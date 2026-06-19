from __future__ import annotations

import gc
import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from importlib.util import find_spec
from pathlib import Path
from typing import Callable

from PIL import Image

from aiwf import __version__
from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.engine import EngineSwitchRequest, EngineTenant
from aiwf.core.domain.wan import WAN_TI2V_5B, WanI2VRequest, WanI2VResult, SAMPLER_TYPES
from aiwf.dev.diagnostics import trace_model_throughput
from aiwf.infrastructure.video import write_frames
from aiwf.infrastructure.wan import WanI2VBackend, WanUnavailable
from aiwf.services.wan_models import (
    WanModelPairCheck,
    wan_lora_matches,
    wan_model_pair_compatibility,
    wan_model_pair_family_key,
    wan_model_quant_family,
    wan_model_stage_role,
    wan_model_storage_family,
    wan_runtime_size_class,
)

logger = logging.getLogger(__name__)

_MAX_DEFAULT_FP8_DEQUANT_GB = 12.0


def _native_fp8_unavailable_reason() -> str | None:
    try:
        import torch
    except Exception as exc:
        return f"PyTorch could not be imported ({exc})."

    try:
        if not torch.cuda.is_available():
            return "CUDA is not available to PyTorch. If `nvidia-smi` reports 'GPU is lost', reboot to reset the driver/GPU."
        major, minor = torch.cuda.get_device_capability()
    except Exception as exc:
        return f"CUDA capability check failed ({exc})."
    if (int(major), int(minor)) < (8, 9):
        return f"GPU compute capability {major}.{minor} is below Ada FP8 tensor-core support (8.9)."
    if not hasattr(torch, "float8_e4m3fn"):
        return "This PyTorch build has no torch.float8_e4m3fn dtype."
    if not hasattr(torch, "_scaled_mm"):
        return "This PyTorch build has no torch._scaled_mm FP8 matmul entry point."
    return None


def _native_fp8_runtime_available() -> bool:
    return _native_fp8_unavailable_reason() is None


def _wan_experimental_formats_enabled() -> bool:
    return os.environ.get("AIWF_WAN_ENABLE_EXPERIMENTAL_FORMATS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _looks_like_incompatible_t5xxl(text_encoder_id: str | None) -> bool:
    text = str(text_encoder_id or "").strip().lower()
    if not text:
        return False
    if any(token in text for token in ("flux", "sd3", "stable-diffusion-3")):
        return True
    return "t5xxl" in text and not any(token in text for token in ("umt5", "nsfw_wan"))


def _video_status(message: str) -> None:
    print(f"[AIWF] Video: {message}", flush=True)
    try:
        from aiwf.dev.diagnostics import trace_safe

        trace_safe("wan.status", message, component="wan.service")
    except Exception:
        logger.debug("Wan status trace failed.", exc_info=True)


def _emit_progress(on_progress, step: int, total: int, steps_per_second=None, message: str | None = None) -> None:
    if on_progress is None:
        return
    try:
        on_progress(step, total, steps_per_second, message)
    except TypeError:
        try:
            on_progress(step, total, steps_per_second)
        except TypeError:
            on_progress(step, total)


def _float_metric(value) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed != parsed or parsed < 0:
        return None
    return parsed


def _int_metric(value) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _str_list_metric(value) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    items: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            items.append(text)
    return items


def _dict_metric(value) -> dict:
    return dict(value) if isinstance(value, dict) else {}


def _dict_list_metric(value) -> list[dict]:
    if not isinstance(value, (list, tuple)):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


@dataclass(frozen=True)
class WanPreflightResult:
    ok: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    model_id: str | None = None
    components_base: str | None = None
    high_noise_model: str | None = None
    low_noise_model: str | None = None
    vae: str | None = None
    text_encoder: str | None = None
    high_noise_lora: str | None = None
    low_noise_lora: str | None = None

    def message(self) -> str:
        parts: list[str] = []
        if self.errors:
            parts.append("Wan model check failed:")
            parts.extend(f"- {error}" for error in self.errors)
        if self.warnings:
            parts.append("Warnings:")
            parts.extend(f"- {warning}" for warning in self.warnings)
        return "\n".join(parts) if parts else "Wan model check passed."


class WanService:
    """Application layer for Wan 2.2 image-to-video."""

    def __init__(
        self,
        flags: RuntimeFlags,
        settings: UserSettings,
        *,
        unload_image_models: Callable[[], None] | None = None,
        supervisor=None,
    ) -> None:
        self.flags = flags
        self.settings = settings
        self._backend = WanI2VBackend(
            async_offload=getattr(flags, "async_offload", True),
            pinned_memory=getattr(flags, "pinned_memory", True),
        )
        self._unload_image_models = unload_image_models
        self.supervisor = supervisor

    def available(self) -> bool:
        return self._backend.available()

    def _cleanup_failed_generation(self) -> None:
        _video_status("Cleaning up failed Wan generation and releasing video VRAM.")
        try:
            self._backend.unload()
        except Exception:
            logger.debug("Wan backend cleanup after failed generation failed.", exc_info=True)

    def unload_models(self) -> None:
        """Release cached Wan models before another GPU video stage starts."""
        _video_status("Unloading Wan video pipeline and clearing VRAM.")
        try:
            self._backend.unload()
        finally:
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if hasattr(torch.cuda, "ipc_collect"):
                        torch.cuda.ipc_collect()
            except Exception:
                logger.debug("Wan CUDA cache cleanup failed.", exc_info=True)

    def acceleration_capabilities(self) -> dict[str, dict[str, object]]:
        from aiwf.infrastructure.torch.wan_perf import describe_wan_acceleration_capabilities

        return describe_wan_acceleration_capabilities()

    def models_dir(self) -> Path:
        return self.flags.resolved_models_dir() / "wan"

    def output_dir(self) -> Path:
        return self.flags.resolved_output_dir() / "video" / "wan"

    def _wan_roots(self) -> list[Path]:
        roots: list[Path] = [self.models_dir()]
        for extra in self.flags.resolved_extra_model_dirs():
            roots.append(extra / "wan")
            roots.append(extra)  # allow wan files directly in an extra models root too
        return roots

    def _wan_weight_roots(self) -> list[Path]:
        """Folders that can contain standalone Wan transformer weights."""
        roots: list[Path] = []
        for root in self._wan_roots():
            roots.extend([
                root / "Safetensor",
                root / "safetensor",
                root / "safetensors",
                root / "GGUF",
                root / "gguf",
                root,
            ])
        deduped: list[Path] = []
        seen: set[Path] = set()
        for root in roots:
            resolved = root.resolve()
            if resolved not in seen:
                seen.add(resolved)
                deduped.append(root)
        return deduped

    def _wan_diffusers_roots(self) -> list[Path]:
        """Folders that can contain full broken-out Diffusers model layouts."""
        roots: list[Path] = []
        for root in self._wan_roots():
            roots.extend([
                root / "Diffusers",
                root / "diffusers",
                root,
            ])
        deduped: list[Path] = []
        seen: set[Path] = set()
        for root in roots:
            resolved = root.resolve()
            if resolved not in seen:
                seen.add(resolved)
                deduped.append(root)
        return deduped

    def _is_components_base(self, path: Path) -> bool:
        text_encoder = path / "text_encoder"
        tokenizer = path / "tokenizer"
        scheduler = path / "scheduler"
        return (
            path.is_dir()
            and (path / "model_index.json").is_file()
            and (text_encoder / "config.json").is_file()
            and (
                (text_encoder / "model.safetensors").is_file()
                or (text_encoder / "model.safetensors.index.json").is_file()
            )
            and (tokenizer / "tokenizer.json").is_file()
            and (scheduler / "scheduler_config.json").is_file()
        )

    def _is_full_fast_5b_diffusers_model(self, path: Path) -> bool:
        """Return whether a Diffusers folder contains the actual 5B transformer."""
        if not (path.is_dir() and (path / "model_index.json").is_file()):
            return False
        transformer_dir = path / "transformer"
        return (transformer_dir / "config.json").is_file() and (
            (transformer_dir / "diffusion_pytorch_model.safetensors").is_file()
            or any(transformer_dir.glob("diffusion_pytorch_model-*.safetensors"))
        )

    def _looks_like_fast_5b_transformer(self, path: Path) -> bool:
        if not path.is_file() or path.suffix.lower() not in {".safetensors", ".gguf"}:
            return False
        name = path.name.lower()
        return "wan" in name and "5b" in name and "ti2v" in name

    def _find_default_fast_5b_transformer(self) -> str | None:
        """Find a local standalone Wan 5B TI2V transformer when no full folder exists."""
        candidates: list[Path] = []
        for root in self._wan_weight_roots() + self._wan_file_candidates():
            if not root.exists():
                continue
            for name in ("wan2.2_ti2v_5B_fp16.safetensors", "wan2.2_ti2v_5b_fp16.safetensors"):
                direct = root / name
                if self._looks_like_fast_5b_transformer(direct):
                    return str(direct.resolve())
            candidates.extend(
                child
                for child in root.rglob("*")
                if child.is_file() and self._looks_like_fast_5b_transformer(child)
            )
        if not candidates:
            return None
        preferred = sorted(
            candidates,
            key=lambda p: (
                0 if "fp16" in p.name.lower() else 1,
                0 if p.suffix.lower() == ".safetensors" else 1,
                p.name.lower(),
            ),
        )[0]
        return str(preferred.resolve())

    def _component_base_missing(self, path: Path) -> list[str]:
        required = [
            path / "model_index.json",
            path / "text_encoder" / "config.json",
            path / "text_encoder" / "model.safetensors",
            path / "tokenizer" / "tokenizer.json",
            path / "scheduler" / "scheduler_config.json",
        ]
        missing = [str(p) for p in required if not p.is_file()]
        if not (path / "text_encoder" / "model.safetensors").is_file() and (
            path / "text_encoder" / "model.safetensors.index.json"
        ).is_file():
            missing = [p for p in missing if not p.endswith("model.safetensors")]
        return missing

    def find_components_base(self) -> str | None:
        """Find a local diffusers folder that can provide Wan shared components."""
        for root in self._wan_diffusers_roots():
            preferred = root / WAN_TI2V_5B.split("/")[-1]
            if self._is_components_base(preferred):
                return str(preferred.resolve())
        for root in self._wan_diffusers_roots():
            if not root.exists():
                continue
            candidates = [root, *[child for child in root.rglob("*") if child.is_dir()]]
            for child in sorted(candidates):
                if self._is_components_base(child):
                    return str(child.resolve())
        return None

    def ensure_components_base(self) -> str:
        base = self.find_components_base()
        if base:
            return base
        raise WanUnavailable(
            "Wan high/low transformer files need a local component base for "
            "text_encoder, tokenizer, and scheduler. Place a Wan component folder such as "
            f"`{WAN_TI2V_5B.split('/')[-1]}` under `{self.models_dir() / 'Diffusers'}` "
            "or download it from Models -> Wan Diffusers folder. AIWF will not auto-download it during generation."
        )

    def _validate_transformer_file(self, path: Path, label: str) -> tuple[list[str], list[str]]:
        errors: list[str] = []
        warnings: list[str] = []
        if not path.is_file():
            return [f"{label} transformer is not a local file: {path}"], warnings

        suffix = path.suffix.lower()
        if suffix not in {".safetensors", ".gguf"}:
            errors.append(f"{label} transformer must be a `.safetensors` or `.gguf` file: {path.name}")
            return errors, warnings

        label_lower = label.lower()
        expected_token = "high" if label_lower.startswith("high") else "low" if label_lower.startswith("low") else ""
        if expected_token and expected_token not in path.name.lower():
            warnings.append(f"{label} transformer filename does not contain `{expected_token}`: {path.name}")

        if suffix == ".gguf":
            from aiwf.infrastructure.wan.transformer_runtime import (
                WanTransformerFormat,
                detect_transformer_format,
                gguf_dequant_stub_allowed,
                gguf_quantized_runtime_enabled,
                gguf_unavailable_message,
            )

            if find_spec("gguf") is None:
                errors.append(f"{label} transformer is GGUF, but the optional `gguf` package is not installed.")
            elif detect_transformer_format(path) == WanTransformerFormat.GGUF_QUANTIZED:
                if not gguf_quantized_runtime_enabled() and not gguf_dequant_stub_allowed():
                    errors.append(gguf_unavailable_message(path, label=label))
                elif gguf_dequant_stub_allowed() and not gguf_quantized_runtime_enabled():
                    warnings.append(
                        f"{label} GGUF dequant stub is enabled - expect very slow load and high RAM. "
                        "Use the default mmap runtime or FP8 safetensors on RTX 40-series."
                    )
                else:
                    warnings.append(
                        f"{label} GGUF uses mmap + on-the-fly dequant (ComfyUI-GGUF style). "
                        "Use Model offload for best speed on 16 GB cards."
                    )
            return errors, warnings

        try:
            from aiwf.infrastructure.wan.comfy_quant_format import inspect_wan_quant_file

            report = inspect_wan_quant_file(path)
            if report.format == "unreadable":
                errors.append(
                    f"{label} transformer safetensors header could not be read: "
                    f"{path.name} ({'; '.join(report.warnings)})"
                )
                return errors, warnings
            if report.tensor_count <= 0:
                errors.append(f"{label} transformer safetensors file has no tensors: {path.name}")
            for missing in report.missing_scale_keys[:5]:
                errors.append(f"{label} transformer is missing FP8 scale tensor: {missing}")
            if report.unsupported_quant_formats:
                errors.append(
                    f"{label} transformer has unsupported quant metadata: "
                    + ", ".join(report.unsupported_quant_formats[:5])
                )
            warnings.extend(f"{label}: {warning}" for warning in report.warnings)
            if report.quantized_weight_count and not report.weight_scale_count:
                warnings.append(
                    f"{label} transformer contains FP8 tensors without obvious scale tensors; "
                    "it may still load if the file is pre-scaled."
                )
            if report.is_comfy_fp8:
                expanded_gb = report.estimated_bf16_expanded_mb / 1024
                if _native_fp8_runtime_available():
                    warnings.append(
                        f"{label} transformer is ComfyUI scaled FP8 "
                        f"({report.quantized_linear_layers} quantized linear layers); AIWF will use the "
                        "experimental native FP8 compatibility path instead of expanding it to bf16."
                    )
                elif expanded_gb > _MAX_DEFAULT_FP8_DEQUANT_GB and os.environ.get("AIWF_WAN_ALLOW_EXPENSIVE_DEQUANT") != "1":
                    unavailable_reason = _native_fp8_unavailable_reason() or "native FP8 runtime is unavailable."
                    errors.append(
                        f"{label} transformer is ComfyUI FP8 ({path.name}). Diffusers cannot consume Comfy FP8 "
                        f"scale tensors directly, and AIWF would need to expand it to about {expanded_gb:.1f} GB "
                        "of bf16 weights for this stage. That path is disabled to prevent a native crash. "
                        f"Native FP8 is unavailable right now: {unavailable_reason}"
                    )
                elif expanded_gb > _MAX_DEFAULT_FP8_DEQUANT_GB:
                    warnings.append(
                        f"{label} transformer will expand Comfy FP8 weights to about {expanded_gb:.1f} GB of bf16 tensors."
                    )
        except Exception as exc:
            errors.append(f"{label} transformer safetensors header could not be read: {path.name} ({exc})")
        return errors, warnings

    def preflight(self, request: WanI2VRequest, *, image_present: bool = True) -> WanPreflightResult:
        """Validate a Wan request before expensive pipeline initialization."""
        errors: list[str] = []
        warnings: list[str] = []

        if not image_present:
            errors.append("Upload a source image to animate.")
        if not self.available():
            errors.append("Wan video is unavailable: update `diffusers` (>=0.35) and install `ftfy`, then restart.")

        high_res: str | None = None
        low_res: str | None = None
        model_res: str | None = None
        requires_dual = (
            request.requires_dual_transformers()
            if callable(getattr(request, "requires_dual_transformers", None))
            else True
        )
        if requires_dual:
            if not request.high_noise_model_id:
                errors.append("Select a High noise transformer.")
            else:
                high_res = self.resolve_model(request.high_noise_model_id)
                e, w = self._validate_transformer_file(Path(high_res), "High noise")
                errors.extend(e)
                warnings.extend(w)

            if not request.low_noise_model_id:
                errors.append("Select a Low noise transformer.")
            else:
                low_res = self.resolve_model(request.low_noise_model_id)
                e, w = self._validate_transformer_file(Path(low_res), "Low noise")
                errors.extend(e)
                warnings.extend(w)

            if high_res and low_res and Path(high_res) == Path(low_res):
                errors.append("High noise and Low noise transformers must be different files.")
            if high_res and low_res:
                if not _wan_experimental_formats_enabled():
                    high_storage = wan_model_storage_family(high_res)
                    low_storage = wan_model_storage_family(low_res)
                    if high_storage != "gguf" or low_storage != "gguf":
                        errors.append(
                            "Stable Wan Video supports GGUF high/low transformer pairs only. "
                            "FP8/safetensors and resident experiments are kept on the dev branch."
                        )
                pair_check = wan_model_pair_compatibility(high_res, low_res)
                errors.extend(pair_check.errors)
                warnings.extend(pair_check.warnings)
        else:
            model_res = self.resolve_model(request.model_id)
            model_path = Path(model_res)
            if model_path.is_file():
                e, w = self._validate_transformer_file(model_path, "Fast 5B")
                errors.extend(e)
                warnings.extend(w)
                if not self._looks_like_fast_5b_transformer(model_path):
                    warnings.append(
                        f"Fast 5B transformer filename does not clearly look like Wan TI2V 5B: {model_path.name}"
                    )
            elif model_path.exists():
                if not self._is_full_fast_5b_diffusers_model(model_path):
                    errors.append(
                        "Fast 5B mode found only a shared component base. Select or install a standalone "
                        "Wan TI2V 5B transformer file, or a full Diffusers folder with transformer/config.json."
                    )
            else:
                errors.append(
                    "Fast 5B mode needs a local Wan TI2V 5B transformer file or full Diffusers folder. "
                    f"Could not resolve locally: {model_res}"
                )

        components_base = self.find_components_base()
        if not components_base:
            preferred = self.models_dir() / "Diffusers" / WAN_TI2V_5B.split("/")[-1]
            missing = self._component_base_missing(preferred)
            missing_text = "\n  ".join(missing) if missing else str(preferred)
            errors.append(
                "Missing local Wan component base. Required files include:\n  "
                f"{missing_text}\n"
                "Install or copy the tokenizer, text_encoder, and scheduler into that folder. "
                "AIWF will not download them during generation."
            )

        vae_id = request.vae_id or self.preferred_vae(request.runtime_mode)
        vae_res = self.resolve_vae(vae_id) if vae_id else None
        if not vae_res:
            errors.append("Missing Wan VAE. Place `wan2.1_vae.safetensors` in `models/VAE` or select a Wan VAE.")
        elif not Path(vae_res).exists():
            errors.append(f"Selected VAE is not local: {vae_res}")
        elif "wan" not in Path(vae_res).name.lower():
            warnings.append(f"Selected VAE does not look Wan-specific: {Path(vae_res).name}")
        else:
            # VAE generation must match the runtime: the A14B high/low pair uses the
            # Wan 2.1 16-channel VAE; the 5B TI2V path uses the Wan 2.2 48-channel VAE.
            # A wrong pick passes name validation but fails late in latent decode, so
            # warn here from the filename (cheap, no tensor read).
            _vae_name = Path(vae_res).name.lower()
            _looks_22 = "2.2" in _vae_name or "wan22" in _vae_name or "_22" in _vae_name
            _looks_21 = "2.1" in _vae_name or "wan21" in _vae_name or "_21" in _vae_name
            if request.requires_dual_transformers() and _looks_22 and not _looks_21:
                errors.append(
                    f"VAE '{Path(vae_res).name}' looks like a Wan 2.2 (48-channel) VAE, but the "
                    "high/low A14B runtime expects the Wan 2.1 (16-channel) VAE "
                    "(`wan2.1_vae.safetensors`). A channel mismatch fails late in latent decode."
                )
            elif not request.requires_dual_transformers() and _looks_21 and not _looks_22:
                errors.append(
                    f"VAE '{Path(vae_res).name}' looks like a Wan 2.1 (16-channel) VAE, but the "
                    "5B TI2V runtime expects the Wan 2.2 (48-channel) VAE (`wan2.2_vae.safetensors`)."
                )

        text_encoder_id = request.text_encoder_path or self.default_text_encoder()
        text_encoder_res = self.resolve_text_encoder(text_encoder_id) if text_encoder_id else None
        if text_encoder_id and not text_encoder_res:
            errors.append(f"Selected Wan text encoder is not local: {text_encoder_id}")
        if _looks_like_incompatible_t5xxl(text_encoder_id) or _looks_like_incompatible_t5xxl(text_encoder_res):
            errors.append(
                "Selected text encoder looks like a Flux/SD3 T5-XXL file. Wan requires UMT5-XXL "
                "(`umt5-xxl`, `umt5`, or `nsfw_wan` naming)."
            )

        high_lora_res = self.resolve_lora(request.high_noise_lora_id)
        low_lora_res = self.resolve_lora(request.low_noise_lora_id)
        runtime_size = wan_runtime_size_class(request.runtime_mode)
        for label, lora in (("High noise LoRA", high_lora_res), ("Low noise LoRA", low_lora_res)):
            if lora and not Path(lora).exists():
                errors.append(f"{label} is not local: {lora}")
            if lora and not wan_lora_matches(lora, size_class=runtime_size):
                expected = "14B high/low" if runtime_size == "14b" else "5B TI2V"
                errors.append(f"{label} does not match the {expected} runtime: {Path(lora).name}")

        return WanPreflightResult(
            ok=not errors,
            errors=tuple(errors),
            warnings=tuple(warnings),
            model_id=model_res,
            components_base=components_base,
            high_noise_model=high_res,
            low_noise_model=low_res,
            vae=vae_res,
            text_encoder=text_encoder_res,
            high_noise_lora=high_lora_res,
            low_noise_lora=low_lora_res,
        )

    def _wan_file_candidates(self) -> list[Path]:
        """Configured extra roots for standalone .safetensors / .gguf Wan weights."""
        cands: list[Path] = []
        for extra in self.flags.resolved_extra_model_dirs():
            if extra.exists():
                cands.append(extra)
            diffusion_models = extra / "diffusion_models"
            if diffusion_models.exists():
                cands.append(diffusion_models)
        return cands

    def _lora_roots(self) -> list[Path]:
        roots: list[Path] = [
            self.flags.resolved_models_dir() / "wan" / "lora",
            self.flags.resolved_models_dir() / "wan" / "loras",
            self.flags.resolved_models_dir() / "Loras",
            self.flags.resolved_models_dir() / "loras",
        ]
        for extra in self.flags.resolved_extra_model_dirs():
            roots.extend([
                extra / "wan" / "lora",
                extra / "wan" / "loras",
                extra / "Loras",
                extra / "loras",
            ])
            if (extra / "loras").exists() or (extra / "Loras").exists():
                roots.extend(child for child in extra.iterdir() if child.is_dir() and "lora" in child.name.lower())
            if extra.exists():
                roots.append(extra)
        deduped: list[Path] = []
        seen: set[Path] = set()
        for root in roots:
            resolved = root.resolve()
            if resolved not in seen:
                seen.add(resolved)
                deduped.append(root)
        return deduped

    def _looks_like_wan_weights(self, name: str) -> bool:
        n = name.lower()
        if n.endswith((".safetensors", ".gguf")):
            if any(k in n for k in ("wan", "i2v", "t2v", "ti2v")):
                return True
        return False

    def _is_lora_path(self, path: Path) -> bool:
        return any(part.lower() in {"lora", "loras"} for part in path.parts)

    def _is_lora_model_candidate(self, path: Path) -> bool:
        try:
            return self._is_lora_path(path.relative_to(self.models_dir()))
        except ValueError:
            return self._is_lora_path(path)

    def _is_valid_wan_weight_path(self, path: Path) -> bool:
        suffix = path.suffix.lower()
        if suffix not in {".safetensors", ".gguf"}:
            return False
        if self._is_lora_model_candidate(path):
            return False
        for root in self._wan_roots():
            try:
                rel = path.relative_to(root)
            except ValueError:
                continue
            if rel.parts:
                bucket = rel.parts[0].lower()
                if bucket == "gguf":
                    return suffix == ".gguf"
                if bucket in {"safetensor", "safetensors"}:
                    return suffix == ".safetensors"
                if bucket == "diffusers":
                    return False
            return True
        return True

    def folder_help(self) -> str:
        return (
            f"**Wan models** -> `{self.models_dir()}`.  \n"
            "Stable video runtime: matched 14B GGUF High Noise + Low Noise transformers. "
            "Use Sequential offload + small size/frames on lower-VRAM cards. "
            "Needs a recent `diffusers` + `ftfy`. "
            "Put Wan high/low transformer weights in `models/wan/Safetensor/` or `models/wan/GGUF/`, "
            "broken-out Diffusers folders in `models/wan/Diffusers/`, "
            "and Wan LoRAs in `models/wan/lora/`. "
            "Add ComfyUI model folders in Settings -> Launch profile -> extra model dirs if you want them scanned. "
            "GGUF files require the optional `gguf` package. "
            "**VAE**: Wan 2.2 I2V (esp. 14B high/low) typically requires the **Wan 2.1 VAE**. Use the VAE selector in the UI or place `wan*vae*.safetensors` in `models/VAE` or a configured extra VAE folder."
        )

    def list_local_models(self) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        seen_file_names: set[str] = set()

        # Full diffusers layouts (dirs with model_index.json)
        for root in self._wan_diffusers_roots():
            if not root.exists():
                continue
            candidates = [root, *[child for child in root.rglob("*") if child.is_dir()]]
            for child in sorted(candidates):
                if child.is_dir() and (child / "model_index.json").exists():
                    try:
                        name = child.relative_to(self.models_dir()).as_posix()
                    except ValueError:
                        name = child.name
                    if name not in seen:
                        seen.add(name)
                        out.append(name)

        # Standalone safetensors / gguf weights (local Wan layout first, Comfy fallback second)
        for root in self._wan_weight_roots() + self._wan_file_candidates():
            if not root.exists():
                continue
            for child in sorted(root.rglob("*")):
                if child.is_file() and self._is_valid_wan_weight_path(child) and self._looks_like_wan_weights(child.name):
                    if child.name in seen_file_names:
                        continue
                    try:
                        name = child.relative_to(self.models_dir()).as_posix()
                    except ValueError:
                        name = child.name
                    if name not in seen:
                        seen.add(name)
                        seen_file_names.add(child.name)
                        out.append(name)

        return out

    def list_local_models_labeled(self) -> list[tuple[str, str]]:
        """Like list_local_models() but returns (display_name, identifier) for Gradio dropdowns.

        The display_name is read from the model file header (arch, precision, size).
        The identifier is the same string returned by list_local_models() - used
        unchanged in WanI2VRequest so no pipeline changes are needed.
        """
        try:
            from aiwf.infrastructure.model_header import read_model_info
        except ImportError:
            return [(n, n) for n in self.list_local_models()]

        out: list[tuple[str, str]] = []
        seen: set[str] = set()
        seen_file_names: set[str] = set()

        # Diffusers folders - no weight file to inspect; use folder name
        for root in self._wan_diffusers_roots():
            if not root.exists():
                continue
            candidates = [root, *[c for c in root.rglob("*") if c.is_dir()]]
            for child in sorted(candidates):
                if child.is_dir() and (child / "model_index.json").exists():
                    try:
                        name = child.relative_to(self.models_dir()).as_posix()
                    except ValueError:
                        name = child.name
                    if name not in seen:
                        seen.add(name)
                        out.append((f"Wan Diffusers - {child.name}", name))

        # Standalone safetensors / gguf files
        for root in self._wan_weight_roots() + self._wan_file_candidates():
            if not root.exists():
                continue
            for child in sorted(root.rglob("*")):
                if not (child.is_file() and self._is_valid_wan_weight_path(child)
                        and self._looks_like_wan_weights(child.name)):
                    continue
                if child.name in seen_file_names:
                    continue
                try:
                    name = child.relative_to(self.models_dir()).as_posix()
                except ValueError:
                    name = child.name
                if name not in seen:
                    seen.add(name)
                    seen_file_names.add(child.name)
                    try:
                        display = read_model_info(child).display_name
                    except Exception:
                        display = child.name
                    out.append((display, name))

        return out

    def list_local_vaes_labeled(self) -> list[tuple[str, str]]:
        """Like list_local_vaes() but returns (display_name, filename) tuples."""
        try:
            from aiwf.infrastructure.model_header import read_model_info
        except ImportError:
            return [(n, n) for n in self.list_local_vaes()]

        names = self.list_local_vaes()
        if names == [""]:
            return [("Default VAE (not installed)", "")]

        labeled: list[tuple[str, str]] = []
        for name in names:
            if not name:
                labeled.append(("Default VAE", ""))
                continue
            # Resolve to absolute path for header reading
            abs_path: Path | None = None
            for root in self._vae_roots():
                candidate = root / name
                if candidate.is_file():
                    abs_path = candidate
                    break
            if abs_path is None:
                labeled.append((name, name))
                continue
            try:
                display = read_model_info(abs_path).display_name
            except Exception:
                display = name
            labeled.append((display, name))
        return labeled

    def list_local_text_encoders_labeled(self) -> list[tuple[str, str]]:
        """Like list_local_text_encoders() but returns (display_name, filename) tuples.

        Filters out T5-XXL (incompatible with Wan) and flags remaining files
        with their detected precision in the label.
        """
        try:
            from aiwf.infrastructure.model_header import read_model_info
        except ImportError:
            return [(n, n) for n in self.list_local_text_encoders()]

        labeled: list[tuple[str, str]] = []
        for name in self.list_local_text_encoders():
            abs_path: Path | None = None
            for root in self._text_encoder_roots():
                candidate = root / name
                if candidate.is_file():
                    abs_path = candidate
                    break
            if abs_path is None:
                labeled.append((name, name))
                continue
            try:
                info = read_model_info(abs_path)
                display = info.display_name
            except Exception:
                display = name
            labeled.append((display, name))
        return labeled

    def _vae_roots(self) -> list[Path]:
        roots: list[Path] = [
            self.flags.resolved_models_dir() / "VAE",
            self.flags.resolved_models_dir() / "vae",
        ]
        for extra in self.flags.resolved_extra_model_dirs():
            roots.extend([
                extra / "VAE",
                extra / "vae",
            ])
            if extra.exists():
                roots.append(extra)
        return roots

    def _text_encoder_roots(self) -> list[Path]:
        roots: list[Path] = [
            self.flags.resolved_models_dir() / "Textencoder",
            self.flags.resolved_models_dir() / "textencoder",
            self.flags.resolved_models_dir() / "TextEncoder",
            self.flags.resolved_models_dir() / "text_encoder",
        ]
        for extra in self.flags.resolved_extra_model_dirs():
            roots.extend([
                extra / "Textencoder",
                extra / "textencoder",
                extra / "TextEncoder",
            ])
        deduped: list[Path] = []
        seen: set[Path] = set()
        for r in roots:
            rp = r.resolve() if not r.is_absolute() else r
            if rp not in seen:
                seen.add(rp)
                deduped.append(r)
        return deduped

    # File stems that signal this is a T5-XXL (Flux/SD3) encoder - NOT UMT5-XXL (Wan).
    # Wan's text encoder is UMT5-XXL; these T5 files will produce garbage output if used.
    _T5_REJECT_STEMS = frozenset([
        "t5xxl_fp16", "t5xxl_fp8_e4m3fn", "t5xxl_fp8_e4m3fn_scaled",
        "t5-v1_1-xxl", "t5_xxl", "t5xxl",
    ])

    def _is_t5xxl_name(self, name: str) -> bool:
        stem = Path(name).stem.lower()
        return stem in self._T5_REJECT_STEMS or stem.startswith("t5xxl")

    def list_local_text_encoders(self) -> list[str]:
        """Scan text-encoder directories for UMT5-XXL files (.safetensors, .gguf).

        T5-XXL files (Flux/SD3) are excluded - they are NOT compatible with Wan.
        Returns filenames (not full paths), or an empty list if none found.
        """
        out: list[str] = []
        seen: set[str] = set()
        for root in self._text_encoder_roots():
            if not root.exists():
                continue
            for child in sorted(root.iterdir()):
                if not child.is_file():
                    continue
                if child.suffix.lower() not in {".safetensors", ".gguf"}:
                    continue
                if self._is_t5xxl_name(child.name):
                    continue
                if child.name not in seen:
                    seen.add(child.name)
                    out.append(child.name)
        return out

    def resolve_text_encoder(self, filename: str | None) -> str | None:
        """Return the absolute path for a text encoder filename, or None."""
        if not filename or not filename.strip():
            return None
        candidate = filename.strip()
        as_path = Path(candidate)
        if as_path.is_file():
            return str(as_path.resolve())
        for root in self._text_encoder_roots():
            p = root / candidate
            if p.is_file():
                return str(p.resolve())
        return None

    def default_text_encoder(self) -> str:
        """Return the best local text encoder filename to use by default.

        Preference order: FP8 safetensors (smallest, fastest) > GGUF > "" (full precision).
        An empty string means "use the full-precision encoder inside components_base/text_encoder/".
        """
        encoders = self.list_local_text_encoders()
        # Prefer explicit FP8 safetensors (usually ~4 GB, much smaller than full-precision)
        for name in encoders:
            n = name.lower()
            if n.endswith(".safetensors") and "fp8" in n:
                return name
        # Next: GGUF (quantized, even smaller, needs gguf package)
        for name in encoders:
            if name.lower().endswith(".gguf"):
                return name
        # Any remaining safetensors
        for name in encoders:
            if name.lower().endswith(".safetensors"):
                return name
        # Fall back to full-precision encoder bundled with components_base
        return ""

    def list_local_vaes(self) -> list[str]:
        """List available VAE files/dirs suitable for Wan (prefers Wan 2.1 VAEs).
        Includes any .safetensors/.pth in the vae roots so user can pick their 2.1 VAE even if name is unusual.
        """
        out: list[str] = []
        seen: set[str] = set()
        for root in self._vae_roots():
            if not root.exists():
                continue
            for child in sorted(root.iterdir()):
                name_lower = child.name.lower()
                if child.is_file() and name_lower.endswith((".safetensors", ".pth", ".pt", ".ckpt")):
                    # include all vae-like files; user knows which is the 2.1 one for their 14B I2V
                    if child.name not in seen:
                        seen.add(child.name)
                        out.append(child.name)
                elif child.is_dir() and (child / "config.json").exists():
                    # diffusers style vae folder
                    if child.name not in seen:
                        seen.add(child.name)
                        out.append(child.name)
        if not out:
            return [""]
        return sorted(
            out,
            key=lambda name: (
                0
                if any(token in name.lower() for token in ("wan2.1_vae", "wan_2.1_vae", "wan21_vae"))
                else 1
                if ("wan" in name.lower() and "vae" in name.lower())
                else 2,
                name.lower(),
            ),
        )

    def preferred_vae(self, runtime_mode: str | None = None) -> str | None:
        vaes = self.list_local_vaes()
        if runtime_mode == "fast_5b":
            for name in vaes:
                lowered = name.lower()
                if "wan2.2" in lowered and "vae" in lowered:
                    return name
        else:
            for name in vaes:
                lowered = name.lower()
                if any(token in lowered for token in ("wan2.1_vae", "wan_2.1_vae", "wan21_vae")):
                    return name
        for name in vaes:
            lowered = name.lower()
            if "wan" in lowered and "vae" in lowered:
                return name
        return None

    def list_local_loras(self, stage: str | None = None) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        seen_file_names: set[str] = set()
        required = (stage or "").strip().lower()
        for root in self._lora_roots():
            if not root.exists():
                continue
            for child in sorted(root.rglob("*")):
                if not child.is_file():
                    continue
                if child.suffix.lower() not in {".safetensors", ".pt", ".pth"}:
                    continue
                if required and required not in child.name.lower():
                    continue
                if child.name in seen_file_names:
                    continue
                try:
                    label = child.relative_to(root).as_posix()
                except ValueError:
                    label = child.name
                if label not in seen:
                    seen.add(label)
                    seen_file_names.add(child.name)
                    out.append(label)
        return out

    def resolve_lora(self, lora_id: str | None) -> str | None:
        if not lora_id or not lora_id.strip():
            return None
        candidate = lora_id.strip()
        as_path = Path(candidate)
        if as_path.exists():
            return str(as_path.resolve())
        for root in self._lora_roots():
            p = root / candidate
            if p.exists():
                return str(p.resolve())
            for child in root.rglob(Path(candidate).name):
                if child.is_file() and child.suffix.lower() in {".safetensors", ".pt", ".pth"}:
                    return str(child.resolve())
        return candidate

    def resolve_vae(self, vae_id: str | None) -> str | None:
        if not vae_id or not vae_id.strip():
            return None
        candidate = vae_id.strip()
        as_path = Path(candidate)
        if as_path.exists():
            return str(as_path.resolve())
        for root in self._vae_roots():
            p = root / candidate
            if p.exists():
                return str(p.resolve())
        return candidate  # let it fail later with clear error if bad

    def resolve_model(self, model_id: str | None) -> str:
        """Local diffusers dir, local .safetensors/.gguf weight file, or fall back to HF repo id."""
        candidate = (model_id or WAN_TI2V_5B).strip() or WAN_TI2V_5B
        as_path = Path(candidate)
        is_default_fast_5b = candidate == WAN_TI2V_5B or candidate.endswith(WAN_TI2V_5B.split("/")[-1])

        # Direct file (absolute or relative that resolves)
        if as_path.is_file() and self._is_valid_wan_weight_path(as_path):
            return str(as_path.resolve())

        # Direct dir (full diffusers layout)
        if as_path.is_dir() and (as_path / "model_index.json").exists():
            return str(as_path)

        # Search roots for exact dir match or file match (by relative path or name).
        # For the default fast-5B id, prefer a complete pipeline folder; if the
        # local match is only the shared component base, fall through and look
        # for a standalone 5B transformer file.
        component_base_match: str | None = None
        for root in self._wan_diffusers_roots():
            local_dir = root / candidate
            if local_dir.is_dir() and (local_dir / "model_index.json").exists():
                if self._is_full_fast_5b_diffusers_model(local_dir) or not is_default_fast_5b:
                    return str(local_dir)
                component_base_match = str(local_dir.resolve())
            for child in root.rglob(Path(candidate).name):
                if child.is_dir() and (child / "model_index.json").exists():
                    if self._is_full_fast_5b_diffusers_model(child) or not is_default_fast_5b:
                        return str(child.resolve())
                    component_base_match = str(child.resolve())

        if is_default_fast_5b:
            default_transformer = self._find_default_fast_5b_transformer()
            if default_transformer:
                return default_transformer
            if component_base_match:
                return component_base_match

        for root in self._wan_weight_roots():
            local_file = root / candidate
            if (
                local_file.is_file()
                and self._is_valid_wan_weight_path(local_file)
            ):
                return str(local_file)

            # Also try without extension if user passed stem
            if not as_path.suffix:
                for ext in (".safetensors", ".gguf"):
                    f = root / (candidate + ext)
                    if f.is_file() and self._is_valid_wan_weight_path(f):
                        return str(f)

            for child in root.rglob(Path(candidate).name):
                if self._is_lora_model_candidate(child):
                    continue
                if child.is_file() and self._is_valid_wan_weight_path(child):
                    return str(child.resolve())

        # Also scan the Comfy/extra candidate file dirs (so bare "wan2.2_....safetensors" resolves)
        for root in self._wan_file_candidates():
            local_file = root / candidate
            if local_file.is_file() and local_file.suffix.lower() in {".safetensors", ".gguf"}:
                return str(local_file)
            if not as_path.suffix:
                for ext in (".safetensors", ".gguf"):
                    f = root / (candidate + ext)
                    if f.is_file():
                        return str(f)

        return candidate

    def _output_path(self) -> Path:
        root = self.output_dir()
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        candidate = root / f"wan-i2v-{stamp}.mp4"
        counter = 1
        while candidate.exists():
            candidate = root / f"wan-i2v-{stamp}-{counter}.mp4"
            counter += 1
        return candidate

    def generate(
        self,
        request: WanI2VRequest,
        image: Image.Image,
        *,
        on_progress=None,
        should_cancel=None,
    ) -> WanI2VResult:
        preflight = self.preflight(request, image_present=image is not None)
        if not preflight.ok:
            raise WanUnavailable(preflight.message())
        request = request.model_copy(
            update={
                "model_id": preflight.model_id or request.model_id,
                "high_noise_model_id": preflight.high_noise_model or request.high_noise_model_id,
                "low_noise_model_id": preflight.low_noise_model or request.low_noise_model_id,
                "vae_id": preflight.vae or request.vae_id,
                "text_encoder_path": preflight.text_encoder or request.text_encoder_path,
                "high_noise_lora_id": preflight.high_noise_lora or request.high_noise_lora_id,
                "low_noise_lora_id": preflight.low_noise_lora or request.low_noise_lora_id,
                "components_base": preflight.components_base or request.components_base,
            }
        )

        if image is None:
            raise WanUnavailable("Upload a source image to animate.")
        if not self.available():
            raise WanUnavailable(
                "Wan video is unavailable - update `diffusers` (>=0.35) and install `ftfy`, then restart."
            )
        tenant_job_id = f"wan_{uuid.uuid4().hex[:8]}"
        if self.supervisor is not None:
            switch = self.supervisor.request_switch(
                EngineSwitchRequest(
                    target=EngineTenant.VIDEO,
                    reason="Wan video generation",
                    job_id=tenant_job_id,
                )
            )
            if not switch.ok:
                raise WanUnavailable(f"GPU busy: {switch.message}")
        try:
            if self._unload_image_models is not None:
                _video_status("Unloading image models before loading video pipeline.")
                try:
                    self._unload_image_models()
                except Exception:
                    logger.exception("Failed to unload image models before Wan generation; continuing.")

            # Expose HF token so large/gated model
            hf_token = getattr(self.settings, "huggingface_token", "").strip() if self.settings else ""
            if hf_token:
                os.environ.setdefault("HF_TOKEN", hf_token)
                os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)

            started = time.perf_counter()
            requested_steps = max(1, int(request.effective_steps() or 1))
            _emit_progress(
                on_progress,
                0,
                requested_steps,
                None,
                "Loading models and encoding inputs",
            )

            try:
                backend_result = self._backend.generate(
                    request,
                    image,
                    on_progress=on_progress,
                    should_cancel=should_cancel,
                )
                backend_metrics = {}
                if isinstance(backend_result, (tuple, list)) and len(backend_result) == 4:
                    frames, h, w, backend_metrics = backend_result
                    backend_metrics = backend_metrics if isinstance(backend_metrics, dict) else {}
                elif isinstance(backend_result, (tuple, list)) and len(backend_result) == 3:
                    frames, h, w = backend_result
                else:
                    raise WanUnavailable("Video backend returned an invalid generation result.")
            except WanUnavailable:
                self._cleanup_failed_generation()
                raise
            except Exception as exc:
                self._cleanup_failed_generation()
                raise WanUnavailable(f"Video generation failed: {exc}") from exc

            backend_elapsed = time.perf_counter() - started
            step_count = max(0, int(backend_metrics.get("step_count") or request.effective_steps()))
            load_seconds = _float_metric(backend_metrics.get("load_seconds")) or 0.0
            preprocess_seconds = _float_metric(backend_metrics.get("preprocess_seconds")) or 0.0
            prompt_encode_seconds = _float_metric(backend_metrics.get("prompt_encode_seconds")) or 0.0
            image_encode_seconds = _float_metric(backend_metrics.get("image_encode_seconds")) or 0.0
            latent_prepare_seconds = _float_metric(backend_metrics.get("latent_prepare_seconds")) or 0.0
            denoise_seconds = _float_metric(backend_metrics.get("denoise_seconds")) or 0.0
            high_denoise_seconds = _float_metric(backend_metrics.get("high_denoise_seconds")) or 0.0
            low_denoise_seconds = _float_metric(backend_metrics.get("low_denoise_seconds")) or 0.0
            pipeline_seconds = _float_metric(backend_metrics.get("pipeline_seconds")) or backend_elapsed
            pipeline_overhead_seconds = _float_metric(backend_metrics.get("pipeline_overhead_seconds"))
            if pipeline_overhead_seconds is None:
                pipeline_overhead_seconds = max(0.0, pipeline_seconds - denoise_seconds)
            vae_decode_seconds = _float_metric(backend_metrics.get("vae_decode_seconds")) or 0.0
            manual_vae_decode = bool(backend_metrics.get("manual_vae_decode"))
            try:
                vae_decode_chunk_frames = max(0, int(backend_metrics.get("vae_decode_chunk_frames") or 0))
            except (TypeError, ValueError):
                vae_decode_chunk_frames = 0
            latent_frame_count = _int_metric(backend_metrics.get("latent_frame_count"))
            temporal_chunks = bool(backend_metrics.get("temporal_chunks"))
            temporal_chunk_size = _int_metric(backend_metrics.get("temporal_chunk_size"))
            temporal_chunk_overlap = _int_metric(backend_metrics.get("temporal_chunk_overlap"))
            transformer_chunks_per_forward = _int_metric(backend_metrics.get("transformer_chunks_per_forward")) or 1
            transformer_forwards_per_step = _int_metric(backend_metrics.get("transformer_forwards_per_step")) or 1
            video_postprocess_seconds = _float_metric(backend_metrics.get("video_postprocess_seconds")) or 0.0
            offload_cleanup_seconds = _float_metric(backend_metrics.get("offload_cleanup_seconds")) or 0.0
            postprocess_seconds = _float_metric(backend_metrics.get("postprocess_seconds")) or 0.0
            steps_per_second = _float_metric(backend_metrics.get("steps_per_second"))
            if steps_per_second is None and denoise_seconds > 0 and step_count > 0:
                steps_per_second = step_count / denoise_seconds
            iterations_per_second = steps_per_second
            fp8_linear_layers = _int_metric(backend_metrics.get("fp8_linear_layers"))
            fp8_fast_mm_calls = _int_metric(backend_metrics.get("fp8_fast_mm_calls"))
            fp8_fallback_calls = _int_metric(backend_metrics.get("fp8_fallback_calls"))
            fp8_fallback_layers = _int_metric(backend_metrics.get("fp8_fallback_layers"))
            fp8_fallback_reasons = _str_list_metric(backend_metrics.get("fp8_fallback_reasons"))
            fp8_strict_mode = bool(backend_metrics.get("fp8_strict_mode"))
            fp8_native_available = bool(backend_metrics.get("fp8_native_available"))
            fp8_profile_enabled = bool(backend_metrics.get("fp8_profile_enabled"))
            fp8_backend = str(backend_metrics.get("fp8_backend") or "").strip()
            fp8_backend_metadata = _dict_metric(backend_metrics.get("fp8_backend_metadata"))
            fp8_linear_shape_count = _int_metric(backend_metrics.get("fp8_linear_shape_count"))
            fp8_linear_shapes = _dict_list_metric(backend_metrics.get("fp8_linear_shapes"))
            fp8_prepare_ms = _float_metric(backend_metrics.get("fp8_prepare_ms")) or 0.0
            fp8_scaled_mm_ms = _float_metric(backend_metrics.get("fp8_scaled_mm_ms")) or 0.0
            fp8_bias_ms = _float_metric(backend_metrics.get("fp8_bias_ms")) or 0.0
            fp8_fallback_ms = _float_metric(backend_metrics.get("fp8_fallback_ms")) or 0.0
            attention_backends = _str_list_metric(backend_metrics.get("attention_backends"))
            attention_optimizations = _str_list_metric(backend_metrics.get("attention_optimizations"))
            stage_transition_count = _int_metric(backend_metrics.get("stage_transition_count"))
            stage_transition_total_ms = _float_metric(backend_metrics.get("stage_transition_total_ms")) or 0.0
            stage_transition_h2d_ms = _float_metric(backend_metrics.get("stage_transition_h2d_ms")) or 0.0
            stage_transition_d2h_ms = _float_metric(backend_metrics.get("stage_transition_d2h_ms")) or 0.0
            stage_transition_cleanup_ms = _float_metric(backend_metrics.get("stage_transition_cleanup_ms")) or 0.0
            stage_transition_events = _dict_list_metric(backend_metrics.get("stage_transition_events"))
            hardware_fingerprint = _dict_metric(backend_metrics.get("hardware_fingerprint"))
            transfer_probe = _dict_metric(backend_metrics.get("transfer_probe"))
            performance_benchmark_notes: list[str] = []
            if fp8_fallback_calls:
                performance_benchmark_notes.append("FP8 fallback occurred; speed result is diagnostic only.")
            if fp8_linear_layers and not fp8_strict_mode:
                performance_benchmark_notes.append("FP8 strict mode was disabled.")
            performance_benchmark_valid = not performance_benchmark_notes
            cache_mode = str(backend_metrics.get("cache_mode") or "").strip()
            vram_reserve_enabled = bool(backend_metrics.get("vram_reserve_enabled"))
            vram_reserve_mb = _int_metric(backend_metrics.get("vram_reserve_mb"))
            vram_limit_mb = _int_metric(backend_metrics.get("vram_limit_mb"))
            vram_total_mb = _int_metric(backend_metrics.get("vram_total_mb"))
            vram_limit_fraction = _float_metric(backend_metrics.get("vram_limit_fraction"))
            if vram_limit_fraction is None:
                vram_limit_fraction = 1.0

            output_path = self._output_path()
            output_path.parent.mkdir(parents=True, exist_ok=True)

            write_started = time.perf_counter()
            _emit_progress(
                on_progress,
                step_count,
                max(1, step_count),
                steps_per_second,
                "Writing video file",
            )
            frame_count = write_frames(
                frames,
                output_path,
                fps=float(getattr(request, "fps", 16) or 16),
            )
            _emit_progress(
                on_progress,
                step_count,
                max(1, step_count),
                steps_per_second,
                "Video file saved",
            )
            video_write_seconds = max(0.0, time.perf_counter() - write_started)
            elapsed = time.perf_counter() - started
            trace_model_throughput(
                kind="wan.video",
                model_id=str(
                    request.model_id
                    if not request.requires_dual_transformers()
                    else request.high_noise_model_id
                    or ""
                ),
                model_name=(
                    str(request.model_id)
                    if not request.requires_dual_transformers()
                    else f"{request.high_noise_model_id} + {request.low_noise_model_id}"
                ),
                app_version=__version__,
                elapsed_seconds=elapsed,
                units=max(1, int(frame_count)),
                units_label="frames",
                runtime_mode=request.runtime_mode,
                high_noise_model_id=request.high_noise_model_id,
                low_noise_model_id=request.low_noise_model_id,
                offload=request.offload,
                fps=int(getattr(request, "fps", 16) or 16),
                step_count=step_count,
                load_seconds=round(load_seconds, 3),
                preprocess_seconds=round(preprocess_seconds, 3),
                prompt_encode_seconds=round(prompt_encode_seconds, 3),
                image_encode_seconds=round(image_encode_seconds, 3),
                latent_prepare_seconds=round(latent_prepare_seconds, 3),
                denoise_seconds=round(denoise_seconds, 3),
                high_denoise_seconds=round(high_denoise_seconds, 3),
                low_denoise_seconds=round(low_denoise_seconds, 3),
                pipeline_seconds=round(pipeline_seconds, 3),
                pipeline_overhead_seconds=round(pipeline_overhead_seconds, 3),
                vae_decode_seconds=round(vae_decode_seconds, 3),
                manual_vae_decode=manual_vae_decode,
                vae_decode_chunk_frames=vae_decode_chunk_frames,
                latent_frame_count=latent_frame_count,
                temporal_chunks=temporal_chunks,
                temporal_chunk_size=temporal_chunk_size,
                temporal_chunk_overlap=temporal_chunk_overlap,
                transformer_chunks_per_forward=transformer_chunks_per_forward,
                transformer_forwards_per_step=transformer_forwards_per_step,
                video_postprocess_seconds=round(video_postprocess_seconds, 3),
                offload_cleanup_seconds=round(offload_cleanup_seconds, 3),
                postprocess_seconds=round(postprocess_seconds, 3),
                video_write_seconds=round(video_write_seconds, 3),
                steps_per_second=round(steps_per_second, 6) if steps_per_second is not None else None,
                iterations_per_second=round(iterations_per_second, 6)
                if iterations_per_second is not None
                else None,
                fp8_linear_layers=fp8_linear_layers,
                fp8_fast_mm_calls=fp8_fast_mm_calls,
                fp8_fallback_calls=fp8_fallback_calls,
                fp8_fallback_layers=fp8_fallback_layers,
                fp8_fallback_reasons=fp8_fallback_reasons,
                fp8_strict_mode=fp8_strict_mode,
                fp8_native_available=fp8_native_available,
                fp8_profile_enabled=fp8_profile_enabled,
                fp8_backend=fp8_backend,
                fp8_backend_metadata=fp8_backend_metadata,
                fp8_linear_shape_count=fp8_linear_shape_count,
                fp8_linear_shapes=fp8_linear_shapes,
                fp8_prepare_ms=round(fp8_prepare_ms, 3),
                fp8_scaled_mm_ms=round(fp8_scaled_mm_ms, 3),
                fp8_bias_ms=round(fp8_bias_ms, 3),
                fp8_fallback_ms=round(fp8_fallback_ms, 3),
                attention_backends=attention_backends,
                attention_optimizations=attention_optimizations,
                stage_transition_count=stage_transition_count,
                stage_transition_total_ms=round(stage_transition_total_ms, 3),
                stage_transition_h2d_ms=round(stage_transition_h2d_ms, 3),
                stage_transition_d2h_ms=round(stage_transition_d2h_ms, 3),
                stage_transition_cleanup_ms=round(stage_transition_cleanup_ms, 3),
                stage_transition_events=stage_transition_events,
                hardware_fingerprint=hardware_fingerprint,
                transfer_probe=transfer_probe,
                performance_benchmark_valid=performance_benchmark_valid,
                performance_benchmark_notes=performance_benchmark_notes,
                cache_mode=cache_mode,
                vram_reserve_enabled=vram_reserve_enabled,
                vram_reserve_mb=vram_reserve_mb,
                vram_limit_mb=vram_limit_mb,
                vram_total_mb=vram_total_mb,
                vram_limit_fraction=round(vram_limit_fraction, 6),
            )

            if steps_per_second is not None and steps_per_second > 0:
                message = (
                    f"{frame_count} frames at {w}x{h} in {elapsed:.1f}s; "
                    f"{steps_per_second:.3f} steps/s ({steps_per_second:.3f} it/s, "
                    f"{1.0 / steps_per_second:.2f} s/it); denoise {denoise_seconds:.1f}s, "
                    f"write {video_write_seconds:.1f}s"
                )
            else:
                message = f"{frame_count} frames at {w}x{h} in {elapsed:.1f}s"
            if fp8_fallback_calls:
                reason = f": {fp8_fallback_reasons[0]}" if fp8_fallback_reasons else ""
                message = (
                    f"{message}; FP8 fallback calls={fp8_fallback_calls} "
                    f"across {fp8_fallback_layers} layers{reason}"
                )
            elif fp8_linear_layers:
                message = f"{message}; FP8 fast path clean ({fp8_linear_layers} layers, 0 fallbacks)"
            if fp8_profile_enabled:
                message = (
                    f"{message}; FP8 profile prepare={fp8_prepare_ms:.1f}ms, "
                    f"mm={fp8_scaled_mm_ms:.1f}ms, bias={fp8_bias_ms:.1f}ms"
                )
            if latent_frame_count:
                message = (
                    f"{message}; latent={latent_frame_count}f, "
                    f"chunks={'on' if temporal_chunks else 'off'}, "
                    f"xfwd/step~{transformer_forwards_per_step}"
                )
            if attention_backends:
                message = f"{message}; attention={','.join(attention_backends)}"
            if stage_transition_count:
                message = (
                    f"{message}; stage_swaps={stage_transition_count} "
                    f"({stage_transition_total_ms:.0f}ms)"
                )
            if cache_mode:
                message = f"{message}; cache={cache_mode}"
            if vram_reserve_enabled and vram_limit_mb and vram_total_mb:
                message = (
                    f"{message}; VRAM cap={vram_limit_mb}/{vram_total_mb} MB "
                    f"(keep_free={vram_reserve_mb} MB)"
                )

            return WanI2VResult(
                output_path=str(output_path),
                frame_count=frame_count,
                fps=int(getattr(request, "fps", 16) or 16),
                width=w,
                height=h,
                elapsed_seconds=round(elapsed, 2),
                step_count=step_count,
                load_seconds=round(load_seconds, 3),
                preprocess_seconds=round(preprocess_seconds, 3),
                prompt_encode_seconds=round(prompt_encode_seconds, 3),
                image_encode_seconds=round(image_encode_seconds, 3),
                latent_prepare_seconds=round(latent_prepare_seconds, 3),
                denoise_seconds=round(denoise_seconds, 3),
                high_denoise_seconds=round(high_denoise_seconds, 3),
                low_denoise_seconds=round(low_denoise_seconds, 3),
                pipeline_seconds=round(pipeline_seconds, 3),
                pipeline_overhead_seconds=round(pipeline_overhead_seconds, 3),
                vae_decode_seconds=round(vae_decode_seconds, 3),
                manual_vae_decode=manual_vae_decode,
                vae_decode_chunk_frames=vae_decode_chunk_frames,
                latent_frame_count=latent_frame_count,
                temporal_chunks=temporal_chunks,
                temporal_chunk_size=temporal_chunk_size,
                temporal_chunk_overlap=temporal_chunk_overlap,
                transformer_chunks_per_forward=transformer_chunks_per_forward,
                transformer_forwards_per_step=transformer_forwards_per_step,
                video_postprocess_seconds=round(video_postprocess_seconds, 3),
                offload_cleanup_seconds=round(offload_cleanup_seconds, 3),
                postprocess_seconds=round(postprocess_seconds, 3),
                video_write_seconds=round(video_write_seconds, 3),
                steps_per_second=round(steps_per_second, 6) if steps_per_second is not None else None,
                iterations_per_second=round(iterations_per_second, 6)
                if iterations_per_second is not None
                else None,
                fp8_linear_layers=fp8_linear_layers,
                fp8_fast_mm_calls=fp8_fast_mm_calls,
                fp8_fallback_calls=fp8_fallback_calls,
                fp8_fallback_layers=fp8_fallback_layers,
                fp8_fallback_reasons=fp8_fallback_reasons,
                fp8_strict_mode=fp8_strict_mode,
                fp8_native_available=fp8_native_available,
                fp8_profile_enabled=fp8_profile_enabled,
                fp8_backend=fp8_backend,
                fp8_backend_metadata=fp8_backend_metadata,
                fp8_linear_shape_count=fp8_linear_shape_count,
                fp8_linear_shapes=fp8_linear_shapes,
                fp8_prepare_ms=round(fp8_prepare_ms, 3),
                fp8_scaled_mm_ms=round(fp8_scaled_mm_ms, 3),
                fp8_bias_ms=round(fp8_bias_ms, 3),
                fp8_fallback_ms=round(fp8_fallback_ms, 3),
                attention_backends=attention_backends,
                attention_optimizations=attention_optimizations,
                stage_transition_count=stage_transition_count,
                stage_transition_total_ms=round(stage_transition_total_ms, 3),
                stage_transition_h2d_ms=round(stage_transition_h2d_ms, 3),
                stage_transition_d2h_ms=round(stage_transition_d2h_ms, 3),
                stage_transition_cleanup_ms=round(stage_transition_cleanup_ms, 3),
                stage_transition_events=stage_transition_events,
                hardware_fingerprint=hardware_fingerprint,
                transfer_probe=transfer_probe,
                performance_benchmark_valid=performance_benchmark_valid,
                performance_benchmark_notes=performance_benchmark_notes,
                cache_mode=cache_mode,
                vram_reserve_enabled=vram_reserve_enabled,
                vram_reserve_mb=vram_reserve_mb,
                vram_limit_mb=vram_limit_mb,
                vram_total_mb=vram_total_mb,
                vram_limit_fraction=round(vram_limit_fraction, 6),
                message=message,
            )
        finally:
            if self.supervisor is not None:
                self.supervisor.request_switch(
                    EngineSwitchRequest(
                        target=EngineTenant.IDLE,
                        reason="Wan video generation complete",
                        job_id=tenant_job_id,
                    )
                )
