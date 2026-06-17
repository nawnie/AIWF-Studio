from __future__ import annotations

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

logger = logging.getLogger(__name__)

_MAX_DEFAULT_FP8_DEQUANT_GB = 12.0


def _native_fp8_runtime_available() -> bool:
    try:
        import torch

        return bool(
            torch.cuda.is_available()
            and hasattr(torch, "float8_e4m3fn")
            and hasattr(torch, "_scaled_mm")
        )
    except Exception:
        return False


def _video_status(message: str) -> None:
    print(f"[AIWF] Video: {message}", flush=True)
    try:
        from aiwf.dev.diagnostics import trace_safe

        trace_safe("wan.status", message, component="wan.service")
    except Exception:
        logger.debug("Wan status trace failed.", exc_info=True)


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


@dataclass(frozen=True)
class WanPreflightResult:
    ok: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
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

        expected_token = "high" if label.lower().startswith("high") else "low"
        if expected_token not in path.name.lower():
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
            from safetensors import safe_open

            saw_tensor = False
            saw_fp8 = False
            saw_scale = False
            fp8_elements = 0
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    saw_tensor = True
                    key_l = key.lower()
                    if key_l.endswith((".weight_scale", ".scale_weight", ".pre_quant_scale")):
                        saw_scale = True
                    try:
                        tensor_slice = handle.get_slice(key)
                        dtype = tensor_slice.get_dtype()
                    except Exception:
                        continue
                    if str(dtype).upper().startswith("F8"):
                        saw_fp8 = True
                        count = 1
                        for dim in tensor_slice.get_shape():
                            count *= dim
                        fp8_elements += count
            if not saw_tensor:
                errors.append(f"{label} transformer safetensors file has no tensors: {path.name}")
            if saw_fp8 and not saw_scale:
                warnings.append(
                    f"{label} transformer contains FP8 tensors without obvious scale tensors; "
                    "it may still load if the file is pre-scaled."
                )
            if saw_fp8 and saw_scale:
                expanded_gb = fp8_elements * 2 / 1024**3
                if _native_fp8_runtime_available():
                    warnings.append(
                        f"{label} transformer is ComfyUI scaled FP8; AIWF will use the experimental native FP8 "
                        "compatibility path instead of expanding it to bf16."
                    )
                elif expanded_gb > _MAX_DEFAULT_FP8_DEQUANT_GB and os.environ.get("AIWF_WAN_ALLOW_EXPENSIVE_DEQUANT") != "1":
                    errors.append(
                        f"{label} transformer is ComfyUI FP8 ({path.name}). Diffusers cannot consume Comfy FP8 "
                        f"scale tensors directly, and AIWF would need to expand it to about {expanded_gb:.1f} GB "
                        "of bf16 weights for this stage. That path is disabled to prevent a native crash. "
                        "Use a Diffusers-format bf16/fp16 transformer or wait for AIWF native Comfy FP8/GGUF runtime support."
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

        vae_id = request.vae_id or self.preferred_vae()
        vae_res = self.resolve_vae(vae_id) if vae_id else None
        if not vae_res:
            errors.append("Missing Wan VAE. Place `wan_2.1_vae.safetensors` in `models/VAE` or select a Wan VAE.")
        elif not Path(vae_res).exists():
            errors.append(f"Selected VAE is not local: {vae_res}")
        elif "wan" not in Path(vae_res).name.lower():
            warnings.append(f"Selected VAE does not look Wan-specific: {Path(vae_res).name}")

        text_encoder_id = request.text_encoder_path or self.default_text_encoder()
        text_encoder_res = self.resolve_text_encoder(text_encoder_id) if text_encoder_id else None
        if text_encoder_id and not text_encoder_res:
            errors.append(f"Selected Wan text encoder is not local: {text_encoder_id}")

        high_lora_res = self.resolve_lora(request.high_noise_lora_id)
        low_lora_res = self.resolve_lora(request.low_noise_lora_id)
        for label, lora in (("High noise LoRA", high_lora_res), ("Low noise LoRA", low_lora_res)):
            if lora and not Path(lora).exists():
                errors.append(f"{label} is not local: {lora}")

        return WanPreflightResult(
            ok=not errors,
            errors=tuple(errors),
            warnings=tuple(warnings),
            components_base=components_base,
            high_noise_model=high_res,
            low_noise_model=low_res,
            vae=vae_res,
            text_encoder=text_encoder_res,
            high_noise_lora=high_lora_res,
            low_noise_lora=low_lora_res,
        )

    def _wan_file_candidates(self) -> list[Path]:
        """Extra locations to scan for standalone .safetensors / .gguf Wan diffusion weights (e.g. your ComfyUI diffusion_models)."""
        cands: list[Path] = []
        # Explicit path you provided for testing your models
        comfy_dm = Path(r"F:\ComfyUI\models\diffusion_models")
        if comfy_dm.exists():
            cands.append(comfy_dm)
        # Also any extra_model_dir at top level (in case user points extra to Comfy models root)
        for extra in self.flags.resolved_extra_model_dirs():
            if extra.exists():
                cands.append(extra)
        return cands

    def _lora_roots(self) -> list[Path]:
        roots: list[Path] = [
            self.flags.resolved_models_dir() / "wan" / "lora",
            self.flags.resolved_models_dir() / "wan" / "loras",
            self.flags.resolved_models_dir() / "Loras",
            self.flags.resolved_models_dir() / "loras",
        ]
        comfy_models = Path(r"F:\ComfyUI\models")
        if comfy_models.exists():
            roots.extend(child for child in comfy_models.iterdir() if child.is_dir() and "lora" in child.name.lower())
        for extra in self.flags.resolved_extra_model_dirs():
            roots.extend([
                extra / "wan" / "lora",
                extra / "wan" / "loras",
                extra / "Loras",
                extra / "loras",
            ])
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
            f"Default model: `{WAN_TI2V_5B}` (5B, the most VRAM-friendly). "
            "8 GB cards: use Sequential offload + small size/frames (slow). "
            "Needs a recent `diffusers` + `ftfy`. "
            "Put Wan high/low transformer weights in `models/wan/Safetensor/` or `models/wan/GGUF/`, "
            "broken-out Diffusers folders in `models/wan/Diffusers/`, "
            "and Wan LoRAs in `models/wan/lora/`. "
            "ComfyUI `diffusion_models/` is still scanned as a fallback if the filename contains 'wan'/'i2v'/'t2v'. "
            "GGUF files require the optional `gguf` package. "
            "**VAE**: Wan 2.2 I2V (esp. 14B high/low) typically requires the **Wan 2.1 VAE**. Use the VAE selector in the UI or place `wan*vae*.safetensors` in `models/VAE` or your Comfy `models/vae/`."
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
        # User's Comfy location for vae (common for Wan 2.1 VAE)
        comfy_vae = Path(r"F:\ComfyUI\models\vae")
        if comfy_vae.exists():
            roots.append(comfy_vae)
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
                if any(token in name.lower() for token in ("wan_2.1_vae", "wan2.1_vae", "wan21_vae"))
                else 1
                if ("wan" in name.lower() and "vae" in name.lower())
                else 2,
                name.lower(),
            ),
        )

    def preferred_vae(self) -> str | None:
        for name in self.list_local_vaes():
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

        # Direct file (absolute or relative that resolves)
        if as_path.is_file() and self._is_valid_wan_weight_path(as_path):
            return str(as_path.resolve())

        # Direct dir (full diffusers layout)
        if as_path.is_dir() and (as_path / "model_index.json").exists():
            return str(as_path)

        # Search roots for exact dir match or file match (by relative path or name)
        for root in self._wan_diffusers_roots():
            local_dir = root / candidate
            if local_dir.is_dir() and (local_dir / "model_index.json").exists():
                return str(local_dir)
            for child in root.rglob(Path(candidate).name):
                if child.is_dir() and (child / "model_index.json").exists():
                    return str(child.resolve())

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
                raise
            except Exception as exc:
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
            video_postprocess_seconds = _float_metric(backend_metrics.get("video_postprocess_seconds")) or 0.0
            offload_cleanup_seconds = _float_metric(backend_metrics.get("offload_cleanup_seconds")) or 0.0
            postprocess_seconds = _float_metric(backend_metrics.get("postprocess_seconds")) or 0.0
            steps_per_second = _float_metric(backend_metrics.get("steps_per_second"))
            if steps_per_second is None and denoise_seconds > 0 and step_count > 0:
                steps_per_second = step_count / denoise_seconds
            iterations_per_second = steps_per_second

            output_path = self._output_path()
            output_path.parent.mkdir(parents=True, exist_ok=True)

            write_started = time.perf_counter()
            frame_count = write_frames(
                frames,
                output_path,
                fps=float(getattr(request, "fps", 16) or 16),
            )
            video_write_seconds = max(0.0, time.perf_counter() - write_started)
            elapsed = time.perf_counter() - started
            trace_model_throughput(
                kind="wan.video",
                model_id=str(request.high_noise_model_id or ""),
                model_name=f"{request.high_noise_model_id} + {request.low_noise_model_id}",
                app_version=__version__,
                elapsed_seconds=elapsed,
                units=max(1, int(frame_count)),
                units_label="frames",
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
                video_postprocess_seconds=round(video_postprocess_seconds, 3),
                offload_cleanup_seconds=round(offload_cleanup_seconds, 3),
                postprocess_seconds=round(postprocess_seconds, 3),
                video_write_seconds=round(video_write_seconds, 3),
                steps_per_second=round(steps_per_second, 6) if steps_per_second is not None else None,
                iterations_per_second=round(iterations_per_second, 6)
                if iterations_per_second is not None
                else None,
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
                video_postprocess_seconds=round(video_postprocess_seconds, 3),
                offload_cleanup_seconds=round(offload_cleanup_seconds, 3),
                postprocess_seconds=round(postprocess_seconds, 3),
                video_write_seconds=round(video_write_seconds, 3),
                steps_per_second=round(steps_per_second, 6) if steps_per_second is not None else None,
                iterations_per_second=round(iterations_per_second, 6)
                if iterations_per_second is not None
                else None,
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
