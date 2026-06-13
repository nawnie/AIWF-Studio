from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from importlib.util import find_spec
from pathlib import Path
from typing import Callable

from PIL import Image

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.wan import WAN_TI2V_5B, WanI2VRequest, WanI2VResult
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


@dataclass(frozen=True)
class WanPreflightResult:
    ok: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    components_base: str | None = None
    high_noise_model: str | None = None
    low_noise_model: str | None = None
    vae: str | None = None
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
    ) -> None:
        self.flags = flags
        self.settings = settings
        self._backend = WanI2VBackend()
        self._unload_image_models = unload_image_models

    def available(self) -> bool:
        return self._backend.available()

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
            if find_spec("gguf") is None:
                errors.append(f"{label} transformer is GGUF, but the optional `gguf` package is not installed.")
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
            f"**Wan models** → `{self.models_dir()}`.  \n"
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

        if image is None:
            raise WanUnavailable("Upload a source image to animate.")
        if not self.available():
            raise WanUnavailable(
                "Wan video is unavailable — update `diffusers` (>=0.35) and install `ftfy`, then restart."
            )
        if self._unload_image_models is not None:
            _video_status("Unloading image models before loading video pipeline.")
            try:
                self._unload_image_models()
            except Exception:
                logger.exception("Failed to unload image models before Wan generation; continuing.")

        # Expose HF token so large/gated model downloads succeed.
        try:
            self.settings.apply_token_env()
        except Exception:
            pass

        # Wan 2.2 I2V always runs a two-stage high-noise + low-noise transformer pair.
        # Both must be selected — there is no single-model path (true even with LoRAs).
        if not (request.high_noise_model_id and request.low_noise_model_id):
            raise WanUnavailable(
                "Select both a High noise model and a Low noise model. Wan 2.2 image-to-video "
                "always runs a two-stage high/low transformer pair (required even when using LoRAs)."
            )

        high_res = preflight.high_noise_model
        low_res = preflight.low_noise_model
        vae_res = preflight.vae
        high_lora_res = preflight.high_noise_lora
        low_lora_res = preflight.low_noise_lora
        try:
            components_base = preflight.components_base
        except WanUnavailable as exc:
            logger.error(
                "Wan video generation unavailable: %s (high=%s low=%s vae=%s high_lora=%s low_lora=%s)",
                exc,
                high_res,
                low_res,
                vae_res,
                high_lora_res,
                low_lora_res,
            )
            raise

        req = request.model_copy(update={
            "high_noise_model_id": high_res,
            "low_noise_model_id": low_res,
            "vae_id": vae_res,
            "high_noise_lora_id": high_lora_res,
            "low_noise_lora_id": low_lora_res,
            "components_base": components_base,
            "steps": request.effective_steps(),
            "boundary_ratio": request.effective_boundary_ratio(),
        })

        start = time.perf_counter()
        try:
            frames, width, height = self._backend.generate(
                req, image, on_progress=on_progress, should_cancel=should_cancel
            )
        except WanUnavailable as exc:
            logger.error(
                "Wan video generation unavailable: %s (high=%s low=%s vae=%s high_lora=%s low_lora=%s)",
                exc,
                high_res,
                low_res,
                vae_res,
                high_lora_res,
                low_lora_res,
            )
            raise
        except Exception:
            logger.exception(
                "Wan video generation failed (high=%s low=%s vae=%s high_lora=%s low_lora=%s)",
                high_res,
                low_res,
                vae_res,
                high_lora_res,
                low_lora_res,
            )
            raise
        elapsed = time.perf_counter() - start

        out = self._output_path()
        count = write_frames(frames, out, fps=request.fps)
        return WanI2VResult(
            output_path=str(out),
            frame_count=count,
            fps=request.fps,
            width=width,
            height=height,
            elapsed_seconds=elapsed,
            message=f"{count} frames at {width}x{height}, {request.fps} fps in {elapsed:.0f}s",
        )
