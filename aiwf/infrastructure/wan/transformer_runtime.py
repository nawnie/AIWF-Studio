"""Wan transformer format detection and loader routing.

ComfyUI only splits at **load** time; everything after that is shared:

- ``UNETLoader`` (safetensors/FP8) vs ``UnetLoaderGGUF`` (mmap + ``GGMLOps``)
- Same ``MODEL`` type → same KSampler / denoise loop → same VAE decode

AIWF should mirror that: format-specific transformer **loader**, shared sampler path.

Today the shared path is diffusers ``WanImageToVideoPipeline`` (UMT5, scheduler,
dual-stage denoise, VAE decode). GGUF needs its own loader into a quantized Wan
transformer backend — not full dequant into diffusers weights.
"""
from __future__ import annotations

import os
from enum import Enum
from pathlib import Path


class WanTransformerFormat(str, Enum):
    """How a standalone Wan transformer weight file should be executed."""

    DIFFUSERS_SAFETENSORS = "diffusers_safetensors"
    COMFY_FP8_SAFETENSORS = "comfy_fp8_safetensors"
    GGUF_QUANTIZED = "gguf_quantized"


_COMFY_FP8_METADATA_SUFFIXES = (
    ".comfy_quant",
    ".weight_scale",
    ".weight_scale_2",
    ".pre_quant_scale",
    ".input_scale",
    ".scale_weight",
    ".scale_input",
)


def detect_transformer_format(path: Path) -> WanTransformerFormat:
    """Classify a local Wan transformer file for runtime routing."""
    suffix = path.suffix.lower()
    if suffix == ".gguf":
        return WanTransformerFormat.GGUF_QUANTIZED
    if suffix != ".safetensors":
        return WanTransformerFormat.DIFFUSERS_SAFETENSORS
    try:
        from safetensors import safe_open

        saw_fp8 = False
        saw_scale = False
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                key_l = key.lower()
                if any(key_l.endswith(s) for s in _COMFY_FP8_METADATA_SUFFIXES):
                    saw_scale = True
                try:
                    dtype = handle.get_slice(key).get_dtype()
                except Exception:
                    continue
                if str(dtype).upper().startswith("F8"):
                    saw_fp8 = True
        if saw_fp8 and saw_scale:
            return WanTransformerFormat.COMFY_FP8_SAFETENSORS
    except Exception:
        pass
    return WanTransformerFormat.DIFFUSERS_SAFETENSORS


def estimate_gguf_expanded_gb(path: Path) -> float:
    """Rough host RAM if GGUF were fully dequantized (legacy diffusers stub path)."""
    try:
        size_gb = path.stat().st_size / (1024**3)
    except OSError:
        return 0.0
    return size_gb * 4.5


def gguf_quantized_runtime_enabled() -> bool:
    """True when the mmap + on-the-fly dequant GGUF backend is available."""
    if os.environ.get("AIWF_WAN_GGUF_RUNTIME", "").strip().lower() in {"0", "false", "no", "off"}:
        return False
    try:
        from importlib.util import find_spec

        return find_spec("gguf") is not None
    except Exception:
        return False


def gguf_dequant_stub_allowed() -> bool:
    """Dev-only escape hatch: expand GGUF to bf16 diffusers weights (slow, huge RAM)."""
    return os.environ.get("AIWF_WAN_ALLOW_EXPENSIVE_DEQUANT", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def gguf_unavailable_message(path: Path, *, label: str = "Transformer") -> str:
    """User-facing explanation when GGUF cannot run."""
    if not gguf_quantized_runtime_enabled():
        return (
            f"{label} GGUF ({path.name}) requires the optional `gguf` package. "
            "Install it (`pip install gguf`) and restart AIWF Studio."
        )
    expanded_gb = estimate_gguf_expanded_gb(path)
    return (
        f"{label} GGUF ({path.name}) could not use the quantized runtime. "
        f"Full dequant expand would need ~{expanded_gb:.0f} GB RAM. "
        "Use FP8 `.safetensors` high/low, or set AIWF_WAN_ALLOW_EXPENSIVE_DEQUANT=1 for the legacy stub."
    )


def require_diffusers_transformer_path(path: Path, *, label: str = "Transformer") -> WanTransformerFormat:
    """Return format for the Wan transformer loader route; raise when unloadable."""
    from aiwf.infrastructure.wan.pipeline import WanUnavailable

    fmt = detect_transformer_format(path)
    if fmt != WanTransformerFormat.GGUF_QUANTIZED:
        return fmt
    if gguf_quantized_runtime_enabled():
        return fmt
    if gguf_dequant_stub_allowed():
        return fmt
    raise WanUnavailable(gguf_unavailable_message(path, label=label))


def describe_comfy_launcher_parity() -> list[str]:
    """Hints aligning AIWF architecture with Comfy's separate loaders."""
    return [
        "Comfy: UNETLoader vs UnetLoaderGGUF — loader only; same MODEL, KSampler, VAE decode after.",
        "AIWF: diffusers pipeline is the shared sampler; safetensors/FP8 and GGUF each have their own loader.",
        "VSR / NVIDIA toolkit / aimdo are Comfy model-management hooks; they do not apply to diffusers Wan yet.",
    ]