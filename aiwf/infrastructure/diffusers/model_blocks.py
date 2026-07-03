from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BlockedImageAsset:
    status: str
    reason: str
    suggested_action: str


_KNOWN_BROKEN_SELECTABLE_IMAGE_ASSETS: dict[str, BlockedImageAsset] = {
    "fluxedupfluxnsfw_110fp8.safetensors": BlockedImageAsset(
        status="broken-runtime",
        reason="Known Flux FP8 selectable failure: checkpoint keys do not match the expected Flux loader schema.",
        suggested_action="Keep blocked until key mapping/loading support is fixed.",
    ),
    "fluxfusionv24stepsggufnf4_v2ggufq4km.gguf": BlockedImageAsset(
        status="broken-runtime",
        reason="Known Flux GGUF/NF4 mismatch: metadata/quantization does not match the current image route.",
        suggested_action="Do not expose as a normal Flux checkpoint until a compatible GGUF/NF4 route exists.",
    ),
    "snofssexnudesandotherfunstuff_distilledv12fp8.safetensors": BlockedImageAsset(
        status="broken-runtime",
        reason="Known Flux/Flux.2 loader-schema mismatch: checkpoint keys do not match the standard Diffusers converter.",
        suggested_action="Keep blocked until this checkpoint has a dedicated loader or a compatible export.",
    ),
    "4xbhi_dat2_multiblurjpg.safetensors": BlockedImageAsset(
        status="broken-runtime",
        reason="Known bad selectable: checkpoint is missing the expected CLIP text model.",
        suggested_action="Classify as an auxiliary/upscale asset instead of a txt2img checkpoint.",
    ),
}

_NON_SELECTABLE_IMAGE_ASSET_DIRS = {
    "inpaint",
    "ultralytics",
    "upscale_models",
}


def known_broken_selectable_image_asset(path: Path | str) -> BlockedImageAsset | None:
    resolved = Path(path)
    entry = _KNOWN_BROKEN_SELECTABLE_IMAGE_ASSETS.get(resolved.name.lower())
    if entry is not None:
        return entry
    # Z-Image GGUF needs fused GGUF CUDA kernels (Hub repo Isotr0py/ggml),
    # which only ship Linux builds. The Windows fallback dequantizes layer by
    # layer, exhausts 16 GB VRAM, and falls into very slow system-memory paging.
    # Blocked until a Windows kernel build or a bf16/FP8 Z-Image transformer route exists.
    if os.name == "nt" and resolved.suffix.lower() == ".gguf":
        compact_name = resolved.name.lower().replace("-", "").replace("_", "")
        in_zimage_dir = any(part.lower() in {"z-image", "zimage"} for part in resolved.parts)
        if in_zimage_dir or "zimage" in compact_name:
            return BlockedImageAsset(
                status="blocked-cleanly",
                reason=(
                    "Z-Image GGUF is unusable on Windows: the fused GGUF CUDA kernels are Linux-only "
                    "and the fallback dequant path exhausts VRAM on 16 GB GPUs."
                ),
                suggested_action="Use a bf16/FP8 Z-Image transformer instead, or run this route on Linux.",
            )
    return None


def is_non_selectable_image_asset_path(path: Path | str) -> bool:
    return any(part.lower() in _NON_SELECTABLE_IMAGE_ASSET_DIRS for part in Path(path).parts)


def is_blocked_selectable_image_asset(path: Path | str) -> bool:
    return known_broken_selectable_image_asset(path) is not None or is_non_selectable_image_asset_path(path)
