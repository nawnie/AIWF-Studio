from __future__ import annotations

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
    return _KNOWN_BROKEN_SELECTABLE_IMAGE_ASSETS.get(Path(path).name.lower())


def is_non_selectable_image_asset_path(path: Path | str) -> bool:
    return any(part.lower() in _NON_SELECTABLE_IMAGE_ASSET_DIRS for part in Path(path).parts)


def is_blocked_selectable_image_asset(path: Path | str) -> bool:
    return known_broken_selectable_image_asset(path) is not None or is_non_selectable_image_asset_path(path)
