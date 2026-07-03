from __future__ import annotations

import json
import logging
import struct
from pathlib import Path

logger = logging.getLogger(__name__)

# Key names aligned with diffusers single-file detection (A1111 sd_models_config.py logic).
UNET_INPUT_KEY = "model.diffusion_model.input_blocks.0.0.weight"
SDXL_OPENCLIP_KEY = "conditioner.embedders.1.model.ln_final.weight"
SDXL_BASE_KEY = "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.bias"
SD3_JOINT_BLOCK_MARKER = "joint_blocks.0"
SD3_DIFFUSERS_BLOCK_MARKER = "transformer_blocks.0"

ARCH_SD15 = "sd15"
ARCH_INPAINT = "inpaint"
ARCH_SDXL = "sdxl"
ARCH_SDXL_INPAINT = "sdxl_inpaint"
ARCH_SDXL_REFINER = "sdxl_refiner"
ARCH_SD35 = "sd35"
ARCH_FLUX = "flux"
ARCH_FLUX_FILL = "flux_fill"
ARCH_FLUX_KONTEXT = "flux_kontext"
ARCH_FLUX2_KLEIN = "flux2_klein"
ARCH_Z_IMAGE = "z_image"
ARCH_QWEN_IMAGE = "qwen_image"
ARCH_QWEN_IMAGE_NUNCHAKU = "qwen_image_nunchaku"
ARCH_SANA = "sana"
ARCH_SANA_VIDEO = "sana_video"
ARCH_UNKNOWN = "unknown"

_QWEN_NUNCHAKU_MARKERS = ("nunchaku", "svdq-int4", "lightningv", "4steps")


def _is_qwen_nunchaku_name(text: str) -> bool:
    lowered = text.lower().replace("_", "-")
    return any(marker in lowered for marker in _QWEN_NUNCHAKU_MARKERS)


def _architecture_from_name(filename: str) -> str | None:
    lower = filename.lower().replace("_", "-")
    compact = lower.replace("-", "")
    if ("qwen-image" in lower or "qwenimage" in compact or "qwen2.0" in lower) and _is_qwen_nunchaku_name(lower):
        return ARCH_QWEN_IMAGE_NUNCHAKU
    if "qwen-image" in lower or "qwenimage" in compact or "qwen2.0" in lower:
        return ARCH_QWEN_IMAGE
    if "sana-video" in lower or "sanavideo" in compact or "sanaimagetovideo" in compact:
        return ARCH_SANA_VIDEO
    if "sana" in lower:
        return ARCH_SANA
    if "z-image" in lower or "zimage" in compact:
        return ARCH_Z_IMAGE
    if "flux.2" in lower or "flux2" in compact or "klein" in lower:
        return ARCH_FLUX2_KLEIN
    if "kontext" in lower:
        return ARCH_FLUX_KONTEXT
    if "flux" in lower and "fill" in lower:
        return ARCH_FLUX_FILL
    if "flux" in lower:
        return ARCH_FLUX
    if "hunyuan" in lower:
        # Hunyuan (3D/video/image) checkpoints are not runnable in AIWF's
        # image routes; classifying them as unknown keeps them out of the
        # selectable model pickers instead of silently failing as SD 1.5.
        return ARCH_UNKNOWN
    return None


def _safetensors_tensor_shapes(path: Path) -> dict[str, list[int]]:
    """Read tensor shapes from a safetensors header without loading weights."""
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size).decode("utf-8"))
    shapes: dict[str, list[int]] = {}
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        shape = meta.get("shape")
        if shape:
            shapes[key] = list(shape)
    return shapes


def _safetensors_metadata(path: Path) -> dict[str, str]:
    """Read safetensors metadata without loading tensor data."""
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_size).decode("utf-8"))
    return {str(key): str(value) for key, value in (header.get("__metadata__") or {}).items()}


def _ckpt_tensor_shapes(path: Path) -> dict[str, list[int]]:
    """Best-effort shape map for legacy .ckpt checkpoints."""
    try:
        import torch

        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            return {}
        state_dict = state.get("state_dict", state)
        shapes: dict[str, list[int]] = {}
        for key, value in state_dict.items():
            if hasattr(value, "shape"):
                shapes[key] = list(value.shape)
        return shapes
    except Exception:
        logger.debug("Could not inspect ckpt shapes for %s", path, exc_info=True)
        return {}


def _shapes_for_checkpoint(path: Path) -> dict[str, list[int]]:
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        try:
            return _safetensors_tensor_shapes(path)
        except Exception:
            logger.debug("Could not read safetensors header for %s", path, exc_info=True)
            return {}
    if suffix in {".ckpt", ".pt"}:
        return _ckpt_tensor_shapes(path)
    return {}


def _metadata_architecture_for_checkpoint(path: Path) -> str | None:
    if path.suffix.lower() != ".safetensors":
        return None
    try:
        metadata = _safetensors_metadata(path)
    except Exception:
        logger.debug("Could not read safetensors metadata for %s", path, exc_info=True)
        return None
    text = " ".join(
        metadata.get(key, "")
        for key in (
            "modelspec.architecture",
            "modelspec.description",
            "modelspec.implementation",
            "modelspec.title",
            "ss_base_model_version",
            "ss_sd_model_name",
            # ComfyUI ModelSave stores the source UNETLoader graph here.
            "prompt",
            "workflow",
        )
    )
    return _architecture_from_name(text) if text else None


def infer_architecture_from_shapes(shapes: dict[str, list[int]], *, filename: str = "") -> str:
    """Classify checkpoint architecture from state-dict key shapes."""
    name_arch = _architecture_from_name(filename)
    if name_arch:
        return name_arch
    if not shapes:
        return ARCH_UNKNOWN

    unet_in = shapes.get(UNET_INPUT_KEY)
    has_sdxl = SDXL_OPENCLIP_KEY in shapes or SDXL_BASE_KEY in shapes
    # The SDXL refiner conditions on OpenCLIP as embedder 0 (no CLIP-L at all);
    # base SDXL keeps OpenCLIP at embedder 1. Without this check the refiner's
    # 4-channel UNet input misclassifies it as an SD 1.5 base checkpoint.
    if "conditioner.embedders.0.model.ln_final.weight" in shapes and not has_sdxl:
        return ARCH_SDXL_REFINER
    lower = filename.lower().replace("_", "-")
    has_sd3 = any(
        SD3_JOINT_BLOCK_MARKER in key or key.startswith(SD3_DIFFUSERS_BLOCK_MARKER)
        for key in shapes
    )

    has_flux_blocks = any(
        key.startswith(
            (
                "double_blocks.0",
                "single_blocks.0",
                "transformer.double_blocks.0",
                "transformer.single_blocks.0",
                "model.diffusion_model.double_blocks.0",
                "model.diffusion_model.single_blocks.0",
            )
        )
        for key in shapes
    )
    if has_flux_blocks:
        # Flux Fill (inpaint/outpaint) widens the image input projection to
        # take the masked-image + mask channels: img_in is [3072, 384] instead
        # of the base model's [3072, 64].
        for img_in_key in ("img_in.weight", "model.diffusion_model.img_in.weight", "x_embedder.weight"):
            shape = shapes.get(img_in_key)
            if shape and len(shape) >= 2 and int(shape[1]) == 384:
                return ARCH_FLUX_FILL
        return ARCH_FLUX

    if has_sd3 or "sd3.5" in lower or "sd35" in lower or "stable-diffusion-3.5" in lower:
        return ARCH_SD35

    if unet_in and len(unet_in) >= 2 and unet_in[1] == 9:
        if has_sdxl:
            return ARCH_SDXL_INPAINT
        return ARCH_INPAINT

    # SDXL must be decided BEFORE the 4-channel SD 1.5 check: SDXL's UNet
    # input is also 4-channel, so checking unet_in first silently downgrades
    # every SDXL base checkpoint to SD 1.5.
    if has_sdxl:
        return ARCH_SDXL

    if unet_in and len(unet_in) >= 2 and unet_in[1] == 4:
        return ARCH_SD15

    if "inpaint" in lower:
        return ARCH_INPAINT
    return ARCH_UNKNOWN


def looks_like_lora_weights(path: Path | str) -> bool:
    """True when a safetensors/ckpt file contains LoRA adapters but no full UNet."""
    shapes = _shapes_for_checkpoint(Path(path))
    if not shapes:
        return False
    if UNET_INPUT_KEY in shapes:
        return False
    return any(
        "lora_down" in key
        or "lora_up" in key
        or ".lora_A." in key
        or ".lora_B." in key
        or ".lora_a." in key
        or ".lora_b." in key
        for key in shapes
    )


def looks_like_controlnet_weights(path: Path | str) -> bool:
    """True when a file contains ControlNet weights, not a base checkpoint."""
    shapes = _shapes_for_checkpoint(Path(path))
    if not shapes:
        return False
    return any(
        key.startswith("controlnet_")
        or key.startswith("control_model.")
        or key.startswith("controlnet.")
        for key in shapes
    )


def detect_checkpoint_architecture(path: Path | str) -> str:
    """Detect SD1.5 / SDXL / inpaint variants from checkpoint weights."""
    resolved = Path(path)
    metadata_arch = _metadata_architecture_for_checkpoint(resolved)
    if metadata_arch:
        return metadata_arch
    shapes = _shapes_for_checkpoint(resolved)
    if shapes:
        return infer_architecture_from_shapes(shapes, filename=resolved.name)

    lower = resolved.name.lower()
    normalized = lower.replace("_", "-")
    if "sd3.5" in normalized or "sd35" in normalized or "stable-diffusion-3.5" in normalized:
        return ARCH_SD35
    if normalized.startswith("sd3-") or normalized.startswith("sd3.") or normalized.startswith("sd3_"):
        return ARCH_SD35
    name_arch = _architecture_from_name(resolved.name)
    if name_arch:
        return name_arch
    if "sd15" in normalized or "sd1.5" in normalized or "v1-5" in normalized or "stable-diffusion-v1-5" in normalized:
        return ARCH_SD15
    if "inpaint" in lower and "xl" in lower.replace("_", " "):
        return ARCH_SDXL_INPAINT
    if "inpaint" in lower:
        return ARCH_INPAINT
    if "xl" in lower or "sdxl" in lower:
        return ARCH_SDXL
    return ARCH_UNKNOWN


def architecture_label(architecture: str) -> str:
    return {
        ARCH_SDXL: "SDXL",
        ARCH_SDXL_INPAINT: "SDXL inpaint",
        ARCH_SDXL_REFINER: "SDXL refiner",
        ARCH_SD35: "SD3.5",
        ARCH_FLUX: "Flux",
        ARCH_FLUX_FILL: "Flux Fill (inpaint)",
        ARCH_FLUX_KONTEXT: "Flux Kontext",
        ARCH_FLUX2_KLEIN: "Flux.2 Klein",
        ARCH_Z_IMAGE: "Z-Image",
        ARCH_QWEN_IMAGE: "Qwen Image",
        ARCH_QWEN_IMAGE_NUNCHAKU: "Qwen Image Nunchaku",
        ARCH_SANA: "Sana",
        ARCH_SANA_VIDEO: "Sana Video",
        ARCH_UNKNOWN: "unknown",
        ARCH_INPAINT: "inpaint",
        ARCH_SD15: "SD1.5",
    }.get(architecture, architecture)


def is_inpaint_architecture(architecture: str) -> bool:
    return architecture in {ARCH_INPAINT, ARCH_SDXL_INPAINT, ARCH_FLUX_FILL}


def is_flux_fill_architecture(architecture: str) -> bool:
    return (architecture or "").lower() == ARCH_FLUX_FILL


def is_sdxl_architecture(architecture: str) -> bool:
    return architecture in {ARCH_SDXL, ARCH_SDXL_INPAINT}


def is_sd3_architecture(architecture: str) -> bool:
    return (architecture or "").lower() in {ARCH_SD35, "sd3", "stable-diffusion-3", "stable-diffusion-3.5"}


def is_flux_architecture(architecture: str) -> bool:
    return (architecture or "").lower() == ARCH_FLUX


def is_flux_kontext_architecture(architecture: str) -> bool:
    return (architecture or "").lower() == ARCH_FLUX_KONTEXT


def is_flux2_klein_architecture(architecture: str) -> bool:
    return (architecture or "").lower() == ARCH_FLUX2_KLEIN


def is_z_image_architecture(architecture: str) -> bool:
    return (architecture or "").lower() == ARCH_Z_IMAGE


def is_qwen_image_architecture(architecture: str) -> bool:
    return (architecture or "").lower() in {ARCH_QWEN_IMAGE, ARCH_QWEN_IMAGE_NUNCHAKU}


def is_qwen_nunchaku_architecture(architecture: str) -> bool:
    return (architecture or "").lower() == ARCH_QWEN_IMAGE_NUNCHAKU


def is_sana_architecture(architecture: str) -> bool:
    return (architecture or "").lower() == ARCH_SANA


def is_sana_video_architecture(architecture: str) -> bool:
    return (architecture or "").lower() == ARCH_SANA_VIDEO


def is_transformer_image_architecture(architecture: str) -> bool:
    return (architecture or "").lower() in {
        ARCH_FLUX,
        ARCH_FLUX_KONTEXT,
        ARCH_FLUX2_KLEIN,
        ARCH_Z_IMAGE,
        ARCH_QWEN_IMAGE,
        ARCH_QWEN_IMAGE_NUNCHAKU,
        ARCH_SANA,
    }
