"""ControlNet pipeline construction for the diffusers backend.

Isolated here so the rest of the backend stays readable. Loads a
``ControlNetModel`` from a single-file checkpoint (cached by path) and builds a
ControlNet pipeline that reuses an already-loaded base pipeline's components, so
switching ControlNet on/off never reloads the base checkpoint.

This module touches torch/diffusers and therefore runs only with the GPU stack
installed; it is import-guarded and unit-tested at the structural level.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLControlNetPipeline,
)

logger = logging.getLogger(__name__)

# Scaffold config for SD1.5 Control LoRA checkpoints (rank128 files from
# comfyanonymous/ControlNet-v1-1_fp16_safetensors). Architecture is identical
# across v1.1 types; the LoRA file carries the specialized weights.
_SD15_CONTROLNET_CONFIG = "lllyasviel/sd-controlnet-canny"


def is_control_lora_checkpoint(path: str | Path) -> bool:
    """True when the file is a SAI-style SD1.5 Control LoRA (rank128) checkpoint."""
    name = Path(path).name.lower()
    if "control_lora" in name:
        return True
    try:
        import safetensors.torch as st

        keys = st.load_file(str(path)).keys()
        return "lora_controlnet" in keys
    except Exception:
        return False


def _load_control_lora_checkpoint(path: Path, *, dtype: torch.dtype) -> ControlNetModel:
    config = ControlNetModel.load_config(_SD15_CONTROLNET_CONFIG)
    model = ControlNetModel.from_config(config)
    model = model.to(dtype=dtype)
    model.load_lora_adapter(str(path), prefix=None)
    return model


class ControlNetModelCache:
    """Loads and caches ControlNetModel weights by checkpoint path."""

    def __init__(self) -> None:
        self._cache: dict[str, ControlNetModel] = {}

    def load(self, path: str, *, dtype: torch.dtype) -> ControlNetModel:
        resolved = Path(path).resolve()
        key = str(resolved)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        if is_control_lora_checkpoint(resolved):
            logger.info("Loading ControlNet LoRA %s", path)
            model = _load_control_lora_checkpoint(resolved, dtype=dtype)
        else:
            logger.info("Loading ControlNet model %s", path)
            model = ControlNetModel.from_single_file(str(resolved), torch_dtype=dtype)
        self._cache[key] = model
        return model

    def clear(self) -> None:
        self._cache.clear()


def _is_sdxl(base_pipe) -> bool:
    return hasattr(base_pipe, "text_encoder_2") and base_pipe.text_encoder_2 is not None


def build_controlnet_pipeline(
    base_pipe,
    controlnet: ControlNetModel | list[ControlNetModel],
    *,
    img2img: bool,
):
    """Build a ControlNet pipeline reusing the base pipeline's components.

    The returned pipeline shares the base UNet/VAE/text-encoders (no reload) and
    adds the ControlNet conditioning branch.
    """
    if _is_sdxl(base_pipe):
        cls = (
            StableDiffusionXLControlNetImg2ImgPipeline
            if img2img
            else StableDiffusionXLControlNetPipeline
        )
        return cls(
            vae=base_pipe.vae,
            text_encoder=base_pipe.text_encoder,
            text_encoder_2=base_pipe.text_encoder_2,
            tokenizer=base_pipe.tokenizer,
            tokenizer_2=base_pipe.tokenizer_2,
            unet=base_pipe.unet,
            controlnet=controlnet,
            scheduler=base_pipe.scheduler,
        )

    cls = (
        StableDiffusionControlNetImg2ImgPipeline
        if img2img
        else StableDiffusionControlNetPipeline
    )
    return cls(
        vae=base_pipe.vae,
        text_encoder=base_pipe.text_encoder,
        tokenizer=base_pipe.tokenizer,
        unet=base_pipe.unet,
        controlnet=controlnet,
        scheduler=base_pipe.scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
