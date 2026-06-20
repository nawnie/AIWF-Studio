from __future__ import annotations

import logging
from pathlib import Path

from aiwf.core.domain.extra_networks import LoraRef
from aiwf.core.domain.models import LoraInfo
from aiwf.infrastructure.diffusers.model_arch import (
    ARCH_INPAINT,
    ARCH_SD15,
    ARCH_SD35,
    ARCH_SDXL,
    is_sd3_architecture,
    is_sdxl_architecture,
)
from aiwf.infrastructure.diffusers.loras import resolve_lora

logger = logging.getLogger(__name__)


def _lora_compatible(base_architecture: str | None, lora_architecture: str | None) -> bool:
    lora_arch = (lora_architecture or "unknown").lower()
    if lora_arch in {"", "unknown"}:
        return True
    base_arch = (base_architecture or "unknown").lower()
    if lora_arch == ARCH_SDXL:
        return is_sdxl_architecture(base_arch)
    if lora_arch == ARCH_SD15:
        return base_arch in {ARCH_SD15, ARCH_INPAINT}
    if lora_arch in {ARCH_SD35, "sd3"}:
        return is_sd3_architecture(base_arch)
    return lora_arch == base_arch


def apply_loras(
    pipe,
    loras: list[LoraRef],
    catalog: list[LoraInfo],
    *,
    base_architecture: str | None = None,
) -> list[str]:
    """Load LoRA adapters onto an active diffusers pipeline. Returns adapter names."""
    if not loras:
        return []

    adapter_names: list[str] = []
    adapter_weights: list[float] = []

    # Pipelines share their UNet (txt2img/img2img/hires), so an adapter loaded
    # via one pipe is already registered when the next pipe asks for it.
    existing = set(getattr(getattr(pipe, "unet", None), "peft_config", None) or {})

    for index, ref in enumerate(loras):
        match = resolve_lora(catalog, ref.name)
        if match is None:
            logger.warning("LoRA not found: %s", ref.name)
            continue
        if not _lora_compatible(base_architecture, match.architecture):
            raise ValueError(
                f"LoRA '{match.title}' targets {match.architecture}, "
                f"but the selected checkpoint is {base_architecture or 'unknown'}."
            )

        adapter_name = f"aiwf_lora_{index}"
        path = Path(match.path)
        if adapter_name in existing:
            adapter_names.append(adapter_name)
            adapter_weights.append(ref.weight)
            logger.debug("LoRA adapter %s already registered; reusing", adapter_name)
            continue
        try:
            if path.is_file():
                pipe.load_lora_weights(str(path.parent), weight_name=path.name, adapter_name=adapter_name)
            else:
                pipe.load_lora_weights(match.path, adapter_name=adapter_name)
            adapter_names.append(adapter_name)
            adapter_weights.append(ref.weight)
            logger.info("Loaded LoRA %s at weight %.2f", match.title, ref.weight)
        except Exception:
            logger.exception("Failed to load LoRA %s", match.title)

    if adapter_names:
        pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)

    return adapter_names


def clear_loras(pipe) -> None:
    try:
        if hasattr(pipe, "unload_lora_weights"):
            pipe.unload_lora_weights()
    except Exception:
        logger.exception("Failed to unload LoRA weights")
