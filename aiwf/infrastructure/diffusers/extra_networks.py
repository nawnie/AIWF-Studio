from __future__ import annotations

import logging
from pathlib import Path

from aiwf.core.domain.extra_networks import LoraRef
from aiwf.core.domain.models import LoraInfo
from aiwf.infrastructure.diffusers.loras import resolve_lora

logger = logging.getLogger(__name__)


def apply_loras(pipe, loras: list[LoraRef], catalog: list[LoraInfo]) -> list[str]:
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