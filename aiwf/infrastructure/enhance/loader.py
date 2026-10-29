from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_extra_arches_loaded = False


def _register_extra_arches() -> None:
    global _extra_arches_loaded
    if _extra_arches_loaded:
        return
    try:
        import spandrel
        import spandrel_extra_arches

        spandrel.MAIN_REGISTRY.add(*spandrel_extra_arches.EXTRA_REGISTRY)
        _extra_arches_loaded = True
    except Exception:
        logger.warning("spandrel_extra_arches not available — CodeFormer models may not load", exc_info=True)
        _extra_arches_loaded = True


def load_spandrel_model(path: str, *, device: torch.device, prefer_half: bool = True):
    from spandrel import ModelLoader

    _register_extra_arches()
    descriptor = ModelLoader(device=device).load_from_file(path)
    if prefer_half and getattr(descriptor, "supports_half", False):
        descriptor.model.half()
    descriptor.model.to(device)
    descriptor.model.eval()
    return descriptor