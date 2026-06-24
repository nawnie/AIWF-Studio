from __future__ import annotations

import gc
import logging

from aiwf.bootstrap import AppContext
from aiwf.core.domain.engine import EngineSwitchRequest, EngineTenant

logger = logging.getLogger(__name__)


def flush_vram() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        logger.debug("VRAM flush failed", exc_info=True)


def unload_all_gpu_models(ctx: AppContext) -> str:
    unloaded: list[str] = []
    errors: list[str] = []

    try:
        ctx.generation.backend.unload()
        unloaded.append("image pipeline")
    except Exception as exc:
        logger.exception("Failed to unload image pipeline")
        errors.append(f"image pipeline ({exc})")

    try:
        from aiwf.web.tabs.wan_i2v import unload_wan_for_context

        if unload_wan_for_context(ctx):
            unloaded.append("Wan video")
    except Exception as exc:
        logger.exception("Failed to unload Wan video pipeline")
        errors.append(f"Wan video ({exc})")

    for label, unload_fn in (
        ("Enhance", ctx.enhance.unload_models),
        ("Face swap", ctx.faceswap.unload),
        ("Segment", ctx.segment.unload),
    ):
        try:
            unload_fn()
            unloaded.append(label)
        except Exception as exc:
            logger.exception("Failed to unload %s models", label)
            errors.append(f"{label} ({exc})")

    flush_vram()

    try:
        ctx.supervisor.request_switch(
            EngineSwitchRequest(
                target=EngineTenant.IDLE,
                reason="Manual GPU model unload from Settings",
            )
        )
    except Exception:
        logger.debug("Could not release GPU tenant after manual unload", exc_info=True)

    if not unloaded and errors:
        return f"**Model unload failed:** {'; '.join(errors)}"
    if errors:
        return (
            f"**Partially unloaded GPU models:** {', '.join(unloaded)}. "
            f"Some engines reported errors: {'; '.join(errors)}"
        )
    if not unloaded:
        return "**No GPU models were loaded.** VRAM cache was cleared anyway."
    return f"**Unloaded GPU models:** {', '.join(unloaded)}."