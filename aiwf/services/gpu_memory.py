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


def _vram_free_gb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            return free / (1024**3)
    except Exception:
        logger.debug("Could not read free VRAM", exc_info=True)
    return None


def unload_all_gpu_models(ctx: AppContext) -> str:
    unloaded: list[str] = []
    errors: list[str] = []
    before = _vram_free_gb()

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

    try:
        from aiwf.services.ltx_diffusers import unload_ltx2b_diffusers_cache

        if unload_ltx2b_diffusers_cache():
            unloaded.append("LTX 2B Diffusers")
    except Exception as exc:
        logger.exception("Failed to unload LTX 2B Diffusers pipeline")
        errors.append(f"LTX 2B Diffusers ({exc})")

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
        status = f"**Model unload failed:** {'; '.join(errors)}"
    elif errors:
        status = (
            f"**Partially unloaded GPU models:** {', '.join(unloaded)}. "
            f"Some engines reported errors: {'; '.join(errors)}"
        )
    elif not unloaded:
        status = "**No GPU models were loaded.** VRAM cache was cleared anyway."
    else:
        status = f"**Unloaded GPU models:** {', '.join(unloaded)}."

    after = _vram_free_gb()
    if before is not None and after is not None:
        status += (
            f" GPU memory free: {before:.1f} → {after:.1f} GB "
            f"(reclaimed {max(0.0, after - before):.1f} GB)."
        )
    return status
