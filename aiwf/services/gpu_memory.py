from __future__ import annotations

import gc
import logging
import os
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiwf.bootstrap import AppContext

logger = logging.getLogger(__name__)


def nvidia_smi_free_bytes(logical_device_index: int = 0) -> int | None:
    """Read free VRAM for a worker's logical CUDA device without importing Torch."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    expected_index: int | None = None
    expected_uuid = ""
    if visible is None:
        expected_index = logical_device_index
    else:
        tokens = [item.strip().lower() for item in visible.split(",")]
        if logical_device_index < 0 or logical_device_index >= len(tokens):
            return None
        token = tokens[logical_device_index]
        if token.isdigit():
            expected_index = int(token)
        elif token.startswith("gpu-"):
            expected_uuid = token.removeprefix("gpu-")
        else:
            return None

    matches: list[int] = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        try:
            physical_index = int(fields[0])
            free_bytes = int(fields[2]) * 1024**2
        except ValueError:
            continue
        row_uuid = fields[1].lower().removeprefix("gpu-")
        if expected_uuid and row_uuid == expected_uuid:
            matches.append(free_bytes)
        elif not expected_uuid and expected_index is not None and physical_index == expected_index:
            matches.append(free_bytes)
    return matches[0] if len(matches) == 1 else None


def measured_cuda_free_bytes(torch_module, device=None) -> int | None:
    """Return conservative device-wide free VRAM, or None if it cannot be matched."""
    try:
        torch_free, _total = torch_module.cuda.mem_get_info(device) if device is not None else torch_module.cuda.mem_get_info()
        torch_free = int(torch_free)
    except Exception:
        logger.debug("Could not read CUDA free memory through PyTorch", exc_info=True)
        return None

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    logical_index = getattr(device, "index", None)
    if logical_index is None:
        try:
            logical_index = int(torch_module.cuda.current_device())
        except Exception:
            logical_index = None
    try:
        props = torch_module.cuda.get_device_properties(device) if device is not None else torch_module.cuda.get_device_properties(logical_index)
        uuid = str(getattr(props, "uuid", "") or "").strip().lower().removeprefix("gpu-")
    except Exception:
        uuid = ""

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return torch_free

    expected_physical = None
    if visible is not None and logical_index is not None:
        tokens = [item.strip() for item in visible.split(",")]
        if 0 <= logical_index < len(tokens):
            token = tokens[logical_index].lower()
            if token.isdigit():
                expected_physical = int(token)
            elif token.startswith("gpu-"):
                visible_uuid = token.removeprefix("gpu-")
                if uuid and visible_uuid != uuid:
                    logger.warning("CUDA_VISIBLE_DEVICES UUID disagrees with the active PyTorch CUDA device")
                    return None
                uuid = visible_uuid
    elif visible is None:
        expected_physical = logical_index

    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            continue
        try:
            physical_index = int(fields[0])
            free_bytes = int(fields[2]) * 1024**2
        except ValueError:
            continue
        row_uuid = fields[1].lower().removeprefix("gpu-")
        if uuid and row_uuid == uuid:
            return min(torch_free, free_bytes)
        if not uuid and expected_physical is not None and physical_index == expected_physical:
            return min(torch_free, free_bytes)
    logger.warning("Could not match active CUDA device to nvidia-smi device inventory")
    return None


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
            free = measured_cuda_free_bytes(torch)
            return free / (1024**3) if free is not None else None
    except Exception:
        logger.debug("Could not read free VRAM", exc_info=True)
    return None


def unload_all_gpu_models(ctx: AppContext) -> str:
    from aiwf.core.domain.engine import EngineSwitchRequest, EngineTenant
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
