"""CUDA VRAM budget helpers.

This is intentionally a PyTorch allocator limit, not a driver-level partition.
It makes AIWF's future CUDA allocations fail at a lower ceiling so the user can
leave headroom for the desktop, games, training, or another local process.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any


@dataclass(frozen=True)
class CudaVramReserveResult:
    enabled: bool
    applied: bool
    requested_reserve_mb: int = 0
    total_mb: int = 0
    limit_mb: int = 0
    fraction: float = 1.0
    message: str = ""


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return max(0, int(raw))
    except ValueError:
        return None


def _env_reserve_mb() -> int | None:
    for name in ("AIWF_CUDA_VRAM_RESERVE_MB", "AIWF_WAN_VRAM_RESERVE_MB"):
        env_value = _env_int(name)
        if env_value is not None:
            return env_value
    return None


def resolve_cuda_vram_reserve_mb(requested_mb: int | None = None) -> int:
    """Resolve env override first, then the caller value."""
    env_value = _env_reserve_mb()
    if env_value is not None:
        return env_value
    try:
        return max(0, int(requested_mb or 0))
    except (TypeError, ValueError):
        return 0


def apply_cuda_vram_reserve(
    *,
    enabled: bool,
    reserve_mb: int | None = None,
    device: int | str | Any = 0,
    torch_module: Any | None = None,
) -> CudaVramReserveResult:
    """Apply a process-local CUDA allocator cap.

    If disabled, this resets PyTorch's per-process CUDA memory fraction to 1.0
    so a prior capped generation does not leave a sticky limit.
    """
    env_requested = _env_reserve_mb()
    requested = resolve_cuda_vram_reserve_mb(reserve_mb)
    enabled = bool(
        (env_requested is not None and requested > 0)
        or (enabled and requested > 0)
    )

    try:
        torch = torch_module
        if torch is None:
            import torch as torch  # type: ignore[no-redef]
    except Exception as exc:
        return CudaVramReserveResult(
            enabled=enabled,
            applied=False,
            requested_reserve_mb=requested,
            message=f"PyTorch import failed: {exc}",
        )

    try:
        if not torch.cuda.is_available():
            return CudaVramReserveResult(
                enabled=enabled,
                applied=False,
                requested_reserve_mb=requested,
                message="CUDA is not available.",
            )
        props = torch.cuda.get_device_properties(device)
        total_mb = max(1, int(getattr(props, "total_memory", 0)) // (1024 * 1024))
    except Exception as exc:
        return CudaVramReserveResult(
            enabled=enabled,
            applied=False,
            requested_reserve_mb=requested,
            message=f"CUDA memory query failed: {exc}",
        )

    if not enabled:
        try:
            torch.cuda.set_per_process_memory_fraction(1.0, device=device)
        except Exception as exc:
            return CudaVramReserveResult(
                enabled=False,
                applied=False,
                total_mb=total_mb,
                limit_mb=total_mb,
                fraction=1.0,
                message=f"Could not reset CUDA memory fraction: {exc}",
            )
        return CudaVramReserveResult(
            enabled=False,
            applied=True,
            requested_reserve_mb=requested,
            total_mb=total_mb,
            limit_mb=total_mb,
            fraction=1.0,
            message=f"CUDA VRAM reserve disabled; AIWF allocator limit restored to {total_mb} MB.",
        )

    # Leave at least 256 MB visible to avoid invalid/zero fractions if the user
    # enters an overly aggressive reserve.
    clamped_reserve = min(requested, max(0, total_mb - 256))
    limit_mb = max(256, total_mb - clamped_reserve)
    fraction = max(0.01, min(1.0, float(limit_mb) / float(total_mb)))
    try:
        torch.cuda.set_per_process_memory_fraction(fraction, device=device)
    except Exception as exc:
        return CudaVramReserveResult(
            enabled=True,
            applied=False,
            requested_reserve_mb=clamped_reserve,
            total_mb=total_mb,
            limit_mb=limit_mb,
            fraction=fraction,
            message=f"Could not apply CUDA VRAM reserve: {exc}",
        )

    return CudaVramReserveResult(
        enabled=True,
        applied=True,
        requested_reserve_mb=clamped_reserve,
        total_mb=total_mb,
        limit_mb=limit_mb,
        fraction=fraction,
        message=(
            f"CUDA VRAM reserve active: {clamped_reserve} MB reserved, "
            f"AIWF allocator limit {limit_mb}/{total_mb} MB ({fraction:.3f})."
        ),
    )
