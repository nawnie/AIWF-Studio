"""Memory/cache policy for AIWF-native Wan high/low stages."""
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class WanStageCacheMode(str, Enum):
    NONE = "none"
    FULL = "full"
    GPU_ACTIVE_CPU_PINNED_STANDBY = "gpu_active_cpu_pinned_standby"
    GPU_ACTIVE_CPU_UNPINNED_STANDBY = "gpu_active_cpu_unpinned_standby"
    DISK_SEQUENTIAL = "disk_sequential"


@dataclass(frozen=True)
class WanStageCacheDecision:
    mode: WanStageCacheMode
    reason: str
    available_ram_gb: float | None = None
    additional_required_gb: float | None = None
    low_stage_estimate_gb: float | None = None
    safety_margin_gb: float = 6.0

    @property
    def uses_disk_at_boundary(self) -> bool:
        return self.mode == WanStageCacheMode.DISK_SEQUENTIAL


def _parse_mode(mode: str | WanStageCacheMode) -> WanStageCacheMode:
    if isinstance(mode, WanStageCacheMode):
        return mode
    return WanStageCacheMode(str(mode))


def stage_cache_uses_cpu_standby(mode: str | WanStageCacheMode) -> bool:
    parsed = _parse_mode(mode)
    return parsed in {
        WanStageCacheMode.FULL,
        WanStageCacheMode.GPU_ACTIVE_CPU_PINNED_STANDBY,
        WanStageCacheMode.GPU_ACTIVE_CPU_UNPINNED_STANDBY,
        WanStageCacheMode.DISK_SEQUENTIAL,
    }


def stage_cache_pins_tensors(mode: str | WanStageCacheMode) -> bool:
    parsed = _parse_mode(mode)
    return parsed in {
        WanStageCacheMode.FULL,
        WanStageCacheMode.GPU_ACTIVE_CPU_PINNED_STANDBY,
    }


def stage_cache_is_gpu_active_cpu_standby(mode: str | WanStageCacheMode) -> bool:
    parsed = _parse_mode(mode)
    return parsed in {
        WanStageCacheMode.GPU_ACTIVE_CPU_PINNED_STANDBY,
        WanStageCacheMode.GPU_ACTIVE_CPU_UNPINNED_STANDBY,
        WanStageCacheMode.DISK_SEQUENTIAL,
    }


def stage_cache_is_disk_sequential(mode: str | WanStageCacheMode) -> bool:
    return _parse_mode(mode) == WanStageCacheMode.DISK_SEQUENTIAL


def select_initial_stage_cache_mode(
    offload: str,
    *,
    fast_quantized_pair: bool,
    pinned_memory: bool = True,
) -> WanStageCacheMode:
    if offload == "model" and fast_quantized_pair:
        if pinned_memory:
            return WanStageCacheMode.GPU_ACTIVE_CPU_PINNED_STANDBY
        return WanStageCacheMode.GPU_ACTIVE_CPU_UNPINNED_STANDBY
    if offload == "none":
        return WanStageCacheMode.FULL
    return WanStageCacheMode.NONE


def available_system_ram_gb() -> float | None:
    try:
        import psutil

        return float(psutil.virtual_memory().available) / 1024**3
    except Exception:
        return None


def estimated_stage_file_gb(path: str | Path | None) -> float:
    if not path:
        return 0.0
    try:
        return max(0.0, float(Path(path).stat().st_size) / 1024**3)
    except OSError:
        return 0.0


def _safety_margin_gb(default: float = 6.0) -> float:
    raw = os.environ.get("AIWF_WAN_CPU_CACHE_SAFETY_GB", "").strip()
    if not raw:
        return float(default)
    try:
        return max(1.0, float(raw))
    except ValueError:
        return float(default)


def resolve_stage_cache_after_pin_probe(
    mode: str | WanStageCacheMode,
    *,
    high_path: str | Path | None,
    low_path: str | Path | None,
    pin_available: bool,
    available_ram_gb: float | None = None,
    safety_margin_gb: float | None = None,
) -> WanStageCacheDecision:
    parsed = _parse_mode(mode)
    safety = _safety_margin_gb() if safety_margin_gb is None else max(1.0, float(safety_margin_gb))
    if parsed not in {
        WanStageCacheMode.GPU_ACTIVE_CPU_PINNED_STANDBY,
        WanStageCacheMode.GPU_ACTIVE_CPU_UNPINNED_STANDBY,
        WanStageCacheMode.DISK_SEQUENTIAL,
    }:
        return WanStageCacheDecision(mode=parsed, reason="cache mode does not use high/low CPU standby")

    low_estimate = estimated_stage_file_gb(low_path)
    available = available_system_ram_gb() if available_ram_gb is None else available_ram_gb
    additional_required = low_estimate + safety

    if pin_available and parsed == WanStageCacheMode.GPU_ACTIVE_CPU_PINNED_STANDBY:
        return WanStageCacheDecision(
            mode=parsed,
            reason="pinned CPU standby is available",
            available_ram_gb=available,
            additional_required_gb=additional_required,
            low_stage_estimate_gb=low_estimate,
            safety_margin_gb=safety,
        )

    if available is not None and available < additional_required:
        return WanStageCacheDecision(
            mode=WanStageCacheMode.DISK_SEQUENTIAL,
            reason=(
                "available system RAM is below the estimated low-stage CPU standby "
                "requirement plus safety margin"
            ),
            available_ram_gb=available,
            additional_required_gb=additional_required,
            low_stage_estimate_gb=low_estimate,
            safety_margin_gb=safety,
        )

    return WanStageCacheDecision(
        mode=WanStageCacheMode.GPU_ACTIVE_CPU_UNPINNED_STANDBY,
        reason="pinning is unavailable, but RAM is sufficient for unpinned CPU standby",
        available_ram_gb=available,
        additional_required_gb=additional_required,
        low_stage_estimate_gb=low_estimate,
        safety_margin_gb=safety,
    )
