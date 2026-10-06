"""Optional NVML probe used to measure real VRAM after each model load."""
from __future__ import annotations

from typing import Callable

from .registry import GIB


def nvml_used_gib(device_index: int = 0) -> Callable[[], float | None]:
    """Return a probe reporting used VRAM in GiB, or one that returns None without NVML."""
    try:
        import pynvml  # provided by the nvidia-ml-py package

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
    except Exception:  # NVML missing, no NVIDIA driver, or no GPU
        return lambda: None

    def probe() -> float | None:
        try:
            return pynvml.nvmlDeviceGetMemoryInfo(handle).used / GIB
        except Exception:
            return None

    return probe
