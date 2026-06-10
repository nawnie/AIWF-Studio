from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from aiwf.core.config.settings import RuntimeFlags

logger = logging.getLogger(__name__)


class DeviceManager:
    def __init__(self, flags: RuntimeFlags | None = None) -> None:
        self._force_cpu = bool(flags.cpu) if flags is not None else False

    def device(self) -> torch.device:
        if self._force_cpu:
            return torch.device("cpu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def dtype(self, no_half: bool = False) -> torch.dtype:
        if no_half:
            return torch.float32
        if self.device().type == "cuda":
            return torch.float16
        return torch.float32

    def describe(self) -> str:
        if self._force_cpu:
            return "CPU (forced by --cpu)"
        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            name = torch.cuda.get_device_name(index)
            props = torch.cuda.get_device_properties(index)
            vram_gb = props.total_memory / (1024**3)
            cuda_ver = torch.version.cuda or "unknown"
            return f"CUDA ({name}, {vram_gb:.1f} GB VRAM, torch cuda {cuda_ver})"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "Apple MPS"
        return "CPU (slow — install CUDA PyTorch for GPU acceleration)"

    def log_status(self) -> None:
        logger.info("Compute device: %s", self.describe())
        if self.device().type == "cpu" and torch.version.cuda is None:
            logger.warning(
                "PyTorch CPU-only build detected. Re-run webui.bat to install the CUDA build, "
                "or set TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124"
            )

    def empty_cache(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()