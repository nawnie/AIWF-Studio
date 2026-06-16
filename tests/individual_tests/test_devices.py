from __future__ import annotations

from unittest.mock import patch

from aiwf.core.config.settings import RuntimeFlags
from aiwf.infrastructure.torch.devices import DeviceManager


def test_device_manager_honors_force_cpu_flag():
    flags = RuntimeFlags(cpu=True)
    manager = DeviceManager(flags)
    assert manager.device().type == "cpu"
    assert "forced" in manager.describe().lower()


@patch("aiwf.infrastructure.torch.devices.torch.cuda.is_available", return_value=True)
def test_device_manager_uses_cuda_when_available(_mock_cuda):
    manager = DeviceManager(RuntimeFlags(cpu=False))
    assert manager.device().type == "cuda"