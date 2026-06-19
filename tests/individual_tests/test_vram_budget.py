from __future__ import annotations

import pytest

from aiwf.infrastructure.torch.vram_budget import apply_cuda_vram_reserve


class _Props:
    total_memory = 16 * 1024 * 1024 * 1024


class _Cuda:
    def __init__(self, *, available: bool = True):
        self.available = available
        self.calls: list[tuple[float, int | str]] = []

    def is_available(self) -> bool:
        return self.available

    def get_device_properties(self, device):
        return _Props()

    def set_per_process_memory_fraction(self, fraction, device=0):
        self.calls.append((fraction, device))


class _Torch:
    def __init__(self, *, available: bool = True):
        self.cuda = _Cuda(available=available)


def test_apply_cuda_vram_reserve_caps_allocator_fraction(monkeypatch):
    monkeypatch.delenv("AIWF_CUDA_VRAM_RESERVE_MB", raising=False)
    monkeypatch.delenv("AIWF_WAN_VRAM_RESERVE_MB", raising=False)
    torch = _Torch()

    result = apply_cuda_vram_reserve(
        enabled=True,
        reserve_mb=1536,
        device=0,
        torch_module=torch,
    )

    assert result.enabled is True
    assert result.applied is True
    assert result.requested_reserve_mb == 1536
    assert result.total_mb == 16384
    assert result.limit_mb == 14848
    assert result.fraction == pytest.approx(0.90625)
    assert len(torch.cuda.calls) == 1
    assert torch.cuda.calls[0][0] == pytest.approx(0.90625)
    assert torch.cuda.calls[0][1] == 0


def test_apply_cuda_vram_reserve_disabled_resets_even_with_slider_value(monkeypatch):
    monkeypatch.delenv("AIWF_CUDA_VRAM_RESERVE_MB", raising=False)
    monkeypatch.delenv("AIWF_WAN_VRAM_RESERVE_MB", raising=False)
    torch = _Torch()

    result = apply_cuda_vram_reserve(
        enabled=False,
        reserve_mb=1536,
        device=0,
        torch_module=torch,
    )

    assert result.enabled is False
    assert result.applied is True
    assert result.requested_reserve_mb == 1536
    assert result.limit_mb == 16384
    assert result.fraction == 1.0
    assert torch.cuda.calls == [(1.0, 0)]


def test_apply_cuda_vram_reserve_env_override_can_force_cap(monkeypatch):
    monkeypatch.setenv("AIWF_CUDA_VRAM_RESERVE_MB", "2048")
    monkeypatch.delenv("AIWF_WAN_VRAM_RESERVE_MB", raising=False)
    torch = _Torch()

    result = apply_cuda_vram_reserve(
        enabled=False,
        reserve_mb=1536,
        device=0,
        torch_module=torch,
    )

    assert result.enabled is True
    assert result.applied is True
    assert result.requested_reserve_mb == 2048
    assert result.limit_mb == 14336
    assert result.fraction == pytest.approx(0.875)
    assert len(torch.cuda.calls) == 1
    assert torch.cuda.calls[0][0] == pytest.approx(0.875)
    assert torch.cuda.calls[0][1] == 0


def test_apply_cuda_vram_reserve_reports_unavailable_cuda(monkeypatch):
    monkeypatch.delenv("AIWF_CUDA_VRAM_RESERVE_MB", raising=False)
    monkeypatch.delenv("AIWF_WAN_VRAM_RESERVE_MB", raising=False)
    torch = _Torch(available=False)

    result = apply_cuda_vram_reserve(
        enabled=True,
        reserve_mb=1024,
        torch_module=torch,
    )

    assert result.enabled is True
    assert result.applied is False
    assert "CUDA is not available" in result.message
    assert torch.cuda.calls == []
