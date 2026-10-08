from __future__ import annotations

from types import SimpleNamespace

from aiwf.services.gpu_memory import nvidia_smi_free_bytes


def test_nvidia_smi_free_bytes_maps_visible_numeric_device(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,0")
    monkeypatch.setattr(
        "aiwf.services.gpu_memory.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout="0, GPU-bbb, 4096\n2, GPU-aaa, 2048\n"
        ),
    )

    assert nvidia_smi_free_bytes(0) == 2048 * 1024**2


def test_nvidia_smi_free_bytes_maps_visible_uuid(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-aaa")
    monkeypatch.setattr(
        "aiwf.services.gpu_memory.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="0, GPU-bbb, 4096\n2, GPU-aaa, 2048\n"),
    )

    assert nvidia_smi_free_bytes() == 2048 * 1024**2


def test_nvidia_smi_free_bytes_fails_closed_on_unmatched_device(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-missing")
    monkeypatch.setattr(
        "aiwf.services.gpu_memory.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout="0, GPU-bbb, 4096\n"),
    )

    assert nvidia_smi_free_bytes() is None
