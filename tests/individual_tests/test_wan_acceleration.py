from __future__ import annotations

from aiwf.infrastructure.torch import wan_perf


def test_wan_acceleration_capabilities_report_expected_keys(monkeypatch):
    monkeypatch.setattr(wan_perf, "_sage_dispatch_available", lambda: True)
    monkeypatch.setattr(wan_perf, "_flash_attn_dispatch_available", lambda: False)
    monkeypatch.setattr(wan_perf, "_module_importable", lambda name: name in {"sageattention", "gguf"})
    monkeypatch.delenv("AIWF_WAN_GGUF_RUNTIME", raising=False)
    monkeypatch.delenv("DIFFUSERS_GGUF_CUDA_KERNELS", raising=False)

    caps = wan_perf.describe_wan_acceleration_capabilities()

    assert caps["diffusers_sage"]["available"] is True
    assert caps["diffusers_flash"]["available"] is False
    assert caps["sageattention_fallback"]["available"] is True
    assert caps["gguf_runtime"]["available"] is True
    assert caps["gguf_cuda_kernels"]["available"] is False
    assert caps["torchao"]["available"] is False


def test_wan_acceleration_capabilities_respect_gguf_env(monkeypatch):
    monkeypatch.setattr(wan_perf, "_sage_dispatch_available", lambda: False)
    monkeypatch.setattr(wan_perf, "_flash_attn_dispatch_available", lambda: False)
    monkeypatch.setattr(wan_perf, "_module_importable", lambda name: name == "gguf")
    monkeypatch.setenv("AIWF_WAN_GGUF_RUNTIME", "0")

    caps = wan_perf.describe_wan_acceleration_capabilities()

    assert caps["gguf_runtime"]["available"] is False
