from __future__ import annotations

import pytest


def test_gguf_linear_dequantizes_quant_payload_on_cuda(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    from aiwf.infrastructure.wan import gguf_runtime

    seen_devices: list[str] = []

    def fake_is_quantized(_tensor):
        return True

    def fake_dequantize(tensor, dtype=None, dequant_dtype=None):
        seen_devices.append(str(tensor.device))
        return torch.ones((2, 3), device=tensor.device, dtype=dtype or torch.float32)

    monkeypatch.setattr(gguf_runtime, "is_quantized", fake_is_quantized)
    monkeypatch.setattr(gguf_runtime, "dequantize_tensor", fake_dequantize)

    layer = gguf_runtime.GGUFLinear(3, 2, bias=False)
    layer.weight = torch.nn.Parameter(torch.ones((2, 3)), requires_grad=False)
    x = torch.ones((4, 3), device="cuda", dtype=torch.float16)

    y = layer(x)

    assert y.shape == (4, 2)
    assert y.device.type == "cuda"
    assert seen_devices and seen_devices[0].startswith("cuda")
