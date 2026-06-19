from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.wan import WAN_RUNTIME_HIGH_LOW, WanI2VRequest
from aiwf.infrastructure.wan.comfy_quant_format import inspect_wan_quant_file
from aiwf.services.wan import WanService


def _write_component_base(service: WanService) -> Path:
    base = service.models_dir() / "Diffusers" / "Wan2.2-TI2V-5B-Diffusers"
    (base / "text_encoder").mkdir(parents=True)
    (base / "tokenizer").mkdir()
    (base / "scheduler").mkdir()
    (base / "model_index.json").write_text("{}", encoding="utf-8")
    (base / "text_encoder" / "config.json").write_text("{}", encoding="utf-8")
    (base / "text_encoder" / "model.safetensors").write_bytes(b"fake")
    (base / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    (base / "scheduler" / "scheduler_config.json").write_text("{}", encoding="utf-8")
    return base


def _svc(tmp_path: Path) -> WanService:
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "out")
    service = WanService(flags, UserSettings())
    service._backend.available = lambda: True
    return service


def test_inspect_wan_quant_file_detects_comfy_fp8(tmp_path: Path):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch float8 unavailable")

    path = tmp_path / "wan-high.safetensors"
    safetensors.save_file(
        {
            "blocks.0.attn1.to_q.weight": torch.randn(32, 16).to(torch.float8_e4m3fn),
            "blocks.0.attn1.to_q.weight_scale": torch.tensor(0.125, dtype=torch.float32),
            "blocks.0.attn1.to_q.input_scale": torch.tensor(1.0, dtype=torch.float32),
        },
        path,
    )

    report = inspect_wan_quant_file(path)

    assert report.format == "comfy_fp8"
    assert report.is_comfy_fp8 is True
    assert report.demo_ready is True
    assert report.quantized_linear_layers == 1
    assert report.weight_scale_count == 1
    assert report.input_scale_count == 1
    assert not report.missing_scale_keys


def test_inspect_wan_quant_file_reports_missing_scales(tmp_path: Path):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch float8 unavailable")

    path = tmp_path / "wan-low.safetensors"
    safetensors.save_file(
        {"blocks.0.attn1.to_q.weight": torch.randn(32, 16).to(torch.float8_e4m3fn)},
        path,
    )

    report = inspect_wan_quant_file(path)

    assert report.format == "comfy_fp8"
    assert report.demo_ready is False
    assert report.missing_scale_keys == ("blocks.0.attn1.to_q.weight_scale",)


def test_inspect_wan_quant_file_reports_unsupported_metadata(tmp_path: Path):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")

    path = tmp_path / "wan-high.safetensors"
    safetensors.save_file(
        {"blocks.0.attn1.to_q.weight": torch.ones(1, dtype=torch.float32)},
        path,
        metadata={
            "_quantization_metadata": json.dumps({"blocks.0": {"format": "nf4"}}),
        },
    )

    report = inspect_wan_quant_file(path)

    assert "nf4" in report.unsupported_quant_formats
    assert report.demo_ready is False


def test_wan_preflight_surfaces_missing_fp8_scale(tmp_path: Path):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch float8 unavailable")

    service = _svc(tmp_path)
    _write_component_base(service)
    high = service.models_dir() / "Safetensor" / "wan-high.safetensors"
    low = service.models_dir() / "Safetensor" / "wan-low.safetensors"
    vae = service.flags.resolved_models_dir() / "VAE" / "wan2.1_vae.safetensors"
    high.parent.mkdir(parents=True)
    vae.parent.mkdir(parents=True)
    safetensors.save_file(
        {"blocks.0.attn1.to_q.weight": torch.randn(32, 16).to(torch.float8_e4m3fn)},
        high,
    )
    safetensors.save_file(
        {
            "blocks.0.attn1.to_q.weight": torch.randn(32, 16).to(torch.float8_e4m3fn),
            "blocks.0.attn1.to_q.weight_scale": torch.tensor(0.125, dtype=torch.float32),
        },
        low,
    )
    safetensors.save_file({"vae.weight": torch.ones(1)}, vae)

    result = service.preflight(
        WanI2VRequest(runtime_mode=WAN_RUNTIME_HIGH_LOW, high_noise_model_id=high.name, low_noise_model_id=low.name)
    )

    assert result.ok is False
    assert any("missing FP8 scale tensor" in error for error in result.errors)
