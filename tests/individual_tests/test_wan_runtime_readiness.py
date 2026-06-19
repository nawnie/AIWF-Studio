from __future__ import annotations

from pathlib import Path

import pytest

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.wan import (
    WAN_RUNTIME_FAST_5B,
    WAN_RUNTIME_HIGH_LOW,
    WAN_RUNTIME_HIGH_LOW_FP8,
    WanI2VRequest,
)
from aiwf.infrastructure.quant.fp8_linear import collect_fp8_linear_metrics
from aiwf.services.wan import WanService
from aiwf.web.app import register_default_tabs
from aiwf.web.registry import WebRegistry


def _svc(tmp_path: Path) -> WanService:
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "out")
    service = WanService(flags, UserSettings())
    service._backend.available = lambda: True
    return service


def _write_component_base(service: WanService, name: str = "Wan2.2-TI2V-5B-Diffusers") -> Path:
    base = service.models_dir() / "Diffusers" / name
    (base / "text_encoder").mkdir(parents=True)
    (base / "tokenizer").mkdir()
    (base / "scheduler").mkdir()
    (base / "model_index.json").write_text("{}", encoding="utf-8")
    (base / "text_encoder" / "config.json").write_text("{}", encoding="utf-8")
    (base / "text_encoder" / "model.safetensors").write_bytes(b"fake")
    (base / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    (base / "scheduler" / "scheduler_config.json").write_text("{}", encoding="utf-8")
    return base


def _write_fake_safetensors(path: Path) -> None:
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    path.parent.mkdir(parents=True, exist_ok=True)
    safetensors.save_file({"blocks.0.weight": torch.ones(1)}, path)


def test_pass5_runtime_mode_contracts_are_explicit():
    fast = WanI2VRequest(runtime_mode=WAN_RUNTIME_FAST_5B)
    quality = WanI2VRequest(runtime_mode=WAN_RUNTIME_HIGH_LOW)
    experimental = WanI2VRequest(runtime_mode=WAN_RUNTIME_HIGH_LOW_FP8)
    populated = WanI2VRequest(
        runtime_mode=WAN_RUNTIME_HIGH_LOW_FP8,
        high_noise_model_id="high.safetensors",
        low_noise_model_id="low.safetensors",
    )

    assert fast.requires_dual_transformers() is False
    assert fast.uses_dual_transformers() is False
    assert quality.requires_dual_transformers() is True
    assert quality.uses_dual_transformers() is False
    assert experimental.requires_dual_transformers() is True
    assert populated.uses_dual_transformers() is True

    with pytest.raises(ValueError, match="runtime_mode"):
        WanI2VRequest(runtime_mode="comfy_backend")


def test_pass5_fast_5b_preflight_is_local_only_and_does_not_need_high_low(tmp_path: Path, monkeypatch):
    service = _svc(tmp_path)
    monkeypatch.setattr(service, "_wan_file_candidates", lambda: [])
    missing = service.preflight(WanI2VRequest(runtime_mode=WAN_RUNTIME_FAST_5B))

    assert missing.ok is False
    assert any("Fast 5B mode needs a local Wan TI2V 5B transformer file" in error for error in missing.errors)

    base = _write_component_base(service)
    vae = service.flags.resolved_models_dir() / "VAE" / "wan2.1_vae.safetensors"
    _write_fake_safetensors(vae)
    base_only = service.preflight(WanI2VRequest(runtime_mode=WAN_RUNTIME_FAST_5B))

    assert base_only.ok is False
    assert any("found only a shared component base" in error for error in base_only.errors)

    transformer = service.models_dir() / "Safetensor" / "wan2.2_ti2v_5B_fp16.safetensors"
    _write_fake_safetensors(transformer)
    ready = service.preflight(WanI2VRequest(runtime_mode=WAN_RUNTIME_FAST_5B))

    assert ready.ok is True
    assert ready.model_id == str(transformer.resolve())
    assert ready.high_noise_model is None
    assert ready.low_noise_model is None
    assert ready.components_base == str(base.resolve())


def test_pass5_high_low_modes_still_require_both_transformers(tmp_path: Path):
    service = _svc(tmp_path)
    _write_component_base(service)
    vae = service.flags.resolved_models_dir() / "VAE" / "wan2.1_vae.safetensors"
    high = service.models_dir() / "Safetensor" / "wan-high.safetensors"
    _write_fake_safetensors(vae)
    _write_fake_safetensors(high)

    for mode in (WAN_RUNTIME_HIGH_LOW, WAN_RUNTIME_HIGH_LOW_FP8):
        result = service.preflight(WanI2VRequest(runtime_mode=mode, high_noise_model_id=high.name))
        assert result.ok is False
        assert any("Select a Low noise transformer" in error for error in result.errors)


def test_pass5_fp8_metric_aggregation_deduplicates_shared_modules(monkeypatch):
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch float8 unavailable")

    monkeypatch.setenv("AIWF_WAN_ALLOW_FP8_FALLBACK", "1")
    from aiwf.infrastructure.quant.fp8_linear import AIWFFP8Linear

    layer = AIWFFP8Linear(16, 32)
    layer.weight = torch.nn.Parameter(
        torch.randn(32, 16).clamp(-2, 2).to(dtype=torch.float8_e4m3fn),
        requires_grad=False,
    )
    layer.weight_scale = torch.ones((), dtype=torch.float32)
    root_a = torch.nn.Sequential(layer)
    root_b = torch.nn.Module()
    root_b.layer = layer

    layer(torch.randn(2, 16, dtype=torch.bfloat16))
    metrics = collect_fp8_linear_metrics(root_a, root_b)

    assert metrics["fp8_linear_layers"] == 1
    assert metrics["fp8_fallback_calls"] == 1
    assert metrics["fp8_fallback_layers"] == 1
    assert metrics["fp8_fallback_reasons"]


def test_pass5_video_tab_is_registered_with_default_ui_tabs():
    registry = WebRegistry()
    register_default_tabs(registry)

    names = [name for name, _builder, _order in registry.tabs]

    assert "Video" in names
    assert len(names) == len(set(names))
