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
from aiwf.services.model_download_catalog import MODEL_DOWNLOAD_CATALOG, QUICK_START_BUNDLES
from aiwf.services.model_setup_manifest import MODEL_SETUP_ROUTES
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
    (base / "model_index.json").write_text('{"_class_name": "WanPipeline"}', encoding="utf-8")
    (base / "text_encoder" / "config.json").write_text('{"hidden_size": 8}', encoding="utf-8")
    (base / "text_encoder" / "model.safetensors").write_bytes(b"fake")
    (base / "tokenizer" / "tokenizer.json").write_text('{"version": "1.0"}', encoding="utf-8")
    (base / "scheduler" / "scheduler_config.json").write_text('{"_class_name": "FlowMatchEulerDiscreteScheduler"}', encoding="utf-8")
    return base


def _write_fake_safetensors(path: Path) -> None:
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    path.parent.mkdir(parents=True, exist_ok=True)
    safetensors.save_file({"blocks.0.weight": torch.ones(1)}, path)


def _write_fake_comfy_fp8_wan_transformer(path: Path, *, in_channels: int) -> None:
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch float8 unavailable")
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {
        "patch_embedding.weight": torch.zeros((1, in_channels, 1, 1, 1), dtype=torch.float32),
        "blocks.0.attn.q.weight": torch.ones((1, 1), dtype=torch.float32).to(torch.float8_e4m3fn),
        "blocks.0.attn.q.weight_scale": torch.ones((1,), dtype=torch.float32),
    }
    for index in range(40):
        tensors[f"blocks.{index}.dummy"] = torch.ones((1,), dtype=torch.float32)
    safetensors.save_file(tensors, path)


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
    vae = service.flags.resolved_models_dir() / "VAE" / "wan2.2_vae.safetensors"
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


def test_wan_component_base_rejects_incomplete_indexed_text_encoder(tmp_path: Path):
    service = _svc(tmp_path)
    base = _write_component_base(service)
    text_encoder = base / "text_encoder"
    (text_encoder / "model.safetensors").unlink()
    (text_encoder / "model.safetensors.index.json").write_text(
        '{"weight_map":{"tensor":"missing-shard.safetensors"}}', encoding="utf-8"
    )

    assert service._is_components_base(base) is False
    assert any("model.safetensors" in item for item in service._component_base_missing(base))


@pytest.mark.parametrize(
    ("relative_path", "contents"),
    [
        ("model_index.json", ""),
        ("model_index.json", "not json"),
        ("model_index.json", "{}"),
        ("text_encoder/config.json", ""),
        ("text_encoder/config.json", "not json"),
        ("text_encoder/config.json", "{}"),
        ("tokenizer/tokenizer.json", ""),
        ("tokenizer/tokenizer.json", "not json"),
        ("tokenizer/tokenizer.json", "{}"),
        ("scheduler/scheduler_config.json", ""),
        ("scheduler/scheduler_config.json", "not json"),
        ("scheduler/scheduler_config.json", "{}"),
    ],
)
def test_wan_component_base_rejects_invalid_metadata(tmp_path: Path, relative_path: str, contents: str):
    service = _svc(tmp_path)
    base = _write_component_base(service)
    invalid_metadata = base / relative_path
    invalid_metadata.write_text(contents, encoding="utf-8")

    assert service._is_components_base(base) is False
    assert str(invalid_metadata) in service._component_base_missing(base)


def test_pass5_fast_5b_preflight_rejects_wan21_vae(tmp_path: Path, monkeypatch):
    service = _svc(tmp_path)
    monkeypatch.setattr(service, "_wan_file_candidates", lambda: [])
    _write_component_base(service)
    transformer = service.models_dir() / "Safetensor" / "wan2.2_ti2v_5B_fp16.safetensors"
    vae = service.flags.resolved_models_dir() / "VAE" / "wan2.1_vae.safetensors"
    _write_fake_safetensors(transformer)
    _write_fake_safetensors(vae)

    result = service.preflight(WanI2VRequest(runtime_mode=WAN_RUNTIME_FAST_5B))

    assert result.ok is False
    assert any("5B TI2V runtime expects the Wan 2.2" in error for error in result.errors)


def test_pass5_fast_5b_preflight_rejects_clear_14b_transformer(tmp_path: Path, monkeypatch):
    service = _svc(tmp_path)
    monkeypatch.setattr(service, "_wan_file_candidates", lambda: [])
    _write_component_base(service)
    transformer = service.models_dir() / "Safetensor" / "wan2.2_i2v_14B_high_noise_fp8.safetensors"
    vae = service.flags.resolved_models_dir() / "VAE" / "wan2.2_vae.safetensors"
    _write_fake_safetensors(transformer)
    _write_fake_safetensors(vae)

    result = service.preflight(
        WanI2VRequest(runtime_mode=WAN_RUNTIME_FAST_5B, model_id=transformer.name)
    )

    assert result.ok is False
    assert any("looks like 14B" in error and "expects Wan TI2V 5B" in error for error in result.errors)


def test_pass5_preflight_rejects_flux_t5xxl_text_encoder(tmp_path: Path, monkeypatch):
    service = _svc(tmp_path)
    monkeypatch.setattr(service, "_wan_file_candidates", lambda: [])
    _write_component_base(service)
    transformer = service.models_dir() / "Safetensor" / "wan2.2_ti2v_5B_fp16.safetensors"
    vae = service.flags.resolved_models_dir() / "VAE" / "wan2.2_vae.safetensors"
    text_encoder = service.flags.resolved_models_dir() / "Textencoder" / "flux_t5xxl_fp16.safetensors"
    _write_fake_safetensors(transformer)
    _write_fake_safetensors(vae)
    _write_fake_safetensors(text_encoder)

    result = service.preflight(
        WanI2VRequest(runtime_mode=WAN_RUNTIME_FAST_5B, text_encoder_path=text_encoder.name)
    )

    assert result.ok is False
    assert any("Flux/SD3 T5-XXL" in error for error in result.errors)


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


def test_pass5_fp8_preflight_rejects_fun_control_channel_count(tmp_path: Path, monkeypatch):
    import aiwf.services.wan as wan_service_module

    service = _svc(tmp_path)
    monkeypatch.setattr(service, "_wan_file_candidates", lambda: [])
    monkeypatch.setattr(wan_service_module, "_native_fp8_runtime_available", lambda: True)
    _write_component_base(service)
    vae = service.flags.resolved_models_dir() / "VAE" / "wan2.1_vae.safetensors"
    high = service.models_dir() / "Safetensor" / "wan2.2_fun_control_high_noise_14B_fp8_scaled.safetensors"
    low = service.models_dir() / "Safetensor" / "wan2.2_fun_control_low_noise_14B_fp8_scaled.safetensors"
    _write_fake_safetensors(vae)
    _write_fake_comfy_fp8_wan_transformer(high, in_channels=52)
    _write_fake_comfy_fp8_wan_transformer(low, in_channels=52)

    result = service.preflight(
        WanI2VRequest(
            runtime_mode=WAN_RUNTIME_HIGH_LOW_FP8,
            high_noise_model_id=high.name,
            low_noise_model_id=low.name,
            offload="streamed",
        )
    )

    assert result.ok is False
    assert any("52-channel patch embedding" in error for error in result.errors)
    assert any("36-channel Wan A14B I2V" in error for error in result.errors)


def test_wan_a14b_setup_bundle_components_pass_fp8_route_preflight(tmp_path: Path, monkeypatch):
    import json
    import aiwf.services.wan as wan_service_module

    service = _svc(tmp_path)
    monkeypatch.setattr(service, "_wan_file_candidates", lambda: [])
    monkeypatch.setattr(wan_service_module, "_native_fp8_runtime_available", lambda: True)

    route = next(route for route in MODEL_SETUP_ROUTES if route.route_key == "pro.video.wan.fp8-pair")
    bundle = QUICK_START_BUNDLES[route.bundle_key]
    entries = {entry.key: entry for entry in MODEL_DOWNLOAD_CATALOG if entry.key in bundle}
    components_entry = entries["wan-ti2v-components"]
    assert components_entry.snapshot_allow_patterns == (
        "model_index.json",
        "scheduler/**",
        "tokenizer/**",
        "text_encoder/config.json",
        "text_encoder/model.safetensors.index.json",
        "text_encoder/model-*.safetensors",
    )
    assert entries["wan-vae-21"].filename.endswith("wan_2.1_vae.safetensors")

    # Materialize the filtered snapshot shape the bundle downloads. The
    # repository is sharded, and the 5B transformer and Wan 2.2 VAE are absent.
    base = _write_component_base(service)
    text_encoder = base / "text_encoder"
    (text_encoder / "model.safetensors").unlink()
    (text_encoder / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "model-00001-of-00003.safetensors"}}),
        encoding="utf-8",
    )
    for shard in range(1, 4):
        (text_encoder / f"model-0000{shard}-of-00003.safetensors").write_bytes(b"fixture shard")

    vae = service.flags.resolved_models_dir() / "VAE" / "wan2.1_vae.safetensors"
    high = service.models_dir() / "Safetensor" / "wan2.2_i2v_14B_high_noise_fp8_scaled.safetensors"
    low = service.models_dir() / "Safetensor" / "wan2.2_i2v_14B_low_noise_fp8_scaled.safetensors"
    _write_fake_safetensors(vae)
    _write_fake_comfy_fp8_wan_transformer(high, in_channels=36)
    _write_fake_comfy_fp8_wan_transformer(low, in_channels=36)

    result = service.preflight(
        WanI2VRequest(
            runtime_mode=WAN_RUNTIME_HIGH_LOW_FP8,
            high_noise_model_id=high.name,
            low_noise_model_id=low.name,
            offload="streamed",
        )
    )

    assert result.ok, result.message()
    assert result.components_base == str(base.resolve())
    assert result.vae == str(vae.resolve())


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
