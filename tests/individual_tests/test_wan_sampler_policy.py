from __future__ import annotations

from pathlib import Path

import pytest

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.wan import WAN_RUNTIME_FAST_5B, WAN_RUNTIME_HIGH_LOW_FP8, WanI2VRequest
from aiwf.services.wan import WanService
from aiwf.infrastructure.wan.sampler_policy import (
    WAN_5B_NATIVE_FLOW_SHIFT,
    WAN_5B_NATIVE_SAMPLER,
    audit_wan_sampler_settings,
)


def test_5b_blocks_euler_with_high_flow_shift():
    request = WanI2VRequest(
        runtime_mode=WAN_RUNTIME_FAST_5B,
        sampler="euler",
        flow_shift=8.0,
    )
    audit = audit_wan_sampler_settings(request)
    assert audit.errors
    assert any("euler" in error.lower() for error in audit.errors)
    assert any("8" in error for error in audit.errors)


def test_5b_auto_corrects_unipc_flow_shift_to_native_value():
    request = WanI2VRequest(
        runtime_mode=WAN_RUNTIME_FAST_5B,
        sampler=WAN_5B_NATIVE_SAMPLER,
        flow_shift=8.0,
    )
    audit = audit_wan_sampler_settings(request)
    assert not audit.errors
    assert audit.corrections
    assert audit.request is not None
    assert audit.request.flow_shift == WAN_5B_NATIVE_FLOW_SHIFT


def test_5b_unipc_native_settings_pass_without_corrections():
    request = WanI2VRequest(
        runtime_mode=WAN_RUNTIME_FAST_5B,
        sampler=WAN_5B_NATIVE_SAMPLER,
        flow_shift=WAN_5B_NATIVE_FLOW_SHIFT,
    )
    audit = audit_wan_sampler_settings(request)
    assert not audit.errors
    assert not audit.corrections


def _write_5b_preflight_fixtures(service: WanService) -> tuple[Path, Path]:
    base = service.models_dir() / "Diffusers" / "Wan2.2-TI2V-5B-Diffusers"
    for sub in ("text_encoder", "tokenizer", "scheduler"):
        (base / sub).mkdir(parents=True)
    (base / "model_index.json").write_text("{}", encoding="utf-8")
    (base / "text_encoder" / "config.json").write_text("{}", encoding="utf-8")
    (base / "text_encoder" / "model.safetensors").write_bytes(b"fake")
    (base / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    (base / "scheduler" / "scheduler_config.json").write_text("{}", encoding="utf-8")

    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    transformer = service.models_dir() / "Safetensor" / "wan2.2_ti2v_5B_fp16.safetensors"
    transformer.parent.mkdir(parents=True)
    safetensors.save_file({"blocks.0.weight": torch.ones(1)}, transformer)
    vae = service.flags.resolved_models_dir() / "VAE" / "wan2.2_vae.safetensors"
    vae.parent.mkdir(parents=True)
    safetensors.save_file({"decoder.conv_in.weight": torch.ones(1)}, vae)
    return transformer, vae


def test_preflight_blocks_5b_euler_high_shift(tmp_path: Path):
    flags = RuntimeFlags(
        data_dir=tmp_path,
        models_dir=tmp_path / "models",
        output_dir=tmp_path / "out",
    )
    service = WanService(flags, UserSettings())
    service._backend.available = lambda: True
    transformer, vae = _write_5b_preflight_fixtures(service)

    result = service.preflight(
        WanI2VRequest(
            runtime_mode=WAN_RUNTIME_FAST_5B,
            sampler="euler",
            flow_shift=8.0,
            vae_id=str(vae),
            model_id=str(transformer),
        )
    )
    assert result.ok is False
    assert any("euler" in error.lower() for error in result.errors)


def test_preflight_auto_corrects_unipc_flow_shift(tmp_path: Path):
    flags = RuntimeFlags(
        data_dir=tmp_path,
        models_dir=tmp_path / "models",
        output_dir=tmp_path / "out",
    )
    service = WanService(flags, UserSettings())
    service._backend.available = lambda: True
    transformer, vae = _write_5b_preflight_fixtures(service)

    result = service.preflight(
        WanI2VRequest(
            runtime_mode=WAN_RUNTIME_FAST_5B,
            sampler="unipc",
            flow_shift=8.0,
            vae_id=str(vae),
            model_id=str(transformer),
        )
    )
    assert result.ok is True
    assert result.audited_request is not None
    assert result.audited_request.flow_shift == WAN_5B_NATIVE_FLOW_SHIFT


def test_14b_fp8_requires_streamed_offload():
    request = WanI2VRequest(
        runtime_mode=WAN_RUNTIME_HIGH_LOW_FP8,
        sampler="euler",
        flow_shift=5.0,
        offload="balanced",
        high_noise_model_id="high.safetensors",
        low_noise_model_id="low.safetensors",
    )
    audit = audit_wan_sampler_settings(request)
    assert any("streamed" in error.lower() for error in audit.errors)