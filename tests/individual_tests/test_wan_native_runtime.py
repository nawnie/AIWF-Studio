from __future__ import annotations

from pathlib import Path

import pytest


def test_wan_ops_factories_create_expected_modules():
    torch = pytest.importorskip("torch")

    from aiwf.infrastructure.quant.fp8_linear import AIWFFP8Linear
    from aiwf.infrastructure.wan.native.ops import WanBF16Ops, WanDiagnosticOps, WanFP8Ops

    bf16 = WanBF16Ops()
    fp8 = WanFP8Ops()
    diagnostic = WanDiagnosticOps(base=bf16)

    assert isinstance(bf16.Linear(4, 8), torch.nn.Linear)
    assert isinstance(fp8.Linear(16, 32, bias=False), AIWFFP8Linear)
    assert isinstance(diagnostic.LayerNorm(4), torch.nn.LayerNorm)
    assert diagnostic.created[0][0] == "LayerNorm"


def test_prepare_wan_i2v_latents_matches_wan_geometry():
    torch = pytest.importorskip("torch")
    Image = pytest.importorskip("PIL.Image")

    from aiwf.infrastructure.wan.native.conditioning import prepare_wan_i2v_latents

    image = Image.new("RGB", (32, 32), "white")
    bundle = prepare_wan_i2v_latents(
        image,
        vae=None,
        width=64,
        height=32,
        frames=81,
        batch_size=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert tuple(bundle.latent.shape) == (2, 16, 21, 4, 8)
    assert tuple(bundle.concat_latent_image.shape) == (2, 16, 21, 4, 8)
    assert tuple(bundle.concat_mask.shape) == (2, 1, 21, 4, 8)
    assert torch.all(bundle.concat_mask[:, :, 0] == 0)
    assert torch.all(bundle.concat_mask[:, :, 1:] == 1)


def test_prepare_wan_i2v_latents_uses_vae_encode_output():
    torch = pytest.importorskip("torch")
    Image = pytest.importorskip("PIL.Image")

    from aiwf.infrastructure.wan.native.conditioning import prepare_wan_i2v_latents

    class _Encoded:
        sample = torch.ones(1, 16, 4, 4)

    class _VAE:
        def encode(self, image_tensor):
            assert image_tensor.shape == (1, 3, 32, 32)
            return _Encoded()

    bundle = prepare_wan_i2v_latents(
        Image.new("RGB", (16, 16), "black"),
        _VAE(),
        width=32,
        height=32,
        frames=9,
        dtype=torch.float32,
    )

    assert tuple(bundle.concat_latent_image.shape) == (1, 16, 3, 4, 4)
    assert torch.all(bundle.concat_latent_image[:, :, 0] == 1)
    assert torch.all(bundle.concat_latent_image[:, :, 1:] == 0)


def test_native_runner_reports_missing_high_low_paths(tmp_path: Path):
    from aiwf.infrastructure.wan.native.ops import WanBF16Ops
    from aiwf.infrastructure.wan.native.runner import AIWFWanRunner

    runner = AIWFWanRunner(ops=WanBF16Ops())
    readiness = runner.inspect_high_low(tmp_path / "missing-high.safetensors", tmp_path / "missing-low.safetensors")

    assert readiness.ok is False
    assert "High-noise transformer does not exist" in readiness.message()
    assert "Low-noise transformer does not exist" in readiness.message()


def test_native_runner_uses_quant_inspector_for_fp8_readiness(tmp_path: Path, monkeypatch):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch float8 unavailable")

    from aiwf.infrastructure.wan.native.ops import WanFP8Ops
    from aiwf.infrastructure.wan.native.runner import AIWFWanRunner

    high = tmp_path / "high.safetensors"
    low = tmp_path / "low.safetensors"
    payload = {
        "blocks.0.attn.q.weight": torch.randn(32, 16).to(torch.float8_e4m3fn),
        "blocks.0.attn.q.weight_scale": torch.tensor(0.125, dtype=torch.float32),
    }
    safetensors.save_file(payload, high)
    safetensors.save_file(payload, low)
    monkeypatch.setattr("aiwf.infrastructure.wan.native.runner.torch_native_fp8_available", lambda: True)

    readiness = AIWFWanRunner(ops=WanFP8Ops()).inspect_high_low(high, low, require_fp8=True)

    assert readiness.ok is True
    assert readiness.high_report is not None
    assert readiness.high_report.quantized_linear_layers == 1
    assert readiness.low_report is not None


def test_native_runner_prepare_latents_records_diagnostics():
    torch = pytest.importorskip("torch")

    from aiwf.infrastructure.wan.native.ops import WanBF16Ops
    from aiwf.infrastructure.wan.native.runner import AIWFWanRunner

    runner = AIWFWanRunner(ops=WanBF16Ops())
    bundle = runner.prepare_latents(
        image=None,
        vae=None,
        width=64,
        height=64,
        frames=17,
        dtype=torch.float32,
    )

    assert bundle.latent_shape == (1, 16, 5, 8, 8)
    assert runner.diagnostics["latent_shape"] == (1, 16, 5, 8, 8)
    assert runner.diagnostics["has_concat_latent_image"] is False
    assert runner.diagnostics["has_concat_mask"] is False


def test_stage_cache_policy_uses_unpinned_standby_when_pin_fails_but_ram_is_available(tmp_path: Path):
    from aiwf.infrastructure.wan.native.memory import (
        WanStageCacheMode,
        resolve_stage_cache_after_pin_probe,
    )

    low = tmp_path / "low.safetensors"
    low.write_bytes(b"x" * 1024)

    decision = resolve_stage_cache_after_pin_probe(
        WanStageCacheMode.GPU_ACTIVE_CPU_PINNED_STANDBY,
        high_path=tmp_path / "high.safetensors",
        low_path=low,
        pin_available=False,
        available_ram_gb=24.0,
        safety_margin_gb=4.0,
    )

    assert decision.mode == WanStageCacheMode.GPU_ACTIVE_CPU_UNPINNED_STANDBY
    assert decision.uses_disk_at_boundary is False
    assert "RAM is sufficient" in decision.reason


def test_stage_cache_policy_uses_disk_sequential_only_when_ram_is_tight(tmp_path: Path):
    from aiwf.infrastructure.wan.native.memory import (
        WanStageCacheMode,
        resolve_stage_cache_after_pin_probe,
    )

    low = tmp_path / "low.safetensors"
    low.write_bytes(b"x" * (2 * 1024 * 1024))

    decision = resolve_stage_cache_after_pin_probe(
        WanStageCacheMode.GPU_ACTIVE_CPU_PINNED_STANDBY,
        high_path=tmp_path / "high.safetensors",
        low_path=low,
        pin_available=False,
        available_ram_gb=1.0,
        safety_margin_gb=4.0,
    )

    assert decision.mode == WanStageCacheMode.DISK_SEQUENTIAL
    assert decision.uses_disk_at_boundary is True
    assert decision.additional_required_gb is not None
