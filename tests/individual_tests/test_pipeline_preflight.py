from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from aiwf.core.config.settings import RuntimeFlags
from aiwf.core.domain.ltx import LTX_FULL_CHECKPOINT, LTX_GEMMA_REPO, LTX_PIPELINE_ONE_STAGE
from aiwf.services.pipeline_preflight import (
    preflight_diffusers_pipeline,
    preflight_ltx_pipeline,
    preflight_onnx_pipeline,
    preflight_qwen_nunchaku_pipeline,
    preflight_sana_video_pipeline,
    preflight_wan_pipeline,
)
from aiwf.services.worker_tenant import python_exe_for_venv


def _onnx_dir(root: Path, *, tokenizer: bool = True) -> Path:
    model = root / "sdxl_onnx"
    for sub in ("text_encoder", "unet", "vae_decoder"):
        path = model / sub
        path.mkdir(parents=True)
        (path / "model.onnx").write_bytes(b"fake")
    if tokenizer:
        tok = model / "tokenizer"
        tok.mkdir()
        (tok / "tokenizer.json").write_text("{}", encoding="utf-8")
    return model


def test_onnx_preflight_passes_complete_folder_with_cuda_provider(tmp_path: Path):
    model = _onnx_dir(tmp_path)

    result = preflight_onnx_pipeline(
        model,
        provider_preference="cuda",
        available_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    assert result.ok
    assert "CUDAExecutionProvider is available" in result.markdown()


def test_onnx_preflight_blocks_missing_tokenizer(tmp_path: Path):
    model = _onnx_dir(tmp_path, tokenizer=False)

    result = preflight_onnx_pipeline(
        model,
        provider_preference="auto",
        available_providers=["CPUExecutionProvider"],
    )

    assert not result.ok
    assert "tokenizer" in result.markdown()
    assert "Expected local tokenizer assets" in result.markdown()


def test_onnx_preflight_blocks_provider_mismatch(tmp_path: Path):
    model = _onnx_dir(tmp_path)

    result = preflight_onnx_pipeline(
        model,
        provider_preference="cuda",
        available_providers=["CPUExecutionProvider"],
    )

    assert not result.ok
    assert "CUDAExecutionProvider is not available" in result.markdown()


def test_onnx_preflight_auto_selects_available_provider(tmp_path: Path):
    model = _onnx_dir(tmp_path)

    result = preflight_onnx_pipeline(
        model,
        provider_preference="auto",
        available_providers=["DmlExecutionProvider", "CPUExecutionProvider"],
    )

    assert result.ok
    assert "auto will use DmlExecutionProvider" in result.markdown()


def test_onnx_backend_exposes_preflight(tmp_path: Path):
    model = _onnx_dir(tmp_path)

    with patch(
        "aiwf.services.pipeline_preflight._load_available_onnx_providers",
        return_value=["CPUExecutionProvider"],
    ):
        from aiwf.infrastructure.onnx.backend import ONNXBackend

        backend = ONNXBackend(tmp_path, provider="cpu")
        result = backend.preflight_checkpoint(model.name)

    assert result.ok
    assert result.metadata["model_dir"] == str(model.resolve())


def test_diffusers_preflight_reports_transformers_5_as_blocked(monkeypatch):
    fake_transformers = types.SimpleNamespace(__version__="5.0.0")
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "diffusers", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "safetensors", types.SimpleNamespace())

    result = preflight_diffusers_pipeline()

    assert not result.ok
    assert "unsupported" in result.markdown()


def test_qwen_nunchaku_preflight_checks_engine_and_assets(tmp_path: Path):
    engine = tmp_path / "engines" / "qwen_nunchaku"
    python = engine / ".venv" / "Scripts" / "python.exe"
    runner = engine / "run_qwen_lightning.py"
    base_dir = tmp_path / "models" / "qwen-image" / "Diffusers" / "Qwen-Image"
    transformer = tmp_path / "models" / "qwen-image" / "Nunchaku" / "svdq-int4_r32-qwen-image-lightningv1.0-4steps.safetensors"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_bytes(b"")
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner.write_text("print('ok')", encoding="utf-8")
    base_dir.mkdir(parents=True)
    (base_dir / "model_index.json").write_text("{}", encoding="utf-8")
    transformer.parent.mkdir(parents=True, exist_ok=True)
    transformer.write_bytes(b"")

    result = preflight_qwen_nunchaku_pipeline(tmp_path)

    assert result.ok
    assert result.metadata["transformer_path"].endswith(transformer.name)
    assert result.metadata["storage_mode"] == "single_transformer_safetensors_plus_base_components"


def test_qwen_nunchaku_preflight_blocks_missing_runtime(tmp_path: Path):
    result = preflight_qwen_nunchaku_pipeline(tmp_path)

    assert not result.ok
    assert "engine runtime missing" in result.markdown().lower() or "missing engine runtime" in result.markdown().lower()


def _write_fake_safetensors(path: Path) -> None:
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    path.parent.mkdir(parents=True, exist_ok=True)
    safetensors.save_file({"blocks.0.weight": torch.ones(1)}, path)


def _write_wan_fast_5b_assets(root: Path) -> RuntimeFlags:
    flags = RuntimeFlags(data_dir=root, models_dir=root / "models", output_dir=root / "outputs")
    component_base = flags.resolved_models_dir() / "wan" / "Diffusers" / "Wan2.2-TI2V-5B-Diffusers"
    (component_base / "text_encoder").mkdir(parents=True)
    (component_base / "tokenizer").mkdir()
    (component_base / "scheduler").mkdir()
    (component_base / "model_index.json").write_text("{}", encoding="utf-8")
    (component_base / "text_encoder" / "config.json").write_text("{}", encoding="utf-8")
    (component_base / "text_encoder" / "model.safetensors").write_bytes(b"fake")
    (component_base / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    (component_base / "scheduler" / "scheduler_config.json").write_text("{}", encoding="utf-8")
    _write_fake_safetensors(
        flags.resolved_models_dir() / "wan" / "Safetensor" / "wan2.2_ti2v_5B_fp16.safetensors"
    )
    _write_fake_safetensors(flags.resolved_models_dir() / "VAE" / "wan2.2_vae.safetensors")
    return flags


def _write_ready_ltx_worker(root: Path) -> None:
    worker = root / "engines" / "ltx" / "worker.py"
    repo = root / "engines" / "ltx" / "LTX-2"
    python = python_exe_for_venv(root / "engines" / "ltx" / ".venv")
    worker.parent.mkdir(parents=True)
    repo.mkdir(parents=True)
    python.parent.mkdir(parents=True)
    worker.write_text("print('worker')", encoding="utf-8")
    python.write_text("", encoding="utf-8")
    (root / "engines.json").write_text(json.dumps({"ltx": {"enabled": True}}), encoding="utf-8")


def test_wan_fast_5b_preflight_reports_ready_local_assets(tmp_path: Path, monkeypatch):
    flags = _write_wan_fast_5b_assets(tmp_path)
    monkeypatch.setattr("aiwf.services.wan.WanService.available", lambda self: True)

    result = preflight_wan_pipeline(flags)

    assert result.ok
    assert result.metadata["runtime_mode"] == "fast_5b"
    assert result.metadata["sampler"] == "unipc"
    assert result.metadata["offload"] == "balanced"


def test_ltx_preflight_uses_installed_one_stage_when_distilled_missing(tmp_path: Path):
    _write_ready_ltx_worker(tmp_path)
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs")
    checkpoint = flags.resolved_models_dir() / "ltx" / "checkpoints" / LTX_FULL_CHECKPOINT
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake")
    gemma = flags.resolved_models_dir() / "ltx" / "text_encoder" / LTX_GEMMA_REPO.split("/", 1)[1]
    gemma.mkdir(parents=True)

    result = preflight_ltx_pipeline(flags)

    assert result.ok
    assert result.metadata["selected_pipeline"] == LTX_PIPELINE_ONE_STAGE
    assert any("falls back" in warning for warning in result.warnings)


def test_ltx_preflight_blocks_missing_launch_checkpoint(tmp_path: Path):
    _write_ready_ltx_worker(tmp_path)
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs")
    gemma = flags.resolved_models_dir() / "ltx" / "text_encoder" / LTX_GEMMA_REPO.split("/", 1)[1]
    gemma.mkdir(parents=True)

    result = preflight_ltx_pipeline(flags)

    assert not result.ok
    assert "missing LTX checkpoint" in result.markdown()


def test_sana_video_preflight_reports_runtime_and_default_model_path(tmp_path: Path):
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs")

    result = preflight_sana_video_pipeline(flags)

    assert result.ok
    assert result.metadata["default_repo"] == "Efficient-Large-Model/SANA-Video_2B_480p_diffusers"
    assert result.metadata["model_path"].endswith("SANA-Video_2B_480p_diffusers")
    assert any("silent MP4" in warning for warning in result.warnings)
