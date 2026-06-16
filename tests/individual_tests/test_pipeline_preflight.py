from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import patch

from aiwf.services.pipeline_preflight import (
    preflight_diffusers_pipeline,
    preflight_onnx_pipeline,
)


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
