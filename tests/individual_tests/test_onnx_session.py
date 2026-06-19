"""tests/test_onnx_session.py — ONNX session module tests (no GPU, no ORT required)."""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import numpy as np


class TestONNXSessionImport:
    def test_module_importable(self) -> None:
        from aiwf.infrastructure.onnx import session  # noqa: F401

    def test_load_session_raises_without_ort(self) -> None:
        """load_session should raise ImportError if onnxruntime not installed."""
        with patch.dict(sys.modules, {"onnxruntime": None}):
            from aiwf.infrastructure.onnx.session import _ort
            with pytest.raises(ImportError, match="onnxruntime"):
                _ort()


class TestONNXBackend:
    def test_discovers_no_models_in_empty_dir(self, tmp_path: Path) -> None:
        from aiwf.infrastructure.onnx.backend import ONNXBackend
        backend = ONNXBackend(tmp_path)
        assert backend.list_checkpoints() == []

    def test_discovers_valid_model_dir(self, tmp_path: Path) -> None:
        # Create fake model structure
        for sub in ("text_encoder", "unet", "vae_decoder"):
            d = tmp_path / "my_sd_model" / sub
            d.mkdir(parents=True)
            (d / "model.onnx").write_bytes(b"fake")
        from aiwf.infrastructure.onnx.backend import ONNXBackend
        backend = ONNXBackend(tmp_path)
        checkpoints = backend.list_checkpoints()
        assert len(checkpoints) == 1
        assert checkpoints[0].id == "my_sd_model"
        assert checkpoints[0].kind == "onnx"

    def test_invalid_dir_not_listed(self, tmp_path: Path) -> None:
        # Missing vae_decoder
        for sub in ("text_encoder", "unet"):
            d = tmp_path / "bad_model" / sub
            d.mkdir(parents=True)
            (d / "model.onnx").write_bytes(b"fake")
        from aiwf.infrastructure.onnx.backend import ONNXBackend
        backend = ONNXBackend(tmp_path)
        assert backend.list_checkpoints() == []

    def test_unload_no_error_when_nothing_loaded(self, tmp_path: Path) -> None:
        from aiwf.infrastructure.onnx.backend import ONNXBackend
        backend = ONNXBackend(tmp_path)
        backend.unload()  # should not raise

    def test_list_samplers_nonempty(self, tmp_path: Path) -> None:
        from aiwf.infrastructure.onnx.backend import ONNXBackend
        backend = ONNXBackend(tmp_path)
        samplers = backend.list_samplers()
        assert len(samplers) > 0

    def test_catalog_methods_include_empty_embeddings(self, tmp_path: Path) -> None:
        from aiwf.infrastructure.onnx.backend import ONNXBackend
        backend = ONNXBackend(tmp_path)
        assert backend.list_loras() == []
        assert backend.list_vaes() == []
        assert backend.list_embeddings() == []


class TestONNXTokenizer:
    def test_tokenize_requires_local_tokenizer_dir(self, tmp_path: Path) -> None:
        from aiwf.infrastructure.onnx.pipeline import _tokenize

        with pytest.raises(FileNotFoundError, match="tokenizer"):
            _tokenize("hello", tmp_path / "missing-tokenizer")

    def test_tokenize_uses_local_files_only(self, tmp_path: Path) -> None:
        from aiwf.infrastructure.onnx.pipeline import _tokenize

        calls = {}
        tokenizer_dir = tmp_path / "tokenizer"
        tokenizer_dir.mkdir()

        class FakeTokenizer:
            @classmethod
            def from_pretrained(cls, path, local_files_only=False):
                calls["path"] = path
                calls["local_files_only"] = local_files_only
                return cls()

            def __call__(self, *args, **kwargs):
                return types.SimpleNamespace(input_ids=np.zeros((1, 77), dtype=np.int64))

        fake_transformers = types.SimpleNamespace(CLIPTokenizer=FakeTokenizer)

        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            tokens = _tokenize("hello", tokenizer_dir)

        assert calls == {"path": str(tokenizer_dir), "local_files_only": True}
        assert tokens.shape == (1, 77)


class TestONNXProviderSelection:
    def test_cpu_fallback_when_nothing_available(self) -> None:
        fake_ort = MagicMock()
        fake_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
        with patch.dict(sys.modules, {"onnxruntime": fake_ort}):
            from importlib import reload
            import aiwf.infrastructure.onnx.session as sess_mod
            reload(sess_mod)
            providers = sess_mod.select_provider("auto")
        assert providers == ["CPUExecutionProvider"]

    def test_auto_selects_cuda_when_available(self) -> None:
        fake_ort = MagicMock()
        fake_ort.get_available_providers.return_value = [
            "CUDAExecutionProvider", "CPUExecutionProvider"
        ]
        with patch.dict(sys.modules, {"onnxruntime": fake_ort}):
            from importlib import reload
            import aiwf.infrastructure.onnx.session as sess_mod
            reload(sess_mod)
            providers = sess_mod.select_provider("auto")
        names = [p if isinstance(p, str) else p[0] for p in providers]
        assert "CUDAExecutionProvider" in names


class TestONNXBootstrap:
    def test_build_context_uses_saved_onnx_model_dir(self, tmp_path: Path) -> None:
        from aiwf.core.config.settings import RuntimeFlags

        captured = {}

        class FakeDevices:
            def __init__(self, flags):
                self.flags = flags

            def log_status(self):
                return None

            def describe(self):
                return "CPU"

            def device(self):
                return "cpu"

        class FakeONNXBackend:
            def __init__(self, models_root, provider="auto", device_id=0):
                captured["models_root"] = Path(models_root)
                captured["provider"] = provider
                captured["device_id"] = device_id

            def list_checkpoints(self):
                return []

            def list_loras(self):
                return []

            def list_vaes(self):
                return []

            def list_samplers(self):
                return []

        (tmp_path / "config.json").write_text(
            '{"onnx_model_dir": "custom-onnx-root"}',
            encoding="utf-8",
        )

        with patch("aiwf.bootstrap.DeviceManager", FakeDevices), patch(
            "aiwf.bootstrap.ONNXBackend", FakeONNXBackend
        ), patch(
            "aiwf.dev.diagnostics.install_dev_diagnostics", lambda ctx: None
        ):
            from aiwf.bootstrap import build_context

            build_context(RuntimeFlags(data_dir=tmp_path, inference_backend="onnx"))

        assert captured["models_root"] == Path("custom-onnx-root")
        assert captured["provider"] == "auto"
