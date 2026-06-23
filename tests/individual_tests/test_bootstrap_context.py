from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from aiwf.core.config.settings import RuntimeFlags


class _FakeDevices:
    def __init__(self, flags):
        self.flags = flags

    def log_status(self):
        return None

    def describe(self):
        return "CPU"

    def device(self):
        return "cpu"

    def empty_cache(self):
        return None


class _FakeBackend:
    def __init__(self, flags, devices):
        self.flags = flags
        self.devices = devices

    def list_checkpoints(self):
        return []

    def list_loras(self):
        return []

    def list_vaes(self):
        return []

    def list_embeddings(self):
        return []

    def list_samplers(self):
        return []


def test_build_context_does_not_download_optional_segment_models_at_boot(tmp_path: Path):
    ensure_default_models = MagicMock()

    with patch("aiwf.bootstrap._create_device_manager", lambda flags: _FakeDevices(flags)), patch(
        "aiwf.bootstrap._create_diffusers_backend", lambda flags, devices: _FakeBackend(flags, devices)
    ), patch(
        "aiwf.bootstrap._create_segment_service",
        lambda flags, settings, devices, *, supervisor: SimpleNamespace(ensure_default_models=ensure_default_models),
    ), patch(
        "aiwf.dev.diagnostics.install_dev_diagnostics", lambda ctx: None
    ):
        from aiwf.bootstrap import build_context

        build_context(RuntimeFlags(data_dir=tmp_path))

    ensure_default_models.assert_not_called()
