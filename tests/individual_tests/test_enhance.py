from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.enhance import RestoreOptions, UpscaleOptions
from aiwf.infrastructure.enhance.catalog import EnhanceModelCatalog
from aiwf.infrastructure.enhance.tiles import combine_grid, split_grid
from aiwf.services.enhance import EnhanceService


@pytest.fixture
def catalog(tmp_path):
    flags = RuntimeFlags(data_dir=tmp_path)
    return EnhanceModelCatalog(flags)


def test_catalog_lists_builtin_models(catalog):
    upscalers = catalog.list_models()
    assert any(model.id == "realesrgan-x4plus" for model in upscalers)
    assert any(model.id == "gfpgan-v1.4" for model in upscalers)
    assert any(model.id == "codeformer-v0.1.0" for model in upscalers)


def test_tile_split_and_combine_roundtrip():
    image = Image.new("RGB", (300, 200), color=(120, 80, 40))
    grid = split_grid(image, tile_w=128, tile_h=128, overlap=16)
    combined = combine_grid(grid)
    assert combined.size == image.size


def test_run_pipeline_restore_then_upscale(tmp_path):
    flags = RuntimeFlags(data_dir=tmp_path)
    settings = UserSettings(save_images=False)
    devices = MagicMock()
    devices.device.return_value = "cpu"
    store = MagicMock()
    service = EnhanceService(flags, settings, devices, store)

    image = Image.new("RGB", (64, 64), color=(200, 100, 50))
    restored = Image.new("RGB", (64, 64), color=(210, 110, 60))
    upscaled = Image.new("RGB", (128, 128), color=(220, 120, 70))

    with patch.object(service, "restore", return_value=restored) as restore_mock, patch.object(
        service, "upscale", return_value=upscaled
    ) as upscale_mock:
        result, infotext = service.run_pipeline(
            image,
            restore=RestoreOptions(model_id="gfpgan-v1.4"),
            upscale=UpscaleOptions(model_id="realesrgan-x4plus", scale=2),
            restore_first=True,
        )

    restore_mock.assert_called_once()
    upscale_mock.assert_called_once()
    assert result.size == (128, 128)
    assert "Restore" in infotext
    assert "Upscale" in infotext