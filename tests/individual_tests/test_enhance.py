from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.enhance import RestoreOptions, UpscaleOptions
from aiwf.infrastructure.enhance.catalog import EnhanceModelCatalog
from aiwf.infrastructure.enhance.tiles import combine_grid, split_grid
from aiwf.infrastructure.storage.filesystem import FilesystemImageStore
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


def test_pro_model_setup_downloads_to_staging_file_and_reports_installed(tmp_path, monkeypatch):
    flags = RuntimeFlags(data_dir=tmp_path)
    service = EnhanceService(
        flags,
        UserSettings(save_images=False),
        MagicMock(),
        FilesystemImageStore(tmp_path / "outputs"),
    )

    restorer = next(model for model in service.list_restorers() if model.id == "gfpgan-v1.4")
    initial = next(model for model in service.list_model_status() if model["id"] == restorer.id)
    assert initial["installed"] is False
    assert initial["installAvailable"] is True

    def fake_urlretrieve(_url, destination):
        Path(destination).write_bytes(b"fixture weights")
        return str(destination), None

    monkeypatch.setattr("aiwf.infrastructure.enhance.catalog.urlretrieve", fake_urlretrieve)
    installed_path = service.prepare_model(restorer.id)

    assert installed_path.is_file()
    assert installed_path.read_bytes() == b"fixture weights"
    assert next(model for model in service.list_model_status() if model["id"] == restorer.id)["installed"] is True
    assert list(installed_path.parent.glob("*.download")) == []


def test_warmed_enhance_catalog_rediscovers_model_in_new_shared_root_without_downloading(tmp_path, monkeypatch):
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models")
    service = EnhanceService(
        flags,
        UserSettings(save_images=False),
        MagicMock(),
        FilesystemImageStore(tmp_path / "outputs"),
    )
    assert next(model for model in service.list_model_status() if model["id"] == "realesrgan-x4plus")["installed"] is False

    shared_root = tmp_path / "shared-models"
    shared_file = shared_root / "upscale_models" / "RealESRGAN_x4plus.pth"
    shared_file.parent.mkdir(parents=True)
    shared_file.write_bytes(b"shared fixture weights")
    flags.extra_model_dirs.append(shared_root)
    service.invalidate_model_catalog()
    monkeypatch.setattr(
        "aiwf.infrastructure.enhance.catalog.urlretrieve",
        lambda *_args: (_ for _ in ()).throw(AssertionError("must reuse the discovered shared model")),
    )

    model = next(item for item in service.list_model_status() if item["id"] == "realesrgan-x4plus")
    installed_path = service.prepare_model("realesrgan-x4plus")

    assert model["installed"] is True
    assert installed_path.resolve() == shared_file.resolve()




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


def test_save_result_writes_enhance_receipt(tmp_path):
    flags = RuntimeFlags(data_dir=tmp_path)
    settings = UserSettings(save_images=True, enhance_output_subdir="enhanced-images")
    devices = MagicMock()
    store = FilesystemImageStore(tmp_path / "outputs", settings=settings)
    service = EnhanceService(flags, settings, devices, store)
    source = Image.new("RGB", (16, 12), color=(10, 20, 30))
    output = Image.new("RGB", (32, 24), color=(40, 50, 60))
    options = UpscaleOptions(model_id="realesrgan-x4plus", scale=2, tile_size=128, tile_overlap=16)

    result = service.save_result(
        output,
        "Upscale: RealESRGAN (2x)",
        source_image=source,
        route="upscale",
        upscale=options,
    )

    assert result.image_path
    assert result.receipt_path
    assert Path(result.image_path).is_file()
    receipt = json.loads(Path(result.receipt_path).read_text(encoding="utf-8"))
    assert receipt["receipt_type"] == "enhance"
    assert receipt["route"] == "upscale"
    assert receipt["input"]["width"] == 16
    assert receipt["output"]["height"] == 24
    assert receipt["upscale"]["model_id"] == "realesrgan-x4plus"
    assert receipt["settings"]["enhance_output_subdir"] == "enhanced-images"
    assert receipt["receipt_id"]


def test_save_result_skips_receipt_when_save_disabled(tmp_path):
    flags = RuntimeFlags(data_dir=tmp_path)
    settings = UserSettings(save_images=False)
    devices = MagicMock()
    store = FilesystemImageStore(tmp_path / "outputs", settings=settings)
    service = EnhanceService(flags, settings, devices, store)

    result = service.save_result(Image.new("RGB", (8, 8)), "Restore", route="restore")

    assert result.image_path is None
    assert result.receipt_path is None
    assert result.message == "Done (save disabled in Settings)"
