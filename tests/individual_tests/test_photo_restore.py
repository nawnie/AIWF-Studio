from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.enhance import RestoreOptions, UpscaleOptions
from aiwf.core.domain.photo_restore import PhotoRestoreOptions
from aiwf.infrastructure.enhance.photo_restore import (
    crop_to_box,
    detect_scratch_mask,
    global_restore_image,
    pad_to_multiple,
    run_photo_restore_stages,
)
from aiwf.services.enhance import EnhanceService


def test_pad_to_multiple_adds_border_and_crop_roundtrip():
    image = Image.new("RGB", (100, 50), color=(40, 80, 120))
    padded, crop_box = pad_to_multiple(image, 8)
    assert padded.size[0] % 8 == 0
    assert padded.size[1] % 8 == 0
    cropped = crop_to_box(padded, crop_box)
    assert cropped.size == image.size


def test_detect_scratch_mask_returns_grayscale():
    arr = np.full((64, 64, 3), 180, dtype=np.uint8)
    for offset in range(0, 48, 3):
        arr[10 + offset, 8 + offset] = 255
        arr[10 + offset, 9 + offset] = 245
    image = Image.fromarray(arr, "RGB")
    mask = detect_scratch_mask(image, sensitivity=0.75, dilation=2)
    assert mask.mode == "L"
    assert np.array(mask).max() > 0


def test_global_restore_preserves_size():
    image = Image.new("RGB", (128, 96), color=(90, 70, 50))
    result = global_restore_image(image, denoise_strength=0.5, color_boost=0.4)
    assert result.size == image.size


def test_run_photo_restore_stages_reports_steps():
    image = Image.new("RGB", (64, 64), color=(100, 100, 100))
    options = PhotoRestoreOptions(
        scratch_detection=False,
        global_restore=True,
        face_restore=False,
    )

    result, steps, crop_box = run_photo_restore_stages(
        image,
        options,
        face_restore_fn=lambda img: img,
    )
    assert "Global restore" in steps
    assert result.size[0] % 8 == 0


def test_enhance_service_run_photo_restore_orders_stages(tmp_path: Path):
    flags = RuntimeFlags(data_dir=tmp_path)
    settings = UserSettings(save_images=False)
    devices = MagicMock()
    devices.device.return_value = "cpu"
    store = MagicMock()
    service = EnhanceService(flags, settings, devices, store)

    image = Image.new("RGB", (64, 64), color=(120, 90, 60))
    global_out = Image.new("RGB", (64, 64), color=(130, 100, 70))
    face_out = Image.new("RGB", (64, 64), color=(140, 110, 80))
    up_out = Image.new("RGB", (128, 128), color=(150, 120, 90))

    options = PhotoRestoreOptions(
        scratch_detection=False,
        global_restore=True,
        face_restore=True,
        restore=RestoreOptions(model_id="gfpgan-v1.4"),
        upscale=UpscaleOptions(model_id="realesrgan-x2plus", scale=2),
    )

    with patch(
        "aiwf.services.enhance.run_photo_restore_stages",
        return_value=(global_out, ["Global restore", "Face enhance (gfpgan-v1.4)"], (0, 0, 64, 64)),
    ), patch.object(service, "restore", return_value=face_out) as restore_mock, patch.object(
        service, "upscale", return_value=up_out
    ) as upscale_mock:
        result, infotext = service.run_photo_restore(image, options)

    restore_mock.assert_not_called()
    upscale_mock.assert_called_once()
    assert result.size == (128, 128)
    assert "Upscale" in infotext