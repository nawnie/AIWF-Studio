from __future__ import annotations

from pathlib import Path

import pytest

from aiwf.core.config.settings import RuntimeFlags
from aiwf.infrastructure.diffusers.controlnet_pipe import is_control_lora_checkpoint
from aiwf.services.controlnet import DOWNLOADABLE_CONTROLNETS, ControlNetService


def test_downloadable_catalog_uses_light_sd15_checkpoints():
    assert DOWNLOADABLE_CONTROLNETS
    for item in DOWNLOADABLE_CONTROLNETS:
        assert item.base == "SD1.5"
        assert item.size_mb <= 150
        assert "control_lora_rank128" in item.filename
        assert item.url.endswith(item.filename)


def test_controlnet_service_lists_light_catalog(tmp_path: Path):
    service = ControlNetService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    keys = {item.key for item in service.list_downloadable()}
    assert {"canny", "depth", "openpose"} <= keys


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("control_lora_rank128_v11p_sd15_canny_fp16.safetensors", True),
        ("control_v11p_sd15_canny_fp16.safetensors", False),
    ],
)
def test_is_control_lora_checkpoint_by_filename(filename: str, expected: bool):
    assert is_control_lora_checkpoint(filename) is expected