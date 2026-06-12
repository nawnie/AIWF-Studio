from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from aiwf.core.config.settings import RuntimeFlags
from aiwf.infrastructure.diffusers.controlnet_pipe import (
    assert_controlnet_checkpoint_compatible,
    infer_controlnet_architecture,
)
from aiwf.services.controlnet import ControlNetService


def test_infer_controlnet_architecture_detects_sd15_lora():
    name = "control_lora_rank128_v11p_sd15_canny_fp16.safetensors"
    assert infer_controlnet_architecture(name) == "sd15"


def test_infer_controlnet_architecture_detects_sdxl_filename():
    assert infer_controlnet_architecture("diffusion_pytorch_model_sdxl_canny.safetensors") == "sdxl"


def test_assert_rejects_sd15_controlnet_with_sdxl_checkpoint():
    with pytest.raises(ValueError, match="SD1.5 ControlNet"):
        assert_controlnet_checkpoint_compatible(
            "control_lora_rank128_v11p_sd15_canny_fp16.safetensors",
            "sdxl",
        )


def test_assert_allows_sd15_pairing():
    assert_controlnet_checkpoint_compatible(
        "control_lora_rank128_v11p_sd15_canny_fp16.safetensors",
        "sd15",
    )


def test_validate_enabled_requires_model_image_and_supported_mode(tmp_path: Path):
    service = ControlNetService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    service.ensure_dir()
    (service.models_dir() / "control_canny.safetensors").write_bytes(b"x")

    with pytest.raises(ValueError, match="only available"):
        service.validate_enabled(enabled=True, mode="inpaint", model_id="control_canny", control_image=None)

    with pytest.raises(ValueError, match="control image"):
        service.validate_enabled(enabled=True, mode="txt2img", model_id="control_canny", control_image=None)

    service.validate_enabled(
        enabled=True,
        mode="txt2img",
        model_id="control_canny",
        control_image=Image.new("RGB", (8, 8), "red"),
    )