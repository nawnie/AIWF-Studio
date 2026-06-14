from __future__ import annotations

import json
import struct
from pathlib import Path

from aiwf.infrastructure.diffusers.model_arch import (
    ARCH_INPAINT,
    ARCH_SD15,
    ARCH_SDXL,
    ARCH_SDXL_INPAINT,
    detect_checkpoint_architecture,
    infer_architecture_from_shapes,
)


def _write_fake_safetensors(path: Path, shapes: dict[str, list[int]]) -> None:
    header = {key: {"dtype": "F32", "shape": shape} for key, shape in shapes.items()}
    payload = json.dumps(header).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(payload)))
        handle.write(payload)


def test_infer_sd15_default():
    assert infer_architecture_from_shapes({}) == ARCH_SD15


def test_infer_sdxl_from_openclip_key():
    shapes = {"conditioner.embedders.1.model.ln_final.weight": [77, 1280]}
    assert infer_architecture_from_shapes(shapes) == ARCH_SDXL


def test_infer_sdxl_inpaint_from_nine_channel_unet():
    shapes = {
        "model.diffusion_model.input_blocks.0.0.weight": [320, 9, 3, 3],
        "conditioner.embedders.1.model.ln_final.weight": [77, 1280],
    }
    assert infer_architecture_from_shapes(shapes) == ARCH_SDXL_INPAINT


def test_infer_sd15_inpaint_from_nine_channel_unet():
    shapes = {"model.diffusion_model.input_blocks.0.0.weight": [320, 9, 3, 3]}
    assert infer_architecture_from_shapes(shapes) == ARCH_INPAINT


def test_detect_from_safetensors_header(tmp_path: Path):
    path = tmp_path / "xl_inpaint.safetensors"
    _write_fake_safetensors(
        path,
        {
            "model.diffusion_model.input_blocks.0.0.weight": [320, 9, 3, 3],
            "conditioner.embedders.1.model.ln_final.weight": [77, 1280],
        },
    )
    assert detect_checkpoint_architecture(path) == ARCH_SDXL_INPAINT


def test_detect_sdxl_filename_fallback(tmp_path: Path):
    path = tmp_path / "juggernaut_xl.safetensors"
    _write_fake_safetensors(path, {})
    assert detect_checkpoint_architecture(path) == ARCH_SDXL