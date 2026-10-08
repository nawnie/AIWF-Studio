from __future__ import annotations

import json
import struct
import zipfile
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest

from aiwf.infrastructure.diffusers import model_arch
from aiwf.infrastructure.diffusers.model_arch import (
    ARCH_FLUX2,
    ARCH_FLUX2_KLEIN,
    ARCH_FLUX_KONTEXT,
    ARCH_INPAINT,
    ARCH_SD15,
    ARCH_SDXL,
    ARCH_SDXL_INPAINT,
    ARCH_UNKNOWN,
    detect_checkpoint_architecture,
    infer_architecture_from_shapes,
    looks_like_controlnet_weights,
)


def _write_fake_safetensors(path: Path, shapes: dict[str, list[int]]) -> None:
    header = {key: {"dtype": "F32", "shape": shape} for key, shape in shapes.items()}
    payload = json.dumps(header).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(payload)))
        handle.write(payload)


def test_infer_unknown_when_no_shape_or_name_evidence():
    assert infer_architecture_from_shapes({}) == ARCH_UNKNOWN


def test_infer_sd15_from_four_channel_unet_key():
    shapes = {"model.diffusion_model.input_blocks.0.0.weight": [320, 4, 3, 3]}
    assert infer_architecture_from_shapes(shapes) == ARCH_SD15


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


def test_flux2_kontext_filename_precedence_preserves_generic_and_klein(tmp_path: Path):
    cases = {
        "flux2-kontext-dev.safetensors": ARCH_FLUX_KONTEXT,
        "flux2-base-Q4_K_M.safetensors": ARCH_FLUX2,
        "flux2-klein-9b.safetensors": ARCH_FLUX2_KLEIN,
    }
    for filename, expected in cases.items():
        path = tmp_path / filename
        _write_fake_safetensors(path, {})
        assert detect_checkpoint_architecture(path) == expected


def test_controlnet_weights_are_detected_as_non_base_checkpoint(tmp_path: Path):
    path = tmp_path / "diffusion_pytorch_model.safetensors"
    _write_fake_safetensors(
        path,
        {
            "controlnet_cond_embedding.conv_in.weight": [16, 3, 3, 3],
            "controlnet_down_blocks.0.weight": [320, 320, 1, 1],
        },
    )

    assert looks_like_controlnet_weights(path)


def test_legacy_checkpoint_shape_inspection_uses_memory_mapping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "ambiguous.ckpt"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data.pkl", b"checkpoint fixture")
    call_options = {}

    def fake_load(_path, **kwargs):
        call_options.update(kwargs)
        return {
            "state_dict": {
                "model.diffusion_model.input_blocks.0.0.weight": SimpleNamespace(shape=(320, 4, 3, 3)),
            },
        }

    monkeypatch.setattr(torch, "load", fake_load)

    assert model_arch.detect_checkpoint_architecture(path) == ARCH_SD15
    assert call_options["mmap"] is True


def test_torchscript_archive_is_not_materialized_for_checkpoint_discovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "inpaint" / "big-lama.pt"
    path.parent.mkdir()
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("big-lama/data.pkl", b"scripted module")
        archive.writestr("big-lama/code/__torch__/lama.py", b"module source")
        archive.writestr("big-lama/constants.pkl", b"constants")

    def unexpected_load(*_args, **_kwargs):
        raise AssertionError("TorchScript archives must not be materialized during discovery")

    monkeypatch.setattr(torch, "load", unexpected_load)

    assert model_arch.is_torchscript_archive(path)
    assert model_arch._ckpt_tensor_shapes(path) == {}
    assert model_arch.detect_checkpoint_architecture(path) == ARCH_UNKNOWN


def test_regular_torch_checkpoint_is_not_mistaken_for_torchscript(tmp_path: Path):
    path = tmp_path / "ordinary.pt"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ordinary/data.pkl", b"checkpoint")
        archive.writestr("ordinary/data/0", b"tensor")

    assert not model_arch.is_torchscript_archive(path)


def test_large_zip_checkpoint_uses_filename_evidence_without_torch_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "ambiguous.ckpt"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ambiguous/data.pkl", b"checkpoint")
    monkeypatch.setattr(model_arch, "_MAX_LEGACY_CHECKPOINT_SHAPE_LOAD_BYTES", 1)

    def unexpected_load(*_args, **_kwargs):
        raise AssertionError("Oversized checkpoint must not be materialized during discovery")

    monkeypatch.setattr(torch, "load", unexpected_load)

    assert model_arch._ckpt_tensor_shapes(path) == {}


def test_small_non_zip_checkpoint_keeps_bounded_shape_inspection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "ambiguous.ckpt"
    path.write_bytes(b"small legacy checkpoint")
    call_options = {}

    def fake_load(_path, **kwargs):
        call_options.update(kwargs)
        return {
            "state_dict": {
                "model.diffusion_model.input_blocks.0.0.weight": SimpleNamespace(shape=(320, 4, 3, 3)),
            },
        }

    monkeypatch.setattr(torch, "load", fake_load)

    assert model_arch.detect_checkpoint_architecture(path) == ARCH_SD15
    assert call_options["weights_only"] is True
    assert "mmap" not in call_options


def test_large_non_zip_checkpoint_is_not_deserialized_for_discovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "ambiguous.ckpt"
    with path.open("wb") as handle:
        handle.truncate(model_arch._MAX_LEGACY_CHECKPOINT_SHAPE_LOAD_BYTES + 1)

    def unexpected_load(*_args, **_kwargs):
        raise AssertionError("Large legacy checkpoint must not be deserialized during discovery")

    monkeypatch.setattr(torch, "load", unexpected_load)

    assert model_arch.detect_checkpoint_architecture(path) == ARCH_UNKNOWN
