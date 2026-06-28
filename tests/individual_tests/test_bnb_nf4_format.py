import json
import struct
from pathlib import Path

import pytest

from aiwf.infrastructure.quant.bnb_nf4_format import (
    Bnb4BitReport,
    build_bnb_4bit_quantization_config,
    inspect_bnb_4bit_safetensors,
    normalize_bnb_4bit_compute_dtype,
    resolve_transformer_load_format,
    runtime_for_ada_4bit,
)


def _write_fake_safetensors(path: Path, keys: list[str], metadata: dict | None = None) -> None:
    header = {key: {"dtype": "I8", "shape": [128, 1], "data_offsets": [0, 128]} for key in keys}
    header["__metadata__"] = metadata or {}
    payload = json.dumps(header).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(payload)))
        handle.write(payload)


def test_detects_nf4_from_quant_state_keys(tmp_path):
    path = tmp_path / "fluxFusionV24StepsGGUFNF4_V2NF4.safetensors"
    _write_fake_safetensors(
        path,
        [
            "transformer_blocks.0.attn.to_q.weight",
            "transformer_blocks.0.attn.to_q.weight.quant_state.bitsandbytes__nf4",
        ],
    )
    report = inspect_bnb_4bit_safetensors(path)
    assert report.is_bnb_4bit
    assert report.format == "bnb_nf4"
    assert report.quant_type == "nf4"
    assert report.layout == "diffusers_or_unknown"
    assert report.quantized_linear_layers == 1
    assert resolve_transformer_load_format(path) == "nf4"


def test_detects_nf4_from_filename_when_sidecars_missing(tmp_path):
    path = tmp_path / "model_nf4.safetensors"
    _write_fake_safetensors(path, ["layer.weight"])
    report = inspect_bnb_4bit_safetensors(path)
    assert report.is_bnb_4bit
    assert report.quant_type == "nf4"
    assert report.layout == "diffusers_or_unknown"


def test_plain_safetensors_not_marked_bnb(tmp_path):
    path = tmp_path / "fluxtrait_v10.safetensors"
    _write_fake_safetensors(path, ["transformer_blocks.0.attn.to_q.weight"])
    report = inspect_bnb_4bit_safetensors(path)
    assert not report.is_bnb_4bit
    assert resolve_transformer_load_format(path) == "safetensors"


def test_build_bnb_config_for_nf4(tmp_path):
    path = tmp_path / "nf4.safetensors"
    _write_fake_safetensors(path, ["w.weight", "w.weight.quant_state.bitsandbytes__nf4"])
    report = inspect_bnb_4bit_safetensors(path)
    import torch

    config = build_bnb_4bit_quantization_config(report, compute_dtype=torch.bfloat16)
    assert config is not None
    assert config.load_in_4bit is True
    assert config.bnb_4bit_quant_type == "nf4"


def test_detects_flux_original_bnb_layout(tmp_path):
    path = tmp_path / "flux_original_nf4.safetensors"
    _write_fake_safetensors(
        path,
        [
            "single_blocks.0.linear1.weight",
            "single_blocks.0.linear1.weight.quant_state.bitsandbytes__nf4",
        ],
    )
    report = inspect_bnb_4bit_safetensors(path)

    assert report.is_bnb_4bit
    assert report.format == "bnb_nf4"
    assert report.quant_type == "nf4"
    assert report.layout == "flux_original_bnb"
    assert report.needs_custom_flux_bnb_loader is True
    assert report.supports_diffusers_single_file is False
    assert report.quantized_linear_layers == 1
    assert any("Diffusers single-file converter" in warning for warning in report.warnings)


def test_build_bnb_config_clamps_float8_compute_dtype(monkeypatch):
    import torch

    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch float8 unavailable")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    report = Bnb4BitReport(path="nf4.safetensors", format="bnb_nf4", quant_type="nf4")
    config = build_bnb_4bit_quantization_config(report, compute_dtype=torch.float8_e4m3fn)

    assert normalize_bnb_4bit_compute_dtype(torch.float8_e4m3fn) is torch.bfloat16
    assert config.bnb_4bit_compute_dtype is torch.bfloat16


def test_detects_fp4_from_quant_state_keys(tmp_path):
    path = tmp_path / "flux_fp4.safetensors"
    _write_fake_safetensors(
        path,
        [
            "transformer_blocks.0.attn.to_q.weight",
            "transformer_blocks.0.attn.to_q.weight.quant_state.bitsandbytes__fp4",
        ],
    )
    report = inspect_bnb_4bit_safetensors(path)
    assert report.is_bnb_4bit
    assert report.format == "bnb_fp4"
    assert report.quant_type == "fp4"
    assert resolve_transformer_load_format(path) == "fp4"


def test_build_bnb_config_for_fp4(tmp_path):
    path = tmp_path / "fp4.safetensors"
    _write_fake_safetensors(path, ["w.weight", "w.weight.quant_state.bitsandbytes__fp4"])
    report = inspect_bnb_4bit_safetensors(path)
    import torch

    config = build_bnb_4bit_quantization_config(report, compute_dtype=torch.bfloat16)
    assert config is not None
    assert config.load_in_4bit is True
    assert config.bnb_4bit_quant_type == "fp4"
    assert config.bnb_4bit_compute_dtype is torch.bfloat16


def test_nvfp4_filename_is_storage_only_not_bnb(tmp_path):
    path = tmp_path / "ltx-2.3-22b-dev-nvfp4.safetensors"
    _write_fake_safetensors(path, ["layer.weight"])
    report = inspect_bnb_4bit_safetensors(path)
    assert report.format == "nvfp4_storage"
    assert report.quant_type == "nvfp4"
    assert report.is_bnb_4bit is False
    assert report.is_storage_only_4bit is True
    assert resolve_transformer_load_format(path) == "safetensors"


def test_fp4_filename_without_bnb_sidecars_is_storage_only(tmp_path):
    path = tmp_path / "model_fp4.safetensors"
    _write_fake_safetensors(path, ["layer.weight"])
    report = inspect_bnb_4bit_safetensors(path)
    assert report.format == "fp4_storage"
    assert report.quant_type == "fp4"
    assert report.is_bnb_4bit is False
    assert report.is_storage_only_4bit is True
    assert build_bnb_4bit_quantization_config(report, compute_dtype=__import__("torch").bfloat16) is None


def test_runtime_for_ada_4bit_prefers_bf16_when_supported(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    runtime = runtime_for_ada_4bit()

    assert runtime.available is True
    assert runtime.backend == "bitsandbytes_4bit"
    assert runtime.compute_dtype == "bfloat16"
    assert "not NVFP4" in runtime.reason
