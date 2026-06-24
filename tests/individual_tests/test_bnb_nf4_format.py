import json
import struct
from pathlib import Path

from aiwf.infrastructure.quant.bnb_nf4_format import (
    build_bnb_4bit_quantization_config,
    inspect_bnb_4bit_safetensors,
    resolve_transformer_load_format,
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
    assert resolve_transformer_load_format(path) == "nf4"


def test_detects_nf4_from_filename_when_sidecars_missing(tmp_path):
    path = tmp_path / "model_nf4.safetensors"
    _write_fake_safetensors(path, ["layer.weight"])
    report = inspect_bnb_4bit_safetensors(path)
    assert report.is_bnb_4bit
    assert report.quant_type == "nf4"


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