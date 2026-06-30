from __future__ import annotations

import json
import struct
from pathlib import Path

from aiwf.core.config.settings import RuntimeFlags
from aiwf.infrastructure.diffusers.vae import scan_vaes


def _write_safetensors_header(path: Path, *, dtype: str = "F16") -> None:
    header = {
        "decoder.weight": {
            "dtype": dtype,
            "shape": [1],
            "data_offsets": [0, 2],
        }
    }
    body = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(body)) + body + b"00")


def test_scan_vaes_labels_size_and_precision(tmp_path: Path):
    models = tmp_path / "models"
    vae_dir = models / "VAE"
    vae_dir.mkdir(parents=True)
    vae = vae_dir / "sdxl_vae_fp16_fix.safetensors"
    _write_safetensors_header(vae, dtype="F16")

    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)
    vaes = scan_vaes(flags)

    assert len(vaes) == 1
    assert vaes[0].filename == vae.name
    assert vaes[0].precision == "fp16"
    assert vaes[0].file_count == 1
    assert vaes[0].size_bytes == vae.stat().st_size
    assert "1 file" in vaes[0].title
    assert "fp16" in vaes[0].title
