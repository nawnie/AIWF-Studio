from pathlib import Path

from aiwf.core.config.settings import RuntimeFlags
from aiwf.infrastructure.diffusers.checkpoints import scan_checkpoints, scan_from_flags
from aiwf.infrastructure.diffusers.model_arch import looks_like_lora_weights


def test_scan_empty_dir(tmp_path: Path):
    assert scan_checkpoints([tmp_path]) == []


def test_scan_finds_nested_checkpoint(tmp_path: Path):
    models = tmp_path / "models"
    nested = models / "Stable-diffusion" / "sub"
    nested.mkdir(parents=True)
    model = nested / "test_model.safetensors"
    model.write_bytes(b"fake model content for test")

    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)
    checkpoints = scan_from_flags(flags)
    assert len(checkpoints) == 1
    assert checkpoints[0].filename == "test_model.safetensors"


def test_scan_skips_loras_folder(tmp_path: Path):
    models = tmp_path / "models"
    lora_dir = models / "Loras"
    lora_dir.mkdir(parents=True)
    (lora_dir / "style.safetensors").write_bytes(b"fake")

    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)
    assert scan_from_flags(flags) == []


def test_looks_like_lora_weights_detects_lora_keys(tmp_path: Path):
    import json
    import struct

    meta = {
        "lora_unet_down_blocks_0.lora_down.weight": {
            "dtype": "F16",
            "shape": [4, 4],
            "data_offsets": [0, 32],
        }
    }
    body = json.dumps(meta).encode("utf-8")
    path = tmp_path / "adapter.safetensors"
    path.write_bytes(struct.pack("<Q", len(body)) + body)
    assert looks_like_lora_weights(path) is True


def test_scan_finds_model_in_models_root(tmp_path: Path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "root_model.ckpt").write_bytes(b"root checkpoint")

    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)
    checkpoints = scan_from_flags(flags)
    assert any(c.filename == "root_model.ckpt" for c in checkpoints)


def test_scan_does_not_fallback_to_neighbor_webui_models(tmp_path: Path):
    models = tmp_path / "models"
    models.mkdir()
    neighbor = tmp_path.parent / "stable-diffusion-webui" / "models" / "Stable-diffusion"
    neighbor.mkdir(parents=True, exist_ok=True)
    fallback_model = neighbor / "legacy_model.safetensors"
    fallback_model.write_bytes(b"legacy checkpoint")

    try:
        flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)
        checkpoints = scan_from_flags(flags)
        assert not any(c.filename == "legacy_model.safetensors" for c in checkpoints)
    finally:
        fallback_model.unlink(missing_ok=True)
