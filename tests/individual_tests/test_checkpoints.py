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
    assert checkpoints[0].file_count == 1
    assert checkpoints[0].size_bytes == model.stat().st_size
    assert "1 file" in checkpoints[0].title
    assert checkpoints[0].asset_summary in checkpoints[0].title


def test_scan_skips_loras_folder(tmp_path: Path):
    models = tmp_path / "models"
    lora_dir = models / "Loras"
    lora_dir.mkdir(parents=True)
    (lora_dir / "style.safetensors").write_bytes(b"fake")

    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)
    assert scan_from_flags(flags) == []


def test_scan_skips_non_image_model_runtime_folders(tmp_path: Path):
    models = tmp_path / "models"
    (models / "Textencoder").mkdir(parents=True)
    (models / "wan" / "Safetensor").mkdir(parents=True)
    (models / "diffusion_models").mkdir(parents=True)
    (models / "upscale_models").mkdir(parents=True)
    (models / "Textencoder" / "umt5-xxl.safetensors").write_bytes(b"text encoder")
    (models / "wan" / "Safetensor" / "wan2.2_ti2v_5B_fp16.safetensors").write_bytes(b"wan")
    (models / "diffusion_models" / "clip_l.safetensors").write_bytes(b"clip")
    (models / "upscale_models" / "4xBHI_dat2_real.safetensors").write_bytes(b"upscaler")
    (models / "root_model.ckpt").write_bytes(b"root checkpoint")

    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)
    checkpoints = scan_from_flags(flags)

    filenames = {c.filename for c in checkpoints}
    assert "root_model.ckpt" in filenames
    assert "umt5-xxl.safetensors" not in filenames
    assert "wan2.2_ti2v_5B_fp16.safetensors" not in filenames
    assert "clip_l.safetensors" not in filenames
    assert "4xBHI_dat2_real.safetensors" not in filenames


def test_known_broken_runtime_assets_do_not_enter_selectable_checkpoint_catalog(tmp_path: Path):
    models = tmp_path / "models"
    good = models / "Stable-diffusion" / "root_model.ckpt"
    flux_fp8 = models / "flux" / "UNet" / "fluxedUpFluxNSFW_110FP8.safetensors"
    flux_gguf = models / "flux" / "GGUF" / "fluxFusionV24StepsGGUFNF4_V2GGUFQ4KM.gguf"
    bad_upscaler = models / "upscale_models" / "4xBHI_dat2_multiblurjpg.safetensors"
    for path in (good, flux_fp8, flux_gguf, bad_upscaler):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake")

    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)
    checkpoints = scan_from_flags(flags)

    filenames = {checkpoint.filename for checkpoint in checkpoints}
    assert "root_model.ckpt" in filenames
    assert "fluxedUpFluxNSFW_110FP8.safetensors" not in filenames
    assert "fluxFusionV24StepsGGUFNF4_V2GGUFQ4KM.gguf" not in filenames
    assert "4xBHI_dat2_multiblurjpg.safetensors" not in filenames


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
