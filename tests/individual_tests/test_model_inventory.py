from __future__ import annotations

import json
import struct
from pathlib import Path

from aiwf.core.config.settings import RuntimeFlags
from aiwf.infrastructure.diffusers.checkpoints import scan_from_flags
from aiwf.infrastructure.diffusers.loras import scan_loras
from aiwf.infrastructure.model_inventory import inventory_path, scan_and_write_model_inventory


def _write_safetensors_header(path: Path, tensors: dict, metadata: dict[str, str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = dict(tensors)
    if metadata:
        header["__metadata__"] = metadata
    body_size = 0
    for item in tensors.values():
        offsets = item.get("data_offsets", [0, 0])
        body_size = max(body_size, int(offsets[1]))
    payload = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(payload)) + payload + (b"\0" * body_size))


def test_inventory_finds_misplaced_sdxl_lora_and_writes_manifest(tmp_path: Path):
    models = tmp_path / "models"
    misplaced = models / "Stable-diffusion" / "style.safetensors"
    _write_safetensors_header(
        misplaced,
        {
            "lora_unet_down_blocks_0.lora_down.weight": {
                "dtype": "F16",
                "shape": [4, 4],
                "data_offsets": [0, 32],
            }
        },
        {
            "ss_network_module": "networks.lora",
            "ss_base_model_version": "sdxl_base_v1-0",
        },
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    records = scan_and_write_model_inventory(flags)
    loras = scan_loras(flags)
    checkpoints = scan_from_flags(flags)

    record = next(item for item in records if item.filename == "style.safetensors")
    assert record.family == "lora"
    assert record.architecture == "sdxl"
    assert record.recommended_subdir == "Loras/SDXL"
    assert record.should_move is True
    assert inventory_path(flags).is_file()
    assert [lora.filename for lora in loras] == ["style.safetensors"]
    assert loras[0].architecture == "sdxl"
    assert checkpoints == []


def test_inventory_excludes_reactor_face_embedding_from_checkpoints(tmp_path: Path):
    models = tmp_path / "models"
    face = models / "reactor" / "faces" / "person.safetensors"
    _write_safetensors_header(
        face,
        {
            "embedding": {"dtype": "F32", "shape": [512], "data_offsets": [0, 2048]},
            "bbox": {"dtype": "F32", "shape": [4], "data_offsets": [2048, 2064]},
            "kps": {"dtype": "F32", "shape": [5, 2], "data_offsets": [2064, 2104]},
        },
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    records = scan_and_write_model_inventory(flags)
    checkpoints = scan_from_flags(flags)

    record = next(item for item in records if item.filename == "person.safetensors")
    assert record.family == "face_embedding"
    assert record.recommended_subdir == "reactor/faces"
    assert checkpoints == []


def test_controlnet_and_wan_loras_do_not_enter_image_lora_catalog(tmp_path: Path):
    models = tmp_path / "models"
    control_lora = models / "controlnet" / "sai_xl_canny_128lora.safetensors"
    wan_lora = models / "Loras" / "Wan" / "motion_wan_rank16.safetensors"
    tensor = {
        "lora_unet_down_blocks_0.lora_down.weight": {
            "dtype": "F16",
            "shape": [4, 4],
            "data_offsets": [0, 32],
        }
    }
    _write_safetensors_header(control_lora, tensor, {"ss_base_model_version": "sdxl_base_v1-0"})
    _write_safetensors_header(wan_lora, tensor, {"ss_base_model_version": "wan2.2"})
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    records = scan_and_write_model_inventory(flags)
    image_loras = scan_loras(flags)

    by_name = {record.filename: record for record in records}
    assert by_name["sai_xl_canny_128lora.safetensors"].family == "controlnet"
    assert by_name["motion_wan_rank16.safetensors"].family == "lora"
    assert by_name["motion_wan_rank16.safetensors"].architecture == "wan"
    assert image_loras == []


def test_sd35_diffusers_folder_is_checkpoint(tmp_path: Path):
    models = tmp_path / "models"
    sd35 = models / "Stable-diffusion" / "stable-diffusion-3.5-medium"
    sd35.mkdir(parents=True)
    (sd35 / "model_index.json").write_text(
        json.dumps({"_class_name": "StableDiffusion3Pipeline"}),
        encoding="utf-8",
    )
    (sd35 / "transformer").mkdir()
    _write_safetensors_header(
        sd35 / "transformer" / "diffusion_pytorch_model.safetensors",
        {
            "transformer_blocks.0.attn.add_q_proj.weight": {
                "dtype": "F16",
                "shape": [4, 4],
                "data_offsets": [0, 32],
            }
        },
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    records = scan_and_write_model_inventory(flags)
    checkpoints = scan_from_flags(flags)

    folder_record = next(item for item in records if item.path == str(sd35.resolve()))
    assert folder_record.family == "checkpoint"
    assert folder_record.architecture == "sd35"
    assert [checkpoint.id for checkpoint in checkpoints] == ["stable-diffusion-3.5-medium"]
    assert checkpoints[0].architecture == "sd35"
