from __future__ import annotations

import json
import struct
from pathlib import Path

from aiwf.core.config.settings import RuntimeFlags
from aiwf.infrastructure.diffusers.checkpoints import scan_from_flags
from aiwf.infrastructure.diffusers.loras import scan_loras
from aiwf.infrastructure.model_header import (
    ARCH_FLUX2_KLEIN_TRANSFORMER,
    ARCH_FLUX_LORA,
    ARCH_FLUX_VAE,
    ARCH_SD35_CHECKPOINT,
    ARCH_T5XXL_ENCODER,
    ARCH_UMT5_ENCODER,
    ARCH_Z_IMAGE_TRANSFORMER,
    read_model_info,
)
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
    assert folder_record.current_subdir == "Stable-diffusion"
    assert folder_record.recommended_subdir == "Stable-diffusion"
    assert folder_record.should_move is False
    assert [checkpoint.id for checkpoint in checkpoints] == ["stable-diffusion-3.5-medium"]
    assert checkpoints[0].architecture == "sd35"


def test_sd35_all_in_one_metadata_is_not_misclassified_as_vae(tmp_path: Path):
    models = tmp_path / "models"
    checkpoint = models / "Stable-diffusion" / "sd3.5_large_fp8_scaled.safetensors"
    _write_safetensors_header(
        checkpoint,
        {},
        {
            "modelspec.architecture": "stable-diffusion-v3.5-large",
            "modelspec.description": (
                "SD3.5-large all-in-one checkpoint with text encoder and vae weights."
            ),
        },
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    records = scan_and_write_model_inventory(flags)
    checkpoints = scan_from_flags(flags)

    record = next(item for item in records if item.filename == checkpoint.name)
    assert record.family == "checkpoint"
    assert record.architecture == "sd35"
    assert record.recommended_subdir == "Stable-diffusion"
    assert [item.id for item in checkpoints] == ["sd3.5_large_fp8_scaled"]
    assert checkpoints[0].architecture == "sd35"


def test_model_header_labels_sd3_joint_blocks_as_sd35(tmp_path: Path):
    model = tmp_path / "models" / "Stable-diffusion" / "sd3_medium.safetensors"
    _write_safetensors_header(
        model,
        {
            "model.diffusion_model.joint_blocks.0.attn.qkv.weight": {
                "dtype": "F16",
                "shape": [4, 4],
                "data_offsets": [0, 32],
            }
        },
    )

    info = read_model_info(model)

    assert info.arch == ARCH_SD35_CHECKPOINT
    assert "SD3.5" in info.display_name


def test_flux_gguf_is_runtime_asset_and_selectable_flux_checkpoint(tmp_path: Path):
    models = tmp_path / "models"
    flux = models / "Stable-diffusion" / "flux1-dev-Q5_K_M.gguf"
    flux.parent.mkdir(parents=True)
    flux.write_bytes(b"GGUF")
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    records = scan_and_write_model_inventory(flags)
    checkpoints = scan_from_flags(flags)

    record = next(item for item in records if item.filename == flux.name)
    assert record.family == "runtime_asset"
    assert record.architecture == "flux"
    assert record.recommended_subdir == "flux/GGUF"
    assert [checkpoint.id for checkpoint in checkpoints] == ["flux1-dev-Q5_K_M"]
    assert checkpoints[0].architecture == "flux"
    assert checkpoints[0].kind == "flux"


def test_flux2_klein_gguf_is_runtime_asset_and_selectable_flux2_checkpoint(tmp_path: Path):
    models = tmp_path / "models"
    klein = models / "Stable-diffusion" / "fluxtraitFLUX2KleinFLUXZ_klein9bV2Q4KM.gguf"
    klein.parent.mkdir(parents=True)
    klein.write_bytes(b"GGUF")
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    info = read_model_info(klein)
    records = scan_and_write_model_inventory(flags)
    checkpoints = scan_from_flags(flags)

    record = next(item for item in records if item.filename == klein.name)
    assert info.arch == ARCH_FLUX2_KLEIN_TRANSFORMER
    assert record.family == "runtime_asset"
    assert record.architecture == "flux2_klein"
    assert record.recommended_subdir == "flux2/GGUF"
    assert [checkpoint.id for checkpoint in checkpoints] == ["fluxtraitFLUX2KleinFLUXZ_klein9bV2Q4KM"]
    assert checkpoints[0].architecture == "flux2_klein"
    assert checkpoints[0].kind == "flux2"


def test_z_image_gguf_is_runtime_asset_and_selectable_z_image_checkpoint(tmp_path: Path):
    models = tmp_path / "models"
    z_image = models / "flux" / "GGUF" / "fluxtraitFLUX2KleinFLUXZ_zImageV2GgufQ4.gguf"
    z_image.parent.mkdir(parents=True)
    z_image.write_bytes(b"GGUF")
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    info = read_model_info(z_image)
    records = scan_and_write_model_inventory(flags)
    checkpoints = scan_from_flags(flags)

    record = next(item for item in records if item.filename == z_image.name)
    assert info.arch == ARCH_Z_IMAGE_TRANSFORMER
    assert record.family == "runtime_asset"
    assert record.architecture == "z_image"
    assert record.current_subdir == "flux/GGUF"
    assert record.recommended_subdir == "z-image/GGUF"
    assert record.should_move is True
    assert [checkpoint.id for checkpoint in checkpoints] == ["fluxtraitFLUX2KleinFLUXZ_zImageV2GgufQ4"]
    assert checkpoints[0].architecture == "z_image"
    assert checkpoints[0].kind == "z-image"


def test_flux_lora_header_overrides_wrong_folder(tmp_path: Path):
    models = tmp_path / "models"
    flux_lora = models / "Loras" / "Wan" / "flux_motion_rank16.safetensors"
    _write_safetensors_header(
        flux_lora,
        {
            "lora_transformer_double_blocks_0_img_attn_qkv.lora_down.weight": {
                "dtype": "F16",
                "shape": [4, 4],
                "data_offsets": [0, 32],
            }
        },
        {"ss_base_model_version": "flux1-dev"},
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    info = read_model_info(flux_lora)
    records = scan_and_write_model_inventory(flags)
    image_loras = scan_loras(flags)

    record = next(item for item in records if item.filename == flux_lora.name)
    assert info.arch == ARCH_FLUX_LORA
    assert record.family == "lora"
    assert record.architecture == "flux"
    assert record.recommended_subdir == "Loras/Flux"
    assert [item.filename for item in image_loras] == [flux_lora.name]
    assert image_loras[0].architecture == "flux"


def test_wan_lora_header_overrides_wrong_flux_folder_and_stays_out_of_image_loras(tmp_path: Path):
    models = tmp_path / "models"
    wan_lora = models / "Loras" / "Flux" / "wan_motion_rank16.safetensors"
    _write_safetensors_header(
        wan_lora,
        {
            "diffusion_model.blocks.0.cross_attn.k.lora_A.weight": {
                "dtype": "F16",
                "shape": [4, 4],
                "data_offsets": [0, 32],
            }
        },
        {"ss_base_model_version": "wan2.2"},
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    records = scan_and_write_model_inventory(flags)
    image_loras = scan_loras(flags)

    record = next(item for item in records if item.filename == wan_lora.name)
    assert record.family == "lora"
    assert record.architecture == "wan"
    assert record.recommended_subdir == "Loras/Wan"
    assert image_loras == []


def test_t5xxl_and_umT5_headers_stay_separate(tmp_path: Path):
    models = tmp_path / "models"
    t5xxl = models / "Textencoder" / "t5xxl_fp8_e4m3fn.safetensors"
    umt5 = models / "Textencoder" / "umt5-xxl_fp8_e4m3fn.safetensors"
    tensor = {
        "encoder.block.0.layer.0.SelfAttention.q.weight": {
            "dtype": "F16",
            "shape": [4, 4],
            "data_offsets": [0, 32],
        }
    }
    _write_safetensors_header(t5xxl, tensor, {})
    _write_safetensors_header(umt5, tensor, {})
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    t5_info = read_model_info(t5xxl)
    umt5_info = read_model_info(umt5)
    records = {item.filename: item for item in scan_and_write_model_inventory(flags)}

    assert t5_info.arch == ARCH_T5XXL_ENCODER
    assert t5_info.is_t5xxl() is True
    assert records[t5xxl.name].family == "text_encoder"
    assert records[t5xxl.name].architecture == "flux"
    assert records[t5xxl.name].recommended_subdir == "flux/Textencoder"
    assert umt5_info.arch == ARCH_UMT5_ENCODER
    assert umt5_info.is_t5xxl() is False
    assert records[umt5.name].architecture == "wan"
    assert records[umt5.name].recommended_subdir == "Textencoder"


def test_flux_ae_vae_is_not_treated_as_wan_vae(tmp_path: Path):
    models = tmp_path / "models"
    ae = models / "VAE" / "ae.safetensors"
    _write_safetensors_header(
        ae,
        {
            "encoder.conv_in.weight": {
                "dtype": "F16",
                "shape": [4, 4],
                "data_offsets": [0, 32],
            }
        },
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    info = read_model_info(ae)
    record = next(item for item in scan_and_write_model_inventory(flags) if item.filename == ae.name)

    assert info.arch == ARCH_FLUX_VAE
    assert record.family == "vae"
    assert record.architecture == "flux"
    assert record.recommended_subdir == "flux/VAE"


def test_embeddings_folder_does_not_enter_checkpoint_catalog(tmp_path: Path):
    models = tmp_path / "models"
    embedding = models / "embeddings" / "EasyNegative.safetensors"
    _write_safetensors_header(
        embedding,
        {
            "emb_params": {
                "dtype": "F16",
                "shape": [77, 768],
                "data_offsets": [0, 1024],
            }
        },
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    records = scan_and_write_model_inventory(flags)
    checkpoints = scan_from_flags(flags)

    record = next(item for item in records if item.filename == "EasyNegative.safetensors")
    assert record.family == "embedding"
    assert record.recommended_subdir == "embeddings"
    assert checkpoints == []


def test_lora_architecture_can_come_from_parent_folder(tmp_path: Path):
    models = tmp_path / "models"
    lora = models / "Loras" / "SDXL" / "folder_tagged_style.safetensors"
    _write_safetensors_header(
        lora,
        {
            "lora_unet_down_blocks_0.lora_down.weight": {
                "dtype": "F16",
                "shape": [4, 4],
                "data_offsets": [0, 32],
            }
        },
        {"ss_network_module": "networks.lora"},
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    records = scan_and_write_model_inventory(flags)
    image_loras = scan_loras(flags)

    record = next(item for item in records if item.filename == "folder_tagged_style.safetensors")
    assert record.family == "lora"
    assert record.architecture == "sdxl"
    assert record.recommended_subdir == "Loras/SDXL"
    assert image_loras[0].architecture == "sdxl"


def test_ltx_assets_are_recommended_to_ltx_folders(tmp_path: Path):
    models = tmp_path / "models"
    checkpoint = models / "Stable-diffusion" / "ltx-2.3-22b-distilled-1.1.safetensors"
    upscaler = models / "upscalers" / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
    for path in (checkpoint, upscaler):
        _write_safetensors_header(
            path,
            {
                "transformer_blocks.0.weight": {
                    "dtype": "F16",
                    "shape": [4, 4],
                    "data_offsets": [0, 32],
                }
            },
        )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    records = {item.filename: item for item in scan_and_write_model_inventory(flags)}

    assert records[checkpoint.name].family == "ltx"
    assert records[checkpoint.name].architecture == "ltx"
    assert records[checkpoint.name].recommended_subdir == "ltx/checkpoints"
    assert records[upscaler.name].recommended_subdir == "ltx/upscalers"
