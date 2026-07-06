from __future__ import annotations

import json
import os
import struct
from pathlib import Path

from aiwf.core.config.settings import RuntimeFlags
from aiwf.infrastructure.diffusers.checkpoints import scan_from_flags
from aiwf.infrastructure.diffusers.loras import scan_loras
from aiwf.infrastructure.model_header import (
    ARCH_FLUX2_KLEIN_TRANSFORMER,
    ARCH_FLUX_LORA,
    ARCH_FLUX_VAE,
    ARCH_LTX_AUDIO_VAE,
    ARCH_LTX_LORA,
    ARCH_LTX_TRANSFORMER,
    ARCH_LTX_VAE,
    ARCH_SD_CHECKPOINT,
    ARCH_SD35_CHECKPOINT,
    ARCH_T5XXL_ENCODER,
    ARCH_UMT5_ENCODER,
    ARCH_Z_IMAGE_TRANSFORMER,
    ROLE_TEXT_ENCODER,
    ROLE_VAE,
    read_model_info,
)
from aiwf.infrastructure.model_inventory import inventory_path, scan_and_write_model_inventory
from aiwf.infrastructure.model_sorter import reorganize_models


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


def test_unknown_safetensor_goes_to_sort_bucket_not_sd15_catalog(tmp_path: Path):
    models = tmp_path / "models"
    unknown = models / "Stable-diffusion" / "mystery.safetensors"
    _write_safetensors_header(unknown, {})
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    records = scan_and_write_model_inventory(flags)
    checkpoints = scan_from_flags(flags)

    record = next(item for item in records if item.filename == "mystery.safetensors")
    assert record.family == "checkpoint"
    assert record.architecture == "unknown"
    assert record.recommended_subdir == "models to sort"
    assert record.should_move is True
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


def test_reorganize_moves_confident_gguf_without_overwriting(tmp_path: Path):
    models = tmp_path / "models"
    misplaced = models / "Stable-diffusion" / "flux1-dev-Q5_K_M.gguf"
    misplaced.parent.mkdir(parents=True)
    misplaced.write_bytes(b"GGUF")
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    actions = reorganize_models(flags)

    moved = [action for action in actions if action.status == "moved"]
    assert [action.filename for action in moved] == ["flux1-dev-Q5_K_M.gguf"]
    assert moved[0].dest_subdir == "flux/GGUF"
    assert not misplaced.exists()
    assert (models / "flux" / "GGUF" / "flux1-dev-Q5_K_M.gguf").is_file()


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


def test_comfy_saved_flux2_klein_safetensor_is_selectable_runtime_asset(tmp_path: Path):
    models = tmp_path / "models"
    klein = models / "flux" / "UNet" / "snofsSexNudesAndOtherFunStuff_v14Distilled.safetensors"
    _write_safetensors_header(
        klein,
        {
            "model.diffusion_model.double_blocks.0.img_attn.qkv.weight": {
                "dtype": "F8_E4M3",
                "shape": [12288, 4096],
                "data_offsets": [0, 1],
            }
        },
        metadata={
            "prompt": json.dumps(
                {
                    "1": {
                        "class_type": "UNETLoader",
                        "inputs": {"unet_name": "flux-2-klein-9b.safetensors"},
                    }
                }
            )
        },
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    info = read_model_info(klein)
    records = scan_and_write_model_inventory(flags)
    checkpoints = scan_from_flags(flags)

    record = next(item for item in records if item.filename == klein.name)
    assert info.arch == ARCH_FLUX2_KLEIN_TRANSFORMER
    assert record.family == "runtime_asset"
    assert record.architecture == "flux2_klein"
    assert record.current_subdir == "flux/UNet"
    assert record.recommended_subdir == "flux2/UNet"
    assert record.should_move is True
    assert [checkpoint.id for checkpoint in checkpoints] == ["snofsSexNudesAndOtherFunStuff_v14Distilled"]
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
    if os.name == "nt":
        # Z-Image GGUF is blocked on Windows: the fused GGUF CUDA kernels are
        # Linux-only and the fallback dequant path pages out 16 GB GPUs.
        assert checkpoints == []
    else:
        assert [checkpoint.id for checkpoint in checkpoints] == ["fluxtraitFLUX2KleinFLUXZ_zImageV2GgufQ4"]
        assert checkpoints[0].architecture == "z_image"
        assert checkpoints[0].kind == "z-image"


def test_qwen_and_sana_diffusers_dirs_are_selectable_runtime_assets(tmp_path: Path):
    models = tmp_path / "models"
    specs = [
        (
            models / "qwen-image" / "Diffusers" / "Qwen-Image-2512",
            "QwenImagePipeline",
            "qwen_image",
            "qwen-image/Diffusers",
            "qwen-image",
        ),
        (
            models / "sana" / "Diffusers" / "Sana_Sprint_1.6B_1024px_diffusers",
            "SanaSprintPipeline",
            "sana",
            "sana/Diffusers",
            "sana",
        ),
        (
            models / "krea2" / "Diffusers" / "Krea-2-Turbo",
            "Krea2Pipeline",
            "krea2",
            "krea2/Diffusers",
            "krea2",
        ),
    ]
    for root, class_name, _, _, _ in specs:
        root.mkdir(parents=True)
        (root / "model_index.json").write_text(json.dumps({"_class_name": class_name}), encoding="utf-8")
        (root / "transformer.safetensors").write_bytes(b"weights")
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    records = scan_and_write_model_inventory(flags)
    checkpoints = scan_from_flags(flags)
    by_path = {Path(record.path): record for record in records}
    by_kind = {checkpoint.kind: checkpoint for checkpoint in checkpoints}

    for root, _, architecture, recommended_subdir, kind in specs:
        record = by_path[root.resolve()]
        assert record.family == "runtime_asset"
        assert record.architecture == architecture
        assert record.recommended_subdir == recommended_subdir
        assert by_kind[kind].architecture == architecture
        assert by_kind[kind].file_count == 2
        assert by_kind[kind].asset_summary.startswith("folder, 2 files")
        assert by_kind[kind].asset_summary in by_kind[kind].title


def test_incomplete_qwen_diffusers_dir_is_not_selectable(tmp_path: Path):
    models = tmp_path / "models"
    root = models / "qwen-image" / "Diffusers" / "Qwen-Image"
    transformer = root / "transformer"
    transformer.mkdir(parents=True)
    (root / "model_index.json").write_text(json.dumps({"_class_name": "QwenImagePipeline"}), encoding="utf-8")
    (transformer / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 1},
                "weight_map": {
                    "transformer_blocks.0.attn.to_q.weight": "diffusion_pytorch_model-00001-of-00009.safetensors"
                },
            }
        ),
        encoding="utf-8",
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    records = scan_and_write_model_inventory(flags)
    checkpoints = scan_from_flags(flags)

    record = next(item for item in records if Path(item.path) == root.resolve())
    assert record.family == "runtime_asset"
    assert record.architecture == "qwen_image"
    assert checkpoints == []


def test_sana_video_diffusers_dir_is_video_runtime_asset_not_image_checkpoint(tmp_path: Path):
    models = tmp_path / "models"
    root = models / "sana-video" / "Diffusers" / "SANA-Video_2B_480p_diffusers"
    root.mkdir(parents=True)
    (root / "model_index.json").write_text(json.dumps({"_class_name": "SanaVideoPipeline"}), encoding="utf-8")
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    records = scan_and_write_model_inventory(flags)
    checkpoints = scan_from_flags(flags)
    record = next(item for item in records if Path(item.path) == root.resolve())

    assert record.family == "runtime_asset"
    assert record.architecture == "sana_video"
    assert record.recommended_subdir == "sana-video/Diffusers"
    assert record.should_move is False
    assert all(checkpoint.id != "SANA-Video_2B_480p_diffusers" for checkpoint in checkpoints)


def test_qwen_nunchaku_transformer_is_tracked_but_not_selectable_in_v1(tmp_path: Path):
    models = tmp_path / "models"
    transformer = models / "qwen-image" / "Nunchaku" / "svdq-int4_r32-qwen-image-lightningv1.0-4steps.safetensors"
    transformer.parent.mkdir(parents=True)
    _write_safetensors_header(
        transformer,
        {
            "transformer_blocks.0.attn.to_q.weight": {
                "dtype": "F16",
                "shape": [4, 4],
                "data_offsets": [0, 32],
            }
        },
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    records = scan_and_write_model_inventory(flags)
    checkpoints = scan_from_flags(flags)

    record = next(item for item in records if item.filename == transformer.name)
    assert record.family == "runtime_asset"
    assert record.architecture == "qwen_image_nunchaku"
    assert record.current_subdir == "qwen-image/Nunchaku"
    assert record.recommended_subdir == "qwen-image/Nunchaku"
    assert all(checkpoint.id != "svdq-int4_r32-qwen-image-lightningv1.0-4steps" for checkpoint in checkpoints)


def test_flux2_and_z_image_component_dirs_are_support_assets_not_checkpoints(tmp_path: Path):
    models = tmp_path / "models"
    component_specs = [
        (
            models / "flux2" / "Components" / "FLUX.2-klein-4B",
            "Flux2KleinPipeline",
            "flux2_klein",
            "flux2/Components",
        ),
        (
            models / "z-image" / "Components" / "Z-Image-Turbo",
            "ZImagePipeline",
            "z_image",
            "z-image/Components",
        ),
    ]
    for component, class_name, _, _ in component_specs:
        (component / "text_encoder").mkdir(parents=True)
        (component / "model_index.json").write_text(
            json.dumps({"_class_name": class_name}),
            encoding="utf-8",
        )
        (component / "text_encoder" / "model-00001-of-00002.safetensors").write_bytes(
            b"not-a-real-safetensors-header"
        )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    records = scan_and_write_model_inventory(flags)
    checkpoints = scan_from_flags(flags)
    by_path = {Path(record.path): record for record in records}

    for component, _, architecture, recommended_subdir in component_specs:
        record = by_path[component.resolve()]
        assert record.family == "text_encoder"
        assert record.architecture == architecture
        assert record.recommended_subdir == recommended_subdir
        assert record.should_move is False
    assert all(record.filename != "model-00001-of-00002.safetensors" for record in records)
    assert checkpoints == []


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


def test_flux2_klein_lora_metadata_routes_to_flux2_lora_folder(tmp_path: Path):
    models = tmp_path / "models"
    klein_lora = models / "Loras" / "party_time_v2.0_klein9b.safetensors"
    _write_safetensors_header(
        klein_lora,
        {
            "lora_transformer_double_blocks_0_img_attn_qkv.lora_A.weight": {
                "dtype": "BF16",
                "shape": [4, 4],
                "data_offsets": [0, 32],
            }
        },
        {"modelspec.architecture": "flux2-klein-9b/lora"},
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    info = read_model_info(klein_lora)
    record = next(item for item in scan_and_write_model_inventory(flags) if item.filename == klein_lora.name)

    assert info.arch == ARCH_FLUX_LORA
    assert record.family == "lora"
    assert record.architecture == "flux2_klein"
    assert record.recommended_subdir == "Loras/Flux2"


def test_ltx_headers_route_gguf_lora_and_vaes_to_ltx_folders(tmp_path: Path):
    models = tmp_path / "models"
    gguf = models / "ltx" / "GGUF" / "ltx23DISTILLEDGGUF_q2k.gguf"
    lora = models / "Stable-diffusion" / "ltx23_ltx2322bDistilled.safetensors"
    video_vae = models / "Stable-diffusion" / "ltx23FP4_ltx23VideoVae.safetensors"
    audio_vae = models / "Stable-diffusion" / "ltx23FP4_ltx23AudioVae.safetensors"
    text_encoder = models / "ltx" / "text_encoder" / "gemma_3_12B_it_fp4_mixed.safetensors"
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"GGUF")
    _write_safetensors_header(
        lora,
        {
            "diffusion_model.transformer_blocks.0.attn.q_proj.lora_A.weight": {
                "dtype": "BF16",
                "shape": [4, 4],
                "data_offsets": [0, 32],
            }
        },
        {"modelspec.title": "LTX 2.3 Distilled LoRA"},
    )
    _write_safetensors_header(
        video_vae,
        {
            "encoder.conv_in.weight": {
                "dtype": "BF16",
                "shape": [4, 4],
                "data_offsets": [0, 32],
            }
        },
        {"config": '{"class_name":"CausalVideoAutoencoder"}', "modelspec.title": "LTX Video VAE"},
    )
    _write_safetensors_header(
        audio_vae,
        {
            "audio_vae.encoder.conv_in.weight": {
                "dtype": "BF16",
                "shape": [4, 4],
                "data_offsets": [0, 32],
            }
        },
        {"modelspec.title": "LTX Audio VAE"},
    )
    _write_safetensors_header(
        text_encoder,
        {
            "model.layers.0.self_attn.q_proj.weight": {
                "dtype": "BF16",
                "shape": [4, 4],
                "data_offsets": [0, 32],
            }
        },
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    by_name = {item.filename: item for item in scan_and_write_model_inventory(flags)}
    info_by_name = {path.name: read_model_info(path) for path in (gguf, lora, video_vae, audio_vae, text_encoder)}

    assert info_by_name[gguf.name].arch == ARCH_LTX_TRANSFORMER
    assert by_name[gguf.name].family == "runtime_asset"
    assert by_name[gguf.name].recommended_subdir == "ltx/GGUF"
    assert info_by_name[lora.name].arch == ARCH_LTX_LORA
    assert by_name[lora.name].family == "lora"
    assert by_name[lora.name].recommended_subdir == "ltx/loras"
    assert info_by_name[video_vae.name].arch == ARCH_LTX_VAE
    assert by_name[video_vae.name].family == "vae"
    assert by_name[video_vae.name].recommended_subdir == "ltx/vae"
    assert info_by_name[audio_vae.name].arch == ARCH_LTX_AUDIO_VAE
    assert by_name[audio_vae.name].family == "vae"
    assert by_name[audio_vae.name].recommended_subdir == "ltx/audio_vae"
    assert info_by_name[text_encoder.name].role == ROLE_TEXT_ENCODER
    assert by_name[text_encoder.name].family == "text_encoder"
    assert by_name[text_encoder.name].recommended_subdir == "ltx/text_encoder"


def test_thumbnail_metadata_does_not_drive_ltx_detection(tmp_path: Path):
    path = tmp_path / "models" / "Stable-diffusion" / "sd15-with-thumbnail.safetensors"
    _write_safetensors_header(
        path,
        {
            "model.diffusion_model.input_blocks.0.0.weight": {
                "dtype": "F16",
                "shape": [320, 4, 3, 3],
                "data_offsets": [0, 23040],
            }
        },
        {
            "modelspec.title": "Stable Diffusion v1.5",
            "modelspec.thumbnail": "data:image/jpeg;base64,this-value-mentions-ltx-by-accident",
        },
    )

    info = read_model_info(path)

    assert info.arch == ARCH_SD_CHECKPOINT


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


def test_qwen_text_encoder_stays_in_textencoder_folder(tmp_path: Path):
    models = tmp_path / "models"
    qwen = models / "Textencoder" / "qwen_3_8b_fp8mixed.safetensors"
    _write_safetensors_header(
        qwen,
        {
            "model.layers.0.self_attn.q_proj.weight": {
                "dtype": "F8_E4M3",
                "shape": [4, 4],
                "data_offsets": [0, 32],
            }
        },
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    info = read_model_info(qwen)
    record = next(item for item in scan_and_write_model_inventory(flags) if item.filename == qwen.name)

    assert info.role == ROLE_TEXT_ENCODER
    assert record.family == "text_encoder"
    assert record.architecture == "unknown"
    assert record.recommended_subdir == "Textencoder"


def test_flux2_vae_role_overrides_transformer_filename(tmp_path: Path):
    models = tmp_path / "models"
    vae = models / "VAE" / "flux2-vae.safetensors"
    _write_safetensors_header(
        vae,
        {
            "encoder.conv_in.weight": {
                "dtype": "F32",
                "shape": [4, 4],
                "data_offsets": [0, 32],
            }
        },
    )
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    info = read_model_info(vae)
    record = next(item for item in scan_and_write_model_inventory(flags) if item.filename == vae.name)

    assert info.role == ROLE_VAE
    assert record.family == "vae"
    assert record.architecture == "flux2_klein"
    assert record.recommended_subdir == "VAE"


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


def test_llm_folder_weights_do_not_enter_checkpoint_catalog(tmp_path: Path):
    models = tmp_path / "models"
    gguf = models / "LLM" / "GGUF" / "gemma-3-12b-it-heretic" / "gemma-3-12b-it-heretic-Q4_K_M.gguf"
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"GGUF")
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=models)

    records = scan_and_write_model_inventory(flags)
    checkpoints = scan_from_flags(flags)
    by_name = {record.filename: record for record in records}

    assert by_name[gguf.name].family == "llm"
    assert by_name[gguf.name].recommended_subdir == "LLM/GGUF"
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

    assert records[checkpoint.name].family == "runtime_asset"
    assert records[checkpoint.name].architecture == "ltx"
    assert records[checkpoint.name].recommended_subdir == "ltx/checkpoints"
    assert records[upscaler.name].recommended_subdir == "ltx/upscalers"
