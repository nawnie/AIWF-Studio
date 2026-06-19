from __future__ import annotations

from pathlib import Path

from aiwf.core.config.settings import RuntimeFlags
from aiwf.infrastructure.diffusers.checkpoints import resolve_search_roots
from aiwf.infrastructure.diffusers.embeddings import resolve_embedding_roots
from aiwf.infrastructure.diffusers.loras import resolve_lora_roots
from aiwf.infrastructure.diffusers.vae import resolve_vae_roots
from aiwf.infrastructure.enhance.catalog import EnhanceModelCatalog
from aiwf.services.controlnet import ControlNetService, resolve_controlnet_roots


def test_extra_checkpoint_and_model_roots_are_scanned(tmp_path: Path):
    primary_models = tmp_path / "models"
    primary_ckpts = primary_models / "Stable-diffusion"
    extra_models = tmp_path / "shared-models"
    extra_ckpts = tmp_path / "shared-checkpoints"
    for path in (
        primary_models,
        primary_ckpts,
        extra_models,
        extra_models / "Stable-diffusion",
        extra_ckpts,
    ):
        path.mkdir(parents=True, exist_ok=True)

    flags = RuntimeFlags(
        data_dir=tmp_path,
        models_dir=primary_models,
        ckpt_dir=primary_ckpts,
        extra_model_dirs=[extra_models],
        extra_ckpt_dirs=[extra_ckpts],
    )

    roots = resolve_search_roots(flags)

    assert primary_ckpts.resolve() in roots
    assert extra_models.resolve() in roots
    assert (extra_models / "Stable-diffusion").resolve() in roots
    assert extra_ckpts.resolve() in roots


def test_extra_model_roots_extend_lora_vae_and_embedding_scans(tmp_path: Path):
    extra_models = tmp_path / "shared-models"
    lora_root = extra_models / "Loras"
    vae_root = extra_models / "VAE"
    embedding_root = extra_models / "Embeddings"
    for path in (lora_root, vae_root, embedding_root):
        path.mkdir(parents=True, exist_ok=True)

    flags = RuntimeFlags(data_dir=tmp_path, extra_model_dirs=[extra_models])

    assert lora_root.resolve() in resolve_lora_roots(flags)
    assert vae_root.resolve() in resolve_vae_roots(flags)
    assert embedding_root.resolve() in resolve_embedding_roots(flags)


def test_direct_category_paths_are_supported_as_extra_model_roots(tmp_path: Path):
    lora_root = tmp_path / "loras"
    vae_root = tmp_path / "vae"
    embedding_root = tmp_path / "embeddings"
    for path in (lora_root, vae_root, embedding_root):
        path.mkdir(parents=True, exist_ok=True)

    flags = RuntimeFlags(
        data_dir=tmp_path,
        extra_model_dirs=[lora_root, vae_root, embedding_root],
    )

    assert lora_root.resolve() in resolve_lora_roots(flags)
    assert vae_root.resolve() in resolve_vae_roots(flags)
    assert embedding_root.resolve() in resolve_embedding_roots(flags)


def test_extra_model_roots_extend_controlnet_and_enhance_scans(tmp_path: Path):
    extra_models = tmp_path / "shared-models"
    controlnet_root = extra_models / "ControlNet"
    upscaler_root = extra_models / "upscale_models"
    restorer_root = extra_models / "facerestore_models"
    controlnet_root.mkdir(parents=True, exist_ok=True)
    upscaler_root.mkdir(parents=True, exist_ok=True)
    restorer_root.mkdir(parents=True, exist_ok=True)
    (controlnet_root / "depth.safetensors").write_text("x", encoding="utf-8")
    annotators = controlnet_root / "Annotators"
    annotators.mkdir()
    (annotators / "body_pose_model.pth").write_text("x", encoding="utf-8")
    union = controlnet_root / "controlnet-union-sdxl-1.0"
    union.mkdir()
    (union / "config.json").write_text("{}", encoding="utf-8")
    (union / "diffusion_pytorch_model.safetensors").write_text("x", encoding="utf-8")
    collection = controlnet_root / "sd_control_collection"
    collection.mkdir()
    (collection / "diffusers_xl_canny_full.safetensors").write_text("x", encoding="utf-8")
    (collection / "ip-adapter_xl.pth").write_text("x", encoding="utf-8")
    (collection / "sai_xl_canny_128lora.safetensors").write_text("x", encoding="utf-8")
    (upscaler_root / "custom-upscale.pth").write_text("x", encoding="utf-8")
    (restorer_root / "custom-face.pth").write_text("x", encoding="utf-8")

    flags = RuntimeFlags(data_dir=tmp_path, extra_model_dirs=[extra_models])

    assert controlnet_root.resolve() in resolve_controlnet_roots(flags)
    controlnet_ids = {model.id for model in ControlNetService(flags).list_models()}
    assert "depth" in controlnet_ids
    assert "controlnet-union-sdxl-1.0" in controlnet_ids
    assert "diffusers_xl_canny_full" in controlnet_ids
    assert "body_pose_model" not in controlnet_ids
    assert "ip-adapter_xl" not in controlnet_ids
    assert "sai_xl_canny_128lora" not in controlnet_ids

    enhance_models = EnhanceModelCatalog(flags).list_models()
    assert any(model.id == "custom-upscale" for model in enhance_models)
    assert any(model.id == "custom-face" for model in enhance_models)
