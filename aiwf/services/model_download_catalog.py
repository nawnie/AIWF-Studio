from __future__ import annotations

from aiwf.core.domain.model_download import CatalogEntry

_CN = "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main"

# Curated starter catalog — expand over time. Users can always paste custom HF/CivitAI URLs.
MODEL_DOWNLOAD_CATALOG: list[CatalogEntry] = [
    # Checkpoints (Hugging Face)
    CatalogEntry(
        key="hf-sd15-pruned",
        title="Stable Diffusion 1.5 Pruned",
        category="checkpoint",
        source="huggingface",
        repo_id="runwayml/stable-diffusion-v1-5",
        filename="v1-5-pruned-emaonly.safetensors",
        size_mb=4067,
        notes="Classic SD1.5 base model.",
    ),
    CatalogEntry(
        key="hf-sdxl-base",
        title="Stable Diffusion XL Base 1.0",
        category="checkpoint",
        source="huggingface",
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        filename="sd_xl_base_1.0.safetensors",
        size_mb=6600,
        notes="SDXL base — pair with a refiner for best results.",
    ),
    # Checkpoints (CivitAI — resolved via API at download time)
    CatalogEntry(
        key="civit-dreamshaper-8",
        title="DreamShaper 8",
        category="checkpoint",
        source="civitai",
        civitai_model_id=4384,
        size_mb=2000,
        notes="Popular SD1.5 checkpoint on CivitAI.",
    ),
    CatalogEntry(
        key="civit-juggernaut-xl",
        title="Juggernaut XL",
        category="checkpoint",
        source="civitai",
        civitai_model_id=133005,
        size_mb=6500,
        notes="Strong general-purpose SDXL checkpoint.",
    ),
    CatalogEntry(
        key="civit-realvisxl",
        title="RealVisXL V4.0",
        category="checkpoint",
        source="civitai",
        civitai_model_id=139562,
        size_mb=6500,
        notes="Photoreal SDXL checkpoint.",
    ),
    # LoRAs
    CatalogEntry(
        key="hf-lora-detail-tweaker",
        title="Detail Tweaker LoRA (SD1.5)",
        category="lora",
        source="huggingface",
        repo_id="goofyai/SDXL_Crystal_clear_4K",
        filename="add_detail.safetensors",
        size_mb=144,
        notes="Adds detail — works on SD1.5 pipelines.",
    ),
    CatalogEntry(
        key="hf-lora-xl-detail",
        title="SDXL Detail LoRA",
        category="lora",
        source="huggingface",
        repo_id="goofyai/SDXL_Crystal_clear_4K",
        filename="SDXLrender_v2.0.safetensors",
        size_mb=218,
    ),
    CatalogEntry(
        key="civit-lora-add-detail",
        title="Add More Details (LoRA)",
        category="lora",
        source="civitai",
        civitai_model_id=82098,
        size_mb=144,
    ),
    # VAE
    CatalogEntry(
        key="hf-vae-mse",
        title="SD VAE FT MSE Original",
        category="vae",
        source="huggingface",
        repo_id="stabilityai/sd-vae-ft-mse-original",
        filename="diffusion_pytorch_model.safetensors",
        size_mb=319,
        notes="Fixes washed-out SD1.5 colors.",
    ),
    CatalogEntry(
        key="hf-vae-sdxl",
        title="SDXL VAE (fp16 fix)",
        category="vae",
        source="huggingface",
        repo_id="madebyollin/sdxl-vae-fp16-fix",
        filename="sdxl_vae.safetensors",
        size_mb=319,
    ),
    # ControlNet (light)
    CatalogEntry(
        key="cn-canny-light",
        title="ControlNet Canny v1.1 Light",
        category="controlnet",
        source="direct",
        url=f"{_CN}/control_lora_rank128_v11p_sd15_canny_fp16.safetensors",
        size_mb=129,
    ),
    CatalogEntry(
        key="cn-depth-light",
        title="ControlNet Depth v1.1 Light",
        category="controlnet",
        source="direct",
        url=f"{_CN}/control_lora_rank128_v11f1p_sd15_depth_fp16.safetensors",
        size_mb=129,
    ),
    CatalogEntry(
        key="cn-openpose-light",
        title="ControlNet OpenPose v1.1 Light",
        category="controlnet",
        source="direct",
        url=f"{_CN}/control_lora_rank128_v11p_sd15_openpose_fp16.safetensors",
        size_mb=129,
    ),
    CatalogEntry(
        key="cn-tile-light",
        title="ControlNet Tile v1.1 Light",
        category="controlnet",
        source="direct",
        url=f"{_CN}/control_lora_rank128_v11f1e_sd15_tile_fp16.safetensors",
        size_mb=129,
    ),
    # Upscalers
    CatalogEntry(
        key="up-realesrgan-x4",
        title="RealESRGAN 4x+",
        category="upscaler",
        source="direct",
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        size_mb=64,
    ),
    CatalogEntry(
        key="up-realesrgan-anime",
        title="RealESRGAN 4x+ Anime6B",
        category="upscaler",
        source="direct",
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        size_mb=18,
    ),
    CatalogEntry(
        key="up-realesrgan-x2",
        title="RealESRGAN 2x+",
        category="upscaler",
        source="direct",
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        size_mb=64,
    ),
    # Face swap
    CatalogEntry(
        key="fs-inswapper",
        title="inswapper_128 (ReActor)",
        category="faceswap",
        source="direct",
        url="https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128.onnx",
        size_mb=554,
    ),
]