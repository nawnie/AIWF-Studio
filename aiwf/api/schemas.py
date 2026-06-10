from __future__ import annotations

from pydantic import BaseModel, Field

from aiwf.core.domain.controlnet import ControlNetUnit


class Txt2ImgPayload(BaseModel):
    prompt: str
    negative_prompt: str = ""
    steps: int = 20
    cfg_scale: float = 7.0
    width: int = 512
    height: int = 512
    seed: int = -1
    sampler: str = "euler_a"
    batch_size: int = 1
    batch_count: int = 1
    clip_skip: int = 1
    enable_hr: bool = False
    hr_scale: float = 2.0
    hr_steps: int = 20
    hr_denoising_strength: float = 0.35
    hr_upscaler: str = "latent"
    checkpoint_id: str | None = None
    vae_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    save_images: bool = True
    prompt_file: str | None = None
    use_prompt_file: bool = False
    style_name: str | None = None
    controlnet_units: list[ControlNetUnit] = Field(default_factory=list)


class Img2ImgPayload(Txt2ImgPayload):
    init_images: list[str] = Field(default_factory=list)
    denoising_strength: float = 0.75


class InpaintPayload(Img2ImgPayload):
    mask_image: str = ""
    mask_blur: int = 4


class BatchImg2ImgPayload(Img2ImgPayload):
    init_images: list[str] = Field(default_factory=list, min_length=1)


class PlotAxisPayload(BaseModel):
    field: str
    values: list = Field(default_factory=list)


class PlotPayload(BaseModel):
    base: Txt2ImgPayload
    axes: list[PlotAxisPayload] = Field(default_factory=list)


class EnhancePayload(BaseModel):
    image: str
    upscaler_id: str | None = None
    scale: float = Field(default=4.0, ge=1.0, le=8.0)
    restorer_id: str | None = None
    restore_visibility: float = Field(default=1.0, ge=0.0, le=1.0)
    codeformer_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    tile_size: int = Field(default=256, ge=0, le=2048)
    tile_overlap: int = Field(default=32, ge=0, le=512)


class ControlNetDetectPayload(BaseModel):
    image: str
    module: str = "canny"
    processor_res: int = Field(default=512, ge=64, le=2048)
    threshold_a: float = 100.0
    threshold_b: float = 200.0
