from __future__ import annotations

from typing import Literal

from PIL import Image
from pydantic import BaseModel, Field, model_validator

from aiwf.core.domain.segment_presets import CUSTOM_SEGMENT_PRESET_ID

ImageWorkflowStage = Literal[
    "auto_mask",
    "inpaint",
    "restore",
    "denoise",
    "tone",
    "upscale",
    "resize",
    "export",
]

IMAGE_WORKFLOW_ORDER: tuple[ImageWorkflowStage, ...] = (
    "auto_mask",
    "inpaint",
    "denoise",
    "restore",
    "tone",
    "upscale",
    "resize",
    "export",
)

IMAGE_STAGE_LABELS: dict[str, str] = {
    "auto_mask": "Auto mask",
    "inpaint": "Inpaint / repair",
    "restore": "Face / detail restore",
    "denoise": "Denoise",
    "tone": "Tone and color",
    "upscale": "AI upscale",
    "resize": "Final resize",
    "export": "Export",
}


class ImageWorkflowSettings(BaseModel):
    stages: list[ImageWorkflowStage] = Field(default_factory=lambda: ["tone", "export"])
    preset: str = "custom"

    mask_preset: str = "person"
    mask_custom_prompt: str = ""
    mask_model_id: str | None = None
    mask_threshold: float = Field(default=0.28, ge=0.05, le=0.95)
    mask_index: int = Field(default=0, ge=0, le=2)
    mask_dilation: int = Field(default=6, ge=0, le=128)
    mask_blur: int = Field(default=3, ge=0, le=64)
    mask_feather: int = Field(default=8, ge=0, le=64)

    inpaint_prompt: str = ""
    inpaint_negative_prompt: str = ""
    checkpoint_id: str | None = None
    sampler: str = "euler_a"
    steps: int = Field(default=24, ge=1, le=150)
    cfg_scale: float = Field(default=6.0, ge=0.0, le=30.0)
    seed: int = -1
    denoising_strength: float = Field(default=0.62, ge=0.0, le=1.0)

    restore_model_id: str | None = None
    restore_visibility: float = Field(default=0.85, ge=0.0, le=1.0)
    codeformer_weight: float = Field(default=0.5, ge=0.0, le=1.0)

    denoise_radius: int = Field(default=1, ge=1, le=4)
    denoise_strength: float = Field(default=0.35, ge=0.0, le=1.0)

    brightness: float = Field(default=1.0, ge=0.25, le=2.0)
    contrast: float = Field(default=1.0, ge=0.25, le=2.0)
    saturation: float = Field(default=1.0, ge=0.0, le=2.0)
    sharpness: float = Field(default=1.0, ge=0.0, le=2.0)

    upscaler_model_id: str | None = None
    upscale_factor: float = Field(default=2.0, ge=1.0, le=8.0)
    tile_size: int = Field(default=256, ge=0, le=2048)
    tile_overlap: int = Field(default=32, ge=0, le=512)

    resize_width: int = Field(default=0, ge=0, le=8192)
    resize_height: int = Field(default=0, ge=0, le=8192)
    keep_aspect: bool = True

    export_format: Literal["png", "jpg", "webp"] = "png"
    export_quality: int = Field(default=95, ge=40, le=100)

    @model_validator(mode="after")
    def normalize_stages(self):
        selected = set(self.stages)
        self.stages = [stage for stage in IMAGE_WORKFLOW_ORDER if stage in selected]
        if "export" not in self.stages:
            self.stages.append("export")
        if "inpaint" in self.stages and not (self.inpaint_prompt or "").strip():
            raise ValueError("Inpaint is enabled, but its prompt is empty.")
        if (
            "auto_mask" in self.stages
            and self.mask_preset == CUSTOM_SEGMENT_PRESET_ID
            and not (self.mask_custom_prompt or "").strip()
        ):
            raise ValueError("Auto mask uses Custom, but its mask prompt is empty.")
        return self


class ImageWorkflowPlan(BaseModel):
    stages: list[ImageWorkflowStage]
    labels: list[str]
    warnings: list[str] = Field(default_factory=list)

    @property
    def summary(self) -> str:
        return " → ".join(self.labels)


class ImageWorkflowResult(BaseModel):
    image: Image.Image
    output_path: str
    manifest_path: str
    mask: Image.Image | None = None
    mask_preview: Image.Image | None = None
    message: str
    stage_log: list[str] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}
