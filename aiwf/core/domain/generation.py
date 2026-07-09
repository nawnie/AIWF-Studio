from __future__ import annotations

from enum import Enum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator
from PIL import Image

from aiwf.core.domain.controlnet import ControlNetUnit


class GenerationMode(str, Enum):
    TXT2IMG = "txt2img"
    IMG2IMG = "img2img"
    INPAINT = "inpaint"


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class GenerationRequest(BaseModel):
    """Backend-neutral request contract for Studio image generation.

    Keep this free of UI widgets and backend-specific objects. Services may
    translate these fields into Diffusers, ONNX, or future runners.
    """

    mode: GenerationMode = GenerationMode.TXT2IMG
    prompt: str = ""
    negative_prompt: str = ""
    steps: int = Field(default=20, ge=1, le=150)
    cfg_scale: float = Field(default=7.0, ge=0.0, le=30.0)
    width: int = Field(default=512, ge=64, le=2048)
    height: int = Field(default=512, ge=64, le=2048)
    seed: int = Field(default=-1)
    sampler: str = "euler_a"
    scheduler: str = "automatic"
    batch_size: int = Field(default=1, ge=1, le=8)
    batch_count: int = Field(default=1, ge=1, le=8)
    denoising_strength: float = Field(default=0.75, ge=0.0, le=1.0)
    mask_blur: int = Field(default=4, ge=0, le=64)
    seam_erode: int = Field(
        default=1,
        ge=0,
        le=32,
        description="Shrink composite mask inward to reduce edge halos after inpaint",
    )
    inpaint_only_masked: bool = False
    inpaint_masked_padding: int = Field(default=32, ge=0, le=256)
    inpaint_mask_content: str = Field(default="original")  # fill | original | latent noise | latent nothing
    clip_skip: int = Field(default=1, ge=1, le=12)
    enable_hr: bool = False
    hr_scale: float = Field(default=2.0, ge=1.0, le=4.0)
    hr_steps: int = Field(default=20, ge=1, le=150)
    hr_denoising_strength: float = Field(default=0.35, ge=0.0, le=1.0)
    hr_upscaler: str = "lanczos"
    save_before_hires: bool = False
    save_interrupted: bool = False
    sdxl_refiner_enabled: bool = False
    sdxl_refiner_checkpoint_id: str | None = None
    sdxl_refiner_steps: int = Field(default=10, ge=1, le=150)
    sdxl_refiner_strength: float = Field(default=0.25, ge=0.0, le=1.0)
    checkpoint_id: str | None = None
    vae_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    save_images: bool = True
    training_metadata: bool = False
    prompt_file: str | None = None
    use_prompt_file: bool = False
    prompt_seed: int | None = None
    style_name: str | None = None
    style_prompt_template: str | None = None
    style_negative_template: str | None = None
    # Multiple units are preserved here; each backend decides how to fan them
    # out or reject unsupported combinations.
    controlnet_units: list[ControlNetUnit] = Field(default_factory=list)
    # Per-request engine selection. None/"aiwf" -> primary (Diffusers),
    # "sdcpp" -> stable-diffusion.cpp. Only honored when the dual backend
    # is active; single-backend deployments ignore it.
    pipeline_backend: str | None = None

    @field_validator("width", "height")
    @classmethod
    def must_be_multiple_of_8(cls, value: int) -> int:
        if value % 8 != 0:
            raise ValueError("dimensions must be a multiple of 8")
        return value

    @field_validator("hr_upscaler")
    @classmethod
    def normalize_hr_upscaler(cls, value: str) -> str:
        normalized = (value or "lanczos").strip().lower().replace(" ", "_")
        if normalized == "latent":
            normalized = "lanczos"
        if normalized not in {"lanczos", "bicubic", "nearest"}:
            raise ValueError("hr_upscaler must be lanczos, bicubic, or nearest")
        return normalized

    model_config = {"arbitrary_types_allowed": True}


class SavedArtifact(BaseModel):
    """A persisted output plus the infotext/metadata written with it."""

    path: str
    infotext: str
    receipt_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GenerationResult(BaseModel):
    """Completed image job result returned by generation services.

    `images` may hold in-memory previews, while `artifacts` records what was
    actually saved to disk for gallery/history workflows.
    """

    job_id: UUID
    images: list[Image.Image]
    seeds: list[int]
    infotexts: list[str]
    artifacts: list[SavedArtifact] = Field(default_factory=list)
    before_hires_images: list[Image.Image] = Field(default_factory=list)
    mode: GenerationMode
    elapsed_seconds: float = 0.0

    model_config = {"arbitrary_types_allowed": True}


class JobProgress(BaseModel):
    job_id: UUID
    state: JobState
    step: int = 0
    total_steps: int = 0
    message: str = ""
    current_image: Image.Image | None = None

    model_config = {"arbitrary_types_allowed": True}

    @property
    def percent(self) -> int:
        if self.total_steps <= 0:
            return 0
        return int(100 * self.step / self.total_steps)


class JobRecord(BaseModel):
    """Queue record shared by UI/status layers and generation runners."""

    id: UUID = Field(default_factory=uuid4)
    request: GenerationRequest
    state: JobState = JobState.QUEUED
    progress: JobProgress | None = None
    result: GenerationResult | None = None
    error: str | None = None

    model_config = {"arbitrary_types_allowed": True}
