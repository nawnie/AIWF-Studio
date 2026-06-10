from __future__ import annotations

from enum import Enum
from typing import Literal
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
    mode: GenerationMode = GenerationMode.TXT2IMG
    prompt: str = ""
    negative_prompt: str = ""
    steps: int = Field(default=20, ge=1, le=150)
    cfg_scale: float = Field(default=7.0, ge=1.0, le=30.0)
    width: int = Field(default=512, ge=64, le=2048)
    height: int = Field(default=512, ge=64, le=2048)
    seed: int = Field(default=-1)
    sampler: str = "euler_a"
    batch_size: int = Field(default=1, ge=1, le=8)
    batch_count: int = Field(default=1, ge=1, le=8)
    denoising_strength: float = Field(default=0.75, ge=0.0, le=1.0)
    mask_blur: int = Field(default=4, ge=0, le=64)
    clip_skip: int = Field(default=1, ge=1, le=12)
    enable_hr: bool = False
    hr_scale: float = Field(default=2.0, ge=1.0, le=4.0)
    hr_steps: int = Field(default=20, ge=1, le=150)
    hr_denoising_strength: float = Field(default=0.35, ge=0.0, le=1.0)
    hr_upscaler: str = "latent"
    checkpoint_id: str | None = None
    vae_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    save_images: bool = True
    prompt_file: str | None = None
    use_prompt_file: bool = False
    prompt_seed: int | None = None
    style_name: str | None = None
    style_prompt_template: str | None = None
    style_negative_template: str | None = None
    controlnet_units: list[ControlNetUnit] = Field(default_factory=list)

    @field_validator("width", "height")
    @classmethod
    def must_be_multiple_of_8(cls, value: int) -> int:
        if value % 8 != 0:
            raise ValueError("dimensions must be a multiple of 8")
        return value

    model_config = {"arbitrary_types_allowed": True}


class SavedArtifact(BaseModel):
    path: str
    infotext: str


class GenerationResult(BaseModel):
    job_id: UUID
    images: list[Image.Image]
    seeds: list[int]
    infotexts: list[str]
    artifacts: list[SavedArtifact] = Field(default_factory=list)
    mode: GenerationMode

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
    id: UUID = Field(default_factory=uuid4)
    request: GenerationRequest
    state: JobState = JobState.QUEUED
    progress: JobProgress | None = None
    result: GenerationResult | None = None
    error: str | None = None

    model_config = {"arbitrary_types_allowed": True}
