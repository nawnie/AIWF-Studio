from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

LTX_ENGINE_NAME = "ltx"
LTX_REPO_URL = "https://github.com/Lightricks/LTX-2.git"
LTX_MODEL_REPO = "Lightricks/LTX-2.3"
LTX_GEMMA_REPO = "google/gemma-3-12b-it-qat-q4_0-unquantized"

LTX_PIPELINE_DISTILLED = "distilled"
LTX_PIPELINE_ONE_STAGE = "one_stage"
LTX_PIPELINES = (LTX_PIPELINE_DISTILLED, LTX_PIPELINE_ONE_STAGE)

LTX_OFFLOAD_MODES = ("none", "cpu", "disk")
LTX_QUANTIZATION_MODES = ("", "fp8-cast", "fp8-scaled-mm")

LTX_DISTILLED_CHECKPOINT = "ltx-2.3-22b-distilled-1.1.safetensors"
LTX_FULL_CHECKPOINT = "ltx-2.3-22b-dev-bf16.safetensors"
LTX_FULL_CHECKPOINT_NVFP4 = "ltx-2.3-22b-dev-nvfp4.safetensors"
LTX_SPATIAL_UPSCALER_X2 = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"


def ltx_frame_count_valid(value: int) -> bool:
    return int(value) >= 9 and (int(value) - 1) % 8 == 0


def snap_ltx_num_frames(value: int) -> int:
    safe = max(9, int(value))
    k = round((safe - 1) / 8)
    return max(9, 8 * k + 1)


class LtxVideoRequest(BaseModel):
    """Request for the isolated LTX 2.3 video worker.

    This stays separate from Wan because LTX uses its own upstream repo,
    package set, Gemma text encoder, and 8k+1 frame convention.
    """

    prompt: str = ""
    negative_prompt: str = ""
    source_image_path: str | None = None
    pipeline: str = LTX_PIPELINE_DISTILLED
    checkpoint_path: str = ""
    spatial_upsampler_path: str = ""
    gemma_root: str = ""
    width: int = Field(default=512, ge=128, le=2048)
    height: int = Field(default=512, ge=128, le=2048)
    num_frames: int = Field(default=81, ge=9, le=257)
    fps: float = Field(default=25.0, ge=1.0, le=60.0)
    seed: int = -1
    steps: int = Field(default=20, ge=1, le=100)
    image_strength: float = Field(default=0.8, ge=0.0, le=1.0)
    offload: str = "cpu"
    quantization: str = "fp8-cast"
    max_batch_size: int = Field(default=1, ge=1, le=4)
    enhance_prompt: bool = False

    @field_validator("pipeline")
    @classmethod
    def _validate_pipeline(cls, value: str) -> str:
        normalized = (value or LTX_PIPELINE_DISTILLED).strip().lower()
        if normalized not in LTX_PIPELINES:
            raise ValueError(f"pipeline must be one of {LTX_PIPELINES}, got {value!r}")
        return normalized

    @field_validator("width", "height")
    @classmethod
    def _validate_size(cls, value: int) -> int:
        if int(value) % 32 != 0:
            raise ValueError("LTX width and height must be divisible by 32")
        return int(value)

    @field_validator("num_frames")
    @classmethod
    def _validate_frames(cls, value: int) -> int:
        if not ltx_frame_count_valid(int(value)):
            raise ValueError("LTX num_frames must be 8*k+1, for example 81 or 121")
        return int(value)

    @field_validator("offload")
    @classmethod
    def _validate_offload(cls, value: str) -> str:
        normalized = (value or "none").strip().lower()
        if normalized not in LTX_OFFLOAD_MODES:
            raise ValueError(f"offload must be one of {LTX_OFFLOAD_MODES}, got {value!r}")
        return normalized

    @field_validator("quantization")
    @classmethod
    def _validate_quantization(cls, value: str) -> str:
        normalized = (value or "").strip().lower()
        if normalized == "none":
            normalized = ""
        if normalized not in LTX_QUANTIZATION_MODES:
            raise ValueError(
                f"quantization must be empty, fp8-cast, or fp8-scaled-mm; got {value!r}"
            )
        return normalized


class LtxVideoResult(BaseModel):
    output_path: str
    message: str = "LTX 2.3 video complete"
    events: list[dict] = Field(default_factory=list)
    has_audio: bool = False
    audio_mode: str = "native"
