from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

LTX_ENGINE_NAME = "ltx"
LTX_REPO_URL = "https://github.com/Lightricks/LTX-2.git"
LTX_MODEL_REPO = "Lightricks/LTX-2.3"
LTX_GEMMA_REPO = "google/gemma-3-12b-it-qat-q4_0-unquantized"

LTX_PIPELINE_DISTILLED = "distilled"
LTX_PIPELINE_ONE_STAGE = "one_stage"
LTX_PIPELINE_DIFFUSERS_2B = "diffusers_2b"
LTX_PIPELINES = (LTX_PIPELINE_DIFFUSERS_2B, LTX_PIPELINE_DISTILLED, LTX_PIPELINE_ONE_STAGE)

LTX_OFFLOAD_MODES = ("none", "cpu", "disk")
LTX_QUANTIZATION_MODES = ("", "fp8-cast", "fp8-scaled-mm")
LTX_GEMMA_BACKEND_HF_SAFETENSORS = "hf_safetensors"
LTX_GEMMA_BACKEND_GGUF = "gguf"
LTX_GEMMA_BACKENDS = (LTX_GEMMA_BACKEND_HF_SAFETENSORS, LTX_GEMMA_BACKEND_GGUF)

LTX_DISTILLED_CHECKPOINT = "ltx-2.3-22b-distilled-1.1.safetensors"
LTX_FULL_CHECKPOINT = "ltx-2.3-22b-dev-bf16.safetensors"
LTX_FULL_CHECKPOINT_FP8 = "ltx-2.3-22b-dev-fp8.safetensors"
LTX_FULL_CHECKPOINT_NVFP4 = "ltx-2.3-22b-dev-nvfp4.safetensors"
LTX_DIFFUSERS_2B_CHECKPOINT = "ltx-video-2b-v0.9.5.safetensors"
LTX_SPATIAL_UPSCALER_X2 = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
LTX_HERETIC_Q3_GGUF = "gemma-3-12b-it-heretic-Q3_K_M.gguf"
LTX_HERETIC_Q3_CONVERTED_FOLDER = "gemma-3-12b-heretic-q3km-converted"
LTX_T5XXL_FP16 = "t5xxl_fp16.safetensors"
LTX_T5_TOKENIZER = "google/t5-v1_1-xxl"


def ltx_frame_count_valid(value: int) -> bool:
    return int(value) >= 9 and (int(value) - 1) % 8 == 0


def snap_ltx_num_frames(value: int) -> int:
    safe = max(9, int(value))
    k = round((safe - 1) / 8)
    return max(9, 8 * k + 1)


class LtxVideoRequest(BaseModel):
    """Request for LTX video generation.

    LTX-2.3 uses the isolated upstream worker and Gemma. The local
    Diffusers 2B fallback uses the app venv, a single-file LTX checkpoint,
    and a T5XXL text encoder.
    """

    prompt: str = ""
    negative_prompt: str = ""
    source_image_path: str | None = None
    pipeline: str = LTX_PIPELINE_ONE_STAGE
    checkpoint_path: str = ""
    spatial_upsampler_path: str = ""
    gemma_root: str = ""
    gemma_backend: str = LTX_GEMMA_BACKEND_HF_SAFETENSORS
    gemma_gguf_path: str = ""
    t5_encoder_path: str = ""
    t5_tokenizer: str = LTX_T5_TOKENIZER
    width: int = Field(default=128, ge=128, le=2048)
    height: int = Field(default=128, ge=128, le=2048)
    num_frames: int = Field(default=9, ge=9, le=257)
    fps: float = Field(default=8.0, ge=1.0, le=60.0)
    seed: int = -1
    steps: int = Field(default=1, ge=1, le=100)
    image_strength: float = Field(default=0.8, ge=0.0, le=1.0)
    offload: str = "disk"
    quantization: str = "fp8-cast"
    max_batch_size: int = Field(default=1, ge=1, le=4)
    enhance_prompt: bool = False

    @field_validator("pipeline")
    @classmethod
    def _validate_pipeline(cls, value: str) -> str:
        normalized = (value or LTX_PIPELINE_ONE_STAGE).strip().lower()
        if normalized not in LTX_PIPELINES:
            raise ValueError(f"pipeline must be one of {LTX_PIPELINES}, got {value!r}")
        return normalized

    @field_validator("gemma_backend")
    @classmethod
    def _validate_gemma_backend(cls, value: str) -> str:
        normalized = (value or LTX_GEMMA_BACKEND_HF_SAFETENSORS).strip().lower()
        if normalized in {"hf", "safetensors", "hf-safetensors"}:
            normalized = LTX_GEMMA_BACKEND_HF_SAFETENSORS
        if normalized not in LTX_GEMMA_BACKENDS:
            raise ValueError(f"gemma_backend must be one of {LTX_GEMMA_BACKENDS}, got {value!r}")
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
