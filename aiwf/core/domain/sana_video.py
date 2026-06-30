from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, field_validator

SANA_VIDEO_MODEL_REPO_480P = "Efficient-Large-Model/SANA-Video_2B_480p_diffusers"
SANA_VIDEO_MODEL_REPO_720P = "Efficient-Large-Model/SANA-Video_2B_720p_diffusers"
SANA_VIDEO_PIPELINE_T2V = "text_to_video"
SANA_VIDEO_PIPELINE_I2V = "image_to_video"
SANA_VIDEO_PIPELINES = (SANA_VIDEO_PIPELINE_T2V, SANA_VIDEO_PIPELINE_I2V)
SANA_VIDEO_QUANTIZATION_AUTO = "auto"
SANA_VIDEO_QUANTIZATION_BF16 = "bf16"
SANA_VIDEO_QUANTIZATION_FP8 = "fp8_layerwise"
SANA_VIDEO_QUANTIZATION_BNB_INT8 = "bnb_int8"
SANA_VIDEO_QUANTIZATION_BNB_NF4 = "bnb_nf4"
SANA_VIDEO_QUANTIZATION_BNB_FP4 = "bnb_fp4"
SANA_VIDEO_QUANTIZATIONS = (
    SANA_VIDEO_QUANTIZATION_AUTO,
    SANA_VIDEO_QUANTIZATION_BF16,
    SANA_VIDEO_QUANTIZATION_FP8,
    SANA_VIDEO_QUANTIZATION_BNB_INT8,
    SANA_VIDEO_QUANTIZATION_BNB_NF4,
    SANA_VIDEO_QUANTIZATION_BNB_FP4,
)
SANA_VIDEO_VAE_TILING_AUTO = "auto"
SANA_VIDEO_VAE_TILING_OFF = "off"
SANA_VIDEO_VAE_TILING_ALWAYS = "always"
SANA_VIDEO_VAE_TILING_MODES = (
    SANA_VIDEO_VAE_TILING_AUTO,
    SANA_VIDEO_VAE_TILING_OFF,
    SANA_VIDEO_VAE_TILING_ALWAYS,
)


class SanaVideoRequest(BaseModel):
    prompt: str = ""
    negative_prompt: str = ""
    source_image_path: str | None = None
    pipeline: str = SANA_VIDEO_PIPELINE_T2V
    model_path: str = ""
    width: int = Field(default=832, ge=128, le=2048)
    height: int = Field(default=480, ge=128, le=2048)
    frames: int = Field(default=81, ge=1, le=257)
    fps: float = Field(default=16.0, ge=1.0, le=60.0)
    seed: int = -1
    steps: int = Field(default=50, ge=1, le=100)
    cfg_scale: float = Field(default=6.0, ge=0.0, le=20.0)
    motion_score: int = Field(default=30, ge=0, le=100)
    use_resolution_binning: bool = True
    max_sequence_length: int = Field(default=300, ge=16, le=1024)
    quantization: str = SANA_VIDEO_QUANTIZATION_AUTO
    vae_tiling: str = SANA_VIDEO_VAE_TILING_AUTO
    offload_text_encoder_after_encode: bool = True
    use_sage_attention: bool = True
    generate_audio: bool = False
    audio_prompt: str = ""
    audio_model_id: str = "mmaudio:small_16k"
    audio_steps: int = Field(default=25, ge=1, le=200)
    audio_cfg: float = Field(default=4.5, ge=0.1, le=10.0)

    @field_validator("pipeline")
    @classmethod
    def _validate_pipeline(cls, value: str) -> str:
        normalized = (value or SANA_VIDEO_PIPELINE_T2V).strip().lower().replace("-", "_")
        if normalized in {"t2v", "txt2video", "txt_to_video"}:
            normalized = SANA_VIDEO_PIPELINE_T2V
        elif normalized in {"i2v", "img2video", "image_to_video"}:
            normalized = SANA_VIDEO_PIPELINE_I2V
        if normalized not in SANA_VIDEO_PIPELINES:
            raise ValueError(f"pipeline must be one of {SANA_VIDEO_PIPELINES}, got {value!r}")
        return normalized

    @field_validator("quantization")
    @classmethod
    def _validate_quantization(cls, value: str) -> str:
        normalized = (value or SANA_VIDEO_QUANTIZATION_AUTO).strip().lower().replace("-", "_")
        aliases = {
            "none": SANA_VIDEO_QUANTIZATION_BF16,
            "default": SANA_VIDEO_QUANTIZATION_BF16,
            "fp8": SANA_VIDEO_QUANTIZATION_FP8,
            "float8": SANA_VIDEO_QUANTIZATION_FP8,
            "fp8_layerwise": SANA_VIDEO_QUANTIZATION_FP8,
            "float8_layerwise": SANA_VIDEO_QUANTIZATION_FP8,
            "int8": SANA_VIDEO_QUANTIZATION_BNB_INT8,
            "8bit": SANA_VIDEO_QUANTIZATION_BNB_INT8,
            "bnb_8bit": SANA_VIDEO_QUANTIZATION_BNB_INT8,
            "nf4": SANA_VIDEO_QUANTIZATION_BNB_NF4,
            "4bit": SANA_VIDEO_QUANTIZATION_BNB_NF4,
            "bnb_4bit": SANA_VIDEO_QUANTIZATION_BNB_NF4,
            "fp4": SANA_VIDEO_QUANTIZATION_BNB_FP4,
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in SANA_VIDEO_QUANTIZATIONS:
            raise ValueError(f"quantization must be one of {SANA_VIDEO_QUANTIZATIONS}, got {value!r}")
        return normalized

    @field_validator("vae_tiling")
    @classmethod
    def _validate_vae_tiling(cls, value: str) -> str:
        normalized = (value or SANA_VIDEO_VAE_TILING_AUTO).strip().lower().replace("-", "_")
        if normalized in {"true", "on", "yes"}:
            normalized = SANA_VIDEO_VAE_TILING_ALWAYS
        elif normalized in {"false", "no"}:
            normalized = SANA_VIDEO_VAE_TILING_OFF
        if normalized not in SANA_VIDEO_VAE_TILING_MODES:
            raise ValueError(f"vae_tiling must be one of {SANA_VIDEO_VAE_TILING_MODES}, got {value!r}")
        return normalized

    @field_validator("width", "height")
    @classmethod
    def _validate_size(cls, value: int) -> int:
        if int(value) % 32 != 0:
            raise ValueError("Sana video width and height must be divisible by 32")
        return int(value)

    @property
    def wants_image_to_video(self) -> bool:
        return self.pipeline == SANA_VIDEO_PIPELINE_I2V or bool(str(self.source_image_path or "").strip())


class SanaVideoResult(BaseModel):
    output_path: str
    message: str = "Sana video complete"
    frames: int = 0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    has_audio: bool = False
    audio_path: str = ""
    video_only_path: str = ""
    infotext: str = ""
    timings: dict[str, float] = Field(default_factory=dict)
    progress: list[dict[str, object]] = Field(default_factory=list)
    attention_backend: str = ""
    quantization: str = ""
    vae_tiling: str = ""
    receipt_path: str = ""

    @property
    def path(self) -> str:
        return self.output_path


class SanaVideoProgressEvent(BaseModel):
    stage: str
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    message: str = ""
    step: int = 0
    total: int = 0
    seconds: float = 0.0


def sana_video_model_folder_name(repo_id: str = SANA_VIDEO_MODEL_REPO_480P) -> str:
    return repo_id.split("/", 1)[-1]


def resolve_sana_video_path(raw: str | None, default: Path, root: Path) -> Path:
    text = str(raw or "").strip()
    path = Path(text) if text else default
    return path.resolve() if path.is_absolute() else (root / path).resolve()
