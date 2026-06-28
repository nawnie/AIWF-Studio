from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, field_validator

SANA_VIDEO_MODEL_REPO_480P = "Efficient-Large-Model/SANA-Video_2B_480p_diffusers"
SANA_VIDEO_MODEL_REPO_720P = "Efficient-Large-Model/SANA-Video_2B_720p_diffusers"
SANA_VIDEO_PIPELINE_T2V = "text_to_video"
SANA_VIDEO_PIPELINE_I2V = "image_to_video"
SANA_VIDEO_PIPELINES = (SANA_VIDEO_PIPELINE_T2V, SANA_VIDEO_PIPELINE_I2V)


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

    @property
    def path(self) -> str:
        return self.output_path


def sana_video_model_folder_name(repo_id: str = SANA_VIDEO_MODEL_REPO_480P) -> str:
    return repo_id.split("/", 1)[-1]


def resolve_sana_video_path(raw: str | None, default: Path, root: Path) -> Path:
    text = str(raw or "").strip()
    path = Path(text) if text else default
    return path.resolve() if path.is_absolute() else (root / path).resolve()
