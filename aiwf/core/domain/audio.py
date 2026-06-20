from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class AudioGenerationOptions(BaseModel):
    """Options for optional text-to-audio generation."""

    prompt: str = ""
    kind: str = "music"  # music | sfx | video_audio
    model_id: str = "facebook/musicgen-small"
    negative_prompt: str = ""
    duration_seconds: float = Field(default=8.0, ge=1.0, le=120.0)
    temperature: float = Field(default=1.0, ge=0.1, le=2.0)
    cfg_coef: float = Field(default=3.0, ge=0.1, le=10.0)
    top_k: int = Field(default=250, ge=0, le=1000)
    steps: int = Field(default=25, ge=1, le=200)
    seed: int = -1


class AudioGenerationResult(BaseModel):
    output_path: str
    prompt: str = ""
    model_id: str = ""
    kind: str = "music"
    duration_seconds: float = 0.0
    sample_rate: int = 0
    message: str = ""
    infotext: str = ""

    @property
    def path(self) -> str:
        return self.output_path


class AudioMuxResult(BaseModel):
    output_path: str
    audio_path: str
    video_path: str
    message: str = ""
    infotext: str = ""

    @classmethod
    def saved(cls, video_path: str | Path, audio_path: str | Path, output_path: str | Path) -> "AudioMuxResult":
        return cls(
            video_path=str(video_path),
            audio_path=str(audio_path),
            output_path=str(output_path),
            message=f"Muxed audio into video -> {output_path}",
            infotext=f"Audio mux: {Path(audio_path).name}",
        )
