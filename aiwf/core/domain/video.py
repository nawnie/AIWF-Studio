from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class VideoInfo(BaseModel):
    """Lightweight description of a video file (no frame data held)."""

    path: str
    frame_count: int = Field(default=0, ge=0)
    fps: float = Field(default=0.0, ge=0.0)
    width: int = Field(default=0, ge=0)
    height: int = Field(default=0, ge=0)
    has_audio: bool = False

    @property
    def duration_seconds(self) -> float:
        return (self.frame_count / self.fps) if self.fps else 0.0


class VideoProcessResult(BaseModel):
    """Outcome of a frame-by-frame video transform."""

    output_path: str
    frames_processed: int = Field(default=0, ge=0)
    fps: float = Field(default=0.0, ge=0.0)
    width: int = Field(default=0, ge=0)
    height: int = Field(default=0, ge=0)
    audio_muxed: bool = False
    message: str = ""
    infotext: str = ""

    @classmethod
    def saved(
        cls,
        output_path: str | Path,
        frame_count: int,
        fps: float,
        width: int,
        height: int,
        infotext: str = "",
    ) -> VideoProcessResult:
        return cls(
            output_path=str(output_path),
            frames_processed=frame_count,
            fps=fps,
            width=width,
            height=height,
            message=f"{frame_count} frame(s) at {width}x{height}, {fps:.0f} fps",
            infotext=infotext,
        )

    @property
    def path(self) -> str:
        return self.output_path

    @property
    def frame_count(self) -> int:
        return self.frames_processed


