from __future__ import annotations

from pydantic import BaseModel, Field


class VsrOptions(BaseModel):
    """Options for NVIDIA RTX Video Super Resolution via Video Effects SDK."""

    effect: str = "SuperRes"
    scale: float = Field(default=2.0, ge=1.0, le=4.0)
    mode: int = Field(default=1, ge=0, le=1)
    strength: float = Field(default=0.6, ge=0.0, le=1.0)
    codec: str = "H264"


class VsrResult(BaseModel):
    output_path: str
    input_width: int = 0
    input_height: int = 0
    output_width: int = 0
    output_height: int = 0
    fps: float = 0.0
    frame_count: int = 0
    message: str = ""
    infotext: str = ""
