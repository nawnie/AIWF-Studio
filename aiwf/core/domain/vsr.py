from __future__ import annotations

from pydantic import BaseModel, Field


class VsrOptions(BaseModel):
    """Options for NVIDIA RTX Video Super Resolution via Video Effects SDK."""

    effect: str = "SuperRes"
    scale: float = Field(default=2.0, ge=1.0, le=4.0)
    mode: int = Field(default=3, ge=0, le=19)
    strength: float = Field(default=0.6, ge=0.0, le=1.0)
    codec: str = "avc1"


class VideoFxDenoiseOptions(BaseModel):
    """Options for NVIDIA Video Effects SDK dedicated denoise."""

    strength: float = Field(default=0.8, ge=0.0, le=1.0)
    codec: str = "avc1"


class VideoFxAigsOptions(BaseModel):
    """Options for NVIDIA AI Green Screen / background processing."""

    mode: int = Field(default=0, ge=0, le=1)
    comp_mode: int = Field(default=6, ge=0, le=6)
    blur_strength: float = Field(default=0.45, ge=0.0, le=1.0)
    background_file: str | None = None
    cuda_graph: bool = False
    codec: str = "avc1"


class VideoFxRelightOptions(BaseModel):
    """Options for NVIDIA Video Effects SDK relighting."""

    hdr_file: str | None = None
    background_mode: int = Field(default=0, ge=0, le=4)
    background: str | None = None
    pan_degrees: float = -90.0
    vfov_degrees: float = Field(default=60.0, ge=1.0, le=179.0)
    autorotate: bool = False
    rotation_rate: float = 20.0
    codec: str = "avc1"


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
