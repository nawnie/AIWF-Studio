from __future__ import annotations

from pydantic import BaseModel, Field


class RifeOptions(BaseModel):
    """Options for RIFE optical-flow frame interpolation."""

    ckpt_name: str = "rife47.pth"
    multiplier: int = Field(default=2, ge=2, le=8)
    scale_factor: float = Field(default=1.0, ge=0.25, le=4.0)
    fast_mode: bool = True
    ensemble: bool = True
    clear_cache_every_n_frames: int = Field(default=10, ge=1, le=1000)
    max_input_frames: int | None = Field(default=None, ge=2)
    target_fps: float | None = Field(default=None, ge=1.0, le=240.0)


class RifeResult(BaseModel):
    output_path: str
    input_frames: int = 0
    output_frames: int = 0
    input_fps: float = 0.0
    output_fps: float = 0.0
    width: int = 0
    height: int = 0
    message: str = ""
    infotext: str = ""
