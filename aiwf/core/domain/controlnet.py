from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class ControlNetUnit(BaseModel):
    enabled: bool = True
    model: str | None = None
    module: str = "none"
    weight: float = Field(default=1.0, ge=0.0, le=2.0)
    image: str | None = None
    mask: str | None = None
    resize_mode: str = "resize"
    processor_res: int = Field(default=512, ge=64, le=4096)
    threshold_a: float = 64.0
    threshold_b: float = 64.0
    guidance_start: float = Field(default=0.0, ge=0.0, le=1.0)
    guidance_end: float = Field(default=1.0, ge=0.0, le=1.0)
    control_mode: str = "balanced"


class ControlNetModelInfo(BaseModel):
    id: str
    title: str
    path: str
    kind: str = "controlnet"

    @classmethod
    def from_path(cls, path: Path) -> ControlNetModelInfo:
        return cls(id=path.stem, title=path.stem, path=str(path))
