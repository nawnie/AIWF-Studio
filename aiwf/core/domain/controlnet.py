from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class ControlNetUnit(BaseModel):
    """One ControlNet conditioning lane from UI through backend execution.

    Image and mask are paths/identifiers here, not decoded image tensors. That
    keeps preprocessing/runtime ownership outside the core domain model.
    """

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
    """Discovered local ControlNet asset exposed to selectors and catalogs."""

    id: str
    title: str
    path: str
    kind: str = "controlnet"

    @classmethod
    def from_path(cls, path: Path) -> ControlNetModelInfo:
        name = path.name if path.is_dir() else path.stem
        return cls(id=name, title=name, path=str(path))
