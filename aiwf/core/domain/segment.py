from __future__ import annotations

from pydantic import BaseModel, Field


class SegmentPoint(BaseModel):
    x: int
    y: int
    label: int = Field(default=1, ge=0, le=1)  # 1 = include, 0 = exclude


class SegmentBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class SegmentRequest(BaseModel):
    text_prompt: str = ""
    box_threshold: float = Field(default=0.3, ge=0.05, le=0.95)
    points: list[SegmentPoint] = Field(default_factory=list)
    box: SegmentBox | None = None
    mask_index: int = Field(default=0, ge=0)
    dilation: int = Field(default=0, ge=0, le=128)
    multimask_output: bool = True


class SamModelInfo(BaseModel):
    id: str
    title: str
    filename: str
    path: str
    architecture: str  # vit_h, vit_l, vit_b