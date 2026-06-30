from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class EnhanceModelKind(str, Enum):
    UPSCALER = "upscaler"
    RESTORER = "restorer"


class EnhanceModel(BaseModel):
    id: str
    title: str
    filename: str
    path: str
    kind: EnhanceModelKind
    architecture: str
    scale: int = Field(default=4, ge=1, le=8)
    download_url: str | None = None


class UpscaleOptions(BaseModel):
    model_id: str
    scale: float = Field(default=4.0, ge=1.0, le=8.0)
    tile_size: int = Field(default=256, ge=0, le=2048)
    tile_overlap: int = Field(default=32, ge=0, le=512)


class RestoreOptions(BaseModel):
    model_id: str
    visibility: float = Field(default=1.0, ge=0.0, le=1.0)
    codeformer_weight: float = Field(default=0.5, ge=0.0, le=1.0)


class EnhanceResult(BaseModel):
    image_path: str | None = None
    receipt_path: str | None = None
    infotext: str = ""
    message: str = ""
