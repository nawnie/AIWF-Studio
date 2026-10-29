from __future__ import annotations

from typing import Literal

from PIL import Image
from pydantic import BaseModel, Field


ImagePipelineStatusValue = Literal["ready", "maturing", "blocked"]


class ImagePipelineStatus(BaseModel):
    route: str
    label: str
    score: float = Field(ge=0.0, le=10.0)
    target_score: float = Field(default=8.0, ge=0.0, le=10.0)
    status: ImagePipelineStatusValue = "maturing"
    supported: bool = True
    benchmark_kind: str | None = None
    benchmark_required: bool = True
    notes: list[str] = Field(default_factory=list)


class ImageMaturityMatrix(BaseModel):
    target_score: float = Field(default=8.0, ge=0.0, le=10.0)
    reference: str = "AUTOMATIC1111 core image workflow maturity"
    routes: list[ImagePipelineStatus] = Field(default_factory=list)

    @property
    def below_target(self) -> list[ImagePipelineStatus]:
        return [route for route in self.routes if route.supported and route.score < route.target_score]


class ImageLabRunResult(BaseModel):
    images: list[Image.Image] = Field(default_factory=list)
    seeds: list[int] = Field(default_factory=list)
    infotexts: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    grid: Image.Image | None = None

    model_config = {"arbitrary_types_allowed": True}
