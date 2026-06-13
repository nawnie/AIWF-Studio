from __future__ import annotations

from itertools import product
from math import ceil, sqrt
from typing import Any

from PIL import Image, ImageDraw
from pydantic import BaseModel, Field, field_validator

from aiwf.core.domain.generation import GenerationMode, GenerationRequest
from aiwf.services.generation import GenerationService

PLOT_FIELDS = {
    "seed",
    "steps",
    "cfg_scale",
    "sampler",
    "width",
    "height",
    "denoising_strength",
    "checkpoint_id",
    "vae_id",
    "clip_skip",
}


class PlotAxis(BaseModel):
    field: str
    values: list[Any] = Field(default_factory=list)

    @field_validator("field")
    @classmethod
    def known_field(cls, value: str) -> str:
        if value not in PLOT_FIELDS:
            raise ValueError(f"Unsupported plot field: {value}")
        return value


class PlotRequest(BaseModel):
    base: GenerationRequest
    axes: list[PlotAxis] = Field(default_factory=list)


class PlotResult(BaseModel):
    labels: list[str]
    images: list[Image.Image]
    grid: Image.Image | None = None
    infotexts: list[str] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


class PlotService:
    def __init__(self, generation: GenerationService) -> None:
        self.generation = generation

    def run(
        self,
        request: PlotRequest,
        *,
        init_images: list[Image.Image] | None = None,
        mask_images: list[Image.Image] | None = None,
    ) -> PlotResult:
        axes = [axis for axis in request.axes if axis.values]
        combinations = list(product(*(axis.values for axis in axes))) if axes else [()]
        images: list[Image.Image] = []
        labels: list[str] = []
        infotexts: list[str] = []

        for values in combinations:
            updates = dict(zip((axis.field for axis in axes), values))
            generation_request = request.base.model_copy(update=updates)
            job = self.generation.submit(
                generation_request,
                init_images=init_images,
                mask_images=mask_images,
            )
            if job.result is None or not job.result.images:
                raise RuntimeError(job.error or "plot generation failed")
            images.append(job.result.images[0])
            labels.append(", ".join(f"{field}={value}" for field, value in updates.items()) or "base")
            infotexts.append(job.result.infotexts[0] if job.result.infotexts else "")

        return PlotResult(labels=labels, images=images, grid=_make_grid(images, labels), infotexts=infotexts)


def _make_grid(images: list[Image.Image], labels: list[str]) -> Image.Image | None:
    if not images:
        return None
    thumb_w = max(image.width for image in images)
    thumb_h = max(image.height for image in images)
    label_h = 24
    cols = ceil(sqrt(len(images)))
    rows = ceil(len(images) / cols)
    grid = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(grid)

    for index, image in enumerate(images):
        row, col = divmod(index, cols)
        x = col * thumb_w
        y = row * (thumb_h + label_h)
        grid.paste(image.convert("RGB"), (x, y))
        draw.text((x + 4, y + thumb_h + 4), labels[index][:80], fill=(0, 0, 0))
    return grid


def plot_request_from_payload(payload: dict[str, Any], mode: GenerationMode = GenerationMode.TXT2IMG) -> PlotRequest:
    axes = [PlotAxis.model_validate(axis) for axis in payload.get("axes", [])]
    base_payload = dict(payload.get("base", {}))
    base_payload.setdefault("mode", mode)
    return PlotRequest(base=GenerationRequest.model_validate(base_payload), axes=axes)
