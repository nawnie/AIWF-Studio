from __future__ import annotations

import random
from collections.abc import Iterable
from typing import Any

from PIL import Image

from aiwf.core.domain.generation import GenerationMode, GenerationRequest
from aiwf.core.domain.image_lab import ImageLabRunResult, ImageMaturityMatrix, ImagePipelineStatus
from aiwf.services.plot import PlotAxis, _make_grid


AXIS_ALIASES = {
    "cfg": "cfg_scale",
    "cfg scale": "cfg_scale",
    "scale": "cfg_scale",
    "sampler_name": "sampler",
    "checkpoint": "checkpoint_id",
    "model": "checkpoint_id",
    "vae": "vae_id",
    "denoise": "denoising_strength",
    "denoising": "denoising_strength",
}


def image_maturity_matrix() -> ImageMaturityMatrix:
    return ImageMaturityMatrix(
        routes=[
            ImagePipelineStatus(
                route="txt2img",
                label="Text to image",
                score=8.3,
                status="ready",
                benchmark_kind="txt2img",
                notes=["Native prompt, sampler, seed, batch, LoRA, VAE, CLIP skip, and PNG metadata path."],
            ),
            ImagePipelineStatus(
                route="img2img",
                label="Image to image",
                score=8.0,
                status="ready",
                benchmark_kind="img2img",
                notes=["Native denoise, batch file loop, variation planning, and loopback chaining."],
            ),
            ImagePipelineStatus(
                route="inpaint",
                label="Inpaint / masked repair",
                score=8.0,
                status="ready",
                benchmark_kind="inpaint",
                notes=["Native mask blur, masked-only mode, padding, fill mode, seam erosion, and batch mask pairing."],
            ),
            ImagePipelineStatus(
                route="hires-refiner",
                label="Hires fix / SDXL refiner",
                score=8.0,
                status="ready",
                benchmark_kind="hires",
                notes=["Native hires second pass and SDXL refiner controls; ControlNet combinations remain intentionally gated."],
            ),
            ImagePipelineStatus(
                route="controlnet",
                label="ControlNet conditioning",
                score=8.1,
                status="ready",
                benchmark_kind="controlnet",
                notes=["Native multi-unit contract, local model discovery, preprocessors, API aliases, and benchmark route."],
            ),
            ImagePipelineStatus(
                route="xyz-plot",
                label="XYZ plots",
                score=8.0,
                status="ready",
                benchmark_kind="txt2img",
                notes=["Native grid generation for seed, steps, cfg, sampler, size, checkpoint, VAE, CLIP skip, and denoise axes."],
            ),
            ImagePipelineStatus(
                route="extras",
                label="Extras / enhance",
                score=8.0,
                status="ready",
                benchmark_kind="receipt",
                notes=["Upscale and restoration save image-sidecar receipts with route, model, input, and output metadata."],
            ),
            ImagePipelineStatus(
                route="segment-inpaint",
                label="Segment to inpaint",
                score=8.0,
                status="ready",
                benchmark_kind="receipt",
                notes=["One-click object replacement runs Auto mask into Inpaint and writes a workflow receipt with mask and repair metadata."],
            ),
            ImagePipelineStatus(
                route="png-api-replay",
                label="PNG/API replay",
                score=8.0,
                status="ready",
                benchmark_kind="txt2img",
                notes=["PNG Info, A1111-compatible API aliases, native API, and generation metadata round trip are present."],
            ),
            ImagePipelineStatus(
                route="flux-txt2img",
                label="Flux text to image",
                score=8.0,
                status="ready",
                benchmark_kind="txt2img",
                notes=["Flux remains txt2img-only for release; bounded smoke plans cover Flux, Flux.2 Klein, and Z-Image routes while advanced modes stay gated."],
            ),
        ]
    )


def maturity_matrix_markdown(matrix: ImageMaturityMatrix | None = None) -> str:
    matrix = matrix or image_maturity_matrix()
    lines = [
        f"### Core image maturity target: {matrix.target_score:.1f}+",
        "",
        "| Route | Score | Status | Benchmark | Notes |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for route in matrix.routes:
        benchmark = route.benchmark_kind or "pending"
        notes = "<br>".join(route.notes)
        lines.append(f"| {route.label} | {route.score:.1f} | {route.status} | {benchmark} | {notes} |")
    return "\n".join(lines)


def normalize_axis_field(field: str | None) -> str:
    normalized = (field or "").strip().lower().replace("-", "_")
    normalized = AXIS_ALIASES.get(normalized, normalized)
    return normalized


def parse_axis_values(raw: str | None) -> list[Any]:
    if raw is None:
        return []
    chunks = [
        chunk.strip()
        for chunk in str(raw).replace("\n", ",").replace(";", ",").split(",")
        if chunk.strip()
    ]
    return [_parse_axis_value(chunk) for chunk in chunks]


def build_plot_axes(axis_specs: Iterable[tuple[str | None, str | None]]) -> list[PlotAxis]:
    axes: list[PlotAxis] = []
    for field, raw_values in axis_specs:
        normalized = normalize_axis_field(field)
        values = parse_axis_values(raw_values)
        if normalized and values:
            axes.append(PlotAxis(field=normalized, values=values))
    return axes


def variation_requests(
    base: GenerationRequest,
    *,
    count: int,
    seed_step: int = 1,
    denoise_step: float = 0.0,
) -> list[GenerationRequest]:
    count = max(1, int(count))
    requests: list[GenerationRequest] = []
    start_seed = base.seed if base.seed >= 0 else random.randint(0, 2**31 - 1)
    for index in range(count):
        seed = start_seed + (index * int(seed_step))
        denoise = min(1.0, max(0.0, base.denoising_strength + (index * float(denoise_step))))
        requests.append(base.model_copy(update={"seed": seed, "denoising_strength": denoise}))
    return requests


class ImageLabService:
    def __init__(self, generation: Any) -> None:
        self.generation = generation

    def run_batch(
        self,
        request: GenerationRequest,
        images: list[Image.Image],
        *,
        masks: list[Image.Image] | None = None,
    ) -> ImageLabRunResult:
        if not images:
            raise ValueError("At least one source image is required.")
        if request.mode == GenerationMode.INPAINT and not masks:
            raise ValueError("Batch inpaint requires at least one mask image.")
        if masks and len(masks) not in {1, len(images)}:
            raise ValueError("Mask count must be one shared mask or match the source image count.")

        result = ImageLabRunResult()
        for index, image in enumerate(images):
            paired_masks = None
            if masks:
                paired_masks = [masks[index if len(masks) > 1 else 0]]
            job = self.generation.submit(request, init_images=[image], mask_images=paired_masks)
            _append_job(result, job, f"{index + 1}/{len(images)}")
        result.grid = _make_grid(result.images, result.labels)
        return result

    def run_loopback(
        self,
        request: GenerationRequest,
        source: Image.Image,
        *,
        iterations: int,
        denoise_decay: float = 1.0,
        seed_mode: str = "increment",
    ) -> ImageLabRunResult:
        iterations = max(1, int(iterations))
        denoise_decay = max(0.0, min(1.0, float(denoise_decay)))
        seed_mode = (seed_mode or "increment").strip().lower()
        current = source
        start_seed = request.seed if request.seed >= 0 else random.randint(0, 2**31 - 1)
        result = ImageLabRunResult()

        for index in range(iterations):
            seed = start_seed
            if seed_mode == "increment":
                seed = start_seed + index
            elif seed_mode == "random":
                seed = random.randint(0, 2**31 - 1)
            denoise = max(0.01, min(1.0, request.denoising_strength * (denoise_decay**index)))
            iteration_request = request.model_copy(
                update={
                    "mode": GenerationMode.IMG2IMG,
                    "seed": seed,
                    "denoising_strength": denoise,
                }
            )
            job = self.generation.submit(iteration_request, init_images=[current])
            _append_job(result, job, f"loop {index + 1}, seed={seed}, denoise={denoise:.3f}")
            current = result.images[-1]
        result.grid = _make_grid(result.images, result.labels)
        return result


def _parse_axis_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        parsed_int = int(value)
    except ValueError:
        parsed_int = None
    if parsed_int is not None and str(parsed_int) == value:
        return parsed_int
    try:
        return float(value)
    except ValueError:
        return value


def _append_job(result: ImageLabRunResult, job: Any, label: str) -> None:
    if getattr(job, "result", None) is None or not job.result.images:
        raise RuntimeError(getattr(job, "error", None) or "image generation failed")
    result.images.extend(job.result.images)
    result.seeds.extend(job.result.seeds)
    result.infotexts.extend(job.result.infotexts)
    for offset, _image in enumerate(job.result.images):
        if len(job.result.images) == 1:
            result.labels.append(label)
        else:
            result.labels.append(f"{label}.{offset + 1}")
