from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from PIL import Image

from aiwf.core.config.settings import UserSettings
from aiwf.core.domain.enhance import RestoreOptions, UpscaleOptions
from aiwf.core.domain.photo_restore import PhotoRestoreOptions
from aiwf.core.domain.generation import GenerationMode, GenerationRequest
from aiwf.core.domain.workflow import (
    WorkflowDefinition,
    WorkflowRunResult,
    WorkflowStep,
    WorkflowStepResult,
    WorkflowStepType,
)
from aiwf.services.enhance import EnhanceService
from aiwf.services.generation import GenerationService
from aiwf.services.segment import SegmentService

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]

_GENERATION_FIELDS = set(GenerationRequest.model_fields.keys()) - {"mode"}


def _generation_params(params: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in params.items() if key in _GENERATION_FIELDS}


def validate_workflow(workflow: WorkflowDefinition, *, has_seed_image: bool) -> None:
    if not workflow.steps:
        raise ValueError("Workflow has no steps.")

    first = workflow.steps[0].type
    if first not in {WorkflowStepType.TXT2IMG} and not has_seed_image:
        raise ValueError(
            f"First step is '{first.value}' but no seed image was provided. "
            "Upload an image or start with txt2img."
        )

    for index, step in enumerate(workflow.steps):
        if index == 0:
            continue
        if step.type == WorkflowStepType.TXT2IMG:
            raise ValueError(f"Step '{step.id}' is txt2img but is not the first step.")
        if step.type in {
            WorkflowStepType.IMG2IMG,
            WorkflowStepType.INPAINT,
            WorkflowStepType.SEGMENT,
            WorkflowStepType.UPSCALE,
            WorkflowStepType.RESTORE,
            WorkflowStepType.ENHANCE,
            WorkflowStepType.PHOTO_RESTORE,
        } and index > 0:
            continue


class _ChainState:
    __slots__ = ("image", "mask")

    def __init__(self, image: Image.Image | None = None, mask: Image.Image | None = None) -> None:
        self.image = image
        self.mask = mask


class WorkflowExecutor:
    def __init__(
        self,
        generation: GenerationService,
        enhance: EnhanceService,
        segment: SegmentService,
        settings: UserSettings,
    ) -> None:
        self.generation = generation
        self.enhance = enhance
        self.segment = segment
        self.settings = settings

    def run(
        self,
        workflow: WorkflowDefinition,
        *,
        seed_image: Image.Image | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> tuple[WorkflowRunResult, list[Image.Image]]:
        validate_workflow(workflow, has_seed_image=seed_image is not None)

        total = len(workflow.steps)
        state = _ChainState(image=seed_image)
        step_images: list[Image.Image] = []
        step_results: list[WorkflowStepResult] = []
        run = WorkflowRunResult(workflow_name=workflow.name)

        for index, step in enumerate(workflow.steps):
            if on_progress:
                on_progress(index + 1, total, f"Running {step.label or step.type.value}…")

            image, result = self._run_step(step, state)
            if image is not None:
                state.image = image
                step_images.append(image)
            elif state.image is not None:
                step_images.append(state.image)

            if workflow.save_intermediate and self.settings.save_images and image is not None:
                subdir = self.settings.workflow_output_subdir
                artifact = self.enhance.store.save(image, result.infotext, subdir)
                result.image_path = artifact.path

            step_results.append(result)

        run.steps = step_results
        if step_results:
            run.final_image_path = step_results[-1].image_path
            run.summary = " → ".join(
                step.label or step.step_type.value for step in step_results
            )
        return run, step_images

    def _run_step(
        self,
        step: WorkflowStep,
        state: _ChainState,
    ) -> tuple[Image.Image | None, WorkflowStepResult]:
        label = step.label or step.type.value
        params = dict(step.params)
        image = state.image

        if step.type == WorkflowStepType.TXT2IMG:
            request = GenerationRequest(
                mode=GenerationMode.TXT2IMG,
                save_images=False,
                **_generation_params(params),
            )
            job = self.generation.submit(request)
            if job.result is None or not job.result.images:
                raise RuntimeError(job.error or "txt2img failed")
            seed = job.result.seeds[0] if job.result.seeds else None
            infotext = job.result.infotexts[0] if job.result.infotexts else ""
            return job.result.images[0], WorkflowStepResult(
                step_id=step.id,
                step_type=step.type,
                label=label,
                infotext=infotext,
                message="txt2img complete",
                seed=seed,
            )

        if image is None:
            raise ValueError(f"Step '{step.id}' ({step.type.value}) requires an image from a previous step.")

        if step.type == WorkflowStepType.IMG2IMG:
            request = GenerationRequest(
                mode=GenerationMode.IMG2IMG,
                save_images=False,
                **_generation_params(params),
            )
            job = self.generation.submit(request, init_images=[image])
            if job.result is None or not job.result.images:
                raise RuntimeError(job.error or "img2img failed")
            seed = job.result.seeds[0] if job.result.seeds else None
            infotext = job.result.infotexts[0] if job.result.infotexts else ""
            return job.result.images[0], WorkflowStepResult(
                step_id=step.id,
                step_type=step.type,
                label=label,
                infotext=infotext,
                message="img2img complete",
                seed=seed,
            )

        if step.type == WorkflowStepType.SEGMENT:
            if image is None:
                raise ValueError(f"Step '{step.id}' (segment) requires an image from a previous step.")
            mask, status = self.segment.segment_from_workflow_params(image, params)
            state.mask = mask
            return None, WorkflowStepResult(
                step_id=step.id,
                step_type=step.type,
                label=label,
                infotext=status,
                message="segment complete",
            )

        if step.type == WorkflowStepType.INPAINT:
            mask = state.mask
            if mask is None and params.get("text_prompt"):
                mask, seg_status = self.segment.segment_from_workflow_params(image, params)
                state.mask = mask
                params = {key: value for key, value in params.items() if key not in {
                    "text_prompt", "box_threshold", "points", "box", "mask_index", "dilation", "model_id",
                }}
            if mask is None:
                raise ValueError(
                    f"Step '{step.id}' (inpaint) needs a mask — add a prior 'segment' step or "
                    "include 'text_prompt' in the inpaint step params."
                )
            request = GenerationRequest(
                mode=GenerationMode.INPAINT,
                save_images=False,
                **_generation_params(params),
            )
            job = self.generation.submit(request, init_images=[image], mask_images=[mask])
            if job.result is None or not job.result.images:
                raise RuntimeError(job.error or "inpaint failed")
            seed = job.result.seeds[0] if job.result.seeds else None
            infotext = job.result.infotexts[0] if job.result.infotexts else ""
            return job.result.images[0], WorkflowStepResult(
                step_id=step.id,
                step_type=step.type,
                label=label,
                infotext=infotext,
                message="inpaint complete",
                seed=seed,
            )

        if step.type == WorkflowStepType.UPSCALE:
            options = UpscaleOptions(
                model_id=params.get("model_id", "realesrgan-x4plus"),
                scale=float(params.get("scale", 4)),
                tile_size=int(params.get("tile_size", self.settings.upscale_tile_size)),
                tile_overlap=int(params.get("tile_overlap", self.settings.upscale_tile_overlap)),
            )
            result_image = self.enhance.upscale(image, options)
            return result_image, WorkflowStepResult(
                step_id=step.id,
                step_type=step.type,
                label=label,
                infotext=f"Upscale {options.model_id} {options.scale}x",
                message="upscale complete",
            )

        if step.type == WorkflowStepType.RESTORE:
            options = RestoreOptions(
                model_id=params.get("model_id", "gfpgan-v1.4"),
                visibility=float(params.get("visibility", 1.0)),
                codeformer_weight=float(params.get("codeformer_weight", 0.5)),
            )
            result_image = self.enhance.restore(image, options)
            return result_image, WorkflowStepResult(
                step_id=step.id,
                step_type=step.type,
                label=label,
                infotext=f"Restore {options.model_id}",
                message="restore complete",
            )

        if step.type == WorkflowStepType.PHOTO_RESTORE:
            restore_opts = None
            if params.get("face_restore", True):
                restore_opts = RestoreOptions(
                    model_id=params.get("restore_model_id", "gfpgan-v1.4"),
                    visibility=float(params.get("visibility", 1.0)),
                    codeformer_weight=float(params.get("codeformer_weight", 0.5)),
                )
            upscale_opts = None
            if params.get("upscale", False):
                upscale_opts = UpscaleOptions(
                    model_id=params.get("upscale_model_id", "realesrgan-x4plus"),
                    scale=float(params.get("scale", 2)),
                    tile_size=int(params.get("tile_size", self.settings.upscale_tile_size)),
                    tile_overlap=int(params.get("tile_overlap", self.settings.upscale_tile_overlap)),
                )
            options = PhotoRestoreOptions(
                scratch_detection=bool(params.get("scratch_detection", True)),
                scratch_inpaint=bool(params.get("scratch_inpaint", True)),
                scratch_sensitivity=float(params.get("scratch_sensitivity", 0.45)),
                global_restore=bool(params.get("global_restore", True)),
                denoise_strength=float(params.get("denoise_strength", 0.65)),
                color_boost=float(params.get("color_boost", 0.55)),
                face_restore=bool(params.get("face_restore", True)),
                restore=restore_opts,
                upscale=upscale_opts,
            )
            result_image, infotext = self.enhance.run_photo_restore(image, options)
            return result_image, WorkflowStepResult(
                step_id=step.id,
                step_type=step.type,
                label=label,
                infotext=infotext,
                message="photo restore complete",
            )

        if step.type == WorkflowStepType.ENHANCE:
            restore_opts = None
            upscale_opts = None
            if params.get("restore", True):
                restore_opts = RestoreOptions(
                    model_id=params.get("restore_model_id", "gfpgan-v1.4"),
                    visibility=float(params.get("visibility", 1.0)),
                    codeformer_weight=float(params.get("codeformer_weight", 0.5)),
                )
            if params.get("upscale", True):
                upscale_opts = UpscaleOptions(
                    model_id=params.get("upscale_model_id", "realesrgan-x4plus"),
                    scale=float(params.get("scale", 2)),
                    tile_size=int(params.get("tile_size", self.settings.upscale_tile_size)),
                    tile_overlap=int(params.get("tile_overlap", self.settings.upscale_tile_overlap)),
                )
            result_image, infotext = self.enhance.run_pipeline(
                image,
                restore=restore_opts,
                upscale=upscale_opts,
                restore_first=bool(params.get("restore_first", True)),
            )
            return result_image, WorkflowStepResult(
                step_id=step.id,
                step_type=step.type,
                label=label,
                infotext=infotext,
                message="enhance pipeline complete",
            )

        raise ValueError(f"Unknown workflow step type: {step.type}")
