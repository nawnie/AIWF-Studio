from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

from aiwf.core.domain.enhance import RestoreOptions, UpscaleOptions
from aiwf.core.domain.generation import GenerationMode, GenerationRequest
from aiwf.core.domain.image_workflow import (
    IMAGE_STAGE_LABELS,
    IMAGE_WORKFLOW_ORDER,
    ImageWorkflowPlan,
    ImageWorkflowResult,
    ImageWorkflowSettings,
)
from aiwf.core.domain.segment import SegmentRequest
from aiwf.core.domain.segment_presets import resolve_segment_text_prompt


IMAGE_WORKFLOW_PRESETS: dict[str, dict[str, object]] = {
    "portrait_cleanup": {
        "stages": ["restore", "tone", "export"],
        "restore_visibility": 0.82,
        "contrast": 1.03,
        "saturation": 1.02,
    },
    "old_photo": {
        "stages": ["denoise", "restore", "tone", "upscale", "export"],
        "denoise_radius": 1,
        "denoise_strength": 0.42,
        "restore_visibility": 0.72,
        "contrast": 1.06,
        "saturation": 0.96,
        "upscale_factor": 2.0,
    },
    "object_replace": {
        "stages": ["auto_mask", "inpaint", "tone", "export"],
        "mask_preset": "person",
        "denoising_strength": 0.68,
    },
    "web_ready": {
        "stages": ["tone", "resize", "export"],
        "contrast": 1.03,
        "saturation": 1.04,
        "resize_width": 1600,
        "resize_height": 0,
        "export_format": "webp",
        "export_quality": 92,
    },
    "custom": {"stages": ["tone", "export"]},
}


def preset_image_settings(name: str) -> ImageWorkflowSettings:
    preset = name if name in IMAGE_WORKFLOW_PRESETS else "custom"
    payload = dict(IMAGE_WORKFLOW_PRESETS[preset])
    # Object replacement needs a user prompt, so keep preset construction valid.
    if "inpaint" in payload.get("stages", []):
        payload.setdefault("inpaint_prompt", "replace the selected area naturally")
    return ImageWorkflowSettings(preset=preset, **payload)


def resolve_image_plan(settings: ImageWorkflowSettings, *, has_uploaded_mask: bool = False) -> ImageWorkflowPlan:
    selected = set(settings.stages)
    stages = [stage for stage in IMAGE_WORKFLOW_ORDER if stage in selected]
    if "export" not in stages:
        stages.append("export")
    warnings: list[str] = []
    if "inpaint" in stages and "auto_mask" not in stages and not has_uploaded_mask:
        warnings.append("Inpaint needs an uploaded mask or Auto mask enabled.")
    if "restore" in stages and not settings.restore_model_id:
        warnings.append("Restore is selected, but no restoration model is selected; the stage will be skipped.")
    if "upscale" in stages and not settings.upscaler_model_id:
        warnings.append("Upscale is selected, but no upscaler is selected; the stage will be skipped.")
    if "resize" in stages and settings.resize_width <= 0 and settings.resize_height <= 0:
        warnings.append("Final resize has no target dimension; the source size will be kept.")
    return ImageWorkflowPlan(stages=stages, labels=[IMAGE_STAGE_LABELS[item] for item in stages], warnings=warnings)


class ImageWorkflowService:
    def __init__(self, ctx) -> None:
        self.ctx = ctx
        self.output_root = Path(ctx.flags.resolved_output_dir()).resolve() / "image-lab"

    def build_plan(self, settings: ImageWorkflowSettings, *, has_uploaded_mask: bool = False) -> ImageWorkflowPlan:
        return resolve_image_plan(settings, has_uploaded_mask=has_uploaded_mask)

    @staticmethod
    def _generation_size(image: Image.Image) -> tuple[int, int]:
        width = max(64, min(2048, int(round(image.width / 8) * 8)))
        height = max(64, min(2048, int(round(image.height / 8) * 8)))
        return width, height

    @staticmethod
    def _apply_denoise(image: Image.Image, radius: int, strength: float) -> Image.Image:
        size = max(3, radius * 2 + 1)
        filtered = image.filter(ImageFilter.MedianFilter(size=size))
        return Image.blend(image, filtered, float(strength))

    @staticmethod
    def _apply_tone(image: Image.Image, settings: ImageWorkflowSettings) -> Image.Image:
        working = ImageEnhance.Brightness(image).enhance(settings.brightness)
        working = ImageEnhance.Contrast(working).enhance(settings.contrast)
        working = ImageEnhance.Color(working).enhance(settings.saturation)
        return ImageEnhance.Sharpness(working).enhance(settings.sharpness)

    @staticmethod
    def _apply_resize(image: Image.Image, settings: ImageWorkflowSettings) -> Image.Image:
        width = int(settings.resize_width or 0)
        height = int(settings.resize_height or 0)
        if width <= 0 and height <= 0:
            return image
        if settings.keep_aspect:
            if width <= 0:
                width = max(1, round(image.width * height / image.height))
            elif height <= 0:
                height = max(1, round(image.height * width / image.width))
            else:
                ratio = min(width / image.width, height / image.height)
                width = max(1, round(image.width * ratio))
                height = max(1, round(image.height * ratio))
        else:
            width = width or image.width
            height = height or image.height
        return image.resize((width, height), Image.Resampling.LANCZOS)

    @staticmethod
    def _atomic_save(image: Image.Image, path: Path, *, quality: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=suffix, dir=str(path.parent))
        os.close(fd)
        temp = Path(temp_name)
        try:
            if suffix in {".jpg", ".jpeg"}:
                image.convert("RGB").save(temp, quality=quality, optimize=True)
            elif suffix == ".webp":
                image.convert("RGB").save(temp, quality=quality, method=6)
            else:
                image.save(temp, optimize=True)
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)

    def process(
        self,
        source: Image.Image,
        settings: ImageWorkflowSettings,
        *,
        uploaded_mask: Image.Image | None = None,
    ) -> ImageWorkflowResult:
        if source is None:
            raise ValueError("Upload a source image first.")
        plan = self.build_plan(settings, has_uploaded_mask=uploaded_mask is not None)
        if any("needs an uploaded mask" in item for item in plan.warnings):
            raise ValueError(plan.warnings[0])

        started = time.perf_counter()
        working = source.convert("RGB")
        mask = uploaded_mask.convert("L").resize(working.size) if uploaded_mask is not None else None
        mask_preview = None
        stage_log: list[str] = []

        for stage in plan.stages:
            if stage == "auto_mask":
                prompt = resolve_segment_text_prompt(settings.mask_preset, settings.mask_custom_prompt)
                request = SegmentRequest(
                    text_prompt=prompt,
                    box_threshold=settings.mask_threshold,
                    mask_index=settings.mask_index,
                    dilation=settings.mask_dilation,
                    mask_blur=settings.mask_blur,
                    feather=settings.mask_feather,
                )
                mask, mask_preview, _candidates, status = self.ctx.segment.segment(
                    working, request, model_id=settings.mask_model_id
                )
                stage_log.append(f"Auto mask: {status}")
            elif stage == "inpaint":
                if mask is None:
                    raise ValueError("Inpaint reached the execution graph without a mask.")
                width, height = self._generation_size(working)
                request = GenerationRequest(
                    mode=GenerationMode.INPAINT,
                    prompt=settings.inpaint_prompt,
                    negative_prompt=settings.inpaint_negative_prompt,
                    checkpoint_id=settings.checkpoint_id,
                    sampler=settings.sampler,
                    steps=settings.steps,
                    cfg_scale=settings.cfg_scale,
                    width=width,
                    height=height,
                    seed=settings.seed,
                    denoising_strength=settings.denoising_strength,
                    mask_blur=0,
                    save_images=False,
                )
                record = self.ctx.generation.submit(request, init_images=[working], mask_images=[mask])
                if record.result is None or not record.result.images:
                    raise RuntimeError(record.error or "Inpaint did not return an image.")
                working = record.result.images[0].convert("RGB")
                if mask.size != working.size:
                    mask = mask.resize(working.size, Image.Resampling.LANCZOS)
                stage_log.append("Inpaint / repair completed")
            elif stage == "restore":
                if settings.restore_model_id:
                    working = self.ctx.enhance.restore(
                        working,
                        RestoreOptions(
                            model_id=settings.restore_model_id,
                            visibility=settings.restore_visibility,
                            codeformer_weight=settings.codeformer_weight,
                        ),
                    )
                    stage_log.append(f"Restore: {settings.restore_model_id}")
                else:
                    stage_log.append("Restore skipped: no model selected")
            elif stage == "denoise":
                working = self._apply_denoise(working, settings.denoise_radius, settings.denoise_strength)
                stage_log.append(
                    f"Denoise: radius={settings.denoise_radius}, strength={settings.denoise_strength:.2f}"
                )
            elif stage == "tone":
                working = self._apply_tone(working, settings)
                stage_log.append("Tone and color adjusted")
            elif stage == "upscale":
                if settings.upscaler_model_id:
                    working = self.ctx.enhance.upscale(
                        working,
                        UpscaleOptions(
                            model_id=settings.upscaler_model_id,
                            scale=settings.upscale_factor,
                            tile_size=settings.tile_size,
                            tile_overlap=settings.tile_overlap,
                        ),
                    )
                    stage_log.append(f"Upscale: {settings.upscaler_model_id} at {settings.upscale_factor:g}×")
                else:
                    stage_log.append("Upscale skipped: no model selected")
            elif stage == "resize":
                before = working.size
                working = self._apply_resize(working, settings)
                stage_log.append(f"Resize: {before[0]}×{before[1]} → {working.width}×{working.height}")

        job_id = f"ilab_{uuid.uuid4().hex[:12]}"
        job_dir = self.output_root / datetime.now().strftime("%Y%m%d") / job_id
        extension = "jpg" if settings.export_format == "jpg" else settings.export_format
        output_path = job_dir / f"image_workflow.{extension}"
        manifest_path = job_dir / "job.json"
        self._atomic_save(working, output_path, quality=settings.export_quality)
        if mask is not None:
            self._atomic_save(mask.convert("L"), job_dir / "mask.png", quality=100)
        elapsed = time.perf_counter() - started
        manifest = {
            "schema": 1,
            "job_id": job_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "resolved_order": plan.stages,
            "settings": settings.model_dump(mode="json"),
            "warnings": plan.warnings,
            "stage_log": stage_log,
            "output_path": str(output_path),
            "elapsed_seconds": elapsed,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return ImageWorkflowResult(
            image=working,
            output_path=str(output_path),
            manifest_path=str(manifest_path),
            mask=mask,
            mask_preview=mask_preview,
            message=f"Completed {len(plan.stages)} stage(s) in {elapsed:.2f}s.",
            stage_log=stage_log,
        )
