from __future__ import annotations

import json

import gradio as gr

from aiwf.bootstrap import AppContext
from aiwf.core.domain.image_workflow import IMAGE_STAGE_LABELS, IMAGE_WORKFLOW_ORDER, ImageWorkflowSettings
from aiwf.core.domain.segment_presets import (
    CUSTOM_SEGMENT_PRESET_ID,
    resolve_segment_preset_config,
    segment_mask_preset_choices,
)
from aiwf.services.image_workflow import ImageWorkflowService, preset_image_settings


def _choices(items) -> list[tuple[str, str]]:
    return [(item.title, item.id) for item in items]


def _normalized_stages(values) -> list[str]:
    selected = {str(item) for item in (values or [])}
    result = [stage for stage in IMAGE_WORKFLOW_ORDER if stage in selected]
    if "export" not in result:
        result.append("export")
    return result


def _order_markdown(stages, warnings: list[str] | None = None) -> str:
    normalized = _normalized_stages(stages)
    flow = " → ".join(IMAGE_STAGE_LABELS[item] for item in normalized)
    warning_text = "" if not warnings else "  \n" + "  \n".join(f"⚠ {item}" for item in warnings)
    return f"**Resolved order:** {flow}{warning_text}"


def build_image_workflow_panel(ctx: AppContext) -> None:
    service = ImageWorkflowService(ctx)
    checkpoint_choices = _choices(ctx.generation.list_checkpoints())
    sampler_choices = [(item.label, item.id) for item in ctx.generation.list_samplers()]
    restorer_choices = _choices(ctx.enhance.list_restorers())
    upscaler_choices = _choices(ctx.enhance.list_upscalers())
    sam_choices = _choices(ctx.segment.list_models())
    default_checkpoint = ctx.settings.last_checkpoint_id or (checkpoint_choices[0][1] if checkpoint_choices else None)
    default_sampler = getattr(ctx.settings, "default_sampler", "euler_a")
    default_restorer = restorer_choices[0][1] if restorer_choices else None
    default_upscaler = upscaler_choices[0][1] if upscaler_choices else None
    default_sam = sam_choices[0][1] if sam_choices else None
    person = resolve_segment_preset_config("person")
    default_stages = ["tone", "export"]

    with gr.Row(equal_height=False, elem_classes=["aiwf-lab-workspace"]):
        with gr.Column(scale=5, min_width=420, elem_classes=["aiwf-panel", "aiwf-lab-controls"]):
            with gr.Row():
                source = gr.Image(label="Source image", type="pil", sources=["upload", "clipboard"])
                uploaded_mask = gr.Image(
                    label="Optional mask (white = process)", type="pil", image_mode="L", sources=["upload", "clipboard"]
                )

            with gr.Row():
                preset = gr.Dropdown(
                    label="Starting preset",
                    choices=[
                        ("Portrait cleanup", "portrait_cleanup"),
                        ("Old photo", "old_photo"),
                        ("Object replace", "object_replace"),
                        ("Web ready", "web_ready"),
                        ("Custom", "custom"),
                    ],
                    value="custom",
                )
                apply_preset = gr.Button("Apply preset", elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"])

            stages = gr.CheckboxGroup(
                label="Processes to run",
                choices=[(IMAGE_STAGE_LABELS[item], item) for item in IMAGE_WORKFLOW_ORDER],
                value=default_stages,
                info="Pick the work. Studio resolves the safe execution order.",
                elem_classes=["aiwf-stage-picker"],
            )
            order_view = gr.Markdown(_order_markdown(default_stages), elem_classes=["aiwf-stage-order"])

            with gr.Group(visible=False, elem_classes=["aiwf-stage-card"]) as auto_mask_panel:
                gr.Markdown("#### Auto mask")
                with gr.Row():
                    mask_preset = gr.Dropdown(
                        label="Subject preset", choices=segment_mask_preset_choices(), value="person"
                    )
                    mask_model = gr.Dropdown(label="SAM model", choices=sam_choices, value=default_sam)
                mask_custom = gr.Textbox(label="Custom mask prompt", visible=False)
                mask_note = gr.Markdown(str(person.get("note") or ""), elem_classes=["aiwf-stage-note"])
                with gr.Row():
                    mask_threshold = gr.Slider(
                        0.05, 0.90, value=person["box_threshold"], step=0.01, label="Threshold"
                    )
                    mask_index = gr.Slider(0, 2, value=person["mask_index"], step=1, label="Candidate")
                with gr.Row():
                    mask_dilation = gr.Slider(0, 64, value=person["dilation"], step=1, label="Dilation")
                    mask_blur = gr.Slider(0, 64, value=person["mask_blur"], step=1, label="Blur")
                    mask_feather = gr.Slider(0, 64, value=person["feather"], step=1, label="Feather")

            with gr.Group(visible=False, elem_classes=["aiwf-stage-card"]) as inpaint_panel:
                gr.Markdown("#### Inpaint / repair")
                inpaint_prompt = gr.Textbox(
                    label="Prompt", lines=3, placeholder="Describe what should replace the selected area"
                )
                inpaint_negative = gr.Textbox(label="Negative prompt", lines=2)
                with gr.Row():
                    checkpoint = gr.Dropdown(
                        label="Checkpoint", choices=checkpoint_choices, value=default_checkpoint, allow_custom_value=True
                    )
                    sampler = gr.Dropdown(
                        label="Sampler", choices=sampler_choices, value=default_sampler, allow_custom_value=True
                    )
                with gr.Row():
                    steps = gr.Slider(1, 80, value=24, step=1, label="Steps")
                    cfg = gr.Slider(0, 20, value=6.0, step=0.1, label="CFG")
                    denoise = gr.Slider(0, 1, value=0.62, step=0.01, label="Denoise")
                    seed = gr.Number(value=-1, precision=0, label="Seed")

            with gr.Group(visible=False, elem_classes=["aiwf-stage-card"]) as restore_panel:
                gr.Markdown("#### Face / detail restore")
                restore_model = gr.Dropdown(
                    label="Restoration model", choices=restorer_choices, value=default_restorer
                )
                with gr.Row():
                    restore_visibility = gr.Slider(0, 1, value=0.85, step=0.01, label="Visibility")
                    codeformer_weight = gr.Slider(0, 1, value=0.5, step=0.01, label="CodeFormer fidelity")

            with gr.Group(visible=False, elem_classes=["aiwf-stage-card"]) as denoise_panel:
                gr.Markdown("#### Denoise")
                with gr.Row():
                    denoise_radius = gr.Slider(1, 4, value=1, step=1, label="Neighborhood radius")
                    denoise_strength = gr.Slider(0, 1, value=0.35, step=0.01, label="Blend strength")

            with gr.Group(visible=True, elem_classes=["aiwf-stage-card"]) as tone_panel:
                gr.Markdown("#### Tone and color")
                with gr.Row():
                    brightness = gr.Slider(0.25, 2, value=1, step=0.01, label="Brightness")
                    contrast = gr.Slider(0.25, 2, value=1, step=0.01, label="Contrast")
                with gr.Row():
                    saturation = gr.Slider(0, 2, value=1, step=0.01, label="Saturation")
                    sharpness = gr.Slider(0, 2, value=1, step=0.01, label="Sharpness")

            with gr.Group(visible=False, elem_classes=["aiwf-stage-card"]) as upscale_panel:
                gr.Markdown("#### AI upscale")
                upscaler_model = gr.Dropdown(label="Upscaler", choices=upscaler_choices, value=default_upscaler)
                with gr.Row():
                    upscale_factor = gr.Slider(1, 8, value=2, step=0.25, label="Scale")
                    tile_size = gr.Slider(0, 1024, value=256, step=32, label="Tile size (0 = full)")
                    tile_overlap = gr.Slider(0, 256, value=32, step=8, label="Tile overlap")

            with gr.Group(visible=False, elem_classes=["aiwf-stage-card"]) as resize_panel:
                gr.Markdown("#### Final resize")
                with gr.Row():
                    resize_width = gr.Number(value=0, precision=0, minimum=0, label="Width (0 = auto)")
                    resize_height = gr.Number(value=0, precision=0, minimum=0, label="Height (0 = auto)")
                    keep_aspect = gr.Checkbox(value=True, label="Keep aspect")

            with gr.Group(visible=True, elem_classes=["aiwf-stage-card"]) as export_panel:
                gr.Markdown("#### Export")
                with gr.Row():
                    export_format = gr.Dropdown(
                        label="Format", choices=[("PNG", "png"), ("JPEG", "jpg"), ("WebP", "webp")], value="png"
                    )
                    export_quality = gr.Slider(40, 100, value=95, step=1, label="Quality")

            with gr.Row():
                plan_button = gr.Button("Build plan", elem_classes=["aiwf-btn-ghost"])
                run_button = gr.Button("Run Image Lab", variant="primary", elem_classes=["aiwf-generate-btn"])

        with gr.Column(scale=6, min_width=420, elem_classes=["aiwf-panel", "aiwf-lab-output"]):
            output = gr.Image(label="Processed image", type="pil", interactive=False)
            status = gr.Markdown("**Ready**", elem_classes=["aiwf-status-bar"])
            plan_json = gr.Code(label="Resolved job plan", language="json", lines=14, interactive=False)
            with gr.Row():
                mask_output = gr.Image(label="Resolved mask", type="pil", interactive=False)
                mask_preview = gr.Image(label="Mask overlay", type="pil", interactive=False)
            manifest = gr.File(label="Job manifest", interactive=False)
            stage_log = gr.Textbox(label="Stage log", lines=8, interactive=False)

    panels = [
        auto_mask_panel,
        inpaint_panel,
        denoise_panel,
        restore_panel,
        tone_panel,
        upscale_panel,
        resize_panel,
        export_panel,
    ]

    def _stage_visibility(selected):
        normalized = _normalized_stages(selected)
        visible = set(normalized)
        return (
            gr.update(value=normalized),
            _order_markdown(normalized),
            *[gr.update(visible=stage in visible) for stage in IMAGE_WORKFLOW_ORDER],
        )

    stages.change(
        _stage_visibility,
        inputs=[stages],
        outputs=[stages, order_view, *panels],
        show_progress=False,
    )

    def _mask_preset_change(preset_id):
        config = resolve_segment_preset_config(preset_id)
        return (
            gr.update(visible=preset_id == CUSTOM_SEGMENT_PRESET_ID),
            gr.update(value=float(config["box_threshold"])),
            gr.update(value=int(config["mask_index"])),
            gr.update(value=int(config["dilation"])),
            gr.update(value=int(config["mask_blur"])),
            gr.update(value=int(config["feather"])),
            str(config.get("note") or "Custom mask profile."),
        )

    mask_preset.change(
        _mask_preset_change,
        inputs=[mask_preset],
        outputs=[mask_custom, mask_threshold, mask_index, mask_dilation, mask_blur, mask_feather, mask_note],
        show_progress=False,
    )

    preset_outputs = [
        stages,
        mask_preset,
        mask_threshold,
        mask_index,
        mask_dilation,
        mask_blur,
        mask_feather,
        inpaint_prompt,
        denoise,
        restore_visibility,
        denoise_radius,
        denoise_strength,
        brightness,
        contrast,
        saturation,
        sharpness,
        upscale_factor,
        resize_width,
        resize_height,
        export_format,
        export_quality,
        order_view,
        *panels,
    ]

    def _apply_preset(name):
        settings = preset_image_settings(str(name or "custom"))
        normalized = _normalized_stages(settings.stages)
        selected = set(normalized)
        return (
            normalized,
            settings.mask_preset,
            settings.mask_threshold,
            settings.mask_index,
            settings.mask_dilation,
            settings.mask_blur,
            settings.mask_feather,
            settings.inpaint_prompt,
            settings.denoising_strength,
            settings.restore_visibility,
            settings.denoise_radius,
            settings.denoise_strength,
            settings.brightness,
            settings.contrast,
            settings.saturation,
            settings.sharpness,
            settings.upscale_factor,
            settings.resize_width,
            settings.resize_height,
            settings.export_format,
            settings.export_quality,
            _order_markdown(normalized),
            *[gr.update(visible=stage in selected) for stage in IMAGE_WORKFLOW_ORDER],
        )

    apply_preset.click(_apply_preset, inputs=[preset], outputs=preset_outputs, show_progress=False)

    controls = [
        stages,
        preset,
        mask_preset,
        mask_custom,
        mask_model,
        mask_threshold,
        mask_index,
        mask_dilation,
        mask_blur,
        mask_feather,
        inpaint_prompt,
        inpaint_negative,
        checkpoint,
        sampler,
        steps,
        cfg,
        seed,
        denoise,
        restore_model,
        restore_visibility,
        codeformer_weight,
        denoise_radius,
        denoise_strength,
        brightness,
        contrast,
        saturation,
        sharpness,
        upscaler_model,
        upscale_factor,
        tile_size,
        tile_overlap,
        resize_width,
        resize_height,
        keep_aspect,
        export_format,
        export_quality,
    ]

    def _settings(*values) -> ImageWorkflowSettings:
        (
            selected_stages,
            preset_value,
            mask_preset_value,
            mask_custom_value,
            mask_model_value,
            mask_threshold_value,
            mask_index_value,
            mask_dilation_value,
            mask_blur_value,
            mask_feather_value,
            inpaint_prompt_value,
            inpaint_negative_value,
            checkpoint_value,
            sampler_value,
            steps_value,
            cfg_value,
            seed_value,
            denoise_value,
            restore_model_value,
            restore_visibility_value,
            codeformer_value,
            denoise_radius_value,
            denoise_strength_value,
            brightness_value,
            contrast_value,
            saturation_value,
            sharpness_value,
            upscaler_value,
            upscale_factor_value,
            tile_size_value,
            tile_overlap_value,
            resize_width_value,
            resize_height_value,
            keep_aspect_value,
            export_format_value,
            export_quality_value,
        ) = values
        return ImageWorkflowSettings(
            stages=_normalized_stages(selected_stages),
            preset=str(preset_value or "custom"),
            mask_preset=str(mask_preset_value or "person"),
            mask_custom_prompt=str(mask_custom_value or ""),
            mask_model_id=str(mask_model_value) if mask_model_value else None,
            mask_threshold=float(mask_threshold_value),
            mask_index=int(mask_index_value),
            mask_dilation=int(mask_dilation_value),
            mask_blur=int(mask_blur_value),
            mask_feather=int(mask_feather_value),
            inpaint_prompt=str(inpaint_prompt_value or ""),
            inpaint_negative_prompt=str(inpaint_negative_value or ""),
            checkpoint_id=str(checkpoint_value) if checkpoint_value else None,
            sampler=str(sampler_value or "euler_a"),
            steps=int(steps_value),
            cfg_scale=float(cfg_value),
            seed=int(seed_value),
            denoising_strength=float(denoise_value),
            restore_model_id=str(restore_model_value) if restore_model_value else None,
            restore_visibility=float(restore_visibility_value),
            codeformer_weight=float(codeformer_value),
            denoise_radius=int(denoise_radius_value),
            denoise_strength=float(denoise_strength_value),
            brightness=float(brightness_value),
            contrast=float(contrast_value),
            saturation=float(saturation_value),
            sharpness=float(sharpness_value),
            upscaler_model_id=str(upscaler_value) if upscaler_value else None,
            upscale_factor=float(upscale_factor_value),
            tile_size=int(tile_size_value),
            tile_overlap=int(tile_overlap_value),
            resize_width=int(resize_width_value or 0),
            resize_height=int(resize_height_value or 0),
            keep_aspect=bool(keep_aspect_value),
            export_format=str(export_format_value or "png"),
            export_quality=int(export_quality_value),
        )

    def _plan(mask, *values):
        try:
            settings = _settings(*values)
            plan = service.build_plan(settings, has_uploaded_mask=mask is not None)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc
        payload = {
            "resolved_order": plan.stages,
            "labels": plan.labels,
            "warnings": plan.warnings,
            "settings": settings.model_dump(mode="json"),
        }
        return json.dumps(payload, indent=2), _order_markdown(plan.stages, plan.warnings)

    def _run(image, mask, *values):
        if image is None:
            raise gr.Error("Upload a source image first.")
        try:
            settings = _settings(*values)
            plan = service.build_plan(settings, has_uploaded_mask=mask is not None)
            result = service.process(image, settings, uploaded_mask=mask)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc
        payload = {
            "resolved_order": plan.stages,
            "labels": plan.labels,
            "warnings": plan.warnings,
            "settings": settings.model_dump(mode="json"),
        }
        return (
            result.image,
            f"**Done** — {result.message}",
            json.dumps(payload, indent=2),
            result.mask,
            result.mask_preview,
            result.manifest_path,
            "\n".join(result.stage_log),
        )

    plan_button.click(_plan, inputs=[uploaded_mask, *controls], outputs=[plan_json, order_view])
    run_button.click(
        _run,
        inputs=[source, uploaded_mask, *controls],
        outputs=[output, status, plan_json, mask_output, mask_preview, manifest, stage_log],
        concurrency_limit=1,
        concurrency_id="aiwf-image-lab-workflow",
        show_progress="full",
    )
