from __future__ import annotations

import gradio as gr

from aiwf.bootstrap import AppContext
from aiwf.core.domain.enhance import RestoreOptions, UpscaleOptions
from aiwf.core.domain.photo_restore import PhotoRestoreOptions
from aiwf.web.registry import WebRegistry


def _model_choices(models) -> list[tuple[str, str]]:
    return [(model.title, model.id) for model in models]


def register_enhance(registry: WebRegistry) -> None:
    @registry.tab("Enhance", order=20)
    def build(ctx: AppContext, tab: gr.Tab | None = None) -> None:
        service = ctx.enhance
        upscalers = service.list_upscalers()
        restorers = service.list_restorers()

        with gr.Column(elem_classes=["aiwf-enhance"]):
            with gr.Column(elem_classes=["aiwf-page-header"]):
                gr.Markdown("Enhance", elem_classes=["aiwf-section-label"])
                gr.Markdown(
                    "Upscale images, restore faces, or repair old photos.",
                    elem_classes=["aiwf-page-intro"],
                )
                folder_help = gr.Markdown(service.catalog.folder_help(), elem_classes=["aiwf-page-path"])

            with gr.Row(equal_height=False, elem_classes=["aiwf-enhance-workspace"]):
                with gr.Column(scale=1, min_width=280, elem_classes=["aiwf-panel"]):
                    source = gr.Image(label="Source image", type="pil", sources=["upload", "clipboard"])
                    refresh_models = gr.Button("Refresh models", elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"])

                with gr.Column(scale=1, min_width=280, elem_classes=["aiwf-panel"]):
                    output = gr.Image(label="Result", type="pil", interactive=False)
                    compare = gr.ImageSlider(
                        label="Before / after",
                        type="pil",
                        interactive=False,
                        visible=False,
                        height=480,
                        elem_classes=["aiwf-compare"],
                    )
                    result_info = gr.Textbox(label="Details", lines=2, interactive=False, elem_classes=["aiwf-gen-info"])
                    status = gr.Markdown("**Ready** — upload an image and choose an action", elem_classes=["aiwf-status-bar"])

            with gr.Tabs(elem_classes=["aiwf-enhance-tabs"]):
                with gr.Tab("Upscale"):
                    upscale_model = gr.Dropdown(
                        label="Upscaler",
                        choices=_model_choices(upscalers),
                        value=upscalers[0].id if upscalers else None,
                    )
                    with gr.Row():
                        upscale_scale = gr.Slider(1, 8, value=4, step=0.5, label="Target scale")
                        upscale_tile = gr.Slider(0, 1024, value=ctx.settings.upscale_tile_size, step=64, label="Tile size (0 = off)")
                    upscale_overlap = gr.Slider(0, 256, value=ctx.settings.upscale_tile_overlap, step=8, label="Tile overlap")
                    run_upscale = gr.Button("Upscale", variant="primary", elem_classes=["aiwf-generate-btn"])

                with gr.Tab("Restore"):
                    restore_model = gr.Dropdown(
                        label="Restoration model",
                        choices=_model_choices(restorers),
                        value=restorers[0].id if restorers else None,
                    )
                    restore_visibility = gr.Slider(0, 1, value=1, step=0.05, label="Effect strength")
                    codeformer_weight = gr.Slider(
                        0,
                        1,
                        value=0.5,
                        step=0.05,
                        label="CodeFormer fidelity",
                        info="Higher = closer to the original face. Only applies to CodeFormer.",
                    )
                    run_restore = gr.Button("Restore faces", variant="primary", elem_classes=["aiwf-generate-btn"])

                with gr.Tab("Old photos"):
                    gr.Markdown(
                        "Multi-stage restoration for faded or damaged prints.",
                        elem_classes=["aiwf-settings-paths"],
                    )
                    photo_preset = gr.Radio(
                        label="Preset",
                        choices=[
                            "Full restore (scratches + global + faces)",
                            "Scratches + global only",
                            "Global + faces",
                            "Faces only",
                        ],
                        value="Full restore (scratches + global + faces)",
                    )
                    with gr.Accordion("Stage 1 — Scratches", open=False):
                        photo_scratches = gr.Checkbox(label="Detect scratches", value=True)
                        photo_inpaint = gr.Checkbox(label="Inpaint scratches", value=True)
                        photo_scratch_sens = gr.Slider(
                            0.05, 0.95, value=0.45, step=0.05, label="Scratch sensitivity"
                        )
                    with gr.Accordion("Stage 2 — Global restore", open=False):
                        photo_global = gr.Checkbox(label="Denoise + color restore", value=True)
                        photo_denoise = gr.Slider(0, 1, value=0.65, step=0.05, label="Denoise strength")
                        photo_color = gr.Slider(0, 1, value=0.55, step=0.05, label="Color / contrast boost")
                    with gr.Accordion("Stages 3–4 — Face enhance", open=False):
                        photo_faces = gr.Checkbox(label="Detect, enhance, and blend faces", value=True)
                        photo_face_model = gr.Dropdown(
                            label="Face model",
                            choices=_model_choices(restorers),
                            value=restorers[0].id if restorers else None,
                        )
                        photo_face_strength = gr.Slider(0, 1, value=0.85, step=0.05, label="Face effect strength")
                        photo_cf_weight = gr.Slider(0, 1, value=0.5, step=0.05, label="CodeFormer fidelity")
                    with gr.Accordion("Stage 5 — Upscale (optional)", open=False):
                        photo_upscale = gr.Checkbox(label="Upscale after restore", value=True)
                        photo_upscale_model = gr.Dropdown(
                            label="Upscaler",
                            choices=_model_choices(upscalers),
                            value=upscalers[0].id if upscalers else None,
                        )
                        photo_upscale_scale = gr.Slider(1, 4, value=2, step=0.5, label="Target scale")
                    run_photo_restore = gr.Button(
                        "Restore old photo", variant="primary", elem_classes=["aiwf-generate-btn"]
                    )

                with gr.Tab("Full pipeline"):
                    pipe_restore = gr.Checkbox(label="Face restoration", value=True)
                    pipe_upscale = gr.Checkbox(label="Upscale", value=True)
                    pipe_restore_first = gr.Radio(
                        label="Order",
                        choices=["Restore then upscale", "Upscale then restore"],
                        value="Restore then upscale",
                    )
                    with gr.Row():
                        pipe_upscale_model = gr.Dropdown(
                            label="Upscaler",
                            choices=_model_choices(upscalers),
                            value=upscalers[0].id if upscalers else None,
                        )
                        pipe_restore_model = gr.Dropdown(
                            label="Restoration model",
                            choices=_model_choices(restorers),
                            value=restorers[0].id if restorers else None,
                        )
                    with gr.Row():
                        pipe_scale = gr.Slider(1, 8, value=4, step=0.5, label="Target scale")
                        pipe_visibility = gr.Slider(0, 1, value=1, step=0.05, label="Restore strength")
                    run_pipeline = gr.Button("Run full pipeline", variant="primary", elem_classes=["aiwf-generate-btn"])

        def _require_image(image):
            if image is None:
                raise gr.Error("Upload a source image first.")
            return image

        def _save_and_return(before, after, infotext: str):
            result = service.save_result(after, infotext)
            status_text = f"**Done** — {result.message}"
            show_compare = before is not None
            return (
                after,
                gr.update(visible=show_compare, value=(before, after) if show_compare else None),
                infotext,
                status_text,
            )

        def do_upscale(image, model_id, scale, tile_size, overlap):
            image = _require_image(image)
            if not model_id:
                raise gr.Error("No upscaler model available. Click Refresh models.")
            ctx.settings.upscale_tile_size = int(tile_size)
            ctx.settings.upscale_tile_overlap = int(overlap)
            ctx.save_settings()
            options = UpscaleOptions(model_id=model_id, scale=float(scale), tile_size=int(tile_size), tile_overlap=int(overlap))
            result = service.upscale(image, options)
            model = service.catalog.get_model(model_id)
            infotext = f"Upscale: {model.title if model else model_id} ({scale}x)"
            return _save_and_return(image, result, infotext)

        run_upscale.click(
            do_upscale,
            inputs=[source, upscale_model, upscale_scale, upscale_tile, upscale_overlap],
            outputs=[output, compare, result_info, status],
        )

        def do_restore(image, model_id, visibility, cf_weight):
            image = _require_image(image)
            if not model_id:
                raise gr.Error("No restoration model available. Click Refresh models.")
            options = RestoreOptions(
                model_id=model_id,
                visibility=float(visibility),
                codeformer_weight=float(cf_weight),
            )
            result = service.restore(image, options)
            model = service.catalog.get_model(model_id)
            infotext = f"Restore: {model.title if model else model_id} (strength {visibility:.2f})"
            return _save_and_return(image, result, infotext)

        run_restore.click(
            do_restore,
            inputs=[source, restore_model, restore_visibility, codeformer_weight],
            outputs=[output, compare, result_info, status],
        )

        def _apply_photo_preset(preset: str):
            full = preset.startswith("Full")
            scratches = full or preset.startswith("Scratches")
            global_on = full or preset.startswith("Global") or preset.startswith("Scratches")
            faces = full or preset.startswith("Global") or preset == "Faces only"
            upscale = full
            return scratches, scratches, faces, global_on, upscale

        photo_preset.change(
            _apply_photo_preset,
            inputs=[photo_preset],
            outputs=[photo_scratches, photo_inpaint, photo_faces, photo_global, photo_upscale],
            show_progress=False,
        )

        def do_photo_restore(
            image,
            do_scratches,
            do_inpaint,
            scratch_sens,
            do_global,
            denoise,
            color_boost,
            do_faces,
            face_model,
            face_strength,
            cf_weight,
            do_upscale,
            up_model,
            up_scale,
        ):
            image = _require_image(image)
            restore_opts = None
            if do_faces:
                if not face_model:
                    raise gr.Error("Select a face restoration model.")
                restore_opts = RestoreOptions(
                    model_id=face_model,
                    visibility=float(face_strength),
                    codeformer_weight=float(cf_weight),
                )
            upscale_opts = None
            if do_upscale:
                if not up_model:
                    raise gr.Error("Select an upscaler model.")
                upscale_opts = UpscaleOptions(
                    model_id=up_model,
                    scale=float(up_scale),
                    tile_size=ctx.settings.upscale_tile_size,
                    tile_overlap=ctx.settings.upscale_tile_overlap,
                )
            options = PhotoRestoreOptions(
                scratch_detection=bool(do_scratches),
                scratch_inpaint=bool(do_inpaint),
                scratch_sensitivity=float(scratch_sens),
                global_restore=bool(do_global),
                denoise_strength=float(denoise),
                color_boost=float(color_boost),
                face_restore=bool(do_faces),
                restore=restore_opts,
                upscale=upscale_opts,
            )
            result, infotext = service.run_photo_restore(image, options)
            return _save_and_return(image, result, infotext)

        run_photo_restore.click(
            do_photo_restore,
            inputs=[
                source,
                photo_scratches,
                photo_inpaint,
                photo_scratch_sens,
                photo_global,
                photo_denoise,
                photo_color,
                photo_faces,
                photo_face_model,
                photo_face_strength,
                photo_cf_weight,
                photo_upscale,
                photo_upscale_model,
                photo_upscale_scale,
            ],
            outputs=[output, compare, result_info, status],
        )

        def do_pipeline(
            image,
            use_restore,
            use_upscale,
            order,
            up_model,
            rest_model,
            scale,
            visibility,
        ):
            image = _require_image(image)
            restore_opts = None
            upscale_opts = None
            if use_restore:
                if not rest_model:
                    raise gr.Error("Select a restoration model.")
                restore_opts = RestoreOptions(model_id=rest_model, visibility=float(visibility))
            if use_upscale:
                if not up_model:
                    raise gr.Error("Select an upscaler model.")
                upscale_opts = UpscaleOptions(
                    model_id=up_model,
                    scale=float(scale),
                    tile_size=ctx.settings.upscale_tile_size,
                    tile_overlap=ctx.settings.upscale_tile_overlap,
                )
            if restore_opts is None and upscale_opts is None:
                raise gr.Error("Enable at least one step in the pipeline.")
            result, infotext = service.run_pipeline(
                image,
                restore=restore_opts,
                upscale=upscale_opts,
                restore_first=order.startswith("Restore"),
            )
            return _save_and_return(image, result, infotext)

        run_pipeline.click(
            do_pipeline,
            inputs=[
                source,
                pipe_restore,
                pipe_upscale,
                pipe_restore_first,
                pipe_upscale_model,
                pipe_restore_model,
                pipe_scale,
                pipe_visibility,
            ],
            outputs=[output, compare, result_info, status],
        )

        def refresh_catalog():
            models = service.refresh_catalog()
            ups = [m for m in models if m.kind.value == "upscaler"]
            rest = [m for m in models if m.kind.value == "restorer"]
            up_choices = _model_choices(ups)
            rest_choices = _model_choices(rest)
            default_up = ups[0].id if ups else None
            default_rest = rest[0].id if rest else None
            return (
                gr.update(choices=up_choices, value=default_up),
                gr.update(choices=up_choices, value=default_up),
                gr.update(choices=up_choices, value=default_up),
                gr.update(choices=rest_choices, value=default_rest),
                gr.update(choices=rest_choices, value=default_rest),
                gr.update(choices=rest_choices, value=default_rest),
                service.catalog.folder_help(),
                f"**{len(ups)}** upscalers · **{len(rest)}** restoration models",
            )

        refresh_models.click(
            refresh_catalog,
            outputs=[
                upscale_model,
                pipe_upscale_model,
                photo_upscale_model,
                restore_model,
                pipe_restore_model,
                photo_face_model,
                folder_help,
                status,
            ],
            show_progress=False,
        )

        if tab is not None:
            tab.select(
                refresh_catalog,
                outputs=[
                    upscale_model,
                    pipe_upscale_model,
                    photo_upscale_model,
                    restore_model,
                    pipe_restore_model,
                    photo_face_model,
                    folder_help,
                    status,
                ],
                show_progress=False,
            )
