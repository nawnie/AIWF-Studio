from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr
from PIL import Image

from aiwf.bootstrap import AppContext
from aiwf.core.domain.generation import GenerationMode, GenerationRequest
from aiwf.services.image_lab import ImageLabService, build_plot_axes, maturity_matrix_markdown
from aiwf.services.plot import PlotRequest
from aiwf.web.registry import WebRegistry


def _checkpoint_choices(ctx: AppContext) -> list[tuple[str, str]]:
    return [(checkpoint.title, checkpoint.id) for checkpoint in ctx.generation.list_checkpoints()]


def _sampler_choices(ctx: AppContext) -> list[tuple[str, str]]:
    return [(sampler.label, sampler.id) for sampler in ctx.generation.list_samplers()]


def _load_file_image(file_obj: Any) -> Image.Image:
    path = _file_path(file_obj)
    if not path:
        raise gr.Error("Could not read uploaded image path.")
    return Image.open(path).convert("RGB")


def _file_path(file_obj: Any) -> str | None:
    if isinstance(file_obj, (str, Path)):
        return str(file_obj)
    return getattr(file_obj, "name", None) or getattr(file_obj, "path", None)


def _load_file_images(files: list[Any] | None) -> list[Image.Image]:
    return [_load_file_image(file) for file in (files or [])]


def _base_request(
    mode: str,
    prompt: str | None,
    negative_prompt: str | None,
    checkpoint_id: str | None,
    sampler: str | None,
    steps: float | int,
    cfg_scale: float | int,
    width: float | int,
    height: float | int,
    seed: float | int,
    denoising_strength: float | int,
    batch_size: float | int = 1,
) -> GenerationRequest:
    return GenerationRequest(
        mode=GenerationMode(mode),
        prompt=prompt or "",
        negative_prompt=negative_prompt or "",
        checkpoint_id=checkpoint_id or None,
        sampler=sampler or "euler_a",
        steps=int(steps),
        cfg_scale=float(cfg_scale),
        width=int(width),
        height=int(height),
        seed=int(seed),
        denoising_strength=float(denoising_strength),
        batch_size=int(batch_size),
        batch_count=1,
    )


def register_image_lab(registry: WebRegistry) -> None:
    @registry.tab("Image Lab", order=2)
    def build(ctx: AppContext, tab: gr.Tab | None = None) -> None:
        checkpoint_choices = _checkpoint_choices(ctx)
        sampler_choices = _sampler_choices(ctx)
        default_checkpoint = ctx.settings.last_checkpoint_id or (checkpoint_choices[0][1] if checkpoint_choices else None)
        default_sampler = getattr(ctx.settings, "default_sampler", "euler_a")

        with gr.Column(elem_classes=["aiwf-settings"]):
            with gr.Column(elem_classes=["aiwf-page-header"]):
                gr.Markdown("Image Lab", elem_classes=["aiwf-section-label"])
                gr.Markdown(
                    "Maturity matrix, XYZ plots, batch image-to-image, inpaint batches, and loopback runs.",
                    elem_classes=["aiwf-page-intro"],
                )

            gr.Markdown(maturity_matrix_markdown(), elem_classes=["aiwf-panel"])

            with gr.Tabs(elem_classes=["aiwf-enhance-tabs"]):
                with gr.Tab("XYZ"):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=1, min_width=300, elem_classes=["aiwf-panel"]):
                            plot_prompt = gr.Textbox(label="Prompt", lines=3)
                            plot_negative = gr.Textbox(label="Negative prompt", lines=2)
                            plot_mode = gr.Dropdown(
                                label="Mode",
                                choices=[
                                    ("Text to image", GenerationMode.TXT2IMG.value),
                                    ("Image to image", GenerationMode.IMG2IMG.value),
                                    ("Inpaint", GenerationMode.INPAINT.value),
                                ],
                                value=GenerationMode.TXT2IMG.value,
                            )
                            plot_checkpoint = gr.Dropdown(
                                label="Checkpoint",
                                choices=checkpoint_choices,
                                value=default_checkpoint,
                            )
                            plot_sampler = gr.Dropdown(
                                label="Sampler",
                                choices=sampler_choices,
                                value=default_sampler,
                            )
                            with gr.Row():
                                plot_width = gr.Number(label="Width", value=ctx.settings.default_width, precision=0)
                                plot_height = gr.Number(label="Height", value=ctx.settings.default_height, precision=0)
                            with gr.Row():
                                plot_steps = gr.Number(label="Steps", value=ctx.settings.default_steps, precision=0)
                                plot_cfg = gr.Number(label="CFG", value=ctx.settings.default_cfg_scale)
                            with gr.Row():
                                plot_seed = gr.Number(label="Seed", value=-1, precision=0)
                                plot_denoise = gr.Slider(0, 1, value=0.75, step=0.01, label="Denoise")
                            plot_source = gr.Image(label="Source image", type="pil", sources=["upload", "clipboard"])
                            plot_mask = gr.Image(label="Mask image", type="pil", sources=["upload", "clipboard"])

                        with gr.Column(scale=1, min_width=300, elem_classes=["aiwf-panel"]):
                            with gr.Row():
                                axis_1 = gr.Dropdown(
                                    label="X axis",
                                    choices=_axis_choices(),
                                    value="seed",
                                    allow_custom_value=True,
                                )
                                axis_1_values = gr.Textbox(label="X values", value="1, 2, 3")
                            with gr.Row():
                                axis_2 = gr.Dropdown(
                                    label="Y axis",
                                    choices=_axis_choices(),
                                    value=None,
                                    allow_custom_value=True,
                                )
                                axis_2_values = gr.Textbox(label="Y values")
                            with gr.Row():
                                axis_3 = gr.Dropdown(
                                    label="Z axis",
                                    choices=_axis_choices(),
                                    value=None,
                                    allow_custom_value=True,
                                )
                                axis_3_values = gr.Textbox(label="Z values")
                            run_plot = gr.Button("Run plot", variant="primary", elem_classes=["aiwf-generate-btn"])
                            plot_grid = gr.Image(label="Grid", type="pil", interactive=False)
                            plot_gallery = gr.Gallery(
                                label="Cells",
                                columns=2,
                                object_fit="contain",
                                height=420,
                                elem_classes=["aiwf-results-gallery"],
                            )
                            plot_status = gr.Markdown("Ready.", elem_classes=["aiwf-status-bar"])

                with gr.Tab("Batch"):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=1, min_width=300, elem_classes=["aiwf-panel"]):
                            batch_prompt = gr.Textbox(label="Prompt", lines=3)
                            batch_negative = gr.Textbox(label="Negative prompt", lines=2)
                            batch_mode = gr.Dropdown(
                                label="Mode",
                                choices=[
                                    ("Image to image", GenerationMode.IMG2IMG.value),
                                    ("Inpaint", GenerationMode.INPAINT.value),
                                ],
                                value=GenerationMode.IMG2IMG.value,
                            )
                            batch_checkpoint = gr.Dropdown(
                                label="Checkpoint",
                                choices=checkpoint_choices,
                                value=default_checkpoint,
                            )
                            batch_sampler = gr.Dropdown(label="Sampler", choices=sampler_choices, value=default_sampler)
                            with gr.Row():
                                batch_steps = gr.Number(label="Steps", value=ctx.settings.default_steps, precision=0)
                                batch_cfg = gr.Number(label="CFG", value=ctx.settings.default_cfg_scale)
                            with gr.Row():
                                batch_width = gr.Number(label="Width", value=ctx.settings.default_width, precision=0)
                                batch_height = gr.Number(label="Height", value=ctx.settings.default_height, precision=0)
                            with gr.Row():
                                batch_seed = gr.Number(label="Seed", value=-1, precision=0)
                                batch_denoise = gr.Slider(0, 1, value=0.75, step=0.01, label="Denoise")
                            source_files = gr.File(
                                label="Source images",
                                file_count="multiple",
                                type="filepath",
                            )
                            mask_files = gr.File(
                                label="Mask images",
                                file_count="multiple",
                                type="filepath",
                            )
                            run_batch = gr.Button("Run batch", variant="primary", elem_classes=["aiwf-generate-btn"])

                        with gr.Column(scale=1, min_width=300, elem_classes=["aiwf-panel"]):
                            batch_grid = gr.Image(label="Grid", type="pil", interactive=False)
                            batch_gallery = gr.Gallery(
                                label="Results",
                                columns=2,
                                object_fit="contain",
                                height=520,
                                elem_classes=["aiwf-results-gallery"],
                            )
                            batch_status = gr.Markdown("Ready.", elem_classes=["aiwf-status-bar"])

                with gr.Tab("Loopback"):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=1, min_width=300, elem_classes=["aiwf-panel"]):
                            loop_source = gr.Image(label="Source image", type="pil", sources=["upload", "clipboard"])
                            loop_prompt = gr.Textbox(label="Prompt", lines=3)
                            loop_negative = gr.Textbox(label="Negative prompt", lines=2)
                            loop_checkpoint = gr.Dropdown(
                                label="Checkpoint",
                                choices=checkpoint_choices,
                                value=default_checkpoint,
                            )
                            loop_sampler = gr.Dropdown(label="Sampler", choices=sampler_choices, value=default_sampler)
                            with gr.Row():
                                loop_steps = gr.Number(label="Steps", value=ctx.settings.default_steps, precision=0)
                                loop_cfg = gr.Number(label="CFG", value=ctx.settings.default_cfg_scale)
                            with gr.Row():
                                loop_width = gr.Number(label="Width", value=ctx.settings.default_width, precision=0)
                                loop_height = gr.Number(label="Height", value=ctx.settings.default_height, precision=0)
                            with gr.Row():
                                loop_seed = gr.Number(label="Seed", value=-1, precision=0)
                                loop_denoise = gr.Slider(0, 1, value=0.75, step=0.01, label="Start denoise")
                            with gr.Row():
                                loop_iterations = gr.Number(label="Iterations", value=4, precision=0)
                                loop_decay = gr.Slider(0.5, 1.0, value=1.0, step=0.01, label="Denoise decay")
                            loop_seed_mode = gr.Dropdown(
                                label="Seed mode",
                                choices=[("Increment", "increment"), ("Fixed", "fixed"), ("Random", "random")],
                                value="increment",
                            )
                            run_loopback = gr.Button("Run loopback", variant="primary", elem_classes=["aiwf-generate-btn"])

                        with gr.Column(scale=1, min_width=300, elem_classes=["aiwf-panel"]):
                            loop_grid = gr.Image(label="Grid", type="pil", interactive=False)
                            loop_gallery = gr.Gallery(
                                label="Iterations",
                                columns=2,
                                object_fit="contain",
                                height=520,
                                elem_classes=["aiwf-results-gallery"],
                            )
                            loop_status = gr.Markdown("Ready.", elem_classes=["aiwf-status-bar"])

        def do_plot(
            prompt,
            negative,
            mode,
            checkpoint_id,
            sampler,
            width,
            height,
            steps,
            cfg,
            seed,
            denoise,
            source_image,
            mask_image,
            x_axis,
            x_values,
            y_axis,
            y_values,
            z_axis,
            z_values,
        ):
            request = _base_request(mode, prompt, negative, checkpoint_id, sampler, steps, cfg, width, height, seed, denoise)
            axes = build_plot_axes(((x_axis, x_values), (y_axis, y_values), (z_axis, z_values)))
            if not axes:
                raise gr.Error("Add at least one axis with values.")
            init_images = [source_image] if request.mode in {GenerationMode.IMG2IMG, GenerationMode.INPAINT} and source_image else None
            mask_images = [mask_image] if request.mode == GenerationMode.INPAINT and mask_image else None
            if request.mode in {GenerationMode.IMG2IMG, GenerationMode.INPAINT} and not init_images:
                raise gr.Error("Source image is required for image-to-image and inpaint plots.")
            if request.mode == GenerationMode.INPAINT and not mask_images:
                raise gr.Error("Mask image is required for inpaint plots.")
            result = ctx.plots.run(PlotRequest(base=request, axes=axes), init_images=init_images, mask_images=mask_images)
            return result.grid, result.images, f"Completed {len(result.images)} plot cell(s)."

        def do_batch(
            prompt,
            negative,
            mode,
            checkpoint_id,
            sampler,
            steps,
            cfg,
            width,
            height,
            seed,
            denoise,
            sources,
            masks,
        ):
            images = _load_file_images(sources)
            if not images:
                raise gr.Error("Upload at least one source image.")
            mask_images = _load_file_images(masks) if masks else None
            request = _base_request(mode, prompt, negative, checkpoint_id, sampler, steps, cfg, width, height, seed, denoise)
            result = ImageLabService(ctx.generation).run_batch(request, images, masks=mask_images)
            return result.grid, result.images, f"Completed {len(result.images)} batch result(s)."

        def do_loopback(
            source,
            prompt,
            negative,
            checkpoint_id,
            sampler,
            steps,
            cfg,
            width,
            height,
            seed,
            denoise,
            iterations,
            decay,
            seed_mode,
        ):
            if source is None:
                raise gr.Error("Source image is required for loopback.")
            request = _base_request(
                GenerationMode.IMG2IMG.value,
                prompt,
                negative,
                checkpoint_id,
                sampler,
                steps,
                cfg,
                width,
                height,
                seed,
                denoise,
            )
            result = ImageLabService(ctx.generation).run_loopback(
                request,
                source,
                iterations=int(iterations),
                denoise_decay=float(decay),
                seed_mode=str(seed_mode or "increment"),
            )
            return result.grid, result.images, f"Completed {len(result.images)} loopback iteration(s)."

        run_plot.click(
            do_plot,
            inputs=[
                plot_prompt,
                plot_negative,
                plot_mode,
                plot_checkpoint,
                plot_sampler,
                plot_width,
                plot_height,
                plot_steps,
                plot_cfg,
                plot_seed,
                plot_denoise,
                plot_source,
                plot_mask,
                axis_1,
                axis_1_values,
                axis_2,
                axis_2_values,
                axis_3,
                axis_3_values,
            ],
            outputs=[plot_grid, plot_gallery, plot_status],
        )
        run_batch.click(
            do_batch,
            inputs=[
                batch_prompt,
                batch_negative,
                batch_mode,
                batch_checkpoint,
                batch_sampler,
                batch_steps,
                batch_cfg,
                batch_width,
                batch_height,
                batch_seed,
                batch_denoise,
                source_files,
                mask_files,
            ],
            outputs=[batch_grid, batch_gallery, batch_status],
        )
        run_loopback.click(
            do_loopback,
            inputs=[
                loop_source,
                loop_prompt,
                loop_negative,
                loop_checkpoint,
                loop_sampler,
                loop_steps,
                loop_cfg,
                loop_width,
                loop_height,
                loop_seed,
                loop_denoise,
                loop_iterations,
                loop_decay,
                loop_seed_mode,
            ],
            outputs=[loop_grid, loop_gallery, loop_status],
        )


def _axis_choices() -> list[str]:
    return [
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
    ]
