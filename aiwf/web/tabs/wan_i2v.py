from __future__ import annotations

import logging

import gradio as gr

from aiwf.bootstrap import AppContext
from aiwf.core.domain.wan import WanI2VRequest, duration_seconds_for_frames, frames_for_duration_seconds
from aiwf.infrastructure.wan import WanUnavailable
from aiwf.services.wan import WanService
from aiwf.web.registry import WebRegistry
from aiwf.web.studio.resolution import (
    ASPECT_RATIO_PRESETS,
    NON_SQUARE_ASPECT_RATIO_PRESETS,
    dimensions_from_generation_preset,
)

_SERVICES: dict[int, WanService] = {}
VIDEO_SIZE_PRESETS: tuple[int, ...] = (480, 568, 640, 768, 896, 1024)
logger = logging.getLogger(__name__)


def _service(ctx: AppContext) -> WanService:
    svc = _SERVICES.get(id(ctx))
    if svc is None:
        svc = WanService(ctx.flags, ctx.settings, unload_image_models=ctx.generation.backend.unload)
        _SERVICES[id(ctx)] = svc
    return svc


def register_wan_i2v(registry: WebRegistry) -> None:
    @registry.tab("Video", order=18)
    def build(ctx: AppContext, tab: gr.Tab | None = None) -> None:
        service = _service(ctx)
        local = service.list_local_models()
        high_lora_choices = service.list_local_loras("high") if hasattr(service, "list_local_loras") else []
        low_lora_choices = service.list_local_loras("low") if hasattr(service, "list_local_loras") else []

        high_choices = [m for m in local if "high" in m.lower()] or local
        low_choices = [m for m in local if "low" in m.lower()] or local
        high_choices = list(dict.fromkeys([*high_choices, *local]))
        low_choices = list(dict.fromkeys([*low_choices, *local]))

        vae_choices = service.list_local_vaes() if hasattr(service, "list_local_vaes") else []
        if not vae_choices or vae_choices == [""]:
            vae_choices = [""]
        preferred_vae = service.preferred_vae() if hasattr(service, "preferred_vae") else None

        default_video_size = 480
        default_video_ratio = "1:1"

        with gr.Column(elem_classes=["aiwf-wan", "aiwf-video", "aiwf-mode-video"]):
            with gr.Column(elem_classes=["aiwf-page-header"]):
                gr.Markdown("Video", elem_classes=["aiwf-section-label"])
                video_mode = gr.Radio(
                    show_label=False,
                    container=False,
                    choices=[("Image2Video", "i2v")],
                    value="i2v",
                    elem_classes=["aiwf-mode-toggle"],
                )
                gr.Markdown(
                    "Animate a still image with Wan 2.2. This workspace is tuned for local testing first: "
                    "**480 base resolution**, **16 fps**, short clips, and explicit **high/low step control**. "
                    "High noise and Low noise transformers are both required, and each stage can carry its own LoRA.",
                    elem_classes=["aiwf-page-intro"],
                )
                gr.Markdown(service.folder_help(), elem_classes=["aiwf-page-path"])

            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=340, elem_classes=["aiwf-panel"]):
                    source = gr.Image(label="Source image", type="pil", sources=["upload", "clipboard"])
                    prompt = gr.Textbox(label="Prompt", lines=3, placeholder="Describe the motion / scene")
                    negative = gr.Textbox(label="Negative prompt", lines=2, value="")

                    gr.Markdown("Models", elem_classes=["aiwf-section-label"])
                    high_noise = gr.Dropdown(
                        label="High noise model",
                        choices=high_choices,
                        value=high_choices[0] if high_choices else None,
                        allow_custom_value=True,
                        info="Required. Early denoising stage.",
                    )
                    low_noise = gr.Dropdown(
                        label="Low noise model",
                        choices=low_choices,
                        value=low_choices[0] if low_choices else None,
                        allow_custom_value=True,
                        info="Required. Late denoising stage.",
                    )

                    gr.Markdown("Stage LoRAs", elem_classes=["aiwf-section-label"])
                    high_lora = gr.Dropdown(
                        label="High noise LoRA",
                        choices=high_lora_choices,
                        value=None,
                        allow_custom_value=True,
                        info="Optional. Shows LoRAs with 'high' in the filename from local and ComfyUI LoRA roots.",
                    )
                    with gr.Row():
                        high_lora_scale = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="High LoRA strength")
                        low_lora_scale = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Low LoRA strength")
                    low_lora = gr.Dropdown(
                        label="Low noise LoRA",
                        choices=low_lora_choices,
                        value=None,
                        allow_custom_value=True,
                        info="Optional. Shows LoRAs with 'low' in the filename from local and ComfyUI LoRA roots.",
                    )

                    gr.Markdown("Runtime", elem_classes=["aiwf-section-label"])
                    offload = gr.Dropdown(
                        label="VRAM / offload",
                        choices=[
                            ("Sequential offload (8 GB, slowest)", "sequential"),
                            ("Model offload (12-16 GB)", "model"),
                            ("No offload (keep on GPU, fastest)", "none"),
                        ],
                        value="model",
                    )
                    vae_id = gr.Dropdown(
                        label="VAE",
                        choices=vae_choices,
                        value=preferred_vae or (vae_choices[0] if vae_choices and vae_choices[0] else None),
                        allow_custom_value=True,
                        info="Wan 2.1 VAE is usually the right choice for Wan 2.2 I2V.",
                    )

                with gr.Column(scale=1, min_width=340, elem_classes=["aiwf-panel"]):
                    gr.Markdown("Resolution", elem_classes=["aiwf-section-label"])
                    with gr.Column(elem_classes=["aiwf-resolution-presets"]):
                        with gr.Row(elem_classes=["aiwf-resolution-row"]):
                            gr.HTML('<div class="aiwf-resolution-heading">Size</div>')
                            resolution_size = gr.Radio(
                                show_label=False,
                                container=False,
                                choices=[(str(size), size) for size in VIDEO_SIZE_PRESETS],
                                value=default_video_size,
                                elem_classes=["aiwf-resolution-toggle", "aiwf-resolution-size"],
                            )
                        with gr.Row(elem_classes=["aiwf-resolution-row"]):
                            gr.HTML('<div class="aiwf-resolution-heading">Ratio</div>')
                            with gr.Column(elem_classes=["aiwf-resolution-ratio-stack"]):
                                resolution_ratio = gr.Radio(
                                    show_label=False,
                                    container=False,
                                    choices=list(NON_SQUARE_ASPECT_RATIO_PRESETS),
                                    value=None,
                                    elem_classes=["aiwf-resolution-toggle", "aiwf-resolution-ratio"],
                                )
                                resolution_ratio_square = gr.Radio(
                                    show_label=False,
                                    container=False,
                                    choices=[("1:1", "1:1")],
                                    value=default_video_ratio,
                                    elem_classes=[
                                        "aiwf-resolution-toggle",
                                        "aiwf-resolution-ratio",
                                        "aiwf-resolution-ratio-square",
                                    ],
                                )
                    with gr.Row():
                        width = gr.Slider(128, 1280, value=480, step=8, label="Width")
                        height = gr.Slider(128, 1280, value=480, step=8, label="Height")

                    gr.Markdown("Motion", elem_classes=["aiwf-section-label"])
                    with gr.Row():
                        fps = gr.Slider(1, 24, value=16, step=1, label="FPS")
                        duration_seconds = gr.Slider(1, 10, value=3, step=1, label="Duration (seconds)")
                    with gr.Row():
                        num_frames = gr.Number(value=49, precision=0, label="Frames", interactive=False)
                        guidance = gr.Slider(1.0, 12.0, value=5.0, step=0.5, label="Guidance (CFG)")
                    frame_summary = gr.Markdown(
                        "**Frames:** 49 · **Duration:** 3.0s snapped for Wan",
                        elem_classes=["aiwf-settings-paths"],
                    )

                    gr.Markdown("Denoising split", elem_classes=["aiwf-section-label"])
                    with gr.Row():
                        high_steps = gr.Slider(1, 30, value=4, step=1, label="High noise steps")
                        low_steps = gr.Slider(1, 30, value=4, step=1, label="Low noise steps")
                    with gr.Row():
                        total_steps = gr.Number(value=8, precision=0, label="Total steps", interactive=False)
                        boundary_ratio = gr.Number(value=0.5, precision=3, label="Stage split", interactive=False)
                    with gr.Row():
                        flow_shift = gr.Slider(1.0, 12.0, value=5.0, step=0.5, label="Flow shift")
                        seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")

                    run = gr.Button("Generate video", variant="primary", elem_classes=["aiwf-generate-btn"])
                    video_out = gr.Video(label="Result", interactive=False)
                    status = gr.Markdown("**Ready** — upload an image and generate.", elem_classes=["aiwf-status-bar"])

        def _active_resolution_ratio(ratio_value, square_ratio_value):
            return square_ratio_value or ratio_value or "1:1"

        def _apply_resolution_preset(size_value, ratio_value, square_ratio_value):
            next_width, next_height = dimensions_from_generation_preset(
                size_value,
                _active_resolution_ratio(ratio_value, square_ratio_value),
            )
            return gr.update(value=next_width), gr.update(value=next_height)

        def _apply_main_resolution_ratio(size_value, ratio_value):
            next_width, next_height = dimensions_from_generation_preset(size_value, ratio_value or "1:1")
            return gr.update(value=next_width), gr.update(value=next_height), gr.update(value=None)

        def _apply_square_resolution_ratio(size_value, square_ratio_value):
            next_width, next_height = dimensions_from_generation_preset(size_value, square_ratio_value or "1:1")
            return gr.update(value=next_width), gr.update(value=next_height), gr.update(value=None)

        resolution_size.change(
            _apply_resolution_preset,
            inputs=[resolution_size, resolution_ratio, resolution_ratio_square],
            outputs=[width, height],
            show_progress=False,
        )
        resolution_ratio.change(
            _apply_main_resolution_ratio,
            inputs=[resolution_size, resolution_ratio],
            outputs=[width, height, resolution_ratio_square],
            show_progress=False,
        )
        resolution_ratio_square.change(
            _apply_square_resolution_ratio,
            inputs=[resolution_size, resolution_ratio_square],
            outputs=[width, height, resolution_ratio],
            show_progress=False,
        )

        def _sync_duration(fps_value, duration_value):
            frames = frames_for_duration_seconds(int(fps_value or 16), float(duration_value or 3))
            snapped = duration_seconds_for_frames(frames, int(fps_value or 16))
            return (
                gr.update(value=frames),
                gr.update(value=f"**Frames:** {frames} · **Duration:** {snapped:.1f}s snapped for Wan"),
            )

        fps.change(
            _sync_duration,
            inputs=[fps, duration_seconds],
            outputs=[num_frames, frame_summary],
            show_progress=False,
        )
        duration_seconds.change(
            _sync_duration,
            inputs=[fps, duration_seconds],
            outputs=[num_frames, frame_summary],
            show_progress=False,
        )

        def _sync_step_split(high_value, low_value):
            high = max(1, int(high_value or 0))
            low = max(1, int(low_value or 0))
            total = high + low
            return gr.update(value=total), gr.update(value=round(high / total, 3))

        high_steps.change(
            _sync_step_split,
            inputs=[high_steps, low_steps],
            outputs=[total_steps, boundary_ratio],
            show_progress=False,
        )
        low_steps.change(
            _sync_step_split,
            inputs=[high_steps, low_steps],
            outputs=[total_steps, boundary_ratio],
            show_progress=False,
        )

        def _run(
            image,
            prompt_v,
            negative_v,
            offload_v,
            width_v,
            height_v,
            frames_v,
            fps_v,
            high_steps_v,
            low_steps_v,
            guidance_v,
            flow_v,
            seed_v,
            high_v,
            low_v,
            vae_v,
            high_lora_v,
            high_lora_scale_v,
            low_lora_v,
            low_lora_scale_v,
            progress=gr.Progress(),
        ):
            if image is None:
                raise gr.Error("Upload a source image first.")
            if not service.available():
                raise gr.Error(
                    "Wan video is unavailable — update diffusers (>=0.35) and install ftfy, then restart."
                )
            if not (high_v and low_v):
                raise gr.Error(
                    "Select BOTH a High noise model and a Low noise model. Wan 2.2 image-to-video "
                    "always runs a two-stage high/low transformer pair."
                )
            request = WanI2VRequest(
                prompt=prompt_v or "",
                negative_prompt=negative_v or "",
                width=int(width_v),
                height=int(height_v),
                num_frames=int(frames_v),
                fps=int(fps_v),
                steps=max(1, int(high_steps_v or 0) + int(low_steps_v or 0)),
                high_noise_steps=int(high_steps_v),
                low_noise_steps=int(low_steps_v),
                guidance_scale=float(guidance_v),
                flow_shift=float(flow_v),
                seed=int(seed_v),
                offload=offload_v,
                high_noise_model_id=high_v or None,
                low_noise_model_id=low_v or None,
                high_noise_lora_id=high_lora_v or None,
                high_noise_lora_scale=float(high_lora_scale_v),
                low_noise_lora_id=low_lora_v or None,
                low_noise_lora_scale=float(low_lora_scale_v),
                boundary_ratio=0.5,
                vae_id=vae_v or None,
            )

            def on_progress(step, tot):
                progress(min(1.0, step / max(1, tot)), desc=f"Video step {step}/{tot}")

            progress(0.0, desc="Loading video pipeline…")
            try:
                result = service.generate(request, image, on_progress=on_progress)
            except WanUnavailable as exc:
                raise gr.Error(str(exc))
            except Exception as exc:
                logger.exception("Video generation failed")
                raise gr.Error(f"Video generation failed: {exc}") from exc
            return result.output_path, f"**Done** — {result.message}"

        run.click(
            _run,
            inputs=[
                source,
                prompt,
                negative,
                offload,
                width,
                height,
                num_frames,
                fps,
                high_steps,
                low_steps,
                guidance,
                flow_shift,
                seed,
                high_noise,
                low_noise,
                vae_id,
                high_lora,
                high_lora_scale,
                low_lora,
                low_lora_scale,
            ],
            outputs=[video_out, status],
            show_progress="minimal",
        )

        if tab is not None:

            def _load_pending():
                img = ctx.infotext_bridge.consume_image()
                return gr.update(value=img) if img is not None else gr.update()

            tab.select(_load_pending, outputs=[source], show_progress=False)
