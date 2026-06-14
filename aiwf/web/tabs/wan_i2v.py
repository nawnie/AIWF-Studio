from __future__ import annotations

import logging

import gradio as gr

from aiwf.bootstrap import AppContext
from aiwf.core.domain.wan import SIGMA_TYPES, SAMPLER_TYPES, WanI2VRequest, duration_seconds_for_frames, frames_for_duration_seconds
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

        # Labeled (display_name, identifier) choices — read from model file headers.
        all_labeled = service.list_local_models_labeled() if hasattr(service, "list_local_models_labeled") else []
        if not all_labeled:
            all_labeled = [(m, m) for m in service.list_local_models()]

        high_lora_choices = service.list_local_loras("high") if hasattr(service, "list_local_loras") else []
        low_lora_choices = service.list_local_loras("low") if hasattr(service, "list_local_loras") else []

        # Sort high/low noise models to the top of each dropdown; unknown-role in both.
        high_labeled = [c for c in all_labeled if "high" in c[0].lower() or "high" in c[1].lower()]
        low_labeled  = [c for c in all_labeled if "low"  in c[0].lower() or "low"  in c[1].lower()]
        other_labeled = [c for c in all_labeled if c not in high_labeled and c not in low_labeled]
        high_labeled = list(dict.fromkeys(high_labeled + other_labeled + all_labeled))
        low_labeled  = list(dict.fromkeys(low_labeled  + other_labeled + all_labeled))

        # Load persisted defaults
        _s = ctx.settings
        _last_high = getattr(_s, "last_wan_high", "")
        _last_low  = getattr(_s, "last_wan_low", "")
        _last_vae  = getattr(_s, "last_wan_vae", "")
        _last_te   = getattr(_s, "last_wan_text_encoder", "")
        _last_offload = getattr(_s, "last_wan_offload", "model")

        def _best_default(labeled: list[tuple[str, str]], persisted: str) -> str | None:
            ids = [v for _, v in labeled]
            if persisted and persisted in ids:
                return persisted
            return ids[0] if ids else None

        vae_labeled = service.list_local_vaes_labeled() if hasattr(service, "list_local_vaes_labeled") else []
        if not vae_labeled:
            vae_labeled = [("Default VAE", "")]
        preferred_vae_id = _last_vae or next(
            (v for _, v in vae_labeled if v and "wan" in v.lower() and "vae" in v.lower()), None
        ) or (vae_labeled[0][1] if vae_labeled else None)

        te_labeled = service.list_local_text_encoders_labeled() if hasattr(service, "list_local_text_encoders_labeled") else []
        default_te_labeled = [("Default (full precision bundled encoder)", "")] + te_labeled
        default_te = _last_te if _last_te else (service.default_text_encoder() if hasattr(service, "default_text_encoder") else "")

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
                    "Wan 2.2 image-to-video — requires a High Noise + Low Noise transformer pair. "
                    "Models are detected automatically from file headers.",
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
                        label="High noise transformer",
                        choices=high_labeled,
                        value=_best_default(high_labeled, _last_high),
                        allow_custom_value=True,
                        info="Required — early denoising stage. FP8 or GGUF Q4_K_M recommended for 16 GB.",
                    )
                    low_noise = gr.Dropdown(
                        label="Low noise transformer",
                        choices=low_labeled,
                        value=_best_default(low_labeled, _last_low),
                        allow_custom_value=True,
                        info="Required — late denoising stage. Must match the High Noise model series.",
                    )
                    text_encoder = gr.Dropdown(
                        label="Text encoder (UMT5-XXL)",
                        choices=default_te_labeled,
                        value=default_te if default_te else "",
                        allow_custom_value=True,
                        info="UMT5-XXL only — FP8 or GGUF saves ~5 GB VRAM vs full precision. Default uses bundled encoder.",
                    )
                    vae_id = gr.Dropdown(
                        label="VAE",
                        choices=vae_labeled,
                        value=preferred_vae_id,
                        allow_custom_value=True,
                        info="Wan 2.1 VAE recommended for Wan 2.2 I2V.",
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
                            ("Model offload (16 GB + FP8 safetensors — recommended)", "model"),
                            ("Sequential offload (8 GB fallback, ~3-10x slower)", "sequential"),
                            ("No offload (fastest, needs 24 GB+ VRAM)", "none"),
                        ],
                        value="model",
                        info="On RTX 4070 Ti 16GB with FP8 safetensors, use Model offload. Sequential moves every layer over PCIe each step.",
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
                        guidance = gr.Slider(1.0, 12.0, value=1.0, step=0.5, label="Guidance (CFG)")
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

                    gr.Markdown("Sampler", elem_classes=["aiwf-section-label"])
                    sampler = gr.Dropdown(
                        label="Sampler",
                        choices=[
                            ("FlowMatch Euler (recommended — fast, 1 NFE/step)", "euler"),
                            ("FlowMatch Heun (2nd-order, higher quality, ~2× slower)", "heun"),
                        ],
                        value="euler",
                        info="Euler is the standard Wan solver. Heun uses a predictor-corrector 2nd-order step — better motion at same step count but roughly doubles inference time.",
                    )
                    sigma_type = gr.Dropdown(
                        label="Scheduler",
                        choices=[
                            ("Beta — smooth motion, best quality at low steps (recommended)", "beta"),
                            ("Simple — linear uniform spacing (fastest)", "simple"),
                            ("Exponential — more detail at high noise", "exponential"),
                            ("Karras — SD-style detail preservation", "karras"),
                        ],
                        value="beta",
                        info="Controls how denoising steps are spaced across the noise range. Beta is the best starting point for Wan.",
                    )
                    with gr.Row():
                        flow_shift = gr.Slider(
                            0.5, 25.0, value=5.0, step=0.5, label="Flow shift",
                            info="Wan 2.2 default: 5.0. Higher values (8-12) push more steps toward high-noise. Try 7-9 for sharper motion.",
                        )
                        seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")

                    gr.Markdown("Temporal chunk / reference", elem_classes=["aiwf-section-label"])
                    gr.Markdown(
                        "Controls how the denoiser slices long clips. **Frame context overlap** = how many frames "
                        "each chunk borrows from the previous chunk — larger values reduce brightness seams and help "
                        "the model 'remember' recent motion. **Image guidance** boosts how strongly the initial "
                        "reference image is applied; increase this (1.5–3.0) if the video drifts away from the source "
                        "at high frame counts.",
                        elem_classes=["aiwf-settings-paths"],
                    )
                    with gr.Row():
                        chunk_size = gr.Slider(
                            4, 64, value=16, step=4, label="Chunk size (frames)",
                            info="Frames processed per transformer pass. Smaller = less VRAM per step but more seams. 16 is optimal for 16 GB.",
                        )
                        chunk_overlap = gr.Slider(
                            0, 32, value=8, step=1, label="Frame context overlap",
                            info="Frames shared between adjacent chunks. Higher = smoother seams + better temporal memory of recent motion. 8 recommended.",
                        )
                    image_guidance_scale = gr.Slider(
                        1.0, 5.0, value=1.0, step=0.1, label="Image guidance scale",
                        info="Boost reference image conditioning. 1.0 = standard. Increase to 1.5–3.0 to reduce drift at 65–81+ frames.",
                    )

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
            sampler_v,
            sigma_type_v,
            flow_v,
            seed_v,
            high_v,
            low_v,
            vae_v,
            text_encoder_v,
            high_lora_v,
            high_lora_scale_v,
            low_lora_v,
            low_lora_scale_v,
            chunk_size_v,
            chunk_overlap_v,
            image_guidance_scale_v,
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
            # Warn if user somehow selected a t5xxl file (shouldn't happen via dropdown but allow_custom_value=True)
            _te_path = str(text_encoder_v or "").strip()
            if _te_path and ("t5xxl" in _te_path.lower()) and not any(k in _te_path.lower() for k in ("umt5", "nsfw_wan")):
                raise gr.Error(
                    f"⚠ '{_te_path}' looks like a T5-XXL file (Flux/SD3). "
                    "T5-XXL is NOT compatible with Wan — it will produce garbage output. "
                    "Select 'Default' or a UMT5-XXL file (umt5-xxl-*.gguf or umt5/nsfw_wan_*.safetensors)."
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
                sampler=str(sampler_v or "euler"),
                sigma_type=str(sigma_type_v or "beta"),
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
                text_encoder_path=_te_path,
                chunk_size=int(chunk_size_v or 16),
                chunk_overlap=int(chunk_overlap_v or 8),
                image_guidance_scale=float(image_guidance_scale_v or 1.0),
            )

            def on_progress(step, tot):
                progress(min(1.0, step / max(1, tot)), desc=f"Video step {step}/{tot}")

            progress(0.0, desc="Loading + encoding for Wan 14B dual-stage I2V (UMT5 text + VAE cond can take 30-180s with low GPU util; watch terminal for [AIWF] Video: and step messages, then 'Video step X/Y' will appear)")
            try:
                result = service.generate(request, image, on_progress=on_progress)
            except WanUnavailable as exc:
                raise gr.Error(str(exc))
            except Exception as exc:
                logger.exception("Video generation failed")
                raise gr.Error(f"Video generation failed: {exc}") from exc

            # Persist last-used model/encoder selections so they restore on next launch
            s = ctx.settings
            changed = False
            for attr, val in [
                ("last_wan_high", str(high_v or "")),
                ("last_wan_low", str(low_v or "")),
                ("last_wan_vae", str(vae_v or "")),
                ("last_wan_text_encoder", str(text_encoder_v or "")),
                ("last_wan_offload", str(offload_v or "model")),
            ]:
                if getattr(s, attr, None) != val:
                    setattr(s, attr, val)
                    changed = True
            if changed:
                try:
                    ctx.save_settings()
                except Exception:
                    pass

            return result.output_path, f"**Done** -- {result.message}"

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
                sampler,
                sigma_type,
                flow_shift,
                seed,
                high_noise,
                low_noise,
                vae_id,
                text_encoder,
                high_lora,
                high_lora_scale,
                low_lora,
                low_lora_scale,
                chunk_size,
                chunk_overlap,
                image_guidance_scale,
            ],
            outputs=[video_out, status],
            show_progress="minimal",
        )

        if tab is not None:

            def _load_pending():
                img = ctx.infotext_bridge.consume_image()
                return gr.update(value=img) if img is not None else gr.update()

            tab.select(_load_pending, outputs=[source], show_progress=False)
