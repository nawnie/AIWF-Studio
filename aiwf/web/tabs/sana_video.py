from __future__ import annotations

from typing import Any

import gradio as gr

from aiwf.bootstrap import AppContext
from aiwf.core.domain.sana_video import (
    SANA_VIDEO_PIPELINE_I2V,
    SANA_VIDEO_PIPELINE_T2V,
    SANA_VIDEO_QUANTIZATION_AUTO,
    SANA_VIDEO_QUANTIZATION_BF16,
    SANA_VIDEO_QUANTIZATION_BNB_FP4,
    SANA_VIDEO_QUANTIZATION_BNB_INT8,
    SANA_VIDEO_QUANTIZATION_BNB_NF4,
    SANA_VIDEO_QUANTIZATION_FP8,
    SANA_VIDEO_VAE_TILING_ALWAYS,
    SANA_VIDEO_VAE_TILING_AUTO,
    SANA_VIDEO_VAE_TILING_OFF,
    SanaVideoRequest,
)
from aiwf.services.sana_video import SanaVideoService
from aiwf.web.registry import WebRegistry

_SERVICES: dict[int, SanaVideoService] = {}

SANA_VIDEO_QUANTIZATION_CHOICES = [
    ("Auto", SANA_VIDEO_QUANTIZATION_AUTO),
    ("BF16", SANA_VIDEO_QUANTIZATION_BF16),
    ("FP8 layerwise", SANA_VIDEO_QUANTIZATION_FP8),
    ("BNB 8-bit", SANA_VIDEO_QUANTIZATION_BNB_INT8),
    ("BNB NF4", SANA_VIDEO_QUANTIZATION_BNB_NF4),
    ("BNB FP4", SANA_VIDEO_QUANTIZATION_BNB_FP4),
]

SANA_VIDEO_VAE_TILING_CHOICES = [
    ("Auto retry", SANA_VIDEO_VAE_TILING_AUTO),
    ("Off", SANA_VIDEO_VAE_TILING_OFF),
    ("Always", SANA_VIDEO_VAE_TILING_ALWAYS),
]

SANA_VIDEO_PIPELINE_CHOICES = [
    ("Text to video", SANA_VIDEO_PIPELINE_T2V),
    ("Image to video", SANA_VIDEO_PIPELINE_I2V),
]


def _service(ctx: AppContext) -> SanaVideoService:
    service = _SERVICES.get(id(ctx))
    if service is None:
        service = SanaVideoService(
            ctx.flags,
            ctx.settings,
            ctx.generation.backend.devices,
            supervisor=getattr(ctx, "supervisor", None),
        )
        _SERVICES[id(ctx)] = service
    return service


def _run_sana_video(
    ctx: AppContext,
    prompt: str,
    negative_prompt: str,
    pipeline: str,
    source_image: str | None,
    model_path: str,
    width: int,
    height: int,
    frames: int,
    fps: float,
    steps: int,
    cfg_scale: float,
    motion_score: int,
    seed: int,
    quantization: str,
    vae_tiling: str,
    offload_text_encoder_after_encode: bool,
    use_sage_attention: bool,
    generate_audio: bool,
    audio_prompt: str,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[str | None, str, dict[str, Any]]:
    service = _service(ctx)
    events: list[dict[str, Any]] = []

    def on_progress(stage: str, value: float, message: str, step: int = 0, total: int = 0, seconds: float = 0.0) -> None:
        progress(float(value), desc=f"{stage}: {message}")
        events.append(
            {
                "stage": stage,
                "progress": value,
                "message": message,
                "step": step,
                "total": total,
                "seconds": seconds,
            }
        )

    try:
        request = SanaVideoRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            pipeline=pipeline,
            source_image_path=source_image if pipeline == SANA_VIDEO_PIPELINE_I2V else None,
            model_path=model_path,
            width=int(width),
            height=int(height),
            frames=int(frames),
            fps=float(fps),
            steps=int(steps),
            cfg_scale=float(cfg_scale),
            motion_score=int(motion_score),
            seed=int(seed),
            quantization=quantization,
            vae_tiling=vae_tiling,
            offload_text_encoder_after_encode=bool(offload_text_encoder_after_encode),
            use_sage_attention=bool(use_sage_attention),
            generate_audio=bool(generate_audio),
            audio_prompt=audio_prompt,
        )
        result = service.generate(request, on_progress=on_progress)
    except Exception as exc:
        details = {"status": "error", "error": str(exc), "events": events}
        return None, f"Error: {exc}", details

    details = {
        "status": "complete",
        "message": result.message,
        "output_path": result.output_path,
        "receipt_path": result.receipt_path,
        "width": result.width,
        "height": result.height,
        "frames": result.frames,
        "fps": result.fps,
        "attention_backend": result.attention_backend,
        "quantization": result.quantization,
        "vae_tiling": result.vae_tiling,
        "timings": result.timings,
        "events": result.progress or events,
    }
    return result.output_path, result.message, details


def register_sana_video(registry: WebRegistry) -> None:
    @registry.tab("Sana Video", order=3)
    def build(ctx: AppContext, _tab=None) -> None:
        service = _service(ctx)
        gr.Markdown(
            "Sana Video is experimental in AIWF. Use short 480p runs first, then raise frames or resolution after one clean receipt."
        )
        gr.Markdown(service.status_markdown())
        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Prompt",
                    lines=4,
                    placeholder="cinematic portrait, subtle camera motion, natural light",
                )
                negative_prompt = gr.Textbox(
                    label="Negative prompt",
                    lines=2,
                    placeholder="low quality, distorted, flicker, watermark",
                )
                pipeline = gr.Radio(
                    label="Pipeline",
                    choices=SANA_VIDEO_PIPELINE_CHOICES,
                    value=SANA_VIDEO_PIPELINE_T2V,
                )
                source_image = gr.Image(
                    label="Source image for image-to-video",
                    type="filepath",
                    sources=["upload", "clipboard"],
                )
                model_path = gr.Textbox(
                    label="Model folder",
                    value=str(service.default_model_path()),
                    info="Diffusers folder containing model_index.json.",
                )
                with gr.Row():
                    width = gr.Slider(256, 1280, value=832, step=32, label="Width")
                    height = gr.Slider(256, 1280, value=480, step=32, label="Height")
                with gr.Row():
                    frames = gr.Slider(5, 257, value=81, step=4, label="Frames")
                    fps = gr.Slider(1, 60, value=16, step=1, label="FPS")
                with gr.Row():
                    steps = gr.Slider(1, 100, value=50, step=1, label="Steps")
                    cfg_scale = gr.Slider(0, 20, value=6.0, step=0.5, label="CFG scale")
                with gr.Row():
                    motion_score = gr.Slider(0, 100, value=30, step=1, label="Motion score")
                    seed = gr.Number(value=-1, precision=0, label="Seed")
                with gr.Accordion("Runtime", open=True):
                    quantization = gr.Dropdown(
                        label="Quantization",
                        choices=SANA_VIDEO_QUANTIZATION_CHOICES,
                        value=SANA_VIDEO_QUANTIZATION_AUTO,
                    )
                    vae_tiling = gr.Dropdown(
                        label="VAE tiling",
                        choices=SANA_VIDEO_VAE_TILING_CHOICES,
                        value=SANA_VIDEO_VAE_TILING_AUTO,
                    )
                    offload_text_encoder = gr.Checkbox(value=True, label="Offload text encoder after prompt encode")
                    use_sage_attention = gr.Checkbox(value=True, label="Use Sage attention when available")
                with gr.Accordion("Audio post-process", open=False):
                    generate_audio = gr.Checkbox(value=False, label="Add audio after video")
                    audio_prompt = gr.Textbox(label="Audio prompt", lines=2)
                generate = gr.Button("Generate Sana video", variant="primary")
            with gr.Column(scale=1):
                output = gr.Video(label="Output")
                status = gr.Markdown("Ready.")
                details = gr.JSON(label="Generation details")

        generate.click(
            lambda *args: _run_sana_video(ctx, *args),
            inputs=[
                prompt,
                negative_prompt,
                pipeline,
                source_image,
                model_path,
                width,
                height,
                frames,
                fps,
                steps,
                cfg_scale,
                motion_score,
                seed,
                quantization,
                vae_tiling,
                offload_text_encoder,
                use_sage_attention,
                generate_audio,
                audio_prompt,
            ],
            outputs=[output, status, details],
        )
