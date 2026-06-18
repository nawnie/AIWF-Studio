from __future__ import annotations

import gradio as gr

from aiwf.bootstrap import AppContext
from aiwf.core.domain.audio import AudioGenerationOptions
from aiwf.services.audio import AudioGenerationService, AudioUnavailable
from aiwf.web.registry import WebRegistry

_SERVICES: dict[int, AudioGenerationService] = {}


def _service(ctx: AppContext) -> AudioGenerationService:
    svc = _SERVICES.get(id(ctx))
    if svc is None:
        svc = AudioGenerationService(
            ctx.flags,
            ctx.settings,
            ctx.generation.backend.devices,
            supervisor=ctx.supervisor,
        )
        _SERVICES[id(ctx)] = svc
    return svc


def register_audio(registry: WebRegistry) -> None:
    @registry.tab("Audio", order=24)
    def build(ctx: AppContext, tab: gr.Tab | None = None) -> None:
        service = _service(ctx)
        music_models = service.music_model_choices()
        sfx_models = service.sfx_model_choices()

        with gr.Column(elem_classes=["aiwf-audio", "aiwf-video"]):
            with gr.Column(elem_classes=["aiwf-page-header"]):
                gr.Markdown("Audio", elem_classes=["aiwf-section-label"])
                gr.Markdown("Generate local music or sound effects, optionally muxed into a video.", elem_classes=["aiwf-page-intro"])
                gr.Markdown(service.folder_help(), elem_classes=["aiwf-page-path"])

            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=340, elem_classes=["aiwf-panel"]):
                    prompt = gr.Textbox(
                        label="Audio prompt",
                        lines=4,
                        placeholder="cinematic ambient score, soft percussion, warm synth pads",
                    )
                    source_video = gr.Video(
                        label="Optional target video",
                        sources=["upload"],
                        interactive=True,
                    )
                    with gr.Row():
                        kind = gr.Radio(
                            label="Type",
                            choices=[("Music", "music"), ("Sound effects", "sfx")],
                            value="music",
                        )
                        model = gr.Dropdown(
                            label="Model",
                            choices=music_models,
                            value=music_models[0][1] if music_models else "facebook/musicgen-small",
                            allow_custom_value=True,
                        )
                    with gr.Row():
                        duration = gr.Slider(1, 120, value=8, step=1, label="Duration (seconds)")
                        seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                    with gr.Row():
                        temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.05, label="Temperature")
                        cfg = gr.Slider(0.1, 10.0, value=3.0, step=0.1, label="Guidance")
                    generate = gr.Button("Generate audio", variant="primary", elem_classes=["aiwf-generate-btn"])

                with gr.Column(scale=1, min_width=340, elem_classes=["aiwf-panel"]):
                    audio_out = gr.Audio(label="Generated audio", type="filepath", interactive=False)
                    video_out = gr.Video(label="Video with audio", interactive=False)
                    status = gr.Markdown("**Ready**", elem_classes=["aiwf-status-bar"])
                    details = gr.Textbox(label="Details", lines=4, interactive=False, elem_classes=["aiwf-gen-info"])

        def _sync_kind(kind_value):
            choices = sfx_models if kind_value == "sfx" else music_models
            fallback = "facebook/audiogen-medium" if kind_value == "sfx" else "facebook/musicgen-small"
            return gr.update(choices=choices, value=choices[0][1] if choices else fallback)

        kind.change(_sync_kind, inputs=[kind], outputs=[model], show_progress=False)

        def _run(prompt_v, video_v, kind_v, model_v, duration_v, seed_v, temperature_v, cfg_v):
            video_path = None
            if video_v is not None:
                video_path = video_v if isinstance(video_v, str) else getattr(video_v, "name", None) or getattr(video_v, "path", None)
            options = AudioGenerationOptions(
                prompt=prompt_v or "",
                kind=str(kind_v or "music"),
                model_id=str(model_v or ("facebook/audiogen-medium" if kind_v == "sfx" else "facebook/musicgen-small")),
                duration_seconds=float(duration_v or 8),
                temperature=float(temperature_v or 1.0),
                cfg_coef=float(cfg_v or 3.0),
                seed=int(seed_v if seed_v is not None else -1),
            )
            try:
                if video_path:
                    audio, muxed = service.generate_and_mux(video_path, options, duration_seconds=float(duration_v or 8))
                    return audio.output_path, muxed.output_path, "**Audio mux complete.**", audio.infotext + "\n" + muxed.infotext
                audio = service.generate(options)
                return audio.output_path, None, "**Audio complete.**", audio.infotext
            except AudioUnavailable as exc:
                raise gr.Error(str(exc)) from exc

        generate.click(
            _run,
            inputs=[prompt, source_video, kind, model, duration, seed, temperature, cfg],
            outputs=[audio_out, video_out, status, details],
            show_progress="minimal",
        )
