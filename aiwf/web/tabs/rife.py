from __future__ import annotations

import gradio as gr

from aiwf.bootstrap import AppContext
from aiwf.core.domain.rife import RifeOptions
from aiwf.infrastructure.rife import RifeUnavailable
from aiwf.services.rife import RifeService
from aiwf.web.registry import WebRegistry

_SERVICES: dict[int, RifeService] = {}


def _service(ctx: AppContext) -> RifeService:
    svc = _SERVICES.get(id(ctx))
    if svc is None:
        svc = RifeService(ctx.flags, ctx.settings, ctx.generation.backend.devices, supervisor=ctx.supervisor)
        _SERVICES[id(ctx)] = svc
    return svc


def register_rife(registry: WebRegistry) -> None:
    @registry.tab("RIFE", order=23)
    def build(ctx: AppContext, tab: gr.Tab | None = None) -> None:
        service = _service(ctx)
        ckpts = service.list_checkpoints()
        default_ckpt = service.default_checkpoint()

        with gr.Column(elem_classes=["aiwf-rife", "aiwf-video"]):
            with gr.Column(elem_classes=["aiwf-page-header"]):
                gr.Markdown("RIFE", elem_classes=["aiwf-section-label"])
                gr.Markdown(
                    "Interpolate video frames with Practical-RIFE.",
                    elem_classes=["aiwf-page-intro"],
                )
                gr.Markdown(service.folder_help(), elem_classes=["aiwf-page-path"])

            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=340, elem_classes=["aiwf-panel"]):
                    source = gr.Video(label="Input video", sources=["upload"])
                    ckpt = gr.Dropdown(
                        label="RIFE model",
                        choices=ckpts,
                        value=default_ckpt if default_ckpt in ckpts else (ckpts[0] if ckpts else None),
                    )
                    multiplier = gr.Slider(2, 4, value=2, step=1, label="Multiplier", info="2x doubles FPS")
                    scale_factor = gr.Dropdown(
                        label="Scale factor",
                        choices=[("Full res", 1.0), ("Half res (less VRAM)", 0.5), ("Quarter res", 0.25)],
                        value=1.0,
                    )
                    with gr.Row():
                        fast_mode = gr.Checkbox(label="Fast mode", value=True)
                        ensemble = gr.Checkbox(label="Ensemble (quality)", value=True)
                    chunk_frames = gr.Slider(
                        2, 64, value=16, step=1, label="Input frames per chunk",
                        info="Lower values reduce RAM/VRAM. The model remains loaded between chunks.",
                    )
                    clear_cache = gr.Slider(1, 50, value=10, step=1, label="Clear CUDA cache every N frame pairs")
                    max_frames = gr.Number(
                        label="Max input frames (0 = all)",
                        value=0,
                        precision=0,
                        info="Cap for quick tests on long files",
                    )
                    refresh = gr.Button("Refresh models", elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"])
                    run = gr.Button("Interpolate", variant="primary", elem_classes=["aiwf-generate-btn"])

                with gr.Column(scale=1, min_width=340, elem_classes=["aiwf-panel"]):
                    output = gr.Video(label="Output video", interactive=False)
                    status = gr.Markdown("**Ready** — upload a video (2+ frames)", elem_classes=["aiwf-status-bar"])
                    details = gr.Textbox(label="Details", lines=3, interactive=False, elem_classes=["aiwf-gen-info"])

        def _refresh():
            choices = service.list_checkpoints()
            default = service.default_checkpoint()
            return gr.update(choices=choices, value=default if default in choices else (choices[0] if choices else None))

        def _run(video, ckpt_name, mult, scale, fast, ens, chunk_n, cache_n, cap):
            if video is None:
                raise gr.Error("Upload an input video first.")
            path = video if isinstance(video, str) else getattr(video, "name", None) or getattr(video, "path", None)
            if not path:
                raise gr.Error("Could not read the uploaded video path.")
            cap_i = int(cap or 0)
            options = RifeOptions(
                ckpt_name=str(ckpt_name or service.default_checkpoint()),
                multiplier=int(mult),
                scale_factor=float(scale),
                fast_mode=bool(fast),
                ensemble=bool(ens),
                clear_cache_every_n_frames=int(cache_n),
                chunk_input_frames=int(chunk_n),
                max_input_frames=None if cap_i <= 0 else cap_i,
            )
            progress_state = {"step": 0, "total": 1}

            def on_progress(step, total):
                progress_state["step"] = step
                progress_state["total"] = max(1, total)

            try:
                result = service.interpolate(path, options, on_progress=on_progress)
            except RifeUnavailable as exc:
                raise gr.Error(str(exc)) from exc

            status_text = f"**Done** — {result.message}"
            return result.output_path, status_text, result.infotext

        refresh.click(_refresh, outputs=[ckpt])
        run.click(
            _run,
            inputs=[source, ckpt, multiplier, scale_factor, fast_mode, ensemble, chunk_frames, clear_cache, max_frames],
            outputs=[output, status, details],
        )
