from __future__ import annotations

import queue
import threading

import gradio as gr

from aiwf.bootstrap import AppContext
from aiwf.core.domain.enhance import RestoreOptions
from aiwf.core.domain.faceswap import FaceSwapOptions
from aiwf.infrastructure.faceswap import FaceSwapUnavailable
from aiwf.web.components.results import result_image
from aiwf.web.registry import WebRegistry


def _download_choices(ctx: AppContext) -> list[tuple[str, str]]:
    choices = []
    for item in ctx.faceswap.list_downloadable():
        mark = "  ✓ installed" if ctx.faceswap.is_installed(item) else ""
        choices.append((f"{item.title} · {item.size_mb}MB{mark}", item.key))
    return choices


def _status_md(ctx: AppContext) -> str:
    models = ctx.faceswap.list_models()
    if not models:
        return "_No face-swap model installed. Download `inswapper_128` below._"
    return "**Installed:** " + ", ".join(m.title for m in models)


def _restorer_choices(ctx: AppContext) -> list[tuple[str, str]]:
    return [(m.title, m.id) for m in ctx.enhance.list_restorers()]


def register_faceswap(registry: WebRegistry) -> None:
    @registry.tab("Face Swap", order=20)
    def build(ctx: AppContext, tab: gr.Tab | None = None) -> None:
        service = ctx.faceswap

        with gr.Column(elem_classes=["aiwf-faceswap"]):
            with gr.Column(elem_classes=["aiwf-page-header"]):
                gr.Markdown("Face Swap (ReActor)", elem_classes=["aiwf-section-label"])
                gr.Markdown(
                    "Swap a face from a **source** image onto a **target** image. "
                    "Needs the optional `insightface` + `onnxruntime` packages and the "
                    "`inswapper_128` model. Use responsibly and only with consent.",
                    elem_classes=["aiwf-page-intro"],
                )
                gr.Markdown(service.folder_help(), elem_classes=["aiwf-page-path"])

            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=300, elem_classes=["aiwf-panel"]):
                    source_image = gr.Image(label="Source face", type="pil", sources=["upload", "clipboard"])
                    source_index = gr.Number(value=0, precision=0, label="Source face #", info="0 = first face")
                with gr.Column(scale=1, min_width=300, elem_classes=["aiwf-panel"]):
                    target_image = gr.Image(label="Target image", type="pil", sources=["upload", "clipboard"])
                    target_index = gr.Number(
                        value=-1, precision=0, label="Target face #", info="-1 = swap every face"
                    )

            with gr.Column(elem_classes=["aiwf-panel"]):
                with gr.Row():
                    restore_face = gr.Checkbox(label="Restore face after swap", value=True)
                    _restorers = _restorer_choices(ctx)
                    restorer = gr.Dropdown(
                        label="Restorer",
                        choices=_restorers,
                        value=(_restorers[0][1] if _restorers else None),
                    )
                with gr.Row():
                    restore_visibility = gr.Slider(0, 1, value=1.0, step=0.05, label="Restore visibility")
                    codeformer_weight = gr.Slider(0, 1, value=0.5, step=0.05, label="CodeFormer weight")
                run_btn = gr.Button("Swap face", variant="primary", elem_classes=["aiwf-generate-btn"])
                status = gr.Markdown(_status_md(ctx), elem_classes=["aiwf-status-bar"])

            result = result_image(label="Result")

            with gr.Accordion("Download model", open=False, elem_classes=["aiwf-prompt-tools"]):
                dl_select = gr.Dropdown(
                    label="Face-swap model", choices=_download_choices(ctx), value=None
                )
                with gr.Row():
                    dl_btn = gr.Button("Download", variant="primary")
                    dl_refresh = gr.Button("Refresh", elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"])
                dl_status = gr.Markdown("", elem_classes=["aiwf-status-bar"])

        def _run(src, tgt, s_idx, t_idx, do_restore, restorer_id, visibility, cf_weight):
            if src is None or tgt is None:
                raise gr.Error("Upload both a source face and a target image.")
            options = FaceSwapOptions(
                source_face_index=max(0, int(s_idx or 0)),
                target_face_index=int(t_idx if t_idx is not None else -1),
                restore_face=bool(do_restore),
                restorer_id=restorer_id,
                restore_visibility=float(visibility),
                codeformer_weight=float(cf_weight),
            )

            restore_fn = None
            if do_restore and restorer_id:
                def restore_fn(image):
                    return ctx.enhance.restore(
                        image,
                        RestoreOptions(
                            model_id=restorer_id,
                            visibility=float(visibility),
                            codeformer_weight=float(cf_weight),
                        ),
                    )

            try:
                output = service.swap(tgt, src, options, restore_fn=restore_fn)
            except FaceSwapUnavailable as exc:
                raise gr.Error(str(exc))
            return output, "**Face swap complete.**"

        run_btn.click(
            _run,
            inputs=[
                source_image,
                target_image,
                source_index,
                target_index,
                restore_face,
                restorer,
                restore_visibility,
                codeformer_weight,
            ],
            outputs=[result, status],
            show_progress="minimal",
        )

        def _refresh_dl():
            return gr.update(choices=_download_choices(ctx)), _status_md(ctx)

        dl_refresh.click(_refresh_dl, outputs=[dl_select, status], show_progress=False)

        def _download(key):
            if not key:
                raise gr.Error("Pick a model to download.")
            item = service.find_downloadable(key)
            if item is None:
                raise gr.Error("Unknown model.")
            if service.is_installed(item):
                yield f"**{item.title}** already installed.", _status_md(ctx)
                return

            progress_q: queue.Queue = queue.Queue()
            result_box: dict = {}

            def worker():
                try:
                    service.download_model(key, on_progress=lambda d, t: progress_q.put((d, t)))
                    result_box["ok"] = True
                except Exception as exc:
                    result_box["error"] = str(exc)
                finally:
                    progress_q.put(None)

            threading.Thread(target=worker, daemon=True).start()
            yield f"**Downloading {item.title}…**", gr.update()

            while True:
                update = progress_q.get()
                if update is None:
                    break
                done, total = update
                if total:
                    pct = int(100 * done / total)
                    yield f"**Downloading {item.title}…** {pct}% ({done / 1_000_000:.0f} MB)", gr.update()

            if result_box.get("error"):
                yield f"**Download failed** — {result_box['error']}", _status_md(ctx)
            else:
                yield f"**{item.title} installed.**", _status_md(ctx)

        dl_btn.click(_download, inputs=[dl_select], outputs=[dl_status, status], show_progress="minimal")

        if tab is not None:
            tab.select(lambda: _status_md(ctx), outputs=[status], show_progress=False)
