from __future__ import annotations

import gradio as gr

from aiwf.bootstrap import AppContext
from aiwf.core.tags import parse_tags_from_params
from aiwf.web.components.infotext import read_png_infotext, store_for_tabs
from aiwf.web.registry import WebRegistry


def register_pnginfo(registry: WebRegistry) -> None:
    @registry.tab("PNG Info", order=40)
    def build(ctx: AppContext, tab: gr.Tab | None = None) -> None:
        with gr.Column(elem_classes=["aiwf-settings"]):
            with gr.Column(elem_classes=["aiwf-page-header"]):
                gr.Markdown("PNG metadata", elem_classes=["aiwf-section-label"])
                gr.Markdown(
                    "Read generation parameters from a PNG and send them to Studio.",
                    elem_classes=["aiwf-page-intro"],
                )

            with gr.Row(equal_height=False):
                with gr.Column(elem_classes=["aiwf-panel"]):
                    source = gr.Image(label="Source image", type="pil", sources=["upload", "clipboard"])
                    with gr.Row():
                        read_btn = gr.Button("Read from image", variant="primary", elem_classes=["aiwf-generate-btn"])
                        send_btn = gr.Button("Send to Studio", elem_classes=["aiwf-btn-ghost"])
                with gr.Column(elem_classes=["aiwf-panel"]):
                    parameters = gr.Textbox(label="Parameters", lines=14, elem_classes=["aiwf-gen-info"])
                    status = gr.Markdown("**Ready** — load an image to begin.", elem_classes=["aiwf-status-bar"])

        def do_read(image):
            if image is None:
                return "", "**Ready** — load an image to begin."
            text, params = read_png_infotext(image)
            if not text:
                return "", "**Error** — no generation parameters in this image."
            tags = parse_tags_from_params(params)
            status = "**Loaded** — parameters extracted from PNG."
            if tags:
                status += "  \n**Tags** " + " · ".join(f"`#{tag}`" for tag in tags)
            return text, status

        def do_send(text: str | None):
            clean = (text or "").strip()
            if not clean:
                raise gr.Error("Load parameters from an image first.")
            from aiwf.core.infotext import parse_infotext

            params = parse_infotext(clean)
            return store_for_tabs(ctx, clean, params)

        read_btn.click(do_read, inputs=[source], outputs=[parameters, status])
        source.change(do_read, inputs=[source], outputs=[parameters, status])
        send_btn.click(
            do_send,
            inputs=[parameters],
            outputs=[status],
            js="""
            (text) => {
                setTimeout(() => {
                    const studioTab = [...document.querySelectorAll(".aiwf-nav-tabs .tab-nav button")]
                        .find((button) => button.textContent.trim() === "Studio");
                    if (studioTab) {
                        studioTab.click();
                    }
                }, 120);
                return [text];
            }
            """,
        )
