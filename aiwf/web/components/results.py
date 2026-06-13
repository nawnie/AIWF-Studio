from __future__ import annotations

import gradio as gr
from PIL import Image


def result_image(label: str = "Output") -> gr.Image:
    """Single-image output sized to fit the full generation."""
    return gr.Image(
        label=label,
        type="pil",
        interactive=False,
        elem_classes=["aiwf-result-image"],
    )


def results_gallery(label: str = "All results", *, columns: int = 2, visible: bool = True) -> gr.Gallery:
    """Gallery that shows complete images without a scrollable thumbnail strip."""
    return gr.Gallery(
        label=label,
        columns=columns,
        object_fit="contain",
        height=None,
        fit_columns=True,
        allow_preview=True,
        visible=visible,
        elem_classes=["aiwf-results-gallery"],
    )


def format_generation_outputs(
    images: list[Image.Image],
    infotext: str,
    status: str,
) -> tuple[Image.Image | None, list[Image.Image], str, str]:
    """Primary full-size preview plus optional gallery list for batches."""
    primary = images[0] if images else None
    return primary, images, infotext, status