from __future__ import annotations

import gradio as gr
from PIL import Image as PILImage


def save_bad_image(ctx, image, infotext: str | None) -> str:
    if image is None:
        raise gr.Error("Generate or select an image first.")
    record = ctx.failure_archive.archive_bad_image(
        image,
        infotext=infotext or "",
        note="Marked from Image tab",
    )
    if not record.ok:
        return f"**Failure gallery** -- saved with archive warnings: {record.archive_dir}"
    return f"**Failure gallery** -- saved bad result: {record.archive_dir}"


def on_gallery_select(settings, evt: gr.SelectData, seeds: list, _img_w: int | None = None, _img_h: int | None = None):
    # Gradio gallery-select payloads vary across versions:
    #   {'image': {'path':.., 'url':..}, 'caption':..}  /  {'path':.., 'url':..}  /  a PIL image.
    # The image output component needs a PIL/ndarray/path/None — returning a raw
    # dict raises ComponentProcessingError, so normalise to a path string here.
    selected_image = evt.value
    if isinstance(selected_image, dict):
        inner = selected_image.get("image", selected_image)
        if isinstance(inner, dict):
            selected_image = inner.get("path") or inner.get("url")
        else:
            selected_image = inner

    seed_update = gr.update()
    width_update = gr.update()
    height_update = gr.update()

    if getattr(settings, "send_seed_on_click", True) and seeds:
        idx = evt.index if isinstance(evt.index, int) else (evt.index[0] if evt.index else 0)
        if 0 <= idx < len(seeds):
            seed_update = gr.update(value=seeds[idx])

    if getattr(settings, "send_size_on_click", True):
        size: tuple[int, int] | None = None
        if isinstance(selected_image, PILImage.Image):
            size = (selected_image.width, selected_image.height)
        elif isinstance(selected_image, str) and selected_image and not selected_image.startswith("http"):
            try:
                with PILImage.open(selected_image) as opened:
                    size = opened.size
            except Exception:
                size = None
        if size:
            width_update = gr.update(value=size[0])
            height_update = gr.update(value=size[1])

    return selected_image, seed_update, width_update, height_update
