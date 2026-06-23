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
    selected_image = evt.value
    if isinstance(selected_image, dict):
        selected_image = selected_image.get("image") or selected_image.get("value")

    seed_update = gr.update()
    width_update = gr.update()
    height_update = gr.update()

    if getattr(settings, "send_seed_on_click", True) and seeds:
        idx = evt.index if isinstance(evt.index, int) else (evt.index[0] if evt.index else 0)
        if 0 <= idx < len(seeds):
            seed_update = gr.update(value=seeds[idx])

    if getattr(settings, "send_size_on_click", True) and isinstance(selected_image, PILImage.Image):
        width_update = gr.update(value=selected_image.width)
        height_update = gr.update(value=selected_image.height)

    return selected_image, seed_update, width_update, height_update
