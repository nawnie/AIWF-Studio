from __future__ import annotations

import gradio as gr

from aiwf.web.studio.helpers import align_compare_pair


def toggle_compare(showing: bool, before, after) -> tuple:
    if before is None or after is None:
        raise gr.Error("Generate an image first to compare.")
    aligned_before, aligned_after = align_compare_pair(before, after)
    new_show = not showing
    if new_show:
        return (
            True,
            gr.update(visible=False),
            gr.update(visible=True, value=(aligned_before, aligned_after)),
            gr.update(value="Hide compare"),
        )
    return (
        False,
        gr.update(visible=True, value=after),
        gr.update(visible=False),
        gr.update(value="Compare"),
    )