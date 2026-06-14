from __future__ import annotations

import gradio as gr

from aiwf.bootstrap import AppContext
from aiwf.web.components.checkpoints import refresh_checkpoints
from aiwf.web.studio.catalogs import StudioCatalogs
from aiwf.web.studio.constants import EMPTY_CANVAS, MODE_TITLES, TOOLBAR_HINTS
from aiwf.web.studio.helpers import mode_from_label


def apply_mode_ui(
    ctx: AppContext,
    mode_label: str,
    editing_mask: bool,
    *,
    current_ckpt: str | None = None,
    hide_empty: bool = False,
) -> tuple:
    mode = mode_from_label(mode_label)
    is_txt = mode == "txt2img"
    is_img = mode == "img2img"
    is_inpaint = mode == "inpaint"
    inpaint_editing = is_inpaint and editing_mask

    ckpt_update, new_map = refresh_checkpoints(ctx, current_value=current_ckpt)

    if is_txt:
        hint = TOOLBAR_HINTS["txt2img"]
        empty = EMPTY_CANVAS["txt2img"]
        show_empty = True
    elif is_img:
        hint = TOOLBAR_HINTS["img2img"]
        empty = EMPTY_CANVAS["img2img"]
        show_empty = True
    elif inpaint_editing:
        hint = TOOLBAR_HINTS["inpaint_edit"]
        empty = EMPTY_CANVAS["inpaint_edit"]
        show_empty = False
    else:
        hint = TOOLBAR_HINTS["inpaint_result"]
        empty = EMPTY_CANVAS["inpaint_result"]
        show_empty = True

    if hide_empty:
        show_empty = False

    return (
        gr.update(value=MODE_TITLES[mode]),
        gr.update(visible=is_txt),
        gr.update(visible=is_txt),
        gr.update(visible=is_img),
        gr.update(visible=is_inpaint, open=is_inpaint),
        gr.update(open=is_img),
        gr.update(visible=is_img or is_inpaint),
        gr.update(visible=is_img or is_inpaint),
        gr.update(visible=(is_img or is_inpaint) and not inpaint_editing),
        gr.update(visible=inpaint_editing),
        gr.update(visible=not inpaint_editing),
        gr.update(visible=inpaint_editing),
        gr.update(value=hint),
        gr.update(value=empty, visible=show_empty),
        gr.update(elem_classes=["aiwf-workspace", f"aiwf-mode-{mode}"]),
        gr.update(elem_classes=["aiwf-studio", f"aiwf-mode-{mode}"]),
        ckpt_update,
        new_map,
    )


def on_mode_change(
    ctx: AppContext,
    mode_label: str,
    editing_mask: bool,
    current_ckpt: str | None = None,
) -> tuple:
    mode = mode_from_label(mode_label)
    if mode == "inpaint":
        editing_mask = True
    else:
        editing_mask = False
    return (
        *apply_mode_ui(ctx, mode_label, editing_mask, current_ckpt=current_ckpt),
        editing_mask,
        False,
        gr.update(visible=False),
        gr.update(visible=False, value="Compare"),
    )