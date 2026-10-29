from __future__ import annotations

import gradio as gr

from aiwf.bootstrap import AppContext
from aiwf.core.domain.segment import SegmentRequest
from aiwf.core.domain.segment_presets import CUSTOM_SEGMENT_PRESET_ID, resolve_segment_text_prompt
from aiwf.infrastructure.diffusers.mask import editor_from_mask, inpaint_session_background, prepare_outpaint
from aiwf.web.studio.helpers import load_uploaded_image, mode_from_label, segment_source_image
from aiwf.web.studio.mode_ui import apply_mode_ui
from aiwf.web.studio.session import StudioSession


def handle_upload(
    ctx: AppContext,
    session: StudioSession,
    file_obj,
    mode_label: str,
    editing_mask: bool,
    current_ckpt: str | None = None,
) -> tuple:
    image = load_uploaded_image(file_obj)
    if image is None:
        return gr.update(), gr.update(), editing_mask, *apply_mode_ui(
            ctx, mode_label, editing_mask, current_ckpt=current_ckpt
        )

    mode = mode_from_label(mode_label)
    if mode == "inpaint":
        editor_val = {"background": image, "layers": [], "composite": None}
        session.sam_mask = None
        session.inpaint.original = image.copy()
        session.inpaint.mask = None
        mode_ui = apply_mode_ui(ctx, mode_label, True, current_ckpt=current_ckpt)
        return gr.update(value=editor_val), gr.update(value=None), True, *mode_ui
    if mode == "img2img":
        mode_ui = apply_mode_ui(ctx, mode_label, False, current_ckpt=current_ckpt)
        return gr.update(), gr.update(value=image), False, *mode_ui
    return gr.update(), gr.update(), editing_mask, *apply_mode_ui(
        ctx, mode_label, editing_mask, current_ckpt=current_ckpt
    )


def start_mask_edit(
    ctx: AppContext,
    session: StudioSession,
    mode_label: str,
    workspace_img,
    source_choice: str,
    editor_value,
    current_ckpt: str | None = None,
) -> tuple:
    background = inpaint_session_background(source_choice, workspace_img, editor_value, session.inpaint_session)
    if background is None:
        raise gr.Error("Upload an image first.")
    editor_val = {"background": background, "layers": [], "composite": None}
    if session.inpaint.mask is not None:
        editor_val = editor_from_mask(background, session.inpaint.mask)
    session.sam_mask = session.inpaint.mask
    mode_ui = apply_mode_ui(ctx, mode_label, True, current_ckpt=current_ckpt)
    return gr.update(value=editor_val), True, *mode_ui


def run_sam(
    ctx: AppContext,
    session: StudioSession,
    preset_id: str,
    custom_prompt: str,
    model_id: str,
    threshold,
    mask_index,
    dilation,
    source_image,
    editor_value,
    mode_label: str,
    current_ckpt: str | None = None,
) -> tuple:
    segment_source = segment_source_image(source_image, editor_value)
    if segment_source is None:
        if session.inpaint.original is not None:
            segment_source = session.inpaint.original
        else:
            raise gr.Error("Upload or generate an image first.")
    prompt = resolve_segment_text_prompt(preset_id, custom_prompt)
    if not prompt:
        raise gr.Error("Choose what to mask, or select Custom and enter a prompt.")
    request = SegmentRequest(
        text_prompt=prompt,
        box_threshold=float(threshold),
        mask_index=int(mask_index),
        dilation=int(dilation or 0),
    )
    mask, preview, candidates, message = ctx.segment.segment(
        segment_source, request, model_id=model_id or None
    )
    session.sam_mask = mask
    session.inpaint.mask = mask.copy()
    editor_val = editor_from_mask(segment_source, mask)
    gallery = gr.update(value=[preview, *candidates], visible=True)
    mode_ui = apply_mode_ui(ctx, mode_label, True, current_ckpt=current_ckpt)
    return (
        gr.update(value=editor_val),
        True,
        gallery,
        f"**SAM:** {message}",
        gr.update(value="Only masked"),
        gr.update(value=32),
        gr.update(value="latent noise"),
        *mode_ui,
    )


def prepare_outpaint_canvas(
    ctx: AppContext,
    session: StudioSession,
    source_image,
    editor_value,
    mode_label: str,
    left,
    right,
    up,
    down,
    fill: str,
    overlap,
) -> tuple:
    src = segment_source_image(source_image, editor_value)
    if src is None:
        raise gr.Error("Upload or generate an image first.")
    try:
        padded, mask = prepare_outpaint(
            src,
            left=int(left),
            right=int(right),
            up=int(up),
            down=int(down),
            fill=fill,
            mask_overlap=int(overlap),
        )
    except ValueError as exc:
        raise gr.Error(str(exc))
    session.sam_mask = mask
    session.inpaint.original = padded.copy()
    session.inpaint.mask = mask.copy()
    editor_val = {"background": padded, "layers": [], "composite": None}
    mode_ui = apply_mode_ui(ctx, mode_label, True)
    new_px = padded.size
    return (
        gr.update(value=editor_val),
        True,
        f"**Outpaint ready** — canvas {new_px[0]}×{new_px[1]}. Set a prompt and Generate.",
        *mode_ui,
    )


def use_result_as_source(
    ctx: AppContext,
    session: StudioSession,
    result_image,
    mode_label: str,
) -> tuple:
    if result_image is None:
        raise gr.Error("Generate an image first.")
    mode = mode_from_label(mode_label)
    if mode == "inpaint":
        editor_val = {"background": result_image, "layers": [], "composite": None}
        if session.inpaint.mask is not None:
            editor_val = editor_from_mask(result_image, session.inpaint.mask)
        session.sam_mask = session.inpaint.mask
        mode_ui = apply_mode_ui(ctx, mode_label, True)
        return gr.update(value=editor_val), gr.update(value=None), True, *mode_ui
    mode_ui = apply_mode_ui(ctx, mode_label, False)
    return gr.update(), gr.update(value=result_image), False, *mode_ui


def on_sam_preset_change(preset_id: str):
    return gr.update(visible=preset_id == CUSTOM_SEGMENT_PRESET_ID)