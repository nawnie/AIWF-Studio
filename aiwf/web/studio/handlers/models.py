from __future__ import annotations

import gradio as gr

from aiwf.bootstrap import AppContext
from aiwf.web.components.checkpoints import format_model_status, refresh_checkpoints
from aiwf.web.studio.helpers import mode_from_label


def refresh_checkpoint_models(ctx: AppContext, mode_label: str, current_ckpt: str | None) -> tuple:
    mode_from_label(mode_label)
    update, new_map = refresh_checkpoints(ctx, rescan=True, current_value=current_ckpt)
    return update, format_model_status(ctx), new_map


def on_checkpoint_change(ctx: AppContext, ckpt_title: str | None, ckpt_map: dict) -> gr.Update:
    if not ckpt_title or not ckpt_map:
        return gr.update()
    ckpt_id = ckpt_map.get(ckpt_title)
    if ckpt_id is None:
        return gr.update(value=f"**Error:** unknown checkpoint {ckpt_title}")
    try:
        ctx.generation.load_checkpoint(ckpt_id)
        base_status = format_model_status(ctx)
        return gr.update(value=f"**Loaded:** {ckpt_title}\n\n{base_status}")
    except Exception as exc:
        return gr.update(value=f"**Load failed:** {ckpt_title} — {exc}")


def refresh_vaes(ctx: AppContext, current=None):
    vaes = ctx.generation.refresh_vae_catalog()
    choices = [("Automatic", None)] + [(item.title, item.id) for item in vaes]
    ids = {item.id for item in vaes}
    value = current if current in ids else None
    return gr.update(choices=choices, value=value)


def refresh_sam_models(ctx: AppContext, current=None):
    models = ctx.segment.refresh_models()
    ids = {model.id for model in models}
    value = current if current in ids else (models[0].id if models else None)
    return gr.update(choices=[(model.title, model.id) for model in models], value=value)


def cn_models_update(ctx: AppContext, current=None):
    models = ctx.controlnet.list_models()
    ids = {m.id for m in models}
    value = current if current in ids else (models[0].id if models else None)
    return gr.update(choices=[(m.title, m.id) for m in models], value=value)


def cn_preview(ctx: AppContext, image, module: str, threshold_a, threshold_b):
    if image is None:
        raise gr.Error("Upload a control image first.")
    return ctx.controlnet.preprocess(
        image,
        module or "none",
        processor_res=512,
        threshold_a=float(threshold_a),
        threshold_b=float(threshold_b),
    )


def on_studio_tab_select(ctx: AppContext, mode_label: str, cn_current, current_ckpt: str | None) -> tuple:
    from aiwf.web.studio.handlers.prompts import pnginfo_pending_hint

    mode_from_label(mode_label)
    ckpt_update, new_map = refresh_checkpoints(ctx, rescan=True, current_value=current_ckpt)
    return (
        ckpt_update,
        format_model_status(ctx),
        new_map,
        cn_models_update(ctx, cn_current),
        pnginfo_pending_hint(ctx),
    )