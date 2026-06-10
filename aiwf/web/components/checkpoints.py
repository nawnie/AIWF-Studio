from __future__ import annotations

import gradio as gr

from aiwf.bootstrap import AppContext
from aiwf.core.domain.models import Checkpoint


def _checkpoint_choices(checkpoints: list[Checkpoint], *, inpaint_only: bool = False) -> list[tuple[str, str]]:
    choices = []
    for checkpoint in checkpoints:
        is_inpaint = checkpoint.kind == "inpaint"
        if inpaint_only and not is_inpaint:
            continue
        label_suffix = " [inpaint]" if is_inpaint else ""
        choices.append((f"{checkpoint.title}{label_suffix}", checkpoint.title))
    return choices


def _default_checkpoint(checkpoints: list[Checkpoint], *, prefer_inpaint: bool = False) -> str | None:
    if not checkpoints:
        return None
    if prefer_inpaint:
        for checkpoint in checkpoints:
            if checkpoint.kind == "inpaint":
                return checkpoint.title
    return checkpoints[0].title


def checkpoint_dropdown(
    ctx: AppContext,
    label: str = "Checkpoint",
    *,
    prefer_inpaint: bool = False,
) -> tuple[gr.Dropdown, dict[str, str]]:
    checkpoints = ctx.generation.list_checkpoints()
    id_map = {c.title: c.id for c in checkpoints}
    choices = _checkpoint_choices(checkpoints, inpaint_only=prefer_inpaint)
    if prefer_inpaint and not choices:
        choices = _checkpoint_choices(checkpoints)

    dropdown = gr.Dropdown(
        label=label,
        choices=choices,
        value=_default_checkpoint(checkpoints, prefer_inpaint=prefer_inpaint),
        allow_custom_value=False,
    )
    return dropdown, id_map


def refresh_checkpoints(
    ctx: AppContext,
    *,
    prefer_inpaint: bool = False,
    rescan: bool = False,
) -> tuple[gr.Dropdown, dict[str, str]]:
    checkpoints = (
        ctx.generation.refresh_checkpoint_catalog()
        if rescan
        else ctx.generation.list_checkpoints()
    )
    id_map = {c.title: c.id for c in checkpoints}
    choices = _checkpoint_choices(checkpoints, inpaint_only=prefer_inpaint)
    if prefer_inpaint and not choices:
        choices = _checkpoint_choices(checkpoints)
    update = gr.update(
        choices=choices,
        value=_default_checkpoint(checkpoints, prefer_inpaint=prefer_inpaint),
    )
    return update, id_map


def format_model_status(ctx: AppContext) -> str:
    checkpoints = ctx.generation.list_checkpoints()
    ckpt_dir = ctx.flags.resolved_ckpt_dir()
    models_dir = ctx.flags.resolved_models_dir()
    if checkpoints:
        names = ", ".join(c.filename for c in checkpoints[:5])
        extra = f" (+{len(checkpoints) - 5} more)" if len(checkpoints) > 5 else ""
        return f"**{len(checkpoints)}** checkpoints · `{ckpt_dir.name}` — {names}{extra}"
    return (
        f"No models found.\n\n"
        f"Place `.safetensors` or `.ckpt` files in:\n"
        f"- `{ckpt_dir}`\n"
        f"- or directly in `{models_dir}`\n\n"
        f"Then click **Refresh models**."
    )