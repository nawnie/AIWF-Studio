from __future__ import annotations

import gradio as gr

from aiwf.bootstrap import AppContext
from aiwf.core.domain.models import Checkpoint
from aiwf.infrastructure.diffusers.model_arch import is_inpaint_architecture


ENGINE_MAPPING = {
    "Flux": {"flux"},
    "Flux 2": {"flux2_klein"},
    "Stable Diffusion 1.5": {"sd15", "inpaint"},
    "Stable Diffusion XL": {"sdxl", "sdxl_inpaint"},
    "Stable Diffusion 3.5": {"sd35"},
    "Z-Image": {"z_image"},
}


def _filter_checkpoints(checkpoints: list[Checkpoint], engine_filter: str) -> list[Checkpoint]:
    if not engine_filter or engine_filter == "All":
        return checkpoints
    allowed = ENGINE_MAPPING.get(engine_filter, set())
    return [c for c in checkpoints if c.architecture in allowed]


def _checkpoint_choices(checkpoints: list[Checkpoint]) -> list[tuple[str, str]]:
    """List checkpoints with optional [inpaint] suffix for display.
    No auto-sorting or preferring — user choice in the dropdown is authoritative.
    (All models can be used for inpaint via the appropriate pipeline; dedicated
    inpaint weights are just better at it.)"""
    choices = []
    for checkpoint in checkpoints:
        is_inpaint = checkpoint.kind == "inpaint"
        label_suffix = " [inpaint]" if is_inpaint else ""
        choices.append((f"{checkpoint.title}{label_suffix}", checkpoint.title))
    return choices


def resolve_default_checkpoint(
    checkpoints: list[Checkpoint],
    last_checkpoint_id: str | None = None,
) -> Checkpoint | None:
    """Pick startup checkpoint: last user selection, else first non-inpaint in catalog.

    Inpaint checkpoints (9-channel UNet) can't run txt2img/img2img, so defaulting to one
    would hand the user a checkpoint that errors on the first generation. Prefer a standard
    checkpoint for the default; the user can still pick an inpaint model explicitly."""
    if not checkpoints:
        return None
    if last_checkpoint_id:
        for checkpoint in checkpoints:
            if checkpoint.id == last_checkpoint_id and not _is_inpaint_checkpoint(checkpoint):
                return checkpoint
    for checkpoint in checkpoints:
        if not _is_inpaint_checkpoint(checkpoint):
            return checkpoint
    return checkpoints[0]


def _is_inpaint_checkpoint(checkpoint: Checkpoint) -> bool:
    return checkpoint.kind == "inpaint" or is_inpaint_architecture(checkpoint.architecture)


def default_checkpoint_title(
    checkpoints: list[Checkpoint],
    last_checkpoint_id: str | None = None,
) -> str | None:
    selected = resolve_default_checkpoint(checkpoints, last_checkpoint_id)
    return selected.title if selected else None


def checkpoint_dropdown(
    ctx: AppContext,
    label: str = "Model",
    engine_filter: str = "All",
) -> tuple[gr.Dropdown, dict[str, str]]:
    checkpoints = ctx.generation.list_checkpoints()
    filtered = _filter_checkpoints(checkpoints, engine_filter)
    id_map = {c.title: c.id for c in filtered}
    choices = _checkpoint_choices(filtered)
    default_title = default_checkpoint_title(filtered, ctx.settings.last_checkpoint_id)
    # Ensure the value is actually present in choices to avoid Gradio warnings
    choice_ids = {c[1] for c in choices}
    if default_title and default_title not in choice_ids:
        default_title = choices[0][1] if choices else None

    dropdown = gr.Dropdown(
        label=label,
        choices=choices,
        value=default_title,
        allow_custom_value=False,
    )
    return dropdown, id_map


def refresh_checkpoints(
    ctx: AppContext,
    *,
    rescan: bool = False,
    current_value: str | None = None,
    engine_filter: str = "All",
) -> tuple[gr.Dropdown, dict[str, str]]:
    """Refresh the checkpoint list.

    If current_value is still present in the (new) choices (by label or id), it is
    preserved. There is no UI logic that auto-swaps or prefers certain models
    (e.g. no forcing inpaint models when in inpaint mode). The user-selected
    value in the dropdown is always respected. If the current value is no longer
    valid (e.g. after a rescan removed it), falls back to the first in the list.
    """
    checkpoints = (
        ctx.generation.refresh_checkpoint_catalog()
        if rescan
        else ctx.generation.list_checkpoints()
    )
    filtered = _filter_checkpoints(checkpoints, engine_filter)
    id_map = {c.title: c.id for c in filtered}
    choices = _checkpoint_choices(filtered)
    valid = {label for (label, _id) in choices} | { _id for (_label, _id) in choices }
    if current_value and current_value in valid:
        update = gr.update(choices=choices, value=current_value)
    else:
        default_title = default_checkpoint_title(filtered, ctx.settings.last_checkpoint_id)
        # Ensure the value is actually present in choices to avoid Gradio warnings
        choice_ids = {c[1] for c in choices}
        if default_title and default_title not in choice_ids:
            default_title = choices[0][1] if choices else None
        update = gr.update(choices=choices, value=default_title)
    return update, id_map


def format_model_status(ctx: AppContext, engine_filter: str = "All") -> str:
    checkpoints = ctx.generation.list_checkpoints()
    filtered = _filter_checkpoints(checkpoints, engine_filter)
    ckpt_dir = ctx.flags.resolved_ckpt_dir()
    models_dir = ctx.flags.resolved_models_dir()
    engine_desc = f" for {engine_filter}" if engine_filter != "All" else ""
    if filtered:
        names = ", ".join(c.filename for c in filtered[:5])
        extra = f" (+{len(filtered) - 5} more)" if len(filtered) > 5 else ""
        return f"**{len(filtered)}** models{engine_desc} · `{ckpt_dir.name}` — {names}{extra}"
    return (
        f"No models{engine_desc} found.\n\n"
        f"Place `.safetensors` or `.ckpt` files in:\n"
        f"- `{ckpt_dir}`\n"
        f"- or directly in `{models_dir}`\n\n"
        f"Then click **Refresh models** or choose another engine."
    )
