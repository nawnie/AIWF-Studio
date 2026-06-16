from __future__ import annotations

import re

import gradio as gr
from PIL import Image as PILImage

from aiwf.bootstrap import AppContext
from aiwf.core.domain.generation import GenerationMode
from aiwf.core.infotext import infotext_to_request_updates, parse_infotext
from aiwf.core.tags import format_tags_display, parse_tags
from aiwf.web.studio.catalogs import StudioCatalogs
from aiwf.web.studio.helpers import mode_from_label, paste_control_values


def append_quick_tag(current: str, selected: str | None) -> tuple:
    if not selected:
        return current or "", gr.update(value=None)
    tags = parse_tags(current or "")
    if selected not in tags:
        tags.append(selected)
    return format_tags_display(tags), gr.update(value=None)


def apply_paste(
    text: str,
    mode_label: str,
    catalogs: StudioCatalogs,
    *,
    default_sampler_label: str,
) -> tuple:
    if not text.strip():
        raise gr.Error("Paste infotext first.")
    mode = mode_from_label(mode_label)
    gen_mode = {
        "txt2img": GenerationMode.TXT2IMG,
        "img2img": GenerationMode.IMG2IMG,
        "inpaint": GenerationMode.INPAINT,
    }[mode]
    updates = infotext_to_request_updates(parse_infotext(text), gen_mode)
    controls = paste_control_values(
        updates,
        sampler_id_to_label=catalogs.sampler_id_to_label,
        default_sampler_label=default_sampler_label,
    )
    return (
        controls["prompt"],
        controls["negative_prompt"],
        gr.update(value=controls["sampler"]),
        gr.update(value=controls["scheduler"]),
        controls["steps"],
        controls["cfg_scale"],
        controls["width"],
        controls["height"],
        controls["seed"],
        controls["clip_skip"],
        controls["enable_hr"],
        controls["hr_scale"],
        controls["hr_steps"],
        controls["hr_denoising_strength"],
        controls["img2img_denoise"],
        controls["inpaint_denoise"],
        controls["mask_blur"],
        controls["tags"],
    )


def pnginfo_pending_hint(ctx: AppContext):
    if ctx.infotext_bridge.pending_text:
        return gr.update(
            value="**PNG Info waiting** — click **From PNG Info** to apply parameters.",
            visible=True,
        )
    return gr.update(value="", visible=False)


def last_generation_infotext(ctx: AppContext) -> str | None:
    for job in ctx.generation.recent_jobs(10):
        result = getattr(job, "result", None)
        if result is not None and getattr(result, "infotexts", None):
            return result.infotexts[-1]
    try:
        root = ctx.flags.resolved_output_dir()
        candidates = sorted(root.rglob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)[:20]
        for path in candidates:
            try:
                with PILImage.open(path) as img:
                    text = (getattr(img, "text", None) or {}).get("parameters")
                if text:
                    return text
            except Exception:
                continue
    except Exception:
        pass
    return None


def reuse_last_generation(
    ctx: AppContext,
    mode_label: str,
    catalogs: StudioCatalogs,
    *,
    default_sampler_label: str,
) -> tuple:
    text = last_generation_infotext(ctx)
    if not text:
        raise gr.Error(
            "No previous generation found — generate once first "
            "(or enable PNG metadata embedding in Settings)."
        )
    applied = apply_paste(text, mode_label, catalogs, default_sampler_label=default_sampler_label)
    return (text, *applied)


def load_bridge(
    ctx: AppContext,
    mode_label: str,
    catalogs: StudioCatalogs,
    *,
    default_sampler_label: str,
) -> tuple:
    if ctx.settings.pnginfo_clear_after_apply:
        text = ctx.infotext_bridge.consume_pending()
    else:
        text = ctx.infotext_bridge.pending_text
    if not text:
        raise gr.Error("No parameters waiting. Use PNG Info → Send first.")
    applied = apply_paste(text, mode_label, catalogs, default_sampler_label=default_sampler_label)
    return (text, *applied, gr.update(value="", visible=False))


def on_lora_pick(ctx: AppContext, lora_id: str | None):
    if not lora_id:
        return gr.update()
    return gr.update(value=ctx.models.lora_strength(lora_id))


def refresh_lora_picker(ctx: AppContext, current):
    ctx.models.refresh_loras()
    choices = ctx.models.lora_choices()
    ids = {value for _, value in choices}
    return gr.update(choices=choices, value=current if current in ids else None)


def add_lora_to_prompt(ctx: AppContext, current_prompt: str, lora_id: str | None, strength) -> str:
    lora = ctx.models.find_lora(lora_id)
    if lora is None:
        raise gr.Error("Pick a LoRA first (hit Refresh if the list is empty).")
    if f"<lora:{lora.id}:" in (current_prompt or ""):
        raise gr.Error("That LoRA is already in the prompt — edit its strength there.")
    tag = f"<lora:{lora.id}:{float(strength):g}>"
    keywords = (ctx.models.lora_keywords(lora.id) or "").strip()
    addition = f"{tag}, {keywords}" if keywords else tag
    text = (current_prompt or "").rstrip().rstrip(",")
    return f"{text}, {addition}" if text else addition


_LORA_TAG_RE = re.compile(r"\s*,?\s*<lora:[^>]+>")


def strip_lora_tags(prompt: str) -> str:
    text = _LORA_TAG_RE.sub("", prompt or "")
    text = re.sub(r"\s*,\s*,+", ", ", text)
    return text.strip().strip(",").strip()


def apply_lora_stack_to_prompt(
    ctx: AppContext,
    current_prompt: str,
    *slot_values,
) -> str:
    """Replace prompt LoRA tags with a compact selected stack.

    slot_values is packed as repeating ``(lora_id, strength)`` pairs followed
    by one ``include_keywords`` boolean.
    """
    include_keywords = bool(slot_values[-1]) if slot_values else True
    pairs = list(zip(slot_values[0:-1:2], slot_values[1:-1:2]))
    seen: set[str] = set()
    additions: list[str] = []
    for lora_id, strength in pairs:
        if not lora_id or lora_id in seen:
            continue
        lora = ctx.models.find_lora(str(lora_id))
        if lora is None:
            continue
        seen.add(lora.id)
        tag = f"<lora:{lora.id}:{float(strength):g}>"
        if include_keywords:
            keywords = (ctx.models.lora_keywords(lora.id) or "").strip()
            additions.append(f"{tag}, {keywords}" if keywords else tag)
        else:
            additions.append(tag)

    base = strip_lora_tags(current_prompt)
    if not additions:
        return base
    suffix = ", ".join(additions)
    return f"{base}, {suffix}" if base else suffix


def refresh_lora_stack(ctx: AppContext, *current_values):
    ctx.models.refresh_loras()
    choices = ctx.models.lora_choices()
    ids = {value for _, value in choices}
    updates = []
    for current in current_values:
        updates.append(gr.update(choices=choices, value=current if current in ids else None))
    return tuple(updates)


def refresh_embedding_picker(ctx: AppContext, current):
    items = ctx.generation.refresh_embedding_catalog()
    choices = [(e.title, e.id) for e in items]
    ids = {value for _, value in choices}
    return gr.update(choices=choices, value=current if current in ids else None)


def append_token(text: str, token: str) -> str:
    base = (text or "").rstrip().rstrip(",")
    return f"{base}, {token}" if base else token


def add_embedding_to(field_value: str, embedding_id: str | None) -> str:
    if not embedding_id:
        raise gr.Error("Pick an embedding first (hit Refresh if the list is empty).")
    if embedding_id in (field_value or ""):
        raise gr.Error("That embedding is already in the prompt.")
    return append_token(field_value, embedding_id)
