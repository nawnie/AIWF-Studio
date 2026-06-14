from __future__ import annotations

import gradio as gr

from aiwf.bootstrap import AppContext
from aiwf.core.domain.generation import GenerationMode
from aiwf.core.infotext import infotext_to_request_updates, parse_infotext


def read_png_infotext(image) -> tuple[str, dict]:
    if image is None:
        return "", {}
    from aiwf.services.metadata import MetadataService

    text = MetadataService().read_infotext(image) or ""
    params = parse_infotext(text) if text else {}
    return text, params


def apply_infotext_to_txt2img(text: str, sampler_labels: dict[str, str]):
    params = parse_infotext(text)
    updates = infotext_to_request_updates(params, GenerationMode.TXT2IMG)
    sampler_id = updates.get("sampler", "euler_a")
    sampler_label = sampler_labels.get(sampler_id, next(iter(sampler_labels.values()), "Euler a"))
    return (
        updates.get("prompt", ""),
        updates.get("negative_prompt", ""),
        gr.update(value=sampler_label),
        updates.get("steps", 20),
        updates.get("cfg_scale", 7.0),
        updates.get("width", 512),
        updates.get("height", 512),
        updates.get("seed", -1),
        updates.get("clip_skip", 1),
        updates.get("enable_hr", False),
        updates.get("hr_scale", 2.0),
        updates.get("hr_steps", 20),
        updates.get("hr_denoising_strength", 0.35),
    )


def apply_infotext_to_img2img(text: str, sampler_labels: dict[str, str]):
    params = parse_infotext(text)
    updates = infotext_to_request_updates(params, GenerationMode.IMG2IMG)
    sampler_id = updates.get("sampler", "euler_a")
    sampler_label = sampler_labels.get(sampler_id, next(iter(sampler_labels.values()), "Euler a"))
    return (
        updates.get("prompt", ""),
        updates.get("negative_prompt", ""),
        gr.update(value=sampler_label),
        updates.get("steps", 20),
        updates.get("cfg_scale", 7.0),
        updates.get("seed", -1),
        updates.get("denoising_strength", 0.75),
        updates.get("clip_skip", 1),
    )


def store_for_tabs(ctx: AppContext, text: str, params: dict) -> str:
    ctx.infotext_bridge.set_pending(text, params)
    return "Parameters ready — open txt2img or img2img and click Paste parameters."