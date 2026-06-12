from __future__ import annotations

import json
import re
from typing import Any

from aiwf.core.domain.generation import GenerationMode, GenerationRequest
from aiwf.core.domain.models import Checkpoint
from aiwf.core.tags import format_tags_infotext, parse_tags_from_params

RE_PARAM = re.compile(r"\s*(\w[\w \-/]+):\s*(\"(?:\\.|[^\"])+\"|[^,]*)(?:,|$)")
RE_IMAGE_SIZE = re.compile(r"^(\d+)x(\d+)$")

SAMPLER_ALIASES: dict[str, str] = {
    "euler a": "euler_a",
    "euler": "euler",
    "heun": "heun",
    "lms": "lms",
    "ddim": "ddim",
    "unipc": "unipc",
    "dpm2": "dpm2",
    "dpm2 a": "dpm2_a",
    "deis": "deis",
    "dpm++ 2m": "dpmpp_2m",
    "dpm++ 2m sde": "dpmpp_2m_sde",
    "dpm++ 3m sde": "dpmpp_3m_sde",
    "dpm++ sde": "dpmpp_sde",
    "dpm++ 2m karras": "dpmpp_2m_karras",
    "sa-solver": "sa_solver",
    "lcm": "lcm",
    "tcd": "tcd",
}


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value[1:-1]
    return value


def parse_infotext(text: str) -> dict[str, Any]:
    """Parse A1111-style generation parameters from PNG metadata or UI paste."""
    text = text.strip()
    if not text:
        return {}

    prompt = ""
    negative_prompt = ""
    done_with_prompt = False

    lines = text.split("\n")
    last_line = lines[-1] if lines else ""
    body_lines = lines[:-1]

    if len(RE_PARAM.findall(last_line)) < 3 and lines:
        body_lines = lines
        last_line = ""

    for line in body_lines:
        line = line.strip()
        if line.startswith("Negative prompt:"):
            done_with_prompt = True
            line = line[len("Negative prompt:") :].strip()
        if done_with_prompt:
            negative_prompt = f"{negative_prompt}\n{line}".strip() if negative_prompt else line
        else:
            prompt = f"{prompt}\n{line}".strip() if prompt else line

    params: dict[str, Any] = {"Prompt": prompt, "Negative prompt": negative_prompt}
    for key, value in RE_PARAM.findall(last_line):
        value = _unquote(value.strip())
        size_match = RE_IMAGE_SIZE.match(value)
        if size_match:
            params[f"{key}-1"] = int(size_match.group(1))
            params[f"{key}-2"] = int(size_match.group(2))
        else:
            params[key] = value

    if "Clip skip" not in params:
        params["Clip skip"] = 1

    return params


def normalize_sampler(name: str | None) -> str | None:
    if not name:
        return None
    return SAMPLER_ALIASES.get(name.strip().lower(), name.strip().lower().replace(" ", "_"))


def infotext_to_request_updates(params: dict[str, Any], mode: GenerationMode) -> dict[str, Any]:
    """Map parsed infotext fields to GenerationRequest kwargs."""
    updates: dict[str, Any] = {
        "prompt": params.get("Prompt", ""),
        "negative_prompt": params.get("Negative prompt", ""),
    }

    if "Steps" in params:
        updates["steps"] = int(params["Steps"])
    if "CFG scale" in params:
        updates["cfg_scale"] = float(params["CFG scale"])
    if "Seed" in params:
        updates["seed"] = int(params["Seed"])
    if "Clip skip" in params:
        updates["clip_skip"] = int(params["Clip skip"])

    sampler = normalize_sampler(str(params.get("Sampler", "")))
    if sampler:
        updates["sampler"] = sampler

    schedule = str(params.get("Schedule type", "")).strip().lower().replace(" ", "_")
    if schedule:
        updates["scheduler"] = schedule

    width = params.get("Size-1") or params.get("Hires resize-1")
    height = params.get("Size-2") or params.get("Hires resize-2")
    if width and height:
        updates["width"] = int(width)
        updates["height"] = int(height)

    if mode == GenerationMode.IMG2IMG and "Denoising strength" in params:
        updates["denoising_strength"] = float(params["Denoising strength"])
    if mode == GenerationMode.INPAINT:
        if "Denoising strength" in params:
            updates["denoising_strength"] = float(params["Denoising strength"])
        if "Mask blur" in params:
            updates["mask_blur"] = int(params["Mask blur"])

    hires_upscale = params.get("Hires upscale")
    hires_resize_x = params.get("Hires resize-1")
    hires_resize_y = params.get("Hires resize-2")
    if hires_upscale or (hires_resize_x and hires_resize_y):
        updates["enable_hr"] = True
        if hires_upscale:
            updates["hr_scale"] = float(hires_upscale)
        if "Hires steps" in params:
            updates["hr_steps"] = int(params["Hires steps"])
        if "Denoising strength" in params and mode == GenerationMode.TXT2IMG:
            updates["hr_denoising_strength"] = float(params["Denoising strength"])

    tags = parse_tags_from_params(params)
    if tags:
        updates["tags"] = tags

    return updates


def format_infotext(
    request: GenerationRequest,
    seed: int,
    checkpoint: Checkpoint,
    *,
    output_width: int | None = None,
    output_height: int | None = None,
) -> str:
    """Build A1111-compatible generation parameters text."""
    width = output_width or request.width
    height = output_height or request.height
    lines = [request.prompt]
    if request.negative_prompt:
        lines.append(f"Negative prompt: {request.negative_prompt}")

    sampler_labels = {
        "euler_a": "Euler a",
        "euler": "Euler",
        "heun": "Heun",
        "lms": "LMS",
        "ddim": "DDIM",
        "unipc": "UniPC",
        "dpm2": "DPM2",
        "dpm2_a": "DPM2 a",
        "deis": "DEIS",
        "dpmpp_2m": "DPM++ 2M",
        "dpmpp_2m_sde": "DPM++ 2M SDE",
        "dpmpp_3m_sde": "DPM++ 3M SDE",
        "dpmpp_sde": "DPM++ SDE",
        "dpmpp_2m_karras": "DPM++ 2M Karras",
        "sa_solver": "SA-Solver",
        "lcm": "LCM",
        "tcd": "TCD",
    }
    sampler_label = sampler_labels.get(request.sampler, request.sampler)
    parts = [
        f"Steps: {request.steps}",
        f"Sampler: {sampler_label}",
        f"CFG scale: {request.cfg_scale}",
        f"Seed: {seed}",
        f"Size: {width}x{height}",
        f"Model: {checkpoint.title}",
    ]

    schedule_labels = {
        "uniform": "Uniform",
        "karras": "Karras",
        "exponential": "Exponential",
        "sgm_uniform": "SGM Uniform",
        "beta": "Beta",
    }
    schedule = getattr(request, "scheduler", "automatic")
    if schedule and schedule != "automatic":
        parts.append(f"Schedule type: {schedule_labels.get(schedule, schedule.title())}")

    if request.clip_skip > 1:
        parts.append(f"Clip skip: {request.clip_skip}")

    if request.mode in (GenerationMode.IMG2IMG, GenerationMode.INPAINT):
        parts.append(f"Denoising strength: {request.denoising_strength}")
    if request.mode == GenerationMode.INPAINT:
        parts.append(f"Mask blur: {request.mask_blur}")

    if request.enable_hr and request.mode == GenerationMode.TXT2IMG:
        parts.extend(
            [
                f"Hires upscale: {request.hr_scale}",
                f"Hires steps: {request.hr_steps}",
                f"Hires upscaler: {request.hr_upscaler}",
                f"Denoising strength: {request.hr_denoising_strength}",
            ]
        )

    if request.tags:
        parts.append(f"Tags: {format_tags_infotext(request.tags)}")

    lines.append(", ".join(parts))
    return "\n".join(line for line in lines if line)