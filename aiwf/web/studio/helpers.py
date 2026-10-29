from __future__ import annotations

from PIL import Image as PILImage

from aiwf.core.domain.models import SCHEDULE_TYPES
from aiwf.core.tags import format_tags_display
from aiwf.web.studio.constants import MODES


def mode_from_label(label: str) -> str:
    for mode_id, mode_label in MODES:
        if mode_label == label or mode_id == label:
            return mode_id
    return "txt2img"


def paste_control_values(
    updates: dict,
    *,
    sampler_id_to_label: dict[str, str],
    default_sampler_label: str,
) -> dict[str, object]:
    sampler_label = sampler_id_to_label.get(updates.get("sampler", "euler_a"), default_sampler_label)
    schedule_labels = {s.id: s.label for s in SCHEDULE_TYPES}
    denoise_strength = updates.get("denoising_strength", 0.75)
    return {
        "prompt": updates.get("prompt", ""),
        "negative_prompt": updates.get("negative_prompt", ""),
        "sampler": sampler_label,
        "scheduler": schedule_labels.get(updates.get("scheduler", "automatic"), "Automatic"),
        "steps": updates.get("steps", 20),
        "cfg_scale": updates.get("cfg_scale", 7.0),
        "width": updates.get("width", 512),
        "height": updates.get("height", 512),
        "seed": updates.get("seed", -1),
        "clip_skip": updates.get("clip_skip", 1),
        "enable_hr": updates.get("enable_hr", False),
        "hr_scale": updates.get("hr_scale", 2.0),
        "hr_steps": updates.get("hr_steps", 20),
        "hr_denoising_strength": updates.get("hr_denoising_strength", 0.35),
        "img2img_denoise": denoise_strength,
        "inpaint_denoise": denoise_strength,
        "mask_blur": updates.get("mask_blur", 4),
        "tags": format_tags_display(updates.get("tags", [])),
    }


def align_compare_pair(before: PILImage.Image | None, after: PILImage.Image | None):
    if before is None or after is None:
        return before, after
    if before.size != after.size:
        before = before.resize(after.size, PILImage.Resampling.LANCZOS)
    return before, after


def format_tag_summary(tags: list[str]) -> str:
    if not tags:
        return ""
    return "**Tags** " + " · ".join(f"`#{tag}`" for tag in tags)


def generation_style_fields(
    style_name: str | None,
    template_prompt: str | None,
    template_negative: str | None,
) -> dict[str, str | None]:
    name = (style_name or "").strip() or None
    positive = (template_prompt or "").strip() or None
    negative = (template_negative or "").strip() or None
    if not name and not positive and not negative:
        return {
            "style_name": None,
            "style_prompt_template": None,
            "style_negative_template": None,
        }
    return {
        "style_name": name,
        "style_prompt_template": positive,
        "style_negative_template": negative,
    }


def segment_source_image(source_image, editor_value):
    if isinstance(editor_value, dict) and editor_value.get("background") is not None:
        return editor_value.get("background")
    return source_image


def load_uploaded_image(file_obj):
    if file_obj is None:
        return None

    if isinstance(file_obj, list):
        if not file_obj:
            return None
        file_obj = file_obj[0]
    path = getattr(file_obj, "name", file_obj)
    if isinstance(path, dict):
        path = path.get("path") or path.get("name")
    return PILImage.open(path).convert("RGB")