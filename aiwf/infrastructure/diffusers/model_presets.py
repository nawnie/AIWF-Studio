"""Per-model generation presets: sane defaults by architecture, overridden by
whatever the user last ran successfully on a given checkpoint.

Resolution order (first match wins):
1. The user's last-used settings for this exact checkpoint id (``UserSettings.model_settings``).
2. A sane built-in default for the checkpoint's architecture (``ARCHITECTURE_PRESETS``).
3. Caller's existing UI value (we return only the keys we have an opinion about).
"""

from __future__ import annotations

from typing import Any

from aiwf.infrastructure.diffusers.model_arch import (
    ARCH_FLUX,
    ARCH_FLUX_KONTEXT,
    ARCH_FLUX2_KLEIN,
    ARCH_INPAINT,
    ARCH_QWEN_IMAGE,
    ARCH_QWEN_IMAGE_NUNCHAKU,
    ARCH_SANA,
    ARCH_SANA_VIDEO,
    ARCH_SD15,
    ARCH_SD35,
    ARCH_SDXL,
    ARCH_SDXL_INPAINT,
    ARCH_Z_IMAGE,
)

# Sane starting points per architecture. Distilled/guidance-embedded DiT models
# (Flux, Flux.2 Klein, Z-Image) want low/zero CFG and fewer steps; classic
# SD1.5/SDXL want the traditional CFG range and more steps.
ARCHITECTURE_PRESETS: dict[str, dict[str, Any]] = {
    ARCH_SD15: {"steps": 20, "cfg_scale": 7.0, "sampler": "euler_a", "scheduler": "automatic", "width": 512, "height": 512},
    ARCH_INPAINT: {"steps": 24, "cfg_scale": 7.0, "sampler": "euler_a", "scheduler": "automatic", "width": 512, "height": 512},
    ARCH_SDXL: {"steps": 28, "cfg_scale": 6.0, "sampler": "dpmpp_2m", "scheduler": "automatic", "width": 1024, "height": 1024},
    ARCH_SDXL_INPAINT: {"steps": 30, "cfg_scale": 6.0, "sampler": "dpmpp_2m", "scheduler": "automatic", "width": 1024, "height": 1024},
    ARCH_SD35: {"steps": 28, "cfg_scale": 4.5, "sampler": "euler_a", "scheduler": "automatic", "width": 1024, "height": 1024},
    ARCH_FLUX: {"steps": 20, "cfg_scale": 0.0, "sampler": "euler_a", "scheduler": "automatic", "width": 1024, "height": 1024},
    ARCH_FLUX_KONTEXT: {"steps": 28, "cfg_scale": 3.5, "sampler": "euler", "scheduler": "automatic", "width": 1024, "height": 1024},
    ARCH_FLUX2_KLEIN: {"steps": 12, "cfg_scale": 1.0, "sampler": "euler", "scheduler": "automatic", "width": 1024, "height": 1024},
    ARCH_Z_IMAGE: {"steps": 8, "cfg_scale": 1.0, "sampler": "euler", "scheduler": "automatic", "width": 1024, "height": 1024},
    ARCH_QWEN_IMAGE: {"steps": 30, "cfg_scale": 4.0, "sampler": "euler", "scheduler": "automatic", "width": 1024, "height": 1024},
    ARCH_QWEN_IMAGE_NUNCHAKU: {"steps": 4, "cfg_scale": 1.0, "sampler": "euler", "scheduler": "automatic", "width": 1024, "height": 1024},
    ARCH_SANA: {"steps": 20, "cfg_scale": 4.5, "sampler": "euler", "scheduler": "automatic", "width": 1024, "height": 1024},
    ARCH_SANA_VIDEO: {"steps": 50, "cfg_scale": 6.0, "sampler": "euler", "scheduler": "automatic", "width": 832, "height": 480},
}

SANA_SPRINT_PRESET: dict[str, Any] = {
    "steps": 2,
    "cfg_scale": 4.5,
    "sampler": "euler",
    "scheduler": "automatic",
    "width": 1024,
    "height": 1024,
}

QWEN_NUNCHAKU_LIGHTNING_PRESET: dict[str, Any] = {
    "steps": 4,
    "cfg_scale": 1.0,
    "sampler": "euler",
    "scheduler": "automatic",
    "width": 1024,
    "height": 1024,
}

# The generation-relevant fields we remember per checkpoint and may apply as
# UI defaults. Keep this list narrow - we don't want to silently change things
# like prompts, seed, or batch size when the user just switches models.
PRESET_FIELDS: tuple[str, ...] = ("steps", "cfg_scale", "sampler", "scheduler", "width", "height", "clip_skip")


def resolve_model_preset(
    model_settings: dict[str, dict[str, Any]],
    checkpoint_id: str | None,
    architecture: str | None,
) -> dict[str, Any]:
    """Return the preset dict to apply for this checkpoint: last-used wins, else
    architecture sane-default, else empty (caller keeps its current UI values)."""
    preset = dict(ARCHITECTURE_PRESETS.get(architecture or "", {}))
    checkpoint_text = (checkpoint_id or "").lower()
    if architecture == ARCH_QWEN_IMAGE and (
        "nunchaku" in checkpoint_text or "svdq-int4" in checkpoint_text or "lightning" in checkpoint_text or "4steps" in checkpoint_text
    ):
        preset.update(QWEN_NUNCHAKU_LIGHTNING_PRESET)
    if architecture == ARCH_SANA and "sprint" in checkpoint_text:
        preset.update(SANA_SPRINT_PRESET)
    if checkpoint_id:
        last_used = model_settings.get(checkpoint_id)
        if last_used:
            preset.update({k: v for k, v in last_used.items() if k in PRESET_FIELDS})
    return preset


def extract_preset_fields(request: Any) -> dict[str, Any]:
    """Pull the rememberable fields off a GenerationRequest-like object."""
    out: dict[str, Any] = {}
    for field in PRESET_FIELDS:
        if hasattr(request, field):
            value = getattr(request, field)
            if value is not None:
                out[field] = value
    return out
