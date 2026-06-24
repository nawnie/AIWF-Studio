"""Detect distilled / few-step checkpoints and recommend safe parameters.

Distilled models (SDXL Lightning, Hyper-SD, SD/SDXL Turbo, LCM, TCD) are trained
to denoise in very few steps and collapse to overexposed, washed-out images at
normal guidance (CFG ~7). They need low CFG (~1-2) and few steps. This module is
the single source of truth for both the auto-guard in the generation service and
the model help box in the UI.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelProfile:
    family: str            # lightning | hyper | turbo | lcm | tcd | flux_fusion | flux2_klein | z_image | standard
    is_distilled: bool
    recommended_cfg: float
    cfg_max: float         # above this, a distilled model overexposes
    recommended_steps: int
    recommended_sampler: str   # sampler id (matches SAMPLER_CLASSES keys)
    recommended_scheduler: str  # schedule type
    help_text: str
    note: str = ""

    @property
    def title(self) -> str:
        labels = {
            "lightning": "Lightning model",
            "hyper": "Hyper-SD model",
            "turbo": "Turbo model",
            "lcm": "LCM model",
            "tcd": "TCD model",
            "flux_fusion": "Flux Fusion model",
            "flux2_klein": "Flux.2 Klein model",
            "z_image": "Z-Image model",
            "standard": "Standard model",
        }
        return labels.get(self.family, "Model")


# family -> (cfg, cfg_max, steps, sampler, scheduler, blurb)
_PROFILES = {
    "lightning": (1.5, 2.5, 6, "euler", "sgm_uniform",
                  "Use CFG 1.0-2.0 and ~4-8 steps. Higher CFG overexposes. Euler + SGM Uniform works well."),
    "hyper": (1.0, 2.0, 8, "ddim", "sgm_uniform",
              "Use CFG ~1.0 and 1-8 steps (match the model's step count). DDIM/TCD with SGM Uniform."),
    "turbo": (1.0, 1.5, 4, "euler_a", "sgm_uniform",
              "Use CFG 1.0 (no guidance) and 1-4 steps. Euler a + SGM Uniform."),
    "lcm": (1.5, 2.0, 6, "lcm", "automatic",
            "Use CFG 1.0-2.0 and 4-8 steps with the LCM sampler."),
    "tcd": (1.5, 2.0, 8, "tcd", "automatic",
            "Use CFG 1.0-2.0 and 4-8 steps with the TCD sampler."),
    "flux_fusion": (1.0, 1.5, 4, "euler", "automatic",
                    "Use Euler, CFG 1, and 4 steps for Flux Fusion / 4-step distilled Flux variants."),
    "flux2_klein": (1.0, 1.5, 12, "euler", "automatic",
                    "Use Euler, CFG 1, and 10-15 steps for Fluxtrait Flux.2 Klein variants."),
    "z_image": (1.0, 1.5, 8, "euler", "automatic",
                "Use Euler, CFG 1, and 8+ steps for Fluxtrait Z-Image Turbo variants."),
}

# Ordered so the most specific / least ambiguous markers win.
_MARKERS = [
    ("lightning", [r"lightning"]),
    ("turbo", [r"turbo"]),
    ("lcm", [r"lcm"]),
    ("tcd", [r"tcd"]),
    ("z_image", [r"z[\s_-]?image", r"zimage"]),
    ("flux_fusion", [r"flux[\s_-]?fusion", r"fusion[\s_-]?v\d"]),
    ("flux2_klein", [r"flux[\s._-]?2", r"klein"]),
    # Hyper-SD only -- must NOT match a baked "HyperVAE" on a normal checkpoint.
    ("hyper", [r"hyper[\s_-]?sd", r"hyper[\s_-]?sdxl"]),
]


def detect_model_profile(*names: str | None) -> ModelProfile:
    """Identify the model family from any of its names (title/filename/id)."""
    blob = " ".join(n for n in names if n).lower()

    family = "standard"
    for fam, patterns in _MARKERS:
        if any(re.search(p, blob) for p in patterns):
            family = fam
            break

    if family == "standard":
        return ModelProfile(
            family="standard",
            is_distilled=False,
            recommended_cfg=7.0,
            cfg_max=30.0,
            recommended_steps=20,
            recommended_sampler="euler_a",
            recommended_scheduler="automatic",
            help_text="Standard model -- CFG ~5-8 and 20-30 steps work well.",
        )

    cfg, cfg_max, steps, sampler, scheduler, blurb = _PROFILES[family]
    return ModelProfile(
        family=family,
        is_distilled=True,
        recommended_cfg=cfg,
        cfg_max=cfg_max,
        recommended_steps=steps,
        recommended_sampler=sampler,
        recommended_scheduler=scheduler,
        help_text=blurb,
        note="Distilled few-step model: high CFG causes overexposure.",
    )
