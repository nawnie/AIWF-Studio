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
    family: str            # lightning | hyper | turbo | lcm | tcd | sdxl_refiner | flux_fusion | flux_fill | flux_kontext | flux2_klein | z_image | krea2_turbo | krea2_raw | anima | qwen_image | qwen_image_nunchaku | sana | sana_video | standard
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
            "sdxl_refiner": "SDXL refiner",
            "flux_fusion": "Flux Fusion model",
            "flux_fill": "Flux Fill model",
            "flux_kontext": "Flux Kontext model",
            "flux2_klein": "Flux.2 Klein model",
            "z_image": "Z-Image model",
            "krea2_turbo": "Krea 2 Turbo model",
            "krea2_raw": "Krea 2 Raw model",
            "anima": "Anima model",
            "qwen_image": "Qwen Image model",
            "qwen_image_nunchaku": "Qwen Image Nunchaku model",
            "sana": "Sana model",
            "sana_sprint": "Sana Sprint model",
            "sana_video": "Sana Video model",
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
    "sdxl_refiner": (6.0, 10.0, 10, "dpmpp_2m", "automatic",
                     "Use the SDXL refiner as an optional second pass with about 10 steps, not as a standalone generator."),
    "flux_fusion": (1.0, 1.5, 4, "euler", "automatic",
                    "Use Euler, CFG 1, and 4 steps for Flux Fusion / 4-step distilled Flux variants."),
    "flux_fill": (3.5, 6.0, 28, "euler", "automatic",
                  "Use Euler, guidance/CFG 3.5, and about 28 steps for Flux Fill inpaint checkpoints."),
    "flux_kontext": (3.5, 6.0, 28, "euler", "automatic",
                     "Use guidance 3.5 and about 28 steps for Flux Kontext checkpoints."),
    "flux2_klein": (1.0, 1.5, 12, "euler", "automatic",
                    "Use Euler, CFG 1, and 10-15 steps for Fluxtrait Flux.2 Klein variants."),
    "z_image": (1.0, 1.5, 8, "euler", "automatic",
                "Use Euler, CFG 1, and 8+ steps for Fluxtrait Z-Image Turbo variants."),
    "krea2_turbo": (0.0, 1.0, 8, "euler", "automatic",
                    "Use Euler, guidance/CFG 0, and 8 steps for Krea 2 Turbo."),
    "krea2_raw": (3.5, 6.0, 52, "euler", "automatic",
                  "Use Euler, guidance/CFG 3.5, and about 52 steps for Krea 2 Raw."),
    "anima": (4.5, 7.0, 36, "euler_a", "automatic",
              "Use CFG 4-5 and 30-50 steps for Anima anime and non-photorealistic image generation."),
    "qwen_image": (4.0, 6.0, 30, "euler", "automatic",
                   "Use true CFG 4 and about 30 steps for Qwen Image first-run quality."),
    "qwen_image_nunchaku": (1.0, 1.5, 4, "euler", "automatic",
                            "Use Euler, CFG 1, and 4 steps for Qwen Image Nunchaku Lightning INT4."),
    "sana": (4.5, 7.0, 20, "euler", "automatic",
             "Use CFG 4.5 and about 20 steps for standard Sana 1024px checkpoints."),
    "sana_sprint": (4.5, 7.0, 2, "euler", "automatic",
                    "Use CFG 4.5 and 2 steps for Sana Sprint checkpoints."),
    "sana_video": (6.0, 7.0, 50, "euler", "automatic",
                   "Use CFG 6 and about 50 steps for Sana Video 480p/720p Diffusers snapshots."),
}

# Ordered so the most specific / least ambiguous markers win.
_MARKERS = [
    ("krea2_turbo", [r"krea[\s_-]?2.*turbo", r"krea2.*turbo"]),
    ("krea2_raw", [r"krea[\s_-]?2.*(?:raw|base)", r"krea2.*(?:raw|base)"]),
    ("anima", [r"(?:^|[\s_./-])anima(?:$|[\s_./-])", r"anima[\s_-]?(?:base|preview)", r"circlestone.*anima"]),
    ("qwen_image_nunchaku", [r"qwen.*(?:nunchaku|svdq-int4|lightningv|4steps)", r"(?:nunchaku|svdq-int4).*qwen"]),
    ("sdxl_refiner", [r"sd[\s_./-]?xl.*refiner", r"sd_xl_refiner", r"stable[\s_./-]?diffusion[\s_./-]?xl.*refiner"]),
    ("lightning", [r"lightning"]),
    ("turbo", [r"turbo"]),
    ("lcm", [r"lcm"]),
    ("tcd", [r"tcd"]),
    ("z_image", [r"z[\s_-]?image", r"zimage"]),
    ("qwen_image", [r"qwen[\s_-]?image", r"qwen2\.?0"]),
    ("sana_video", [r"sana[\s_-]?video", r"sanaimagetovideo", r"sanavideo"]),
    ("sana_sprint", [r"sana[\s_-]?sprint"]),
    ("sana", [r"sana"]),
    ("flux_fill", [r"flux.*fill", r"fill.*flux"]),
    ("flux_kontext", [r"flux[\s_-]?kontext", r"kontext"]),
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
    non_distilled = {"sdxl_refiner", "flux_fill", "flux_kontext", "krea2_raw", "anima", "qwen_image", "sana", "sana_video"}
    note = "" if family in non_distilled else "Distilled few-step model: high CFG causes overexposure."
    return ModelProfile(
        family=family,
        is_distilled=family not in non_distilled,
        recommended_cfg=cfg,
        cfg_max=cfg_max,
        recommended_steps=steps,
        recommended_sampler=sampler,
        recommended_scheduler=scheduler,
        help_text=blurb,
        note=note,
    )
