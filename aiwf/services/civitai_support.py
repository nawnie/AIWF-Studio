"""Evidence-bounded mapping from Civitai resource categories to Studio routes.

Civitai's resource type is metadata, not a promise that a type is a standalone
generation model or trainable target. Keep this catalog explicit so the UI can
show real route limits and PC validation state without inferring support from a
download or filename.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone


_RESOURCES = (
    {
        "id": "Checkpoint",
        "label": "Checkpoint",
        "generation": "Partial: native image and video families have separate local routes; the file must match a supported architecture and format.",
        "training": "Partial: ED2 full image training and Kohya image LoRA training are available through the legacy Training tab for supported bases. No general video full-finetune route is wired.",
        "verification": "Validate each family and asset. Qwen Image 2.1 is not tested on this PC.",
    },
    {
        "id": "LORA",
        "label": "LoRA / LyCORIS",
        "generation": "Partial: classic SD-family runtime LoRA and Wan stage LoRAs are family-specific; several newer transformer image families remain blocked.",
        "training": "Partial: Kohya covers SD 1.x, SDXL, and Flux. Wan video-LoRA and Qwen Image 2.1 LoRA training are not wired.",
        "verification": "Validate the selected base, adapter format, and stage. No single type-wide pass claim.",
    },
    {
        "id": "TextualInversion",
        "label": "Textual inversion / embedding",
        "generation": "Limited to compatible prompt-embedding loaders and model families; Civitai downloads are not auto-imported as runnable assets.",
        "training": "No current Studio textual-inversion trainer route.",
        "verification": "Per-loader and base-family validation required.",
    },
    {
        "id": "Hypernetwork",
        "label": "Hypernetwork",
        "generation": "No current native Hypernetwork route.",
        "training": "No current Hypernetwork trainer route.",
        "verification": "Not tested on this PC; route is not implemented.",
    },
    {
        "id": "AestheticGradient",
        "label": "Aesthetic gradient",
        "generation": "No current native Aesthetic Gradient route.",
        "training": "No current Aesthetic Gradient trainer route.",
        "verification": "Not tested on this PC; route is not implemented.",
    },
    {
        "id": "Controlnet",
        "label": "ControlNet",
        "generation": "Partial: local ControlNet conditioning is limited to compatible SD 1.x / SDXL image routes.",
        "training": "No current ControlNet trainer route.",
        "verification": "Validate the exact control model, base family, and preprocessor.",
    },
    {
        "id": "VAE",
        "label": "VAE",
        "generation": "Partial: external VAE selection is available on compatible classic image routes; newer families use family-specific bundled components.",
        "training": "No current VAE trainer route.",
        "verification": "Validate VAE architecture and exact base compatibility.",
    },
    {
        "id": "MotionModule",
        "label": "Motion module",
        "generation": "No generic Civitai MotionModule loader. Studio video families use their own Wan, LTX, or Sana Video routes.",
        "training": "No video MotionModule / video LoRA trainer route is wired.",
        "verification": "Not tested on this PC; route is not implemented.",
    },
    {
        "id": "Upscaler",
        "label": "Upscaler",
        "generation": "Partial: selected image upscaling and restoration workers are available as post-processing, not as a base generator.",
        "training": "No current upscaler trainer route.",
        "verification": "Validate the exact upscaler worker and asset format.",
    },
    {
        "id": "Poses",
        "label": "Poses",
        "generation": "Pose data can serve as conditioning input when a compatible control route is available; it is not a standalone generator.",
        "training": "No pose-resource trainer route.",
        "verification": "Validate the consuming control route and pose format.",
    },
    {
        "id": "Wildcards",
        "label": "Wildcards",
        "generation": "Prompt text assets are not model weights; Civitai wildcard import and expansion are not currently wired.",
        "training": "Not a model-training target.",
        "verification": "Not tested on this PC; import/expansion route is not implemented.",
    },
    {
        "id": "Workflows",
        "label": "Workflows",
        "generation": "Studio has its own workflow builder, but does not import and execute arbitrary Civitai workflow packages.",
        "training": "Not a model-training target.",
        "verification": "Civitai workflow import is not implemented or tested on this PC.",
    },
    {
        "id": "Detection",
        "label": "Detection",
        "generation": "Some local segmentation/detection features exist, but Civitai Detection packs do not have a universal loader.",
        "training": "No Civitai Detection trainer route.",
        "verification": "Validate each detection worker and model format.",
    },
    {
        "id": "Other",
        "label": "Other",
        "generation": "No generic route; asset type, architecture, and format must be identified first.",
        "training": "No generic route.",
        "verification": "Not tested on this PC; classify and validate before enabling.",
    },
)


def civitai_support_catalog() -> dict:
    """Return a fresh JSON-safe snapshot for the local Studio support panel."""
    return {
        "schema": "aiwf.civitai-resource-support.v1",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "source": "Studio route inspection; Civitai category labels are descriptive metadata",
        "limitations": [
            "Civitai owns the current ModelType enum; this table tracks documented categories but is not an exhaustive or permanent copy of that enum.",
            "Generation and training are family-specific. A Civitai type label alone cannot prove either capability.",
            "A feature marked implemented has not thereby passed a local runtime smoke; use the verification text for the actual gate.",
        ],
        "resources": deepcopy(list(_RESOURCES)),
    }
