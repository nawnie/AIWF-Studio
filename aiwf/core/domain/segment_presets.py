from __future__ import annotations

from dataclasses import asdict, dataclass

CUSTOM_SEGMENT_PRESET_ID = "__custom__"


@dataclass(frozen=True)
class SegmentMaskPreset:
    """Text-guided mask preset plus edge-treatment defaults.

    The prompt alone is not a useful preset. These values intentionally tune
    detection strictness and post-processing for the expected subject shape.
    """

    id: str
    label: str
    prompt: str
    box_threshold: float = 0.30
    mask_index: int = 0
    dilation: int = 4
    mask_blur: int = 2
    feather: int = 6
    note: str = ""


SEGMENT_MASK_PRESETS: tuple[SegmentMaskPreset, ...] = (
    SegmentMaskPreset("person", "Person", "person", 0.28, 0, 6, 3, 8, "Balanced full-body coverage."),
    SegmentMaskPreset("face", "Face", "face", 0.32, 0, 8, 3, 12, "Extra edge room for cheeks, ears, and blending."),
    SegmentMaskPreset("hair", "Hair", "hair", 0.24, 1, 4, 2, 12, "Lower threshold and soft edge for fine strands."),
    SegmentMaskPreset("hand", "Hand", "hand", 0.34, 0, 10, 3, 8, "More dilation helps include fingers."),
    SegmentMaskPreset("arm", "Arm", "arm", 0.31, 0, 7, 3, 8),
    SegmentMaskPreset("leg", "Leg", "leg", 0.31, 0, 7, 3, 8),
    SegmentMaskPreset("shirt", "Shirt", "shirt", 0.30, 0, 5, 2, 6),
    SegmentMaskPreset("clothes", "Clothes", "clothes", 0.28, 0, 6, 3, 7),
    SegmentMaskPreset("dress", "Dress", "dress", 0.29, 0, 6, 3, 8),
    SegmentMaskPreset("pants", "Pants", "pants", 0.31, 0, 5, 2, 6),
    SegmentMaskPreset("shoes", "Shoes", "shoes", 0.35, 0, 4, 2, 5),
    SegmentMaskPreset("hat", "Hat", "hat", 0.34, 0, 5, 2, 6),
    SegmentMaskPreset("glasses", "Glasses", "glasses", 0.38, 1, 3, 1, 3, "Small-object profile; inspect candidate masks."),
    SegmentMaskPreset("dog", "Dog", "dog", 0.29, 0, 6, 3, 9),
    SegmentMaskPreset("cat", "Cat", "cat", 0.29, 0, 6, 3, 9),
    SegmentMaskPreset("bird", "Bird", "bird", 0.30, 0, 5, 2, 7),
    SegmentMaskPreset("horse", "Horse", "horse", 0.29, 0, 6, 3, 8),
    SegmentMaskPreset("car", "Car", "car", 0.32, 0, 4, 2, 5),
    SegmentMaskPreset("tree", "Tree", "tree", 0.24, 0, 3, 2, 8, "Branches may need manual refinement."),
    SegmentMaskPreset("building", "Building", "building", 0.30, 0, 2, 1, 3),
    SegmentMaskPreset("sky", "Sky", "sky", 0.22, 0, 2, 2, 5, "Low threshold for broad continuous regions."),
    SegmentMaskPreset("water", "Water", "water", 0.22, 0, 2, 2, 5),
    SegmentMaskPreset("flower", "Flower", "flower", 0.32, 0, 4, 2, 5),
    SegmentMaskPreset("furniture", "Furniture", "furniture", 0.31, 0, 4, 2, 5),
    SegmentMaskPreset("text", "Text", "text", 0.40, 0, 2, 1, 2, "Use a high threshold; OCR-shaped masks may still need cleanup."),
    SegmentMaskPreset("sign", "Sign", "sign", 0.36, 0, 3, 1, 3),
    SegmentMaskPreset("glare", "Glare", "glare", 0.25, 0, 8, 4, 10, "Approximate text guidance; review before inpainting."),
    SegmentMaskPreset("scratch", "Scratch", "scratch", 0.24, 0, 12, 3, 5, "Thin defects need dilation; manual painting may be more reliable."),
    SegmentMaskPreset("stain", "Stain", "stain", 0.25, 0, 8, 4, 10, "Approximate defect mask; inspect the overlay."),
    SegmentMaskPreset("watermark", "Watermark", "watermark", 0.34, 0, 5, 2, 5),
    SegmentMaskPreset("face_hair", "Face & hair", "face.hair", 0.27, 0, 8, 3, 12),
    SegmentMaskPreset("person_clothes", "Person & clothes", "person.clothes", 0.27, 0, 7, 3, 9),
)

PRESET_BY_ID: dict[str, SegmentMaskPreset] = {preset.id: preset for preset in SEGMENT_MASK_PRESETS}


def segment_mask_preset_choices() -> list[tuple[str, str]]:
    return [(preset.label, preset.id) for preset in SEGMENT_MASK_PRESETS] + [
        ("Custom…", CUSTOM_SEGMENT_PRESET_ID),
    ]


def resolve_segment_preset(preset_id: str | None) -> SegmentMaskPreset | None:
    return PRESET_BY_ID.get(preset_id or "")


def resolve_segment_preset_config(preset_id: str | None) -> dict[str, object]:
    preset = resolve_segment_preset(preset_id)
    if preset is None:
        return {
            "box_threshold": 0.30,
            "mask_index": 0,
            "dilation": 4,
            "mask_blur": 2,
            "feather": 6,
            "note": "Custom prompt: inspect the mask and tune edge controls for the subject.",
        }
    return asdict(preset)


def resolve_segment_text_prompt(preset_id: str | None, custom_text: str = "") -> str:
    if preset_id == CUSTOM_SEGMENT_PRESET_ID:
        return (custom_text or "").strip()
    preset = resolve_segment_preset(preset_id)
    if preset is not None:
        return preset.prompt
    return (custom_text or preset_id or "").strip()
