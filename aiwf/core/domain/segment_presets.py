from __future__ import annotations

from dataclasses import dataclass

CUSTOM_SEGMENT_PRESET_ID = "__custom__"


@dataclass(frozen=True)
class SegmentMaskPreset:
    id: str
    label: str
    prompt: str


SEGMENT_MASK_PRESETS: tuple[SegmentMaskPreset, ...] = (
    SegmentMaskPreset("person", "Person", "person"),
    SegmentMaskPreset("face", "Face", "face"),
    SegmentMaskPreset("hair", "Hair", "hair"),
    SegmentMaskPreset("hand", "Hand", "hand"),
    SegmentMaskPreset("arm", "Arm", "arm"),
    SegmentMaskPreset("leg", "Leg", "leg"),
    SegmentMaskPreset("shirt", "Shirt", "shirt"),
    SegmentMaskPreset("clothes", "Clothes", "clothes"),
    SegmentMaskPreset("dress", "Dress", "dress"),
    SegmentMaskPreset("pants", "Pants", "pants"),
    SegmentMaskPreset("shoes", "Shoes", "shoes"),
    SegmentMaskPreset("hat", "Hat", "hat"),
    SegmentMaskPreset("glasses", "Glasses", "glasses"),
    SegmentMaskPreset("dog", "Dog", "dog"),
    SegmentMaskPreset("cat", "Cat", "cat"),
    SegmentMaskPreset("bird", "Bird", "bird"),
    SegmentMaskPreset("horse", "Horse", "horse"),
    SegmentMaskPreset("car", "Car", "car"),
    SegmentMaskPreset("tree", "Tree", "tree"),
    SegmentMaskPreset("building", "Building", "building"),
    SegmentMaskPreset("sky", "Sky", "sky"),
    SegmentMaskPreset("water", "Water", "water"),
    SegmentMaskPreset("flower", "Flower", "flower"),
    SegmentMaskPreset("furniture", "Furniture", "furniture"),
    SegmentMaskPreset("text", "Text", "text"),
    SegmentMaskPreset("sign", "Sign", "sign"),
    SegmentMaskPreset("glare", "Glare", "glare"),
    SegmentMaskPreset("scratch", "Scratch", "scratch"),
    SegmentMaskPreset("stain", "Stain", "stain"),
    SegmentMaskPreset("watermark", "Watermark", "watermark"),
    SegmentMaskPreset("face_hair", "Face & hair", "face.hair"),
    SegmentMaskPreset("person_clothes", "Person & clothes", "person.clothes"),
)

PRESET_BY_ID: dict[str, SegmentMaskPreset] = {preset.id: preset for preset in SEGMENT_MASK_PRESETS}


def segment_mask_preset_choices() -> list[tuple[str, str]]:
    return [(preset.label, preset.id) for preset in SEGMENT_MASK_PRESETS] + [
        ("Custom…", CUSTOM_SEGMENT_PRESET_ID),
    ]


def resolve_segment_text_prompt(preset_id: str | None, custom_text: str = "") -> str:
    if preset_id == CUSTOM_SEGMENT_PRESET_ID:
        return (custom_text or "").strip()
    preset = PRESET_BY_ID.get(preset_id or "")
    if preset is not None:
        return preset.prompt
    return (custom_text or preset_id or "").strip()