from aiwf.core.domain.segment_presets import (
    CUSTOM_SEGMENT_PRESET_ID,
    resolve_segment_text_prompt,
    segment_mask_preset_choices,
)


def test_segment_mask_preset_choices_include_custom():
    choices = segment_mask_preset_choices()
    assert choices[0] == ("Person", "person")
    assert choices[-1] == ("Custom…", CUSTOM_SEGMENT_PRESET_ID)


def test_resolve_segment_text_prompt_uses_preset():
    assert resolve_segment_text_prompt("face") == "face"
    assert resolve_segment_text_prompt("face_hair") == "face.hair"


def test_resolve_segment_text_prompt_uses_custom_when_selected():
    assert resolve_segment_text_prompt(CUSTOM_SEGMENT_PRESET_ID, "car.window") == "car.window"
    assert resolve_segment_text_prompt(CUSTOM_SEGMENT_PRESET_ID, "") == ""