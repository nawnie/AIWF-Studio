from __future__ import annotations

from aiwf.core.domain.audio_lab import AudioLabSettings
from aiwf.core.domain.image_workflow import ImageWorkflowSettings
from aiwf.core.domain.segment_presets import resolve_segment_preset_config
from aiwf.services.audio_lab import parse_audio_command, resolve_audio_plan
from aiwf.services.image_workflow import resolve_image_plan
from aiwf.web.tabs.audio import normalize_audio_stages
from aiwf.web.tabs.video_lab import normalize_video_stages, video_stages_from_settings
from aiwf.services.video_lab import preset_settings


def test_segment_presets_include_real_edge_profiles() -> None:
    face = resolve_segment_preset_config("face")
    scratch = resolve_segment_preset_config("scratch")
    assert face["box_threshold"] == 0.32
    assert face["feather"] == 12
    assert scratch["dilation"] == 12
    assert scratch["feather"] == 5
    assert "manual" in str(scratch["note"]).lower()


def test_image_plan_uses_canonical_order_and_mandatory_export() -> None:
    settings = ImageWorkflowSettings(
        stages=["upscale", "auto_mask", "restore", "denoise"],
        restore_model_id="restore",
        upscaler_model_id="upscale",
    )
    plan = resolve_image_plan(settings)
    assert plan.stages == ["auto_mask", "denoise", "restore", "upscale", "export"]
    assert plan.labels[0] == "Auto mask"


def test_image_inpaint_warns_without_any_mask_source() -> None:
    settings = ImageWorkflowSettings(stages=["inpaint"], inpaint_prompt="repair")
    plan = resolve_image_plan(settings, has_uploaded_mask=False)
    assert any("uploaded mask" in warning for warning in plan.warnings)


def test_video_stage_selection_is_user_driven_but_export_is_mandatory() -> None:
    assert normalize_video_stages(["sharpen", "deinterlace"]) == ["deinterlace", "sharpen", "export"]
    family = preset_settings("old_family_film")
    stages = video_stages_from_settings(family)
    assert "deinterlace" in stages
    assert "stabilize" in stages
    assert "audio_cleanup" in stages
    assert stages[-1] == "export"


def test_audio_stage_selection_and_plan_order() -> None:
    normalized = normalize_audio_stages(["limiter", "gate", "eq"])
    assert normalized == ["gate", "eq", "limiter", "export"]
    settings = AudioLabSettings(stages=normalized)
    plan = resolve_audio_plan(settings)
    assert plan.stages == normalized


def test_daw_command_parser_handles_measure_transposition() -> None:
    result = parse_audio_command("Modulate the second chorus starting at measure 64 up three semitones")
    assert result.understood
    assert result.operation == "transpose_region"
    assert result.parameters["start_measure"] == 64
    assert result.parameters["semitones"] == 3


def test_daw_command_parser_handles_unison_orchestration() -> None:
    result = parse_audio_command(
        "Add a Cello track in unison of trumpet track 2 one octave below at 60% velocity "
        "with a fade out tapering four beats before the end, pan this track to the right speaker"
    )
    assert result.understood
    assert result.operation == "duplicate_orchestrate_track"
    assert result.parameters["instrument"] == "Cello"
    assert result.parameters["interval_semitones"] == -12
    assert result.parameters["velocity_scale"] == 0.6
    assert result.parameters["pan"] == 1.0


def test_segment_feather_preserves_core_and_adds_a_soft_outer_edge() -> None:
    from PIL import Image, ImageDraw
    from aiwf.services.segment import _feather_mask

    mask = Image.new("L", (41, 41), 0)
    ImageDraw.Draw(mask).rectangle((14, 14, 26, 26), fill=255)
    feathered = _feather_mask(mask, 10)
    assert feathered.getpixel((20, 20)) == 255
    assert 0 < feathered.getpixel((11, 20)) < 255
    assert feathered.getpixel((0, 0)) == 0


def test_image_custom_auto_mask_requires_an_actual_prompt() -> None:
    import pytest
    from pydantic import ValidationError
    from aiwf.core.domain.segment_presets import CUSTOM_SEGMENT_PRESET_ID

    with pytest.raises(ValidationError, match="mask prompt is empty"):
        ImageWorkflowSettings(stages=["auto_mask"], mask_preset=CUSTOM_SEGMENT_PRESET_ID)
