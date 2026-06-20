from types import SimpleNamespace

import pytest

from aiwf.web.app import register_default_tabs
from aiwf.web.registry import WebRegistry
from aiwf.web.tabs.settings import TAB_VISIBILITY_CHOICES


def _noop(_ctx, _tab):
    return None


def test_visible_tabs_hides_secondary_tabs_from_settings():
    registry = WebRegistry()
    registry.tab("Image", order=1)(_noop)
    registry.tab("Models", order=2)(_noop)
    registry.tab("Settings", order=90)(_noop)

    ctx = SimpleNamespace(settings=SimpleNamespace(hidden_tabs=["Models"]))

    visible = [name for name, _builder, _order in registry.visible_tabs(ctx)]

    assert visible == ["Image", "Settings"]


def test_visible_tabs_keeps_pinned_tabs_even_if_hidden():
    registry = WebRegistry()
    registry.tab("Image", order=1)(_noop)
    registry.tab("Settings", order=90)(_noop)

    ctx = SimpleNamespace(settings=SimpleNamespace(hidden_tabs=["Image", "Settings"]))

    visible = [name for name, _builder, _order in registry.visible_tabs(ctx)]

    assert visible == ["Image", "Settings"]


def test_default_tabs_include_shipped_workspace_tabs():
    registry = WebRegistry()
    register_default_tabs(registry)

    names = [name for name, _builder, _order in registry.tabs]

    for expected in ("Image", "Models", "Segment", "Face Swap", "Video", "RIFE", "Settings"):
        assert expected in names
    for hidden_by_default in ("Chat", "Training"):
        assert hidden_by_default not in names

    assert names[:3] == ["Image", "Video", "Models"]


def test_training_tab_stays_disabled_even_with_wip_tabs(monkeypatch):
    monkeypatch.setenv("AIWF_ENABLE_WIP_TABS", "1")
    registry = WebRegistry()

    register_default_tabs(registry)

    names = [name for name, _builder, _order in registry.tabs]
    assert "Training" not in names
    assert "Face Swap" in names
    assert "Chat" in names
    assert "Workflows" in names


def test_settings_visibility_choices_include_secondary_shipped_tabs():
    for expected in ("Models", "Segment", "Enhance", "Face Swap", "Video", "RIFE"):
        assert expected in TAB_VISIBILITY_CHOICES
    for hidden_by_default in ("Chat", "Training"):
        assert hidden_by_default not in TAB_VISIBILITY_CHOICES

    assert TAB_VISIBILITY_CHOICES[:3] == ["Video", "Models", "Enhance"]


def test_wan_video_route_filters_keep_runtime_families_separate():
    from aiwf.core.domain.wan import WAN_RUNTIME_FAST_5B, WAN_RUNTIME_HIGH_LOW, WAN_RUNTIME_HIGH_LOW_FP8
    from aiwf.web.tabs.wan_i2v import (
        _default_offload_for_runtime,
        _model_allowed_for_runtime,
        _offload_choices_for_runtime,
    )

    assert _model_allowed_for_runtime("wan2.2_ti2v_5B_fp16.safetensors", WAN_RUNTIME_FAST_5B)
    assert not _model_allowed_for_runtime("Wan2.2-I2V-A14B-HighNoise-Q3_K_S.gguf", WAN_RUNTIME_FAST_5B)
    assert not _model_allowed_for_runtime("DasiwaWAN22I2V14BLightspeed_boundbiteHighV10.safetensors", WAN_RUNTIME_FAST_5B)

    assert _model_allowed_for_runtime("DasiwaWAN22I2V14BLightspeed_boundbiteHighV10.safetensors", WAN_RUNTIME_HIGH_LOW_FP8)
    assert not _model_allowed_for_runtime("Wan2.2-I2V-A14B-HighNoise-Q3_K_S.gguf", WAN_RUNTIME_HIGH_LOW_FP8)

    assert _model_allowed_for_runtime("Wan2.2-I2V-A14B-HighNoise-Q3_K_S.gguf", WAN_RUNTIME_HIGH_LOW)
    assert not _model_allowed_for_runtime("DasiwaWAN22I2V14BLightspeed_boundbiteHighV10.safetensors", WAN_RUNTIME_HIGH_LOW)

    assert _offload_choices_for_runtime(WAN_RUNTIME_HIGH_LOW_FP8) == [
        ("Tested 14B FP8: streamed group offload", "streamed")
    ]
    assert _default_offload_for_runtime(WAN_RUNTIME_HIGH_LOW_FP8, "balanced") == "streamed"


def test_wan_video_display_path_must_exist(tmp_path):
    from aiwf.infrastructure.video import VideoError
    from aiwf.web.tabs.wan_i2v import _existing_video_output_path

    out = tmp_path / "clip.mp4"
    out.write_bytes(b"fake mp4")

    assert _existing_video_output_path(out, "Wan") == str(out.resolve())
    with pytest.raises(VideoError, match="Wan did not create a video file"):
        _existing_video_output_path(tmp_path / "missing.mp4", "Wan")
