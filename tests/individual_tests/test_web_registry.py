from types import SimpleNamespace

import pytest

from aiwf.web.app import register_default_tabs
from aiwf.core.config.settings import UserSettings
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

    for expected in ("Image", "Image Lab", "Chat", "Models", "Segment", "Face Swap", "Video", "RIFE", "Training", "Settings"):
        assert expected in names

    assert names[:3] == ["Image", "Image Lab", "Video"]


def test_default_user_settings_show_core_navigation_only():
    registry = WebRegistry()
    register_default_tabs(registry)
    ctx = SimpleNamespace(settings=UserSettings())

    visible = [name for name, _builder, _order in registry.visible_tabs(ctx)]

    assert visible == ["Image", "Image Lab", "Video", "Sana Video", "Chat", "Video Lab", "Audio Lab", "Training", "Settings"]


def test_topbar_uses_provided_checkpoint_count_without_rescan():
    from aiwf.core.config.settings import RuntimeFlags
    from aiwf.web.app import _topbar_runtime_html

    def fail_scan():
        raise AssertionError("checkpoint scan should not run")

    ctx = SimpleNamespace(
        flags=RuntimeFlags(),
        generation=SimpleNamespace(
            list_checkpoints=fail_scan,
            backend=SimpleNamespace(devices=SimpleNamespace(describe=lambda: "CPU (test)")),
        ),
    )

    html = _topbar_runtime_html(ctx, checkpoint_count=7)

    assert "Models" in html
    assert ">7<" in html


def test_training_tab_is_shipped_even_with_wip_tabs(monkeypatch):
    monkeypatch.setenv("AIWF_ENABLE_WIP_TABS", "1")
    registry = WebRegistry()

    register_default_tabs(registry)

    names = [name for name, _builder, _order in registry.tabs]
    assert "Training" in names
    assert "Face Swap" in names
    assert "Chat" in names
    assert "Workflows" in names


def test_settings_visibility_choices_include_secondary_shipped_tabs():
    for expected in ("Image Lab", "Models", "Chat", "Segment", "Enhance", "Face Swap", "Video", "RIFE", "Training"):
        assert expected in TAB_VISIBILITY_CHOICES

    assert TAB_VISIBILITY_CHOICES[:3] == ["Image Lab", "Video", "Models"]


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


def test_wan_service_factory_passes_image_backend_unload_hook(tmp_path):
    from aiwf.core.config.settings import RuntimeFlags
    from aiwf.web.tabs.wan_i2v import _SERVICES, _service

    calls = []
    backend = SimpleNamespace(unload=lambda: calls.append("unload"))
    ctx = SimpleNamespace(
        flags=RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "out"),
        settings=UserSettings(),
        generation=SimpleNamespace(backend=backend),
        supervisor=None,
        failure_archive=None,
        genlog=None,
    )

    try:
        service = _service(ctx)
        assert callable(service._unload_image_models)

        service._unload_image_models()

        assert calls == ["unload"]
    finally:
        _SERVICES.pop(id(ctx), None)


def test_wan_video_display_path_must_exist(tmp_path):
    from aiwf.infrastructure.video import VideoError
    from aiwf.web.tabs.wan_i2v import _existing_video_output_path

    out = tmp_path / "clip.mp4"
    out.write_bytes(b"fake mp4")

    assert _existing_video_output_path(out, "Wan") == str(out.resolve())
    with pytest.raises(VideoError, match="Wan did not create a video file"):
        _existing_video_output_path(tmp_path / "missing.mp4", "Wan")


def test_wan_video_step_summary_is_route_specific():
    from aiwf.core.domain.wan import WAN_RUNTIME_FAST_5B, WAN_RUNTIME_HIGH_LOW
    from aiwf.web.tabs.wan_i2v import _dual_step_split_from_total, _step_summary_for_runtime

    assert _step_summary_for_runtime(WAN_RUNTIME_FAST_5B, 6, 4) == (6, 1.0)
    assert _step_summary_for_runtime(WAN_RUNTIME_HIGH_LOW, 6, 4) == (10, 0.6)
    assert _dual_step_split_from_total(8) == (4, 4)
    assert _dual_step_split_from_total(9) == (5, 4)


def test_wan_route_switch_filters_low_model_from_normalized_high(tmp_path):
    from aiwf.bootstrap import build_context
    from aiwf.core.config.settings import RuntimeFlags
    from aiwf.core.domain.wan import WAN_RUNTIME_FAST_5B, WAN_RUNTIME_HIGH_LOW, WAN_RUNTIME_HIGH_LOW_FP8
    from aiwf.web.app import create_web_ui

    models = tmp_path / "models"
    safetensors = models / "wan" / "Safetensor"
    gguf = models / "wan" / "GGUF"
    loras = models / "Loras" / "Wan"
    vae = models / "VAE"
    output = tmp_path / "outputs"
    for folder in (safetensors, gguf, loras, vae, output):
        folder.mkdir(parents=True, exist_ok=True)

    for name in (
        "wan2.2_ti2v_5B_fp16.safetensors",
        "wan2.2_i2v_14B_high_noise_fp8.safetensors",
        "wan2.2_i2v_14B_low_noise_fp8.safetensors",
    ):
        (safetensors / name).write_bytes(b"")
    for name in (
        "Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf",
        "Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf",
    ):
        (gguf / name).write_bytes(b"")
    (vae / "wan2.2_vae.safetensors").write_bytes(b"")
    (vae / "wan2.1_vae.safetensors").write_bytes(b"")
    for name in (
        "motion_5b_ti2v_rank16.safetensors",
        "motion_a14b_high_rank16.safetensors",
        "motion_a14b_low_rank16.safetensors",
        "style_neutral_rank16.safetensors",
    ):
        (loras / name).write_bytes(b"")

    ctx = build_context(RuntimeFlags(data_dir=tmp_path, models_dir=models, output_dir=output))
    demo, *_ = create_web_ui(ctx)

    def callback(name):
        for event in demo.fns.values() if isinstance(demo.fns, dict) else demo.fns:
            fn = getattr(event, "fn", None)
            if getattr(fn, "__name__", "") == name:
                return fn
        raise AssertionError(f"Missing Gradio callback: {name}")

    def component_value(label):
        for component in demo.config.get("components", []):
            props = component.get("props") or {}
            if props.get("label") == label:
                return props.get("value")
        raise AssertionError(f"Missing Gradio component: {label}")

    sync_runtime = callback("_sync_runtime_choices")
    fast_high = component_value("5B transformer")
    assert fast_high == "Safetensor/wan2.2_ti2v_5B_fp16.safetensors"

    fp8 = sync_runtime(
        WAN_RUNTIME_HIGH_LOW_FP8,
        WAN_RUNTIME_FAST_5B,
        fast_high,
        None,
        None,
        "",
        "balanced",
        8,
        4,
        512,
        512,
        81,
        False,
        24,
        0,
        1.0,
        "unipc",
        5.0,
    )
    assert fp8[0].get("value") == "Safetensor/wan2.2_i2v_14B_high_noise_fp8.safetensors"
    assert fp8[1].get("value") == "Safetensor/wan2.2_i2v_14B_low_noise_fp8.safetensors"
    assert fp8[1].get("visible") is True
    assert fp8[2].get("visible") is True
    assert fp8[4].get("choices") == [
        "Wan/motion_a14b_high_rank16.safetensors",
        "Wan/style_neutral_rank16.safetensors",
    ]
    assert fp8[5].get("choices") == [
        "Wan/motion_a14b_low_rank16.safetensors",
        "Wan/style_neutral_rank16.safetensors",
    ]
    assert fp8[8].get("value") == "streamed"

    gguf_route = sync_runtime(
        WAN_RUNTIME_HIGH_LOW,
        WAN_RUNTIME_HIGH_LOW_FP8,
        fp8[0].get("value"),
        fp8[1].get("value"),
        fp8[3].get("value"),
        "",
        fp8[8].get("value"),
        fp8[9].get("value"),
        fp8[10].get("value"),
        512,
        512,
        81,
        False,
        24,
        0,
        1.0,
        "euler",
        5.0,
    )
    assert gguf_route[0].get("value") == "GGUF/Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf"
    assert gguf_route[1].get("value") == "GGUF/Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf"
    assert gguf_route[8].get("value") == "model"

    fast_route = sync_runtime(
        WAN_RUNTIME_FAST_5B,
        WAN_RUNTIME_HIGH_LOW,
        gguf_route[0].get("value"),
        gguf_route[1].get("value"),
        gguf_route[3].get("value"),
        "",
        gguf_route[8].get("value"),
        gguf_route[9].get("value"),
        gguf_route[10].get("value"),
        512,
        512,
        81,
        False,
        24,
        0,
        1.0,
        "unipc",
        5.0,
    )
    assert fast_route[0].get("value") == "Safetensor/wan2.2_ti2v_5B_fp16.safetensors"
    assert fast_route[1].get("value") is None
    assert fast_route[1].get("interactive") is False
    assert fast_route[1].get("visible") is False
    assert fast_route[2].get("visible") is False
    assert fast_route[4].get("label") == "5B LoRA"
    assert fast_route[4].get("choices") == [
        "Wan/motion_5b_ti2v_rank16.safetensors",
        "Wan/style_neutral_rank16.safetensors",
    ]
    assert fast_route[4].get("interactive") is True
    assert fast_route[5].get("choices") == []
    assert fast_route[5].get("interactive") is False
    assert fast_route[5].get("visible") is False
    assert fast_route[10].get("visible") is False
    assert fast_route[12].get("visible") is False
    assert fast_route[13].get("visible") is False

    missing_runtime_route = sync_runtime(
        None,
        WAN_RUNTIME_HIGH_LOW,
        gguf_route[0].get("value"),
        gguf_route[1].get("value"),
        gguf_route[3].get("value"),
        "",
        gguf_route[8].get("value"),
        gguf_route[9].get("value"),
        gguf_route[10].get("value"),
        512,
        512,
        81,
        False,
        24,
        0,
        1.0,
        "unipc",
        5.0,
    )
    assert missing_runtime_route[0].get("value") == "Safetensor/wan2.2_ti2v_5B_fp16.safetensors"
    assert missing_runtime_route[1].get("value") is None
    assert missing_runtime_route[1].get("interactive") is False
