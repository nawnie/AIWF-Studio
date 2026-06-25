from __future__ import annotations

import json
from types import SimpleNamespace

import gradio as gr

from aiwf.services.audio_lab import AudioLabService
from aiwf.web.registry import WebRegistry
from aiwf.web.tabs.audio import _build_mix_panel
from aiwf.web.tabs.image_workflow import build_image_workflow_panel
from aiwf.web.tabs.video_lab import register_video_lab


def _item(item_id: str, title: str, *, label: str | None = None):
    return SimpleNamespace(id=item_id, title=title, label=label or title)


def _context(tmp_path):
    class Flags:
        def resolved_output_dir(self):
            return str(tmp_path / "outputs")

    return SimpleNamespace(
        flags=Flags(),
        settings=SimpleNamespace(last_checkpoint_id=None, default_sampler="euler_a", hidden_tabs=[]),
        generation=SimpleNamespace(
            list_checkpoints=lambda: [_item("checkpoint", "Checkpoint")],
            list_samplers=lambda: [_item("euler_a", "Euler A", label="Euler A")],
        ),
        enhance=SimpleNamespace(
            list_restorers=lambda: [_item("codeformer", "CodeFormer")],
            list_upscalers=lambda: [_item("realesrgan", "RealESRGAN")],
        ),
        segment=SimpleNamespace(list_models=lambda: [_item("sam", "SAM")]),
    )


def _callbacks(demo: gr.Blocks, suffix: str):
    result = [fn for fn in demo.fns.values() if fn.fn and fn.fn.__qualname__.endswith(suffix)]
    assert result, f"Callback not registered: {suffix}"
    return result


def _assert_contract(fn, args) -> tuple | list | object:
    result = fn.fn(*args)
    count = len(result) if isinstance(result, (tuple, list)) else 1
    assert count == len(fn.outputs)
    return result


def test_image_video_audio_dynamic_stage_callbacks_match_outputs(tmp_path) -> None:
    ctx = _context(tmp_path)

    with gr.Blocks() as image_demo:
        build_image_workflow_panel(ctx)
    image_visibility = _callbacks(image_demo, ".<locals>._stage_visibility")[0]
    _assert_contract(image_visibility, [["auto_mask", "tone"]])
    image_preset = _callbacks(image_demo, ".<locals>._apply_preset")[0]
    _assert_contract(image_preset, ["old_photo"])
    image_plan = _callbacks(image_demo, ".<locals>._plan")[0]
    image_values = [component.value for component in image_plan.inputs]
    image_result = _assert_contract(image_plan, image_values)
    assert json.loads(image_result[0])["resolved_order"][-1] == "export"

    registry = WebRegistry()
    register_video_lab(registry)
    _name, video_builder, _order = registry.tabs[0]
    with gr.Blocks() as video_demo:
        video_builder(ctx, None)
    video_visibility = _callbacks(video_demo, ".<locals>._stage_visibility")[0]
    _assert_contract(video_visibility, [["deinterlace", "audio_cleanup"]])
    video_preset = _callbacks(video_demo, ".<locals>._apply_preset")[0]
    result = _assert_contract(video_preset, ["old_family_film"])
    assert len(result) >= 50  # preset applies stage-specific parameters, not just booleans

    with gr.Blocks() as audio_demo:
        _build_mix_panel(ctx, AudioLabService(tmp_path / "outputs"))
    audio_visibility = _callbacks(audio_demo, ".<locals>._stage_visibility")[0]
    _assert_contract(audio_visibility, [["gate", "eq", "pan"]])
    audio_preset = _callbacks(audio_demo, ".<locals>._apply_preset")[0]
    _assert_contract(audio_preset, ["music_sweeten"])
    audio_plan = _callbacks(audio_demo, ".<locals>._plan")[0]
    audio_values = [component.value for component in audio_plan.inputs]
    audio_result = _assert_contract(audio_plan, audio_values)
    assert json.loads(audio_result[0])["resolved_order"][-1] == "export"
