from __future__ import annotations

from pathlib import Path

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.services.pipeline_registry import PipelineRegistry


def test_pipeline_registry_lists_image_launch_choices(tmp_path: Path):
    registry = PipelineRegistry(RuntimeFlags(data_dir=tmp_path), UserSettings())

    choices = registry.launch_choices()

    assert ("Diffusers pipeline (default)", "diffusers") in choices
    assert ("ONNX Runtime pipeline", "onnx") in choices


def test_pipeline_registry_lists_qwen_nunchaku_image_pipeline(tmp_path: Path):
    registry = PipelineRegistry(RuntimeFlags(data_dir=tmp_path), UserSettings())

    ids = {pipeline.id for pipeline in registry.image_pipelines()}

    assert {"diffusers", "qwen-image", "qwen-nunchaku", "sana", "onnx"}.issubset(ids)


def test_pipeline_registry_reports_missing_default_onnx_folder(tmp_path: Path):
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models")
    registry = PipelineRegistry(flags, UserSettings())

    text = registry.status_markdown()

    assert "ONNX Runtime pipeline" in text
    assert "Needs setup" in text
    assert str((tmp_path / "models" / "onnx").resolve()) in text


def test_pipeline_registry_accepts_configured_onnx_folder(tmp_path: Path):
    onnx_root = tmp_path / "onnx-models"
    onnx_root.mkdir()
    registry = PipelineRegistry(
        RuntimeFlags(data_dir=tmp_path),
        UserSettings(onnx_model_dir=str(onnx_root)),
    )

    onnx = [pipeline for pipeline in registry.image_pipelines() if pipeline.id == "onnx"][0]

    assert onnx.ready
    assert str(onnx_root.resolve()) in onnx.message


def test_pipeline_registry_lists_wan_diffusers_and_gguf_methods(tmp_path: Path):
    registry = PipelineRegistry(RuntimeFlags(data_dir=tmp_path), UserSettings())

    ids = {pipeline.id for pipeline in registry.video_pipelines()}

    assert {"wan-diffusers", "wan-gguf", "sana-video", "ltx-2.3"}.issubset(ids)


def test_pipeline_registry_marks_ltx_missing_until_worker_ready(tmp_path: Path):
    registry = PipelineRegistry(RuntimeFlags(data_dir=tmp_path), UserSettings())

    ltx = [pipeline for pipeline in registry.video_pipelines() if pipeline.id == "ltx-2.3"][0]

    assert not ltx.ready
    assert "enabled=true" in ltx.message or "missing" in ltx.message


def test_pipeline_registry_marks_sana_video_waiting_for_snapshot(tmp_path: Path):
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models")
    registry = PipelineRegistry(flags, UserSettings())

    sana_video = [pipeline for pipeline in registry.video_pipelines() if pipeline.id == "sana-video"][0]

    assert not sana_video.ready
    assert "SANA-Video snapshot" in sana_video.message


def test_pipeline_registry_marks_qwen_nunchaku_missing_until_runtime_ready(tmp_path: Path):
    registry = PipelineRegistry(RuntimeFlags(data_dir=tmp_path), UserSettings())

    qwen = [pipeline for pipeline in registry.image_pipelines() if pipeline.id == "qwen-nunchaku"][0]

    assert not qwen.ready
    assert "missing" in qwen.message
