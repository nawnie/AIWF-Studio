from __future__ import annotations

import json
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


def test_pipeline_registry_marks_qwen_and_sana_missing_until_snapshots_exist(tmp_path: Path):
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models")
    registry = PipelineRegistry(flags, UserSettings())

    by_id = {pipeline.id: pipeline for pipeline in registry.image_pipelines()}

    assert not by_id["qwen-image"].ready
    assert "complete Qwen Image Diffusers snapshot" in by_id["qwen-image"].message
    assert not by_id["sana"].ready
    assert "complete Sana Diffusers snapshot" in by_id["sana"].message


def test_pipeline_registry_marks_incomplete_qwen_snapshot_not_ready(tmp_path: Path):
    models = tmp_path / "models"
    root = models / "qwen-image" / "Diffusers" / "Qwen-Image"
    transformer = root / "transformer"
    transformer.mkdir(parents=True)
    (root / "model_index.json").write_text(json.dumps({"_class_name": "QwenImagePipeline"}), encoding="utf-8")
    (transformer / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 1},
                "weight_map": {
                    "transformer_blocks.0.attn.to_q.weight": "diffusion_pytorch_model-00001-of-00009.safetensors"
                },
            }
        ),
        encoding="utf-8",
    )
    registry = PipelineRegistry(RuntimeFlags(data_dir=tmp_path, models_dir=models), UserSettings())

    qwen = [pipeline for pipeline in registry.image_pipelines() if pipeline.id == "qwen-image"][0]

    assert not qwen.ready
    assert "incomplete Diffusers snapshot" in qwen.message


def test_pipeline_registry_marks_sana_ready_when_snapshot_is_complete(tmp_path: Path):
    models = tmp_path / "models"
    root = models / "sana" / "Diffusers" / "Sana_Sprint_0.6B_1024px_diffusers"
    root.mkdir(parents=True)
    (root / "model_index.json").write_text(json.dumps({"_class_name": "SanaSprintPipeline"}), encoding="utf-8")
    (root / "transformer.safetensors").write_bytes(b"fake")
    registry = PipelineRegistry(RuntimeFlags(data_dir=tmp_path, models_dir=models), UserSettings())

    sana = [pipeline for pipeline in registry.image_pipelines() if pipeline.id == "sana"][0]

    assert sana.ready
    assert str(root.resolve()) in sana.message


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

    assert {"wan-diffusers", "wan-gguf", "sana-video", "ltx-2b-diffusers", "ltx-2.3"}.issubset(ids)


def test_pipeline_registry_marks_wan_routes_missing_until_preflight_ready(tmp_path: Path):
    registry = PipelineRegistry(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"), UserSettings())

    by_id = {pipeline.id: pipeline for pipeline in registry.video_pipelines()}

    assert not by_id["wan-diffusers"].ready
    assert "local Wan TI2V 5B model did not resolve" in by_id["wan-diffusers"].message
    assert not by_id["wan-gguf"].ready
    assert "matched Wan GGUF high and low" in by_id["wan-gguf"].message


def test_pipeline_registry_surfaces_recent_wan_runtime_failure(tmp_path: Path):
    outputs = tmp_path / "outputs"
    index = outputs / "failures" / "index.jsonl"
    index.parent.mkdir(parents=True)
    index.write_text(
        json.dumps(
            {
                "created_at": "2026-06-30T01:28:31Z",
                "kind": "video",
                "stage": "wan-video",
                "request": {"runtime_mode": "fast_5b"},
                "error": {"type": "WanUnavailable", "message": "Video generation failed: Allocation on device"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    registry = PipelineRegistry(
        RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=outputs),
        UserSettings(),
    )

    wan = [pipeline for pipeline in registry.video_pipelines() if pipeline.id == "wan-diffusers"][0]

    assert "last runtime failure 2026-06-30 01:28:31 UTC" in wan.message
    assert "WanUnavailable" in wan.message


def test_pipeline_registry_marks_ltx_missing_until_worker_ready(tmp_path: Path):
    registry = PipelineRegistry(RuntimeFlags(data_dir=tmp_path), UserSettings())

    ltx = [pipeline for pipeline in registry.video_pipelines() if pipeline.id == "ltx-2.3"][0]

    assert not ltx.ready
    assert "enabled=true" in ltx.message or "missing" in ltx.message


def test_pipeline_registry_marks_ltx2b_ready_when_assets_exist(tmp_path: Path):
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models")
    checkpoint = flags.resolved_models_dir() / "ltx" / "checkpoints" / "ltx-video-2b-v0.9.5.safetensors"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake")
    t5 = flags.resolved_models_dir() / "flux" / "Textencoder" / "t5xxl_fp16.safetensors"
    t5.parent.mkdir(parents=True)
    t5.write_bytes(b"fake")
    registry = PipelineRegistry(flags, UserSettings())

    ltx = [pipeline for pipeline in registry.video_pipelines() if pipeline.id == "ltx-2b-diffusers"][0]

    assert ltx.ready
    assert str(checkpoint.resolve()) in ltx.message


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
