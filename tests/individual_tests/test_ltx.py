from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.ltx import (
    LTX_FULL_CHECKPOINT,
    LTX_GEMMA_REPO,
    LTX_PIPELINE_DISTILLED,
    LTX_PIPELINE_ONE_STAGE,
    LtxVideoRequest,
    snap_ltx_num_frames,
)
from aiwf.services.ltx import LtxService, LtxUnavailable
from aiwf.services.worker_tenant import WorkerTenantRegistry, python_exe_for_venv


def test_ltx_frame_count_validation():
    assert snap_ltx_num_frames(80) == 81
    assert LtxVideoRequest(prompt="dance", num_frames=81).num_frames == 81

    with pytest.raises(ValueError, match="8\\*k\\+1"):
        LtxVideoRequest(prompt="dance", num_frames=82)


def test_ltx_request_requires_32_multiple_resolution():
    with pytest.raises(ValueError, match="divisible by 32"):
        LtxVideoRequest(prompt="dance", width=500, height=512)


def test_ltx_service_blocks_when_engine_missing(tmp_path: Path):
    service = LtxService(
        RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs"),
        UserSettings(),
    )

    with pytest.raises(LtxUnavailable, match="LTX 2.3 engine is not ready"):
        service.generate(LtxVideoRequest(prompt="simple dance", pipeline=LTX_PIPELINE_DISTILLED))


def test_ltx_service_blocks_when_models_missing_after_engine_ready(tmp_path: Path):
    worker = tmp_path / "engines" / "ltx" / "worker.py"
    repo = tmp_path / "engines" / "ltx" / "LTX-2"
    python = python_exe_for_venv(tmp_path / "engines" / "ltx" / ".venv")
    worker.parent.mkdir(parents=True)
    repo.mkdir(parents=True)
    python.parent.mkdir(parents=True)
    worker.write_text("print('worker')", encoding="utf-8")
    python.write_text("", encoding="utf-8")
    (tmp_path / "engines.json").write_text(
        json.dumps({"ltx": {"enabled": True}}),
        encoding="utf-8",
    )
    registry = WorkerTenantRegistry(tmp_path)
    service = LtxService(
        RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs"),
        UserSettings(),
        registry=registry,
    )

    with pytest.raises(LtxUnavailable, match="checkpoint missing"):
        service.generate(LtxVideoRequest(prompt="simple dance", pipeline=LTX_PIPELINE_DISTILLED))


def test_ltx_default_launch_uses_installed_one_stage_when_distilled_missing(tmp_path: Path):
    service = LtxService(
        RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs"),
        UserSettings(),
    )
    checkpoint = service.default_checkpoint_path(LTX_PIPELINE_ONE_STAGE)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake")

    assert service.default_launch_pipeline() == LTX_PIPELINE_ONE_STAGE

    payload = service._resolve_request(LtxVideoRequest(prompt="simple dance"))

    assert payload["pipeline"] == LTX_PIPELINE_ONE_STAGE
    assert payload["checkpoint_path"] == str(checkpoint.resolve())


def test_ltx_service_reports_native_audio_when_output_has_audio(tmp_path: Path, monkeypatch):
    worker = tmp_path / "engines" / "ltx" / "worker.py"
    repo = tmp_path / "engines" / "ltx" / "LTX-2"
    python = python_exe_for_venv(tmp_path / "engines" / "ltx" / ".venv")
    worker.parent.mkdir(parents=True)
    repo.mkdir(parents=True)
    python.parent.mkdir(parents=True)
    worker.write_text("print('worker')", encoding="utf-8")
    python.write_text("", encoding="utf-8")
    (tmp_path / "engines.json").write_text(json.dumps({"ltx": {"enabled": True}}), encoding="utf-8")

    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs")
    checkpoint = flags.resolved_models_dir() / "ltx" / "checkpoints" / LTX_FULL_CHECKPOINT
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fake")
    gemma = flags.resolved_models_dir() / "ltx" / "text_encoder" / LTX_GEMMA_REPO.split("/", 1)[1]
    gemma.mkdir(parents=True)
    service = LtxService(flags, UserSettings(), registry=WorkerTenantRegistry(tmp_path))

    def fake_run_worker(_job_id, _command, _events):
        request_files = sorted((flags.resolved_output_dir() / "ltx-videos" / "requests").glob("*.json"))
        payload = json.loads(request_files[-1].read_text(encoding="utf-8"))
        Path(payload["output_path"]).write_bytes(b"video")

    monkeypatch.setattr(service, "_run_worker", fake_run_worker)
    monkeypatch.setattr("aiwf.services.ltx.VideoProcessor.probe", lambda self, path: SimpleNamespace(has_audio=True))

    result = service.generate(LtxVideoRequest(prompt="simple dance", pipeline=LTX_PIPELINE_ONE_STAGE))

    assert result.has_audio is True
    assert result.audio_mode == "native"
    assert "native audio" in result.message
