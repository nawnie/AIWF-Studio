from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.engine import EngineSwitchRequest, EngineTenant
from aiwf.core.domain.sana_video import SanaVideoRequest, SanaVideoResult
from aiwf.services.audio import AudioGenerationService
from aiwf.services.engine_supervisor import EngineSupervisor
from aiwf.services.gpu_tenant_lock import GpuTenantLock
from aiwf.services.sana_video import SanaVideoService, SanaVideoUnavailable


def _supervisor() -> EngineSupervisor:
    supervisor = EngineSupervisor(gpu_lock=GpuTenantLock())
    supervisor._flush_cuda = lambda: None  # type: ignore[method-assign]
    return supervisor


def _service(supervisor: EngineSupervisor, *, data_dir: Path, unload=None) -> SanaVideoService:
    return SanaVideoService(
        RuntimeFlags(data_dir=data_dir),
        UserSettings(),
        supervisor=supervisor,
        unload_image_models=unload,
    )


def test_sana_acquires_video_tenant_unloads_images_then_generates_and_releases(monkeypatch, tmp_path):
    supervisor = _supervisor()
    events: list[tuple[str, EngineTenant]] = []
    service = _service(
        supervisor,
        data_dir=tmp_path,
        unload=lambda: events.append(("unload", supervisor.active_tenant)),
    )

    def fake_generate(_request, *, on_progress=None):
        events.append(("generate", supervisor.active_tenant))
        return "synthetic-result"

    monkeypatch.setattr(service, "_generate_for_request", fake_generate)

    result = service.generate(SanaVideoRequest(prompt="synthetic lifecycle"))

    assert result == "synthetic-result"
    assert events == [
        ("unload", EngineTenant.VIDEO),
        ("generate", EngineTenant.VIDEO),
    ]
    assert supervisor.active_tenant == EngineTenant.IDLE
    assert supervisor._gpu_lock.is_free


def test_sana_denial_preserves_existing_tenant_and_skips_unload_and_generation(monkeypatch, tmp_path):
    supervisor = _supervisor()
    assert supervisor.request_switch(
        EngineSwitchRequest(target=EngineTenant.IMAGE, job_id="image-owner")
    ).ok
    events: list[str] = []
    service = _service(supervisor, data_dir=tmp_path, unload=lambda: events.append("unload"))
    monkeypatch.setattr(
        service,
        "_generate_for_request",
        lambda *_args, **_kwargs: pytest.fail("denied Sana request reached generation"),
    )

    with pytest.raises(SanaVideoUnavailable, match="GPU busy"):
        service.generate(SanaVideoRequest(prompt="synthetic lifecycle"))

    assert events == []
    assert supervisor.active_tenant == EngineTenant.IMAGE
    assert supervisor._gpu_lock.active_job_id == "image-owner"


def test_sana_unload_failure_skips_generation_and_releases_video_tenant(monkeypatch, tmp_path):
    supervisor = _supervisor()

    def fail_unload():
        raise RuntimeError("synthetic unload failure")

    service = _service(supervisor, data_dir=tmp_path, unload=fail_unload)
    monkeypatch.setattr(
        service,
        "_generate_for_request",
        lambda *_args, **_kwargs: pytest.fail("generation continued after unload failure"),
    )

    with pytest.raises(SanaVideoUnavailable, match="Could not unload.*synthetic unload failure"):
        service.generate(SanaVideoRequest(prompt="synthetic lifecycle"))

    assert supervisor.active_tenant == EngineTenant.IDLE
    assert supervisor._gpu_lock.is_free


def test_nested_audio_tenant_borrows_sana_video_ownership_without_releasing_it(monkeypatch, tmp_path):
    supervisor = _supervisor()
    service = _service(supervisor, data_dir=tmp_path)
    audio = AudioGenerationService(
        RuntimeFlags(data_dir=tmp_path),
        UserSettings(),
        supervisor=supervisor,
    )
    observed: list[tuple[str, EngineTenant]] = []

    def fake_generate(_request, *, on_progress=None):
        with audio._gpu_tenant("synthetic nested audio callback"):
            observed.append(("audio", supervisor.active_tenant))
        observed.append(("after-audio", supervisor.active_tenant))
        return "synthetic-result"

    monkeypatch.setattr(service, "_generate_for_request", fake_generate)

    result = service.generate(SanaVideoRequest(prompt="synthetic lifecycle"))

    assert result == "synthetic-result"
    assert observed == [
        ("audio", EngineTenant.VIDEO),
        ("after-audio", EngineTenant.VIDEO),
    ]
    assert supervisor.active_tenant == EngineTenant.IDLE
    assert supervisor._gpu_lock.is_free


def test_sana_releases_video_tenant_before_audio_acquires_audio_tenant(monkeypatch, tmp_path):
    supervisor = _supervisor()
    service = _service(supervisor, data_dir=tmp_path)
    events: list[tuple[str, EngineTenant]] = []

    def fake_generate(request, *, on_progress=None):
        events.append((f"video-generate-audio-{request.generate_audio}", supervisor.active_tenant))
        return SanaVideoResult(output_path=str(tmp_path / "silent.mp4"), has_audio=False)

    class FakeAudioGenerationService:
        def __init__(self, *_args, **_kwargs):
            pass

        def _mmaudio_variant_ready(self, _variant):
            return True

        def _mmaudio_runtime_import_error(self):
            return ""

        def generate_and_mux(self, video_path, _options, *, duration_seconds):
            events.append(("audio-before-acquire", supervisor.active_tenant))
            with supervisor.tenant_session(EngineTenant.AUDIO, reason="test audio"):
                events.append(("audio-running", supervisor.active_tenant))
            events.append(("audio-after-release", supervisor.active_tenant))
            return (
                SimpleNamespace(output_path=str(tmp_path / "audio.wav")),
                SimpleNamespace(output_path=str(tmp_path / "muxed.mp4")),
            )

    monkeypatch.setattr(service, "_generate_for_request", fake_generate)
    monkeypatch.setattr("aiwf.services.sana_video.AudioGenerationService", FakeAudioGenerationService)

    result = service.generate(SanaVideoRequest(prompt="synthetic", generate_audio=True))

    assert events == [
        ("video-generate-audio-False", EngineTenant.VIDEO),
        ("audio-before-acquire", EngineTenant.IDLE),
        ("audio-running", EngineTenant.AUDIO),
        ("audio-after-release", EngineTenant.IDLE),
    ]
    assert result.output_path == str(tmp_path / "muxed.mp4")
    assert result.audio_path == str(tmp_path / "audio.wav")
    assert result.video_only_path == str(tmp_path / "silent.mp4")
    assert result.has_audio is True
    receipt = json.loads(Path(result.receipt_path).read_text(encoding="utf-8"))
    assert receipt["status"] == "ok"
    assert receipt["request"]["generate_audio"] is True
    assert receipt["result"]["output_path"] == result.output_path
    assert receipt["result"]["has_audio"] is True
    assert supervisor.active_tenant == EngineTenant.IDLE


def test_sana_audio_failure_receipt_marks_silent_video_as_partial(monkeypatch, tmp_path):
    supervisor = _supervisor()
    service = _service(supervisor, data_dir=tmp_path)
    monkeypatch.setattr(
        service,
        "_generate_for_request",
        lambda *_args, **_kwargs: SanaVideoResult(output_path=str(tmp_path / "silent.mp4")),
    )

    class FailingAudioGenerationService:
        def __init__(self, *_args, **_kwargs):
            pass

        def _mmaudio_variant_ready(self, _variant):
            return True

        def _mmaudio_runtime_import_error(self):
            return ""

        def generate_and_mux(self, *_args, **_kwargs):
            raise RuntimeError("synthetic audio failure")

    monkeypatch.setattr("aiwf.services.sana_video.AudioGenerationService", FailingAudioGenerationService)

    with pytest.raises(SanaVideoUnavailable, match="Silent video preserved"):
        service.generate(SanaVideoRequest(prompt="synthetic", generate_audio=True))

    receipt = json.loads((tmp_path / "_local" / "logs" / "sana_video_latest.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "error"
    assert receipt["request"]["generate_audio"] is True
    assert receipt["partial_artifact"] == {
        "path": str(tmp_path / "silent.mp4"),
        "kind": "video_without_requested_audio",
    }
    assert receipt["error"]["message"] == "synthetic audio failure"


def test_pro_factory_wires_backend_unload_into_sana_service(monkeypatch, tmp_path):
    from aiwf.web import pro_api

    calls = []

    def unload():
        calls.append("unload")

    backend = SimpleNamespace(devices=SimpleNamespace(), unload=unload)
    ctx = SimpleNamespace(
        flags=RuntimeFlags(data_dir=tmp_path),
        settings=UserSettings(),
        generation=SimpleNamespace(backend=backend),
        supervisor=_supervisor(),
        sana_video=None,
    )
    monkeypatch.setattr(pro_api, "_SANA_VIDEO_SERVICES", {})

    service = pro_api._sana_video_service(ctx)

    assert service._unload_image_models is not unload
    service._unload_image_models()
    assert calls == ["unload"]


def test_sana_cached_model_release_borrows_video_tenant_and_confirms_eviction(tmp_path):
    supervisor = _supervisor()
    service = _service(supervisor, data_dir=tmp_path)
    service._empty_cuda_cache = lambda: None
    service._prepared_pipeline = object()
    service._prepared_pipeline_key = ("fixture",)
    service._prepared_pipeline_info = {"quantization": "bf16"}

    with supervisor.tenant_session(
        EngineTenant.VIDEO,
        reason="existing Sana owner",
        job_id="existing-video-owner",
    ):
        assert service.release_cached_model_for_modality_switch() is True
        assert service._prepared_pipeline is None
        assert service._prepared_pipeline_key is None
        assert supervisor.active_tenant == EngineTenant.VIDEO

    assert supervisor.active_tenant == EngineTenant.IDLE
    assert supervisor._gpu_lock.is_free


def test_sana_cached_model_release_refuses_other_tenant_and_keeps_cache(tmp_path):
    supervisor = _supervisor()
    service = _service(supervisor, data_dir=tmp_path)
    service._prepared_pipeline = object()
    service._prepared_pipeline_key = ("fixture",)
    service._prepared_pipeline_info = {"quantization": "bf16"}
    assert supervisor.request_switch(
        EngineSwitchRequest(target=EngineTenant.IMAGE, job_id="image-owner")
    ).ok
    pipeline = service._prepared_pipeline

    assert service.release_cached_model_for_modality_switch() is False
    assert service._prepared_pipeline is pipeline
    assert service._prepared_pipeline_key == ("fixture",)
    assert supervisor.active_tenant == EngineTenant.IMAGE
    assert supervisor._gpu_lock.active_job_id == "image-owner"


def test_sana_cached_model_release_refuses_while_its_operation_lock_is_busy(tmp_path):
    supervisor = _supervisor()
    service = _service(supervisor, data_dir=tmp_path)
    service._prepared_pipeline = object()
    assert service._operation_lock.acquire(blocking=False)
    try:
        assert service.release_cached_model_for_modality_switch() is False
        assert service._prepared_pipeline is not None
        assert supervisor.active_tenant == EngineTenant.IDLE
    finally:
        service._operation_lock.release()
