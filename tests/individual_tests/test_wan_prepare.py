from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.engine import EngineTenant
from aiwf.core.domain.wan import WAN_RUNTIME_FAST_5B, WAN_RUNTIME_HIGH_LOW, WanI2VRequest
from aiwf.infrastructure.wan import pipeline as wan_pipeline
from aiwf.services.wan import WanPreflightResult, WanService


@pytest.fixture(autouse=True)
def _make_device_memory_probe_deterministic(monkeypatch):
    monkeypatch.setattr(
        "aiwf.services.gpu_memory.subprocess.run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError("nvidia-smi test isolation")),
    )


def _fake_torch(monkeypatch):
    fake = ModuleType("torch")
    fake.cuda = SimpleNamespace(is_available=lambda: False, current_device=lambda: 0)
    monkeypatch.setitem(sys.modules, "torch", fake)


def test_backend_prepare_reuses_fast5b_loader_config_without_generation(tmp_path, monkeypatch):
    backend = wan_pipeline.WanI2VBackend()
    _fake_torch(monkeypatch)
    monkeypatch.setattr(wan_pipeline, "_require_wan", lambda: None)
    monkeypatch.setattr(
        wan_pipeline,
        "apply_cuda_vram_reserve",
        lambda **_kwargs: SimpleNamespace(enabled=False, applied=True, message=""),
    )
    received = []

    def ensure(**kwargs):
        received.append(kwargs)
        backend._pipe = object()
        backend._key = ("fast5b", kwargs["model_id"])
        return backend._pipe

    monkeypatch.setattr(backend, "_ensure_single_5b", ensure)
    request = WanI2VRequest(runtime_mode=WAN_RUNTIME_FAST_5B, model_id="C:/models/wan-5b.safetensors")

    result = backend.prepare(request)

    assert result["loaded"] is True
    assert backend.is_prepared(request) is True
    assert len(received) == 1
    assert received[0]["model_id"] == request.model_id
    assert received[0]["flow_shift"] == request.flow_shift
    assert received[0]["sampler"] == request.sampler
    assert not hasattr(backend, "generate_called")


def test_backend_prepare_reuses_high_low_loader_config_without_generation(tmp_path, monkeypatch):
    backend = wan_pipeline.WanI2VBackend()
    _fake_torch(monkeypatch)
    monkeypatch.setattr(wan_pipeline, "_require_wan", lambda: None)
    monkeypatch.setattr(
        wan_pipeline,
        "apply_cuda_vram_reserve",
        lambda **_kwargs: SimpleNamespace(enabled=False, applied=True, message=""),
    )
    received = []

    def ensure(**kwargs):
        received.append(kwargs)
        backend._pipe = object()
        backend._key = ("dual", kwargs["high_noise_model_id"], kwargs["low_noise_model_id"])
        return backend._pipe

    monkeypatch.setattr(backend, "_ensure", ensure)
    request = WanI2VRequest(
        runtime_mode=WAN_RUNTIME_HIGH_LOW,
        high_noise_model_id="C:/models/wan-high.safetensors",
        low_noise_model_id="C:/models/wan-low.safetensors",
    )

    result = backend.prepare(request)

    assert result["loaded"] is True
    assert backend.is_prepared(request) is True
    assert len(received) == 1
    assert received[0]["high_noise_model_id"] == request.high_noise_model_id
    assert received[0]["low_noise_model_id"] == request.low_noise_model_id
    assert received[0]["vae_id"] == request.vae_id


def test_service_prepare_owns_video_tenant_and_unloads_image_before_backend(tmp_path, monkeypatch):
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "out")
    service = WanService(flags, UserSettings())
    events = []
    request = WanI2VRequest(runtime_mode=WAN_RUNTIME_FAST_5B, model_id="wan-5b.safetensors")
    service.preflight = lambda *_args, **_kwargs: WanPreflightResult(
        ok=True, errors=(), warnings=(), model_id="/local/wan-5b.safetensors", audited_request=request,
    )
    service.available = lambda: True
    service._prepare_headroom_issue = lambda _request: None
    service._unload_image_models = lambda: events.append("image-unload")
    service._backend.prepare = lambda selected: events.append(("prepare", selected.model_id)) or {"loaded": True, "cacheMode": "mock"}
    service._backend.is_prepared = lambda selected: selected.model_id == "/local/wan-5b.safetensors"

    class Supervisor:
        def request_switch(self, switch):
            events.append(("tenant", switch.target))
            return SimpleNamespace(ok=True, message="ok")

    service.supervisor = Supervisor()

    result = service.prepare(request)

    assert result["loaded"] is True
    assert result["resident"] is True
    assert result["modelId"] == "/local/wan-5b.safetensors"
    assert events[0][0] == "tenant" and events[0][1] == EngineTenant.VIDEO
    assert events[1] == "image-unload"
    assert events[2] == ("prepare", "/local/wan-5b.safetensors")
    assert events[-1][0] == "tenant" and events[-1][1] == EngineTenant.IDLE


def test_service_prepare_fails_closed_when_image_model_cannot_be_unloaded(tmp_path):
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "out")
    service = WanService(flags, UserSettings())
    request = WanI2VRequest(runtime_mode=WAN_RUNTIME_FAST_5B, model_id="wan-5b.safetensors")
    service.preflight = lambda *_args, **_kwargs: WanPreflightResult(
        ok=True, errors=(), warnings=(), model_id=request.model_id, audited_request=request,
    )
    service.available = lambda: True
    service._prepare_headroom_issue = lambda _request: None
    service._unload_image_models = lambda: (_ for _ in ()).throw(RuntimeError("busy"))
    backend_calls = []
    service._backend.prepare = lambda *_args, **_kwargs: backend_calls.append("prepare")

    try:
        service.prepare(request)
    except wan_pipeline.WanUnavailable as exc:
        assert "Could not unload image models" in str(exc)
    else:
        raise AssertionError("Wan preparation continued without freeing the image backend")
    assert backend_calls == []


def test_service_prepare_fails_closed_when_image_unload_returns_false(tmp_path):
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "out")
    service = WanService(flags, UserSettings())
    request = WanI2VRequest(runtime_mode=WAN_RUNTIME_FAST_5B, model_id="wan-5b.safetensors")
    service.preflight = lambda *_args, **_kwargs: WanPreflightResult(
        ok=True, errors=(), warnings=(), model_id=request.model_id, audited_request=request,
    )
    service.available = lambda: True
    service._prepare_headroom_issue = lambda _request: None
    service._unload_image_models = lambda: False
    backend_calls = []
    service._backend.prepare = lambda *_args, **_kwargs: backend_calls.append("prepare")

    try:
        service.prepare(request)
    except wan_pipeline.WanUnavailable as exc:
        assert "Could not confirm image model release" in str(exc)
    else:
        raise AssertionError("Wan preparation continued after image unload returned false")
    assert backend_calls == []


def test_service_prepare_rejects_unconfirmed_backend_cache_identity(tmp_path):
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "out")
    service = WanService(flags, UserSettings())
    request = WanI2VRequest(runtime_mode=WAN_RUNTIME_FAST_5B, model_id="wan-5b.safetensors")
    service.preflight = lambda *_args, **_kwargs: WanPreflightResult(
        ok=True, errors=(), warnings=(), model_id=request.model_id, audited_request=request,
    )
    service.available = lambda: True
    service._prepare_headroom_issue = lambda _request: None
    service._backend.prepare = lambda _selected: {"loaded": True, "cacheMode": "mock"}
    service._backend.is_prepared = lambda _selected: False

    try:
        service.prepare(request)
    except wan_pipeline.WanUnavailable as exc:
        assert "did not confirm" in str(exc)
    else:
        raise AssertionError("Wan service reported a load without backend cache confirmation")


def test_service_switches_cached_wan_variant_before_headroom_check(tmp_path):
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "out")
    service = WanService(flags, UserSettings())
    service.available = lambda: True
    service._unload_image_models = lambda: True
    cached = {"model": ""}
    events = []

    def preflight(request, **_kwargs):
        return WanPreflightResult(
            ok=True, errors=(), warnings=(), model_id=request.model_id, audited_request=request,
        )

    service.preflight = preflight
    service._backend.has_cached_pipeline = lambda: bool(cached["model"])
    service._backend.is_prepared = lambda request: cached["model"] == request.model_id

    def unload():
        events.append(("unload", cached["model"]))
        cached["model"] = ""

    def prepare(request):
        events.append(("prepare", request.model_id))
        cached["model"] = request.model_id
        return {"loaded": True, "cacheMode": "mock"}

    service._backend.unload = unload
    service._backend.prepare = prepare

    def headroom(request):
        events.append(("headroom", request.model_id, cached["model"]))
        return "low VRAM" if cached["model"] else None

    service._prepare_headroom_issue = headroom
    request_a = WanI2VRequest(runtime_mode=WAN_RUNTIME_FAST_5B, model_id="wan-a.safetensors")
    request_b = WanI2VRequest(runtime_mode=WAN_RUNTIME_FAST_5B, model_id="wan-b.safetensors")

    service.prepare(request_a)
    result = service.prepare(request_b)

    assert result["resident"] is True
    assert cached["model"] == "wan-b.safetensors"
    assert events == [
        ("headroom", "wan-a.safetensors", ""),
        ("prepare", "wan-a.safetensors"),
        ("unload", "wan-a.safetensors"),
        ("headroom", "wan-b.safetensors", ""),
        ("prepare", "wan-b.safetensors"),
    ]


def test_service_does_not_prepare_wan_variant_when_cached_pipeline_eviction_is_unconfirmed(tmp_path):
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "out")
    service = WanService(flags, UserSettings())
    service.available = lambda: True
    service._unload_image_models = lambda: True
    cached = {"model": "wan-a.safetensors"}
    prepare_calls = []
    service.preflight = lambda request, **_kwargs: WanPreflightResult(
        ok=True, errors=(), warnings=(), model_id=request.model_id, audited_request=request,
    )
    service._backend.has_cached_pipeline = lambda: bool(cached["model"])
    service._backend.is_prepared = lambda request: cached["model"] == request.model_id
    service._backend.unload = lambda: None
    service._backend.prepare = lambda request: prepare_calls.append(request.model_id)
    service._prepare_headroom_issue = lambda _request: pytest.fail("headroom must not be checked until eviction is confirmed")
    request_b = WanI2VRequest(runtime_mode=WAN_RUNTIME_FAST_5B, model_id="wan-b.safetensors")

    with pytest.raises(wan_pipeline.WanUnavailable, match="could not confirm release of the previous pipeline"):
        service.prepare(request_b)

    assert cached["model"] == "wan-a.safetensors"
    assert prepare_calls == []


def test_service_prepare_rejects_concurrent_wan_operation(tmp_path):
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "out")
    service = WanService(flags, UserSettings())
    request = WanI2VRequest(runtime_mode=WAN_RUNTIME_FAST_5B, model_id="wan-5b.safetensors")
    service._operation_lock.acquire()
    try:
        try:
            service.prepare(request)
        except wan_pipeline.WanUnavailable as exc:
            assert "another prepare or generation" in str(exc)
        else:
            raise AssertionError("Concurrent Wan preparation was admitted")
    finally:
        service._operation_lock.release()


def test_prepare_headroom_defers_with_less_than_six_gib_free(tmp_path, monkeypatch):
    fake = ModuleType("torch")
    fake.cuda = SimpleNamespace(is_available=lambda: True, mem_get_info=lambda: (int(1.7 * 1024**3), int(24 * 1024**3)))
    monkeypatch.setitem(sys.modules, "torch", fake)
    request = WanI2VRequest(runtime_mode=WAN_RUNTIME_FAST_5B, model_id="wan-5b.safetensors", offload="balanced")
    message = WanService._prepare_headroom_issue(request)
    assert message is not None
    assert "1.7 GB VRAM is free" in message
    assert "requires at least 6 GB" in message


def test_prepare_headroom_requires_twenty_gib_for_resident_mode(tmp_path, monkeypatch):
    fake = ModuleType("torch")
    fake.cuda = SimpleNamespace(is_available=lambda: True, mem_get_info=lambda: (int(12 * 1024**3), int(24 * 1024**3)))
    monkeypatch.setitem(sys.modules, "torch", fake)
    request = WanI2VRequest(runtime_mode=WAN_RUNTIME_FAST_5B, model_id="wan-5b.safetensors", offload="resident")
    message = WanService._prepare_headroom_issue(request)
    assert message is not None
    assert "requires at least 20 GB" in message


def test_service_releases_cached_pipeline_under_video_tenant_and_confirms_eviction(tmp_path):
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "out")
    service = WanService(flags, UserSettings())
    service._backend._pipe = object()
    service._backend._key = ("cached",)
    events = []

    class Supervisor:
        def request_switch(self, switch):
            events.append(switch.target)
            return SimpleNamespace(ok=True, message="ok")

    service.supervisor = Supervisor()

    def unload():
        service._backend._pipe = None
        service._backend._key = None
        service._backend._prepared_identity = None

    service._backend.unload = unload

    assert service.release_cached_model_for_modality_switch() is True
    assert events == [EngineTenant.VIDEO, EngineTenant.IDLE]
    assert service._backend.has_cached_pipeline() is False


def test_service_does_not_evict_cached_pipeline_when_video_tenant_is_busy(tmp_path):
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "out")
    service = WanService(flags, UserSettings())
    service._backend._pipe = object()
    service._backend._key = ("cached",)
    unload_calls = []

    class Supervisor:
        def request_switch(self, _switch):
            return SimpleNamespace(ok=False, message="busy")

    service.supervisor = Supervisor()
    service._backend.unload = lambda: unload_calls.append("unload")

    assert service.release_cached_model_for_modality_switch() is False
    assert unload_calls == []
    assert service._backend.has_cached_pipeline() is True
