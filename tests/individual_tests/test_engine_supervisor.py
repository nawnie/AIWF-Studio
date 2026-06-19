from __future__ import annotations

import threading
import time

from aiwf.core.domain.engine import EngineSwitchRequest, EngineTenant
from aiwf.core.domain.job_status import JobPhase
from aiwf.services import engine_supervisor as engine_supervisor_module
from aiwf.services.engine_supervisor import EngineSupervisor
from aiwf.services.gpu_tenant_lock import GpuTenantLock


def _supervisor(monkeypatch) -> EngineSupervisor:
    supervisor = EngineSupervisor(gpu_lock=GpuTenantLock())
    monkeypatch.setattr(supervisor, "_flush_cuda", lambda: None)
    return supervisor


def test_request_switch_denies_different_gpu_heavy_tenant(monkeypatch):
    supervisor = _supervisor(monkeypatch)

    first = supervisor.request_switch(
        EngineSwitchRequest(target=EngineTenant.IMAGE, job_id="image-1")
    )
    second = supervisor.request_switch(
        EngineSwitchRequest(target=EngineTenant.VIDEO, job_id="video-1")
    )

    assert first.ok
    assert not second.ok
    assert second.active == EngineTenant.IMAGE
    assert "image" in second.message.lower()


def test_request_switch_denies_same_tenant_different_job(monkeypatch):
    supervisor = _supervisor(monkeypatch)

    first = supervisor.request_switch(
        EngineSwitchRequest(target=EngineTenant.IMAGE, job_id="image-1")
    )
    second = supervisor.request_switch(
        EngineSwitchRequest(target=EngineTenant.IMAGE, job_id="image-2")
    )

    assert first.ok
    assert not second.ok


def test_request_switch_is_idempotent_for_same_owner(monkeypatch):
    supervisor = _supervisor(monkeypatch)

    first = supervisor.request_switch(
        EngineSwitchRequest(target=EngineTenant.ENHANCE, job_id="enhance-1")
    )
    second = supervisor.request_switch(
        EngineSwitchRequest(target=EngineTenant.ENHANCE, job_id="enhance-1")
    )

    assert first.ok
    assert second.ok
    assert supervisor.active_tenant == EngineTenant.ENHANCE


def test_tenant_session_is_reentrant_for_same_thread_owner(monkeypatch):
    supervisor = _supervisor(monkeypatch)

    with supervisor.tenant_session(EngineTenant.ENHANCE, reason="outer") as outer:
        assert supervisor.active_tenant == EngineTenant.ENHANCE
        with supervisor.tenant_session(EngineTenant.ENHANCE, reason="inner") as inner:
            assert inner == outer
            assert supervisor.active_tenant == EngineTenant.ENHANCE
        assert supervisor.active_tenant == EngineTenant.ENHANCE

    assert supervisor.active_tenant == EngineTenant.IDLE


def test_tenant_session_allows_nested_gpu_work_under_outer_owner(monkeypatch):
    supervisor = _supervisor(monkeypatch)

    with supervisor.tenant_session(EngineTenant.IMAGE, reason="outer image") as outer:
        assert supervisor.active_tenant == EngineTenant.IMAGE
        with supervisor.tenant_session(EngineTenant.ENHANCE, reason="postprocess") as nested:
            assert nested == outer
            assert supervisor.active_tenant == EngineTenant.IMAGE
        assert supervisor.active_tenant == EngineTenant.IMAGE

    assert supervisor.active_tenant == EngineTenant.IDLE


def test_borrow_active_tenant_allows_nested_service_call(monkeypatch):
    supervisor = _supervisor(monkeypatch)
    assert supervisor.request_switch(
        EngineSwitchRequest(target=EngineTenant.IMAGE, job_id="image-1")
    ).ok

    with supervisor.borrow_active_tenant(EngineTenant.IMAGE, job_id="image-1") as owner:
        with supervisor.tenant_session(EngineTenant.ENHANCE, reason="postprocess") as nested:
            assert owner == "image-1"
            assert nested == owner
            assert supervisor.active_tenant == EngineTenant.IMAGE

    assert supervisor.request_switch(
        EngineSwitchRequest(target=EngineTenant.IDLE, job_id="image-1")
    ).ok
    assert supervisor.active_tenant == EngineTenant.IDLE


def test_request_switch_releases_owner_then_allows_next_tenant(monkeypatch):
    supervisor = _supervisor(monkeypatch)

    assert supervisor.request_switch(
        EngineSwitchRequest(target=EngineTenant.IMAGE, job_id="image-1")
    ).ok
    assert supervisor.request_switch(
        EngineSwitchRequest(target=EngineTenant.IDLE, job_id="image-1")
    ).ok
    next_result = supervisor.request_switch(
        EngineSwitchRequest(target=EngineTenant.VIDEO, job_id="video-1")
    )

    assert next_result.ok
    assert supervisor.active_tenant == EngineTenant.VIDEO


def test_request_switch_waits_when_allowed(monkeypatch):
    supervisor = _supervisor(monkeypatch)
    assert supervisor.request_switch(
        EngineSwitchRequest(target=EngineTenant.IMAGE, job_id="image-1")
    ).ok
    result_holder = []

    def wait_for_video():
        result_holder.append(
            supervisor.request_switch(
                EngineSwitchRequest(
                    target=EngineTenant.VIDEO,
                    job_id="video-1",
                    allow_wait=True,
                )
            )
        )

    thread = threading.Thread(target=wait_for_video)
    thread.start()
    time.sleep(0.1)
    assert result_holder == []

    supervisor.request_switch(
        EngineSwitchRequest(target=EngineTenant.IDLE, job_id="image-1")
    )
    thread.join(timeout=2.0)

    assert result_holder and result_holder[0].ok
    assert supervisor.active_tenant == EngineTenant.VIDEO


class _FakeOllama:
    def __init__(self, ok: bool):
        self.ok = ok
        self.unloaded: list[str] = []

    def unload(self, model: str) -> bool:
        self.unloaded.append(model)
        return self.ok


def test_chat_preemption_denies_when_unload_fails(monkeypatch):
    supervisor = _supervisor(monkeypatch)
    client = _FakeOllama(ok=False)
    supervisor.set_ollama_client(client)
    supervisor.set_chat_model("llama3:8b")
    assert supervisor.request_switch(
        EngineSwitchRequest(target=EngineTenant.CHAT, job_id="chat")
    ).ok

    result = supervisor.request_switch(
        EngineSwitchRequest(target=EngineTenant.IMAGE, job_id="image-1")
    )

    assert not result.ok
    assert client.unloaded == ["llama3:8b"]
    assert supervisor.active_tenant == EngineTenant.CHAT


def test_chat_preemption_allows_heavy_tenant_after_unload(monkeypatch):
    supervisor = _supervisor(monkeypatch)
    client = _FakeOllama(ok=True)
    supervisor.set_ollama_client(client)
    supervisor.set_chat_model("llama3:8b")
    assert supervisor.request_switch(
        EngineSwitchRequest(target=EngineTenant.CHAT, job_id="chat")
    ).ok

    result = supervisor.request_switch(
        EngineSwitchRequest(target=EngineTenant.IMAGE, job_id="image-1")
    )

    assert result.ok
    assert client.unloaded == ["llama3:8b"]
    assert supervisor.active_tenant == EngineTenant.IMAGE


class _SilentFakeProcess:
    pid = 12345

    def __init__(self):
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def send_signal(self, _signal):
        self.terminated = True
        self.returncode = -15

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.returncode = -9


def test_worker_heartbeat_timeout_fails_silent_job(monkeypatch, tmp_path):
    monkeypatch.setattr(engine_supervisor_module, "SUBPROCESS_HEARTBEAT_TIMEOUT", 0.01)
    supervisor = _supervisor(monkeypatch)
    job_id = supervisor.begin_job("kohya", tmp_path)
    supervisor.mark_running(job_id, "running")
    proc = _SilentFakeProcess()

    with supervisor._lock:
        supervisor._procs[job_id] = proc
        supervisor._worker_last_event_at[job_id] = time.monotonic() - 1.0

    supervisor._monitor_worker_heartbeat(job_id, proc)

    record = supervisor.get_job(job_id)
    assert proc.terminated
    assert record is not None
    assert record.phase == JobPhase.FAILED
    assert "heartbeat" in record.error_detail.lower()
