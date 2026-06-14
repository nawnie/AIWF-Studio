"""Dev-only diagnostics tests — run with: pytest -m dev"""

from __future__ import annotations

import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiwf.api.v1.client_log import build_client_log_router
from aiwf import __version__
from aiwf.core.config.settings import RuntimeFlags
from aiwf.core.domain.generation import GenerationMode, GenerationRequest, JobRecord
from aiwf.core.events.types import AppStarted
from aiwf.core.events.bus import EventBus
from aiwf.core.events.types import JobFailed, JobQueued
from aiwf.dev.diagnostics import DevDiagnostics, install_dev_diagnostics, trace_model_throughput, trace_safe
from aiwf.services.queue import JobQueue

pytestmark = pytest.mark.dev


def test_dev_diagnostics_writes_json_trace(tmp_path, monkeypatch):
    monkeypatch.setenv("AIWF_DEV_TRACE", "1")
    diag = DevDiagnostics(tmp_path / "outputs")
    diag.trace("test.category", "hello", foo="bar", count=3)

    log_path = tmp_path / "outputs" / "dev-trace.log"
    assert log_path.is_file()
    row = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert row["category"] == "test.category"
    assert row["message"] == "hello"
    assert row["foo"] == "bar"
    assert row["count"] == 3


def test_dev_diagnostics_disabled_by_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AIWF_DEV_TRACE", "0")
    diag = DevDiagnostics(tmp_path / "outputs")
    diag.trace("test.category", "silent")
    assert not (tmp_path / "outputs" / "dev-trace.log").exists()


def test_trace_model_throughput_records_app_version_and_rate(tmp_path, monkeypatch):
    monkeypatch.setenv("AIWF_DEV_TRACE", "1")
    diag = DevDiagnostics(tmp_path / "outputs")
    from aiwf.dev import diagnostics as diagnostics_module

    monkeypatch.setattr(diagnostics_module, "_installed", diag)

    trace_model_throughput(
        kind="txt2img",
        model_id="model-a",
        model_name="Model A",
        elapsed_seconds=2.0,
        units=8,
        units_label="steps",
        app_version="9.9.9",
    )

    row = json.loads((tmp_path / "outputs" / "dev-trace.log").read_text(encoding="utf-8").strip())
    assert row["category"] == "model.rate"
    assert row["app_version"] == "9.9.9"
    assert row["units_per_second"] == 4.0
    assert row["units_label"] == "steps"


def test_install_subscribes_job_events(tmp_path, monkeypatch):
    monkeypatch.setenv("AIWF_DEV_TRACE", "1")
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models")
    events = EventBus()
    queue = JobQueue(events)
    ctx = SimpleNamespace(
        flags=flags,
        events=events,
        generation=SimpleNamespace(active_job=lambda: None),
        settings=SimpleNamespace(last_checkpoint_id=None),
        runtime_port=7860,
    )
    install_dev_diagnostics(ctx)

    request = GenerationRequest(mode=GenerationMode.TXT2IMG, width=512, height=512)
    record = JobRecord(request=request)
    queue.enqueue(record)
    events.publish(JobQueued(record.id, request))
    events.publish(JobFailed(record.id, "boom"))

    log_path = flags.resolved_output_dir() / "dev-trace.log"
    text = log_path.read_text(encoding="utf-8")
    assert "job.queued" in text
    assert "job.failed" in text


def test_install_logs_app_version_on_startup(tmp_path, monkeypatch):
    monkeypatch.setenv("AIWF_DEV_TRACE", "1")
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models")
    events = EventBus()
    ctx = SimpleNamespace(
        flags=flags,
        events=events,
        generation=SimpleNamespace(active_job=lambda: None),
        settings=SimpleNamespace(last_checkpoint_id=None),
        runtime_port=7860,
    )
    install_dev_diagnostics(ctx)
    events.publish(AppStarted())

    row = json.loads((flags.resolved_output_dir() / "dev-trace.log").read_text(encoding="utf-8").strip())
    assert row["category"] == "app.started"
    assert row["app_version"] == __version__


def test_trace_safe_noop_without_install():
    trace_safe("noop", "should not raise")


def test_client_events_endpoint(tmp_path):
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models")
    ctx = SimpleNamespace(flags=flags)
    app = FastAPI()
    app.include_router(build_client_log_router(ctx), prefix="/api/v1")
    client = TestClient(app)

    response = client.post(
        "/api/v1/client-events",
        json={
            "action": "generate_click",
            "detail": "txt2img",
            "session_id": "sess-test",
            "context": {"busy": False, "mode": "txt2img"},
        },
    )

    assert response.status_code == 200
    log_path = tmp_path / "outputs" / "client-events.jsonl"
    assert log_path.is_file()
    row = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert row["action"] == "generate_click"
    assert row["session_id"] == "sess-test"


def test_client_errors_jsonl_includes_context(tmp_path):
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models")
    ctx = SimpleNamespace(flags=flags)
    app = FastAPI()
    app.include_router(build_client_log_router(ctx), prefix="/api/v1")
    client = TestClient(app)

    response = client.post(
        "/api/v1/client-errors",
        json={
            "kind": "unhandledrejection",
            "message": "get_data is null",
            "session_id": str(uuid4()),
            "context": {"recent_actions": [{"name": "generate_click"}]},
        },
    )

    assert response.status_code == 200
    jsonl = tmp_path / "outputs" / "client-errors.jsonl"
    assert jsonl.is_file()
    row = json.loads(jsonl.read_text(encoding="utf-8").strip())
    assert row["message"] == "get_data is null"
    assert row["context"]["recent_actions"][0]["name"] == "generate_click"
