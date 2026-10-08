from __future__ import annotations

import importlib.util
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from aiwf.core.domain.workflow import WorkflowDefinition, WorkflowRunResult, WorkflowStep, WorkflowStepResult, WorkflowStepType


ROOT = Path(os.environ.get("AIWF_TEST_REPO_ROOT", Path(__file__).parents[2])).resolve()


def _fake_generation(tmp_path: Path, models: list[tuple[str, str, bool]] | None = None):
    records = []
    for model_id, architecture, installed in models or [("fake-sdxl", "sdxl", True)]:
        path = tmp_path / f"{model_id}.safetensors"
        if installed:
            path.write_bytes(b"fake checkpoint fixture")
        records.append(
            SimpleNamespace(
                id=model_id,
                title=model_id,
                filename=path.name,
                path=str(path),
                architecture=architecture,
            )
        )

    class FakeGeneration:
        def list_checkpoints(self):
            return records

        def resolve_checkpoint(self, checkpoint_id=None):
            if checkpoint_id is None:
                return records[0] if records else None
            return next((item for item in records if item.id == checkpoint_id), None)

    return FakeGeneration()


def _load_staged(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_pro_workflow_submit_status_idempotency_and_unknown_node_rejection(tmp_path, monkeypatch):
    _load_staged(
        "aiwf.services.pro_workflow_runs",
        ROOT / "aiwf" / "services" / "pro_workflow_runs.py",
    )
    api = _load_staged("staged_pro_workflow_api", ROOT / "aiwf" / "web" / "pro_api.py")
    calls = []

    class FakeWorkflowService:
        def run(self, workflow, *, seed_image=None, on_progress=None, on_step_complete=None):
            calls.append(workflow.name)
            on_progress(1, 1, "Running fake node")
            detail = WorkflowStepResult(
                step_id="generate",
                step_type=WorkflowStepType.TXT2IMG,
                label="generate",
                message="synthetic complete",
                seed=23,
            )
            on_step_complete(1, detail)
            return WorkflowRunResult(workflow_name=workflow.name, summary="generate"), [Image.new("RGB", (24, 18))]

    class Flags:
        data_dir = tmp_path

        @staticmethod
        def resolved_output_dir():
            return tmp_path / "outputs"

    ctx = SimpleNamespace(
        flags=Flags(),
        workflows=FakeWorkflowService(),
        generation=_fake_generation(tmp_path),
    )
    release_calls = []
    for name in (
        "_release_cached_audio_model",
        "_release_cached_wan_model",
        "_release_cached_sana_video",
        "_unload_cached_ltx_model",
    ):
        monkeypatch.setattr(api, name, lambda _ctx, name=name: release_calls.append(name) or True)
    app = FastAPI()
    app.include_router(api.build_router(ctx))
    client = TestClient(app)
    body = {
        "workflow": {
            "name": "synthetic API run",
            "steps": [
                {
                    "id": "generate",
                    "type": "txt2img",
                    "params": {"prompt": "fake only", "checkpoint_id": "fake-sdxl"},
                }
            ],
        },
        "idempotencyKey": "api-test-key",
    }
    submitted = client.post("/api/pro/workflows/runs", json=body)
    assert submitted.status_code == 202
    assert release_calls == [
        "_release_cached_audio_model",
        "_release_cached_wan_model",
        "_release_cached_sana_video",
        "_unload_cached_ltx_model",
    ]
    run_id = submitted.json()["runId"]

    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        status = client.get(f"/api/pro/workflows/runs/{run_id}")
        if status.json()["status"] == "completed":
            break
        time.sleep(0.01)
    assert status.status_code == 200
    assert status.json()["steps"][0]["status"] == "completed"
    assert status.json()["steps"][0]["receipt"]["seed"] == 23
    generation = status.json()["steps"][0]["generation"]
    assert generation == {
        "checkpointId": "fake-sdxl",
        "mode": "txt2img",
        "route": "image.sdxl",
        "modelFamily": "sdxl",
        "resident": None,
    }
    # The fake generation backend intentionally exposes no residency API.
    persisted = api._pro_workflow_run_service(ctx).get(run_id)
    assert persisted["steps"][0]["generation"]["resident"] is None

    # Retries return the accepted run even if the installed-model catalog has
    # changed since its original submission.
    ctx.generation = _fake_generation(tmp_path, [])
    repeated = client.post("/api/pro/workflows/runs", json=body)
    assert repeated.status_code == 202
    assert repeated.json()["runId"] == run_id
    assert len(release_calls) == 4, "idempotent retries must not evict models again"
    assert calls == ["synthetic API run"]

    from fastapi import HTTPException

    def refuse_audio_release(_ctx):
        raise HTTPException(status_code=409, detail="Audio still owns its model.")

    monkeypatch.setattr(api, "_release_cached_audio_model", refuse_audio_release)
    blocked_body = {
        **body,
        "idempotencyKey": "blocked-model-release",
        "workflow": {**body["workflow"], "name": "blocked synthetic API run"},
    }
    blocked = client.post("/api/pro/workflows/runs", json=blocked_body)
    assert blocked.status_code == 409
    assert blocked.json()["detail"] == "Audio still owns its model."
    assert calls == ["synthetic API run"], "no workflow generation may start after a failed cache release"

    changed_body = {
        **body,
        "workflow": {
            "name": "synthetic API run",
            "steps": [
                {
                    "id": "generate",
                    "type": "txt2img",
                    "params": {"prompt": "different", "checkpoint_id": "fake-sdxl"},
                }
            ],
        },
    }
    conflict = client.post("/api/pro/workflows/runs", json=changed_body)
    assert conflict.status_code == 409

    from fastapi import HTTPException
    inpaint_sd35 = WorkflowDefinition(
        name="inpaint unsupported family",
        steps=[
            WorkflowStep(
                id="inpaint",
                type=WorkflowStepType.INPAINT,
                params={"checkpoint_id": "fake-sd35"},
            )
        ],
    )
    ctx.generation = _fake_generation(tmp_path, [("fake-sd35", "sd35", True)])
    try:
        api._validate_workflow_generation_targets(ctx, inpaint_sd35)
    except HTTPException as exc:
        assert exc.status_code == 422
    else:
        raise AssertionError("SD 3.5 inpaint was accepted")

    flux_fill_inpaint = WorkflowDefinition(
        name="Flux Fill inpaint",
        steps=[
            WorkflowStep(
                id="inpaint",
                type=WorkflowStepType.INPAINT,
                params={"checkpoint_id": "fake-flux-fill"},
            )
        ],
    )
    ctx.generation = _fake_generation(tmp_path, [("fake-flux-fill", "flux_fill", True)])
    api._validate_workflow_generation_targets(ctx, flux_fill_inpaint)

    unsupported = {
        "workflow": {"name": "unsupported", "steps": [{"id": "x", "type": "video", "params": {}}]}
    }
    rejected = client.post("/api/pro/workflows/runs", json=unsupported)
    assert rejected.status_code == 422

    missing_checkpoint = {
        "workflow": {
            "name": "missing checkpoint",
            "steps": [{"id": "generate", "type": "txt2img", "params": {"prompt": "fake only"}}],
        }
    }
    missing = client.post("/api/pro/workflows/runs", json=missing_checkpoint)
    assert missing.status_code == 422

    multiple_images = {
        "workflow": {
            "name": "multiple image batch",
            "steps": [
                {
                    "id": "generate",
                    "type": "txt2img",
                    "params": {"prompt": "fake only", "checkpoint_id": "fake-sdxl", "batch_size": 2},
                }
            ],
        }
    }
    multiple = client.post("/api/pro/workflows/runs", json=multiple_images)
    assert multiple.status_code == 422
    assert "one image per node" in multiple.json()["detail"]

    special_route = {
        "workflow": {
            "name": "unsupported model family",
            "steps": [
                {
                    "id": "generate",
                    "type": "txt2img",
                    "params": {"prompt": "fake only", "checkpoint_id": "fake-flux"},
                }
            ],
        }
    }
    ctx.generation = _fake_generation(tmp_path, [("fake-flux", "flux", True)])
    unsupported_model = client.post("/api/pro/workflows/runs", json=special_route)
    assert unsupported_model.status_code == 422


def test_concurrent_first_requests_share_one_durable_service(tmp_path):
    _load_staged(
        "aiwf.services.pro_workflow_runs",
        ROOT / "aiwf" / "services" / "pro_workflow_runs.py",
    )
    api = _load_staged("staged_pro_workflow_api_concurrent", ROOT / "aiwf" / "web" / "pro_api.py")

    class Flags:
        data_dir = tmp_path

    ctx = SimpleNamespace(flags=Flags(), workflows=SimpleNamespace(run=lambda *_args, **_kwargs: None))
    with ThreadPoolExecutor(max_workers=12) as pool:
        services = list(pool.map(lambda _index: api._pro_workflow_run_service(ctx), range(48)))
    assert all(service is services[0] for service in services)


def test_receipt_id_does_not_depend_on_intermediate_file_existence(tmp_path):
    api = _load_staged("staged_pro_workflow_api_receipt", ROOT / "aiwf" / "web" / "pro_api.py")

    class Flags:
        @staticmethod
        def resolved_output_dir():
            return tmp_path / "outputs"

    output = tmp_path / "outputs" / "step.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), (12, 34, 56)).save(output, format="PNG")
    ctx = SimpleNamespace(flags=Flags())
    record = {
        "run_id": "a" * 32,
        "workflow": {"name": "receipt"},
        "status": "completed",
        "steps": [
            {
                "step_id": "generate",
                "type": "txt2img",
                "label": "generate",
                "status": "completed",
                "params_sha256": "b" * 64,
                "started_at": "2026-10-03T00:00:00+00:00",
                "completed_at": "2026-10-03T00:00:01+00:00",
                "receipt": {"message": "ok", "image_path": str(output)},
            }
        ],
    }
    before = api._pro_workflow_run_payload(ctx, record)["steps"][0]
    output.unlink()
    after = api._pro_workflow_run_payload(ctx, record)["steps"][0]
    assert before["receiptId"] == after["receiptId"]
    assert before["receipt"]["image_url"] is not None
    assert after["receipt"]["image_url"] is None


def test_cancel_route_reports_cancelling_then_stops_at_node_boundary(tmp_path):
    _load_staged(
        "aiwf.services.pro_workflow_runs",
        ROOT / "aiwf" / "services" / "pro_workflow_runs.py",
    )
    api = _load_staged("staged_pro_workflow_api_cancel", ROOT / "aiwf" / "web" / "pro_api.py")
    entered = threading.Event()
    release = threading.Event()
    second_node_calls = []

    class FakeWorkflowService:
        def run(self, workflow, *, seed_image=None, on_progress=None, on_step_complete=None):
            on_progress(1, 2, "Running fake generation")
            entered.set()
            assert release.wait(5)
            on_step_complete(
                1,
                WorkflowStepResult(
                    step_id="generate",
                    step_type=WorkflowStepType.TXT2IMG,
                    label="generate",
                    message="synthetic complete",
                    seed=23,
                ),
            )
            on_progress(2, 2, "Running fake upscale")
            second_node_calls.append("upscale")
            return WorkflowRunResult(workflow_name=workflow.name), [Image.new("RGB", (24, 18))]

    class Flags:
        data_dir = tmp_path

        @staticmethod
        def resolved_output_dir():
            return tmp_path / "outputs"

    app = FastAPI()
    app.include_router(
        api.build_router(
            SimpleNamespace(
                flags=Flags(),
                workflows=FakeWorkflowService(),
                generation=_fake_generation(tmp_path),
            )
        )
    )
    client = TestClient(app)
    body = {
        "workflow": {
            "name": "cancel API run",
            "steps": [
                {
                    "id": "generate",
                    "type": "txt2img",
                    "params": {"prompt": "fake only", "checkpoint_id": "fake-sdxl"},
                },
                {"id": "upscale", "type": "upscale", "params": {"scale": 2}},
            ],
        }
    }
    submitted = client.post("/api/pro/workflows/runs", json=body)
    assert submitted.status_code == 202
    run_id = submitted.json()["runId"]
    assert entered.wait(1)

    try:
        cancelled = client.post(f"/api/pro/workflows/runs/{run_id}/cancel")
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelling"
    finally:
        release.set()

    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        status = client.get(f"/api/pro/workflows/runs/{run_id}")
        if status.json()["status"] == "cancelled":
            break
        time.sleep(0.01)
    assert status.json()["status"] == "cancelled"
    assert status.json()["steps"][0]["status"] == "completed"
    assert status.json()["steps"][1]["status"] == "cancelled"
    assert second_node_calls == []

