from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
import os
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.generation import GenerationMode, GenerationRequest, GenerationResult, JobRecord, JobState
from aiwf.core.domain.workflow import WorkflowDefinition, WorkflowRunResult, WorkflowStep, WorkflowStepResult, WorkflowStepType


REPO_ROOT = Path(os.environ.get("AIWF_TEST_REPO_ROOT", Path(__file__).parents[2])).resolve()
_MODULE_PATH = REPO_ROOT / "aiwf" / "services" / "pro_workflow_runs.py"
_SPEC = importlib.util.spec_from_file_location("staged_pro_workflow_runs", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(_MODULE)
ProWorkflowRunService = _MODULE.ProWorkflowRunService


def _workflow() -> WorkflowDefinition:
    return WorkflowDefinition(
        name="cpu fake",
        steps=[WorkflowStep(id="generate", type=WorkflowStepType.TXT2IMG, params={"prompt": "synthetic"})],
    )


def _detail() -> WorkflowStepResult:
    return WorkflowStepResult(
        step_id="generate",
        step_type=WorkflowStepType.TXT2IMG,
        label="generate",
        message="synthetic complete",
        seed=17,
    )


def _wait_terminal(service, run_id: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = service.get(run_id)
        if record and record["status"] in {"completed", "failed", "cancelled"}:
            return record
        time.sleep(0.01)
    raise AssertionError(f"workflow did not finish: {service.get(run_id)}")


def test_partial_record_write_removes_temporary_file(tmp_path):
    service = ProWorkflowRunService(tmp_path / "runs", lambda *_args, **_kwargs: None)
    run_id = "d" * 32
    original_write_text = Path.write_text

    def partial_write_text(self, data, encoding=None, errors=None, newline=None):
        if self.suffix == ".tmp":
            with self.open("w", encoding=encoding, errors=errors, newline=newline) as stream:
                stream.write(data[:16])
            raise OSError("synthetic partial write failure")
        return original_write_text(self, data, encoding=encoding, errors=errors, newline=newline)

    try:
        with patch.object(Path, "write_text", partial_write_text):
            try:
                service._write({"run_id": run_id, "status": "queued"})
            except OSError as exc:
                assert "partial write failure" in str(exc)
            else:
                raise AssertionError("synthetic write failure was not raised")
    finally:
        service._executor.shutdown(wait=True)

    assert list(service.root.glob(f".{run_id}.json.*.tmp")) == []
    assert not service._record_path(run_id).exists()


def test_success_persists_receipt_output_and_reloads(tmp_path):
    calls = []
    output = tmp_path / "outputs" / "workflow.png"

    def execute(workflow, *, seed_image=None, on_progress, on_step_complete):
        calls.append(workflow.name)
        on_progress(1, 1, "running generate")
        image = Image.new("RGB", (32, 24), (10, 20, 30))
        on_step_complete(1, _detail())
        return WorkflowRunResult(workflow_name=workflow.name, summary="generate"), [image]

    def save(image, _infotext):
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output)
        return type("Saved", (), {"path": str(output)})()

    service = ProWorkflowRunService(tmp_path / "runs", execute, save_output=save)
    created = service.submit(_workflow(), idempotency_key="success-key")
    finished = _wait_terminal(service, created["run_id"])
    assert finished["status"] == "completed", finished.get("error")
    assert finished["steps"][0]["status"] == "completed"
    assert finished["steps"][0]["receipt"]["seed"] == 17
    assert finished["output_path"] == str(output)
    assert output.is_file()
    assert len(calls) == 1

    reloaded = ProWorkflowRunService(tmp_path / "runs", execute)
    assert reloaded.get(created["run_id"])["status"] == "completed"
    repeated = reloaded.submit(_workflow(), idempotency_key="success-key")
    assert repeated["run_id"] == created["run_id"]
    assert len(calls) == 1


def test_idempotency_key_cannot_be_reused_for_different_payload(tmp_path):
    gate = threading.Event()

    def execute(workflow, *, seed_image=None, on_progress, on_step_complete):
        gate.wait(1)
        image = Image.new("RGB", (16, 16))
        on_progress(1, 1, "running")
        on_step_complete(1, _detail())
        return WorkflowRunResult(workflow_name=workflow.name), [image]

    service = ProWorkflowRunService(tmp_path / "runs", execute)
    service.submit(_workflow(), idempotency_key="same-key")
    changed = WorkflowDefinition(
        name="different",
        steps=[WorkflowStep(id="generate", type=WorkflowStepType.TXT2IMG, params={"prompt": "different"})],
    )
    try:
        try:
            service.submit(changed, idempotency_key="same-key")
        except ValueError as exc:
            assert "different workflow request" in str(exc)
        else:
            raise AssertionError("mismatched idempotency key was accepted")
    finally:
        gate.set()


def test_changed_persisted_seed_image_is_rejected_before_execution(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def execute(workflow, *, seed_image=None, on_progress, on_step_complete):
        calls.append(workflow.name)
        if workflow.name == "blocker":
            entered.set()
            assert release.wait(3)
        on_progress(1, 1, "running generate")
        on_step_complete(1, _detail())
        return WorkflowRunResult(workflow_name=workflow.name), [Image.new("RGB", (8, 8))]

    service = ProWorkflowRunService(tmp_path / "runs", execute, workers=1)
    blocker = WorkflowDefinition(name="blocker", steps=_workflow().steps)
    target = WorkflowDefinition(name="seeded", steps=_workflow().steps)
    first = service.submit(blocker)
    assert entered.wait(5)
    submitted_image = Image.new("RGB", (12, 9), (1, 2, 3))
    second = service.submit(target, seed_image=submitted_image)
    Image.new("RGB", (12, 9), (9, 8, 7)).save(second["source_image_path"], format="PNG")
    try:
        release.set()
        assert _wait_terminal(service, first["run_id"])["status"] == "completed"
        failed = _wait_terminal(service, second["run_id"])
    finally:
        release.set()
        service._executor.shutdown(wait=True)

    assert failed["status"] == "failed"
    assert "inputs changed" in failed["error"]["message"]
    assert calls == ["blocker"]


def test_unchanged_persisted_seed_image_is_passed_to_executor(tmp_path):
    observed = []
    source = Image.new("RGB", (12, 9), (4, 5, 6))

    def execute(workflow, *, seed_image=None, on_progress, on_step_complete):
        observed.append((seed_image.size, seed_image.getpixel((0, 0))))
        on_progress(1, 1, "running generate")
        on_step_complete(1, _detail())
        return WorkflowRunResult(workflow_name=workflow.name), [Image.new("RGB", (8, 8))]

    service = ProWorkflowRunService(tmp_path / "runs", execute)
    created = service.submit(_workflow(), seed_image=source)
    finished = _wait_terminal(service, created["run_id"])
    service._executor.shutdown(wait=True)

    assert finished["status"] == "completed", finished.get("error")
    assert observed == [((12, 9), (4, 5, 6))]


def test_seed_image_outside_run_store_is_rejected(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def execute(workflow, *, seed_image=None, on_progress, on_step_complete):
        calls.append(workflow.name)
        if workflow.name == "blocker":
            entered.set()
            assert release.wait(3)
        on_progress(1, 1, "running generate")
        on_step_complete(1, _detail())
        return WorkflowRunResult(workflow_name=workflow.name), [Image.new("RGB", (8, 8))]

    run_root = tmp_path / "runs"
    service = ProWorkflowRunService(run_root, execute, workers=1)
    blocker = WorkflowDefinition(name="blocker", steps=_workflow().steps)
    target = WorkflowDefinition(name="external", steps=_workflow().steps)
    first = service.submit(blocker)
    assert entered.wait(5)
    source = Image.new("RGB", (12, 9), (1, 2, 3))
    second = service.submit(target, seed_image=source)
    outside_source = tmp_path / "outside.png"
    source.save(outside_source, format="PNG")
    changed_record = service.get(second["run_id"])
    changed_record["source_image_path"] = str(outside_source)
    service._write(changed_record)
    try:
        release.set()
        assert _wait_terminal(service, first["run_id"])["status"] == "completed"
        failed = _wait_terminal(service, second["run_id"])
    finally:
        release.set()
        service._executor.shutdown(wait=True)

    assert failed["status"] == "failed"
    assert "outside the run store" in failed["error"]["message"]
    assert calls == ["blocker"]


def test_cancel_stops_between_nodes_without_repeating_completed_node(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    calls = []

    workflow = WorkflowDefinition(
        name="cancel after generation",
        steps=[
            WorkflowStep(id="generate", type=WorkflowStepType.TXT2IMG, params={"prompt": "synthetic"}),
            WorkflowStep(id="upscale", type=WorkflowStepType.UPSCALE, params={"scale": 2}),
        ],
    )

    def execute(workflow, *, seed_image=None, on_progress, on_step_complete):
        calls.append("generate")
        on_progress(1, 1, "running")
        entered.set()
        assert release.wait(3)
        on_step_complete(1, _detail())
        raise AssertionError("second node should not start after cancellation")

    service = ProWorkflowRunService(tmp_path / "runs", execute)
    created = service.submit(workflow)
    assert entered.wait(5)
    try:
        assert service.cancel(created["run_id"])["status"] == "cancelling"
        assert service._cancel_events[created["run_id"]].is_set()
    finally:
        release.set()
    finished = _wait_terminal(service, created["run_id"])
    assert finished["status"] == "cancelled"
    assert finished["steps"][0]["status"] == "completed"
    assert finished["steps"][1]["status"] == "cancelled"
    assert calls == ["generate"]


def test_cancel_queued_run_releases_token_and_never_executes(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def execute(workflow, *, seed_image=None, on_progress, on_step_complete):
        calls.append(workflow.name)
        if workflow.name == "blocking":
            entered.set()
            assert release.wait(3)
        on_progress(1, 1, "running generate")
        on_step_complete(1, _detail())
        return WorkflowRunResult(workflow_name=workflow.name), [Image.new("RGB", (8, 8))]

    service = ProWorkflowRunService(tmp_path / "runs", execute, workers=1)
    blocking = WorkflowDefinition(name="blocking", steps=_workflow().steps)
    queued = WorkflowDefinition(name="queued", steps=_workflow().steps)
    first = service.submit(blocking)
    assert entered.wait(5)
    second = service.submit(queued)
    try:
        cancelled = service.cancel(second["run_id"])
        assert cancelled["status"] == "cancelled"
        assert second["run_id"] not in service._cancel_events
    finally:
        release.set()

    assert _wait_terminal(service, first["run_id"])["status"] == "completed"
    service._executor.shutdown(wait=True)
    assert service.get(second["run_id"])["status"] == "cancelled"
    assert second["run_id"] not in service._cancel_events
    assert calls == ["blocking"]


def test_scheduler_rejection_persists_failed_receipt_and_cleans_token(tmp_path):
    workflow = _workflow()
    calls = []
    service = ProWorkflowRunService(tmp_path / "runs", lambda *args, **kwargs: calls.append((args, kwargs)))
    with patch.object(service._executor, "submit", side_effect=RuntimeError("executor is shut down")):
        record = service.submit(workflow, idempotency_key="scheduler-rejected")

    assert record["status"] == "failed"
    assert record["error"]["code"] == "scheduler_unavailable"
    assert record["steps"][0]["status"] == "skipped"
    assert record["run_id"] not in service._cancel_events
    assert service.submit(workflow, idempotency_key="scheduler-rejected")["run_id"] == record["run_id"]
    assert calls == []
    service._executor.shutdown(wait=True)


def test_malformed_status_type_fails_stale_recovery_without_crash(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    workflow = _workflow()
    run_id = "9" * 32
    (root / f"{run_id}.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "run_id": run_id,
                "status": ["running"],
                "workflow": workflow.model_dump(mode="json"),
                "request_sha256": _MODULE._canonical_hash(
                    {"workflow": workflow.model_dump(mode="json"), "seed": None}
                ),
                "idempotency_key": "invalid-status-retry",
                "current_step": {"step_id": "active", "index": 1, "total": 1},
                "steps": [{"step_id": "active", "status": "running"}],
            }
        ),
        encoding="utf-8",
    )

    calls = []
    service = ProWorkflowRunService(root, lambda *args, **kwargs: calls.append((args, kwargs)))
    recovered = service.get(run_id)

    assert recovered["status"] == "failed"
    assert recovered["recovery"] == "invalid_record_without_replay"
    assert recovered["steps"][0]["status"] == "failed"
    repeated = service.submit(workflow, idempotency_key="invalid-status-retry")
    assert repeated["run_id"] == run_id
    assert repeated["status"] == "failed"
    assert calls == []


def test_duplicate_malformed_status_keys_fail_closed_without_skipping_recovery(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    workflow = _workflow()
    request_hash = _MODULE._canonical_hash(
        {"workflow": workflow.model_dump(mode="json"), "seed": None}
    )
    run_ids = ("8" * 32, "7" * 32)
    for run_id in run_ids:
        (root / f"{run_id}.json").write_text(
            json.dumps(
                {
                    "schema": 1,
                    "run_id": run_id,
                    "status": ["running"],
                    "workflow": workflow.model_dump(mode="json"),
                    "request_sha256": request_hash,
                    "idempotency_key": "duplicate-invalid-status",
                    "current_step": {"step_id": "active", "index": 1, "total": 1},
                    "steps": [{"step_id": "active", "status": "running"}],
                }
            ),
            encoding="utf-8",
        )

    calls = []
    service = ProWorkflowRunService(root, lambda *args, **kwargs: calls.append((args, kwargs)))
    for run_id in run_ids:
        recovered = service.get(run_id)
        assert recovered["status"] == "failed"
        assert recovered["recovery"] == "invalid_record_without_replay"
    try:
        service.submit(workflow, idempotency_key="duplicate-invalid-status")
    except ValueError as exc:
        assert "multiple recovered workflow runs" in str(exc)
    else:
        raise AssertionError("ambiguous malformed run key was accepted")
    assert calls == []


def test_malformed_step_status_fails_stale_recovery_without_crash(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    run_id = "6" * 32
    (root / f"{run_id}.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "run_id": run_id,
                "status": "running",
                "workflow": _workflow().model_dump(mode="json"),
                "current_step": {"step_id": "active", "index": 2, "total": 2},
                "steps": [
                    {"step_id": "done", "status": "completed", "receipt": {"message": "preserve"}},
                    {"step_id": "active", "status": ["running"]},
                ],
            }
        ),
        encoding="utf-8",
    )

    calls = []
    service = ProWorkflowRunService(root, lambda *args, **kwargs: calls.append((args, kwargs)))
    recovered = service.get(run_id)

    assert recovered["status"] == "failed"
    assert recovered["recovery"] == "invalid_record_without_replay"
    assert recovered["steps"][0]["status"] == "completed"
    assert recovered["steps"][0]["receipt"] == {"message": "preserve"}
    assert recovered["steps"][1]["status"] == "failed"
    assert recovered["steps"][1]["receipt"]["code"] == "invalid_run_record"
    assert calls == []


def test_cancel_at_node_boundary_prevents_next_node_start(tmp_path):
    service_ref = {}
    run_id_ref = {}
    allow_start = threading.Event()
    second_node_calls = []
    workflow = WorkflowDefinition(
        name="cancel before upscale",
        steps=[
            WorkflowStep(id="generate", type=WorkflowStepType.TXT2IMG, params={"prompt": "synthetic"}),
            WorkflowStep(id="upscale", type=WorkflowStepType.UPSCALE, params={"scale": 2}),
        ],
    )

    def execute(workflow, *, seed_image=None, on_progress, on_step_complete):
        assert allow_start.wait(1)
        on_progress(1, 2, "running generate")
        on_step_complete(1, _detail())
        service_ref["service"].cancel(run_id_ref["run_id"])
        on_progress(2, 2, "running upscale")
        second_node_calls.append("upscale")
        raise AssertionError("upscale started after cancellation")

    service = ProWorkflowRunService(tmp_path / "runs", execute)
    service_ref["service"] = service
    created = service.submit(workflow)
    run_id_ref["run_id"] = created["run_id"]
    allow_start.set()
    finished = _wait_terminal(service, created["run_id"])
    assert finished["status"] == "cancelled"
    assert finished["steps"][0]["status"] == "completed"
    assert finished["steps"][1]["status"] == "cancelled"
    assert second_node_calls == []


def test_cancel_after_last_node_boundary_allows_finalization(tmp_path):
    service_ref = {}
    run_id_ref = {}
    finalizing = threading.Event()
    release = threading.Event()

    def execute(workflow, *, seed_image=None, on_progress, on_step_complete):
        on_progress(1, 1, "running generate")
        on_step_complete(1, _detail())
        finalizing.set()
        assert release.wait(2)
        return WorkflowRunResult(workflow_name=workflow.name), [Image.new("RGB", (16, 16))]

    service = ProWorkflowRunService(tmp_path / "runs", execute)
    service_ref["service"] = service
    created = service.submit(_workflow())
    run_id_ref["run_id"] = created["run_id"]
    assert finalizing.wait(1)
    assert service.get(created["run_id"])["status"] == "finalizing"
    assert service.cancel(created["run_id"])["status"] == "finalizing"
    release.set()
    finished = _wait_terminal(service, created["run_id"])
    assert finished["status"] == "completed"


def test_failure_marks_active_node_and_skips_following_nodes(tmp_path):
    workflow = WorkflowDefinition(
        name="fail",
        steps=[
            WorkflowStep(id="generate", type=WorkflowStepType.TXT2IMG, params={"prompt": "synthetic"}),
            WorkflowStep(id="upscale", type=WorkflowStepType.UPSCALE, params={"scale": 2}),
        ],
    )

    def execute(_workflow, *, seed_image=None, on_progress, on_step_complete):
        on_progress(1, 2, "running generate")
        raise RuntimeError("synthetic backend failure")

    service = ProWorkflowRunService(tmp_path / "runs", execute)
    created = service.submit(workflow)
    finished = _wait_terminal(service, created["run_id"])
    assert finished["status"] == "failed"
    assert finished["steps"][0]["status"] == "failed"
    assert finished["steps"][1]["status"] == "skipped"
    assert finished["error"]["code"] == "workflow_failed"


def test_stale_run_is_recovered_without_reexecution(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    workflow = _workflow()
    run_id = "a" * 32
    (root / f"{run_id}.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "run_id": run_id,
                "status": "running",
                "idempotency_key": "stale-key",
                "request_sha256": _MODULE._canonical_hash({"workflow": workflow.model_dump(mode="json"), "seed": None}),
                "workflow": workflow.model_dump(mode="json"),
                "current_step": {"step_id": "generate", "index": 1, "total": 1},
                "steps": [
                    {"step_id": "generate", "status": "running"},
                    {"step_id": "later", "status": "queued"},
                ],
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def execute(*args, **kwargs):
        calls.append(True)
        raise AssertionError("recovered work must not rerun")

    service = ProWorkflowRunService(root, execute)
    recovered = service.get(run_id)
    assert recovered["status"] == "failed"
    assert recovered["recovery"] == "interrupted_without_replay"
    assert recovered["steps"][0]["status"] == "failed"
    assert recovered["steps"][1]["status"] == "skipped"
    repeated = service.submit(workflow, idempotency_key="stale-key")
    assert repeated["run_id"] == run_id
    assert calls == []


def test_malformed_run_id_cannot_redirect_stale_recovery_write(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    sentinel = tmp_path / "sentinel.json"
    sentinel.write_text("preserve-me", encoding="utf-8")
    run_id = "a" * 32
    (root / f"{run_id}.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "run_id": "../sentinel",
                "status": "running",
                "idempotency_key": "redirect-key",
                "workflow": _workflow().model_dump(mode="json"),
                "current_step": {"step_id": "generate", "index": 1, "total": 1},
                "steps": [{"step_id": "generate", "status": "running"}],
            }
        ),
        encoding="utf-8",
    )

    service = ProWorkflowRunService(root, lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError()))

    assert sentinel.read_text(encoding="utf-8") == "preserve-me"
    assert service.get(run_id) is None
    assert service.lookup_idempotency(_workflow(), idempotency_key="redirect-key") is None
    assert not (tmp_path / "sentinel.json.tmp").exists()


def test_duplicate_recovered_idempotency_key_fails_closed(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    workflow = _workflow()
    request_hash = _MODULE._canonical_hash(
        {"workflow": workflow.model_dump(mode="json"), "seed": None}
    )
    for run_id in ("a" * 32, "b" * 32):
        (root / f"{run_id}.json").write_text(
            json.dumps(
                {
                    "schema": 1,
                    "run_id": run_id,
                    "status": "running",
                    "workflow": workflow.model_dump(mode="json"),
                    "request_sha256": request_hash,
                    "idempotency_key": "duplicate-key",
                    "current_step": {"step_id": "generate", "index": 1, "total": 1},
                    "steps": [{"step_id": "generate", "status": "running"}],
                }
            ),
            encoding="utf-8",
        )
    interrupted_id = "c" * 32
    (root / f"{interrupted_id}.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "run_id": interrupted_id,
                "status": "running",
                "workflow": workflow.model_dump(mode="json"),
                "request_sha256": request_hash,
                "idempotency_key": "duplicate-key",
                "current_step": {"step_id": "generate", "index": 1, "total": 1},
                "steps": [{"step_id": "generate", "status": "running"}],
            }
        ),
        encoding="utf-8",
    )

    calls = []
    service = ProWorkflowRunService(root, lambda *args, **kwargs: calls.append((args, kwargs)))
    try:
        service.submit(workflow, idempotency_key="duplicate-key")
    except ValueError as exc:
        assert "multiple recovered workflow runs" in str(exc)
    else:
        raise AssertionError("ambiguous recovered idempotency key was accepted")

    for run_id in ("a" * 32, "b" * 32):
        recovered = service.get(run_id)
        assert recovered["status"] == "failed"
        assert recovered["recovery"] == "interrupted_without_replay"
    interrupted = service.get(interrupted_id)
    assert interrupted["status"] == "failed"
    assert interrupted["recovery"] == "interrupted_without_replay"
    assert calls == []


def test_malformed_stale_receipt_fails_closed_without_crashing_service(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    workflow = _workflow()
    run_id = "e" * 32
    request_hash = _MODULE._canonical_hash(
        {"workflow": workflow.model_dump(mode="json"), "seed": None}
    )
    (root / f"{run_id}.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "run_id": run_id,
                "status": "running",
                "workflow": workflow.model_dump(mode="json"),
                "request_sha256": request_hash,
                "idempotency_key": "malformed-key",
                "current_step": ["not", "an", "object"],
                "steps": [
                    {"step_id": "generate", "status": "running"},
                    "broken step receipt",
                    {"step_id": "later", "status": "queued"},
                ],
            }
        ),
        encoding="utf-8",
    )

    calls = []
    service = ProWorkflowRunService(root, lambda *args, **kwargs: calls.append((args, kwargs)))
    recovered = service.get(run_id)
    repeated = service.submit(workflow, idempotency_key="malformed-key")

    assert recovered["status"] == "failed"
    assert recovered["recovery"] == "invalid_record_without_replay"
    assert [step["status"] for step in recovered["steps"]] == ["failed", "skipped"]
    assert repeated["run_id"] == run_id
    assert calls == []


def test_malformed_workflow_receipt_recovers_to_readable_failed_run(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    run_id = "f" * 32
    (root / f"{run_id}.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "run_id": run_id,
                "status": "running",
                "workflow": ["malformed"],
                "idempotency_key": "invalid-workflow-shape",
                "request_sha256": "untrusted-digest",
                "current_step": {"step_id": "active", "index": 2, "total": 2},
                "steps": [
                    {"step_id": "completed", "status": "completed", "receipt": {"message": "kept"}},
                    {"step_id": "active", "status": "running"},
                ],
            }
        ),
        encoding="utf-8",
    )

    calls = []
    service = ProWorkflowRunService(root, lambda *args, **kwargs: calls.append((args, kwargs)))
    recovered = service.get(run_id)

    assert recovered["status"] == "failed"
    assert recovered["recovery"] == "invalid_record_without_replay"
    assert recovered["workflow"] == {}
    assert recovered["steps"][0]["receipt"] == {"message": "kept"}
    assert recovered["steps"][1]["status"] == "failed"
    try:
        service.lookup_idempotency(_workflow(), idempotency_key="invalid-workflow-shape")
    except ValueError as exc:
        assert "different workflow request" in str(exc)
    else:
        raise AssertionError("invalid stale receipt must not create a missing-record lookup")
    assert calls == []


def test_actual_staged_workflow_service_executor_use_fake_generation(tmp_path):
    def load_staged(module_name: str, path: Path):
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    executor_name = "aiwf.services.workflow_executor"
    workflow_name = "aiwf.services.workflow"
    old_executor = sys.modules.get(executor_name)
    old_workflow = sys.modules.get(workflow_name)
    try:
        load_staged(executor_name, REPO_ROOT / "aiwf" / "services" / "workflow_executor.py")
        workflow_module = load_staged(workflow_name, REPO_ROOT / "aiwf" / "services" / "workflow.py")
        source = Image.new("RGB", (64, 64), (2, 3, 4))

        class FakeGeneration:
            calls = []

            def submit(self, request, init_images=None, mask_images=None):
                self.calls.append(request.mode)
                image = source.copy()
                return JobRecord(
                    request=request,
                    state=JobState.COMPLETED,
                    result=GenerationResult(
                        job_id=__import__("uuid").uuid4(),
                        images=[image],
                        seeds=[19],
                        infotexts=["cpu fake"],
                        mode=request.mode,
                    ),
                )

        class FakeEnhance:
            store = None

            @staticmethod
            def upscale(image, _options):
                return image.resize((image.width * 2, image.height * 2))

        generation = FakeGeneration()
        workflow_service = workflow_module.WorkflowService(
            RuntimeFlags(data_dir=tmp_path),
            UserSettings(save_images=False),
            generation,
            FakeEnhance(),
            object(),
        )
        workflow = WorkflowDefinition(
            name="actual staged executor",
            steps=[
                WorkflowStep(
                    id="generate",
                    type=WorkflowStepType.TXT2IMG,
                    params={"prompt": "synthetic", "save_images": True},
                ),
                WorkflowStep(id="upscale", type=WorkflowStepType.UPSCALE, params={"scale": 2}),
            ],
        )
        service = ProWorkflowRunService(tmp_path / "runs", workflow_service.run)
        record = service.submit(workflow)
        finished = _wait_terminal(service, record["run_id"])
        assert finished["status"] == "completed", finished.get("error")
        assert [step["status"] for step in finished["steps"]] == ["completed", "completed"]
        assert finished["steps"][0]["receipt"]["seed"] == 19
        assert generation.calls == [GenerationMode.TXT2IMG]
    finally:
        if old_executor is None:
            sys.modules.pop(executor_name, None)
        else:
            sys.modules[executor_name] = old_executor
        if old_workflow is None:
            sys.modules.pop(workflow_name, None)
        else:
            sys.modules[workflow_name] = old_workflow

