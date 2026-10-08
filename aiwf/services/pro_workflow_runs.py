from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from PIL import Image

from aiwf.core.domain.workflow import WorkflowDefinition
from aiwf.services.image_artifacts import image_artifact_dimensions


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_hash(payload: Any) -> str:
    packed = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def _image_hash(image: Image.Image) -> str:
    packed = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(str(packed.size).encode("ascii"))
    digest.update(packed.tobytes())
    return digest.hexdigest()


def _is_run_id(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 32 and all(
        character in "0123456789abcdef" for character in value.lower()
    )


class WorkflowRunCancelled(RuntimeError):
    """Raised only at a node boundary after a cancel request."""


class ProWorkflowRunService:
    """Durable Pro run records around the existing WorkflowService executor.

    The service persists intent before scheduling. Any non-terminal record found
    after restart is marked interrupted and is never replayed automatically.
    """

    def __init__(
        self,
        root: Path,
        run_workflow: Callable[..., tuple[Any, list[Image.Image]]],
        *,
        save_output: Callable[[Image.Image, str], Any] | None = None,
        output_root: Path | None = None,
        workers: int = 1,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_workflow = run_workflow
        self.save_output = save_output
        self.output_root = Path(output_root).resolve() if output_root is not None else None
        self._lock = threading.RLock()
        self._cancel_events: dict[str, threading.Event] = {}
        self._executor = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="pro-workflow")
        self._idempotency: dict[str, str] = {}
        self._idempotency_conflicts: set[str] = set()
        self._recover_stale_runs()

    def _record_path(self, run_id: str) -> Path:
        if not _is_run_id(run_id):
            raise ValueError("Workflow run ID is invalid.")
        return self.root / f"{run_id.lower()}.json"

    def _read(self, run_id: str) -> dict[str, Any] | None:
        # On Windows, a reader can temporarily prevent atomic replacement of
        # the same record. Use the service lock for every in-process read.
        with self._lock:
            try:
                record = json.loads(self._record_path(run_id).read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                return None
            if (
                not isinstance(record, dict)
                or not _is_run_id(record.get("run_id"))
                or record["run_id"].lower() != run_id.lower()
                or not isinstance(record.get("workflow"), dict)
                or not isinstance(record.get("steps"), list)
                or any(not isinstance(step, dict) for step in record["steps"])
            ):
                return None
            return record

    def _write(self, record: dict[str, Any]) -> None:
        path = self._record_path(str(record["run_id"]))
        # Use a per-write temporary name so interrupted writes or another
        # process cannot collide on the old fixed `.json.tmp` path. Windows
        # scanners can briefly hold the destination during replacement, so
        # retry only transient access/busy errors for a short bounded window.
        temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temp.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
            for attempt in range(5):
                try:
                    os.replace(temp, path)
                    return
                except PermissionError:
                    if attempt == 4:
                        raise
                    time.sleep(0.01 * (2**attempt))
        finally:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass

    def _recover_stale_runs(self) -> None:
        for path in self.root.glob("*.json"):
            if not _is_run_id(path.stem):
                continue
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                not isinstance(record, dict)
                or not _is_run_id(record.get("run_id"))
                or record["run_id"].lower() != path.stem.lower()
            ):
                continue
            status = record.get("status")
            key = record.get("idempotency_key")
            if key:
                key = str(key)
                run_id = record["run_id"].lower()
                if key not in self._idempotency_conflicts:
                    previous_run_id = self._idempotency.get(key)
                    if previous_run_id is not None and previous_run_id != run_id:
                        self._idempotency.pop(key, None)
                        self._idempotency_conflicts.add(key)
                    else:
                        self._idempotency[key] = run_id
            valid_statuses = {
                "queued",
                "running",
                "cancelling",
                "finalizing",
                "completed",
                "failed",
                "cancelled",
            }
            if not isinstance(status, str) or status not in valid_statuses:
                self._fail_malformed_stale_run(record, record.get("steps"))
                continue
            if status in {"queued", "running", "cancelling", "finalizing"}:
                steps = record.get("steps")
                current_step = record.get("current_step")
                valid_step_statuses = {"queued", "running", "completed", "failed", "cancelled", "skipped"}
                if (
                    not isinstance(steps, list)
                    or any(not isinstance(step, dict) for step in steps)
                    or any(
                        not isinstance(step.get("status"), str)
                        or step["status"] not in valid_step_statuses
                        for step in steps
                        if isinstance(step, dict)
                    )
                    or (current_step is not None and not isinstance(current_step, dict))
                    or not isinstance(record.get("workflow"), dict)
                ):
                    self._fail_malformed_stale_run(record, steps)
                    continue
                current_step_id = str((record.get("current_step") or {}).get("step_id") or "")
                for step in record.get("steps", []):
                    if step.get("status") == "running" and step.get("step_id") == current_step_id:
                        step["status"] = "failed"
                        step["completed_at"] = _utc_now()
                        step["receipt"] = {
                            "error": "Studio restarted while this node was running.",
                            "code": "process_restarted",
                        }
                    elif step.get("status") in {"queued", "running"}:
                        step["status"] = "skipped"
                        step["completed_at"] = _utc_now()
                record["status"] = "failed"
                record["updated_at"] = _utc_now()
                record["completed_at"] = record["updated_at"]
                record["current_step"] = None
                record["error"] = {
                    "code": "process_restarted",
                    "message": "Studio restarted before this workflow completed. It was not replayed.",
                }
                record["recovery"] = "interrupted_without_replay"
                self._write(record)

    def _fail_malformed_stale_run(self, record: dict[str, Any], steps: Any) -> None:
        now = _utc_now()
        valid_steps = [step for step in steps if isinstance(step, dict)] if isinstance(steps, list) else []
        valid_step_statuses = {"queued", "running", "completed", "failed", "cancelled", "skipped"}
        for step in valid_steps:
            step_status = step.get("status")
            if step_status == "running":
                step["status"] = "failed"
                step["completed_at"] = now
                step["receipt"] = {
                    "error": "Studio restarted with a malformed workflow run receipt.",
                    "code": "invalid_run_record",
                }
            elif step_status == "queued":
                step["status"] = "skipped"
                step["completed_at"] = now
            elif not isinstance(step_status, str) or step_status not in valid_step_statuses:
                step["status"] = "failed"
                step["completed_at"] = now
                step["receipt"] = {
                    "error": "Studio restarted with a malformed workflow step receipt.",
                    "code": "invalid_run_record",
                }
        record["steps"] = valid_steps
        if not isinstance(record.get("workflow"), dict):
            record["workflow"] = {}
        record["status"] = "failed"
        record["updated_at"] = now
        record["completed_at"] = now
        record["current_step"] = None
        record["error"] = {
            "code": "invalid_run_record",
            "message": "Studio restarted with a malformed workflow run receipt. It was not replayed.",
        }
        record["recovery"] = "invalid_record_without_replay"
        self._write(record)

    def submit(
        self,
        workflow: WorkflowDefinition,
        *,
        seed_image: Image.Image | None = None,
        idempotency_key: str | None = None,
        step_route_metadata: dict[str, dict[str, str | None]] | None = None,
    ) -> dict[str, Any]:
        workflow_json = workflow.model_dump(mode="json")
        with self._lock:
            if idempotency_key:
                previous = self.lookup_idempotency(workflow, seed_image=seed_image, idempotency_key=idempotency_key)
                if previous:
                    return previous

            request_hash = _canonical_hash(
                {"workflow": workflow_json, "seed": _image_hash(seed_image) if seed_image is not None else None}
            )

            run_id = uuid4().hex
            source_path = None
            if seed_image is not None:
                source_path = self.root / f"{run_id}.source.png"
                seed_image.convert("RGB").save(source_path, format="PNG")
            now = _utc_now()
            record: dict[str, Any] = {
                "schema": 1,
                "run_id": run_id,
                "status": "queued",
                "created_at": now,
                "updated_at": now,
                "workflow": workflow_json,
                "source_image_path": str(source_path) if source_path else None,
                "request_sha256": request_hash,
                "idempotency_key": idempotency_key,
                "steps": [
                    {
                        "step_id": step.id,
                        "type": step.type.value,
                        "label": step.label or step.type.value,
                        **(
                            {
                                "generation": {
                                    "checkpoint_id": str(step.params.get("checkpoint_id") or "").strip() or None,
                                    "mode": step.type.value,
                                    "route": (step_route_metadata or {}).get(step.id, {}).get("route"),
                                    "model_family": (step_route_metadata or {}).get(step.id, {}).get("model_family"),
                                    # Workflow execution has no residency receipt contract yet.
                                    # Keep this unknown until a backend explicitly confirms it.
                                    "resident": None,
                                }
                            }
                            if step.type.value in {"txt2img", "img2img", "inpaint"}
                            else {}
                        ),
                        "status": "queued",
                        "params_sha256": _canonical_hash(step.params),
                        "started_at": None,
                        "completed_at": None,
                        "receipt": None,
                    }
                    for step in workflow.steps
                ],
                "current_step": None,
                "output_path": None,
                "error": None,
            }
            self._write(record)
            if idempotency_key:
                self._idempotency[idempotency_key] = run_id
            self._cancel_events[run_id] = threading.Event()
            try:
                self._executor.submit(self._execute, run_id)
            except Exception as exc:
                self._cancel_events.pop(run_id, None)
                self._finish_error(
                    run_id,
                    "failed",
                    "scheduler_unavailable",
                    f"Workflow run could not be scheduled: {exc}",
                    None,
                )
                return self._read(run_id) or record
            return record

    def lookup_idempotency(
        self,
        workflow: WorkflowDefinition,
        *,
        seed_image: Image.Image | None = None,
        idempotency_key: str | None,
    ) -> dict[str, Any] | None:
        """Return an existing matching request, or reject key reuse.

        Routes can call this before dynamic model-availability checks so a
        retry remains idempotent after the model catalog changes.
        """
        if not idempotency_key:
            return None
        request_hash = _canonical_hash(
            {
                "workflow": workflow.model_dump(mode="json"),
                "seed": _image_hash(seed_image) if seed_image is not None else None,
            }
        )
        with self._lock:
            if idempotency_key in self._idempotency_conflicts:
                raise ValueError("The idempotency key matches multiple recovered workflow runs; refusing an ambiguous retry.")
            previous_id = self._idempotency.get(idempotency_key)
            if not previous_id:
                return None
            previous = self._read(previous_id)
            if previous is None:
                raise RuntimeError("The idempotent workflow receipt is missing.")
            if previous.get("request_sha256") != request_hash:
                raise ValueError("The idempotency key was already used for a different workflow request.")
            return previous

    def get(self, run_id: str) -> dict[str, Any] | None:
        if len(run_id) != 32 or any(ch not in "0123456789abcdef" for ch in run_id.lower()):
            return None
        return self._read(run_id)

    def has_active_runs(self) -> bool:
        """Return whether any queued or executing workflow can still use models."""
        with self._lock:
            for path in self.root.glob("*.json"):
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if isinstance(record, dict) and record.get("status") in {
                    "queued", "running", "cancelling", "finalizing",
                }:
                    return True
        return False

    def cancel(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self.get(run_id)
            if record is None:
                return None
            if record.get("status") == "queued":
                record["status"] = "cancelled"
                record["updated_at"] = _utc_now()
                record["completed_at"] = record["updated_at"]
                for step in record["steps"]:
                    step["status"] = "cancelled"
                    step["completed_at"] = record["completed_at"]
            elif record.get("status") == "running":
                record["status"] = "cancelling"
                record["updated_at"] = _utc_now()
                self._cancel_events.setdefault(run_id, threading.Event()).set()
            self._write(record)
            if record.get("status") == "cancelled":
                # The executor task may still be queued and will return before
                # entering _execute's try/finally. Drop its cancellation token
                # only after the terminal state is durable.
                self._cancel_events.pop(run_id, None)
            return record

    def _execute(self, run_id: str) -> None:
        with self._lock:
            record = self._read(run_id)
            if record is None or record.get("status") != "queued":
                return
            record["status"] = "running"
            record["started_at"] = _utc_now()
            record["updated_at"] = record["started_at"]
            self._write(record)
            cancel_event = self._cancel_events.setdefault(run_id, threading.Event())

        current_index: int | None = None

        def on_progress(index: int, _total: int, _message: str) -> None:
            nonlocal current_index
            now = _utc_now()
            with self._lock:
                if cancel_event.is_set():
                    raise WorkflowRunCancelled("Workflow cancellation requested at a node boundary.")
                latest = self._read(run_id)
                if latest is None:
                    raise RuntimeError("Workflow run receipt disappeared.")
                current_index = index - 1
                step = latest["steps"][current_index]
                step["status"] = "running"
                step["started_at"] = now
                latest["current_step"] = {
                    "step_id": step["step_id"],
                    "index": index,
                    "total": len(latest["steps"]),
                }
                latest["updated_at"] = now
                self._write(latest)

        def on_step_complete(index: int, detail: Any) -> None:
            nonlocal current_index
            completed_index = index - 1
            now = _utc_now()
            with self._lock:
                latest = self._read(run_id)
                if latest is None:
                    raise RuntimeError("Workflow run receipt disappeared.")
                step = latest["steps"][completed_index]
                step["status"] = "completed"
                step["completed_at"] = now
                step["receipt"] = {
                    "message": detail.message,
                    "seed": detail.seed,
                    "image_path": detail.image_path,
                    "infotext_sha256": hashlib.sha256(detail.infotext.encode("utf-8")).hexdigest(),
                }
                if index == len(latest["steps"]):
                    latest["status"] = "finalizing"
                    latest["current_step"] = None
                latest["updated_at"] = now
                self._write(latest)
            current_index = None
            if cancel_event.is_set():
                raise WorkflowRunCancelled("Workflow cancellation requested at a node boundary.")

        try:
            record = self._read(run_id)
            workflow = WorkflowDefinition.model_validate(record["workflow"])
            seed_image = None
            source_path = record.get("source_image_path")
            if source_path:
                resolved_source = Path(source_path).resolve()
                try:
                    resolved_source.relative_to(self.root)
                except ValueError as exc:
                    raise RuntimeError("Workflow source image is outside the run store.") from exc
                with Image.open(resolved_source) as opened:
                    seed_image = opened.convert("RGB")
            expected_request_hash = _canonical_hash(
                {
                    "workflow": workflow.model_dump(mode="json"),
                    "seed": _image_hash(seed_image) if seed_image is not None else None,
                }
            )
            if record.get("request_sha256") != expected_request_hash:
                raise RuntimeError("Persisted workflow inputs changed after submission.")
            workflow_result, images = self.run_workflow(
                workflow,
                seed_image=seed_image,
                on_progress=on_progress,
                on_step_complete=on_step_complete,
            )
            final_image = images[-1] if images else None
            output_path = workflow_result.final_image_path
            output_dimensions = image_artifact_dimensions(output_path) if output_path else None
            publishable = False
            if output_path and output_dimensions is not None:
                try:
                    candidate = Path(output_path).resolve()
                    publishable = self.output_root is None or candidate.is_relative_to(self.output_root)
                except (OSError, ValueError):
                    publishable = False
            if not publishable and final_image is not None and self.save_output is not None:
                saved = self.save_output(final_image, f"Workflow: {workflow.name}")
                output_path = str(getattr(saved, "path", saved))
                output_dimensions = image_artifact_dimensions(output_path)
                try:
                    candidate = Path(output_path).resolve()
                    publishable = (
                        output_dimensions is not None
                        and (self.output_root is None or candidate.is_relative_to(self.output_root))
                    )
                except (OSError, ValueError):
                    publishable = False
            if self.output_root is not None and not publishable:
                raise RuntimeError("Workflow finished without a valid, publishable image output artifact.")
            with self._lock:
                latest = self._read(run_id)
                if latest is None:
                    return
                if cancel_event.is_set():
                    raise WorkflowRunCancelled("Workflow cancellation requested before final receipt commit.")
                for index, step in enumerate(latest["steps"]):
                    step["status"] = "completed"
                    step["completed_at"] = step.get("completed_at") or _utc_now()
                    detail = workflow_result.steps[index] if index < len(workflow_result.steps) else None
                    if detail is not None and step["receipt"] is None:
                        step["receipt"] = {
                            "message": detail.message,
                            "seed": detail.seed,
                            "image_path": detail.image_path,
                            "infotext_sha256": hashlib.sha256(detail.infotext.encode("utf-8")).hexdigest(),
                        }
                latest["status"] = "completed"
                latest["completed_at"] = _utc_now()
                latest["updated_at"] = latest["completed_at"]
                latest["current_step"] = None
                latest["output_path"] = output_path
                latest["summary"] = workflow_result.summary
                self._write(latest)
        except WorkflowRunCancelled as exc:
            self._finish_error(run_id, "cancelled", "cancelled", str(exc), current_index)
        except Exception as exc:
            self._finish_error(run_id, "failed", "workflow_failed", str(exc), current_index)
        finally:
            with self._lock:
                self._cancel_events.pop(run_id, None)

    def _finish_error(
        self,
        run_id: str,
        status: str,
        code: str,
        message: str,
        failed_index: int | None,
    ) -> None:
        with self._lock:
            record = self._read(run_id)
            if record is None:
                return
            now = _utc_now()
            if failed_index is not None and 0 <= failed_index < len(record["steps"]):
                step = record["steps"][failed_index]
                step["status"] = status
                step["completed_at"] = now
                step["receipt"] = {"error": message}
            for step in record["steps"]:
                if step["status"] == "queued":
                    step["status"] = "cancelled" if status == "cancelled" else "skipped"
                    step["completed_at"] = now
            record["status"] = status
            record["updated_at"] = now
            record["completed_at"] = now
            record["current_step"] = None
            record["error"] = {"code": code, "message": message}
            self._write(record)
