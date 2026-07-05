"""Background render-queue worker for the paid media center (v5 worker pass).

Design goals:
- The queue file (render_queue.json) stays the single source of truth. The worker
  claims jobs from it, mutates status/progress under a lock, and persists every step.
- Execution is pluggable. Real services (Wan/LTX render, upscale, export) register
  executors by job kind via ``register_executor``. If no executor is registered for
  a kind, the job fails fast with a clear error instead of pretending to render.
- A built-in ``local-echo`` executor exists only for pipeline testing and is
  labeled as such in receipts. Nothing is silently simulated.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# executor signature: fn(job: dict, report: Callable[[int, str], None], ctx: Any) -> dict
Executor = Callable[[dict[str, Any], Callable[[int, str], None], Any], dict[str, Any]]

_EXECUTORS: dict[str, Executor] = {}
_WORKERS: dict[str, "QueueWorker"] = {}
_WORKERS_LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def register_executor(kind: str, fn: Executor) -> None:
    """Register a real executor for a job kind (e.g. 'workflow', 'export', 'upscale')."""
    _EXECUTORS[str(kind)] = fn
    logger.info("paid_worker: registered executor for kind=%s", kind)


def registered_kinds() -> list[str]:
    return sorted(_EXECUTORS)


def _local_echo_executor(job: dict[str, Any], report: Callable[[int, str], None], ctx: Any) -> dict[str, Any]:
    """Test-lane executor. Stages progress and writes a manifest; produces no media."""
    stages = ["validate payload", "resolve inputs", "dispatch", "collect outputs", "finalize"]
    for index, stage in enumerate(stages):
        report(int((index + 1) / len(stages) * 100), stage)
        time.sleep(0.4)
    return {
        "executor": "local-echo",
        "simulated": True,
        "note": "local-echo executor ran; no real render backend is registered for this kind.",
        "payloadKeys": sorted((job.get("payload") or {}).keys()) if isinstance(job.get("payload"), dict) else [],
    }


def _workflow_plan_executor(job: dict[str, Any], report: Callable[[int, str], None], ctx: Any) -> dict[str, Any]:
    """Validation-only workflow-plan executor. It never renders media."""
    payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
    blocks = payload.get("blocks") if isinstance(payload, dict) and isinstance(payload.get("blocks"), list) else []
    report(35, "validate workflow code blocks")
    time.sleep(0.1)
    report(70, "preserve queue order")
    time.sleep(0.1)
    report(100, "write validation-only receipt")
    return {
        "executor": "workflow-plan",
        "rendered": False,
        "note": "Workflow plan was validated and recorded only; no model pipeline was executed.",
        "blockCount": len(blocks),
        "payloadKeys": sorted(payload.keys()) if isinstance(payload, dict) else [],
    }


register_executor("local-echo", _local_echo_executor)
register_executor("workflow-plan", _workflow_plan_executor)


class QueueWorker:
    """Single-threaded worker bound to one runtime directory."""

    def __init__(self, ctx: Any, queue_path: Path, receipts_path: Path, logs_dir: Path) -> None:
        self.ctx = ctx
        self.queue_path = queue_path
        self.receipts_path = receipts_path
        self.logs_dir = logs_dir
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.current_job_id: str | None = None
        self.started_at: str | None = None
        self.completed = 0
        self.failed = 0

    # ---- queue file helpers -------------------------------------------------

    def _load_queue(self) -> dict[str, Any]:
        try:
            if self.queue_path.is_file():
                payload = json.loads(self.queue_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    payload.setdefault("jobs", [])
                    return payload
        except (OSError, json.JSONDecodeError):
            logger.debug("paid_worker: could not read queue file", exc_info=True)
        return {"schema": "aiwf.render-queue.v1", "updatedAt": _now(), "jobs": []}

    def _save_queue(self, queue: dict[str, Any]) -> None:
        queue["updatedAt"] = _now()
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        self.queue_path.write_text(json.dumps(queue, indent=2), encoding="utf-8")

    def _mutate_job(self, job_id: str, **fields: Any) -> dict[str, Any] | None:
        with self.lock:
            queue = self._load_queue()
            for job in queue.get("jobs", []):
                if isinstance(job, dict) and job.get("id") == job_id:
                    job.update(fields)
                    job["updatedAt"] = _now()
                    self._save_queue(queue)
                    return job
        return None

    def _claim_next(self) -> dict[str, Any] | None:
        with self.lock:
            queue = self._load_queue()
            jobs = [j for j in queue.get("jobs", []) if isinstance(j, dict)]
            queued = [j for j in jobs if j.get("status") == "queued"]
            if not queued:
                return None
            queued.sort(key=lambda j: str(j.get("createdAt") or ""))
            job = queued[0]
            job["status"] = "running"
            job["progress"] = 0
            job["startedAt"] = _now()
            job["updatedAt"] = _now()
            self._save_queue(queue)
            return dict(job)

    # ---- logging / receipts -------------------------------------------------

    def _log_path(self, job_id: str) -> Path:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        return self.logs_dir / f"{job_id}.log"

    def _append_log(self, job_id: str, line: str) -> None:
        try:
            with self._log_path(job_id).open("a", encoding="utf-8") as handle:
                handle.write(f"[{_now()}] {line}\n")
        except OSError:
            logger.debug("paid_worker: could not append job log", exc_info=True)

    def _append_receipt(self, entry: dict[str, Any]) -> None:
        with self.lock:
            payload: dict[str, Any] = {"schema": "aiwf.receipts.v1", "receipts": []}
            try:
                if self.receipts_path.is_file():
                    loaded = json.loads(self.receipts_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        payload = loaded
                        payload.setdefault("receipts", [])
            except (OSError, json.JSONDecodeError):
                pass
            payload["receipts"] = [entry, *payload.get("receipts", [])][:500]
            payload["updatedAt"] = _now()
            self.receipts_path.parent.mkdir(parents=True, exist_ok=True)
            self.receipts_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ---- execution ----------------------------------------------------------

    def _is_cancelled(self, job_id: str) -> bool:
        queue = self._load_queue()
        for job in queue.get("jobs", []):
            if isinstance(job, dict) and job.get("id") == job_id:
                return job.get("status") in {"cancelled", "paused"}
        return True

    def run_job(self, job: dict[str, Any]) -> dict[str, Any]:
        job_id = str(job.get("id"))
        kind = str(job.get("kind") or "workflow")
        executor = _EXECUTORS.get(kind)
        self.current_job_id = job_id
        self._append_log(job_id, f"claimed job kind={kind} label={job.get('label')}")

        def report(progress: int, stage: str) -> None:
            if self._stop.is_set() or self._is_cancelled(job_id):
                raise JobCancelled(job_id)
            self._mutate_job(job_id, progress=max(0, min(100, int(progress))), stage=stage)
            self._append_log(job_id, f"{progress:>3}% {stage}")

        try:
            if executor is None:
                raise RuntimeError(
                    f"No executor registered for kind '{kind}'. "
                    f"Registered kinds: {', '.join(registered_kinds()) or 'none'}. "
                    "Register one via aiwf.web.paid_worker.register_executor()."
                )
            result = executor(job, report, self.ctx)
            updated = self._mutate_job(job_id, status="completed", progress=100, result=result, finishedAt=_now())
            self.completed += 1
            self._append_log(job_id, "completed")
            self._append_receipt({"id": f"rcpt-{job_id}", "jobId": job_id, "kind": kind, "label": job.get("label"), "status": "completed", "at": _now(), "result": result})
            return updated or job
        except JobCancelled:
            self._append_log(job_id, "cancelled/paused by user; worker released job")
            return job
        except Exception as exc:  # noqa: BLE001 - worker must survive executor failures
            detail = f"{type(exc).__name__}: {exc}"
            self._append_log(job_id, f"failed: {detail}")
            self._append_log(job_id, traceback.format_exc(limit=6))
            updated = self._mutate_job(job_id, status="failed", error=detail, finishedAt=_now())
            self.failed += 1
            self._append_receipt({"id": f"rcpt-{job_id}", "jobId": job_id, "kind": kind, "label": job.get("label"), "status": "failed", "at": _now(), "error": detail})
            return updated or job
        finally:
            self.current_job_id = None

    def run_next(self) -> dict[str, Any] | None:
        job = self._claim_next()
        if job is None:
            return None
        return self.run_job(job)

    # ---- thread lifecycle ---------------------------------------------------

    def start(self) -> bool:
        if self.is_running:
            return False
        self._stop.clear()
        self.started_at = _now()
        self._thread = threading.Thread(target=self._loop, name="aiwf-paid-queue-worker", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> bool:
        if not self.is_running:
            return False
        self._stop.set()
        return True

    @property
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive() and not self._stop.is_set())

    def _loop(self) -> None:
        logger.info("paid_worker: loop started")
        while not self._stop.is_set():
            try:
                if self.run_next() is None:
                    self._stop.wait(1.5)
            except Exception:  # noqa: BLE001
                logger.exception("paid_worker: loop error")
                self._stop.wait(3.0)
        logger.info("paid_worker: loop stopped")

    def status(self) -> dict[str, Any]:
        return {
            "running": self.is_running,
            "currentJobId": self.current_job_id,
            "startedAt": self.started_at,
            "completed": self.completed,
            "failed": self.failed,
            "registeredKinds": registered_kinds(),
        }


class JobCancelled(Exception):
    def __init__(self, job_id: str) -> None:
        super().__init__(f"job {job_id} cancelled")
        self.job_id = job_id


def get_worker(ctx: Any, queue_path: Path, receipts_path: Path, logs_dir: Path) -> QueueWorker:
    key = str(queue_path.resolve())
    with _WORKERS_LOCK:
        worker = _WORKERS.get(key)
        if worker is None:
            worker = QueueWorker(ctx, queue_path, receipts_path, logs_dir)
            _WORKERS[key] = worker
        return worker
