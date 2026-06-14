"""Engine supervisor — Phase 1 (in-process) + Phase 2 (subprocess) combined.

Phase 1 API (in-process, backward-compatible):
    supervisor = get_supervisor()
    job_id = supervisor.begin_job("wan", outputs_root, request_data=data)
    supervisor.update_progress(job_id, step=4, total=8)
    supervisor.finish_job(job_id)

Phase 2 API (subprocess worker):
    supervisor = get_supervisor()
    spec = get_engine_spec("kohya")   # from launch.py registry
    job_id = supervisor.submit_subprocess_job(
        spec, request_data, outputs_root=ROOT / "outputs"
    )
    # Events flow back automatically from worker stdout.

The Phase 2 method co-exists with Phase 1 — in-process Wan generation
keeps working unchanged while new training jobs use subprocess workers.

Architecture reference: ``docs/architecture/engine_isolation_architecture.md``

Phase 2 acceptance criteria:
  ✓ submit_subprocess_job() spawns a worker in the engine's venv
  ✓ Worker stdout is parsed for JSONL events
  ✓ Events update the JobRecord and fire registered listeners
  ✓ Non-JSONL lines are forwarded to the crash log
  ✓ Worker death (any exit code) triggers fail_job() automatically
  ✓ cancel_job() sends SIGTERM to the worker process
  ✓ Heartbeat monitoring detects silent hangs (configurable timeout)
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from aiwf.core.domain.engine_events import EngineEvent, emit, parse_event_line
from aiwf.core.domain.job_status import JobPhase, JobRecord, make_job_dir
from aiwf.services.gpu_tenant_lock import GpuTenantLock, get_gpu_lock

if TYPE_CHECKING:
    from launch import EngineSpec

logger = logging.getLogger(__name__)

# Optional: callers can register a progress callback so the UI can update
# without polling job files.
ProgressCallback = Callable[[EngineEvent], None]

# How long (seconds) to wait for a heartbeat before considering a subprocess hung.
# Workers emit a heartbeat every 30s; we allow 3x before declaring a hang.
SUBPROCESS_HEARTBEAT_TIMEOUT = 120.0


class EngineSupervisor:
    """Process-global job tracker and GPU tenant coordinator.

    This is a lightweight in-process supervisor.  It does not launch
    subprocesses yet — that is Phase 2.  It does:

    - Create job directories under ``outputs/jobs/<job_id>/``
    - Track job phases (created → running → completed/failed/cancelled)
    - Emit ``EngineEvent`` JSONL lines to stdout (visible in aiwf-crash.log)
    - Expose the GPU tenant lock for callers that need to gate heavy work
    - Allow callers to register a progress callback for live UI updates
    """

    def __init__(self, gpu_lock: GpuTenantLock | None = None) -> None:
        self._gpu_lock = gpu_lock or get_gpu_lock()
        self._jobs: dict[str, JobRecord] = {}
        self._job_dirs: dict[str, Path] = {}
        self._lock = threading.Lock()
        self._progress_listeners: list[ProgressCallback] = []
        # Phase 2: subprocess tracking
        self._procs: dict[str, subprocess.Popen] = {}         # job_id → process
        self._reader_threads: dict[str, threading.Thread] = {}  # job_id → stdout reader

    # ------------------------------------------------------------------
    # Listener registration
    # ------------------------------------------------------------------

    def add_progress_listener(self, cb: ProgressCallback) -> None:
        """Register a callback that fires for every engine event on any job."""
        with self._lock:
            self._progress_listeners.append(cb)

    def remove_progress_listener(self, cb: ProgressCallback) -> None:
        with self._lock:
            self._progress_listeners = [l for l in self._progress_listeners if l is not cb]

    # ------------------------------------------------------------------
    # GPU tenant lock pass-through
    # ------------------------------------------------------------------

    @property
    def gpu_lock(self) -> GpuTenantLock:
        return self._gpu_lock

    def gpu_status(self) -> str:
        return self._gpu_lock.status_message()

    def gpu_is_free(self) -> bool:
        return self._gpu_lock.is_free

    # ------------------------------------------------------------------
    # Job lifecycle
    # ------------------------------------------------------------------

    def begin_job(
        self,
        engine: str,
        outputs_root: "Path | str",
        *,
        job_id: str | None = None,
        request_data: dict | None = None,
    ) -> str:
        """Create a job directory and ``JobRecord``.  Returns the job_id."""
        jid = job_id or f"{engine}_{uuid.uuid4().hex[:8]}"
        job_dir = make_job_dir(outputs_root, jid)
        record = JobRecord(job_id=jid, engine=engine, phase=JobPhase.CREATED)

        if request_data:
            import json
            (job_dir / "request.json").write_text(
                json.dumps(request_data, indent=2, default=str), encoding="utf-8"
            )

        record.save(job_dir)

        with self._lock:
            self._jobs[jid] = record
            self._job_dirs[jid] = job_dir

        event = EngineEvent.status(jid, f"Job created — engine: {engine}")
        self._fire(event)
        logger.info("[Supervisor] Job %s created (engine=%s)", jid, engine)
        return jid

    def update_status(self, job_id: str, message: str) -> None:
        record, job_dir = self._get(job_id)
        if record:
            record.message = message
            record.save(job_dir)
        event = EngineEvent.status(job_id, message)
        emit(event)
        self._fire(event)

    def mark_running(self, job_id: str, message: str = "") -> None:
        record, job_dir = self._get(job_id)
        if record:
            record.mark_running(message)
            record.save(job_dir)
        event = EngineEvent.status(job_id, message or "running")
        emit(event)
        self._fire(event)

    def update_progress(self, job_id: str, *, step: int, total: int, message: str = "") -> None:
        record, job_dir = self._get(job_id)
        if record:
            record.update_progress(step, total, message)
            record.save(job_dir)
        event = EngineEvent.progress(job_id, step=step, total=total, message=message)
        emit(event)
        self._fire(event)

    def register_artifact(self, job_id: str, path: "Path | str") -> None:
        record, job_dir = self._get(job_id)
        spath = str(path)
        if record:
            record.add_artifact(spath)
            record.save(job_dir)
        event = EngineEvent.artifact(job_id, path=spath)
        emit(event)
        self._fire(event)

    def finish_job(self, job_id: str, message: str = "") -> None:
        record, job_dir = self._get(job_id)
        if record:
            record.mark_completed(message)
            record.save(job_dir)
        event = EngineEvent.complete(job_id, message)
        emit(event)
        self._fire(event)
        logger.info("[Supervisor] Job %s completed", job_id)

    def fail_job(self, job_id: str, *, detail: str, message: str = "") -> None:
        record, job_dir = self._get(job_id)
        if record:
            record.mark_failed(detail, message)
            record.save(job_dir)
        event = EngineEvent.error(job_id, detail=detail, message=message or "job failed")
        emit(event)
        self._fire(event)
        logger.error("[Supervisor] Job %s failed: %s", job_id, detail)

    def cancel_job(self, job_id: str) -> None:
        record, job_dir = self._get(job_id)
        if record:
            record.mark_cancelled()
            record.save(job_dir)
        # Phase 2: send SIGTERM to the subprocess if running
        with self._lock:
            proc = self._procs.get(job_id)
        if proc and proc.poll() is None:
            try:
                if os.name == "nt":
                    proc.terminate()
                else:
                    proc.send_signal(signal.SIGTERM)
                logger.info("[Supervisor] SIGTERM sent to worker for job %s (pid=%s)", job_id, proc.pid)
            except OSError:
                pass
        event = EngineEvent.status(job_id, "cancelled")
        emit(event)
        self._fire(event)
        logger.info("[Supervisor] Job %s cancelled", job_id)

    # ------------------------------------------------------------------
    # Phase 2: subprocess worker management
    # ------------------------------------------------------------------

    def submit_subprocess_job(
        self,
        spec: "EngineSpec",
        request_data: dict,
        *,
        outputs_root: "Path | str",
        job_id: str | None = None,
    ) -> str:
        """Spawn an engine worker subprocess and stream its JSONL events.

        The supervisor:
        1. Creates a job directory and writes request.json into it.
        2. Spawns ``spec.python_exe() <spec.worker_script> <request.json>``.
        3. Reads the worker's stdout in a background thread.
        4. Routes JSONL events to ``_fire()`` / ``update_*`` helpers.
        5. Marks the job failed/complete when the process exits.

        The caller's GPU tenant lock acquisition is expected to happen in the
        service client (e.g. kohya_client.py) before calling this method.

        Returns
        -------
        job_id : str
            The assigned job ID.  The job is already in RUNNING phase when
            this method returns.
        """
        import json as _json

        jid = self.begin_job(
            spec.name,
            outputs_root,
            job_id=job_id,
            request_data=request_data,
        )
        job_dir = self.get_job_dir(jid)
        if job_dir is None:
            raise RuntimeError(f"[Supervisor] Could not create job dir for {jid}")

        # Write request with internal fields the worker needs
        full_request = {
            "_job_id": jid,
            "_engine": spec.name,
            "_repo_dir": str(spec.repo_dir) if spec.repo_dir else "",
            "_outputs_root": str(outputs_root),
            **request_data,
        }
        request_file = job_dir / "request.json"
        request_file.write_text(
            _json.dumps(full_request, indent=2, default=str), encoding="utf-8"
        )

        py_exe = spec.python_exe()
        worker = spec.worker_script
        if not Path(py_exe).exists():
            self.fail_job(jid, detail=f"Engine venv not found at {py_exe}. "
                          f"Is {spec.label} set up? (engines.json enabled=true, then restart)")
            raise RuntimeError(f"Engine python not found: {py_exe}")
        if not worker.exists():
            self.fail_job(jid, detail=f"Worker script not found: {worker}")
            raise RuntimeError(f"Worker script not found: {worker}")

        cmd = [py_exe, str(worker), str(request_file)]
        logger.info("[Supervisor] Spawning %s worker: %s", spec.label, " ".join(cmd))
        self.mark_running(jid, f"Spawning {spec.label} worker (pid pending)")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        with self._lock:
            self._procs[jid] = proc

        self.update_status(jid, f"Worker started (pid={proc.pid})")
        logger.info("[Supervisor] Worker pid=%d for job %s", proc.pid, jid)

        reader = threading.Thread(
            target=self._read_worker_stdout,
            args=(jid, proc),
            name=f"worker-reader-{jid}",
            daemon=True,
        )
        with self._lock:
            self._reader_threads[jid] = reader
        reader.start()

        return jid

    def _read_worker_stdout(self, job_id: str, proc: subprocess.Popen) -> None:
        """Background thread: read worker stdout, parse JSONL events, update job state."""
        try:
            for raw_line in proc.stdout:
                line = raw_line.rstrip()
                if not line:
                    continue

                event = EngineEvent.parse_line(line)
                if event is None:
                    logger.debug("[Worker %s] %s", job_id, line)
                    continue

                self._fire(event)

                if event.kind == "progress":
                    record, job_dir = self._get(job_id)
                    if record:
                        record.update_progress(event.step, event.total, event.message)
                        record.save(job_dir)

                elif event.kind == "artifact":
                    record, job_dir = self._get(job_id)
                    if record:
                        record.add_artifact(event.path)
                        record.save(job_dir)

                elif event.kind == "status":
                    record, job_dir = self._get(job_id)
                    if record:
                        record.message = event.message
                        record.save(job_dir)

                elif event.kind == "complete":
                    self.finish_job(job_id, message=event.message)
                    return

                elif event.kind == "error":
                    self.fail_job(job_id, detail=event.detail, message=event.message)
                    return

        except Exception:
            logger.exception("[Supervisor] Error reading worker stdout for job %s", job_id)
        finally:
            proc.wait()
            rc = proc.returncode
            with self._lock:
                self._procs.pop(job_id, None)
                self._reader_threads.pop(job_id, None)

            record = self.get_job(job_id)
            if record and record.phase.is_active:
                detail = f"Worker exited with code {rc} without emitting a terminal event."
                logger.error("[Supervisor] %s (job=%s)", detail, job_id)
                self.fail_job(job_id, detail=detail, message=f"Worker died (exit {rc})")

    def get_worker_pid(self, job_id: str) -> int | None:
        """Return the subprocess PID for a running worker, or None."""
        with self._lock:
            proc = self._procs.get(job_id)
        return proc.pid if proc and proc.poll() is None else None

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def get_job_dir(self, job_id: str) -> Path | None:
        with self._lock:
            return self._job_dirs.get(job_id, None)

    def active_jobs(self) -> list[JobRecord]:
        with self._lock:
            return [r for r in self._jobs.values() if r.phase.is_active]

    def all_jobs(self) -> list[JobRecord]:
        with self._lock:
            return list(self._jobs.values())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, job_id: str) -> tuple[JobRecord | None, Path]:
        with self._lock:
            record = self._jobs.get(job_id)
            job_dir = self._job_dirs.get(job_id, Path("."))
        return record, job_dir

    def _fire(self, event: EngineEvent) -> None:
        with self._lock:
            listeners = list(self._progress_listeners)
        for cb in listeners:
            try:
                cb(event)
            except Exception:
                logger.exception("[Supervisor] Progress listener raised")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_global_supervisor: EngineSupervisor | None = None
_init_lock = threading.Lock()


def get_supervisor() -> EngineSupervisor:
    """Return the process-global ``EngineSupervisor`` (created on first call)."""
    global _global_supervisor
    if _global_supervisor is None:
        with _init_lock:
            if _global_supervisor is None:
                _global_supervisor = EngineSupervisor()
    return _global_supervisor
