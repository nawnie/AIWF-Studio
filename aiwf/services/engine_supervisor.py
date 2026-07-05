"""Engine supervisor - Phase 1 (in-process) + Phase 2 (subprocess) combined.

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

The Phase 2 method co-exists with Phase 1 - in-process Wan generation
keeps working unchanged while new training jobs use subprocess workers.

Architecture reference: ``docs/architecture/engine_isolation_architecture.md``

Phase 2 acceptance criteria:
  OK submit_subprocess_job() spawns a worker in the engine's venv
  OK Worker stdout is parsed for JSONL events
  OK Events update the JobRecord and fire registered listeners
  OK Non-JSONL lines are forwarded to the crash log
  OK Worker death (any exit code) triggers fail_job() automatically
  OK cancel_job() sends SIGTERM to the worker process
  OK Heartbeat monitoring detects silent hangs (configurable timeout)
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator

from aiwf.core.domain.engine import EngineSwitchRequest, EngineSwitchResult, EngineTenant, EngineStatus
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
    subprocesses yet - that is Phase 2.  It does:

    - Create job directories under ``outputs/jobs/<job_id>/``
    - Track job phases (created -> running -> completed/failed/cancelled)
    - Emit ``EngineEvent`` JSONL lines to stdout (visible in aiwf-crash.log)
    - Expose the GPU tenant lock for callers that need to gate heavy work
    - Allow callers to register a progress callback for live UI updates
    """

    def __init__(
        self,
        gpu_lock: GpuTenantLock | None = None,
        ollama_client=None,
    ) -> None:
        self._gpu_lock = gpu_lock or get_gpu_lock()
        self._jobs: dict[str, JobRecord] = {}
        self._job_dirs: dict[str, Path] = {}
        self._lock = threading.Lock()
        self._progress_listeners: list[ProgressCallback] = []
        # Phase 2: subprocess tracking
        self._procs: dict[str, subprocess.Popen] = {}         # job_id -> process
        self._reader_threads: dict[str, threading.Thread] = {}  # job_id -> stdout reader
        self._monitor_threads: dict[str, threading.Thread] = {}  # job_id -> heartbeat monitor
        self._worker_last_event_at: dict[str, float] = {}
        # Engine tenant tracking
        self._engine_status = EngineStatus()
        self._ollama_client = ollama_client  # injected by AppContext when available
        self._active_chat_model: str = ""
        self._active_tenant_job_id: str = ""
        self._job_tenants: dict[str, tuple[EngineTenant, str]] = {}
        self._tenant_local = threading.local()
        if self._ollama_client is not None:
            self._gpu_lock.register_ollama_unload(self._unload_chat_model)

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
    # Engine tenant management
    # ------------------------------------------------------------------

    @property
    def active_tenant(self) -> EngineTenant:
        with self._lock:
            return self._engine_status.active

    def engine_status(self) -> EngineStatus:
        with self._lock:
            return self._engine_status

    def request_switch(self, request: EngineSwitchRequest) -> EngineSwitchResult:
        """Acquire, wait for, deny, or release the active GPU tenant."""
        if request.target == EngineTenant.IDLE:
            return self._release_active_tenant(request)

        job_id = (request.job_id or "").strip()
        if not job_id:
            job_id = (
                f"{request.target.value}_{uuid.uuid4().hex[:8]}"
                if request.target.is_gpu_heavy()
                else request.target.value
            )

        lock_tenant = request.target.value
        granted = (
            self._gpu_lock.wait_acquire(lock_tenant, job_id)
            if request.allow_wait
            else self._gpu_lock.try_acquire(lock_tenant, job_id)
        )
        if not granted:
            active = self._tenant_from_lock(self._gpu_lock.active_tenant)
            return EngineSwitchResult(
                False,
                active,
                self._gpu_lock.blocked_message(lock_tenant) or "GPU tenant switch blocked.",
            )

        if request.target.is_gpu_heavy():
            with self._lock:
                prior = self._engine_status.active
            if prior.is_gpu_heavy() and prior != request.target:
                self._flush_cuda()

        with self._lock:
            current = self._engine_status.active
            self._active_tenant_job_id = job_id
            self._engine_status.record_switch(
                request.target,
                f"Switched from {current.value} to {request.target.value}: {request.reason}",
            )
        logger.info(
            "[Supervisor] Tenant switch: %s -> %s (%s, job=%s)",
            current.value,
            request.target.value,
            request.reason or "no reason given",
            job_id,
        )
        return EngineSwitchResult(True, request.target, f"Switched to {request.target.friendly_name()}.")

    def _release_active_tenant(self, request: EngineSwitchRequest) -> EngineSwitchResult:
        with self._lock:
            current = self._engine_status.active
            owner_id = self._active_tenant_job_id

        requested_owner = (request.job_id or "").strip()
        if current == EngineTenant.IDLE:
            return EngineSwitchResult(True, EngineTenant.IDLE, "GPU is already idle.")
        if requested_owner and owner_id and requested_owner != owner_id:
            return EngineSwitchResult(
                False,
                current,
                f"Cannot release {current.friendly_name()}: active owner is {owner_id}, not {requested_owner}.",
            )

        if owner_id:
            self._gpu_lock.release(current.value, owner_id)

        with self._lock:
            prior = self._engine_status.active
            self._active_tenant_job_id = ""
            self._engine_status.record_switch(
                EngineTenant.IDLE,
                f"Released {prior.value}: {request.reason}",
            )
        logger.info(
            "[Supervisor] Tenant release: %s -> idle (%s)",
            prior.value,
            request.reason or "no reason given",
        )
        return EngineSwitchResult(True, EngineTenant.IDLE, "GPU tenant released.")

    @staticmethod
    def _tenant_from_lock(value: str | None) -> EngineTenant:
        if not value:
            return EngineTenant.IDLE
        for tenant in EngineTenant:
            if tenant.value == value:
                return tenant
        aliases = {
            "generation": EngineTenant.IMAGE,
            "wan": EngineTenant.VIDEO,
            "kohya": EngineTenant.LORA_TRAINING,
            "ed2": EngineTenant.FULL_TRAINING,
            "ollama": EngineTenant.CHAT,
        }
        return aliases.get(value, EngineTenant.IDLE)

    @contextmanager
    def tenant_session(
        self,
        target: EngineTenant,
        *,
        reason: str = "",
        job_id: str | None = None,
        allow_wait: bool = False,
    ) -> Iterator[str]:
        """Context manager that owns a tenant until the block exits."""
        stack = getattr(self._tenant_local, "stack", None)
        if stack is None:
            stack = {}
            self._tenant_local.stack = stack
        key = target.value
        existing = stack.get(key)
        if existing is not None:
            owner_id, depth, owns_lock = existing
            stack[key] = (owner_id, depth + 1, owns_lock)
            try:
                yield owner_id
            finally:
                owner_id, depth, owns_lock = stack[key]
                if depth <= 1:
                    stack.pop(key, None)
                else:
                    stack[key] = (owner_id, depth - 1, owns_lock)
            return

        if stack and self.active_tenant != EngineTenant.IDLE:
            owner_id = next(iter(stack.values()))[0]
            stack[key] = (owner_id, 1, False)
            try:
                yield owner_id
            finally:
                owner_id, depth, owns_lock = stack[key]
                if depth <= 1:
                    stack.pop(key, None)
                else:
                    stack[key] = (owner_id, depth - 1, owns_lock)
            return

        owner_id = job_id or f"{target.value}_{uuid.uuid4().hex[:8]}"
        result = self.request_switch(
            EngineSwitchRequest(
                target=target,
                reason=reason,
                job_id=owner_id,
                allow_wait=allow_wait,
            )
        )
        if not result.ok:
            raise RuntimeError(result.message)
        stack[key] = (owner_id, 1, True)
        try:
            yield owner_id
        finally:
            existing = stack.get(key)
            if existing is not None:
                existing_owner_id, depth, owns_lock = existing
                if depth <= 1:
                    stack.pop(key, None)
                else:
                    stack[key] = (existing_owner_id, depth - 1, owns_lock)
                    return
                if not owns_lock:
                    return
            self.request_switch(
                EngineSwitchRequest(
                    target=EngineTenant.IDLE,
                    reason=f"{reason or target.friendly_name()} complete",
                    job_id=owner_id,
                )
            )

    @contextmanager
    def borrow_active_tenant(
        self,
        target: EngineTenant,
        *,
        job_id: str,
    ) -> Iterator[str]:
        """Mark the current thread as covered by an already-acquired tenant.

        This is for in-process callbacks that run under a manual
        ``request_switch`` owner, such as image post-processing. It does not
        acquire or release the global lock; it only lets nested service calls
        reuse the caller's ownership instead of being denied as competitors.
        """
        owner_id = (job_id or "").strip()
        if not owner_id:
            raise ValueError("borrow_active_tenant requires a job_id")
        with self._lock:
            if self._engine_status.active != target or self._active_tenant_job_id != owner_id:
                raise RuntimeError(
                    f"Cannot borrow {target.friendly_name()}: active owner is "
                    f"{self._active_tenant_job_id or 'none'}."
                )

        stack = getattr(self._tenant_local, "stack", None)
        if stack is None:
            stack = {}
            self._tenant_local.stack = stack
        key = target.value
        existing = stack.get(key)
        if existing is not None:
            existing_owner_id, depth, owns_lock = existing
            stack[key] = (existing_owner_id, depth + 1, owns_lock)
            try:
                yield existing_owner_id
            finally:
                existing_owner_id, depth, owns_lock = stack[key]
                if depth <= 1:
                    stack.pop(key, None)
                else:
                    stack[key] = (existing_owner_id, depth - 1, owns_lock)
            return

        stack[key] = (owner_id, 1, False)
        try:
            yield owner_id
        finally:
            existing = stack.get(key)
            if existing is not None:
                existing_owner_id, depth, owns_lock = existing
                if depth <= 1:
                    stack.pop(key, None)
                else:
                    stack[key] = (existing_owner_id, depth - 1, owns_lock)

    def set_ollama_client(self, client) -> None:
        """Install the Ollama client used to evict chat models before heavy work."""
        with self._lock:
            self._ollama_client = client
        self._gpu_lock.register_ollama_unload(self._unload_chat_model)

    def set_chat_model(self, model_name: str) -> None:
        """Remember which Ollama model is loaded so we can unload it on switch."""
        with self._lock:
            self._active_chat_model = model_name

    def _unload_chat_model(self) -> bool:
        with self._lock:
            client = self._ollama_client
            model = self._active_chat_model
        if client is None or not model:
            logger.warning("[Supervisor] Cannot unload chat model: Ollama client/model is not registered")
            return False
        try:
            ok = bool(client.unload(model))
        except Exception:
            logger.exception("[Supervisor] Ollama unload failed")
            return False
        if ok:
            logger.info("[Supervisor] Ollama model %r unloaded before GPU-heavy switch", model)
        else:
            logger.warning("[Supervisor] Ollama model %r refused unload", model)
        return ok

    def _flush_cuda(self) -> None:
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

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

        event = EngineEvent.status(jid, f"Job created - engine: {engine}")
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
        if not proc or proc.poll() is not None:
            self._release_job_tenant(job_id, "Job cancelled")
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
        tenant: EngineTenant | None = None,
        allow_wait: bool = False,
    ) -> str:
        """Spawn an engine worker subprocess and stream its JSONL events.

        The supervisor:
        1. Creates a job directory and writes request.json into it.
        2. Spawns ``spec.python_exe() <spec.worker_script> <request.json>``.
        3. Reads the worker's stdout in a background thread.
        4. Routes JSONL events to ``_fire()`` / ``update_*`` helpers.
        5. Marks the job failed/complete when the process exits.

        If *tenant* is provided, this method acquires GPU ownership before
        spawning and releases it when the worker process exits.

        Returns
        -------
        job_id : str
            The assigned job ID.  The job is already in RUNNING phase when
            this method returns.
        """
        import json as _json

        jid = job_id or f"{spec.name}_{uuid.uuid4().hex[:8]}"
        tenant_acquired = False
        if tenant is not None:
            switch = self.request_switch(
                EngineSwitchRequest(
                    target=tenant,
                    reason=f"{spec.label} worker",
                    job_id=jid,
                    allow_wait=allow_wait,
                )
            )
            if not switch.ok:
                raise RuntimeError(switch.message)
            tenant_acquired = True
            with self._lock:
                self._job_tenants[jid] = (tenant, jid)

        try:
            jid = self.begin_job(
                spec.name,
                outputs_root,
                job_id=jid,
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
                self._worker_last_event_at[jid] = time.monotonic()

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

            monitor = threading.Thread(
                target=self._monitor_worker_heartbeat,
                args=(jid, proc),
                name=f"worker-monitor-{jid}",
                daemon=True,
            )
            with self._lock:
                self._monitor_threads[jid] = monitor
            monitor.start()

            return jid
        except Exception:
            if tenant_acquired:
                self._release_job_tenant(jid, "Worker failed to start")
            raise

    def _read_worker_stdout(self, job_id: str, proc: subprocess.Popen) -> None:
        """Background thread: read worker stdout, parse JSONL events, update job state."""
        try:
            for raw_line in proc.stdout:
                line = raw_line.rstrip()
                if not line:
                    continue

                self._touch_worker_heartbeat(job_id)
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
                self._monitor_threads.pop(job_id, None)
                self._worker_last_event_at.pop(job_id, None)

            record = self.get_job(job_id)
            if record and record.phase.is_active:
                detail = f"Worker exited with code {rc} without emitting a terminal event."
                logger.error("[Supervisor] %s (job=%s)", detail, job_id)
                self.fail_job(job_id, detail=detail, message=f"Worker died (exit {rc})")
            self._release_job_tenant(job_id, "Worker exited")

    def _touch_worker_heartbeat(self, job_id: str) -> None:
        """Record that a worker produced stdout and is still responsive."""
        with self._lock:
            if job_id in self._procs:
                self._worker_last_event_at[job_id] = time.monotonic()

    def _monitor_worker_heartbeat(self, job_id: str, proc: subprocess.Popen) -> None:
        """Fail and terminate an active worker that stops emitting stdout."""
        timeout = max(float(SUBPROCESS_HEARTBEAT_TIMEOUT), 0.01)
        interval = min(1.0, max(0.01, timeout / 4.0))
        while proc.poll() is None:
            time.sleep(interval)
            with self._lock:
                last_event_at = self._worker_last_event_at.get(job_id)
                record = self._jobs.get(job_id)
            if last_event_at is None:
                return
            if record is None or not record.phase.is_active:
                return
            if time.monotonic() - last_event_at <= timeout:
                continue

            detail = f"Worker heartbeat timed out after {timeout:.0f}s without output."
            logger.error("[Supervisor] %s (job=%s, pid=%s)", detail, job_id, getattr(proc, "pid", "?"))
            self._terminate_worker_process(job_id, proc)
            record = self.get_job(job_id)
            if record and record.phase.is_active:
                self.fail_job(job_id, detail=detail, message="Worker heartbeat timed out")
            self._release_job_tenant(job_id, "Worker heartbeat timed out")
            return

    def _terminate_worker_process(self, job_id: str, proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
            return
        try:
            if os.name == "nt":
                proc.terminate()
            else:
                proc.send_signal(signal.SIGTERM)
            logger.info("[Supervisor] SIGTERM sent to unresponsive worker for job %s", job_id)
        except OSError:
            return
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
                logger.warning("[Supervisor] SIGKILL sent to unresponsive worker for job %s", job_id)
            except OSError:
                pass

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
            return self._job_dirs.get(job_id)

    def list_jobs(self) -> list[JobRecord]:
        with self._lock:
            return list(self._jobs.values())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, job_id: str) -> tuple[JobRecord | None, Path | None]:
        with self._lock:
            return self._jobs.get(job_id), self._job_dirs.get(job_id)

    def _release_job_tenant(self, job_id: str, reason: str) -> None:
        with self._lock:
            entry = self._job_tenants.pop(job_id, None)
        if entry is None:
            return
        tenant, owner_id = entry
        self.request_switch(
            EngineSwitchRequest(
                target=EngineTenant.IDLE,
                reason=reason,
                job_id=owner_id,
            )
        )

    def _fire(self, event: EngineEvent) -> None:
        with self._lock:
            listeners = list(self._progress_listeners)
        for cb in listeners:
            try:
                cb(event)
            except Exception:
                logger.exception("[Supervisor] Progress listener error")


# ---------------------------------------------------------------------------
# Process-global singleton
# ---------------------------------------------------------------------------

_supervisor: EngineSupervisor | None = None
_supervisor_lock = threading.Lock()


def get_supervisor() -> EngineSupervisor:
    global _supervisor
    if _supervisor is None:
        with _supervisor_lock:
            if _supervisor is None:
                _supervisor = EngineSupervisor()
    return _supervisor
