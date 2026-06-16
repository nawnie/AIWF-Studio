"""
aiwf/services/process_supervisor.py

Single place to launch, stream, and kill subprocess workers.

Used by KohyaRunner, ED2Runner, and any future subprocess-based engine.

Key design points
-----------------
* ``start()`` spawns a non-shell subprocess and returns a line iterator
  (generator).  The caller drives the iterator to consume stdout.
* ``stop()`` sends SIGTERM (Windows: terminate()) and then waits for the
  process to exit.  Uses psutil for recursive child-tree kill if available.
* A double-start guard prevents two workers with the same name from running
  simultaneously (each name maps to one slot).
* ``shell=False`` always.  ``CREATE_NEW_PROCESS_GROUP`` on Windows.
* Callers that want to *fire-and-forget* should run the generator in a
  background thread and collect lines into a log file.

Typical usage
-------------
    sup = ProcessSupervisor()

    # Blocking iteration (in a background thread):
    for line in sup.start("kohya-lora", command):
        log_file.write(line + "\\n")

    # Or fire-and-forget via thread:
    def _run():
        for line in sup.start("kohya-lora", command):
            pass  # lines already streamed to log inside start()

    t = threading.Thread(target=_run, daemon=True)
    t.start()

    # Later:
    sup.stop("kohya-lora")
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Iterator

from aiwf.core.domain.worker import WorkerCommand

logger = logging.getLogger(__name__)


class ProcessSupervisor:
    """Manages a pool of named subprocess workers.

    Each worker slot is identified by a *name* string.  Only one process per
    name can run at a time.  Attempting to start a worker whose slot is
    already occupied raises ``RuntimeError`` unless the previous process has
    already exited.
    """

    def __init__(self) -> None:
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, worker_name: str, command: WorkerCommand, *, check: bool = False) -> Iterator[str]:
        """Spawn the worker and yield stdout lines until the process exits.

        This is a *generator*.  Lines are yielded as they arrive (unbuffered
        on the child side via ``-u`` / ``PYTHONUNBUFFERED``).

        Raises
        ------
        RuntimeError
            If a process named *worker_name* is already running.
        """
        with self._lock:
            existing = self._procs.get(worker_name)
            if existing is not None and existing.poll() is None:
                raise RuntimeError(
                    f"ProcessSupervisor: worker '{worker_name}' is already running "
                    f"(pid={existing.pid}).  Call stop() first."
                )

        env = {**os.environ, **command.env, "PYTHONUNBUFFERED": "1"}
        cwd = self._resolve_cwd(command.cwd)

        popen_kwargs: dict = dict(
            args=command.args,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Platform-specific process-group flags so we can kill the whole tree
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True

        logger.info(
            "[ProcessSupervisor] Starting worker '%s': %s",
            worker_name,
            " ".join(str(a) for a in command.args),
        )

        proc = subprocess.Popen(**popen_kwargs)

        with self._lock:
            self._procs[worker_name] = proc

        logger.info("[ProcessSupervisor] Worker '%s' started (pid=%d)", worker_name, proc.pid)

        rc: int | None = None
        try:
            for line in proc.stdout:
                yield line.rstrip("\n")
        finally:
            proc.stdout.close()
            proc.wait()
            rc = proc.returncode
            logger.info(
                "[ProcessSupervisor] Worker '%s' exited (rc=%s)", worker_name, rc
            )
            with self._lock:
                # Only clear the slot if it still points to *this* process
                if self._procs.get(worker_name) is proc:
                    del self._procs[worker_name]
        if check and rc not in (0, None):
            raise RuntimeError(f"Worker '{worker_name}' exited with code {rc}.")

    def stop(self, worker_name: str, timeout: float = 10.0) -> str:
        """Terminate the named worker and wait for it to exit.

        Returns a human-readable status string.

        If ``psutil`` is available, sends SIGTERM to the full process subtree
        (handles workers that spawn child processes themselves).  Falls back to
        ``proc.terminate()`` if psutil is not installed.
        """
        with self._lock:
            proc = self._procs.get(worker_name)

        if proc is None:
            return f"Worker '{worker_name}' is not registered."

        if proc.poll() is not None:
            return f"Worker '{worker_name}' already exited (rc={proc.returncode})."

        # Attempt psutil recursive kill first
        killed_tree = False
        try:
            import psutil
            parent = psutil.Process(proc.pid)
            children = parent.children(recursive=True)
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            parent.terminate()
            gone, alive = psutil.wait_procs([parent] + children, timeout=timeout)
            for p in alive:
                try:
                    p.kill()
                except psutil.NoSuchProcess:
                    pass
            killed_tree = True
        except ImportError:
            pass
        except Exception as exc:
            logger.warning("[ProcessSupervisor] psutil tree-kill failed: %s", exc)

        if not killed_tree:
            # Fallback: plain terminate
            try:
                proc.terminate()
            except OSError:
                pass
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                    proc.wait(timeout=5.0)
                except Exception:
                    pass

        with self._lock:
            if self._procs.get(worker_name) is proc:
                del self._procs[worker_name]

        logger.info("[ProcessSupervisor] Worker '%s' stopped.", worker_name)
        return f"Worker '{worker_name}' stopped."

    def is_running(self, worker_name: str) -> bool:
        """Return True if a live process is registered under *worker_name*."""
        with self._lock:
            proc = self._procs.get(worker_name)
        return proc is not None and proc.poll() is None

    def get_pid(self, worker_name: str) -> int | None:
        """Return the PID of the named worker, or None if not running."""
        with self._lock:
            proc = self._procs.get(worker_name)
        if proc is not None and proc.poll() is None:
            return proc.pid
        return None

    def running_workers(self) -> list[str]:
        """Return names of all currently-live workers."""
        with self._lock:
            return [name for name, p in self._procs.items() if p.poll() is None]

    def stop_all(self, timeout: float = 10.0) -> None:
        """Stop every registered worker (called on app shutdown)."""
        with self._lock:
            names = list(self._procs.keys())
        for name in names:
            try:
                self.stop(name, timeout=timeout)
            except Exception as exc:
                logger.warning("[ProcessSupervisor] stop_all: error stopping '%s': %s", name, exc)

    @staticmethod
    def _resolve_cwd(cwd: Path) -> Path:
        """Return a subprocess-safe cwd, with Unix /tmp compatibility on Windows."""
        path = Path(cwd)
        if path.exists():
            return path

        if os.name == "nt" and path.as_posix() in {"/tmp", "\\tmp"}:
            return Path(tempfile.gettempdir())

        raise FileNotFoundError(f"Worker cwd does not exist: {path}")


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------


_process_supervisor: "ProcessSupervisor | None" = None
_ps_lock = __import__("threading").Lock()


def get_process_supervisor() -> "ProcessSupervisor":
    """Return the process-global ProcessSupervisor instance."""
    global _process_supervisor
    if _process_supervisor is None:
        with _ps_lock:
            if _process_supervisor is None:
                _process_supervisor = ProcessSupervisor()
    return _process_supervisor
