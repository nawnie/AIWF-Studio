"""
aiwf/services/training/kohya_runner.py

Thin subprocess runner for Kohya LoRA training.

Builds a WorkerCommand pointing at engines/kohya/worker.py (in the Kohya
engine venv) and delegates to ProcessSupervisor.  Engine path resolution
is fully deferred — this module never imports launch.py at load time.

Rule enforced here: optional engines must never become mandatory boot deps.
  - No top-level import of launch, toml, kohya, or any engine package.
  - Engine venv paths resolved lazily inside start(), only when called.
  - If the engine is not configured, raises RuntimeError with a clear message.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Iterator

from aiwf.core.domain.worker import WorkerCommand
from aiwf.services.process_supervisor import ProcessSupervisor, get_process_supervisor
from aiwf.services.training.kohya_config import build_kohya_toml, write_kohya_toml

logger = logging.getLogger(__name__)

# Path to the worker script relative to the repo root — never changes.
_WORKER_RELATIVE = Path("engines") / "kohya" / "worker.py"


class KohyaEngineNotReady(RuntimeError):
    """Raised when the Kohya engine venv is not configured or not found."""


class KohyaRunner:
    """Launch and manage a Kohya LoRA training subprocess.

    Args:
        python_exe:   Absolute path to the Python binary inside the Kohya
                      engine venv.  When None, resolved lazily from the
                      engines registry on first call to start().
        repo_root:    Root of the AIWF Studio repo (used to locate the
                      worker script and engines/ directory).
        supervisor:   ProcessSupervisor instance.  Defaults to the global one.
    """

    def __init__(
        self,
        python_exe: Path | None = None,
        repo_root: Path | None = None,
        supervisor: ProcessSupervisor | None = None,
    ) -> None:
        self._python_exe = python_exe
        self._repo_root  = repo_root or _find_repo_root()
        self._supervisor = supervisor or get_process_supervisor()
        self._lock       = threading.Lock()
        self._active_worker: str | None = None   # slot name in ProcessSupervisor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, request, *, job_id: str | None = None) -> Iterator[str]:
        """Spawn a Kohya training worker and yield stdout log lines.

        Args:
            request:  KohyaLoraRequest or dict.
            job_id:   Optional job ID (generated if not supplied).

        Yields
        ------
        str
            Raw log lines from the worker process (stripped of newlines).

        Raises
        ------
        KohyaEngineNotReady
            If the Kohya engine venv is not configured.
        RuntimeError
            If a Kohya job is already running.
        """
        python_exe = self._resolve_python_exe()
        jid        = job_id or f"kohya_{uuid.uuid4().hex[:8]}"

        with self._lock:
            if self._active_worker is not None and self._supervisor.is_running(self._active_worker):
                raise RuntimeError(
                    f"A Kohya training job is already running (slot={self._active_worker}). "
                    "Call stop() first."
                )
            self._active_worker = jid

        # Write request.json into a temp directory that persists for the run
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"kohya_{jid}_"))
        req_dict = _to_dict(request)
        req_dict.setdefault("_job_id", jid)
        req_dict.setdefault("_engine", "kohya")
        req_dict.setdefault("_repo_dir", "")   # worker will use its own heuristic

        request_file = tmp_dir / "request.json"
        request_file.write_text(
            json.dumps(req_dict, indent=2, default=str), encoding="utf-8"
        )

        worker_script = self._repo_root / _WORKER_RELATIVE
        if not worker_script.exists():
            raise KohyaEngineNotReady(
                f"Kohya worker script not found at {worker_script}. "
                "Ensure AIWF Studio is installed correctly."
            )

        cmd = WorkerCommand(
            args=[str(python_exe), str(worker_script), str(request_file)],
            cwd=tmp_dir,
            env={"PYTHONUNBUFFERED": "1"},
            name=jid,
        )

        logger.info("[KohyaRunner] Starting job %s with %s", jid, python_exe)
        try:
            yield from self._supervisor.start(jid, cmd, check=True)
        finally:
            with self._lock:
                if self._active_worker == jid:
                    self._active_worker = None

    def stop(self) -> str:
        """Stop the currently-running Kohya job."""
        with self._lock:
            slot = self._active_worker
        if slot is None:
            return "No active Kohya job."
        result = self._supervisor.stop(slot)
        with self._lock:
            self._active_worker = None
        return result

    def is_running(self) -> bool:
        with self._lock:
            slot = self._active_worker
        return slot is not None and self._supervisor.is_running(slot)

    # ------------------------------------------------------------------
    # Config preview (no subprocess)
    # ------------------------------------------------------------------

    def preview_toml(self, request) -> str:
        """Return the TOML config that would be generated for *request*."""
        return build_kohya_toml(request)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_python_exe(self) -> Path:
        """Return the Kohya venv Python, resolving lazily if needed."""
        if self._python_exe is not None:
            return self._python_exe

        # Deferred import — NEVER at module level
        try:
            from launch import _build_engine_registry  # type: ignore[import]
            specs = {s.name: s for s in _build_engine_registry()}
            spec  = specs.get("kohya")
            if spec is not None and spec.is_ready():
                self._python_exe = Path(spec.python_exe())
                return self._python_exe
        except Exception as exc:
            logger.debug("[KohyaRunner] Engine registry lookup failed: %s", exc)

        raise KohyaEngineNotReady(
            "Kohya engine venv is not configured or not ready. "
            "Set 'enabled': true for the kohya engine in engines.json, "
            "run the venv setup, then restart AIWF Studio."
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_runner: KohyaRunner | None = None
_runner_lock = threading.Lock()


def get_kohya_runner() -> KohyaRunner:
    """Return the process-global KohyaRunner instance."""
    global _runner
    if _runner is None:
        with _runner_lock:
            if _runner is None:
                _runner = KohyaRunner()
    return _runner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_repo_root() -> Path:
    """Best-effort: walk up from this file to find the repo root."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "launch.py").exists() or (parent / "engines").is_dir():
            return parent
    return here.parents[3]  # fallback: 4 levels up from this file


def _to_dict(request) -> dict:
    if isinstance(request, dict):
        return dict(request)
    if hasattr(request, "model_dump"):
        return request.model_dump()
    return vars(request)
