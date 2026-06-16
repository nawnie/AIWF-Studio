"""
aiwf/services/training/ed2_runner.py

Thin subprocess runner for EveryDream2 full fine-tuning.

Same pattern as kohya_runner.py: wraps ProcessSupervisor, resolves the
ED2 engine venv lazily, never imports launch.py at module load time.

Rule enforced here: optional engines must never become mandatory boot deps.
"""
from __future__ import annotations

import json
import logging
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Iterator

from aiwf.core.domain.worker import WorkerCommand
from aiwf.services.process_supervisor import ProcessSupervisor, get_process_supervisor
from aiwf.services.training.ed2_config import build_ed2_config

logger = logging.getLogger(__name__)

_WORKER_RELATIVE = Path("engines") / "ed2" / "worker.py"


class ED2EngineNotReady(RuntimeError):
    """Raised when the ED2 engine venv is not configured or not found."""


class ED2Runner:
    """Launch and manage an EveryDream2 training subprocess.

    Args:
        python_exe:  Absolute path to the Python binary in the ED2 venv.
                     When None, resolved lazily from the engines registry.
        repo_root:   Root of the AIWF Studio repo.
        supervisor:  ProcessSupervisor instance.
    """

    def __init__(
        self,
        python_exe: Path | None = None,
        repo_dir: Path | None = None,
        repo_root: Path | None = None,
        supervisor: ProcessSupervisor | None = None,
    ) -> None:
        self._python_exe = python_exe
        self._repo_root  = repo_root or _find_repo_root()
        self._repo_dir   = repo_dir
        self._supervisor = supervisor or get_process_supervisor()
        self._lock       = threading.Lock()
        self._active_worker: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, request, *, job_id: str | None = None) -> Iterator[str]:
        """Spawn an ED2 training worker and yield stdout log lines.

        Args:
            request:  ED2TrainingRequest or dict.
            job_id:   Optional job ID (generated if not supplied).

        Raises
        ------
        ED2EngineNotReady
            If the ED2 engine venv is not configured.
        RuntimeError
            If an ED2 job is already running.
        """
        python_exe = self._resolve_python_exe()
        repo_dir = self._resolve_repo_dir()
        jid        = job_id or f"ed2_{uuid.uuid4().hex[:8]}"

        with self._lock:
            if self._active_worker is not None and self._supervisor.is_running(self._active_worker):
                raise RuntimeError(
                    f"An ED2 training job is already running (slot={self._active_worker}). "
                    "Call stop() first."
                )
            self._active_worker = jid

        tmp_dir = Path(tempfile.mkdtemp(prefix=f"ed2_{jid}_"))
        req_dict = _to_dict(request)
        req_dict.setdefault("_job_id", jid)
        req_dict.setdefault("_engine", "ed2")
        if not req_dict.get("_repo_dir"):
            req_dict["_repo_dir"] = str(repo_dir)

        request_file = tmp_dir / "request.json"
        request_file.write_text(
            json.dumps(req_dict, indent=2, default=str), encoding="utf-8"
        )

        worker_script = self._repo_root / _WORKER_RELATIVE
        if not worker_script.exists():
            raise ED2EngineNotReady(
                f"ED2 worker script not found at {worker_script}. "
                "Ensure AIWF Studio is installed correctly."
            )

        cmd = WorkerCommand(
            args=[str(python_exe), str(worker_script), str(request_file)],
            cwd=tmp_dir,
            env={"PYTHONUNBUFFERED": "1"},
            name=jid,
        )

        logger.info("[ED2Runner] Starting job %s with %s", jid, python_exe)
        try:
            yield from self._supervisor.start(jid, cmd)
        finally:
            with self._lock:
                if self._active_worker == jid:
                    self._active_worker = None

    def stop(self) -> str:
        """Stop the currently-running ED2 job."""
        with self._lock:
            slot = self._active_worker
        if slot is None:
            return "No active ED2 job."
        result = self._supervisor.stop(slot)
        with self._lock:
            self._active_worker = None
        return result

    def is_running(self) -> bool:
        with self._lock:
            slot = self._active_worker
        return slot is not None and self._supervisor.is_running(slot)

    def preview_config(self, request) -> dict:
        """Return the train.json dict that would be generated for *request*."""
        return build_ed2_config(request)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_python_exe(self) -> Path:
        if self._python_exe is not None:
            return self._python_exe

        # Deferred import — NEVER at module level
        try:
            from launch import _build_engine_registry  # type: ignore[import]
            specs = {s.name: s for s in _build_engine_registry()}
            spec  = specs.get("ed2")
            if spec is not None and spec.is_ready():
                self._python_exe = Path(spec.python_exe())
                if spec.repo_dir is not None:
                    self._repo_dir = Path(spec.repo_dir)
                return self._python_exe
        except Exception as exc:
            logger.debug("[ED2Runner] Engine registry lookup failed: %s", exc)

        raise ED2EngineNotReady(
            "ED2 engine venv is not configured or not ready. "
            "Set 'enabled': true for the ed2 engine in engines.json, "
            "run the venv setup, then restart AIWF Studio."
        )

    def _resolve_repo_dir(self) -> Path:
        if self._repo_dir is not None:
            return self._repo_dir

        return self._repo_root / "engines" / "ed2" / "EveryDream2trainer"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_runner: ED2Runner | None = None
_runner_lock = threading.Lock()


def get_ed2_runner() -> ED2Runner:
    global _runner
    if _runner is None:
        with _runner_lock:
            if _runner is None:
                _runner = ED2Runner()
    return _runner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "launch.py").exists() or (parent / "engines").is_dir():
            return parent
    return here.parents[3]


def _to_dict(request) -> dict:
    if isinstance(request, dict):
        return dict(request)
    if hasattr(request, "model_dump"):
        return request.model_dump()
    return vars(request)
