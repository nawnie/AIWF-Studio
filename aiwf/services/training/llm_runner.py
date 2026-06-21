"""Thin subprocess runner for AI bot text-model training."""
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
from aiwf.services.training.llm_config import build_llm_training_config

logger = logging.getLogger(__name__)

_WORKER_RELATIVE = Path("engines") / "llm" / "worker.py"


class LLMEngineNotReady(RuntimeError):
    """Raised when the LLM training engine venv is not configured or not found."""


class LLMBotTrainerRunner:
    """Launch and manage a local Causal LM post-training subprocess."""

    def __init__(
        self,
        python_exe: Path | None = None,
        repo_root: Path | None = None,
        supervisor: ProcessSupervisor | None = None,
    ) -> None:
        self._python_exe = python_exe
        self._repo_root = repo_root or _find_repo_root()
        self._supervisor = supervisor or get_process_supervisor()
        self._lock = threading.Lock()
        self._active_worker: str | None = None

    def start(self, request, *, job_id: str | None = None) -> Iterator[str]:
        """Spawn an LLM training worker and yield stdout log lines."""
        python_exe = self._resolve_python_exe()
        jid = job_id or f"llm_{uuid.uuid4().hex[:8]}"

        with self._lock:
            if self._active_worker is not None and self._supervisor.is_running(self._active_worker):
                raise RuntimeError(
                    f"An AI bot training job is already running (slot={self._active_worker}). "
                    "Call stop() first."
                )
            self._active_worker = jid

        tmp_dir = Path(tempfile.mkdtemp(prefix=f"llm_{jid}_"))
        req_dict = _to_dict(request)
        req_dict.setdefault("_job_id", jid)
        req_dict.setdefault("_engine", "llm")

        request_file = tmp_dir / "request.json"
        request_file.write_text(json.dumps(req_dict, indent=2, default=str), encoding="utf-8")

        worker_script = self._repo_root / _WORKER_RELATIVE
        if not worker_script.exists():
            raise LLMEngineNotReady(
                f"LLM worker script not found at {worker_script}. "
                "Ensure AIWF Studio is installed correctly."
            )

        cmd = WorkerCommand(
            args=[str(python_exe), str(worker_script), str(request_file)],
            cwd=tmp_dir,
            env={"PYTHONUNBUFFERED": "1"},
            name=jid,
        )

        logger.info("[LLMBotTrainerRunner] Starting job %s with %s", jid, python_exe)
        try:
            yield from self._supervisor.start(jid, cmd, check=True)
        finally:
            with self._lock:
                if self._active_worker == jid:
                    self._active_worker = None

    def stop(self) -> str:
        """Stop the currently-running LLM training job."""
        with self._lock:
            slot = self._active_worker
        if slot is None:
            return "No active AI bot training job."
        result = self._supervisor.stop(slot)
        with self._lock:
            self._active_worker = None
        return result

    def is_running(self) -> bool:
        with self._lock:
            slot = self._active_worker
        return slot is not None and self._supervisor.is_running(slot)

    def preview_config(self, request) -> dict:
        return build_llm_training_config(request)

    def _resolve_python_exe(self) -> Path:
        if self._python_exe is not None:
            return self._python_exe

        try:
            from launch import _build_engine_registry  # type: ignore[import]

            specs = {s.name: s for s in _build_engine_registry()}
            spec = specs.get("llm")
            if spec is not None and spec.is_ready():
                self._python_exe = Path(spec.python_exe())
                return self._python_exe
        except Exception as exc:
            logger.debug("[LLMBotTrainerRunner] Engine registry lookup failed: %s", exc)

        raise LLMEngineNotReady(
            "LLM trainer venv is not configured or not ready. "
            "Set 'enabled': true for the llm engine in engines.json, "
            "run launch.py so the venv can be prepared, then restart AIWF Studio."
        )


_runner: LLMBotTrainerRunner | None = None
_runner_lock = threading.Lock()


def get_llm_runner() -> LLMBotTrainerRunner:
    global _runner
    if _runner is None:
        with _runner_lock:
            if _runner is None:
                _runner = LLMBotTrainerRunner()
    return _runner


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
