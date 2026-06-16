"""
aiwf/core/domain/worker.py

Typed contract shared between ProcessSupervisor and all subprocess runners.

WorkerCommand describes what to launch.
WorkerResult describes what came back once a job finishes.

These are intentionally pure-Python dataclasses with no torch/diffusers
imports so they can be used safely in UI code and tests alike.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class WorkerCommand:
    """Everything needed to spawn a subprocess worker.

    Args:
        args:             Full command list passed to ``subprocess.Popen``.
                          Must be absolute paths; no shell=True.
        cwd:              Working directory for the subprocess.
        env:              Environment variables to *merge* into ``os.environ``
                          for the child process.  Keys already in the parent
                          environment are overridden; all others are inherited.
        name:             Human-readable label used in log messages and the
                          process-supervisor's tracking dict.
        timeout_seconds:  If set, the supervisor will SIGTERM the process after
                          this many seconds of wall-clock time.  ``None`` means
                          no timeout (run until done or explicitly cancelled).
    """

    args: list[str]
    cwd: Path
    env: dict[str, str]
    name: str
    timeout_seconds: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.args:
            raise ValueError("WorkerCommand.args must not be empty")
        if not self.name:
            raise ValueError("WorkerCommand.name must not be empty")


# ---------------------------------------------------------------------------
# Result status constants
# ---------------------------------------------------------------------------

STATUS_COMPLETED = "completed"
STATUS_FAILED    = "failed"
STATUS_CANCELLED = "cancelled"
STATUS_TIMEOUT   = "timeout"

TERMINAL_STATUSES: frozenset[str] = frozenset(
    {STATUS_COMPLETED, STATUS_FAILED, STATUS_CANCELLED, STATUS_TIMEOUT}
)


@dataclass(frozen=True)
class WorkerResult:
    """Outcome of a completed (or aborted) subprocess worker.

    Args:
        job_id:        Matches the ``job_id`` assigned by EngineSupervisor.
        status:        One of the STATUS_* constants above.
        return_code:   OS exit code, or ``None`` if the process was killed
                       before it had a chance to exit.
        output_paths:  Paths to output files produced by the worker
                       (images, videos, checkpoints, etc.).
        logs_path:     Path to the worker's captured log/stdout file,
                       or ``None`` if logging was not captured.
        error_message: Human-readable error summary (empty string on success).
    """

    job_id: str
    status: str
    return_code: Optional[int]
    output_paths: list[Path]
    logs_path: Optional[Path]
    error_message: str

    def __post_init__(self) -> None:
        if self.status not in TERMINAL_STATUSES:
            raise ValueError(
                f"WorkerResult.status must be one of {sorted(TERMINAL_STATUSES)}, "
                f"got {self.status!r}"
            )

    # ------------------------------------------------------------------
    # Convenience queries
    # ------------------------------------------------------------------

    @property
    def ok(self) -> bool:
        return self.status == STATUS_COMPLETED

    @property
    def failed(self) -> bool:
        return self.status in (STATUS_FAILED, STATUS_TIMEOUT)

    @property
    def cancelled(self) -> bool:
        return self.status == STATUS_CANCELLED

    @classmethod
    def success(
        cls,
        job_id: str,
        output_paths: list[Path] | None = None,
        logs_path: Optional[Path] = None,
        return_code: int = 0,
    ) -> "WorkerResult":
        return cls(
            job_id=job_id,
            status=STATUS_COMPLETED,
            return_code=return_code,
            output_paths=output_paths or [],
            logs_path=logs_path,
            error_message="",
        )

    @classmethod
    def failure(
        cls,
        job_id: str,
        error_message: str,
        return_code: Optional[int] = None,
        logs_path: Optional[Path] = None,
    ) -> "WorkerResult":
        return cls(
            job_id=job_id,
            status=STATUS_FAILED,
            return_code=return_code,
            output_paths=[],
            logs_path=logs_path,
            error_message=error_message,
        )

    @classmethod
    def cancelled_result(
        cls,
        job_id: str,
        return_code: Optional[int] = None,
        logs_path: Optional[Path] = None,
    ) -> "WorkerResult":
        return cls(
            job_id=job_id,
            status=STATUS_CANCELLED,
            return_code=return_code,
            output_paths=[],
            logs_path=logs_path,
            error_message="Job was cancelled.",
        )
