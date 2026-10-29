"""Base protocol for AIWF engine worker subprocesses.

Worker contract
---------------
Every engine worker is a standalone Python script that:

1. Reads its job request from a JSON file passed as the first CLI argument.
2. Emits JSONL events to stdout (one event per line, flushed immediately).
3. Exits 0 on success, non-zero on failure.

The supervisor spawns the worker, captures its stdout, parses JSONL events,
and routes them to progress listeners and the job record on disk.

JSONL event format
------------------
Each line is a JSON object matching ``EngineEvent`` from
``aiwf.core.domain.engine_events``:

    {"kind": "progress", "job_id": "wan_abc123", "step": 4, "total": 8, ...}
    {"kind": "artifact", "job_id": "wan_abc123", "path": "/abs/path/to/video.mp4"}
    {"kind": "complete",  "job_id": "wan_abc123", "message": "done"}
    {"kind": "error",    "job_id": "wan_abc123", "detail": "OOM", "message": "failed"}

Non-JSON lines (plain log output) are forwarded to the crash log but ignored
for job tracking purposes.

Usage in a worker script
------------------------
    from aiwf.engine_workers.base import WorkerContext, emit_progress, emit_artifact, emit_complete, emit_error

    with WorkerContext.from_argv() as ctx:
        emit_progress(ctx.job_id, step=0, total=10, message="Loading model")
        # ... do work ...
        emit_artifact(ctx.job_id, path="/path/to/output.mp4")
        emit_complete(ctx.job_id, message="Training complete")
"""
from __future__ import annotations

import json
import signal
import sys
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# JSONL event emission (mirrors engine_events.py but self-contained so
# workers don't need to import the full aiwf package — they may run in a
# separate venv that doesn't have aiwf installed).
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _emit(obj: dict) -> None:
    """Write one JSONL event line to stdout, flushed immediately."""
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def emit_status(job_id: str, message: str) -> None:
    _emit({"kind": "status", "job_id": job_id, "message": message, "ts": _ts()})


def emit_progress(job_id: str, *, step: int, total: int, message: str = "") -> None:
    _emit({"kind": "progress", "job_id": job_id, "step": step,
           "total": total, "message": message, "ts": _ts()})


def emit_artifact(job_id: str, *, path: str) -> None:
    _emit({"kind": "artifact", "job_id": job_id, "path": path, "ts": _ts()})


def emit_complete(job_id: str, message: str = "") -> None:
    _emit({"kind": "complete", "job_id": job_id, "message": message, "ts": _ts()})


def emit_error(job_id: str, *, detail: str, message: str = "job failed") -> None:
    _emit({"kind": "error", "job_id": job_id,
           "detail": detail, "message": message, "ts": _ts()})


def emit_heartbeat(job_id: str) -> None:
    _emit({"kind": "heartbeat", "job_id": job_id, "ts": _ts()})


# ---------------------------------------------------------------------------
# WorkerContext — reads the job request and wraps the run lifecycle
# ---------------------------------------------------------------------------

@dataclass
class WorkerContext:
    """Holds the job request and job_id for a running worker.

    Typical use::

        with WorkerContext.from_argv() as ctx:
            emit_progress(ctx.job_id, step=0, total=10)
            run_training(ctx.request)
            emit_complete(ctx.job_id)

    The context manager catches any unhandled exception and emits an error
    event before re-raising, so the supervisor always receives a terminal event.
    """
    job_id: str
    request: dict[str, Any]
    request_file: Path
    _raised: bool = field(default=False, repr=False, init=False)

    @classmethod
    def from_argv(cls) -> "WorkerContext":
        """Read the job request from the path given as sys.argv[1].

        The supervisor writes the request to ``<job_dir>/request.json``
        and passes the path as the first argument when spawning the worker.
        """
        if len(sys.argv) < 2:
            print(
                "Usage: python worker.py <path/to/request.json>",
                file=sys.stderr,
                flush=True,
            )
            sys.exit(2)

        request_file = Path(sys.argv[1])
        if not request_file.exists():
            print(f"ERROR: Request file not found: {request_file}", file=sys.stderr, flush=True)
            sys.exit(2)

        raw = json.loads(request_file.read_text(encoding="utf-8"))
        job_id: str = raw.get("_job_id", "unknown")
        return cls(job_id=job_id, request=raw, request_file=request_file)

    def __enter__(self) -> "WorkerContext":
        # Install SIGTERM handler so the supervisor can cleanly cancel workers.
        def _handle_sigterm(signum, frame):  # noqa: ANN001
            emit_error(self.job_id, detail="Worker received SIGTERM — cancelled by supervisor.",
                       message="cancelled")
            sys.exit(1)

        signal.signal(signal.SIGTERM, _handle_sigterm)
        emit_status(self.job_id, "Worker started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # noqa: ANN001
        if exc_type is not None:
            detail = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
            emit_error(self.job_id, detail=detail, message=str(exc_val) or "Worker crashed")
            # Don't suppress — let the process exit with non-zero code
            return False
        return False


# ---------------------------------------------------------------------------
# Convenience: run a worker main() with standard error handling
# ---------------------------------------------------------------------------

def run_worker(main_fn) -> None:  # noqa: ANN001
    """Call ``main_fn(ctx)`` inside a WorkerContext; exit with appropriate code.

    Usage::

        def main(ctx: WorkerContext) -> None:
            emit_progress(ctx.job_id, step=0, total=10, message="start")
            ...
            emit_complete(ctx.job_id)

        if __name__ == "__main__":
            run_worker(main)
    """
    try:
        ctx = WorkerContext.from_argv()
        with ctx:
            main_fn(ctx)
    except SystemExit:
        raise
    except Exception:
        # WorkerContext.__exit__ already emitted the error event;
        # exit with non-zero so the supervisor knows it failed.
        sys.exit(1)
