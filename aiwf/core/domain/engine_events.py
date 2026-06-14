"""Engine event types — emitted by generation/training workers over stdout (JSONL).

Workers write one JSON object per line to stdout.  The UI supervisor reads and
parses these lines to update job status, progress bars, and artifact lists.

Wire a worker like this::

    from aiwf.core.domain.engine_events import EngineEvent, emit
    emit(EngineEvent.status(job_id, "loading model"))
    emit(EngineEvent.progress(job_id, step=3, total=30))
    emit(EngineEvent.artifact(job_id, path="outputs/videos/job_001.mp4"))
    emit(EngineEvent.complete(job_id))

The supervisor (``aiwf.services.engine_supervisor``) calls
``EngineEvent.parse_line()`` on each stdout line.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Event payload types
# ---------------------------------------------------------------------------

EventKind = Literal["status", "progress", "artifact", "complete", "error", "heartbeat"]


@dataclass(frozen=True)
class EngineEvent:
    """One structured line written by a worker to stdout (or parsed by the supervisor)."""

    kind: EventKind
    job_id: str
    # Populated depending on kind:
    message: str = ""
    step: int = 0
    total: int = 0
    path: str = ""
    detail: str = ""
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def status(cls, job_id: str, message: str) -> "EngineEvent":
        return cls(kind="status", job_id=job_id, message=message)

    @classmethod
    def progress(cls, job_id: str, *, step: int, total: int, message: str = "") -> "EngineEvent":
        return cls(kind="progress", job_id=job_id, step=step, total=total, message=message)

    @classmethod
    def artifact(cls, job_id: str, *, path: str, message: str = "") -> "EngineEvent":
        return cls(kind="artifact", job_id=job_id, path=path, message=message)

    @classmethod
    def complete(cls, job_id: str, message: str = "") -> "EngineEvent":
        return cls(kind="complete", job_id=job_id, message=message)

    @classmethod
    def error(cls, job_id: str, *, detail: str, message: str = "") -> "EngineEvent":
        return cls(kind="error", job_id=job_id, detail=detail, message=message)

    @classmethod
    def heartbeat(cls, job_id: str) -> "EngineEvent":
        return cls(kind="heartbeat", job_id=job_id)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        # Use "kind" as the canonical key — matches worker base.py helpers.
        d: dict[str, Any] = {"kind": self.kind, "job_id": self.job_id, "ts": self.ts}
        if self.message:
            d["message"] = self.message
        if self.kind == "progress":
            d["step"] = self.step
            d["total"] = self.total
        if self.kind == "artifact" and self.path:
            d["path"] = self.path
        if self.kind == "error" and self.detail:
            d["detail"] = self.detail
        return d

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def parse_line(cls, line: str) -> "EngineEvent | None":
        """Parse one stdout line from a worker.  Returns None for non-JSON lines.

        Accepts both ``"kind"`` (canonical, from worker base.py) and legacy
        ``"event"`` key for backward compatibility.
        """
        line = line.strip()
        if not line or not line.startswith("{"):
            return None
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            return None
        # Accept "kind" (canonical) or "event" (legacy)
        kind = d.get("kind") or d.get("event")
        if kind not in {"status", "progress", "artifact", "complete", "error", "heartbeat"}:
            return None
        return cls(
            kind=kind,
            job_id=d.get("job_id", ""),
            message=d.get("message", ""),
            step=int(d.get("step", 0)),
            total=int(d.get("total", 0)),
            path=d.get("path", ""),
            detail=d.get("detail", ""),
            ts=d.get("ts", datetime.now(timezone.utc).isoformat()),
        )


# ---------------------------------------------------------------------------
# Worker-side helper: emit to stdout
# ---------------------------------------------------------------------------

def emit(event: EngineEvent, *, file=None) -> None:
    """Write an engine event as a JSONL line to stdout (or *file*)."""
    print(event.to_jsonl(), file=file or sys.stdout, flush=True)


def parse_event_line(line: str) -> "EngineEvent | None":
    """Module-level alias for EngineEvent.parse_line() — convenience import."""
    return EngineEvent.parse_line(line)
