"""Job lifecycle domain — status enum, JobRecord, and job directory layout.

Every heavy operation (video generation, training, etc.) becomes a *job*.
The engine supervisor creates a ``JobRecord`` when a job is submitted and
updates it as the engine emits events.

Job directory layout (under ``outputs/jobs/<job_id>/``)::

    request.json          — serialised input parameters
    status.json           — current JobRecord snapshot
    stdout.jsonl          — raw JSONL events from the worker
    stderr.log            — worker stderr (plain text)
    artifacts/            — output files (mp4, safetensors, …)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class JobPhase(str, Enum):
    """Lifecycle states for a heavy compute job."""
    CREATED = "created"
    VALIDATED = "validated"
    QUEUED = "queued"
    BLOCKED = "blocked"       # GPU owned by another tenant
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in (JobPhase.COMPLETED, JobPhase.FAILED, JobPhase.CANCELLED)

    @property
    def is_active(self) -> bool:
        return self in (JobPhase.QUEUED, JobPhase.RUNNING)


@dataclass
class JobRecord:
    """Mutable snapshot of a job's state, persisted to ``status.json``."""

    job_id: str
    engine: str                          # e.g. "wan", "generation", "kohya"
    phase: JobPhase = JobPhase.CREATED
    message: str = ""
    step: int = 0
    total_steps: int = 0
    artifact_paths: list[str] = field(default_factory=list)
    error_detail: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: str = ""
    finished_at: str = ""

    # ------------------------------------------------------------------
    # Convenience mutators
    # ------------------------------------------------------------------

    def mark_running(self, message: str = "") -> None:
        self.phase = JobPhase.RUNNING
        self.started_at = datetime.now(timezone.utc).isoformat()
        if message:
            self.message = message

    def mark_completed(self, message: str = "") -> None:
        self.phase = JobPhase.COMPLETED
        self.finished_at = datetime.now(timezone.utc).isoformat()
        if message:
            self.message = message

    def mark_failed(self, detail: str, message: str = "") -> None:
        self.phase = JobPhase.FAILED
        self.finished_at = datetime.now(timezone.utc).isoformat()
        self.error_detail = detail
        if message:
            self.message = message

    def mark_cancelled(self) -> None:
        self.phase = JobPhase.CANCELLED
        self.finished_at = datetime.now(timezone.utc).isoformat()

    def mark_blocked(self, reason: str) -> None:
        self.phase = JobPhase.BLOCKED
        self.message = reason

    def update_progress(self, step: int, total: int, message: str = "") -> None:
        self.step = step
        self.total_steps = total
        if message:
            self.message = message

    def add_artifact(self, path: str) -> None:
        if path not in self.artifact_paths:
            self.artifact_paths.append(path)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["phase"] = self.phase.value
        return d

    def save(self, job_dir: "Path | str") -> None:
        """Write current state to ``<job_dir>/status.json``."""
        Path(job_dir).mkdir(parents=True, exist_ok=True)
        status_path = Path(job_dir) / "status.json"
        status_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, job_dir: "Path | str") -> "JobRecord":
        """Load a ``JobRecord`` from ``<job_dir>/status.json``."""
        data = json.loads((Path(job_dir) / "status.json").read_text(encoding="utf-8"))
        data["phase"] = JobPhase(data["phase"])
        return cls(**data)


def make_job_dir(outputs_root: "Path | str", job_id: str) -> Path:
    """Create and return the job directory tree."""
    job_dir = Path(outputs_root) / "jobs" / job_id
    (job_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    return job_dir
