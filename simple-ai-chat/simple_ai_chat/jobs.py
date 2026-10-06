"""Priority job queue with model affinity.

Every job names the model it needs.  Picking strictly by priority makes a
single GPU thrash: TTS chunk, image, TTS chunk, image... each one a model
swap.  The queue therefore scores jobs as::

    score = priority + aging_per_sec * seconds_waiting + affinity_bonus (if model resident)

so work for an already-loaded model runs first, while aging guarantees that a
job for a cold model is eventually picked even under a steady stream of
resident-model work.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Collection


@dataclass
class Job:
    model_id: str
    priority: int = 50
    payload: Any = None
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    submitted_at: float = 0.0
    seq: int = 0


class JobQueue:
    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        aging_per_sec: float = 1.0,
        affinity_bonus: float = 25.0,
    ) -> None:
        self._clock = clock
        self._aging = aging_per_sec
        self._affinity = affinity_bonus
        self._jobs: dict[str, Job] = {}
        self._seq = 0
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._jobs)

    def submit(self, job: Job) -> Job:
        with self._lock:
            if job.job_id in self._jobs:
                raise ValueError(f"duplicate job id {job.job_id!r}")
            self._seq += 1
            job.seq = self._seq
            job.submitted_at = self._clock()
            self._jobs[job.job_id] = job
            return job

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            return self._jobs.pop(job_id, None) is not None

    def remove(self, job_id: str) -> Job:
        with self._lock:
            return self._jobs.pop(job_id)

    def score(self, job: Job, resident: Collection[str], now: float | None = None) -> float:
        now = self._clock() if now is None else now
        bonus = self._affinity if job.model_id in resident else 0.0
        return job.priority + self._aging * max(0.0, now - job.submitted_at) + bonus

    def _ordered_locked(self, resident: Collection[str]) -> list[Job]:
        now = self._clock()
        return sorted(self._jobs.values(), key=lambda j: (-self.score(j, resident, now), j.seq))

    def ordered(self, resident: Collection[str] = ()) -> list[Job]:
        """Jobs best-first without removing them (ties broken FIFO)."""
        with self._lock:
            return self._ordered_locked(resident)

    def pick(self, resident: Collection[str] = ()) -> Job | None:
        with self._lock:
            ordered = self._ordered_locked(resident)
            if not ordered:
                return None
            return self._jobs.pop(ordered[0].job_id)
