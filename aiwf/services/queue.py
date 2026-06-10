from __future__ import annotations

import threading
from collections import deque
from uuid import UUID

from PIL import Image

from aiwf.core.domain.errors import GenerationCancelledError
from aiwf.core.domain.generation import JobProgress, JobRecord, JobState
from aiwf.core.events.bus import EventBus
from aiwf.core.events.types import JobCancelled, JobFailed, JobFinished, JobQueued, JobStarted


class JobQueue:
    """Serializes GPU work and tracks lifecycle — replaces ad-hoc shared.state + fifo_lock."""

    def __init__(self, events: EventBus) -> None:
        self._events = events
        self._lock = threading.Lock()
        self._jobs: dict[UUID, JobRecord] = {}
        self._order: deque[UUID] = deque()
        self._active: UUID | None = None
        self._cancel_requested: set[UUID] = set()
        self._seq: dict[UUID, int] = {}
        self._counter = 0
        self._max_history = 100

    def enqueue(self, record: JobRecord) -> JobRecord:
        with self._lock:
            self._counter += 1
            self._seq[record.id] = self._counter
            self._jobs[record.id] = record
            self._order.append(record.id)
            self._prune_locked()
        self._events.publish(JobQueued(record.id, record.request))
        return record

    def _prune_locked(self) -> None:
        """Drop the oldest finished jobs beyond the history cap (caller holds lock)."""
        if len(self._jobs) <= self._max_history:
            return
        finished = [
            job_id
            for job_id in sorted(self._jobs, key=lambda jid: self._seq.get(jid, 0))
            if job_id != self._active and job_id not in self._order
        ]
        excess = len(self._jobs) - self._max_history
        for job_id in finished[:excess]:
            self._jobs.pop(job_id, None)
            self._seq.pop(job_id, None)
            self._cancel_requested.discard(job_id)

    def get(self, job_id: UUID) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def active(self) -> JobRecord | None:
        with self._lock:
            if self._active is None:
                return None
            return self._jobs.get(self._active)

    def list_recent(self, limit: int = 20) -> list[JobRecord]:
        with self._lock:
            jobs = list(self._jobs.values())
            seq = dict(self._seq)
        jobs.sort(key=lambda item: seq.get(item.id, 0), reverse=True)
        return jobs[: max(1, limit)]

    def update_progress(
        self,
        job_id: UUID,
        step: int,
        total_steps: int,
        message: str,
        preview: Image.Image | None = None,
    ) -> JobProgress | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            job.progress = JobProgress(
                job_id=job_id,
                state=job.state,
                step=step,
                total_steps=total_steps,
                message=message,
                current_image=preview,
            )
            return job.progress

    def should_cancel(self, job_id: UUID) -> bool:
        with self._lock:
            return job_id in self._cancel_requested

    def request_cancel(self, job_id: UUID | None = None) -> None:
        with self._lock:
            target = job_id or self._active
            if target is None:
                return
            self._cancel_requested.add(target)
            job = self._jobs.get(target)
            if job:
                job.state = JobState.CANCELLED

    def run_next(self, worker) -> JobRecord | None:
        with self._lock:
            if self._active is not None:
                return None
            if not self._order:
                return None
            job_id = self._order.popleft()
            self._active = job_id
            record = self._jobs[job_id]
            record.state = JobState.RUNNING

        self._events.publish(JobStarted(job_id, record.request))
        try:
            if self.should_cancel(job_id):
                raise GenerationCancelledError()
            result = worker(record)
            record.result = result
            record.state = JobState.COMPLETED
            self._events.publish(JobFinished(job_id, result))
            return record
        except GenerationCancelledError:
            record.state = JobState.CANCELLED
            self._events.publish(JobCancelled(job_id))
            return record
        except Exception as exc:
            record.state = JobState.FAILED
            record.error = str(exc)
            self._events.publish(JobFailed(job_id, str(exc)))
            raise
        finally:
            with self._lock:
                self._cancel_requested.discard(job_id)
                if self._active == job_id:
                    self._active = None
