from __future__ import annotations

import threading
import time
from collections import deque
from uuid import UUID

from PIL import Image

from aiwf.core.domain.errors import GenerationCancelledError
from aiwf.core.domain.generation import JobProgress, JobRecord, JobState
from aiwf.core.events.bus import EventBus
from aiwf.core.events.types import JobCancelled, JobFailed, JobFinished, JobQueued, JobStarted
from aiwf.dev.diagnostics import trace_exception_safe, trace_safe


class JobQueue:
    """Serializes GPU work and tracks lifecycle — replaces ad-hoc shared.state + fifo_lock."""

    def __init__(self, events: EventBus) -> None:
        self._events = events
        self._lock = threading.Lock()
        self._slot_ready = threading.Condition(self._lock)
        self._jobs: dict[UUID, JobRecord] = {}
        self._order: deque[UUID] = deque()
        self._active: UUID | None = None
        self._cancel_requested: set[UUID] = set()
        self._seq: dict[UUID, int] = {}
        self._counter = 0
        self._max_history = 100
        # Last time each running job reported any progress (model-load message,
        # "Encoding prompt", a denoise step, ...). Used by a stall watchdog to
        # detect a job that has gone completely silent — see seconds_since_progress.
        self._progress_at: dict[UUID, float] = {}
        # Jobs the watchdog has force-failed because they stalled. The worker
        # thread for one of these may still be alive (stuck in a blocking call
        # we cannot interrupt) — this set lets run_next() avoid clobbering the
        # forced FAILED state if that thread ever does return.
        self._abandoned: set[UUID] = set()

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
            self._progress_at[job_id] = time.monotonic()
            return job.progress

    def seconds_since_progress(self, job_id: UUID) -> float | None:
        """How long since this job last reported progress, or None if unknown."""
        with self._lock:
            last = self._progress_at.get(job_id)
        if last is None:
            return None
        return time.monotonic() - last

    def force_fail_stalled(self, job_id: UUID, reason: str) -> bool:
        """Forcibly fail a job that has gone silent for too long.

        Used by a stall watchdog when a worker thread is stuck inside a
        blocking call (e.g. CPU-offload weight transfer / text encoding) that
        offers no cooperative cancellation point. We cannot kill that thread,
        but we *can* stop it from blocking everything else forever: mark the
        job FAILED, release its slot, and let the caller force-release the
        GPU tenant lock. Returns True if it actually force-failed something.
        """
        with self._lock:
            if self._active != job_id:
                return False
            job = self._jobs.get(job_id)
            if job is None or job.state != JobState.RUNNING:
                return False
            job.state = JobState.FAILED
            job.error = reason
            self._abandoned.add(job_id)
            self._cancel_requested.discard(job_id)
            self._active = None
            self._slot_ready.notify_all()
        trace_safe("queue.force_failed", reason, job_id=str(job_id))
        self._events.publish(JobFailed(job_id, reason))
        return True

    def is_abandoned(self, job_id: UUID) -> bool:
        """True if a stall watchdog already force-failed this job.

        Lets a worker that's about to return late (after being force-failed
        for stalling) check whether it should skip post-processing/saving
        instead of touching a tenant lock that's already been released.
        """
        with self._lock:
            return job_id in self._abandoned

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

    def run_next(self, worker, *, block: bool = False) -> JobRecord | None:
        with self._lock:
            while self._active is not None:
                if not block:
                    return None
                self._slot_ready.wait()
            if not self._order:
                return None
            job_id = self._order.popleft()
            self._active = job_id
            record = self._jobs[job_id]
            record.state = JobState.RUNNING
            self._progress_at[job_id] = time.monotonic()

        self._events.publish(JobStarted(job_id, record.request))
        try:
            if self.should_cancel(job_id):
                raise GenerationCancelledError()
            result = worker(record)
            with self._lock:
                abandoned = job_id in self._abandoned
            if abandoned:
                # A stall watchdog already force-failed this job and released
                # its slot/lock. The worker thread was stuck and has now
                # returned late — don't resurrect it as COMPLETED.
                trace_safe(
                    "queue.late_return_after_force_fail",
                    "Worker returned after being force-failed for stalling",
                    job_id=str(job_id),
                )
                return record
            record.result = result
            record.state = JobState.COMPLETED
            self._events.publish(JobFinished(job_id, result))
            return record
        except GenerationCancelledError:
            with self._lock:
                abandoned = job_id in self._abandoned
            if not abandoned:
                record.state = JobState.CANCELLED
                trace_safe("queue.cancelled", "Worker cancelled", job_id=str(job_id))
                self._events.publish(JobCancelled(job_id))
            return record
        except Exception as exc:
            with self._lock:
                abandoned = job_id in self._abandoned
            if not abandoned:
                record.state = JobState.FAILED
                record.error = str(exc)
                trace_exception_safe(
                    "queue.worker_failed",
                    exc,
                    job_id=str(job_id),
                    mode=record.request.mode.value,
                    checkpoint_id=record.request.checkpoint_id,
                )
                self._events.publish(JobFailed(job_id, str(exc)))
            raise
        finally:
            with self._lock:
                self._cancel_requested.discard(job_id)
                self._abandoned.discard(job_id)
                self._progress_at.pop(job_id, None)
                if self._active == job_id:
                    self._active = None
                self._slot_ready.notify_all()
