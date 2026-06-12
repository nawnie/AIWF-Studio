import threading
import time

from aiwf.core.domain.generation import GenerationMode, GenerationRequest, GenerationResult, JobRecord, JobState
from aiwf.core.events.bus import EventBus
from aiwf.services.queue import JobQueue


def _result(job: JobRecord) -> GenerationResult:
    return GenerationResult(
        job_id=job.id,
        mode=job.request.mode,
        images=[],
        infotexts=[],
        seeds=[],
    )


def test_queue_runs_single_job():
    events = EventBus()
    queue = JobQueue(events)
    record = JobRecord(request=GenerationRequest())
    queue.enqueue(record)

    def worker(job: JobRecord):
        return _result(job)

    finished = queue.run_next(worker)
    assert finished is not None
    assert finished.state == JobState.COMPLETED
    queue.request_cancel(record.id)
    assert queue.should_cancel(record.id)


def test_run_next_blocks_until_active_job_finishes():
    events = EventBus()
    queue = JobQueue(events)
    first = JobRecord(request=GenerationRequest())
    second = JobRecord(request=GenerationRequest())
    queue.enqueue(first)
    queue.enqueue(second)

    started = threading.Event()
    release = threading.Event()
    order: list[int] = []

    def worker_first(job: JobRecord):
        order.append(1)
        started.set()
        assert release.wait(timeout=5)
        return _result(job)

    def worker_second(job: JobRecord):
        order.append(2)
        return _result(job)

    first_thread = threading.Thread(target=lambda: queue.run_next(worker_first, block=True))
    second_thread = threading.Thread(target=lambda: queue.run_next(worker_second, block=True))
    first_thread.start()
    assert started.wait(timeout=2)
    second_thread.start()
    time.sleep(0.2)
    assert order == [1]

    release.set()
    first_thread.join(timeout=5)
    second_thread.join(timeout=5)
    assert order == [1, 2]