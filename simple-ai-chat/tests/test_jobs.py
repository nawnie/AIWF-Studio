from __future__ import annotations

import pytest

from simple_ai_chat.jobs import Job, JobQueue


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock():
    return Clock()


def test_higher_priority_first_then_fifo(clock):
    queue = JobQueue(clock=clock, aging_per_sec=0.0, affinity_bonus=0.0)
    low = queue.submit(Job("a", priority=10))
    first = queue.submit(Job("b", priority=50))
    second = queue.submit(Job("c", priority=50))
    assert [j.job_id for j in queue.ordered()] == [first.job_id, second.job_id, low.job_id]
    assert queue.pick() is first
    assert len(queue) == 2


def test_affinity_prefers_resident_model(clock):
    queue = JobQueue(clock=clock, aging_per_sec=0.0, affinity_bonus=25.0)
    image = queue.submit(Job("image", priority=70))
    tts = queue.submit(Job("tts", priority=60))
    assert queue.pick(resident={"tts"}) is tts
    assert queue.pick(resident={"tts"}) is image


def test_aging_prevents_starvation(clock):
    queue = JobQueue(clock=clock, aging_per_sec=1.0, affinity_bonus=25.0)
    cold = queue.submit(Job("image", priority=50))
    clock.now = 30.0  # cold job has waited 30s: 50 + 30 = 80
    hot = queue.submit(Job("tts", priority=50))  # 50 + 0 + 25 = 75
    assert queue.pick(resident={"tts"}) is cold
    assert queue.pick(resident={"tts"}) is hot


def test_cancel_and_duplicate_ids(clock):
    queue = JobQueue(clock=clock)
    job = queue.submit(Job("a", job_id="j1"))
    with pytest.raises(ValueError):
        queue.submit(Job("a", job_id="j1"))
    assert queue.cancel(job.job_id)
    assert not queue.cancel(job.job_id)
    assert queue.pick() is None
