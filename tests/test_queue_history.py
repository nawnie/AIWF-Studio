from __future__ import annotations

from aiwf.core.events.bus import EventBus
from aiwf.core.domain.generation import GenerationRequest, JobRecord
from aiwf.services.queue import JobQueue


def _q():
    return JobQueue(EventBus())


def test_list_recent_is_chronological_newest_first():
    q = _q()
    records = [q.enqueue(JobRecord(request=GenerationRequest(prompt=f"p{i}"))) for i in range(5)]
    recent = q.list_recent(10)
    assert [j.id for j in recent] == [r.id for r in reversed(records)]


def test_list_recent_respects_limit():
    q = _q()
    for i in range(10):
        q.enqueue(JobRecord(request=GenerationRequest(prompt=str(i))))
    assert len(q.list_recent(3)) == 3


def test_history_is_bounded():
    q = _q()
    q._max_history = 5
    for i in range(20):
        record = q.enqueue(JobRecord(request=GenerationRequest(prompt=str(i))))
        # simulate the job leaving the pending queue (finished)
        q._order.clear()
    assert len(q._jobs) <= q._max_history + 1
