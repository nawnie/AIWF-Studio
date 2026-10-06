from __future__ import annotations

import pytest

from simple_ai_chat.registry import Residency
from simple_ai_chat.vram import Action, Resident, VramPlanner

from .conftest import spec


def resident(model_id, gib, residency=Residency.WARM, priority=50, last_used=0.0, busy=False):
    return Resident(model_id, gib, residency, priority, last_used, busy)


def residents(*items):
    return {r.model_id: r for r in items}


@pytest.fixture
def planner():
    return VramPlanner(14.5)


def test_noop_when_already_resident(planner):
    target = spec("a", weights_gib=2.0)
    plan = planner.plan(target, 2.0, residents(resident("a", 2.0)))
    assert plan.action is Action.NOOP and plan.runnable


def test_load_when_it_fits(planner):
    plan = planner.plan(spec("a", weights_gib=4.0), 4.0, residents(resident("brain", 9.0, Residency.PINNED, 100)))
    assert plan.action is Action.LOAD
    assert plan.evict == ()
    assert plan.free_before_gib == pytest.approx(5.5)


def test_evicts_lowest_priority_then_lru(planner):
    current = residents(
        resident("brain", 9.0, Residency.PINNED, 100),
        resident("tts", 2.0, priority=60, last_used=1.0),
        resident("router", 1.0, priority=40, last_used=5.0),
        resident("asr", 1.5, priority=60, last_used=0.5),
    )
    # free = 14.5 - 13.5 = 1.0; needs 3.0 -> evict router (prio 40) then asr (older than tts)
    plan = planner.plan(spec("x", weights_gib=3.0), 3.0, current)
    assert plan.action is Action.LOAD
    assert plan.evict == ("router", "asr")
    assert plan.restore_after == ()


def test_pinned_evicted_last_and_marked_for_restore(planner):
    current = residents(resident("brain", 9.0, Residency.PINNED, 100), resident("asr", 1.5, priority=60))
    plan = planner.plan(spec("big", weights_gib=10.0), 10.0, current)
    assert plan.evict == ("asr", "brain")
    assert plan.restore_after == ("brain",)


def test_busy_models_are_never_evicted(planner):
    current = residents(resident("brain", 9.0, Residency.PINNED, 100, busy=True), resident("asr", 1.5))
    plan = planner.plan(spec("big", weights_gib=10.0), 10.0, current)
    assert plan.action is Action.WAIT
    assert "brain" in plan.reason
    assert not plan.runnable


def test_exclusive_clears_every_idle_model(planner):
    current = residents(resident("brain", 9.0, Residency.PINNED, 100), resident("router", 1.0, priority=40))
    plan = planner.plan(spec("image", exclusive=True, weights_gib=2.0), 2.0, current)
    assert plan.action is Action.LOAD
    assert set(plan.evict) == {"brain", "router"}
    assert plan.restore_after == ("brain",)


def test_exclusive_waits_for_busy_even_if_it_would_fit(planner):
    current = residents(resident("asr", 1.0, busy=True))
    plan = planner.plan(spec("image", exclusive=True, weights_gib=2.0), 2.0, current)
    assert plan.action is Action.WAIT


def test_impossible_when_larger_than_budget(planner):
    plan = planner.plan(spec("huge", weights_gib=20.0), 20.0, {})
    assert plan.action is Action.IMPOSSIBLE
    assert "budget" in plan.reason


def test_cpu_models_never_need_vram(planner):
    full = residents(resident("brain", 14.5, Residency.PINNED, 100, busy=True))
    plan = planner.plan(spec("embed", residency="cpu", weights_gib=1.0), 0.0, full)
    assert plan.action is Action.LOAD and plan.evict == ()


def test_budget_must_be_positive():
    with pytest.raises(ValueError):
        VramPlanner(0)
