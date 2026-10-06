from __future__ import annotations

import threading

import pytest

from simple_ai_chat.backends.fake import FakeBackend
from simple_ai_chat.jobs import Job, JobQueue
from simple_ai_chat.manager import LoadError, ModelManager
from simple_ai_chat.registry import KNOWN_BACKENDS
from simple_ai_chat.vram import Action


def make_manager(registry, backend=None, **kwargs):
    backend = backend or FakeBackend()
    manager = ModelManager(registry, {name: backend for name in KNOWN_BACKENDS}, **kwargs)
    return manager, backend


def test_ensure_loaded_evicts_and_tracks_residents(small_registry):
    manager, backend = make_manager(small_registry)
    manager.ensure_loaded("brain")      # 6 + 0.5 + 2 + 1 = 9.5
    manager.ensure_loaded("router")     # 1.0
    manager.ensure_loaded("tts")        # 2.5 -> 13.0 used
    plan = manager.ensure_loaded("asr")  # 1.5 needs 1.5 free -> fits exactly
    assert plan.action is Action.LOAD and plan.evict == ()
    plan = manager.ensure_loaded("sim")  # 5.0 -> evict router(40), then tts/asr..., then brain
    assert plan.evict[0] == "router"
    assert "sim" in manager.resident_ids()
    assert ("unload", "router") in backend.calls


def test_exclusive_job_saves_and_restores_pinned_brain(small_registry):
    manager, backend = make_manager(small_registry)
    manager.ensure_loaded("brain")
    with manager.using("image") as plan:
        assert plan.restore_after == ("brain",)
        assert manager.resident_ids() == ["image"]
        assert manager.snapshot()["pending_restore"] == ["brain"]
    # image is transient -> unloaded; brain restored with its saved KV state
    assert manager.resident_ids() == ["brain"]
    assert backend.calls[-3:] == [("unload", "image"), ("load", "brain"), ("restore_state", "brain")]
    assert ("save_state", "brain") in backend.calls


def test_hold_restore_avoids_reloading_brain_between_chain_steps(small_registry):
    manager, backend = make_manager(small_registry)
    manager.ensure_loaded("brain")
    with manager.hold_restore():
        with manager.using("pe"):
            pass
        assert "brain" not in manager.resident_ids()
        with manager.using("image"):
            pass
    assert manager.resident_ids() == ["brain"]
    assert backend.calls.count(("load", "brain")) == 2  # initial load + one restore


def test_loading_pinned_on_demand_restores_its_state(small_registry):
    manager, backend = make_manager(small_registry)
    manager.ensure_loaded("brain")
    with manager.hold_restore():
        with manager.using("image"):
            pass
        manager.ensure_loaded("brain")  # requested before the hold ended
    assert ("restore_state", "brain") in backend.calls
    assert manager.snapshot()["pending_restore"] == []


def test_measurement_replaces_estimate(small_registry):
    backend = FakeBackend(actual_gib={"asr": 2.25})
    manager, _ = make_manager(small_registry, backend, gpu_used_gib=backend.gpu_used_gib)
    manager.ensure_loaded("asr")
    assert manager.need_gib("asr") == pytest.approx(2.25)
    assert manager.snapshot()["resident"][0]["measured"] is True


def test_backend_failure_raises_and_keeps_restore_pending(small_registry):
    backend = FakeBackend(fail_on={"image"})
    manager, _ = make_manager(small_registry, backend)
    manager.ensure_loaded("brain")
    with pytest.raises(LoadError, match="simulated"):
        manager.ensure_loaded("image")
    assert manager.resident_ids() == []
    assert manager.restore_pinned() == ["brain"]


def test_busy_model_blocks_unload_and_eviction(small_registry):
    manager, _ = make_manager(small_registry)
    with manager.using("brain"):
        with pytest.raises(LoadError):
            manager.unload("brain")
        plan = manager.plan("image")
        assert plan.action is Action.WAIT
        with pytest.raises(LoadError, match="busy"):
            manager.ensure_loaded("image")
    assert manager.unload("brain")
    assert not manager.unload("brain")


def test_disabled_model_cannot_load(small_registry):
    manager, _ = make_manager(small_registry)
    with pytest.raises(LoadError, match="disabled"):
        manager.ensure_loaded("websim")


def test_next_runnable_skips_jobs_that_must_wait(small_registry):
    manager, _ = make_manager(small_registry)
    queue = JobQueue(aging_per_sec=0.0)
    image_job = queue.submit(Job("image", priority=90))
    asr_job = queue.submit(Job("asr", priority=10))
    with manager.using("brain"):
        assert manager.next_runnable(queue) is asr_job  # image must wait for the busy brain
    assert manager.next_runnable(queue) is image_job
    assert manager.next_runnable(queue) is None


def test_concurrent_users_of_same_model(small_registry):
    manager, backend = make_manager(small_registry)
    barrier = threading.Barrier(4)
    errors: list[BaseException] = []

    def worker():
        try:
            with manager.using("router"):
                barrier.wait(timeout=5)
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert backend.calls.count(("load", "router")) == 1
    assert manager.snapshot()["resident"][0]["busy"] is False
