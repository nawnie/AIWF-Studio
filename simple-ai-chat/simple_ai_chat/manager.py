"""ModelManager: owns what is resident in VRAM and executes VramPlanner plans.

Typical use::

    with manager.using("qwen-image-2.1"):   # evicts the brain (KV saved), loads image model
        run_image_job()
    # image model is transient -> unloaded; pinned brain restored (KV restored)

All state changes happen under one re-entrant lock, so loads are serialized:
only one model is ever being loaded onto the GPU at a time.
"""
from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from typing import Callable, Iterator, Mapping

from .backends.base import Backend, BackendError
from .jobs import Job, JobQueue
from .registry import ModelSpec, Registry, Residency
from .vram import Action, Plan, Resident, VramPlanner

logger = logging.getLogger(__name__)


class LoadError(RuntimeError):
    """A model could not be made resident."""

    def __init__(self, message: str, plan: Plan | None = None) -> None:
        super().__init__(message)
        self.plan = plan


class ModelManager:
    def __init__(
        self,
        registry: Registry,
        backends: Mapping[str, Backend],
        *,
        planner: VramPlanner | None = None,
        clock: Callable[[], float] = time.monotonic,
        gpu_used_gib: Callable[[], float | None] | None = None,
    ) -> None:
        self.registry = registry
        self.planner = planner or VramPlanner(registry.hardware.vram_budget_gib)
        self._backends = dict(backends)
        self._clock = clock
        self._gpu_used = gpu_used_gib
        self._resident: dict[str, Resident] = {}
        self._busy: dict[str, int] = {}
        self._measured: dict[str, float] = {}
        self._pending_restore: list[str] = []
        self._restore_holds = 0
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def resident_ids(self) -> list[str]:
        with self._lock:
            return list(self._resident)

    def need_gib(self, model_id: str) -> float:
        spec = self.registry.get(model_id)
        if not spec.on_gpu:
            return 0.0
        return self._measured.get(model_id, spec.vram_gib())

    def plan(self, model_id: str) -> Plan:
        spec = self.registry.get(model_id)
        with self._lock:
            return self.planner.plan(spec, self.need_gib(model_id), self._resident)

    def record_measurement(self, model_id: str, gib: float) -> None:
        """Store a measured VRAM footprint; future plans use it instead of the estimate."""
        if gib > 0:
            self._measured[model_id] = gib

    def snapshot(self) -> dict:
        with self._lock:
            used = self.planner.used_gib(self._resident)
            return {
                "budget_gib": round(self.planner.budget_gib, 2),
                "used_gib": round(used, 2),
                "free_gib": round(self.planner.budget_gib - used, 2),
                "resident": [
                    {
                        "id": r.model_id,
                        "vram_gib": round(r.vram_gib, 2),
                        "residency": r.residency.value,
                        "busy": r.busy,
                        "measured": r.model_id in self._measured,
                    }
                    for r in self._resident.values()
                ],
                "pending_restore": list(self._pending_restore),
            }

    # ------------------------------------------------------------------
    # Loading / unloading
    # ------------------------------------------------------------------

    def ensure_loaded(self, model_id: str) -> Plan:
        """Make *model_id* resident, evicting per plan.  Raises LoadError if it cannot."""
        spec = self.registry.get(model_id)
        if not spec.enabled:
            raise LoadError(f"{model_id} is disabled in the registry")
        with self._lock:
            plan = self.planner.plan(spec, self.need_gib(model_id), self._resident)
            if plan.action is Action.NOOP:
                return plan
            if not plan.runnable:
                raise LoadError(f"cannot load {model_id}: {plan.reason}", plan)

            for victim in plan.evict:
                self._evict(victim, save_state=victim in plan.restore_after)
            for pinned in plan.restore_after:
                if pinned not in self._pending_restore:
                    self._pending_restore.append(pinned)

            self._load(spec)
            if model_id in self._pending_restore:
                # Loaded on demand before restore_pinned() got to it; still restore its state.
                self._pending_restore.remove(model_id)
                self._restore_state(spec)
            return plan

    def unload(self, model_id: str) -> bool:
        with self._lock:
            if model_id not in self._resident:
                return False
            if self._busy.get(model_id):
                raise LoadError(f"{model_id} is busy and cannot be unloaded")
            self._evict(model_id, save_state=False)
            return True

    @contextmanager
    def hold_restore(self) -> Iterator[None]:
        """Defer restoring pinned models until a multi-step chain finishes.

        Without this, "prompt enhancer -> image model" would evict the brain,
        restore it, and evict it again between the two steps.
        """
        with self._lock:
            self._restore_holds += 1
        try:
            yield
        finally:
            with self._lock:
                self._restore_holds -= 1
                if self._restore_holds == 0:
                    self.restore_pinned()

    def restore_pinned(self) -> list[str]:
        """Reload pinned models evicted by earlier plans, if they fit without evicting another pinned model."""
        restored: list[str] = []
        with self._lock:
            if self._restore_holds:
                return restored
            for model_id in list(self._pending_restore):
                spec = self.registry.get(model_id)
                plan = self.planner.plan(spec, self.need_gib(model_id), self._resident)
                if plan.action is Action.NOOP:
                    self._pending_restore.remove(model_id)
                    continue
                if not plan.runnable or plan.restore_after:
                    continue
                for victim in plan.evict:
                    self._evict(victim, save_state=False)
                try:
                    self._load(spec)
                except LoadError:
                    logger.exception("[vram] failed to restore pinned model %s", model_id)
                    continue
                self._pending_restore.remove(model_id)
                self._restore_state(spec)
                restored.append(model_id)
        return restored

    @contextmanager
    def using(self, model_id: str) -> Iterator[Plan]:
        """Hold *model_id* resident and busy for the duration of a job."""
        spec = self.registry.get(model_id)
        with self._lock:
            plan = self.ensure_loaded(model_id)
            self._busy[model_id] = self._busy.get(model_id, 0) + 1
            self._touch(model_id, busy=True)
        try:
            yield plan
        finally:
            with self._lock:
                remaining = self._busy.get(model_id, 1) - 1
                if remaining > 0:
                    self._busy[model_id] = remaining
                else:
                    self._busy.pop(model_id, None)
                self._touch(model_id, busy=remaining > 0)
                if remaining <= 0 and spec.residency is Residency.TRANSIENT and model_id in self._resident:
                    self._evict(model_id, save_state=False)
                self.restore_pinned()

    def next_runnable(self, queue: JobQueue) -> Job | None:
        """Remove and return the best queued job whose model can be made resident now."""
        with self._lock:
            resident = set(self._resident)
            for job in queue.ordered(resident):
                if job.model_id not in self.registry or not self.registry.get(job.model_id).enabled:
                    continue
                if self.plan(job.model_id).runnable:
                    queue.remove(job.job_id)
                    return job
            return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _backend_for(self, spec: ModelSpec) -> Backend:
        try:
            return self._backends[spec.backend]
        except KeyError as exc:
            raise LoadError(f"no backend registered for {spec.backend!r} ({spec.id})") from exc

    def _touch(self, model_id: str, *, busy: bool) -> None:
        resident = self._resident.get(model_id)
        if resident is not None:
            resident.busy = busy
            resident.last_used = self._clock()

    def _load(self, spec: ModelSpec) -> None:
        backend = self._backend_for(spec)
        before = self._gpu_used() if (self._gpu_used and spec.on_gpu) else None
        try:
            backend.load(spec)
        except BackendError as exc:
            raise LoadError(f"backend failed to load {spec.id}: {exc}") from exc
        estimate = self.need_gib(spec.id)
        vram = estimate
        if before is not None:
            after = self._gpu_used()
            if after is not None and after - before > 0:
                vram = after - before
                self.record_measurement(spec.id, vram)
                if abs(vram - estimate) > 0.5:
                    logger.info("[vram] %s measured %.2f GiB vs estimate %.2f GiB", spec.id, vram, estimate)
        self._resident[spec.id] = Resident(
            model_id=spec.id,
            vram_gib=vram,
            residency=spec.residency,
            priority=spec.priority,
            last_used=self._clock(),
        )
        logger.info("[vram] loaded %s (%.2f GiB)", spec.id, vram)

    def _restore_state(self, spec: ModelSpec) -> None:
        restore = getattr(self._backend_for(spec), "restore_state", None)
        if callable(restore) and not restore(spec.id):
            logger.info("[vram] %s restored without saved state (will re-prefill)", spec.id)

    def _evict(self, model_id: str, *, save_state: bool) -> None:
        spec = self.registry.get(model_id)
        backend = self._backend_for(spec)
        if save_state:
            save = getattr(backend, "save_state", None)
            if callable(save) and not save(model_id):
                logger.warning("[vram] could not save state for %s before eviction", model_id)
        backend.unload(model_id)
        self._resident.pop(model_id, None)
        logger.info("[vram] evicted %s", model_id)
