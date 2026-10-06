"""Byte-budget VRAM planner.

llama-server router mode and llama-swap budget by *model count*.  On a single
16 GB card that is not enough: a 5.5 GiB brain, a 1 GiB ASR model and a 9 GiB
image model cannot be reasoned about as "three models".  This planner works in
GiB and answers one question: *what has to leave VRAM so this model fits?*

Eviction policy:

1. Busy models are never evicted; if they block the load the plan says WAIT.
2. Warm/transient models go first, lowest priority then least recently used.
3. Pinned models (the brain) go last and are reported in ``restore_after`` so
   the manager can bring them back once the job is done.
4. An *exclusive* model evicts every idle GPU resident, because image
   generation activations need the headroom even when the weights would fit.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from .registry import ModelSpec, Residency


class Action(str, Enum):
    NOOP = "noop"              # already resident
    LOAD = "load"              # evict ``evict`` (possibly empty), then load
    WAIT = "wait"              # busy models block the load; retry later
    IMPOSSIBLE = "impossible"  # larger than the whole budget


@dataclass
class Resident:
    model_id: str
    vram_gib: float
    residency: Residency
    priority: int
    last_used: float = 0.0
    busy: bool = False


@dataclass(frozen=True)
class Plan:
    target: str
    action: Action
    need_gib: float
    free_before_gib: float
    evict: tuple[str, ...] = ()
    restore_after: tuple[str, ...] = ()
    reason: str = ""

    @property
    def runnable(self) -> bool:
        return self.action in (Action.NOOP, Action.LOAD)


class VramPlanner:
    def __init__(self, budget_gib: float) -> None:
        if budget_gib <= 0:
            raise ValueError("VRAM budget must be positive")
        self.budget_gib = budget_gib

    def used_gib(self, residents: Mapping[str, Resident]) -> float:
        return sum(r.vram_gib for r in residents.values())

    def plan(self, spec: ModelSpec, need_gib: float, residents: Mapping[str, Resident]) -> Plan:
        free = self.budget_gib - self.used_gib(residents)

        if spec.id in residents:
            return Plan(spec.id, Action.NOOP, need_gib, free, reason="already resident")

        if not spec.on_gpu or need_gib <= 0:
            return Plan(spec.id, Action.LOAD, 0.0, free, reason="CPU-resident; no VRAM needed")

        if need_gib > self.budget_gib:
            return Plan(
                spec.id,
                Action.IMPOSSIBLE,
                need_gib,
                free,
                reason=(
                    f"needs {need_gib:.2f} GiB but the whole budget is {self.budget_gib:.2f} GiB; "
                    "lower ctx, quantize KV, or pick a smaller quant"
                ),
            )

        if free >= need_gib and not spec.exclusive:
            return Plan(spec.id, Action.LOAD, need_gib, free, reason="fits in free VRAM")

        idle = [r for r in residents.values() if not r.busy and r.vram_gib > 0]
        busy = [r for r in residents.values() if r.busy and r.vram_gib > 0]
        # Non-pinned first (by priority, then LRU); pinned last.
        idle.sort(key=lambda r: (r.residency is Residency.PINNED, r.priority, r.last_used))

        evict: list[Resident] = []
        freed = free
        for candidate in idle:
            if not spec.exclusive and freed >= need_gib:
                break
            evict.append(candidate)
            freed += candidate.vram_gib

        if freed < need_gib:
            blockers = ", ".join(r.model_id for r in busy) or "nothing"
            return Plan(
                spec.id,
                Action.WAIT,
                need_gib,
                free,
                reason=f"needs {need_gib:.2f} GiB, only {freed:.2f} GiB freeable; busy: {blockers}",
            )

        if spec.exclusive and busy:
            blockers = ", ".join(r.model_id for r in busy)
            return Plan(
                spec.id,
                Action.WAIT,
                need_gib,
                free,
                reason=f"exclusive load waits for busy models: {blockers}",
            )

        restore = tuple(r.model_id for r in evict if r.residency is Residency.PINNED)
        return Plan(
            spec.id,
            Action.LOAD,
            need_gib,
            free,
            evict=tuple(r.model_id for r in evict),
            restore_after=restore,
            reason="exclusive: clearing GPU" if spec.exclusive else f"evicting {len(evict)} model(s)",
        )
