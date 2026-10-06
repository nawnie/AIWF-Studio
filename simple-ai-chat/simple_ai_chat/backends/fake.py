"""In-memory backend for tests and ``--dry-run`` planning.

It pretends to load models and tracks the VRAM they would use, so the whole
scheduler can be exercised on a machine without a GPU.
"""
from __future__ import annotations

from ..registry import ModelSpec
from .base import BackendError


class FakeBackend:
    def __init__(self, *, fail_on: set[str] | None = None, actual_gib: dict[str, float] | None = None) -> None:
        self.loaded: dict[str, float] = {}
        self.calls: list[tuple[str, str]] = []
        self.saved: set[str] = set()
        self._fail_on = set(fail_on or ())
        self._actual = dict(actual_gib or {})

    def load(self, spec: ModelSpec) -> None:
        self.calls.append(("load", spec.id))
        if spec.id in self._fail_on:
            raise BackendError(f"simulated load failure for {spec.id}")
        self.loaded[spec.id] = self._actual.get(spec.id, spec.vram_gib())

    def unload(self, model_id: str) -> None:
        self.calls.append(("unload", model_id))
        self.loaded.pop(model_id, None)

    def save_state(self, model_id: str) -> bool:
        self.calls.append(("save_state", model_id))
        self.saved.add(model_id)
        return True

    def restore_state(self, model_id: str) -> bool:
        self.calls.append(("restore_state", model_id))
        return model_id in self.saved

    def gpu_used_gib(self) -> float:
        return sum(self.loaded.values())
