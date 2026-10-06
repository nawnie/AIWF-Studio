"""Backend contract used by the ModelManager.

A backend knows how to put one model into (or take it out of) a runtime:
a llama-server child process, an sd.cpp server, a TTS process, and so on.
The manager never touches processes directly; it only calls these methods.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..registry import ModelSpec


class BackendError(RuntimeError):
    """Raised by a backend when a model fails to load or unload."""


@runtime_checkable
class Backend(Protocol):
    def load(self, spec: ModelSpec) -> None:
        """Start serving *spec*; return only once it is ready for requests."""

    def unload(self, model_id: str) -> None:
        """Stop serving *model_id* and release its VRAM."""


@runtime_checkable
class StatefulBackend(Backend, Protocol):
    """Backends that can persist a model's KV cache across an eviction."""

    def save_state(self, model_id: str) -> bool:
        """Persist conversation state before eviction; True on success."""

    def restore_state(self, model_id: str) -> bool:
        """Restore state saved by :meth:`save_state`; True on success."""
