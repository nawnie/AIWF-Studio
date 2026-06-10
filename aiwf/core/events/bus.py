from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")
Listener = Callable[[Any], None]


class EventBus:
    """Lightweight pub/sub for lifecycle hooks — replaces scattered script_callbacks."""

    def __init__(self) -> None:
        self._listeners: dict[type, list[Listener]] = defaultdict(list)

    def subscribe(self, event_type: type[T], listener: Callable[[T], None]) -> None:
        self._listeners[event_type].append(listener)

    def publish(self, event: Any) -> None:
        for listener in list(self._listeners[type(event)]):
            try:
                listener(event)
            except Exception:
                logger.exception("Event listener failed for %s", type(event).__name__)