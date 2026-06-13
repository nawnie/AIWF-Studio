"""Example plugin - shows how to hook into Aiwf Webui without touching core code."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiwf.bootstrap import AppContext

logger = logging.getLogger(__name__)


def setup(ctx: AppContext) -> None:
    ctx.events.subscribe(
        __import__("aiwf.core.events.types", fromlist=["AppStarted"]).AppStarted,
        lambda _event: logger.debug("Example hello plugin received app start."),
    )
