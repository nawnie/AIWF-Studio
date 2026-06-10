"""Example plugin — shows how to hook into Aiwf Webui without touching core code."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiwf.bootstrap import AppContext


def setup(ctx: AppContext) -> None:
    ctx.events.subscribe(
        __import__("aiwf.core.events.types", fromlist=["AppStarted"]).AppStarted,
        lambda _event: print("[example_hello] Aiwf Webui is ready."),
    )