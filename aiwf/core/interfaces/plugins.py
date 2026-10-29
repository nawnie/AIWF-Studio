from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel

if TYPE_CHECKING:
    from aiwf.bootstrap import AppContext


class AiwfPlugin(Protocol):
    """Extension point — plugins register tabs, samplers, hooks without touching core."""

    name: str
    version: str

    def on_load(self, ctx: AppContext) -> None: ...


class PluginInfo(BaseModel):
    id: str
    name: str
    version: str = "0.0.0"
    description: str = ""
    path: str = ""
    enabled: bool = True
    error: str | None = None
