from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import gradio as gr

if TYPE_CHECKING:
    from aiwf.bootstrap import AppContext

TabBuilder = Callable[["AppContext", "gr.Tab | None"], None]
PINNED_TABS = {"Image", "Settings"}


@dataclass
class WebRegistry:
    tabs: list[tuple[str, TabBuilder, int]] = field(default_factory=list)

    def tab(self, name: str, order: int = 100):
        def decorator(builder: TabBuilder) -> TabBuilder:
            self.tabs.append((name, builder, order))
            self.tabs.sort(key=lambda item: item[2])
            return builder

        return decorator

    def visible_tabs(self, ctx: AppContext) -> list[tuple[str, TabBuilder, int]]:
        hidden = set(getattr(ctx.settings, "hidden_tabs", []))
        return [
            (name, builder, order)
            for name, builder, order in self.tabs
            if name not in hidden or name in PINNED_TABS
        ]

    def mount(self, ctx: AppContext) -> None:
        with gr.Tabs(elem_classes=["aiwf-nav-tabs"]):
            for name, builder, _order in self.visible_tabs(ctx):
                with gr.Tab(name, elem_classes=["aiwf-nav-tab"]) as tab:
                    builder(ctx, tab)
