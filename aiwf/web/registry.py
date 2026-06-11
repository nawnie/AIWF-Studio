from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import gradio as gr

if TYPE_CHECKING:
    from aiwf.bootstrap import AppContext

TabBuilder = Callable[["AppContext", "gr.Tab | None"], None]


@dataclass
class WebRegistry:
    tabs: list[tuple[str, TabBuilder, int]] = field(default_factory=list)

    def tab(self, name: str, order: int = 100):
        def decorator(builder: TabBuilder) -> TabBuilder:
            self.tabs.append((name, builder, order))
            self.tabs.sort(key=lambda item: item[2])
            return builder

        return decorator

    def mount(self, ctx: AppContext) -> None:
        with gr.Tabs(elem_classes=["aiwf-nav-tabs"]):
            for name, builder, _order in self.tabs:
                with gr.Tab(name, elem_classes=["aiwf-nav-tab"]) as tab:
                    builder(ctx, tab)