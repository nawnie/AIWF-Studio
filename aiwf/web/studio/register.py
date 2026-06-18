from __future__ import annotations

from aiwf.web.registry import WebRegistry
from aiwf.web.studio.tab import build_studio_tab


def register_studio(registry: WebRegistry) -> None:
    @registry.tab("Image", order=1)
    def build(ctx, tab=None):
        build_studio_tab(ctx, tab)
