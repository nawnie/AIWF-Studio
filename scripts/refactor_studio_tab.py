"""One-shot helper: extract studio.py body into studio/tab.py."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "aiwf/web/studio.py").read_text(encoding="utf-8")

start = src.index("def register_studio")
body = src[start:]
old_header = (
    "def register_studio(registry: WebRegistry) -> None:\n"
    "    @registry.tab(\"Studio\", order=1)\n"
    "    def build(ctx: AppContext, tab: gr.Tab | None = None) -> None:"
)
body = body.replace(old_header, "def build_studio_tab(ctx: AppContext, tab: gr.Tab | None = None) -> None:")

header = """from __future__ import annotations

import gradio as gr

from aiwf.bootstrap import AppContext
from aiwf.core.domain.models import SCHEDULE_TYPES
from aiwf.core.domain.segment_presets import segment_mask_preset_choices
from aiwf.web.components.checkpoints import checkpoint_dropdown, format_model_status
from aiwf.web.components.results import results_gallery
from aiwf.web.studio.catalogs import StudioCatalogs
from aiwf.web.studio.constants import EMPTY_CANVAS, MODE_TITLES, MODES, TOOLBAR_HINTS
from aiwf.web.studio.generation_runner import GenerationRunner
from aiwf.web.studio.handlers import compare as compare_handlers
from aiwf.web.studio.handlers import inpaint as inpaint_handlers
from aiwf.web.studio.handlers import models as model_handlers
from aiwf.web.studio.handlers import prompts as prompt_handlers
from aiwf.web.studio.handlers import reactor as reactor_handlers
from aiwf.web.studio.handlers import styles as style_handlers
from aiwf.web.studio.mode_ui import apply_mode_ui, on_mode_change
from aiwf.web.studio.session import StudioSession

"""

(ROOT / "aiwf/web/studio/tab.py").write_text(header + body, encoding="utf-8")
print("wrote tab.py")