from __future__ import annotations

import html
import logging
import platform
import threading
from functools import lru_cache
from pathlib import Path

import gradio as gr

from aiwf.bootstrap import AppContext
from aiwf.web.components.checkpoints import resolve_default_checkpoint
from aiwf.web.registry import WebRegistry
from aiwf.web.studio import register_studio
from aiwf.web.tabs.enhance import register_enhance
from aiwf.web.tabs.faceswap import register_faceswap
from aiwf.web.tabs.history import register_history
from aiwf.web.tabs.wan_i2v import register_wan_i2v
from aiwf.web.tabs.library import register_library
from aiwf.web.tabs.model_manager import register_model_manager
from aiwf.web.tabs.pnginfo import register_pnginfo
from aiwf.web.tabs.settings import register_settings
from aiwf.web.tabs.workflows import register_workflows
from aiwf.web.theme import build_theme, theme_css_overrides

logger = logging.getLogger(__name__)
_STATIC_DIR = Path(__file__).resolve().parents[2] / "static"


@lru_cache(maxsize=8)
def _static_text(name: str) -> str:
    path = _STATIC_DIR / name
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _preload_default_checkpoint(ctx: AppContext) -> None:
    try:
        checkpoints = ctx.generation.list_checkpoints()
        target = resolve_default_checkpoint(checkpoints, ctx.settings.last_checkpoint_id)
        if target is None:
            return
        ctx.generation.load_checkpoint(target.id)
        logger.info("Preloaded checkpoint: %s", target.title)
    except Exception:
        logger.exception("Background checkpoint preload failed")


def _topbar_runtime_html(ctx: AppContext) -> str:
    try:
        import torch

        torch_version = torch.__version__.split("+", 1)[0]
    except Exception:
        torch_version = "unavailable"

    if ctx.flags.xformers:
        attention = "xFormers"
    elif ctx.flags.opt_sdp_attention or ctx.flags.opt_split_attention:
        attention = "SDP"
    else:
        attention = "Default"

    device = ctx.generation.backend.devices.describe()
    if device.startswith("CUDA ("):
        device = device.removeprefix("CUDA (").rstrip(")")
    elif device.startswith("CPU ("):
        device = "CPU"

    badges = [
        ("Python", platform.python_version()),
        ("Torch", torch_version),
        ("Attention", attention),
        ("Device", device),
        ("Models", str(len(ctx.generation.list_checkpoints()))),
    ]
    chips = "".join(
        (
            '<span class="aiwf-runtime-chip">'
            f'<span class="aiwf-runtime-label">{html.escape(label)}</span>'
            f'<span class="aiwf-runtime-value">{html.escape(value)}</span>'
            "</span>"
        )
        for label, value in badges
    )
    return f'<div class="aiwf-runtime-summary" aria-label="Runtime summary">{chips}</div>'


def create_web_ui(ctx: AppContext) -> tuple[gr.Blocks, object, str, str]:
    registry = WebRegistry()
    register_studio(registry)
    register_model_manager(registry)
    register_enhance(registry)
    register_faceswap(registry)
    register_workflows(registry)
    register_history(registry)
    register_wan_i2v(registry)
    register_library(registry)
    register_pnginfo(registry)
    register_settings(registry)

    with gr.Blocks(title="AIWF Studio", elem_classes=["aiwf-app"]) as demo:
        gr.HTML(
            """
            <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
            <meta name="mobile-web-app-capable" content="yes">
            <meta name="apple-mobile-web-app-capable" content="yes">
            <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
            """,
            elem_classes=["aiwf-viewport-meta"],
            visible=False,
        )
        gr.HTML(theme_css_overrides(preset=ctx.settings.accent_preset), visible=False)
        gr.HTML(
            (
                '<div id="aiwf-client-settings" hidden '
                f'data-title-progress="{str(ctx.settings.live_preview_title_progress).lower()}">'
                "</div>"
            )
        )
        gr.HTML(
            f"""
            <header class="aiwf-topbar" aria-label="Application header">
                <div class="aiwf-topbar-start">
                    <div class="aiwf-brand-lockup">
                        <span class="aiwf-logo" aria-hidden="true">
                            <svg width="28" height="28" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <rect x="1" y="1" width="26" height="26" rx="8" stroke="url(#aiwf-logo-stroke)" stroke-width="1.5"/>
                                <path d="M8 18L14 8L20 18" stroke="url(#aiwf-logo-stroke)" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"/>
                                <circle cx="14" cy="15" r="2.25" fill="url(#aiwf-logo-fill)"/>
                                <defs>
                                    <linearGradient id="aiwf-logo-stroke" x1="4" y1="4" x2="24" y2="24">
                                        <stop stop-color="#8fd8c4"/>
                                        <stop offset="1" stop-color="#5da892"/>
                                    </linearGradient>
                                    <linearGradient id="aiwf-logo-fill" x1="12" y1="13" x2="16" y2="17">
                                        <stop stop-color="#b6eadc"/>
                                        <stop offset="1" stop-color="#68c3aa"/>
                                    </linearGradient>
                                </defs>
                            </svg>
                        </span>
                        <div class="aiwf-brand-text">
                            <span class="aiwf-name">AIWF</span>
                            <span class="aiwf-edition">Studio</span>
                        </div>
                    </div>
                    <p class="aiwf-tagline">Professional diffusion workspace</p>
                </div>
                <div class="aiwf-topbar-end">
                    {_topbar_runtime_html(ctx)}
                    <span class="aiwf-status-pill" id="aiwf-topbar-status" data-state="ready" role="status" aria-live="polite">
                        <span class="aiwf-status-dot" aria-hidden="true"></span>
                        <span class="aiwf-status-label">Ready</span>
                    </span>
                </div>
            </header>
            """,
            elem_classes=["aiwf-topbar-wrap"],
        )
        registry.mount(ctx)

        def startup():
            threading.Thread(
                target=_preload_default_checkpoint,
                args=(ctx,),
                name="aiwf-checkpoint-preload",
                daemon=True,
            ).start()

        demo.load(fn=startup, inputs=None, outputs=None, show_progress=False)

    return (
        demo,
        build_theme(dark=ctx.flags.theme == "dark", accent_preset=ctx.settings.accent_preset),
        _static_text("style.css"),
        _static_text("studio.js"),
    )
