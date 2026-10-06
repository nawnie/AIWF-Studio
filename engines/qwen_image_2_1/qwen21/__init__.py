"""Qwen Image 2.1 Studio — ComfyUI-backed generate/edit/train desktop app.

Pure-Python modules (presets, workflow builder, ComfyUI client, training
config generators) live here and never import PySide6, so they can be unit
tested and reused from the CLI. The GUI lives in ``qwen21.ui``.
"""
from __future__ import annotations

__version__ = "0.1.0"
