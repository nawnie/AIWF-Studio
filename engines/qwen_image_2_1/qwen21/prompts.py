"""Bundled text assets (prompt-enhancer system prompts from the MIT-licensed Comfy-Org templates)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


@lru_cache(maxsize=None)
def load_pe_system_prompt(kind: str) -> str:
    """kind: 't2i' or 'i2i'. Strips the '#!' attribution header lines."""
    path = PROMPTS_DIR / f"pe_{kind}_system.txt"
    if not path.is_file():
        return ""
    lines = path.read_text(encoding="utf-8").split("\n")
    i = 0
    while i < len(lines) and lines[i].startswith("#!"):
        i += 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    return "\n".join(lines[i:]).strip()
