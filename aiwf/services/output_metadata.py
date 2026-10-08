"""Studio output listing and provenance, without importing the Pro web module.

The unified engine API (``aiwf/web/unified_api.py``) lists recent Studio outputs and reads
their generation settings for ``/outputs`` and ``catalog-outputs``. Those routes used to borrow
three private helpers from ``aiwf/web/pro_api.py``; importing that module pulls in torch and the
whole Pro runtime (about 5 s and several hundred MB), which defeats the point of the standalone
engine API host (``aiwf/engine_api.py``) that the native Windows app runs.

These functions follow the same logic as ``pro_api._recent_paths_from_disk``,
``pro_api._read_output_infotext`` and ``pro_api._settings_from_infotext`` (Pillow and the light
``aiwf.core.infotext`` parser only). ``tests/individual_tests/test_output_metadata.py`` compares
both implementations on the same files, so a change to Pro's helpers that this copy does not
follow fails a test instead of drifting silently.
"""

from __future__ import annotations

import heapq
from pathlib import Path
from typing import Any

from aiwf.core.infotext import parse_infotext

# the same limits Pro uses for its recent-outputs dock
RECENT_SCAN_LIMIT = 400
RECENT_INFOTEXT_MAX_CHARS = 20_000
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".avif"}


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _optional_int(value: Any, *, minimum: int = 1) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= minimum else None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def recent_paths_from_disk(root: Path, *, limit: int) -> list[Path]:
    """Newest image files under root (depth-first walk, skipping hidden folders), newest first."""
    if not root.exists():
        return []
    heap: list[tuple[float, str, Path]] = []
    inspected = 0
    stack = [root]
    # this loop walks the folder tree but stops after a bounded number of image files
    while stack and inspected < RECENT_SCAN_LIMIT:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            if inspected >= RECENT_SCAN_LIMIT:
                break
            try:
                if entry.is_dir():
                    if not entry.name.startswith("."):
                        stack.append(entry)
                    continue
                if entry.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                inspected += 1
                stat = entry.stat()
            except OSError:
                continue
            row = (stat.st_mtime, str(entry), entry)
            if len(heap) < limit:
                heapq.heappush(heap, row)
            else:
                heapq.heappushpop(heap, row)
    return [item[2] for item in sorted(heap, reverse=True)]


def read_output_infotext(path: Path, fallback: Any = "") -> str:
    """A1111-style "parameters" text from the image, or its .txt sidecar."""
    from PIL import Image

    text = _clean(fallback)
    if text:
        return text[:RECENT_INFOTEXT_MAX_CHARS]
    try:
        with Image.open(path) as image:
            image_text = getattr(image, "text", None) or {}
            text = _clean(image_text.get("parameters"))
            if not text:
                image_info = getattr(image, "info", None) or {}
                text = _clean(image_info.get("parameters"))
    except Exception:  # noqa: BLE001 - an unreadable image simply has no settings
        text = ""
    if text:
        return text[:RECENT_INFOTEXT_MAX_CHARS]

    sidecar = path.with_suffix(".txt")
    try:
        if sidecar.is_file():
            return sidecar.read_text(encoding="utf-8", errors="replace")[:RECENT_INFOTEXT_MAX_CHARS].strip()
    except OSError:
        return ""
    return ""


def settings_from_infotext(infotext: str) -> dict[str, Any]:
    """Generation settings (prompt, seed, size, model, ...) in Pro's camelCase field names."""
    text = _clean(infotext)
    if not text:
        return {}
    try:
        params = parse_infotext(text)
    except Exception:  # noqa: BLE001 - malformed text means no settings, as in Pro
        return {}

    settings: dict[str, Any] = {}
    # this block maps A1111 parameter names onto the fields Studio's UIs use
    for key, field in (("Prompt", "prompt"), ("Negative prompt", "negativePrompt"), ("Sampler", "sampler"),
                       ("Schedule type", "scheduler"), ("Model", "modelName")):
        value = _clean(params.get(key))
        if value:
            settings[field] = value
    steps = _optional_int(params.get("Steps"))
    if steps is not None:
        settings["steps"] = steps
    cfg_scale = _optional_float(params.get("CFG scale"))
    if cfg_scale is not None:
        settings["cfgScale"] = cfg_scale
    seed = _optional_int(params.get("Seed"), minimum=0)
    if seed is not None:
        settings["seed"] = seed
    width = _optional_int(params.get("Size-1") or params.get("Hires resize-1"))
    height = _optional_int(params.get("Size-2") or params.get("Hires resize-2"))
    if width is not None:
        settings["width"] = width
    if height is not None:
        settings["height"] = height
    return settings
