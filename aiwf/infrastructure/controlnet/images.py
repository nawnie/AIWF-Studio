"""Control-image decoding shared by the service (UI) and backend (API) paths."""
from __future__ import annotations

import base64
import binascii
import io
from pathlib import Path

from PIL import Image


def decode_control_image(value: str | None) -> Image.Image | None:
    """Decode a ControlNetUnit.image: base64 data URL, raw base64, or a file path."""
    if not value:
        return None
    candidate = value.strip()
    if candidate.startswith("data:") and "," in candidate:
        candidate = candidate.split(",", 1)[1]
    try:
        raw = base64.b64decode(candidate, validate=True)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except (binascii.Error, ValueError, OSError):
        pass
    path = Path(value)
    if path.is_file():
        return Image.open(path).convert("RGB")
    return None
