from __future__ import annotations

import warnings
from pathlib import Path

from PIL import Image


MAX_IMAGE_ARTIFACT_BYTES = 512 * 1024 * 1024
MAX_IMAGE_ARTIFACT_PIXELS = 80_000_000
_FORMAT_BY_SUFFIX = {
    ".avif": "AVIF",
    ".bmp": "BMP",
    ".gif": "GIF",
    ".jpeg": "JPEG",
    ".jpg": "JPEG",
    ".png": "PNG",
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".webp": "WEBP",
}


def _inspect_image(path: str, size_bytes: int, expected_format: str) -> tuple[int, int] | None:
    if size_bytes <= 0 or size_bytes > MAX_IMAGE_ARTIFACT_BYTES:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                if image.format != expected_format:
                    return None
                width, height = image.size
                if width <= 0 or height <= 0 or width * height > MAX_IMAGE_ARTIFACT_PIXELS:
                    return None
                image.verify()
            with Image.open(path) as image:
                if image.format != expected_format or image.size != (width, height):
                    return None
                image.load()
        return width, height
    except (OSError, ValueError, SyntaxError, EOFError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        return None


def image_artifact_dimensions(path: str | Path) -> tuple[int, int] | None:
    """Validate a bounded image artifact without materializing its pixel array."""
    try:
        resolved = Path(path).resolve()
        expected_format = _FORMAT_BY_SUFFIX.get(resolved.suffix.lower())
        if expected_format is None:
            return None
        stat = resolved.stat()
        if not resolved.is_file():
            return None
    except OSError:
        return None
    return _inspect_image(str(resolved), stat.st_size, expected_format)
