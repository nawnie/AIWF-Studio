from __future__ import annotations

from PIL import Image

from aiwf.infrastructure.diffusers.mask import align_to_multiple_of_8

# Longest-side caps for upload resize. 768 is the safe max for ~8GB VRAM (16:9 → 768×432).
MAX_BUCKET_8GB = 768

RESOLUTION_BUCKETS: tuple[int, ...] = (
    512,
    576,
    640,
    704,
    MAX_BUCKET_8GB,
)

DEFAULT_UPLOAD_BUCKET = MAX_BUCKET_8GB

GENERATION_SIZE_PRESETS: tuple[int, ...] = (
    512,
    568,
    640,
    768,
    896,
    1024,
    1080,
)

ASPECT_RATIO_PRESETS: tuple[tuple[str, str], ...] = (
    ("1:1", "1:1"),
    ("4:3", "4:3"),
    ("3:4", "3:4"),
    ("3:2", "3:2"),
    ("2:3", "2:3"),
    ("16:9", "16:9"),
    ("9:16", "9:16"),
)

NON_SQUARE_ASPECT_RATIO_PRESETS: tuple[tuple[str, str], ...] = tuple(
    item for item in ASPECT_RATIO_PRESETS if item[1] != "1:1"
)


def _example_dims(long_side: int) -> tuple[int, int, int, int]:
    """Return (w_16_9, h_16_9, w_1_1, h_1_1) aligned to multiples of 8."""
    w16 = long_side
    h16 = max(8, (long_side * 9 // 16) // 8 * 8)
    return w16, h16, long_side, long_side


def bucket_label(long_side: int, *, recommended: bool = False) -> str:
    w16, h16, w1, h1 = _example_dims(long_side)
    tag = " · 8GB safe default" if recommended else ""
    return f"{long_side} px longest — 16:9 {w16}×{h16} · square {w1}×{h1}{tag}"


BUCKET_CHOICES: list[tuple[str, int]] = [
    ("Original (keep size, align to ×8)", 0),
    *[
        (bucket_label(value, recommended=(value == DEFAULT_UPLOAD_BUCKET)), value)
        for value in RESOLUTION_BUCKETS
    ],
]


def _align(value: float) -> int:
    return max(64, int(round(value / 8.0)) * 8)


def dimensions_from_generation_preset(size: int | str, ratio: str) -> tuple[int, int]:
    """Return SD-friendly dimensions from a long-side size and aspect ratio."""
    try:
        long_side = int(size)
    except (TypeError, ValueError):
        long_side = 512
    long_side = max(64, min(2048, _align(long_side)))

    try:
        raw_w, raw_h = (int(part) for part in str(ratio or "1:1").split(":", 1))
    except (TypeError, ValueError):
        raw_w, raw_h = 1, 1
    if raw_w <= 0 or raw_h <= 0:
        raw_w, raw_h = 1, 1

    if raw_w >= raw_h:
        width = long_side
        height = _align(long_side * raw_h / raw_w)
    else:
        width = _align(long_side * raw_w / raw_h)
        height = long_side
    return width, height


def resize_to_bucket(image: Image.Image, bucket: int) -> tuple[Image.Image, str]:
    """Scale so the longest side fits ``bucket``; dimensions stay multiples of 8."""
    rgb = image.convert("RGB")
    if bucket <= 0:
        w, h = align_to_multiple_of_8(rgb.width, rgb.height)
        if (w, h) != rgb.size:
            rgb = rgb.resize((w, h), Image.Resampling.LANCZOS)
        return rgb, f"{w}×{h} (original, aligned)"

    safe_bucket = min(int(bucket), MAX_BUCKET_8GB) if bucket > MAX_BUCKET_8GB else int(bucket)
    width, height = rgb.size
    long_side = max(width, height)
    if long_side <= safe_bucket:
        w, h = align_to_multiple_of_8(width, height)
        if (w, h) != rgb.size:
            rgb = rgb.resize((w, h), Image.Resampling.LANCZOS)
        return rgb, f"{w}×{h} (already under {safe_bucket}px)"

    scale = safe_bucket / long_side
    new_w = max(8, int(width * scale) // 8 * 8)
    new_h = max(8, int(height * scale) // 8 * 8)
    resized = rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return resized, f"{width}×{height} → {new_w}×{new_h} (max side {safe_bucket}px)"
