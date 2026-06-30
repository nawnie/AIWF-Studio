from __future__ import annotations

import json
import struct
from pathlib import Path


def asset_size_bytes(path: Path | str) -> int:
    resolved = Path(path)
    try:
        if resolved.is_file():
            return resolved.stat().st_size
        if resolved.is_dir():
            return sum(item.stat().st_size for item in resolved.rglob("*") if item.is_file())
    except OSError:
        return 0
    return 0


def asset_file_count(path: Path | str) -> int:
    resolved = Path(path)
    try:
        if resolved.is_file():
            return 1
        if resolved.is_dir():
            return sum(1 for item in resolved.rglob("*") if item.is_file())
    except OSError:
        return 0
    return 0


def format_size(size_bytes: int) -> str:
    if size_bytes <= 0:
        return "unknown size"
    units = ("bytes", "KB", "MB", "GB", "TB")
    value = float(size_bytes)
    unit_index = 0
    while value >= 1024.0 and unit_index < len(units) - 1:
        value /= 1024.0
        unit_index += 1
    unit = units[unit_index]
    if unit == "bytes":
        return f"{int(value)} bytes"
    decimals = 2 if value < 10 else 1
    return f"{value:.{decimals}f} {unit}"


def asset_shape_label(path: Path | str, *, size_bytes: int | None = None, file_count: int | None = None) -> str:
    resolved = Path(path)
    count = asset_file_count(resolved) if file_count is None else max(0, int(file_count))
    size = asset_size_bytes(resolved) if size_bytes is None else max(0, int(size_bytes))
    size_label = format_size(size)
    if resolved.is_dir():
        noun = "file" if count == 1 else "files"
        return f"folder, {count} {noun}, {size_label}"
    if count <= 1:
        return f"1 file, {size_label}"
    noun = "file" if count == 1 else "files"
    return f"{count} {noun}, {size_label}"


def precision_from_text(*values: str | None) -> str | None:
    text = " ".join(value or "" for value in values).lower().replace("_", "-")
    compact = text.replace("-", "").replace(" ", "")
    if any(token in text for token in ("bf16", "bfloat16")) or "bfloat16" in compact:
        return "bf16"
    if any(token in text for token in ("fp16", "float16", "half")) or "float16" in compact:
        return "fp16"
    if any(token in text for token in ("fp32", "float32")) or "float32" in compact:
        return "fp32"
    if any(token in text for token in ("fp8", "float8")) or "float8" in compact:
        return "fp8"
    return None


def safetensors_precision(path: Path | str) -> str | None:
    resolved = Path(path)
    if resolved.suffix.lower() != ".safetensors" or not resolved.is_file():
        return None
    try:
        with resolved.open("rb") as handle:
            header_size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_size).decode("utf-8"))
    except Exception:
        return None
    dtype_counts: dict[str, int] = {}
    for key, value in header.items():
        if key == "__metadata__" or not isinstance(value, dict):
            continue
        dtype = str(value.get("dtype") or "").upper()
        if dtype:
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
    if not dtype_counts:
        return None
    dtype = max(dtype_counts.items(), key=lambda item: item[1])[0]
    return {
        "BF16": "bf16",
        "F16": "fp16",
        "F32": "fp32",
        "F8_E4M3": "fp8",
        "F8_E5M2": "fp8",
    }.get(dtype)
