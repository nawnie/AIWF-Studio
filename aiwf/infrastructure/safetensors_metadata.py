from __future__ import annotations

import json
import logging
import struct
from pathlib import Path

logger = logging.getLogger(__name__)


def safetensors_file_is_structurally_valid(path: Path | str) -> bool:
    """Check header framing, tensor byte ranges, and total payload size only."""
    resolved = Path(path)
    try:
        size = resolved.stat().st_size
        with resolved.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                return False
            header_size = struct.unpack("<Q", prefix)[0]
            if header_size < 2 or header_size > min(size - 8, 100 * 1024 * 1024):
                return False
            header = json.loads(stream.read(header_size).decode("utf-8"))
        if not isinstance(header, dict):
            return False
        dtype_sizes = {
            "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
            "F8_E4M3FN": 1, "F8_E4M3FNUZ": 1, "F8_E5M2FNUZ": 1,
            "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
            "U32": 4, "I32": 4, "F32": 4,
            "U64": 8, "I64": 8, "F64": 8,
        }
        data_size = size - 8 - header_size
        tensor_ranges = []
        for key, tensor in header.items():
            if key == "__metadata__":
                continue
            dtype = tensor.get("dtype") if isinstance(tensor, dict) else None
            if not isinstance(dtype, str) or dtype not in dtype_sizes:
                return False
            shape, offsets = tensor.get("shape"), tensor.get("data_offsets")
            if not isinstance(shape, list) or not all(isinstance(dim, int) and dim >= 0 for dim in shape):
                return False
            if not isinstance(offsets, list) or len(offsets) != 2 or not all(isinstance(pos, int) for pos in offsets):
                return False
            numel = 1
            for dim in shape:
                numel *= dim
            if offsets[0] < 0 or offsets[1] < offsets[0] or offsets[1] > data_size:
                return False
            if offsets[1] - offsets[0] != numel * dtype_sizes[dtype]:
                return False
            tensor_ranges.append((offsets[0], offsets[1]))
        tensor_ranges.sort()
        cursor = 0
        for start, end in tensor_ranges:
            if start != cursor:
                return False
            cursor = end
        return cursor == data_size
    except (OSError, UnicodeDecodeError, ValueError, struct.error):
        return False

CHECKPOINT_HEADER_KEYS = (
    "ss_sd_model_name",
    "ss_base_model_version",
    "ss_resolution",
    "ss_num_train_images",
    "ss_training_finished_at",
    "ss_output_name",
    "modelspec.title",
    "modelspec.architecture",
    "modelspec.implementation",
)

LORA_HEADER_KEYS = (
    "ss_output_name",
    "ss_sd_model_name",
    "ss_base_model_version",
    "ss_resolution",
    "ss_tag_frequency",
    "ss_num_train_images",
    "modelspec.title",
    "modelspec.architecture",
    "modelspec.implementation",
    "modelspec.trigger_words",
)


def read_safetensors_metadata(path: Path | str) -> dict[str, str]:
    """Read header metadata from a safetensors file without loading weights."""
    resolved = Path(path)
    if resolved.suffix.lower() != ".safetensors" or not resolved.is_file():
        return {}
    try:
        from safetensors import safe_open

        with safe_open(str(resolved), framework="pt") as handle:
            raw = handle.metadata()
            return {str(key): str(value) for key, value in (raw or {}).items()}
    except Exception:
        logger.debug("Could not read safetensors metadata for %s", resolved, exc_info=True)
        return {}


def file_size_label(path: Path | str) -> str:
    resolved = Path(path)
    try:
        size = resolved.stat().st_size
    except OSError:
        return "unknown"
    if size >= 1024**3:
        return f"{size / 1024**3:.2f} GB"
    if size >= 1024**2:
        return f"{size / 1024**2:.1f} MB"
    return f"{size / 1024:.0f} KB"


def suggest_lora_keywords(metadata: dict[str, str], *, limit: int = 8) -> str:
    """Best-effort trigger words from LoRA header metadata."""
    trigger = metadata.get("modelspec.trigger_words", "").strip()
    if trigger:
        return trigger

    raw = metadata.get("ss_tag_frequency", "").strip()
    if not raw:
        return ""

    try:
        frequencies = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return ""

    if not isinstance(frequencies, dict):
        return ""

    flattened: dict[str, float] = {}
    for tag, count in frequencies.items():
        if isinstance(count, dict):
            for nested_tag, nested_count in count.items():
                try:
                    flattened[str(nested_tag)] = flattened.get(str(nested_tag), 0.0) + float(nested_count)
                except (TypeError, ValueError):
                    continue
            continue
        try:
            flattened[str(tag)] = flattened.get(str(tag), 0.0) + float(count)
        except (TypeError, ValueError):
            continue

    if not flattened:
        return ""

    ranked = sorted(flattened.items(), key=lambda item: (-item[1], item[0]))
    tags = [str(tag) for tag, _count in ranked[:limit]]
    return ", ".join(tags)


def format_metadata_block(metadata: dict[str, str], keys: tuple[str, ...]) -> list[str]:
    lines: list[str] = []
    for key in keys:
        value = metadata.get(key, "").strip()
        if value:
            label = key.replace("ss_", "").replace("modelspec.", "").replace("_", " ").title()
            lines.append(f"- **{label}:** {value}")
    return lines
