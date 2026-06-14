from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

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

    ranked = sorted(frequencies.items(), key=lambda item: (-float(item[1]), str(item[0])))
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