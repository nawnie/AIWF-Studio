from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aiwf import __version__

_PROMPT_KEYS = {
    "prompt",
    "negative_prompt",
    "style_prompt_template",
    "style_negative_template",
    "prompt_file",
    "use_prompt_file",
    "prompt_seed",
}


class GenerationLogService:
    """Append-only local generation telemetry.

    Genlog is intentionally opt-in. Entries should be useful for timing and
    settings comparisons without storing prompt text.
    """

    schema_version = 1

    def __init__(self, output_dir: Path | str, *, enabled: bool = False, path: Path | str | None = None) -> None:
        self.enabled = bool(enabled)
        self.path = Path(path) if path is not None else Path(output_dir) / "genlog" / "generation-log.jsonl"

    def append(self, entry: dict[str, Any]) -> Path | None:
        if not self.enabled:
            return None
        payload = {
            "schema_version": self.schema_version,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "app_version": __version__,
            **entry,
        }
        safe_payload = _scrub_prompt_fields(payload)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(safe_payload, sort_keys=True, default=str) + "\n")
        return self.path


def _scrub_prompt_fields(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.lower() in _PROMPT_KEYS:
                continue
            cleaned[key_text] = _scrub_prompt_fields(item)
        return cleaned
    if isinstance(value, (list, tuple)):
        return [_scrub_prompt_fields(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
