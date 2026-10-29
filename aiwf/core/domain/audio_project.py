from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from aiwf.core.domain.audio import AudioGenerationOptions


class AudioProjectTrack(BaseModel):
    """One copied audio asset and the generation details needed to identify it."""

    model_config = ConfigDict(extra="forbid")

    track_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    asset_ref: str
    source_name: str = Field(min_length=1, max_length=255)
    prompt: str = ""
    model_id: str = ""
    kind: str = "music"
    duration_seconds: float = Field(default=0.0, ge=0.0, le=3600.0)
    sample_rate: int = Field(default=0, ge=0, le=384000)
    license_notice: str | None = None
    license: dict[str, Any] | None = None
    consent_status: str | None = None

    @field_validator("asset_ref")
    @classmethod
    def validate_asset_ref(cls, value: str) -> str:
        if "\\" in value or "\x00" in value:
            raise ValueError("Audio asset references must use safe project-relative paths.")
        path = PurePosixPath(value)
        if path.is_absolute() or len(path.parts) != 2 or path.parts[0] != "assets":
            raise ValueError("Audio asset references must point to one file under assets/.")
        if path.parts[1] in {"", ".", ".."}:
            raise ValueError("Audio asset references cannot traverse project paths.")
        return value


class AudioProjectManifest(BaseModel):
    """Versioned local project data. Asset bytes are stored beside this manifest."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    project_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    name: str = Field(min_length=1, max_length=100)
    created_at: str
    updated_at: str
    options: AudioGenerationOptions = Field(default_factory=AudioGenerationOptions)
    track: AudioProjectTrack | None = None


def validate_project_id(project_id: str) -> str:
    """Return a canonical project ID or reject paths, aliases, and traversal."""
    normalized = str(project_id or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{32}", normalized) is None:
        raise ValueError("Audio project ID must be a 32-character hexadecimal ID.")
    return normalized
