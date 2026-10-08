"""Folder setup for the guided first run: where Studio saves images and where it finds models.

AIWF Studio for Windows (native/) runs a setup wizard on first launch and from Configure
Studio. The wizard reads and changes the same folder settings Pro's Settings page uses,
which live in launch.json next to the code (RuntimeFlags.data_dir):

    output_dir        where generated images are saved            (default <studio>/outputs)
    models_dir        the main model library                      (default <studio>/models)
    ckpt_dir          image checkpoints                           (default <models_dir>/Stable-diffusion)
    extra_model_dirs  more model libraries, one folder per line   (searched, never written)
    extra_ckpt_dirs   more checkpoint folders, one per line       (searched, never written)

Only these five keys are ever changed here. Every other launch.json value, including keys
this version does not know, is written back exactly as it was. New values take effect the
next time Pro or the engine API starts, the same rule launch.json always had.

The module is torch-free and imports only the launch-settings model, so the small engine API
(aiwf/engine_api.py) can serve it without loading the diffusion runtime.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from aiwf.core.config.launch import LaunchSettings, launch_settings_path

# --- the folder keys this module owns --------------------------------------------------
SINGLE_FOLDERS = ("output_dir", "models_dir", "ckpt_dir")
FOLDER_LISTS = ("extra_model_dirs", "extra_ckpt_dirs")
FOLDER_FIELDS = SINGLE_FOLDERS + FOLDER_LISTS

# what each folder means, in the words the wizard and agents show to a person
FOLDER_LABELS = {
    "output_dir": "Generated images are saved here.",
    "models_dir": "Main model library (Studio downloads and finds models here).",
    "ckpt_dir": "Image checkpoints (SD, SDXL, Flux and similar single-file models).",
    "extra_model_dirs": "More model libraries Studio searches but never writes to.",
    "extra_ckpt_dirs": "More checkpoint folders Studio searches but never writes to.",
}


class SetupError(ValueError):
    """A folder value that cannot be saved; code and message go back to the UI unchanged."""

    def __init__(self, code: str, message: str, field: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.field = field


# --- reading ------------------------------------------------------------------------------
def _read_raw(path: Path) -> dict[str, Any]:
    """launch.json as a plain dict, or {} when it is missing or unreadable (Pro then uses defaults)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _folder_facts(path: Path) -> dict[str, Any]:
    """Whether a folder exists, can be written, and how much space its drive has left."""
    facts: dict[str, Any] = {"path": str(path), "exists": path.is_dir(), "writable": False, "free_bytes": None}
    # the nearest existing parent decides the drive and whether a missing folder could be created
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        facts["free_bytes"] = shutil.disk_usage(probe).free
    except OSError:
        pass
    facts["writable"] = os.access(probe, os.W_OK)
    return facts


def _resolved(data_dir: Path, saved: LaunchSettings) -> dict[str, Any]:
    """The folders Studio would actually use with these saved values (same rules as RuntimeFlags)."""
    from aiwf.core.config.settings import RuntimeFlags

    flags = saved.to_runtime_flags(RuntimeFlags(data_dir=data_dir))
    return {
        "output_dir": flags.resolved_output_dir(),
        "models_dir": flags.resolved_models_dir(),
        "ckpt_dir": flags.resolved_ckpt_dir(),
        "extra_model_dirs": flags.resolved_extra_model_dirs(),
        "extra_ckpt_dirs": flags.resolved_extra_ckpt_dirs(),
    }


def describe_setup(data_dir: Path) -> dict[str, Any]:
    """Current folder settings with the effective folders, defaults and drive facts."""
    data_dir = Path(data_dir).resolve()
    path = launch_settings_path(data_dir)
    raw = _read_raw(path)
    try:
        saved = LaunchSettings.model_validate(raw)
    except ValueError:
        saved = LaunchSettings()
    effective = _resolved(data_dir, saved)
    defaults = _resolved(data_dir, LaunchSettings())

    # this loop describes each single folder: what is saved, what is used, and the default
    folders: dict[str, Any] = {}
    for field in SINGLE_FOLDERS:
        folders[field] = {
            "label": FOLDER_LABELS[field],
            "saved": str(getattr(saved, field) or ""),
            "default": str(defaults[field]),
            **_folder_facts(effective[field]),
        }
    # this loop describes the folder lists; effective lists can include auto-discovered libraries
    for field in FOLDER_LISTS:
        lines = [line.strip() for line in str(getattr(saved, field) or "").splitlines() if line.strip()]
        folders[field] = {
            "label": FOLDER_LABELS[field],
            "saved": lines,
            "effective": [_folder_facts(item) for item in effective[field]],
        }

    venv_python = data_dir / "venv" / "Scripts" / "python.exe"
    return {
        "schema_version": "1",
        "studio_root": str(data_dir),
        "launch_file": str(path),
        "launch_file_exists": path.is_file(),
        "folders": folders,
        "python": {"path": str(venv_python), "exists": venv_python.is_file()},
        "applies_on_restart": True,
        "note": "Folder changes apply the next time AIWF Studio Pro or the engine API starts.",
    }


# --- writing --------------------------------------------------------------------------------
def _clean_folder(field: str, value: str, *, create_missing: bool) -> str:
    """Validate one folder: absolute, a real directory (created when asked), not a file."""
    text = (value or "").strip().strip('"')
    if not text:
        return ""   # empty means "use the default"
    if any(ord(ch) < 32 for ch in text):
        raise SetupError("invalid_folder", "The folder name contains control characters.", field)
    path = Path(text)
    if not path.is_absolute():
        raise SetupError("relative_folder", f"Use a full folder path such as D:\\Models (got '{text}').", field)
    if path.exists() and not path.is_dir():
        raise SetupError("not_a_folder", f"{text} is a file, not a folder.", field)
    if not path.exists():
        if not create_missing:
            raise SetupError("folder_missing", f"{text} does not exist. Create it or choose another folder.", field)
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise SetupError("folder_not_created", f"Could not create {text}: {exc.strerror or exc}", field) from exc
    return str(path.resolve())


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write launch.json through a temporary file in the same folder, then swap it in."""
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    handle, temp_name = tempfile.mkstemp(prefix=".launch-", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def save_folders(data_dir: Path, changes: dict[str, Any], *, create_missing: bool = False) -> dict[str, Any]:
    """Change only the given folder keys in launch.json and return the new description.

    changes: any of the five folder keys. Single folders take a string ("" = default);
    folder lists take a list of strings. Unknown keys are refused, so a typo cannot
    silently do nothing.
    """
    data_dir = Path(data_dir).resolve()
    unknown = sorted(set(changes) - set(FOLDER_FIELDS))
    if unknown:
        raise SetupError("unknown_field", f"Only folder settings can be changed here, not: {', '.join(unknown)}.")

    path = launch_settings_path(data_dir)
    raw = _read_raw(path)
    if not raw:
        # no saved profile yet: start from Pro's defaults so the file is complete and valid
        raw = json.loads(LaunchSettings().model_dump_json())

    # this loop validates every requested change before anything is written
    updates: dict[str, str] = {}
    for field, value in changes.items():
        if field in SINGLE_FOLDERS:
            if not isinstance(value, str):
                raise SetupError("invalid_folder", "Expected one folder path.", field)
            updates[field] = _clean_folder(field, value, create_missing=create_missing)
        else:
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise SetupError("invalid_folder", "Expected a list of folder paths.", field)
            cleaned: list[str] = []
            for item in value:
                folder = _clean_folder(field, item, create_missing=create_missing)
                if folder and folder.casefold() not in {existing.casefold() for existing in cleaned}:
                    cleaned.append(folder)
            updates[field] = "\n".join(cleaned)

    merged = {**raw, **updates}
    try:
        LaunchSettings.model_validate(merged)
    except ValueError as exc:
        raise SetupError("invalid_launch_settings", f"launch.json would not be valid: {exc}") from exc
    _write_atomic(path, merged)

    result = describe_setup(data_dir)
    result["saved_fields"] = sorted(updates)
    return result
