from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingEngineStatus:
    name: str
    label: str
    enabled: bool
    repo_dir: Path
    venv_dir: Path
    python_exe: Path
    worker_script: Path
    entry_script: Path | None
    ready: bool
    messages: tuple[str, ...]
    uses_studio_venv: bool = False

    def markdown_line(self) -> str:
        mark = "OK" if self.ready else "Missing"
        state = "enabled" if self.enabled else "disabled"
        if self.uses_studio_venv:
            state = f"{state}, Studio runtime"
        details = "; ".join(self.messages) if self.messages else "ready"
        return f"- **{self.label}:** {mark} ({state}) - {details}"


_ENGINE_DEFAULTS = {
    "kohya": {
        "label": "Kohya LoRA",
        "repo_dir": "engines/kohya/kohya_ss",
        "venv_dir": "engines/kohya/.venv",
        "worker_script": "engines/kohya/worker.py",
        "entry_script": "",
    },
    "ed2": {
        "label": "ED2 Full Fine-tune",
        "repo_dir": "training/EveryDream2trainer",
        "venv_dir": "engines/ed2/.venv",
        "worker_script": "engines/ed2/worker.py",
        "entry_script": "train.py",
    },
}

FULL_TUNE_ENGINE = "ed2"


def training_engine_statuses(repo_root: Path | str | None = None) -> dict[str, TrainingEngineStatus]:
    """Return training engine readiness without importing launch or engine deps."""
    root = Path(repo_root).resolve() if repo_root is not None else _repo_root()
    config = _load_engines_config(root / "engines.json")
    if "engines" in config and isinstance(config["engines"], dict):
        config = config["engines"]
    return {
        name: _status_for_engine(root, name, defaults, config.get(name, {}))
        for name, defaults in _ENGINE_DEFAULTS.items()
    }


def training_status_markdown(statuses: dict[str, TrainingEngineStatus]) -> str:
    if not statuses:
        return "**Training engines:** no engine definitions found."
    lines = ["**Training engine setup**"]
    lines.extend(status.markdown_line() for status in statuses.values())
    lines.append("")
    lines.append("Training engines are optional. Install an add-on from the Training tab or enable one in `engines.json`, then restart through `launch.py` so the engine environment can be prepared.")
    return "\n".join(lines)


def ready_engine_choices(statuses: dict[str, TrainingEngineStatus]) -> list[str]:
    choices: list[str] = []
    if statuses.get("kohya") and statuses["kohya"].ready:
        choices.append("Kohya LoRA")
    if statuses.get("ed2") and statuses["ed2"].ready:
        choices.append("ED2 Full Fine-tune")
    return choices


def full_tune_engine_status(statuses: dict[str, TrainingEngineStatus] | None = None) -> TrainingEngineStatus:
    """Return the mandatory full fine-tune engine status.

    ED2 is the only supported full fine-tune path. This does not make ED2 a
    boot dependency; it only makes ED2 mandatory when starting a full tune.
    """
    statuses = statuses or training_engine_statuses()
    return statuses[FULL_TUNE_ENGINE]


def full_tune_ready(statuses: dict[str, TrainingEngineStatus] | None = None) -> bool:
    return full_tune_engine_status(statuses).ready


def full_tune_status_markdown(statuses: dict[str, TrainingEngineStatus] | None = None) -> str:
    status = full_tune_engine_status(statuses)
    if status.ready:
        return "**Full fine-tune:** OK - ED2 is ready."
    return (
        "**Full fine-tune requires ED2.**\n\n"
        + status.markdown_line()
        + "\n\nED2 remains optional at app startup, but a full tune must use ED2."
    )


def _status_for_engine(root: Path, name: str, defaults: dict[str, str], raw_config) -> TrainingEngineStatus:
    section = raw_config if isinstance(raw_config, dict) else {}
    enabled = bool(section.get("enabled", False))
    repo_dir = _resolve(root, section.get("repo_dir") or defaults["repo_dir"])
    venv_dir, uses_studio_venv = _resolve_venv_dir(root, section, defaults["venv_dir"])
    worker_script = _resolve(root, section.get("worker_script") or defaults["worker_script"])
    entry_script_raw = section.get("entry_script") or defaults.get("entry_script") or ""
    entry_script = (repo_dir / entry_script_raw).resolve() if entry_script_raw else None
    python_exe = _python_exe(venv_dir)

    messages: list[str] = []
    if not enabled:
        messages.append("set enabled=true in engines.json")
    if not repo_dir.exists():
        messages.append(f"repo folder missing: {repo_dir}")
    if not python_exe.exists():
        messages.append(f"engine runtime missing: {python_exe}")
    if not worker_script.exists():
        messages.append(f"AIWF worker missing: {worker_script}")
    if entry_script is not None and not entry_script.exists():
        messages.append(f"engine entry script missing: {entry_script}")
    ready = (
        enabled
        and repo_dir.exists()
        and python_exe.exists()
        and worker_script.exists()
        and (entry_script is None or entry_script.exists())
    )
    return TrainingEngineStatus(
        name=name,
        label=defaults["label"],
        enabled=enabled,
        repo_dir=repo_dir,
        venv_dir=venv_dir,
        python_exe=python_exe,
        worker_script=worker_script,
        entry_script=entry_script,
        ready=ready,
        messages=tuple(messages),
        uses_studio_venv=uses_studio_venv,
    )


def _load_engines_config(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(key): value for key, value in raw.items() if not str(key).startswith("_")}


def _resolve(root: Path, raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (root / path).resolve()


def _resolve_venv_dir(root: Path, section: dict, fallback: str) -> tuple[Path, bool]:
    if "venv_dir" in section:
        raw = section.get("venv_dir")
        if raw is None:
            return (root / "venv").resolve(), True
        raw_text = str(raw).strip().lower()
        if raw_text in {"", "studio", "shared", "main", "aiwf"}:
            return (root / "venv").resolve(), True
        return _resolve(root, str(raw)), False
    return _resolve(root, fallback), False


def _python_exe(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "launch.py").is_file() and (parent / "engines").is_dir():
            return parent
    return here.parents[3]
