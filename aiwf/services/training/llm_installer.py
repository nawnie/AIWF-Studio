"""Configure the optional AI bot trainer engine."""
from __future__ import annotations

import json
from pathlib import Path


class LLMTrainerInstallError(RuntimeError):
    """Raised when the LLM trainer engine cannot be configured."""


def install_llm_trainer_addon(repo_root: Path | str | None = None) -> list[str]:
    """Enable the built-in LLM trainer engine in engines.json.

    This does not start training and does not install packages immediately.
    The next launch.py startup prepares engines/llm/.venv from
    engines/llm/requirements.txt when the engine is enabled.
    """
    root = Path(repo_root).resolve() if repo_root is not None else _repo_root()
    worker = root / "engines" / "llm" / "worker.py"
    requirements = root / "engines" / "llm" / "requirements.txt"
    if not worker.exists():
        raise LLMTrainerInstallError(f"LLM trainer worker missing: {worker}")
    if not requirements.exists():
        raise LLMTrainerInstallError(f"LLM trainer requirements missing: {requirements}")

    path = root / "engines.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        raw = {}
    if not isinstance(raw, dict):
        raw = {}

    raw["llm"] = {
        "enabled": True,
        "venv_dir": "engines/llm/.venv",
        "_comment": "Built-in AI bot trainer. Supports LoRA, QLoRA, and full fine-tune via TRL.",
    }
    path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    return [
        f"Configured LLM trainer in {path}",
        "Restart AIWF Studio through launch.py so engines/llm/.venv can be prepared.",
    ]


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "launch.py").is_file():
            return parent
    return here.parents[3]
