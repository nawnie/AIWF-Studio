from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Callable, Sequence


ED2_REPO_URL = "https://github.com/victorchall/EveryDream2trainer.git"
ED2_ADDON_RELATIVE = Path("training") / "EveryDream2trainer"


class ED2InstallError(RuntimeError):
    """Raised when the ED2 add-on cannot be installed or configured."""


RunCommand = Callable[..., subprocess.CompletedProcess]


def install_ed2_addon(
    repo_root: Path | str | None = None,
    *,
    python_exe: str | Path | None = None,
    install_requirements: bool = True,
    run_command: RunCommand = subprocess.run,
) -> list[str]:
    """Install EveryDream2 as an optional AIWF training add-on.

    The add-on repo lives under ``training/EveryDream2trainer``. Dependencies
    are installed from AIWF's overlay requirements, not ED2's legacy upstream
    requirements file.
    """
    root = Path(repo_root).resolve() if repo_root is not None else _repo_root()
    target = root / ED2_ADDON_RELATIVE
    logs: list[str] = []

    if target.exists():
        if not (target / "train.py").exists():
            raise ED2InstallError(f"ED2 add-on folder exists but train.py is missing: {target}")
        logs.append(f"ED2 add-on already present at {target}")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        _run_checked(
            ["git", "clone", ED2_REPO_URL, str(target)],
            cwd=root,
            run_command=run_command,
        )
        logs.append(f"Cloned ED2 add-on into {target}")

    if install_requirements:
        requirements = root / "engines" / "ed2" / "requirements.txt"
        if not requirements.exists():
            raise ED2InstallError(f"ED2 overlay requirements missing: {requirements}")
        py = str(python_exe or sys.executable)
        _run_checked(
            [py, "-m", "pip", "install", "--disable-pip-version-check", "-r", str(requirements)],
            cwd=root,
            run_command=run_command,
        )
        logs.append("Installed ED2 overlay requirements into the active Studio runtime")

    _configure_engines_json(root, target)
    logs.append("Configured ED2 in engines.json using Studio runtime mode")
    return logs


def _run_checked(command: Sequence[str], *, cwd: Path, run_command: RunCommand) -> None:
    result = run_command(
        list(command),
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        output = (result.stdout or "").strip()
        raise ED2InstallError(f"Command failed: {' '.join(command)}\n{output}")


def _configure_engines_json(root: Path, target: Path) -> None:
    path = root / "engines.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        raw = {}
    if not isinstance(raw, dict):
        raw = {}

    raw["ed2"] = {
        "enabled": True,
        "repo_dir": str(target.relative_to(root)).replace("\\", "/"),
        "venv_dir": "studio",
        "_comment": "Installed as an optional Training tab add-on. Uses AIWF's ED2 overlay and skips upstream legacy requirements.",
    }
    path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "launch.py").is_file():
            return parent
    return here.parents[3]
