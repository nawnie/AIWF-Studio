from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aiwf.core.domain.worker import WorkerCommand


_STUDIO_VENV_ALIASES = {"", "studio", "shared", "main", "aiwf", "core"}


@dataclass(frozen=True)
class WorkerTenantSpec:
    name: str
    label: str
    worker_script: str
    venv_dir: str = ""
    repo_dir: str = ""
    entry_script: str = ""
    enabled: bool = False
    timeout_seconds: int | None = None


@dataclass(frozen=True)
class WorkerTenantStatus:
    name: str
    label: str
    enabled: bool
    ready: bool
    repo_root: Path
    worker_script: Path
    venv_dir: Path
    python_exe: Path
    repo_dir: Path | None
    entry_script: Path | None
    uses_studio_venv: bool
    messages: tuple[str, ...]

    def markdown_line(self) -> str:
        mark = "OK" if self.ready else "Missing"
        state = "enabled" if self.enabled else "disabled"
        if self.uses_studio_venv:
            state = f"{state}, Studio runtime"
        details = "; ".join(self.messages) if self.messages else "ready"
        return f"- **{self.label}:** {mark} ({state}) - {details}"


DEFAULT_WORKER_TENANTS: dict[str, WorkerTenantSpec] = {
    "wan": WorkerTenantSpec(
        name="wan",
        label="Wan Video Engine",
        worker_script="engines/wan/worker.py",
        venv_dir="engines/wan/.venv",
        enabled=False,
        timeout_seconds=None,
    ),
    "ltx": WorkerTenantSpec(
        name="ltx",
        label="LTX 2.3 Video Engine",
        worker_script="engines/ltx/worker.py",
        repo_dir="engines/ltx/LTX-2",
        venv_dir="engines/ltx/.venv",
        enabled=False,
        timeout_seconds=None,
    ),
    "kohya": WorkerTenantSpec(
        name="kohya",
        label="Kohya LoRA Trainer",
        worker_script="engines/kohya/worker.py",
        repo_dir="engines/kohya/kohya_ss",
        venv_dir="engines/kohya/.venv",
        enabled=False,
        timeout_seconds=None,
    ),
    "ed2": WorkerTenantSpec(
        name="ed2",
        label="EveryDream2 Full Trainer",
        worker_script="engines/ed2/worker.py",
        repo_dir="engines/ed2/EveryDream2trainer",
        venv_dir="engines/ed2/.venv",
        entry_script="train.py",
        enabled=False,
        timeout_seconds=None,
    ),
    "llm": WorkerTenantSpec(
        name="llm",
        label="AI Bot Trainer",
        worker_script="engines/llm/worker.py",
        venv_dir="engines/llm/.venv",
        enabled=False,
        timeout_seconds=None,
    ),
}


def load_engines_config(repo_root: Path | str | None = None) -> dict[str, Any]:
    root = Path(repo_root).resolve() if repo_root is not None else repo_root_from_here()
    path = root / "engines.json"
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    if "engines" in raw and isinstance(raw["engines"], dict):
        raw = raw["engines"]
    return {str(key): value for key, value in raw.items() if not str(key).startswith("_")}


class WorkerTenantRegistry:
    """Resolve optional engine workers into launchable subprocess commands.

    Tenants keep heavy engines in their own repos/venvs when configured, while
    still allowing the Studio venv fallback for lightweight or transitional
    workers. Status is intentionally descriptive so Settings can explain what
    is missing without importing those engines.
    """

    def __init__(
        self,
        repo_root: Path | str | None = None,
        specs: dict[str, WorkerTenantSpec] | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve() if repo_root is not None else repo_root_from_here()
        self.specs = dict(specs or DEFAULT_WORKER_TENANTS)
        self.config = load_engines_config(self.repo_root)

    def names(self) -> list[str]:
        return list(self.specs)

    def status(self, name: str) -> WorkerTenantStatus:
        if name not in self.specs:
            raise KeyError(f"Unknown worker tenant: {name}")
        spec = self._merged_spec(name)
        worker_script = self._resolve(spec.worker_script)
        repo_dir = self._resolve(spec.repo_dir) if spec.repo_dir else None
        entry_script = (repo_dir / spec.entry_script).resolve() if repo_dir is not None and spec.entry_script else None
        venv_dir, uses_studio_venv = self._resolve_venv_dir(spec.venv_dir)
        python_exe = python_exe_for_venv(venv_dir)

        messages: list[str] = []
        if not spec.enabled:
            messages.append("set enabled=true in engines.json")
        if not worker_script.exists():
            messages.append(f"AIWF worker missing: {worker_script}")
        if not python_exe.exists():
            messages.append(f"engine runtime missing: {python_exe}")
        if repo_dir is not None and not repo_dir.exists():
            messages.append(f"repo folder missing: {repo_dir}")
        if entry_script is not None and not entry_script.exists():
            messages.append(f"engine entry script missing: {entry_script}")

        ready = spec.enabled and not messages
        return WorkerTenantStatus(
            name=spec.name,
            label=spec.label,
            enabled=spec.enabled,
            ready=ready,
            repo_root=self.repo_root,
            worker_script=worker_script,
            venv_dir=venv_dir,
            python_exe=python_exe,
            repo_dir=repo_dir,
            entry_script=entry_script,
            uses_studio_venv=uses_studio_venv,
            messages=tuple(messages),
        )

    def statuses(self) -> dict[str, WorkerTenantStatus]:
        return {name: self.status(name) for name in self.names()}

    def build_command(
        self,
        name: str,
        request_json: Path | str,
        *,
        extra_args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: Path | str | None = None,
    ) -> WorkerCommand:
        status = self.status(name)
        if not status.ready:
            details = "; ".join(status.messages) if status.messages else "not ready"
            raise RuntimeError(f"Worker tenant '{name}' is not ready: {details}")
        child_env = dict(env or {})
        child_env["AIWF_ROOT"] = str(self.repo_root)
        child_env["PYTHONPATH"] = _prepend_pythonpath(self.repo_root, child_env.get("PYTHONPATH"))
        args = [
            str(status.python_exe),
            str(status.worker_script),
            str(Path(request_json).resolve()),
            *(extra_args or []),
        ]
        return WorkerCommand(
            args=args,
            cwd=Path(cwd).resolve() if cwd is not None else (status.repo_dir or self.repo_root),
            env=child_env,
            name=name,
            timeout_seconds=self._merged_spec(name).timeout_seconds,
        )

    def _merged_spec(self, name: str) -> WorkerTenantSpec:
        base = self.specs[name]
        section = self.config.get(name, {})
        if not isinstance(section, dict):
            section = {}
        timeout_raw = section.get("timeout_seconds", base.timeout_seconds)
        timeout = int(timeout_raw) if timeout_raw not in (None, "") else None
        return WorkerTenantSpec(
            name=base.name,
            label=str(section.get("label") or base.label),
            worker_script=str(section.get("worker_script") or base.worker_script),
            repo_dir=str(section.get("repo_dir") or base.repo_dir),
            venv_dir=str(section.get("venv_dir") if "venv_dir" in section else base.venv_dir),
            entry_script=str(section.get("entry_script") or base.entry_script),
            enabled=bool(section.get("enabled", base.enabled)),
            timeout_seconds=timeout,
        )

    def _resolve(self, raw: str | Path) -> Path:
        path = Path(raw)
        return path.resolve() if path.is_absolute() else (self.repo_root / path).resolve()

    def _resolve_venv_dir(self, raw: str | Path) -> tuple[Path, bool]:
        raw_text = str(raw).strip()
        if raw_text.lower() in _STUDIO_VENV_ALIASES:
            return (self.repo_root / "venv").resolve(), True
        return self._resolve(raw_text), False


def python_exe_for_venv(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def repo_root_from_here() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "launch.py").is_file() and (parent / "engines").is_dir():
            return parent
    return here.parents[2]


def _prepend_pythonpath(repo_root: Path, value: str | None) -> str:
    root = str(repo_root)
    if not value:
        return root
    parts = value.split(os.pathsep)
    if root in parts:
        return value
    return root + os.pathsep + value
