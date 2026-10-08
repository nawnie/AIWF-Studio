from __future__ import annotations

import os
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from aiwf.services.worker_tenant import WorkerTenantRegistry


_INSTALL_LOCK = threading.Lock()
_ACTIVE_INSTALLS: dict[str, tuple[subprocess.Popen[Any], Path]] = {}


def _paths(repo_root: Path | str | None = None) -> tuple[Path, Path]:
    root = Path(repo_root).resolve() if repo_root is not None else WorkerTenantRegistry().repo_root
    script = (root / "scripts" / "bootstrap_qwen_nunchaku.ps1").resolve()
    if script.parent != (root / "scripts").resolve() or not script.is_file():
        raise FileNotFoundError(f"Qwen Nunchaku bootstrap is unavailable at {script}")
    return root, script


def start_qwen_nunchaku_engine_install(
    repo_root: Path | str | None = None,
    data_root: Path | str | None = None,
) -> dict[str, Any]:
    """Start the fixed, isolated Qwen Nunchaku bootstrap and capture its output."""
    root, script = _paths(repo_root)
    runtime_root = Path(data_root).resolve() if data_root is not None else root
    root_key = os.path.normcase(str(runtime_root))
    with _INSTALL_LOCK:
        active = _ACTIVE_INSTALLS.get(root_key)
        if active is not None and active[0].poll() is None:
            return {
                "status": "already_running",
                "pid": active[0].pid,
                "logPath": str(active[1]),
                "message": "Qwen Nunchaku runtime setup is already running.",
            }

        log_dir = root / "outputs" / "engine-installs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"qwen-nunchaku-bootstrap-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}.log"
        command = [
            "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", str(script), "-DataRoot", str(runtime_root),
        ]
        output = log_path.open("x", encoding="utf-8", errors="replace")
        popen_kwargs: dict[str, Any] = {
            "cwd": str(root),
            "stdout": output,
            "stderr": subprocess.STDOUT,
            "stdin": subprocess.DEVNULL,
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            process = subprocess.Popen(command, **popen_kwargs)
        except Exception:
            output.close()
            log_path.unlink(missing_ok=True)
            raise
        output.close()
        _ACTIVE_INSTALLS[root_key] = (process, log_path)
        return {
            "status": "started",
            "pid": process.pid,
            "logPath": str(log_path),
            "message": "Qwen Nunchaku isolated runtime setup started. Model generation stays blocked until route readiness is verified.",
        }


def qwen_nunchaku_engine_install_status(data_root: Path | str | None = None) -> dict[str, Any]:
    root = Path(data_root).resolve() if data_root is not None else WorkerTenantRegistry().repo_root
    with _INSTALL_LOCK:
        active = _ACTIVE_INSTALLS.get(os.path.normcase(str(root)))
        if active is None:
            return {"status": "idle", "running": False, "logPath": ""}
        process, log_path = active
        exit_code = process.poll()
        return {
            "status": "running" if exit_code is None else "finished",
            "running": exit_code is None,
            "exitCode": exit_code,
            "pid": process.pid,
            "logPath": str(log_path),
        }
