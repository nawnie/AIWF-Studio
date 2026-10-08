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


def start_ltx_engine_install(repo_root: Path | str | None = None) -> dict[str, Any]:
    """Start the fixed LTX 2.3 bootstrap script and capture its output."""
    root = Path(repo_root).resolve() if repo_root is not None else WorkerTenantRegistry().repo_root
    script = (root / "scripts" / "bootstrap_ltx.ps1").resolve()
    if not script.is_file() or script.parent != (root / "scripts").resolve():
        raise FileNotFoundError(f"LTX bootstrap script is unavailable at {script}")

    root_key = os.path.normcase(str(root))
    with _INSTALL_LOCK:
        active = _ACTIVE_INSTALLS.get(root_key)
        if active is not None and active[0].poll() is None:
            return {
                "status": "already_running",
                "pid": active[0].pid,
                "logPath": str(active[1]),
                "message": "LTX engine setup is already running.",
            }

        log_dir = root / "outputs" / "engine-installs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"ltx-bootstrap-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}.log"
        command = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-Enable",
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
            "message": "LTX engine setup started. Wait for it to finish, then retry video route readiness.",
        }


def ltx_engine_install_status(repo_root: Path | str | None = None) -> dict[str, Any]:
    root = Path(repo_root).resolve() if repo_root is not None else WorkerTenantRegistry().repo_root
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
