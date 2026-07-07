from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import urllib.parse
from pathlib import Path

import launch
from aiwf.runtime.bootstrap_env import apply_from_argv

PROFILE_PATH = launch.ROOT / "_local" / "backend_profile.json"
VALID_BACKENDS = {"diffusers", "sdcpp", "onnx"}
LAUNCH_PROFILE_ONLY_FLAGS = {"--terminal", "--set-default", "--skip-frontend-build"}


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _falsey(value: str | None) -> bool:
    return (value or "").strip().lower() in {"0", "false", "no", "off"}


def _extract_port(argv: list[str], default: int = 7860) -> int:
    for index, item in enumerate(argv):
        if item == "--port" and index + 1 < len(argv):
            try:
                return int(argv[index + 1])
            except ValueError:
                return default
        if item.startswith("--port="):
            try:
                return int(item.split("=", 1)[1])
            except ValueError:
                return default
    return default


def _browser_app_command(url: str, *, profile_dir: Path) -> list[str] | None:
    candidates = ["chrome", "chrome.exe", "msedge", "msedge.exe"]
    for name in candidates:
        executable = shutil.which(name)
        if executable:
            return [
                executable,
                f"--user-data-dir={profile_dir}",
                "--new-window",
                "--start-maximized",
                f"--app={url}",
            ]

    known_paths = [
        Path(os.environ.get("ProgramFiles", "")) / "Google" / "Chrome" / "Application" / "chrome.exe",
        Path(os.environ.get("ProgramFiles(x86)", "")) / "Google" / "Chrome" / "Application" / "chrome.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "Chrome" / "Application" / "chrome.exe",
        Path(os.environ.get("ProgramFiles", "")) / "Microsoft" / "Edge" / "Application" / "msedge.exe",
        Path(os.environ.get("ProgramFiles(x86)", "")) / "Microsoft" / "Edge" / "Application" / "msedge.exe",
    ]
    for path in known_paths:
        if path.is_file():
            return [
                str(path),
                f"--user-data-dir={profile_dir}",
                "--new-window",
                "--start-maximized",
                f"--app={url}",
            ]
    return None


def _open_loading_window(port: int) -> subprocess.Popen | None:
    if _falsey(os.environ.get("AIWF_PRO_LOADING_WINDOW")):
        return None
    loading_file = launch.ROOT / "static" / "pro_startup_loading.html"
    if not loading_file.is_file():
        return None
    target_url = f"http://127.0.0.1:{port}"
    ready_url = f"{target_url}/api/pro/startup"
    loading_url = (
        loading_file.resolve().as_uri()
        + "?"
        + urllib.parse.urlencode({"target": target_url, "ready": ready_url})
    )
    profile_dir = launch.ROOT / "_local" / "pro-loading-browser-profile"
    profile_dir.mkdir(parents=True, exist_ok=True)
    command = _browser_app_command(loading_url, profile_dir=profile_dir)
    if command is None:
        return None
    kwargs: dict[str, object] = {
        "cwd": str(launch.ROOT),
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        return subprocess.Popen(command, **kwargs)
    except OSError:
        return None


def _normalize_backend(value: str | None) -> str:
    normalized = (value or "").strip().lower().replace("_", "-")
    aliases = {
        "stable-diffusion.cpp": "sdcpp",
        "stable-diffusion-cpp": "sdcpp",
        "sd-cpp": "sdcpp",
        "sdcpp": "sdcpp",
        "diffusers": "diffusers",
        "onnx": "onnx",
    }
    return aliases.get(normalized, "")


def _read_default_backend() -> str:
    try:
        data = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return "diffusers"
    backend = _normalize_backend(str(data.get("backend") or "")) if isinstance(data, dict) else ""
    return backend if backend in VALID_BACKENDS else "diffusers"


def _write_default_backend(backend: str) -> None:
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text(json.dumps({"backend": backend}, indent=2), encoding="utf-8")


def _hidden_launch_stdio() -> tuple[object | None, object | None]:
    log_dir = launch.ROOT / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout = (log_dir / "pro-hidden-launch.out.log").open("a", encoding="utf-8")
        stderr = (log_dir / "pro-hidden-launch.err.log").open("a", encoding="utf-8")
        return stdout, stderr
    except OSError:
        return subprocess.DEVNULL, subprocess.DEVNULL


def _extract_option(argv: list[str], name: str) -> tuple[str | None, list[str]]:
    cleaned: list[str] = []
    value: str | None = None
    skip_next = False
    for index, item in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if item == name:
            if index + 1 < len(argv):
                value = argv[index + 1]
                skip_next = True
            continue
        if item.startswith(f"{name}="):
            value = item.split("=", 1)[1]
            continue
        cleaned.append(item)
    return value, cleaned


def _strip_profile_only_args(argv: list[str]) -> list[str]:
    cleaned: list[str] = []
    skip_next = False
    for index, item in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if item in LAUNCH_PROFILE_ONLY_FLAGS:
            continue
        if item == "--backend":
            skip_next = True
            continue
        if item.startswith("--backend="):
            continue
        cleaned.append(item)
    return cleaned


def _ensure_frontend(passthrough: list[str]) -> None:
    if "--skip-frontend-build" in passthrough or _truthy(os.environ.get("AIWF_SKIP_PRO_FRONTEND_UPDATE")):
        return
    script = launch.ROOT / "scripts" / "ensure_pro_frontend.py"
    if not script.is_file():
        return
    command = [launch.python(), str(script)]
    if "--terminal" not in sys.argv[1:]:
        command.append("--quiet")
    try:
        subprocess.check_call(command, cwd=str(launch.ROOT))
    except subprocess.CalledProcessError as exc:
        # Do not hide a missing frontend, but do not brick an already-built app for a failed rebuild.
        dist_index = launch.ROOT / "frontend" / "dist" / "index.html"
        if not dist_index.is_file():
            raise
        print(f"[AIWF] Frontend update check failed with {exc.returncode}; using existing frontend build.", flush=True)


def main() -> None:
    argv = sys.argv[1:]
    show_terminal = "--terminal" in argv
    set_default = "--set-default" in argv

    backend_arg, with_backend_removed = _extract_option(argv, "--backend")
    backend = _normalize_backend(backend_arg) or _read_default_backend()
    if backend not in VALID_BACKENDS:
        raise SystemExit(f"Unsupported backend profile: {backend_arg!r}")
    if set_default:
        _write_default_backend(backend)

    passthrough = _strip_profile_only_args(with_backend_removed)
    loading_window = None
    if not show_terminal and "--no-autolaunch" not in passthrough:
        loading_window = _open_loading_window(_extract_port(passthrough))
        if loading_window is not None:
            passthrough.append("--no-autolaunch")

    sys.path.insert(0, str(launch.ROOT))
    apply_from_argv(passthrough)
    os.environ.setdefault("XFORMERS_FORCE_DISABLE_TRITON", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    skip_prepare = "--skip-prepare-environment" in passthrough
    skip_install = "--skip-install" in passthrough
    launch.prepare(skip_prepare, skip_install, passthrough)
    _ensure_frontend(argv)
    pro_argv = launch.strip_launch_only_args(passthrough)

    env = os.environ.copy()
    env.setdefault("XFORMERS_FORCE_DISABLE_TRITON", "1")
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    env["AIWF_PROFILE_BACKEND"] = backend
    env["PYTHONPATH"] = (
        str(launch.ROOT)
        if "PYTHONPATH" not in env
        else str(launch.ROOT) + os.pathsep + env["PYTHONPATH"]
    )

    command = [launch.python(), str(launch.ROOT / "webui_backend_profile.py"), "--backend", backend, *pro_argv]
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_CONSOLE if show_terminal else subprocess.CREATE_NO_WINDOW
        if show_terminal:
            print(f"[AIWF Pro] Opening backend terminal with {backend} profile.")
            proc = subprocess.Popen(command, cwd=str(launch.ROOT), env=env, creationflags=creationflags)
        else:
            stdout, stderr = _hidden_launch_stdio()
            proc = subprocess.Popen(
                command,
                cwd=str(launch.ROOT),
                env=env,
                creationflags=creationflags,
                stdout=stdout,
                stderr=stderr,
            )
        raise SystemExit(proc.wait())

    os.execvpe(launch.python(), command, env)


if __name__ == "__main__":
    main()
