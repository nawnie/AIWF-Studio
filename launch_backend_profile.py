from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import launch
from aiwf.runtime.bootstrap_env import apply_from_argv

PROFILE_PATH = launch.ROOT / "_local" / "backend_profile.json"
VALID_BACKENDS = {"diffusers", "sdcpp", "onnx"}


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


def main() -> None:
    argv = sys.argv[1:]
    show_terminal = "--terminal" in argv
    set_default = "--set-default" in argv
    argv = [item for item in argv if item not in {"--terminal", "--set-default"}]

    backend_arg, passthrough = _extract_option(argv, "--backend")
    backend = _normalize_backend(backend_arg) or _read_default_backend()
    if backend not in VALID_BACKENDS:
        raise SystemExit(f"Unsupported backend profile: {backend_arg!r}")
    if set_default:
        _write_default_backend(backend)

    sys.path.insert(0, str(launch.ROOT))
    apply_from_argv(passthrough)
    os.environ.setdefault("XFORMERS_FORCE_DISABLE_TRITON", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    skip_prepare = "--skip-prepare-environment" in passthrough
    skip_install = "--skip-install" in passthrough
    launch.prepare(skip_prepare, skip_install, passthrough)
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
        raise SystemExit(proc.wait())

    os.execvpe(launch.python(), command, env)


if __name__ == "__main__":
    main()
