#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import venv
from pathlib import Path


def _python_path(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def main() -> int:
    parser = argparse.ArgumentParser(description="Install or repair the isolated AIWF Audio Lab engine.")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--upgrade", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    root = Path(args.repo).expanduser().resolve()
    engine_dir = root / "engines" / "audio_lab"
    venv_dir = engine_dir / ".venv"
    requirements = engine_dir / "requirements.txt"
    runner = engine_dir / "runner.py"
    if not requirements.is_file() or not runner.is_file():
        raise SystemExit("Audio Lab engine files are missing. Extract the Studio update into the repository root again.")

    if not _python_path(venv_dir).is_file():
        print(f"Creating isolated environment: {venv_dir}", flush=True)
        venv.EnvBuilder(with_pip=True, clear=False).create(venv_dir)

    python = _python_path(venv_dir)
    subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], check=True)
    command = [str(python), "-m", "pip", "install", "-r", str(requirements)]
    if args.upgrade:
        command.append("--upgrade")
    subprocess.run(command, check=True)
    check = subprocess.run(
        [str(python), str(runner), "self-test"], capture_output=True, text=True, check=False
    )
    if check.returncode != 0:
        raise SystemExit(check.stderr or check.stdout or "Audio Lab self-test failed")
    payload = {
        "installed": True,
        "python": str(python),
        "venv": str(venv_dir),
        "self_test": json.loads(check.stdout),
    }
    if args.json:
        print(json.dumps(payload))
    else:
        print("Audio Lab engine is ready.")
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
