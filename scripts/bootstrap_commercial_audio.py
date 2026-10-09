#!/usr/bin/env python3
"""Install one of AIWF Studio's commercially licensed audio engines.

    python scripts/bootstrap_commercial_audio.py --engine acestep   [--models-dir D:\\Models] [--json]
    python scripts/bootstrap_commercial_audio.py --engine moss-sfx  [--models-dir D:\\Models] [--json]

Each engine gets:
  1. its upstream code, cloned at a pinned commit into engines/<engine>/ (not committed to git);
  2. its own isolated Python environment built with uv (Studio's environment is untouched);
  3. its weights, downloaded at a pinned revision into <models dir>/audio/<name> so they live in
     Studio's model library with everything else (existing complete files are reused);
  4. a self-test run by AIWF's worker in that environment.
The last line printed is a JSON receipt. Studio runs this only when the person presses the
engine's Install button; nothing here runs at Studio start.

Licences (aiwf/services/audio_licenses.py): ACE-Step 1.5 code and weights MIT; MOSS-TTS code and
MOSS-SoundEffect v2.0 weights Apache-2.0.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ---- what each engine is, pinned -------------------------------------------------------------------
ENGINES = {
    "acestep": {
        "repo_url": "https://github.com/ace-step/ACE-Step-1.5.git",
        "commit": "ca1e85fe9430179831e6bc6be790c332190a3866",
        "clone": "engines/acestep/ACE-Step-1.5",
        "worker": "engines/acestep/aiwf_worker.py",
        "weights_repo": "ACE-Step/Ace-Step1.5",
        "weights_revision": "19671f406d603126926c1b7e2adc169acbcade22",
        "weights_folder": "ACE-Step-1.5",
        "sparse": None,
    },
    "moss-sfx": {
        "repo_url": "https://github.com/OpenMOSS/MOSS-TTS.git",
        "commit": "934d6826b084c46a0d033402174d5f8ac4ed2519",
        "clone": "engines/moss_sfx/MOSS-TTS",
        "worker": "engines/moss_sfx/aiwf_worker.py",
        "weights_repo": "OpenMOSS-Team/MOSS-SoundEffect-v2.0",
        "weights_revision": "e35df4d82fbe87fcd5d14e5d100e349c0c3c076d",
        "weights_folder": "MOSS-SoundEffect-v2.0",
        "sparse": ["moss_soundeffect_v2"],
    },
}
CU128_INDEX = "https://download.pytorch.org/whl/cu128"


def _run(command: list[str], *, cwd: Path | None = None, env: dict | None = None) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=str(cwd) if cwd else None, env=env, check=True)


def _python_in(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


# ---- 1. code at the pinned commit -------------------------------------------------------------------
def ensure_code(root: Path, spec: dict) -> Path:
    clone = root / spec["clone"]
    if not (clone / ".git").is_dir():
        clone.parent.mkdir(parents=True, exist_ok=True)
        if spec["sparse"]:
            _run(["git", "clone", "--quiet", "--filter=blob:limit=5m", "--no-checkout", spec["repo_url"], str(clone)])
            _run(["git", "-C", str(clone), "sparse-checkout", "set", *spec["sparse"]])
        else:
            _run(["git", "clone", "--quiet", spec["repo_url"], str(clone)])
    current = subprocess.run(["git", "-C", str(clone), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if current != spec["commit"]:
        _run(["git", "-C", str(clone), "fetch", "--quiet", "origin", spec["commit"]])
        _run(["git", "-C", str(clone), "checkout", "--quiet", spec["commit"]])
    return clone


# ---- 2. the isolated environment ----------------------------------------------------------------------
def ensure_environment(engine: str, clone: Path) -> Path:
    uv = shutil.which("uv")
    if not uv:
        raise SystemExit("uv is required to build the audio engine environment (https://docs.astral.sh/uv/).")
    if engine == "acestep":
        # ACE-Step ships a uv.lock: sync exactly that set (Windows gets torch 2.7.1+cu128)
        _run([uv, "sync", "--frozen", "--no-dev"], cwd=clone)
        return _python_in(clone / ".venv")
    venv = clone.parent / ".venv"
    if not _python_in(venv).is_file():
        _run([uv, "venv", "--python", "3.12", str(venv)])
    package = clone / "moss_soundeffect_v2"
    _run([uv, "pip", "install", "--python", str(_python_in(venv)), "--extra-index-url", CU128_INDEX,
          "--index-strategy", "unsafe-best-match", "-e", f"{package}[torch-cu128]"])
    return _python_in(venv)


# ---- 3. weights in Studio's model library ----------------------------------------------------------------
def ensure_weights(spec: dict, models_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    destination = models_dir / "audio" / spec["weights_folder"]
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=spec["weights_repo"], revision=spec["weights_revision"], local_dir=str(destination))
    return destination


# ---- 4. self-test ---------------------------------------------------------------------------------------
def self_test(root: Path, python: Path, spec: dict, weights: Path) -> dict:
    result = subprocess.run([str(python), str(root / spec["worker"]), "self-test", str(weights)],
                            capture_output=True, text=True)
    lines = [line for line in result.stdout.splitlines() if line.strip().startswith("{")]
    payload = json.loads(lines[-1]) if lines else {"ok": False, "error": (result.stderr or result.stdout)[-2000:]}
    if not payload.get("ok"):
        raise SystemExit(f"Self-test failed: {json.dumps(payload)}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Install a commercially licensed AIWF audio engine.")
    parser.add_argument("--engine", required=True, choices=sorted(ENGINES))
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--models-dir", default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    root = Path(args.repo).resolve()
    models_dir = Path(args.models_dir).resolve() if args.models_dir else root / "models"
    spec = ENGINES[args.engine]
    clone = ensure_code(root, spec)
    python = ensure_environment(args.engine, clone)
    weights = ensure_weights(spec, models_dir)
    test = self_test(root, python, spec, weights)
    receipt = {"ok": True, "engine": args.engine, "code": str(clone), "commit": spec["commit"],
               "python": str(python), "weights": str(weights), "weights_revision": spec["weights_revision"],
               "self_test": test}
    print(json.dumps(receipt) if args.json else json.dumps(receipt, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
