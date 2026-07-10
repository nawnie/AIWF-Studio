#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


MUSICGEN_REPO = "facebook/musicgen-small"
MUSICGEN_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer_config.json",
)


def _run(command: list[str], *, cwd: Path, label: str) -> None:
    print(f"[AIWF] {label}", flush=True)
    result = subprocess.run(command, cwd=str(cwd), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}.")


def _ensure_main_dependencies() -> None:
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError("PyTorch is missing from the Studio environment. Repair the main AIWF install first.")
    missing = [
        requirement
        for module, requirement in (
            ("transformers", "transformers>=4.31,<5"),
            ("scipy", "scipy>=1.11,<2"),
            ("huggingface_hub", "huggingface-hub>=0.27,<2"),
        )
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        _run(
            [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *missing],
            cwd=Path.cwd(),
            label="Installing minimum MusicGen dependencies",
        )


def _download_musicgen(root: Path) -> Path:
    from huggingface_hub import snapshot_download

    destination = root / "models" / "audio" / "MusicGen" / "musicgen-small"
    destination.mkdir(parents=True, exist_ok=True)
    print(f"[AIWF] Downloading {MUSICGEN_REPO} to {destination}", flush=True)
    snapshot_download(
        repo_id=MUSICGEN_REPO,
        local_dir=str(destination),
        allow_patterns=list(MUSICGEN_FILES),
    )
    missing = [name for name in MUSICGEN_FILES if not (destination / name).is_file()]
    if missing:
        raise RuntimeError(f"MusicGen download is incomplete. Missing: {', '.join(missing)}")
    return destination


def _install_mmaudio(root: Path) -> tuple[Path, Path]:
    powershell = shutil.which("pwsh") or shutil.which("powershell")
    if not powershell:
        raise RuntimeError("PowerShell is required to install the isolated MMAudio engine.")
    script = root / "scripts" / "bootstrap_mmaudio.ps1"
    _run(
        [powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
        cwd=root,
        label="Installing or repairing the isolated MMAudio engine",
    )
    engine_root = root / "engines" / "audio" / "MMAudio"
    if os.name == "nt":
        engine_python = root / "engines" / "audio" / ".venv" / "Scripts" / "python.exe"
    else:
        engine_python = root / "engines" / "audio" / ".venv" / "bin" / "python"
    if not engine_python.is_file():
        raise RuntimeError(f"MMAudio Python was not created: {engine_python}")
    _run(
        [
            str(engine_python),
            "-c",
            "from mmaudio.eval_utils import all_model_cfg; all_model_cfg['small_16k'].download_if_needed()",
        ],
        cwd=engine_root,
        label="Downloading the minimum MMAudio Small 16 kHz model",
    )
    return engine_root, engine_python


def _install_audio_lab(root: Path) -> Path:
    script = root / "scripts" / "bootstrap_audio_lab.py"
    _run(
        [sys.executable, str(script), "--repo", str(root), "--json"],
        cwd=root,
        label="Installing or repairing the isolated Audio Lab DSP engine",
    )
    if os.name == "nt":
        python = root / "engines" / "audio_lab" / ".venv" / "Scripts" / "python.exe"
    else:
        python = root / "engines" / "audio_lab" / ".venv" / "bin" / "python"
    if not python.is_file():
        raise RuntimeError(f"Audio Lab Python was not created: {python}")
    return python


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install AIWF's minimum local audio models and isolated dependencies."
    )
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    root = Path(args.repo).expanduser().resolve()
    _ensure_main_dependencies()
    musicgen = _download_musicgen(root)
    mmaudio, mmaudio_python = _install_mmaudio(root)
    audio_lab_python = _install_audio_lab(root)
    payload = {
        "ok": True,
        "musicgen": str(musicgen),
        "mmaudio": str(mmaudio),
        "mmaudio_python": str(mmaudio_python),
        "audio_lab_python": str(audio_lab_python),
        "license": "MusicGen and MMAudio released model weights are CC-BY-NC 4.0 / non-commercial research assets.",
    }
    if args.json:
        print(json.dumps(payload))
    else:
        print("Minimum Audio setup is ready.")
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
