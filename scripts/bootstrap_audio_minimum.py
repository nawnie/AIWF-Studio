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
MUSICGEN_REQUIRED_FILES = tuple(name for name in MUSICGEN_FILES if name != "special_tokens_map.json")
MUSICGEN_LAYOUTS = (
    Path("audio") / "MusicGen" / "musicgen-small",
    Path("MusicGen") / "musicgen-small",
)


def _safe_model_roots(models_dir: Path, extra_model_dirs: list[Path]) -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()
    for value in (models_dir, *extra_model_dirs):
        try:
            root = value.expanduser().resolve()
        except (OSError, RuntimeError):
            continue
        key = str(root).casefold()
        if key not in seen:
            seen.add(key)
            roots.append(root)
    return roots


def _musicgen_files_ready(path: Path) -> bool:
    try:
        if any(not (path / name).is_file() or (path / name).stat().st_size <= 0 for name in MUSICGEN_REQUIRED_FILES):
            return False
        from aiwf.services.audio import _is_complete_safetensors

        return _is_complete_safetensors(path / "model.safetensors")
    except OSError:
        return False


def _find_musicgen_small(model_roots: list[Path]) -> Path | None:
    for root in model_roots:
        for relative in MUSICGEN_LAYOUTS:
            candidate = root / relative
            try:
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(root)
            except (OSError, RuntimeError, ValueError):
                continue
            if _musicgen_files_ready(resolved):
                return resolved
    return None


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


def _download_musicgen(root: Path, models_dir: Path, extra_model_dirs: list[Path]) -> Path:
    shared = _find_musicgen_small(_safe_model_roots(models_dir, extra_model_dirs))
    if shared is not None:
        print(f"[AIWF] Reusing complete MusicGen Small model at {shared}", flush=True)
        return shared
    from huggingface_hub import snapshot_download

    destination = models_dir / "audio" / "MusicGen" / "musicgen-small"
    destination.mkdir(parents=True, exist_ok=True)
    print(f"[AIWF] Downloading {MUSICGEN_REPO} to {destination}", flush=True)
    snapshot_download(
        repo_id=MUSICGEN_REPO,
        local_dir=str(destination),
        allow_patterns=list(MUSICGEN_FILES),
    )
    missing = [name for name in MUSICGEN_REQUIRED_FILES if not (destination / name).is_file()]
    if missing:
        raise RuntimeError(f"MusicGen download is incomplete. Missing: {', '.join(missing)}")
    return destination


def _import_shared_mmaudio(model_roots: list[Path], engine_root: Path) -> list[Path]:
    from aiwf.services.audio import _MMAUDIO_SHARED_LAYOUTS, _is_nonempty_file, _mmaudio_variant_files

    target_root = engine_root.resolve()
    for shared_root in model_roots:
        try:
            target_root.relative_to(shared_root)
        except (OSError, RuntimeError, ValueError):
            continue
        raise RuntimeError(f"MMAudio install target is inside a configured read-only model root: {target_root}")
    copied: list[Path] = []
    for relative in _mmaudio_variant_files("small_16k"):
        target = target_root / relative
        try:
            target.resolve(strict=False).relative_to(target_root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(f"Unsafe MMAudio install destination: {target}") from exc
        if _is_nonempty_file(target):
            continue
        for root in model_roots:
            for layout in _MMAUDIO_SHARED_LAYOUTS:
                candidate = root / layout / relative
                try:
                    source = candidate.resolve(strict=True)
                    source.relative_to(root)
                except (OSError, RuntimeError, ValueError):
                    continue
                if not _is_nonempty_file(source):
                    continue
                if source == target.resolve(strict=False):
                    break
                target.parent.mkdir(parents=True, exist_ok=True)
                resolved_target = target.resolve(strict=False)
                try:
                    resolved_target.relative_to(target_root)
                except (OSError, RuntimeError, ValueError) as exc:
                    raise RuntimeError(f"Unsafe MMAudio install destination: {target}") from exc
                shutil.copy2(source, resolved_target)
                if _is_nonempty_file(resolved_target):
                    copied.append(resolved_target)
                    break
            if _is_nonempty_file(target):
                break
    return copied


def _install_mmaudio(root: Path, model_roots: list[Path]) -> tuple[Path, Path]:
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
    _install_mmaudio_clip_assets(engine_python, model_roots)
    _import_shared_mmaudio(model_roots, engine_root)
    from aiwf.services.audio import _is_nonempty_file, _mmaudio_variant_files

    missing = [relative for relative in _mmaudio_variant_files("small_16k") if not _is_nonempty_file(engine_root / relative)]
    if missing:
        _run(
            [
                str(engine_python),
                "-c",
                "from mmaudio.eval_utils import all_model_cfg; all_model_cfg['small_16k'].download_if_needed()",
            ],
            cwd=engine_root,
            label="Downloading missing MMAudio Small 16 kHz model assets",
        )
    return engine_root, engine_python


def _install_mmaudio_clip_assets(engine_python: Path, model_roots: list[Path]) -> Path:
    from aiwf.services.audio import (
        _MMAUDIO_CLIP_ALLOW_PATTERNS,
        _MMAUDIO_CLIP_REPO,
        _find_mmaudio_clip_hub_cache,
    )

    existing = _find_mmaudio_clip_hub_cache(model_roots)
    if existing is not None:
        print(f"[AIWF] Reusing MMAudio CLIP encoder cache at {existing}", flush=True)
        return existing
    if not model_roots:
        raise RuntimeError("No configured model root is available for the MMAudio CLIP encoder cache.")
    model_root = model_roots[0].resolve()
    cache_root = model_root / "hub"
    try:
        cache_root.resolve(strict=False).relative_to(model_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(f"Unsafe MMAudio CLIP cache destination: {cache_root}") from exc
    program = (
        "import os,sys; os.environ['HF_HUB_CACHE']=sys.argv[1]; os.environ['HF_HUB_OFFLINE']='0'; "
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download(repo_id={_MMAUDIO_CLIP_REPO!r}, cache_dir=sys.argv[1], "
        f"allow_patterns={list(_MMAUDIO_CLIP_ALLOW_PATTERNS)!r})"
    )
    _run(
        [str(engine_python), "-c", program, str(cache_root)],
        cwd=model_root,
        label=f"Installing MMAudio CLIP encoder {_MMAUDIO_CLIP_REPO}",
    )
    installed = _find_mmaudio_clip_hub_cache(model_roots)
    if installed is None:
        raise RuntimeError(
            f"MMAudio CLIP encoder {_MMAUDIO_CLIP_REPO} setup finished, but open_clip_config.json or model weights are missing."
        )
    return installed


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
    parser.add_argument("--models-dir", default="")
    parser.add_argument("--extra-model-dir", action="append", default=[])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    root = Path(args.repo).expanduser().resolve()
    sys.path.insert(0, str(root))
    models_dir = Path(args.models_dir).expanduser().resolve() if args.models_dir else root / "models"
    model_roots = _safe_model_roots(models_dir, [Path(value) for value in args.extra_model_dir])
    _ensure_main_dependencies()
    musicgen = _download_musicgen(root, models_dir, [Path(value) for value in args.extra_model_dir])
    mmaudio, mmaudio_python = _install_mmaudio(root, model_roots)
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
