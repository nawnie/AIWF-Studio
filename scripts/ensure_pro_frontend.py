from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PATCH_MARKER = "// AIWF-SDCPP-BACKEND-OPTION"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _log(message: str, *, quiet: bool = False) -> None:
    if not quiet:
        print(f"[AIWF frontend] {message}", flush=True)


def patch_app_tsx(root: Path, *, quiet: bool = False) -> bool:
    app = root / "frontend" / "src" / "App.tsx"
    if not app.is_file():
        _log("frontend/src/App.tsx was not found; skipping dropdown patch.", quiet=quiet)
        return False
    text = app.read_text(encoding="utf-8")
    if "{ id: 'sdcpp'" in text:
        return False

    target = "const RUNTIME_BACKEND_OPTIONS = [\n  { id: 'diffusers', label: 'Diffusers' },\n  { id: 'onnx', label: 'ONNX' },\n]"
    replacement = (
        "const RUNTIME_BACKEND_OPTIONS = [\n"
        "  { id: 'diffusers', label: 'Diffusers' },\n"
        "  { id: 'sdcpp', label: 'stable-diffusion.cpp' }, " + PATCH_MARKER + "\n"
        "  { id: 'onnx', label: 'ONNX' },\n"
        "]"
    )
    if target in text:
        text = text.replace(target, replacement, 1)
    else:
        needle = "  { id: 'diffusers', label: 'Diffusers' },"
        if needle not in text:
            raise RuntimeError("Could not locate RUNTIME_BACKEND_OPTIONS in frontend/src/App.tsx")
        text = text.replace(needle, needle + "\n  { id: 'sdcpp', label: 'stable-diffusion.cpp' }, " + PATCH_MARKER, 1)

    app.write_text(text, encoding="utf-8", newline="\n")
    _log("Added stable-diffusion.cpp to the native React Settings backend dropdown.", quiet=quiet)
    return True


def _latest_mtime(paths: list[Path]) -> float:
    latest = 0.0
    for path in paths:
        if not path.exists():
            continue
        if path.is_file():
            latest = max(latest, path.stat().st_mtime)
            continue
        for child in path.rglob("*"):
            try:
                if child.is_file() and "node_modules" not in child.parts:
                    latest = max(latest, child.stat().st_mtime)
            except OSError:
                pass
    return latest


def needs_build(root: Path, *, patched: bool) -> bool:
    dist_index = root / "frontend" / "dist" / "index.html"
    if patched or not dist_index.is_file():
        return True
    source_mtime = _latest_mtime(
        [
            root / "frontend" / "src",
            root / "frontend" / "package.json",
            root / "frontend" / "package-lock.json",
            root / "frontend" / "vite.config.ts",
            root / "frontend" / "index.html",
        ]
    )
    return source_mtime > dist_index.stat().st_mtime


def build_frontend(root: Path, *, quiet: bool = False) -> None:
    frontend = root / "frontend"
    if not frontend.is_dir():
        _log("frontend directory was not found; skipping build.", quiet=quiet)
        return
    npm = "npm.cmd" if os.name == "nt" else "npm"
    if not (frontend / "node_modules").exists():
        if (frontend / "package-lock.json").is_file():
            _log("Installing frontend packages with npm ci.", quiet=quiet)
            subprocess.check_call([npm, "ci"], cwd=str(frontend))
        else:
            _log("Installing frontend packages with npm install.", quiet=quiet)
            subprocess.check_call([npm, "install"], cwd=str(frontend))
    _log("Building Pro frontend.", quiet=quiet)
    subprocess.check_call([npm, "run", "build"], cwd=str(frontend))


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch and optionally build the AIWF Pro frontend.")
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    root = _repo_root()
    patched = patch_app_tsx(root, quiet=args.quiet)
    if args.no_build:
        return
    if needs_build(root, patched=patched):
        build_frontend(root, quiet=args.quiet)
    else:
        _log("Frontend build is current.", quiet=args.quiet)


if __name__ == "__main__":
    main()
