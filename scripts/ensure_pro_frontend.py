from __future__ import annotations

import argparse
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
    _log("Added stable-diffusion.cpp to the native React Settings backend dropdown source.", quiet=quiet)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch the AIWF Pro frontend source for sd.cpp backend selection.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    patched = patch_app_tsx(_repo_root(), quiet=args.quiet)
    if patched:
        _log("Frontend source changed. Run the normal Pro frontend build before packaging releases.", quiet=args.quiet)


if __name__ == "__main__":
    main()
