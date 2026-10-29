"""Create the offline installer payload from explicit, verified build inputs.

This script does not download dependencies. Supply an unpacked CPython 3.12
Windows x64 install-only runtime with venv and ensurepip, and see BUILDING.md
for the remaining native, engine, redistributable, and wheel inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def require_file(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"Required build input is missing: {path}")


def require_pe_machine(path: Path, expected: int, label: str) -> None:
    """Reject helpers built for the wrong Windows architecture before staging."""
    try:
        with path.open("rb") as stream:
            dos = stream.read(64)
            if len(dos) != 64 or dos[:2] != b"MZ":
                raise ValueError("missing DOS header")
            stream.seek(struct.unpack_from("<I", dos, 0x3C)[0])
            header = stream.read(6)
            if len(header) != 6 or header[:4] != b"PE\0\0":
                raise ValueError("missing PE header")
            machine = struct.unpack_from("<H", header, 4)[0]
            if machine != expected:
                raise ValueError(f"machine 0x{machine:04x}")
    except (OSError, ValueError, struct.error) as exc:
        raise SystemExit(f"{label} is not the required Windows architecture: {path} ({exc})") from exc


def verified_wheels(wheelhouse: Path) -> list[Path]:
    provenance = ROOT / "packaging" / "wheel-provenance.json"
    require_file(provenance)
    manifest = json.loads(provenance.read_text(encoding="utf-8"))
    if manifest.get("schema") != "aes-wheel-provenance-v1":
        raise SystemExit("Unsupported wheel provenance manifest")
    requirements = ROOT / "native" / "AIWF.Desktop" / "engine-api-requirements.txt"
    require_file(requirements)
    expected = set()
    for raw in requirements.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            name, sep, version = line.partition("==")
            if not sep or not name or not version:
                raise SystemExit(f"Expected a pinned requirement: {line}")
            expected.add((name.lower().replace("-", "_"), version))
    result, seen = [], set()
    for entry in manifest.get("files", []):
        name = entry.get("filename", "")
        parts = name.split("-")
        if Path(name).name != name or len(parts) < 5 or not name.endswith(".whl"):
            raise SystemExit("Invalid wheel filename in provenance manifest")
        identity = (parts[0].lower().replace("-", "_"), parts[1])
        if identity not in expected or identity in seen:
            raise SystemExit(f"Unexpected or duplicate wheel: {name}")
        path = wheelhouse / name
        require_file(path)
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != entry.get("sha256"):
            raise SystemExit(f"Wheel SHA-256 mismatch: {name}")
        seen.add(identity)
        result.append(path)
    if seen != expected:
        raise SystemExit(f"Wheelhouse does not cover locked requirements: {sorted(expected - seen)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path, help="New, empty destination; must not already exist")
    parser.add_argument("--python-runtime", type=Path, required=True, help="Unpacked Windows CPython runtime")
    parser.add_argument("--cpp-engine", type=Path, default=ROOT / "dependencies" / "stable-diffusion.cpp-cuda13")
    parser.add_argument("--wheelhouse", type=Path, default=ROOT / "packaging" / "wheelhouse")
    parser.add_argument("--native-build", type=Path, default=ROOT / "native" / "bin" / "Release")
    parser.add_argument("--vc-runtime", type=Path, default=ROOT / "packaging" / "vc-runtime")
    parser.add_argument("--credential-bridge", type=Path, default=ROOT / "packaging" / "AesCredentialBridge.dll",
                        help="32-bit NSIS credential bridge; use the same path when compiling setup")
    args = parser.parse_args()
    stage = args.stage.expanduser().resolve()
    if stage.exists():
        raise SystemExit("Stage already exists; choose a new path")

    python = args.python_runtime.expanduser().resolve()
    cpp = args.cpp_engine.expanduser().resolve()
    native = args.native_build.expanduser().resolve()
    vc = args.vc_runtime.expanduser().resolve()
    wheels = args.wheelhouse.expanduser().resolve()
    bridge = args.credential_bridge.expanduser().resolve()
    for path in [
        python / "python.exe", python / "Lib" / "venv" / "__init__.py",
        python / "Lib" / "ensurepip" / "__init__.py",
        cpp / "bin" / "sd-cli.exe", cpp / "bin" / "cublas64_13.dll",
        cpp / "bin" / "cublasLt64_13.dll", native / "AESStudio.exe",
        native / "native-build-info.json", vc / "vcruntime140.dll",
        vc / "vcruntime140_1.dll", vc / "msvcp140.dll", vc / "vcomp140.dll",
        vc / "Redist.txt", ROOT / "packaging" / "uninstall_worker.exe", bridge,
    ]:
        require_file(path)
    for path, machine, label in [
        (python / "python.exe", 0x8664, "Python runtime"),
        (cpp / "bin" / "sd-cli.exe", 0x8664, "Image engine"),
        (native / "AESStudio.exe", 0x8664, "Native application"),
        (ROOT / "packaging" / "uninstall_worker.exe", 0x8664, "Uninstall helper"),
        (bridge, 0x014c, "NSIS credential bridge"),
    ]:
        require_pe_machine(path, machine, label)
    subprocess.run([str(python / "python.exe"), "-I", "-B", "-c",
                    "import sys; raise SystemExit(not (sys.implementation.name == 'cpython' "
                    "and (3, 12, 13) <= sys.version_info[:3] < (3, 13)))"],
                   check=True, capture_output=True)
    wheel_files = verified_wheels(wheels)
    # The executable and API are one versioned unit; reject stale native builds.
    subprocess.run([sys.executable, "-B", "-m", "aiwf.native_compat", "verify",
                    "--source-root", str(ROOT), "--build-info", str(native / "native-build-info.json"),
                    "--exe", str(native / "AESStudio.exe")], cwd=ROOT, check=True)

    stage.mkdir(parents=True)
    def ignore(_directory: str, names: list[str]) -> list[str]:
        return [n for n in names if n == "__pycache__" or n.endswith(
            (".pyc", ".pyo", ".pdb", ".lib", ".exp", ".ilk", ".obj", ".iobj", ".ipdb"))]
    def tree(source: Path, destination: Path) -> None:
        shutil.copytree(source, destination, ignore=ignore, symlinks=False)

    tree(ROOT / "aiwf", stage / "aiwf")
    tree(cpp, stage / "dependencies" / "stable-diffusion.cpp-cuda13")
    tree(ROOT / "dependencies" / "windows-py312", stage / "dependencies" / "windows-py312")
    tree(ROOT / "scripts", stage / "scripts")
    tree(ROOT / "licenses", stage / "licenses")
    tree(native, stage / "native")
    for dll in vc.glob("*.dll"):
        shutil.copy2(dll, stage / "native" / dll.name)
        shutil.copy2(dll, stage / "dependencies" / "stable-diffusion.cpp-cuda13" / "bin" / dll.name)
    shutil.copy2(vc / "Redist.txt", stage / "licenses" / "Microsoft-Visual-Cpp-Redist.txt")
    (stage / "native" / "AIWF.Desktop").mkdir()
    for name in ["engine-api-preflight.py", "engine-api-requirements.txt"]:
        shutil.copy2(ROOT / "native" / "AIWF.Desktop" / name, stage / "native" / "AIWF.Desktop" / name)
    tree(python, stage / "runtime" / "python")
    (stage / "wheels").mkdir()
    for wheel in wheel_files:
        shutil.copy2(wheel, stage / "wheels" / wheel.name)
    (stage / "packaging").mkdir()
    for name in ["finalize_install.py", "setup_settings.py", "uninstall_worker.exe"]:
        shutil.copy2(ROOT / "packaging" / name, stage / "packaging" / name)
    for name in ["LICENSE", "THIRD-PARTY-NOTICES.md"]:
        shutil.copy2(ROOT / name, stage / name)
    shutil.copy2(ROOT / "packaging" / "README-AES-Studio.txt", stage / "README-AES-Studio.txt")
    files = [p for p in stage.rglob("*") if p.is_file()]
    print("Staged", len(files), "files,", sum(p.stat().st_size for p in files), "bytes at", stage)


if __name__ == "__main__":
    main()
