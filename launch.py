from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parent
VENV = ROOT / "venv"
REQUIREMENTS = ROOT / "requirements.txt"
TORCH_INDEX = os.environ.get("TORCH_INDEX_URL", "https://download.pytorch.org/whl/cu124")
TORCH_CUDA_VERSION = os.environ.get("TORCH_CUDA_VERSION", "2.6.0+cu124")
TORCHVISION_CUDA_VERSION = os.environ.get("TORCHVISION_CUDA_VERSION", "0.21.0+cu124")
XFORMERS_PACKAGE = os.environ.get("XFORMERS_PACKAGE", "xformers==0.0.29.post3")


def print_startup_banner() -> None:
    width = min(max(shutil.get_terminal_size((96, 24)).columns, 78), 100)
    inner = width - 4
    logo = [
        r"     _      ___ __        __ _____     ____  _____  _   _  ____   ___   ___",
        "    / \\    |_ _|\\ \\      / /|  ___|   / ___||_   _|| | | ||  _ \\ |_ _| / _ \\",
        r"   / _ \    | |  \ \ /\ / / | |_      \___ \  | |  | | | || | | | | | | | | |",
        r"  / ___ \   | |   \ V  V /  |  _|      ___) | | |  | |_| || |_| | | | | |_| |",
        r" /_/   \_\ |___|   \_/\_/   |_|       |____/  |_|   \___/ |____/ |___| \___/",
    ]
    lines = [
        "",
        *logo,
        "",
        "Local-first image generation workspace",
        "Boot sequence: environment -> model cache -> Gradio UI",
        "Quiet launch enabled | Android-ready API surface",
        "",
    ]
    border = "+" + "=" * (width - 2) + "+"
    art = [border]
    for line in lines:
        art.append("| " + line[:inner].center(inner) + " |")
    art.append(border)
    pulse = "\033[1;96m" if sys.stdout.isatty() else ""
    reset = "\033[0m" if sys.stdout.isatty() else ""
    print(f"{pulse}" + "\n".join(art) + f"{reset}\n", flush=True)


def python() -> str:
    if os.name == "nt":
        candidate = VENV / "Scripts" / "python.exe"
    else:
        candidate = VENV / "bin" / "python"
    return str(candidate if candidate.exists() else sys.executable)


def _run_ok(command: list[str], *, quiet: bool = False) -> bool:
    stdout = subprocess.DEVNULL if quiet else None
    stderr = subprocess.DEVNULL if quiet else None
    return subprocess.call(command, stdout=stdout, stderr=stderr) == 0


def torch_cuda_ready(py: str) -> bool:
    return _run_ok(
        [
            py,
            "-c",
            "import torch; raise SystemExit(0 if torch.cuda.is_available() and torch.version.cuda else 1)",
        ],
        quiet=True,
    )


def requirements_satisfied(py: str) -> bool:
    if not REQUIREMENTS.exists():
        return True
    script = dedent(
        f"""
        from importlib import metadata
        from packaging.requirements import Requirement

        requirements = {str(REQUIREMENTS)!r}
        for raw in open(requirements, encoding="utf-8"):
            line = raw.split("#", 1)[0].strip()
            if not line or line.startswith("-"):
                continue
            req = Requirement(line)
            try:
                version = metadata.version(req.name)
            except metadata.PackageNotFoundError:
                raise SystemExit(1)
            if req.specifier and version not in req.specifier:
                raise SystemExit(1)
        """
    )
    return _run_ok([py, "-c", script], quiet=True)


def install_cuda_torch(py: str) -> None:
    print(f"Installing CUDA PyTorch from {TORCH_INDEX} ...")
    subprocess.check_call(
        [
            py,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--disable-pip-version-check",
            "--upgrade",
            "--force-reinstall",
            f"torch=={TORCH_CUDA_VERSION}",
            f"torchvision=={TORCHVISION_CUDA_VERSION}",
            "--index-url",
            TORCH_INDEX,
        ]
    )


def install_xformers(py: str) -> None:
    print(f"Installing {XFORMERS_PACKAGE} (no-deps to protect CUDA torch)...")
    subprocess.check_call(
        [
            py,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--disable-pip-version-check",
            "--upgrade",
            XFORMERS_PACKAGE,
            "--no-deps",
        ]
    )


def prepare(skip_prepare: bool, skip_install: bool, argv: list[str]) -> None:
    if skip_prepare:
        return
    if not VENV.exists():
        print("Creating virtual environment...")
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV)])

    py = python()
    if skip_install:
        if "--xformers" in argv and not _run_ok([py, "-c", "import xformers"], quiet=True):
            install_xformers(py)
        return

    if not torch_cuda_ready(py):
        install_cuda_torch(py)

    if not requirements_satisfied(py):
        print("Installing missing Python requirements...")
        subprocess.check_call(
            [
                py,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--disable-pip-version-check",
                "-r",
                str(REQUIREMENTS),
            ]
        )

    if not torch_cuda_ready(py):
        print("WARNING: CUDA is still not available after install. Generation will use CPU.")

    if "--xformers" in argv and not _run_ok([py, "-c", "import xformers"], quiet=True):
        install_xformers(py)


def main() -> None:
    print_startup_banner()
    os.environ.setdefault("XFORMERS_FORCE_DISABLE_TRITON", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    argv = sys.argv[1:]
    skip_prepare = "--skip-prepare-environment" in argv
    skip_install = "--skip-install" in argv
    prepare(skip_prepare, skip_install, argv)

    env = os.environ.copy()
    env.setdefault("XFORMERS_FORCE_DISABLE_TRITON", "1")
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    env["PYTHONPATH"] = str(ROOT) if "PYTHONPATH" not in env else f"{ROOT}{os.pathsep}{env['PYTHONPATH']}"
    subprocess.check_call([python(), str(ROOT / "webui.py"), *argv], env=env)


if __name__ == "__main__":
    main()
