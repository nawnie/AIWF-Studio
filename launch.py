from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent
from typing import Callable

ROOT = Path(__file__).resolve().parent
VENV = ROOT / "venv"
REQUIREMENTS = ROOT / "requirements.txt"
ENGINES_CONFIG = ROOT / "engines.json"
TORCH_INDEX = os.environ.get("TORCH_INDEX_URL", "https://download.pytorch.org/whl/cu124")
PYPI_INDEX = os.environ.get("PYPI_INDEX_URL", "https://pypi.org/simple")
TORCH_CUDA_VERSION = os.environ.get("TORCH_CUDA_VERSION", "2.6.0+cu124")
TORCHVISION_CUDA_VERSION = os.environ.get("TORCHVISION_CUDA_VERSION", "0.21.0+cu124")
TORCHAUDIO_CUDA_VERSION = os.environ.get("TORCHAUDIO_CUDA_VERSION", "2.6.0+cu124")
XFORMERS_PACKAGE = os.environ.get("XFORMERS_PACKAGE", "xformers==0.0.29.post3")
LAUNCH_ONLY_FLAGS = {
    "--install-sageattention",
    "--sageattention",
    "--skip-sageattention",
    "--skip-ltx",
}


# ---------------------------------------------------------------------------
# Engine venv specification
# ---------------------------------------------------------------------------

@dataclass
class EngineSpec:
    """Declares a backend engine's Python environment and entry point.

    Each engine that needs dependency isolation (Kohya, ED2) gets its own
    venv. The generation engine shares the main venv for now â€” the architecture
    doc recommends splitting only if an incompatible dependency stack appears.

    Fields
    ------
    name:
        Short identifier used in engines.json and the GPU tenant lock.
    label:
        Human-readable name for log messages.
    venv_dir:
        Where the engine's venv lives (or will be created).  ``None`` means
        "use the main AIWF venv" (generation engine case).
    worker_script:
        The subprocess entry point â€” written by AIWF, not the upstream tool.
    repo_dir:
        If the engine is a git-cloned tool (kohya_ss, EveryDream2trainer),
        this is where it lives.  Setup steps install its requirements after
        cloning.
    repo_requirements:
        requirements.txt inside the cloned repo (relative to repo_dir).
    extra_requirements:
        AIWF overlay requirements file (engines/<name>/requirements.txt).
    skip_flag:
        CLI flag that disables setup for this engine this session.
    enabled_by_default:
        Whether this engine is set up even without engines.json opt-in.
        Generation = True; training engines = False.
    cuda_torch:
        Whether this engine needs CUDA torch (True for all GPU engines).
    manual_bootstrap_script:
        Specialized bootstrap path for engines that must not use the generic
        torch/install flow.
    setup_hook:
        Optional callable for extra post-install setup steps.
    """
    name: str
    label: str
    worker_script: Path
    venv_dir: Path | None = None          # None â†’ use main venv
    repo_dir: Path | None = None
    repo_requirements: str = "requirements.txt"
    extra_requirements: Path | None = None
    skip_flag: str = ""
    enabled_by_default: bool = False
    cuda_torch: bool = True
    manual_bootstrap_script: str = ""
    setup_hook: Callable[["EngineSpec"], None] | None = field(default=None, repr=False)

    @property
    def effective_venv(self) -> Path:
        """Return the venv to use â€” falls back to main AIWF venv."""
        return self.venv_dir if self.venv_dir is not None else VENV

    def python_exe(self) -> str:
        """Path to the Python executable in this engine's venv."""
        venv = self.effective_venv
        if os.name == "nt":
            return str(venv / "Scripts" / "python.exe")
        return str(venv / "bin" / "python")

    def is_ready(self) -> bool:
        """True if the engine's venv python exists."""
        return Path(self.python_exe()).exists()


def _load_engines_config() -> dict:
    """Read engines.json, return empty dict if missing or malformed."""
    if not ENGINES_CONFIG.exists():
        return {}
    try:
        raw = json.loads(ENGINES_CONFIG.read_text(encoding="utf-8"))
        # Strip comment keys
        return {k: v for k, v in raw.items() if not k.startswith("_")}
    except Exception:
        return {}


def _engine_enabled(name: str, config: dict, default: bool = False) -> bool:
    """Return whether an engine is enabled per engines.json."""
    section = config.get(name, {})
    if isinstance(section, dict):
        return bool(section.get("enabled", default))
    return default


def _engine_path(name: str, key: str, config: dict, fallback: str) -> Path:
    """Resolve a path from engines.json, relative to ROOT."""
    section = config.get(name, {})
    raw = section.get(key, "") if isinstance(section, dict) else ""
    p = Path(raw or fallback)
    return p if p.is_absolute() else ROOT / p


def _engine_venv_dir(name: str, config: dict, fallback: str) -> Path | None:
    """Resolve an engine venv path.

    Training engines default to isolated venvs, but an explicit shared/studio
    sentinel lets experiments run in the main AIWF venv without making the
    engine a mandatory boot dependency.
    """
    section = config.get(name, {})
    if isinstance(section, dict) and "venv_dir" in section:
        raw = section.get("venv_dir")
        if raw is None:
            return None
        raw_text = str(raw).strip().lower()
        if raw_text in {"", "studio", "shared", "main", "aiwf"}:
            return None
        p = Path(str(raw))
        return p if p.is_absolute() else ROOT / p
    p = Path(fallback)
    return p if p.is_absolute() else ROOT / p


# ---------------------------------------------------------------------------
# Built-in engine registry
# ---------------------------------------------------------------------------

def _build_engine_registry() -> list[EngineSpec]:
    """Return the canonical list of AIWF engine specs, merged with engines.json."""
    cfg = _load_engines_config()

    gen_venv_raw = _engine_path("generation", "venv_dir", cfg, "")
    generation = EngineSpec(
        name="generation",
        label="Generation (stable image reference)",
        worker_script=ROOT / "engines" / "generation" / "worker.py",
        # venv_dir=None â†’ uses main venv (ROOT/venv)
        venv_dir=gen_venv_raw if str(gen_venv_raw) != str(ROOT) else None,
        enabled_by_default=True,
        cuda_torch=True,
    )

    wan = EngineSpec(
        name="wan",
        label="Wan video engine",
        worker_script=ROOT / "engines" / "wan" / "worker.py",
        venv_dir=_engine_venv_dir("wan", cfg, "engines/wan/.venv"),
        extra_requirements=ROOT / "engines" / "wan" / "requirements.txt",
        skip_flag="--skip-wan",
        enabled_by_default=False,
        cuda_torch=True,
    )

    kohya = EngineSpec(
        name="kohya",
        label="Kohya LoRA trainer",
        worker_script=ROOT / "engines" / "kohya" / "worker.py",
        venv_dir=_engine_venv_dir("kohya", cfg, "engines/kohya/.venv"),
        repo_dir=_engine_path("kohya", "repo_dir", cfg, "engines/kohya/kohya_ss"),
        repo_requirements="requirements.txt",
        extra_requirements=ROOT / "engines" / "kohya" / "requirements.txt",
        skip_flag="--skip-kohya",
        enabled_by_default=False,
        cuda_torch=True,
    )

    ed2 = EngineSpec(
        name="ed2",
        label="EveryDream2 full trainer",
        worker_script=ROOT / "engines" / "ed2" / "worker.py",
        venv_dir=_engine_venv_dir("ed2", cfg, "engines/ed2/.venv"),
        repo_dir=_engine_path("ed2", "repo_dir", cfg, "engines/ed2/EveryDream2trainer"),
        repo_requirements="requirements.txt",
        extra_requirements=ROOT / "engines" / "ed2" / "requirements.txt",
        skip_flag="--skip-ed2",
        enabled_by_default=False,
        cuda_torch=True,
    )

    ltx = EngineSpec(
        name="ltx",
        label="LTX 2.3 video engine",
        worker_script=ROOT / "engines" / "ltx" / "worker.py",
        venv_dir=_engine_venv_dir("ltx", cfg, "engines/ltx/.venv"),
        repo_dir=_engine_path("ltx", "repo_dir", cfg, "engines/ltx/LTX-2"),
        skip_flag="--skip-ltx",
        enabled_by_default=False,
        cuda_torch=False,
        manual_bootstrap_script="scripts/bootstrap_ltx.ps1",
    )

    llm = EngineSpec(
        name="llm",
        label="AI bot trainer",
        worker_script=ROOT / "engines" / "llm" / "worker.py",
        venv_dir=_engine_venv_dir("llm", cfg, "engines/llm/.venv"),
        extra_requirements=ROOT / "engines" / "llm" / "requirements.txt",
        skip_flag="--skip-llm",
        enabled_by_default=False,
        cuda_torch=True,
    )

    return [generation, wan, ltx, kohya, ed2, llm]


def print_startup_banner() -> None:
    width = min(max(shutil.get_terminal_size((96, 24)).columns, 78), 100)
    inner = width - 4
    logo = [
        '    _      ___  __        __  _____      ____    _____   _   _   ____    ___    ___  ',
        '   / \\    |_ _| \\ \\      / / |  ___|    / ___|  |_   _| | | | | |  _ \\  |_ _|  / _ \\ ',
        '  / _ \\    | |   \\ \\ /\\ / /  | |_       \\___ \\    | |   | | | | | | | |  | |  | | | |',
        ' / ___ \\   | |    \\ V  V /   |  _|       ___) |   | |   | |_| | | |_| |  | |  | |_| |',
        '/_/   \\_\\ |___|    \\_/\\_/    |_|        |____/    |_|    \\___/  |____/  |___|  \\___/ ',
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
            f"torchaudio=={TORCHAUDIO_CUDA_VERSION}",
            "--index-url",
            TORCH_INDEX,
            "--extra-index-url",
            PYPI_INDEX,
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


def sageattention_ready(py: str) -> bool:
    """Return True when sageattention >=1.0.0 is importable.

    SageAttention is an optional accelerator, not a correctness dependency.
    Newer 2.x builds are benchmark-gated on this Windows/CUDA setup before
    being promoted.
    """
    script = (
        "from importlib.metadata import version, PackageNotFoundError\n"
        "from packaging.version import Version\n"
        "try:\n"
        "    v = Version(version('sageattention'))\n"
        "    raise SystemExit(0 if v >= Version('1.0.0') else 1)\n"
        "except PackageNotFoundError:\n"
        "    raise SystemExit(1)\n"
    )
    return _run_ok([py, "-c", script], quiet=True)


def install_sageattention(py: str) -> None:
    """Install SageAttention for faster Wan attention.

    We install the latest available package only as a best-effort optimization.
    Wan must still run correctly when this install fails or the package is not
    compatible with the local CUDA/Triton stack.

    For a current upstream 2.x test lane, use a copied venv and install:
      pip install sageattention==2.2.0 --no-build-isolation
    """
    pkgs = ["sageattention"]
    if os.name == "nt":
        pkgs.append("triton-windows")
    print(f"Installing SageAttention: {', '.join(pkgs)} ...")
    try:
        subprocess.check_call(
            [
                py,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--disable-pip-version-check",
                "--upgrade",
                *pkgs,
            ]
        )
        print("SageAttention installed â€” Wan attention will use optimised kernels.")
    except subprocess.CalledProcessError:
        print(
            "NOTE: SageAttention auto-install skipped (optional). "
            "Wan will work normally using torch SDPA.\n"
            "  To install manually: pip install sageattention"
        )


def should_install_sageattention(argv: list[str]) -> bool:
    """SageAttention is opt-in at setup time; runtime falls back when missing."""
    return "--install-sageattention" in argv or "--sageattention" in argv


def strip_launch_only_args(argv: list[str]) -> list[str]:
    """Remove setup-only flags before invoking webui.py's stricter parser."""
    return [arg for arg in argv if arg not in LAUNCH_ONLY_FLAGS]


def _prepare_engine_venv(spec: EngineSpec, argv: list[str]) -> None:
    """Create and populate the venv for a training engine (Kohya / ED2).

    Skipped if:
    - spec.skip_flag is present in argv
    - The repo_dir doesn't exist yet (user hasn't cloned the tool)
    """
    if spec.skip_flag and spec.skip_flag in argv:
        print(f"[{spec.label}] Skipped ({spec.skip_flag} passed).")
        return

    if spec.manual_bootstrap_script:
        print(
            f"[{spec.label}] Uses a specialized bootstrap. Run "
            f".\\{spec.manual_bootstrap_script} -Enable from the repo root; "
            "generic launch.py setup will not alter this engine."
        )
        return

    if spec.repo_dir and not spec.repo_dir.exists():
        print(
            f"[{spec.label}] Repository not found at {spec.repo_dir}. "
            f"Clone it and set enabled=true in engines.json to enable training.\n"
            f"  git clone ... {spec.repo_dir}"
        )
        return

    shared_main_venv = spec.venv_dir is None
    venv_dir = spec.effective_venv
    if not shared_main_venv and not venv_dir.exists():
        print(f"[{spec.label}] Creating engine venv at {venv_dir} ...")
        subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])

    py = spec.python_exe()

    if not shared_main_venv and spec.cuda_torch and not torch_cuda_ready(py):
        print(f"[{spec.label}] Installing CUDA torch ...")
        install_cuda_torch(py)

    if spec.extra_requirements and spec.extra_requirements.exists():
        scope = "shared Studio venv" if shared_main_venv else "engine venv"
        print(f"[{spec.label}] Installing AIWF engine overlay into {scope} ...")
        subprocess.check_call(
            [py, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
             "-r", str(spec.extra_requirements)]
        )

    if spec.repo_dir and spec.repo_dir.exists():
        repo_req = spec.repo_dir / spec.repo_requirements
        if repo_req.exists() and not shared_main_venv:
            print(f"[{spec.label}] Installing {spec.repo_dir.name} requirements ...")
            subprocess.check_call(
                [py, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
                 "-r", str(repo_req)]
            )
        elif repo_req.exists() and shared_main_venv:
            print(
                f"[{spec.label}] Sharing Studio venv; skipping upstream "
                f"{spec.repo_dir.name}/{spec.repo_requirements} to avoid legacy core pins."
            )

    if spec.setup_hook:
        spec.setup_hook(spec)

    print(f"[{spec.label}] Engine venv ready.")


def prepare(skip_prepare: bool, skip_install: bool, argv: list[str]) -> None:
    if skip_prepare:
        return

    # ------------------------------------------------------------------
    # Main AIWF venv (UI shell + generation engine, shared)
    # ------------------------------------------------------------------
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

    # Optional accelerator. Do not build/install it during normal startup; Wan
    # keeps a torch SDPA fallback and benchmark-gates SageAttention separately.
    if should_install_sageattention(argv) and not sageattention_ready(py):
        install_sageattention(py)

    if "--xformers" in argv and not _run_ok([py, "-c", "import xformers"], quiet=True):
        install_xformers(py)

    # ------------------------------------------------------------------
    # Training engine venvs (Kohya, ED2, LLM) â€” opt-in via engines.json
    # ------------------------------------------------------------------
    engines_cfg = _load_engines_config()
    for spec in _build_engine_registry():
        if spec.enabled_by_default:
            continue  # generation engine already handled above
        if not _engine_enabled(spec.name, engines_cfg, default=False):
            continue
        if spec.skip_flag and spec.skip_flag in argv:
            continue
        _prepare_engine_venv(spec, argv)


def _tee_run(cmd: list[str], env: dict, log_path: Path) -> int:
    """Run cmd, streaming output to both the console and a rolling crash log."""
    sep = "=" * 72
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(log_path), "a", encoding="utf-8", errors="replace", buffering=1) as log_f:
        header = "\n" + sep + "\n[AIWF launch] " + " ".join(cmd) + "\n" + sep + "\n"
        log_f.write(header)
        log_f.flush()

        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )

        try:
            while True:
                chunk = proc.stdout.read(256)
                if not chunk:
                    break
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                log_f.write(chunk.decode("utf-8", errors="replace"))
                log_f.flush()
        except Exception:
            pass

        proc.wait()
        exit_code = proc.returncode
        log_f.write("\n[AIWF launch] Process exited with code " + str(exit_code) + "\n")
        return exit_code


def main() -> None:
    print_startup_banner()
    argv = sys.argv[1:]
    sys.path.insert(0, str(ROOT))
    from aiwf.runtime.bootstrap_env import apply_from_argv

    apply_from_argv(argv)
    os.environ.setdefault("XFORMERS_FORCE_DISABLE_TRITON", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    skip_prepare = "--skip-prepare-environment" in argv
    skip_install = "--skip-install" in argv
    prepare(skip_prepare, skip_install, argv)
    webui_argv = strip_launch_only_args(argv)

    env = os.environ.copy()
    env.setdefault("XFORMERS_FORCE_DISABLE_TRITON", "1")
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    env["PYTHONPATH"] = str(ROOT) if "PYTHONPATH" not in env else str(ROOT) + os.pathsep + env["PYTHONPATH"]

    # Crash logs live under logs/ so the repo root stays user-facing.
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    crash_log = ROOT / "logs" / "aiwf-crash.log"
    ret = _tee_run([python(), str(ROOT / "webui.py"), *webui_argv], env=env, log_path=crash_log)
    if ret != 0:
        print(
            "\n[AIWF] Process exited with code " + str(ret) + ". Full output saved to: " + str(crash_log),
            flush=True,
        )
        sys.exit(ret)


if __name__ == "__main__":
    main()
