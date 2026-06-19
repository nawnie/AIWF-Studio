# Dependency Policy

AIWF Studio is local-first ML software, so dependency upgrades must optimize for
install reliability before novelty.

## Supported Python

- Default target: Python 3.11 or 3.12.
- Minimum declared support: Python 3.10.
- Python 3.13: experimental until the CUDA and optional-engine wheels are boring
  across Windows installs.

Python 3.13 compatibility work is welcome, but do not make 3.13 mandatory until
these pass in a clean environment:

```powershell
python launch.py
python -m pytest tests/ -q
python -c "from aiwf.bootstrap import build_context; from aiwf.web.app import create_web_ui; ctx = build_context(); create_web_ui(ctx)"
```

## High-Risk Packages

Treat these as compatibility-sensitive:

- `torch`, `torchvision`, `xformers`
- `diffusers`
- `transformers` (must stay `<5` unless checkpoint loading is revalidated)
- `gradio`
- `opencv-python-headless`
- `onnxruntime` / `onnxruntime-gpu`
- `spandrel`, `spandrel-extra-arches`
- optional engine stacks under `engines/`

Do not casually bump these versions. Each bump needs a short compatibility note,
targeted tests, and a manual smoke if it touches generation, video, training, or
the Gradio UI.

## Optional Engines

Optional engines must not become boot dependencies. The core app must launch if
Wan, Kohya, ED2, RIFE, face swap, or future engines are missing.

Rules:

- Engine-specific imports stay inside functions, workers, or methods.
- Engine repos live under `engines/<name>/`.
- Engine venvs live under `engines/<name>/.venv` unless explicitly configured to
  use the Studio runtime.
- Shared Studio runtime mode may install AIWF overlay requirements only; do not
  install upstream legacy requirements into the main `venv`.
- Heavy engine work runs through `ProcessSupervisor` and `EngineSupervisor`.

## ED2 Fork

ED2 is the full fine-tune engine. Install Shawn's fork by setting:

```powershell
$env:AIWF_ED2_REPO_URL = "https://github.com/<account>/EveryDream2trainer.git"
```

Then use the Training tab installer or call `install_ed2_addon()`. The clone
target is:

```text
engines/ed2/EveryDream2trainer
```

The installer configures `engines.json` to use Studio runtime mode by default so
ED2 remains optional at app boot and avoids upstream legacy requirement pins.
