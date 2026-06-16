# Contributing to AIWF Studio

AIWF Studio is a local-first AI runtime and Gradio application built around explicit dependency injection, typed domain models, isolated worker processes, and a large regression test suite.

The best contributions improve reliability, engine isolation, memory behavior, setup clarity, tests, and developer ergonomics.

## 1. Prerequisites

- Git
- Python 3.10
- Windows 10/11 is the primary tested local setup
- NVIDIA GPU recommended for generation and training workflows
- Enough disk space for the repo-local `venv/`, model files, and optional engine venvs

Do not commit local model files, generated outputs, venvs, logs, or private config.

## 2. Clone the repository

```powershell
git clone https://github.com/nawnie/AIWF-Studio.git
cd AIWF-Studio
```

If you are working from a fork, clone your fork and add the upstream repo:

```powershell
git remote add upstream https://github.com/nawnie/AIWF-Studio.git
```

## 3. Bootstrap the core Studio environment

For a contributor setup that avoids optional training/video engines on first run:

```powershell
python launch.py --skip-sageattention --skip-wan --skip-kohya --skip-ed2
```

The first run creates `venv/`, installs CUDA Torch and Python requirements, then starts the web UI. Once the UI starts, stop it with `Ctrl+C` if you only wanted to prepare the environment.

After bootstrap, run commands through the repo venv:

```powershell
.\venv\Scripts\python.exe -m pytest --collect-only -q tests
.\venv\Scripts\python.exe -m pytest tests -q
```

As of this contributor guide, the suite collects 715+ tests. In this workspace, the verified collection count was 722 tests. Treat the exact number as a moving target and trust the local `pytest --collect-only` result.

## 4. Run the app locally

After the environment is ready:

```powershell
.\venv\Scripts\python.exe -m aiwf.app
```

Or use the launcher:

```powershell
.\webui-user.bat
```

The default Gradio URL is usually:

```text
http://127.0.0.1:7860
```

Model files are not committed. Put checkpoints under:

```text
models/Stable-diffusion/
```

Put LoRAs under:

```text
models/Loras/
```

Put VAEs under:

```text
models/VAE/
```

## 5. Optional engine venvs

AIWF Studio can run optional heavy engines outside the core UI process.

```powershell
.\scripts\bootstrap_engine.ps1 wan
.\scripts\bootstrap_engine.ps1 kohya
```

Enable engines in `engines.json` only after their venvs and upstream repos are ready. Optional engines must never become mandatory app boot dependencies.

Current engine boundaries:

- Core Studio UI and image reference path use `venv/`.
- Wan video can use `engines/wan/.venv`.
- Kohya training can use `engines/kohya/.venv`.
- EveryDream2 can use an isolated venv or the explicitly configured Studio/shared mode.

## 6. Architecture rules

- UI callbacks orchestrate; heavy work belongs in services, infrastructure, or subprocess workers.
- Use `AppContext`; do not introduce global `shared` state.
- Build generation requests with typed domain models.
- Use `ProcessSupervisor` for worker subprocesses.
- Keep worker communication JSONL-based and testable.
- Keep optional engines optional at import time and at app boot.
- Do not copy incompatible source from legacy web UIs or abandoned plugins.

Useful starting docs:

- `ARCHITECTURE.md`
- `docs/ENGINE_ISOLATION.md`
- `docs/WORKER_PROTOCOL.md`
- `docs/TRAINING_ENGINE_ROADMAP.md`

## 7. Before opening a PR

Run the focused tests for your change, then the full suite when practical:

```powershell
.\venv\Scripts\python.exe -m pytest tests/test_<area>.py -q
.\venv\Scripts\python.exe -m pytest tests -q
```

For UI-only changes, also smoke import the UI:

```powershell
.\venv\Scripts\python.exe -c "from aiwf.bootstrap import build_context; from aiwf.web.app import create_web_ui; ctx = build_context(); create_web_ui(ctx)"
```

For docs-only changes, run test collection at minimum:

```powershell
.\venv\Scripts\python.exe -m pytest --collect-only -q tests
```

## 8. Good first contributions

Good starter work includes:

- Improving setup docs
- Adding tests around existing services
- Tightening worker protocol validation
- Improving model catalog safety checks
- Making engine readiness errors clearer
- Adding small UI smoke tests
- Adding receipts or checks around model operations

Avoid large feature rewrites until you understand the tenant, worker, and service boundaries.

## 9. Pull request checklist

- Explain the user-visible behavior change.
- Mention any new dependencies.
- Include test output.
- Do not commit model weights, outputs, local config, or venv files.
- Keep optional engines optional.
- Keep UI callbacks thin and service-oriented.
- Update docs only when behavior or architecture actually changed.
