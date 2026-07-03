# AIWF Studio Agent Instructions

## Core Behavior

- Work inside this repository unless Shawn explicitly gives another path.
- Prefer `F:\AIWF_Studio` and `venv\Scripts\python.exe` for local commands.
- Read the files you will change before editing them.
- Keep patches narrow and preserve user work in a dirty tree.
- Use built-in file, edit, and shell tools first. Use MCP only when the task needs connected apps, external docs, or explicit connector access.
- After edits, run the smallest useful verification command and report the result.

## Skill Routing

- Use the global `aiwf` skill first for non-trivial AIWF work.
- Use `aiwf-deep-research` before broad model, Civitai, Hugging Face, arXiv, GitHub, academic, or source-weighted research.
- Use `build-web-apps:react-best-practices` for React/Pro UI changes.
- Use `avoid-ai-writing` for tracked docs, README text, release notes, commit/PR text, UI copy, comments, and docstrings.

## Pro And Gradio Workflow

- Pro means the production FastAPI + React UI: `webui_pro.py`, `aiwf/app_pro.py`, `aiwf/web/pro_api.py`, and `frontend/`.
- Gradio is the broader lab/blueprint surface under `aiwf/web/`.
- Unless Shawn says Gradio, assume UI requests target Pro.
- For new pipeline-facing UI work, validate in this order:
  1. Plain Python module or service.
  2. Focused Python compile, unit, preflight, or no-GUI smoke.
  3. Gradio callback or lab wiring when the feature is still being proven.
  4. Pro API contract in `aiwf/web/pro_api.py`.
  5. React UI in `frontend/`.
  6. Pro build and live API/UI smoke when the change affects startup or runtime wiring.
- Use `docs/agent-workflows/GRADIO_TO_PRO_REACT.md` for the full checklist.

## Runtime Policy

- Check `agents_runtime.md` before changing this workstation's attention, precision, VAE, or offload defaults.
- Check `docs/agent-workflows/MODEL_FAMILY_ATTENTION.md` before changing family-specific routing.
- Do not apply one global attention/offload patch across SD, SDXL, DiT, Flux, Qwen, Sana, and video routes without a family-specific smoke.
- Do not download large models, run VRAM-heavy generation, or start training unless Shawn asks for it.
- Prefer metadata, preflight, list-mode, dry-run, and API probes before GPU work.

## Release And Install Validation Notes

Last GitHub install pass: 2026-07-03.

- Pushed `main` through commit `36098907` (`Fix installer default model bootstrap`).
- Fresh GitHub clone was installed at `C:\AIWF_Studio_install_test`.
- Installer command used:
  ```powershell
  powershell -NoProfile -ExecutionPolicy Bypass -File C:\AIWF_Studio_install_test\scripts\install_aiwf_studio.ps1 -Mode express -SkipPrerequisites
  ```
- The installer created a Python 3.10 venv, installed CUDA PyTorch/runtime requirements, downloaded the default SD 1.5 fp16 model, ran `npm ci`, built Pro, and created Desktop shortcuts.
- Pro started from the C: install on default `http://127.0.0.1:7860` and served Pro API/UI requests, including bootstrap, runtime, capabilities, settings, logs, and data.
- The optional NVIDIA VideoFX SDK was not installed; the installer soft-failed that feature as expected.
- A local pip freeze receipt was written at `C:\AIWF_Studio_install_test\install_receipts\pip-freeze-20260703.txt`.
- Runtime versions observed in that install: `torch 2.6.0+cu124`, CUDA 12.4 available, `diffusers 0.38.0`, `transformers 4.57.6`, `fastapi 0.139.0`.

Do not treat the C: install clone as the main working tree. Use it to verify clean installs from GitHub.

## Current Public Anchors

- `README.md`
- `docs/FEATURES.md`
- `docs/IMAGE_MATURITY_MATRIX.md`
- `docs/qa/README.md`
- `docs/ARCHITECTURE.md`
- `docs/agent-workflows/`

Archived or ignored planning files under `_trash/`, `_local/`, `.codex/`, and `plan.md` are not current release guidance unless Shawn asks for history.

## End-Of-Task Report

At the end of every task, stop using tools and report:

- What changed
- Files edited or created
- Commands run
- Test/build result
- Remaining issue, if any
- Recommended next step

Do not claim success unless verification actually ran and passed.
