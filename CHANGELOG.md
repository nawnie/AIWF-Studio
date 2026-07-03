# Changelog

## 2026-07-03

### Pro Release State

- Documented the current Pro-first release state in `README.md`, including what works now, what is blocked, and which model families still need more receipts.
- Kept the public root layout focused on user-facing files, with docs and utility scripts moved into `docs/` and `scripts/`.
- Confirmed that `AGENTS.md` stays local-only through `.gitignore` and is not tracked in the public repo.

### Pro UI And API

- Validated Pro from a fresh GitHub clone at `C:\AIWF_Studio_install_test`.
- Confirmed that the clean install starts the FastAPI plus React/Vite Pro app and serves the built frontend.
- Confirmed Pro API responses for bootstrap, runtime, capabilities, settings, logs, and data.
- Kept Gradio Lab available as the broader test surface while README guidance points new users to Pro first.

### Pipeline Routing

- Kept SDXL refiner checkpoints out of the base-model picker so users do not try to generate from a refiner alone.
- Added Flux Fill routing as an inpaint-only path instead of a normal txt2img checkpoint.
- Added clearer Windows blocking for Z-Image GGUF, where fused GGUF CUDA kernels are Linux-only and the fallback path can exhaust 16 GB GPUs.
- Improved Z-Image text-encoder loading so CUDA systems can use bitsandbytes 4-bit loading when available.
- Rechecked prompt-encoder VRAM before moving large encoders onto GPU, falling back to CPU encode instead of pushing the driver into system-memory paging.

### Installer And Requirements

- Fixed the default SD 1.5 bootstrap step so the installer runs it with the repo root on the Python import path.
- Added `bitsandbytes>=0.46.0` to shared requirements for quantized transformer and text-encoder routes.
- Verified Express install from GitHub: Python 3.10 venv, CUDA PyTorch/runtime requirements, SD 1.5 fp16 default model, frontend `npm ci`, Pro build, and Desktop shortcuts.
- Captured the clean-install runtime versions: `torch 2.6.0+cu124`, CUDA 12.4 available, `diffusers 0.38.0`, `transformers 4.57.6`, and `fastapi 0.139.0`.
- Left NVIDIA VideoFX SDK as an optional user-installed dependency; the installer soft-fails VSR when the SDK is missing.

## 2026-06-12

### Wan I2V

- Made Wan 2.2 image-to-video require both a high-noise and a low-noise model (the two-stage transformer pair Wan 2.2 always needs, even with LoRAs). Removed the single-model load path entirely from the UI, service, and backend so it no longer falls through to downloading the default HF repo when models are already selected.
- Removed the "Model (single / fallback)" dropdown and the "Download base model" button; High noise + Low noise are now the required model selectors. The local `Wan2.2-TI2V-5B-Diffusers` folder is used only as the text-encoder / VAE / scheduler component provider, not as a generation model.
- Selecting fewer than both models now raises a clear "select both" error at every layer instead of silently triggering a download.

### UI Updates

- Added Studio size preset controls alongside the existing width and height sliders, with one-click preset sizes and aspect ratios tuned for SD 1.5 and SDXL workflows.
- Restyled the new size and ratio controls to better match the app's dark professional theme, including the later hotfix pass that tightened the outer tray spacing around each button.
- Reworked the Settings page structure so generation defaults, model paths, launch profile, access and security, and remote access each have clearer separation.
- Moved remote-session tooling into its own Settings tab and kept live session/network details grouped there instead of crowding the main workspace settings surface.

### Hotfixes

- Added launch-profile support for extra model-library roots and extra checkpoint roots so shared installs can be scanned without manually repointing every built-in folder.
- Added import buttons in Settings for existing AUTOMATIC1111 and ComfyUI installs. These imports merge shared model paths into AIWF's next-start launch profile instead of replacing the user's current entries.
- Expanded model discovery so imported libraries now cover more than checkpoints: LoRAs, VAEs, embeddings, ControlNet models, upscalers, and face-restoration models can all be discovered from shared external folders.
- Tightened tests around launch-path parsing, imported path merging, and shared-library scanning to keep the new settings and UI behavior from regressing.
