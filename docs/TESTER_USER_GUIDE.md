# AIWF Studio Tester Guide

This guide is for early Windows testers running AIWF Studio Pro.

## Start Pro

Use the Desktop shortcut or run:

```powershell
AIWF Studio Pro.bat
```

Pro starts without terminal windows by default. To debug startup in a visible terminal:

```powershell
AIWF Studio Pro.bat --terminal
```

## Install

Recommended:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\install_aiwf_studio.ps1 -Mode express
```

Express install creates the venv, installs runtime packages, builds the Pro frontend, downloads the default SD 1.5 fp16 model, and creates shortcuts.

If your system Python is the wrong version, install Python 3.10 or 3.11 with `pyenv-win`, then run:

```powershell
pyenv install 3.10.11
pyenv local 3.10.11
py -3.10 -m venv venv
venv\Scripts\python.exe -m pip install -U pip
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\install_aiwf_studio.ps1 -Mode express -SkipPrerequisites
```

Conda option:

```powershell
conda create -n aiwf-studio python=3.10 -y
conda activate aiwf-studio
python -m pip install -U pip
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\install_aiwf_studio.ps1 -Mode express -SkipPrerequisites
```

If you use Conda, launch from that activated environment or point `PYTHON` at the Conda Python before starting `AIWF Studio Pro.bat`.

## Recovery Buttons

Open **Settings -> System & Launch**.

- **Open support terminal** opens PowerShell in the repo with the venv activated.
- **Unload model** frees the current image model from memory.
- **Restart backend** restarts the FastAPI backend and reconnects the Pro window.
- **Reload app window** refreshes the React frontend.

Use **Unload model** before switching from a large model to another large model on 8-12 GB cards.

## Error Reports

When Pro catches a generation or model-loading error, it opens an error report dialog.

Use **Copy report** and send that text with the issue. It includes:

- selected model id and display name
- backend route and device
- model path, architecture, and header metadata when readable
- receipt or failure log path when available
- browser and app version

Use **Save local report** to write the report into the local client-error log. The Logs workspace lists the log files.

## Model Files

Drag model files into Pro or use **Settings -> Upload model files**. The sorter reads headers where possible and moves files into the matching folder without overwriting existing files.

Supported model file extensions include `.safetensors`, `.ckpt`, `.pt`, `.pth`, `.bin`, `.gguf`, and `.onnx`.

## Current v1 Limits

Anima split-file generation and Qwen Image Nunchaku are coming soon. Their files can be sorted locally, but Pro does not list them for download or generation in v1.

Qwen Image, Sana, Krea2, Flux, Wan, and LTX support depends on complete local model folders and matching sidecar files. If a model is missing a VAE, text encoder, shard, or gated config, Pro should report that in the popup instead of failing only in a terminal.
