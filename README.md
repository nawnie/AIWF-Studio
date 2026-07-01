# AIWF Studio

[![Windows](https://img.shields.io/badge/Windows-local--first-0078D4?logo=windows11&logoColor=white)](#quick-start)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-Pro%20UI-61DAFB?logo=react&logoColor=black)](frontend/)
[![TypeScript](https://img.shields.io/badge/TypeScript-Pro%20frontend-3178C6?logo=typescript&logoColor=white)](frontend/)
[![Vite](https://img.shields.io/badge/Vite-build-646CFF?logo=vite&logoColor=white)](frontend/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Pro%20API-009688?logo=fastapi&logoColor=white)](#what-works-on-main)
[![Gradio](https://img.shields.io/badge/Gradio-Lab%20UI-F97316?logo=gradio&logoColor=white)](#aiwf-studio-gradio-lab)
[![Diffusers](https://img.shields.io/badge/Diffusers-local%20pipelines-FFD21E?logo=huggingface&logoColor=black)](#image-generation)
[![NVIDIA RTX / VFX SDK](https://img.shields.io/badge/NVIDIA%20RTX-VFX%20SDK-76B900?logo=nvidia&logoColor=white)](https://docs.nvidia.com/maxine/vfx/index.html)

Local-first creative AI workspace for Windows and NVIDIA GPUs, focused on image generation, inpainting, video generation, and video-audio post-processing.

AIWF Studio is a clean-room rebuild of the AUTOMATIC1111-style Stable Diffusion web UI. The project keeps the wiring explicit: typed requests, predictable model folders, isolated engines, and no legacy global `shared` state.

Built around local model folders and explicit routes for Stable Diffusion, SDXL, SD3.5, Flux, Wan, LTX, RIFE, MMAudio, React, FastAPI, Gradio, Diffusers, and NVIDIA RTX workflows.

This `main` branch is the stable sharing branch. It only advertises features intended for normal local use. Experimental work lives on `dev`.

- Full feature inventory: [`docs/FEATURES.md`](docs/FEATURES.md)
- LoRA pipeline direction: [`docs/LORA_PIPELINE_STRATEGY.md`](docs/LORA_PIPELINE_STRATEGY.md)

## Start Here

New users should start with **AIWF Studio Pro**. It is the cleaner React app and the steadier path for normal local use. Use **AIWF Studio Gradio Lab** for the broader beta workspace where pipeline experiments land first.

Both app tracks read and write the same model folders, output history, and settings. Switching between them is safe.

### AIWF Studio Pro

<p align="center">
  <img src="static/icons/aiwf-studio-pro.png" alt="AIWF Studio Pro icon" width="96">
</p>

**Stable UI track.** Best for Create, Models, Data, Monitor, Logs, and Settings.

```bat
AIWF Studio Pro.bat
```

```powershell
python launch_pro.py
```

<p align="center">
  <img src="docs/assets/aiwf-studio-pro-sana-sprint.png" alt="AIWF Studio Pro image generation workspace" width="100%">
</p>

### AIWF Studio Gradio Lab

<p align="center">
  <img src="static/icons/aiwf-studio-gradio-lab.png" alt="AIWF Studio Gradio Lab icon" width="96">
</p>

**Beta workspace.** Best for image, inpaint, ControlNet, enhance, segment, and video testing.

```bat
AIWF Studio Gradio Lab.bat
```

```powershell
python launch_gradio.py
```

<p align="center">
  <img src="docs/assets/aiwf-studio-gradio-lab-continuous.png" alt="AIWF Studio Gradio Lab continuous image workspace" width="100%">
</p>

## Release Focus

Current focus: image generation, inpainting, video generation, and video-audio post-processing must be reliable before the project takes on more feature work.

- New user-facing features are paused until those paths pass local smoke tests.
- Optimization work is allowed when it improves an existing path and has a fallback.
- Benchmark claims need timing receipts from this repo, not upstream marketing numbers.
- Optional engines, model weights, SDKs, and generated outputs stay local and are not committed.

## Quick Start

On Windows, use the installer:

```bat
Install AIWF Studio.bat
```

Choose **Express**. It checks or installs Git, uv, Python 3.10, and Node.js LTS; prepares the AIWF runtime; builds the Pro frontend; and creates Desktop shortcuts for Pro and Gradio Lab.

Manual launchers:

```bat
AIWF Studio Pro.bat
AIWF Studio Gradio Lab.bat
```

Python entry points:

```powershell
python launch_pro.py
python launch_gradio.py
```

Older compatibility entry points still work:

```powershell
python launch.py        # Gradio Studio
webui.bat               # Gradio Studio
python webui_pro.py     # Pro API/React app, build frontend/ first
```

Optional local speed/settings logging:

```powershell
python launch.py --genlog
```

`--genlog` writes JSONL entries to `outputs/genlog/generation-log.jsonl` for SD, SDXL, and Wan runs. It records timings, runtime route/pipeline, settings, models, and LoRAs, but not prompt text. The flag is off by default.

## Runtime Folders

AIWF Studio creates and uses these local runtime folders:

```text
models/
outputs/
prompts/
wildcards/
workflows/
```

Common model locations:

```text
models/Stable-diffusion/   checkpoints
models/Loras/              LoRAs
models/VAE/                VAEs
models/ControlNet/         ControlNet models
models/sam/                SAM weights
models/wan/GGUF/           Wan high/low GGUF transformers
models/wan/Diffusers/      Wan shared components
models/flux/GGUF/          Flux GGUF transformers
models/flux/UNet/          Flux safetensors transformers
models/flux/Textencoder/   Flux CLIP-L and T5-XXL encoders
models/flux/VAE/           Flux ae.safetensors VAE
models/ltx/checkpoints/    LTX 2.3 checkpoints
models/ltx/upscalers/      LTX 2.3 spatial upscalers
models/ltx/text_encoder/   LTX 2.3 Gemma text encoder snapshot
models/insightface/        ReActor inswapper ONNX models
models/reactor/faces/      saved ReActor face models
```

No hard links or junctions are required. To reuse an existing A1111, ComfyUI, or shared model library, open **Settings -> Model paths** and add the folders as extra scan roots. Optional SDK/app paths, such as NVIDIA VideoFX executables, live under **Settings -> Engines & pipelines -> External tool paths**.

## What Works On Main

### Image Generation

- txt2img, img2img, and inpaint
- Stable Diffusion 1.5, SDXL, SD3.5, and Flux txt2img checkpoint loading through Diffusers
- sampler, scheduler, steps, CFG, seed, size, VAE, clip skip, and hires fix controls
- live preview, interrupt, continuous generation, and job history
- prompt styles, wildcards, prompt files, dynamic prompt syntax, and Compel support
- LoRA selection, keyword expansion, saved aliases/strengths, and runtime adapter loading for supported Diffusers image families
- PNG metadata and PNG Info import back into the Image tab

### UI And Monitoring

- FastAPI + React/TypeScript/Vite Pro app with left-rail navigation
- Create, Models, Data, Monitor, Logs, and Settings workspaces
- scroll-safe panels, popup tool windows, and resizable workspace columns
- runtime monitor for backend state, queue health, logs, resources, and recent receipts
- browser-side Transformers.js prompt helper loaded only when **Analyze prompt** is clicked
- Pro API endpoints for runtime, bootstrap, generation, data, logs, and settings

### Inpaint, Masking, And Segment

- inpaint image/mask editor flow
- keep-original / last-result source handling
- SAM-assisted mask presets when SAM models are installed locally
- outpaint canvas expansion
- SAM mask generation when SAM weights are installed
- text-guided boxes through GroundingDINO when the optional dependency is available

### ControlNet

- single ControlNet unit in the Image advanced panel
- local ControlNet model selection
- built-in lightweight preprocessors where available

### Models And LoRA

- local checkpoint, LoRA, VAE, ControlNet, SAM, and enhancement model scanning
- SD3.5 Diffusers-folder checkpoints are supported in `models/Stable-diffusion/`
- Flux split-model txt2img is supported from `models/flux/GGUF/` or `models/flux/UNet/` with local CLIP-L, T5-XXL, and `ae.safetensors`
- model aliases and trigger-word helpers
- curated download entries for common local model folders
- import helpers for model folders from another local install
- SD/SDXL/SD3.5-style runtime LoRA loading through Diffusers adapter APIs
- Wan stage LoRAs for supported 5B and high/low transformer routes, with runtime-aware filtering
- Flux/new transformer-image LoRA and ONNX LoRA are intentionally blocked until their pipeline-specific appliers are implemented and tested

### Enhance

- image upscale
- GFPGAN / CodeFormer-style restoration when models are installed
- old-photo restore pipeline
- tiled upscale controls for local VRAM limits

### Image Lab

- maturity matrix tracking each image route against the AUTOMATIC1111 parity baseline: [`docs/IMAGE_MATURITY_MATRIX.md`](docs/IMAGE_MATURITY_MATRIX.md)
- XYZ plot runner, batch img2img/inpaint runner, and loopback runner
- native `GET /api/v1/image/maturity` endpoint

### Video

- Wan image-to-video through three explicit local routes: 5B safetensors, 14B FP8/safetensors, or matched GGUF High Noise + Low Noise transformer pairs
- optional LTX 2.3 text/image-to-video through an isolated worker engine
- optional RIFE post-processing to write 30 FPS or 60 FPS output after generation
- optional ReActor post-processing from the first key frame, an uploaded image, or a saved face model
- optional NVIDIA RTX VSR / Video Effects SDK upscale post-processing when the SDK is installed
- optional generated audio muxing after video when a supported local audio backend is installed
- optional video-conditioned audio post-processing through MMAudio, installed in an isolated engine venv
- standalone RIFE frame interpolation tab for existing videos
- standalone Audio tab for generating music or sound effects after a video
- local Wan component folder support for tokenizer, text encoder, scheduler, and VAE
- conservative route selection so users cannot mix 5B, 14B FP8/safetensors, and GGUF settings by accident

Wan optimization work is still active. FP8, resident high/low mode, streamed block offload, SageAttention, and similar accelerator paths must stay benchmark-gated.
