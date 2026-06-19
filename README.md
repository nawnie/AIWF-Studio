# AIWF Studio

**Local-first AI image generation for Windows, NVIDIA GPUs, Stable Diffusion, ControlNet, inpainting, enhancement, and Wan GGUF image-to-video.**

AIWF Studio is a clean-room rebuild of the AUTOMATIC1111-style Stable Diffusion web UI. It is designed as a serious local creative workspace: explicit wiring, typed requests, predictable model folders, and no legacy global `shared` state.

This `main` branch is the stable sharing branch. It only advertises features that are intended to work for normal local use. Experimental work lives on `dev`.

## What Works On Main

### Image Generation

- txt2img, img2img, and inpaint
- Stable Diffusion 1.5 and SDXL checkpoint loading through Diffusers
- sampler, scheduler, steps, CFG, seed, size, VAE, clip skip, and hires fix controls
- live preview, interrupt, continuous generation, and job history
- prompt styles, wildcards, prompt files, dynamic prompt syntax, and Compel support
- LoRA selection and keyword expansion from the local model catalog
- PNG metadata and PNG Info import back into the Image tab

### Inpaint And Masking

- inpaint image/mask editor flow
- keep-original / last-result source handling
- SAM-assisted mask presets when SAM models are installed locally
- outpaint canvas expansion

### ControlNet

- single ControlNet unit in the Image advanced panel
- local ControlNet model selection
- built-in lightweight preprocessors where available

### Models

- local checkpoint, LoRA, VAE, ControlNet, SAM, and enhancement model scanning
- model aliases and trigger-word helpers
- curated download entries for common local model folders
- import helpers for model folders from another local install

### Enhance

- image upscale
- GFPGAN / CodeFormer style restoration when models are installed
- old-photo restore pipeline
- tiled upscale controls for local VRAM limits

### Segment

- SAM mask generation when SAM weights are installed
- text-guided boxes through GroundingDINO when the optional dependency is available

### Video

- Wan image-to-video through matched **GGUF High Noise + Low Noise** transformer pairs
- optional RIFE post-processing to write 30 FPS or 60 FPS output after generation
- optional ReActor post-processing from the first key frame, an uploaded image, or a saved face model
- optional NVIDIA RTX VSR / Video Effects SDK upscale post-processing when the SDK is installed
- optional generated audio muxing after video when AudioCraft or Transformers MusicGen is installed
- standalone RIFE frame interpolation tab for existing videos
- standalone Audio tab for generating music or sound effects after a video
- local Wan component folder support for tokenizer, text encoder, scheduler, and VAE
- conservative default UI: GGUF only on `main`

FP8 Wan, resident high/low mode, streamed block offload, and other video experiments are intentionally not exposed on this branch.

### Library, History, And Settings

- generated-output history
- library search over saved outputs
- saved workspace settings
- launch settings for GPU/network/runtime behavior
- Tailscale-friendly remote access information

### API

- native `/api/v1`
- A1111-style `/sdapi/v1` compatibility adapter

## Quick Start

Run:

```bat
webui.bat
```

Or:

```powershell
python launch.py
```

AIWF Studio creates and uses local runtime folders:

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
models/insightface/        ReActor inswapper ONNX models
models/reactor/faces/      saved ReActor face models
```

## Wan GGUF Video Setup

For the stable Video tab, use a matched pair:

```text
models/wan/GGUF/
  ...high...gguf
  ...low...gguf
```

You also need local Wan shared components under:

```text
models/wan/Diffusers/Wan2.2-TI2V-5B-Diffusers/
```

The Video tab blocks FP8/safetensors high-low transformer experiments on `main` so shared users get the least fragile path first.

## Remote Access

Use Tailscale when possible. If you launch with network listening enabled, add authentication before using AIWF outside a trusted local network.

## Project Shape

- `main` is the stable runtime branch for users.
- `dev` keeps broader experiments and active research work.
- `docs/`, `tests/`, and `scripts/` are part of the public maintainability story.
- runtime data such as models, outputs, local configs, and agent notes are ignored.

Useful project docs:

- `ARCHITECTURE.md`
- `CONTRIBUTING.md`
- `docs/DEPENDENCY_POLICY.md`
- `docs/ENGINE_ISOLATION.md`
- `docs/TRAINING_ENGINE_ROADMAP.md`

## WIP And Help Wanted

These areas exist as work-in-progress or need more hardware coverage before they should be treated as stable:

- Wan FP8 high/low video speed path
- Wan resident / streamed offload modes
- training engines
- Ollama or llama.cpp chat workspace
- Face Swap tab
- workflow authoring
- model conversion and quantization tools
- plugin ecosystem
- richer generated-audio controls and model installers
- AMD, Intel, Linux, and lower-VRAM validation
- installer polish and first-run onboarding

## Credits

AIWF Studio is clean-room code, but it is built in conversation with the local AI community: Stable Diffusion, Diffusers, ControlNet, Segment Anything, GroundingDINO, Real-ESRGAN, GFPGAN, CodeFormer, Wan, ComfyUI-GGUF, and the AUTOMATIC1111 web UI ecosystem.

## Come Build This

This project is now bigger than one person, even with AI help. If you care about local-first creative AI, consumer GPU workflows, open tooling, clean Python architecture, or making powerful generation tools easier for regular people to run, help create this with us ✨
