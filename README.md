# AIWF Studio

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![NVIDIA RTX / VFX SDK](https://img.shields.io/badge/NVIDIA%20RTX-VFX%20SDK-76B900?logo=nvidia&logoColor=white)](https://docs.nvidia.com/maxine/vfx/index.html)

**Local-first AI image, inpainting, video, and video-audio tooling for Windows, NVIDIA GPUs, Stable Diffusion, ControlNet, enhancement, and Wan.**

AIWF Studio is a clean-room rebuild of the AUTOMATIC1111-style Stable Diffusion web UI. It is designed as a serious local creative workspace: explicit wiring, typed requests, predictable model folders, and no legacy global `shared` state.

This `main` branch is the stable sharing branch. It only advertises features that are intended to work for normal local use. Experimental work lives on `dev`.

## Release Gate

Current focus: make image generation, inpainting, video generation, and video-audio post-processing boringly reliable before adding more features.

- New user-facing features are paused until those paths pass local smoke tests.
- Optimization work is allowed when it improves an existing path and has a fallback.
- Benchmark claims need timing receipts from this repo, not upstream marketing numbers.
- Optional engines, model weights, SDKs, and generated outputs stay local and are not committed.

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

- Wan image-to-video through three explicit local routes: 5B safetensors, 14B FP8/safetensors, or matched GGUF High Noise + Low Noise transformer pairs
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

The Video tab keeps 5B safetensors, 14B FP8/safetensors, and GGUF high/low pairs as separate runtime routes. The UI should filter settings based on that route so a user cannot accidentally send GGUF options into a safetensors backend or vice versa.

## Video Audio Setup

The near-term audio path is VAP: video audio post-processing. AIWF generates or accepts a video first, then an optional local audio backend creates audio and muxes it back into the MP4.

The first video-conditioned backend is MMAudio. It is isolated under:

```text
engines/audio/
```

Bootstrap script:

```powershell
scripts/bootstrap_mmaudio.ps1
```

MMAudio is optional and soft-fails when not installed so the visual video output is preserved.

Important license note: MMAudio code is MIT licensed, but the released checkpoints are CC-BY-NC 4.0, so this route should be treated as non-commercial unless you have separate permission.

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
- `docs/ATTRIBUTION.md`
- `docs/DEPENDENCY_POLICY.md`
- `docs/ENGINE_ISOLATION.md`
- `docs/MAINTAINER_NOTES.md`
- `docs/TRAINING_ENGINE_ROADMAP.md`

## License And Third-party Status

This is a practical release checklist, not legal advice.

- AIWF Studio's own repo license is not declared yet. Until a root `LICENSE` is added, the code is public/source-available but not formally open source.
- Model weights, generated outputs, NVIDIA SDK binaries, MMAudio checkout files, and large engine repos are local-only and ignored by git.
- Users are responsible for the licenses of checkpoints, LoRAs, VAEs, ControlNet models, SAM weights, Wan files, and audio models they install.
- NVIDIA Video Effects / VFX SDK support is optional. AIWF does not vendor or redistribute NVIDIA SDK binaries or models.
- MMAudio checkpoints are CC-BY-NC 4.0. Do not present MMAudio-backed audio as commercial-safe without separate permission.
- InsightFace code is MIT, but InsightFace-trained models and the inswapper face-swap model require separate license care for non-local or commercial use. Face swapping must only be used with consent and applicable-law compliance.
- Segment Anything is Apache-2.0; AIWF's segment/inpaint path is clean-room integration, with attribution kept in `docs/ATTRIBUTION.md`.

Before a broader public release, choose a root repo license and keep optional restricted components clearly marked as local/user-installed.

## SageAttention And SDK Cache

SageAttention is a promising Wan/video optimization, and the upstream project is Apache-2.0. It belongs in `F:\sdks` as a future accelerator reference and disposable test lane, not as a required runtime dependency yet.

Current rule for `main`: do not wire SageAttention as a required path until a copied-venv test proves installability, output quality, and speed on this Windows/NVIDIA setup. Wan should keep working through the existing torch SDPA fallback when SageAttention is missing.

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

Optional video post-processing can use NVIDIA Video Effects / VFX SDK components for RTX VSR-style upscale, cleanup, AI green screen, and relighting when the user installs the NVIDIA SDK locally. See `docs/ATTRIBUTION.md` for third-party credits and source links.

## Come Build This

This project is now bigger than one person, even with AI help. If you care about local-first creative AI, consumer GPU workflows, open tooling, clean Python architecture, or making powerful generation tools easier for regular people to run, help create this with us ✨
