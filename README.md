# AIWF Studio

<p align="center">
  <strong>Local-first Stable Diffusion WebUI rebuild for creators, tinkerers, and local AI workflows.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Stable%20Diffusion-WebUI-00b894?style=for-the-badge" alt="Stable Diffusion WebUI" />
  <img src="https://img.shields.io/badge/Local%20AI-Creative%20Workspace-6750A4?style=for-the-badge" alt="Local AI Creative Workspace" />
  <img src="https://img.shields.io/badge/Gradio-Python%203.10-ff7c00?style=for-the-badge" alt="Gradio Python 3.10" />
  <img src="https://img.shields.io/badge/A1111--style-Clean%20Room-blue?style=for-the-badge" alt="A1111 style clean-room rebuild" />
</p>

AIWF Studio is a local-first creative workspace for Stable Diffusion-style image generation and the next generation of local creative AI tooling.

The project started as a clean-room rebuild of the AUTOMATIC1111-style web UI, but the goal is larger than cloning an old interface. AIWF Studio is being built as a structured, maintainable creative console: explicit services, typed requests, predictable folders, clean APIs, and isolated heavy engines instead of one fragile Python dependency soup.

> Current status: active early development. The image-generation workspace is the current usable core. Video generation, local chat orchestration, LoRA training, and full-model training are being designed as isolated engine workers so they can become part of one app without forcing every dependency into one environment.

## Also known as / search terms

AIWF Studio is relevant to people searching for:

- Stable Diffusion WebUI
- Automatic1111 alternative
- A1111-style web UI
- local Stable Diffusion UI
- local AI image generation
- Gradio Stable Diffusion app
- Python Stable Diffusion interface
- img2img, txt2img, and inpaint UI
- ControlNet workflow UI
- LoRA training workspace
- local diffusion creative studio
- self-hosted AI art tools
- Windows CUDA Stable Diffusion launcher
- clean-room Stable Diffusion WebUI rebuild
- ComfyUI / Forge / AUTOMATIC1111 adjacent local AI tooling

## What makes it different

AIWF Studio is not trying to be another pile of extensions glued onto global state. It is a local creative AI console built around service boundaries, typed requests, repo-local folders, and isolated worker processes.

The current direction is a familiar image-generation workspace with a cleaner backend shape:

- Stable Diffusion-style txt2img, img2img, and inpaint workflows.
- A1111-style `/sdapi/v1` compatibility where useful.
- Native `/api/v1` routes for cleaner integrations.
- Gradio-based local UI.
- Python 3.10+ Windows-first development path.
- Local model folders for checkpoints, LoRAs, VAEs, prompts, wildcards, workflows, and outputs.
- Future isolated engines for video generation, LoRA training, full-model training, and local chat orchestration.

## Why this exists

Local AI tools are powerful, but many of them grew through extension stacks, monkey patches, global state, and dependency collisions. That makes them flexible, but it also makes them hard for normal users and painful for developers to maintain.

AIWF Studio is trying a cleaner approach:

- one stable user-facing app
- explicit service boundaries
- typed request/response models
- local folders that are easy to understand
- API routes that external tools can call reliably
- heavy backends isolated behind worker processes when needed

The aim is not to hide complexity with magic. The aim is to put the complexity in the architecture instead of dumping it on the user.

## Core architecture principles

AIWF Studio is built around a few non-negotiable rules:

- No global `shared` state.
- No mystery callbacks or monkey-patched extension hooks.
- UI actions call services, not deep Torch code directly.
- Requests flow through typed domain models instead of ad-hoc dictionaries.
- Runtime data lives in predictable repo-local folders unless the user configures otherwise.
- Heavy engines should be isolated when their dependencies or GPU usage would destabilize the main UI.

## Current feature set

### Studio image workspace

The current app focuses on a modern image-generation workflow:

- txt2img, img2img, and inpaint
- live preview, continuous generation, and interrupt
- hires fix, CFG, steps, sampler, clip skip, and VAE selection
- tags, PNG metadata, seed reuse, and before/after compare
- dynamic prompts, wildcards, prompt files, and Compel support
- style presets with editable templates
- single-unit ControlNet in Studio Advanced
- SAM-assisted masking for inpaint
- ReActor-style face swap on results

### Extra workspace tabs

Current/active tabs include:

- Models
- Segment
- Enhance
- Workflows
- Face Swap
- Library
- PNG Info
- History
- Settings

### API surface

AIWF Studio includes:

- native `/api/v1` routes
- A1111-style `/sdapi/v1` compatibility adapter
- API/security controls for local and network usage

## Engine isolation direction

The planned architecture keeps AIWF Studio as one app for the user while allowing heavy tools to run in separate environments under the hood.

```text
AIWF Studio UI
Python 3.10 main environment
Gradio / routing / config / logs / model browser / phone companion
        |
        |-- Image + Wan + LTX generation engine
        |       separate generation worker/venv when needed
        |
        |-- Ollama local chat
        |       external local service / GPU tenant
        |
        |-- Kohya LoRA training engine
        |       separate training venv
        |
        `-- EveryDream2-compatible full-training engine
                separate Python 3.10 training venv
```

The UI should not reload into different Python environments when a tab is selected. The Gradio shell stays stable. Engines move underneath it.

This makes the app easier to maintain because each backend can keep the dependency stack it needs:

- generation can use its own Torch/CUDA/diffusers/video tooling
- Kohya can use its own LoRA training stack
- EveryDream2-compatible training can use its own full-training stack
- Ollama can remain an external local model service

The user still launches one program and works from one interface.

## Video and advanced generation roadmap

The next-generation design work is focused on making heavy video models practical in a local-first app without destabilizing the current image workspace.

Planned/experimental targets include:

- Wan video generation as an isolated GPU engine
- LTX-Video support as part of the generation-engine roadmap
- RIFE/RIF frame interpolation as an optional post-processing engine
- NVIDIA NVENC-based video export
- strict GPU tenant locking so training, video, and chat do not fight for VRAM
- process-level cleanup so a failed video worker does not crash the main UI

The first implementation target is not to make every backend live in the main app process. The target is supervised workers, structured logs, clean stop/cleanup, and result paths returned to the UI.

## Training roadmap

Training support is planned as an isolated service layer rather than a direct import into the main UI.

Planned training modes:

- Simple LoRA training through a Kohya/sd-scripts style backend
- Advanced LoRA training with more exposed parameters
- EveryDream2-compatible full-model training through a separate Python 3.10 worker environment

AIWF Studio should own:

- dataset validation
- caption pairing checks
- config generation
- process launch
- live log streaming
- stop/cleanup controls
- output registration in the model manager

The training backend should own the actual training internals. This avoids turning the main app into a dependency battlefield.

## Quick start

```bat
webui-user.bat
```

Or:

```powershell
python launch.py
```

The launcher creates and uses a repo-local `venv/` when needed, installs the required Python packages, and starts the web UI.

For development:

```powershell
python -m pytest tests/ -q
python -m aiwf.app
```

## Requirements

Current project baseline:

- Python 3.10+
- Windows-first local development path
- NVIDIA CUDA path supported by the launcher
- Torch CUDA wheels installed by `launch.py` using the configured PyTorch CUDA index
- Gradio-based local web UI

The project can expose remote access, but it is intended to be local-first by default.

## Folder layout

By default the app uses dedicated local folders inside this repo:

```text
models/
outputs/
prompts/
wildcards/
workflows/
```

Put checkpoints in:

```text
models/Stable-diffusion/
```

Put LoRAs in:

```text
models/Lora/
```

Put VAEs in:

```text
models/VAE/
```

Runtime folders such as `models/`, `outputs/`, and `venv/` are local user data and should not be committed to the public repo.

## Remote access and security

AIWF Studio includes Tailscale-aware connection info in Settings. Tailscale is the preferred remote path when you want phone or tablet access without broadly exposing the app.

Security guidance:

- `--listen` makes the UI reachable from other devices on your network.
- Add `username:password` auth before using remote access outside a trusted desk setup.
- Treat Gradio public share links as convenience tools, not private tunnels.
- Tailscale is the safest built-in option for routine remote use.

## Workflows status

The Workflows tab is useful but still experimental. Treat it as an active work area until it gets deeper validation, stronger schema checks, and better authoring tools.

## Near-term priorities

- Stabilize the current image-generation workspace.
- Harden model/library scanning and user setup flows.
- Mature workflow authoring and validation.
- Add engine-supervisor infrastructure for isolated workers.
- Add training tabs without merging training dependencies into the main UI environment.
- Add video-generation research paths behind safe process boundaries.
- Improve docs, screenshots, and release examples for public testing.

## Credits and thanks

AIWF Studio is clean-room code, but it is absolutely standing in conversation with the wider local-image community.

- [AUTOMATIC1111 / stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- [ControlNet](https://github.com/lllyasviel/ControlNet)
- [sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet)
- [ReActor](https://github.com/Gourieff/sd-webui-reactor)
- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [Diffusers](https://github.com/huggingface/diffusers)
- [GFPGAN](https://github.com/TencentARC/GFPGAN)
- [CodeFormer](https://github.com/sczhou/CodeFormer)
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)

See [docs/ATTRIBUTION.md](docs/ATTRIBUTION.md) for the fuller attribution trail.

## Clean-room rule

Allowed:

- studying behavior
- reading public docs
- reimplementing compatible ideas
- building compatibility layers where licenses allow it

Not allowed:

- copying incompatible source
- importing abandoned plugin code wholesale
- recreating legacy global-state architecture
- hiding third-party licensing requirements from users

## Project note

This repo is moving quickly. The public goal is to make AIWF Studio useful early while keeping the architecture clean enough to grow into image generation, video generation, local chat, LoRA training, full-model training, and remote companion workflows without turning into one giant fragile plugin stack.
