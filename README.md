# AIWF Studio

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![NVIDIA RTX / VFX SDK](https://img.shields.io/badge/NVIDIA%20RTX-VFX%20SDK-76B900?logo=nvidia&logoColor=white)](https://docs.nvidia.com/maxine/vfx/index.html)

Local-first image, inpainting, video, and video-audio tooling for Windows and NVIDIA GPUs.

AIWF Studio is a clean-room rebuild of the AUTOMATIC1111-style Stable Diffusion web UI. The goal is a local creative workstation with explicit wiring, typed requests, predictable model folders, and no legacy global `shared` state.

This `main` branch is the stable sharing branch. It only advertises features intended for normal local use. Experimental work lives on `dev`.

For a complete GitHub-facing inventory, see [`docs/FEATURES.md`](docs/FEATURES.md).
For the multi-pipeline LoRA design direction, see [`docs/LORA_PIPELINE_STRATEGY.md`](docs/LORA_PIPELINE_STRATEGY.md).

## UI Rebuild

AIWF Studio ships three interchangeable web UIs on top of the same backend:

- **Studio** (`webui.bat` / `python launch.py`, `aiwf/app.py`) - the original Gradio-based tabbed workspace. Still the default and most complete surface for image, inpaint, ControlNet, enhance, segment, and video.
- **Modern** (`webui_modern.py`, `aiwf/app_modern.py`) - a restyled Gradio shell (`aiwf/web/modern/`) with the same backend, aimed at a cleaner layout pass.
- **Pro** (`webui_pro.py`, `aiwf/app_pro.py`) - a from-scratch FastAPI + React/TypeScript/Vite frontend (`frontend/`) talking to a dedicated `aiwf/web/pro_api.py` API. This is the active UI rebuild track and the long-term direction for the project; it covers the Create, Models, Data, Monitor, Logs, and Settings workspaces, including runtime monitoring and a lazy browser-side prompt helper. It needs a frontend build (`cd frontend && npm install && npm run build`) before `webui_pro.py` will serve it.

All three read and write the same model folders, history, and settings, so switching between them is safe. Studio remains the one to recommend for new users until Pro reaches parity.

## Release Gate

Current focus: image generation, inpainting, video generation, and video-audio post-processing must be reliable before the project takes on more feature work.

- New user-facing features are paused until those paths pass local smoke tests.
- Optimization work is allowed when it improves an existing path and has a fallback.
- Benchmark claims need timing receipts from this repo, not upstream marketing numbers.
- Optional engines, model weights, SDKs, and generated outputs stay local and are not committed.

## What Works On Main

### Image Generation

- txt2img, img2img, and inpaint
- Stable Diffusion 1.5, SDXL, SD3.5, and Flux txt2img checkpoint loading through Diffusers
- sampler, scheduler, steps, CFG, seed, size, VAE, clip skip, and hires fix controls
- live preview, interrupt, continuous generation, and job history
- prompt styles, wildcards, prompt files, dynamic prompt syntax, and Compel support
- LoRA selection, keyword expansion, saved aliases/strengths, and runtime adapter loading for supported Diffusers image families
- PNG metadata and PNG Info import back into the Image tab

### Pro UI And Monitoring

- FastAPI + React/TypeScript/Vite app shell with left-rail navigation
- Create, Models, Data, Monitor, Logs, and Settings workspaces
- preset 1 cream/navy/coral visual direction, with dark planned as preset 2
- scroll-safe panels, popup tool windows, and resizable workspace columns
- runtime monitor for backend state, queue health, logs, resources, and recent receipts
- browser-side Transformers.js prompt helper loaded only when **Analyze prompt** is clicked
- Pro API endpoints for runtime, bootstrap, generation, data, logs, and settings

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
- SD3.5 Diffusers-folder checkpoints are supported in `models/Stable-diffusion/`
- Flux split-model txt2img is supported from `models/flux/GGUF/` or `models/flux/UNet/` with local CLIP-L, T5-XXL, and `ae.safetensors`
- model aliases and trigger-word helpers
- curated download entries for common local model folders
- import helpers for model folders from another local install

### LoRA Status

- SD/SDXL/SD3.5-style runtime LoRA loading is wired through Diffusers adapter APIs.
- Studio includes LoRA prompt insertion, LoRA stack composition, trigger-word helpers, aliases, and saved default strengths.
- Model Manager includes LoRA metadata controls and a LoRA fuse worker for supported Diffusers exports.
- Wan supports stage LoRAs for supported 5B and high/low transformer routes, with runtime-aware filtering.
- Flux/new transformer-image LoRA and ONNX LoRA are intentionally blocked until their pipeline-specific appliers are implemented and tested.

### Enhance

- image upscale
- GFPGAN / CodeFormer-style restoration when models are installed
- old-photo restore pipeline
- tiled upscale controls for local VRAM limits

### Segment

- SAM mask generation when SAM weights are installed
- text-guided boxes through GroundingDINO when the optional dependency is available

### Image Lab

- maturity matrix tracking each image route against the AUTOMATIC1111 parity baseline (`docs/IMAGE_MATURITY_MATRIX.md`)
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

Recent local receipts:

- [Wan post-driver benchmark, June 20 2026](docs/benchmark-reports/wan-post-driver-20260620.md)
- [Fluxtrait Flux.2 Klein / Z-Image settings check](docs/benchmark-reports/fluxtrait-settings-check-20260620.md)

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

Run the classic Studio UI:

```bat
webui.bat
```

Or:

```powershell
python launch.py
```

Other UI entry points, see [UI Rebuild](#ui-rebuild):

```powershell
python webui_modern.py   # Modern Gradio shell
python webui_pro.py      # Pro FastAPI + React frontend (build frontend/ first)
```

Optional local speed/settings logging:

```powershell
python launch.py --genlog
```

`--genlog` writes JSONL entries to `outputs/genlog/generation-log.jsonl` for
SD, SDXL, and Wan runs. It records timings, runtime route/pipeline, settings,
models, and LoRAs, but not prompt text. The flag is off by default.

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

No hard links or junctions are required. To reuse an existing A1111, ComfyUI,
or shared model library, open **Settings -> Model paths** and add the folders as
extra scan roots. Optional SDK/app paths, such as NVIDIA VideoFX executables,
live under **Settings -> Engines & pipelines -> External tool paths**.

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

## LTX 2.3 Video Setup

LTX 2.3 is optional and runs in `engines/ltx/.venv`, not the main Studio venv.
Install or repair the engine from **Settings -> Engines & pipelines**, or run:

```powershell
.\scripts\bootstrap_ltx.ps1 -Enable
```

The default LTX route expects:

```text
models/ltx/checkpoints/ltx-2.3-22b-distilled-1.1.safetensors
models/ltx/upscalers/ltx-2.3-spatial-upscaler-x2-1.1.safetensors
models/ltx/text_encoder/gemma-3-12b-it-qat-q4_0-unquantized/
```

Use the Models tab's **Video LTX 2.3** quick-start bundle to download those
assets when Hugging Face access is configured.

## Flux Image Setup

The current Flux route is text-to-image only. It supports Flux transformer files in `.gguf` or `.safetensors` form, plus local split text/VAE assets:

```text
models/flux/GGUF/          flux transformer .gguf files
models/flux/UNet/          flux transformer .safetensors files
models/flux/Textencoder/   clip_l.safetensors and t5xxl_fp16.safetensors
models/flux/VAE/           ae.safetensors
```

Distilled/4-step Flux models without a guidance block run with CFG forced to `0.0`. Full Flux-dev style models with guidance tensors can use the CFG control. Flux LoRA, ControlNet, img2img, and inpaint are intentionally blocked until those paths are wired and tested.

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
- `frontend/` is the React/TypeScript/Vite source for the Pro UI; build it with `npm install && npm run build` to populate `frontend/dist`, which `webui_pro.py` serves.
- `docs/`, `tests/`, and `scripts/` are part of the public maintainability story.
- runtime data such as models, outputs, local configs, and agent notes are ignored.

Useful project docs:

- `ARCHITECTURE.md`
- `CONTRIBUTING.md`
- `docs/ATTRIBUTION.md`
- `docs/DEPENDENCY_POLICY.md`
- `docs/ENGINE_ISOLATION.md`
- `docs/MAINTAINER_NOTES.md`
- `docs/PATH_CONFIGURATION.md`
- `docs/TRAINING_ENGINE_ROADMAP.md`

## License And Third-party Status

This is a practical release checklist, not legal advice.

- AIWF Studio's own code is licensed under the **AIWF Non-Commercial Attribution License v1.0** in `LICENSE`.
- Non-commercial use, study, modification, and sharing are allowed with attribution.
- Commercial use requires separate written permission from the project owner. Contact: https://github.com/nawnie
- Model weights, generated outputs, NVIDIA SDK binaries, MMAudio checkout files, and large engine repos are local-only and ignored by git.
- Users are responsible for the licenses of checkpoints, LoRAs, VAEs, ControlNet models, SAM weights, Wan files, and audio models they install.
- Stable Diffusion 3.5 model weights are released under Stability AI's Community License and may require Hugging Face gate acceptance before download.
- NVIDIA Video Effects / VFX SDK support is optional. AIWF does not vendor or redistribute NVIDIA SDK binaries or models.
- MMAudio checkpoints are CC-BY-NC 4.0. Do not present MMAudio-backed audio as commercial-safe without separate permission.
- InsightFace code is MIT, but InsightFace-trained models and the inswapper face-swap model require separate license care for non-local or commercial use. Face swapping must only be used with consent and applicable-law compliance.
- Segment Anything is Apache-2.0; AIWF's segment/inpaint path is clean-room integration, with attribution kept in `docs/ATTRIBUTION.md`.

Keep optional restricted components clearly marked as local/user-installed.

## SageAttention And SDK Cache

SageAttention is a candidate Wan/video optimization, and the upstream project is Apache-2.0. It belongs in a shared local SDK/cache folder as a future accelerator reference and disposable test lane, not as a required runtime dependency yet.

Current rule for `main`: do not wire SageAttention as a required path until a copied-venv test proves installability, output quality, and speed on this Windows/NVIDIA setup. Wan should keep working through the existing torch SDPA fallback when SageAttention is missing.

## WIP And Help Wanted

These areas exist as work-in-progress or need more hardware coverage before they should be treated as stable:

- Pro UI (React/Vite frontend) feature parity with Studio
- Wan FP8 high/low video speed path
- Wan resident / streamed offload modes
- training engines (Chat and Training tabs are hidden by default in Studio/Modern)
- Ollama or llama.cpp chat workspace
- Face Swap tab
- workflow authoring
- model conversion and quantization tools
- plugin system
- richer generated-audio controls and model installers
- AMD, Intel, Linux, and lower-VRAM validation
- installer polish and first-run onboarding

## Credits

AIWF Studio is clean-room code. It draws from established local AI tooling around Stable Diffusion, Diffusers, ControlNet, Segment Anything, GroundingDINO, Real-ESRGAN, GFPGAN, CodeFormer, Wan, ComfyUI-GGUF, and the AUTOMATIC1111 web UI.

Optional video post-processing can use NVIDIA Video Effects / VFX SDK components for RTX VSR-style upscale, cleanup, AI green screen, and relighting when the user installs the NVIDIA SDK locally. See `docs/ATTRIBUTION.md` for third-party credits and source links.

## Contributing

AIWF Studio needs focused help from people who work on local creative AI, Windows/NVIDIA workflows, Python services, frontend rebuilds, model runtime tooling, and install flows for normal PC users. Open an issue with a narrow repro or send a PR that includes the check you ran.
