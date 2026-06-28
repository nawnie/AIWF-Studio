# AIWF Studio Feature List

This file is the repo-facing feature inventory for GitHub readers. It separates
working paths from planned or gated paths so the README can stay honest as the
project grows.

## Application Surfaces

- Studio UI: original Gradio workspace launched by `webui.bat` or `python launch.py`.
- Modern UI: cleaner Gradio shell launched by `python webui_modern.py`.
- Pro UI: FastAPI plus React/TypeScript/Vite shell launched by `python webui_pro.py`.
- Shared backend: all UI shells use the same local model folders, outputs, settings, and services.
- Pro API: `/api/pro/runtime`, `/api/pro/bootstrap`, `/api/pro/generate`, `/api/pro/data`, `/api/pro/logs`, and `/api/pro/settings`.
- Compatibility API: native `/api/v1` plus A1111-style `/sdapi/v1` adapter.

## Pro UI Rebuild

- Left-rail app navigation for Create, Models, Data, Monitor, Logs, and Settings.
- Menu bar actions for workspace navigation, layout reset, tool popups, and prompt copy.
- Preset 1 visual direction: cream workspace, navy navigation, teal/coral accents.
- Prompt panel with model, engine, aspect, sampler, seed, size, batch, steps, and CFG controls.
- Prompt helper card using a lazy browser-side Transformers.js model for lightweight prompt structure suggestions.
- Canvas preview with scrollable output stage, action chips, and bottom output dock.
- Model inventory grouped by detected engine family.
- Data workspace for recent output receipts, artifact buckets, observed resolutions, and output root status.
- Logs workspace for runtime status, backend-discovered log files, event rows, and resource snapshots.
- Settings workspace for backend paths, generation defaults, layout memory, and runtime flags.
- Monitor workspace for runtime state, resource meters, queue status, and recent events.
- Scroll-safe panels and modal bodies for long settings and dense controls.

## Image Generation

- txt2img, img2img, and inpaint modes through the Diffusers backend.
- Stable Diffusion 1.5, SDXL, SD3.5, Flux, Flux.2 Klein, Z-Image, Qwen Image, SANA, and related local checkpoint families where wired.
- sampler, scheduler, steps, CFG, seed, size, VAE, clip skip, and hires fix controls where supported by the selected family.
- live preview, interrupt, continuous generation, and job history in Studio.
- prompt styles, wildcards, prompt files, dynamic prompt syntax, and Compel support.
- textual inversion embedding discovery and prompt insertion.
- PNG metadata writing and PNG Info import back into generation controls.
- prompt receipt summaries after generation.

## LoRA And Extra Networks

- local LoRA scanning from dedicated LoRA folders and configured extra roots.
- LoRA metadata inspection, aliases, saved strengths, and trigger-word helpers.
- Studio prompt tools for quick LoRA insertion and multi-slot LoRA stack composition.
- Runtime SD/SDXL/SD3.5-style LoRA application through Diffusers adapter loading.
- Runtime compatibility checks that block mismatched LoRA architecture families.
- LoRA fuse worker for CPU-side Diffusers folder export with one or more selected LoRAs.
- Wan stage LoRAs for supported 5B and high/low transformer routes.
- Wan LoRA filename/runtime filtering so 5B and 14B/A14B LoRAs are not casually mixed.
- Flux and newer transformer-image LoRA application is intentionally blocked until wired and tested.
- ONNX LoRA support is not implemented; it requires model merge/export handling.

## Inpaint, Masking, And Control

- inpaint image/mask editor flow.
- keep-original and last-result source handling.
- outpaint canvas expansion.
- SAM-assisted mask presets when local SAM models are installed.
- single ControlNet unit in the Image advanced panel.
- local ControlNet model discovery and selection.
- lightweight preprocessors where available.
- Control LoRA detection for supported SD1.5 ControlNet LoRA checkpoints.

## Model Management

- local checkpoint, LoRA, VAE, ControlNet, SAM, enhancement, Wan, Flux, LTX, ReActor, and audio model scanning.
- model sorter and header inspection for common checkpoint, LoRA, quantized, and video model formats.
- curated download catalog entries for common local model folders.
- import helpers for model folders from A1111, ComfyUI, and shared libraries.
- model aliases and prompt trigger-word helpers.
- checkpoint blend worker.
- conversion worker for supported single-file to Diffusers-folder exports.
- quantization receipt/export tooling for selected safetensors workflows.

## Enhance And Segment

- image upscale.
- GFPGAN and CodeFormer style restoration when models are installed.
- old-photo restore pipeline.
- tiled upscale controls for local VRAM limits.
- SAM mask generation when SAM weights are installed.
- GroundingDINO text-guided boxes when the optional dependency is available.

## Image Lab

- image route maturity matrix tracking against AUTOMATIC1111 parity expectations.
- XYZ plot runner.
- batch img2img/inpaint runner.
- loopback runner.
- native `GET /api/v1/image/maturity` endpoint.

## Video

- Wan image-to-video through explicit local routes: 5B safetensors, 14B FP8/safetensors, or matched GGUF High Noise plus Low Noise transformer pairs.
- Wan runtime filtering for high/low pair selection, VAE generation, text encoder kind, and LoRA compatibility.
- Wan preflight and route diagnostics before launch.
- optional LTX 2.3 text/image-to-video through an isolated worker engine.
- optional RIFE post-processing to write 30 FPS or 60 FPS output after generation.
- standalone RIFE frame interpolation tab for existing videos.
- optional ReActor post-processing from first key frame, uploaded image, or saved face model.
- optional NVIDIA RTX VSR / Video Effects SDK post-processing when the SDK is installed.
- conservative route selection so incompatible Wan model families are not mixed by accident.

## Audio And Video-Audio

- standalone Audio tab for generating music or sound effects after a video.
- optional video-conditioned audio post-processing through MMAudio in an isolated engine venv.
- generated audio muxing after video when a supported local audio backend is installed.
- audio lab architecture for future multitrack project, waveform, MIDI, and plugin workflows.

## Monitoring, Logs, And Data

- generated-output history and library search over saved outputs.
- JSONL generation logging with `--genlog` for SD, SDXL, and Wan timings without prompt text.
- Pro Data workspace for artifact receipts and output health.
- Pro Logs workspace for runtime rows, event rows, discovered log files, and resource snapshots.
- Pro Monitor workspace for runtime state, resource meters, queue status, and recent events.

## Settings, Paths, And Remote Access

- saved workspace settings.
- launch settings for GPU, network, API, attention, and runtime behavior.
- extra model-library roots and extra checkpoint roots.
- external tool path settings for SDKs and optional engines.
- Tailscale-friendly remote access guidance.
- security warnings for network listening without auth.

## Training And Local AI Development

- training tabs are hidden/opt-in while generation paths are hardened.
- Kohya LoRA training engine status and isolated engine design exist, but training is not a default user path.
- ED2 full fine-tune worker scaffolding exists behind isolated engine setup.
- LLM LoRA/QLoRA/full fine-tune config builders exist for future local assistant work.
- Training must remain human-confirmed; no assistant or UI automation should auto-start training.

## Experimental Or Gated

- Flux LoRA, Flux ControlNet, Flux img2img, and Flux inpaint are blocked until wired and tested.
- ONNX LoRA, textual inversion, and VAE swapping are not implemented.
- TensorRT engine cache, shape ranges, and LoRA refit are Engine Lab only.
- SageAttention and similar acceleration paths remain benchmark-gated.
- Wan resident/streamed offload modes remain under performance validation.
- VQA/image QA helper for Pro UI is planned but not wired yet.
- Plugin ecosystem and workflow authoring are planned.
