# AIWF Studio for Windows: native app design (MVP)

Status: MVP built and running, 2026-10-07. Owner: Claude (session f4defb94). Local only.

## 0. Where the MVP stands (2026-10-07)

- **Builds with VS 2022 Build Tools alone** (`build.ps1`, MSVC v143, Windows App SDK 1.7 self-contained). Build Tools lack the UWP C++ component that normally runs the XAML compiler from a C++ project; `AIWF.Desktop/DesktopBuildHooks.targets` schedules the Windows App SDK's own XAML passes instead (see its header).
- **Seven pages working against live engines**: Home (NVML + per-engine VRAM, engine start/stop), Chat (streaming, 38 tokens/s on qwen3-30b-a3b; context-backed questions are recorded in local project history), Create (Qwen Image 2.1, about 40 s per draft image), Datasets, Train (weights, download/key dialog, import, dry-run plan), Projects, Settings.
- **Engine API host**: the native app does not need Pro's web server. `python -m aiwf.engine_api` serves the same `/api/pro/unified/*` router on 127.0.0.1:7870 (no torch/diffusers import, about 1 s to start); the app starts it automatically. Image jobs were added to that shared API (`generate_image`, `image_status`, `cancel_image`), so the CLI and MCP got them too.
- **Measured RAM** (`docs/ram-receipt.json`): native shell 126-150 MB private in 1 process; the same React UI in an Edge app window 718 MB private in 13 processes.
- Per-user installer: `install.ps1`; external process startup remains configurable through the per-user engine manifest. Demo script: `docs/NVIDIA_INCEPTION_DEMO.md`. Report: `F:\UserFolders\Desktop\04 Reports & Results\aiwf-native-mvp-2026-10-07\`.
- Current: a per-user PowerShell installer, project-root discovery for the engine API, and a portable shipped engine manifest. Optional external service process paths still belong in the per-user `engines.json` override.
- Not yet: in-app training start, chat Markdown rendering, and automatic discovery of external engine executables.

## 1. Product in one sentence

AIWF Studio is a local creative and AI workstation: create and refine media, chat with local models, work with audio, organize projects and datasets, and prepare model training through one project-aware experience over modular local engines.

Studio already has working Chat and Audio experiences. This document describes the native Windows MVP; its page list is not the full Studio product inventory. The native MVP currently documents Chat, while Audio is a Studio-wide capability whose native page placement is not specified here.

## 1.1 Product positioning and UX differentiation

### Direction

Position Studio around **confidence in local creative work across engines and media**, not around owning the most elaborate canvas or node graph. Invoke already presents a layer-based canvas and visual workflows; ComfyUI's core strength is its flexible node-based engine. Studio should make local capabilities easier to choose, run, understand, and return to across image creation, Chat, Audio, and the other supported workspaces.

Product promise: **Studio helps people do useful creative and AI work on their own machine, shows what their setup can run, and keeps the work understandable and recoverable.** This is a design direction, not a claim that every capability below is already complete.

### Experience principles to build toward

1. **Show a working route before asking users to configure one.** Use the machine's detected hardware, available engines, installed models, and route support to present valid choices. Explain unavailable choices in plain language and offer the next useful action.
2. **Make local readiness legible.** Present engine and model status, resource needs, progress, and actionable errors where the user is deciding what to run. Keep GPU and VRAM details available without making them the first thing every creator must understand.
3. **Keep a project coherent across workspaces.** Chat, Audio, image and video work, datasets, outputs, and project history should feel connected by the selected project and make it easy to find the inputs and results of prior work.
4. **Offer a clear path from guided task to inspectable workflow.** Keep common actions direct. Let users inspect or adjust the route and settings when they want more control; do not require graph editing for routine tasks.
5. **Make recovery and repeat work part of the flow.** Preserve the relationship between a result and its project, model, engine, settings, and status. Explain failures and provide safe retry or recovery actions where supported.
6. **Keep local control visible.** Be clear about where models and outputs live, which engines are running, and when an action needs network access or a large download.

### What not to compete on

- Do not present a canvas or node graph as Studio's unique advantage; both are established parts of the alternatives.
- Do not use a larger feature checklist or raw model count as the main product story.
- Do not conceal route limits or maturity differences behind a simple Generate button. A straightforward flow should still tell users what will run and what happened.

### How to tell whether the direction is working

- Measure time to a first successful task for each supported workspace.
- Track whether preflight failures explain the cause and lead to a successful next action.
- Check whether users can return to a project and understand or repeat a previous result.

These are proposed product measures; this document does not claim they are currently instrumented.

## 2. Why native, and what each surface is for

| Surface | Role | Why |
|---|---|---|
| **AIWF Studio for Windows** (this app, C++/WinRT + WinUI 3) | The product people install and use | Real Windows app: Fluent controls, Mica, system theme, accessibility, no browser engine. A browser tab costs hundreds of MB; RAM and VRAM belong to the models. |
| React "Pro" web UI | Developer and QA surface | Fast iteration, Playwright tests, same API. Not the shipping UI. |
| Gradio lab | Server-side product | Remote/server deployments. |
| MCP server + CLI | Agents and automation | Same operations, generated from the same contract. |

All four talk to the **same local engine API** (`/api/pro/unified/*`, defined once in `aiwf/services/unified_contract.py`). The native app adds nothing the API lacks; anything new goes into the API first, so every surface gets it.

## 3. Architecture

```
AIWF Studio for Windows (C++/WinRT, WinUI 3, Windows App SDK 1.7, MSVC v143)
 |-- Engine supervisor (C++): starts/stops engines with CreateProcess + a Job Object
 |     -> closing the app ends every engine it started; nothing is left running
 |-- GPU monitor (C++): NVML from the NVIDIA driver (utilization, VRAM, temperature, power,
 |     and which engine process holds how much VRAM)
 '-- Engine client (C++): loopback HTTP + JSON to the engine API
        |
        v  127.0.0.1 only
AIWF engine API  (aiwf/engine_api.py on :7870, or inside AIWF Studio Pro on :7860)  /api/pro/unified/*
 |-- Dataset Studio  (catalog, immutable text packages)        127.0.0.1:8796
 |-- ReTrain          (package import, dry-run plans, weights)   127.12.6.3:8787
 |-- Qwen Chat        (llama.cpp router + gateway + tools)       127.0.0.1:8080
 '-- ComfyUI          (Qwen Image 2.1 generation)                127.0.0.1:8188
```

The engines stay separate processes (each has its own Python/CUDA environment), but the user never meets them as separate programs: they appear as **engines** with one status light each.

## 4. UX principles

1. **Tasks, not programs.** Navigation says Create, Datasets, Train, Chat, not "Dataset Studio" or "ReTrain".
2. **One project in focus.** The project picker in the title area scopes every page; every action lands in that project's history.
3. **Engines start themselves.** Opening Chat with the chat engine off offers one button: Start. Home shows all engines and has Start all.
4. **The GPU is visible.** A live VRAM bar shows which engine holds memory, with a one-click way to free it. Trust comes from seeing that work happens here.
5. **Nothing surprising.** Downloads, key entry and anything that uses the network or many GB ask first, and show size and progress with a real Cancel.
6. **Native behavior.** System theme and accent, Mica, keyboard navigation, screen-reader names, standard dialogs, remembered window size.

## 5. Information architecture (MVP pages)

| Page | Purpose | Key elements |
|---|---|---|
| **Home** | Workstation at a glance | GPU card (name, VRAM bar split by engine, utilization, temperature, power); engine list with status and Start/Stop; Start all; current project and recent activity |
| **Chat** | Talk to local models | Model picker (loaded models first), streaming replies, Stop, New chat, "Include project context" toggle |
| **Create** | Generate images | Prompt, size, Generate, live status, result gallery with Open and Show in folder |
| **Datasets** | Training data | Published text packages (name, revision, rows); catalog recent Studio images into the project |
| **Train** | Prepare fine-tuning | Base models with on-PC / downloadable / needs-key states; download dialog with size, progress and Cancel; Hugging Face key dialog; import a package revision; dry-run plan with gates and VRAM estimate |
| **Projects** | Organize work | Create/select projects; activity timeline |
| **Settings** | Engines and app | Engine endpoints and status, app memory use, about |

## 6. MVP scope (for the NVIDIA Inception demo)

In: the seven pages above; engine supervisor with Job Object; NVML telemetry including per-engine VRAM; streaming chat; Qwen Image 2.1 generation; model weights download/key flow; package import and dry-run plan; projects. Measured RAM of the native shell vs. the web UI in a browser.

Out of the native MVP: training start/monitoring UI, bringing every Studio-wide workspace into the native shell, installer (MSIX), auto-update, crash reporting, telemetry opt-in, localization. Studio-wide Chat and Audio already exist; this boundary describes native MVP scope, not a claim that those capabilities are absent from Studio.

## 7. NVIDIA technology on show

CUDA 13 engines; int8/NVFP4 quantized diffusion (ComfyUI kitchen kernels); llama.cpp CUDA inference; NVML live telemetry with per-process VRAM; RTX Video (VideoFX) in the Studio engine; QLoRA fine-tuning plans sized against the actual card's VRAM.

## 8. Risks and decisions

- Windows App SDK 1.7 is used because 1.8/2.x split into many packages that C++ `packages.config` projects must list by hand; moving to 2.x later is a project-file change.
- Engines keep their own environments; the supervisor reads a JSON engine manifest so paths are configuration, not code.
- The React UI stays as the developer surface and test harness for the same API.

## 9. Competitive references

- [Invoke](https://invoke.ai/) and its [Canvas guide](https://invoke-ai.github.io/InvokeAI-7/users-guide/canvas/introduction/) describe its canvas and workflow strengths.
- [ComfyUI documentation](https://docs.comfy.org/essentials/core-concepts/links) describes its node-based interface and customizable generation engine.
- Studio context: [README](../README.md) and [feature inventory](../docs/FEATURES.md). These describe different Studio surfaces and maturity levels; verify the current implementation before turning a design direction into a capability claim.
