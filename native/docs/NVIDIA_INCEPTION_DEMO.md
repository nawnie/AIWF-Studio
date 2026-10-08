# AIWF Studio for Windows: NVIDIA Inception demo script

Status: MVP, 2026-10-07. Everything below was run on the demo PC (RTX 4070 Ti SUPER 16 GB, driver 616.92, CUDA 13.4). Numbers come from `ram-receipt.json` and the runs listed under "Measured".

## The pitch in three sentences

AIWF Studio turns an RTX PC into a private generative-AI workstation: chat with local LLMs, generate images, curate training data and plan fine-tunes, all on the user's own GPU. It is a real Windows app (C++/WinRT, WinUI 3, Windows App SDK) on top of CUDA engines (llama.cpp, ComfyUI with int8 diffusion, PyTorch fine-tuning), and nothing leaves the PC. The app shows exactly what the GPU is doing: which engine holds how much video memory, live.

## Before the meeting (10 minutes)

1. Open **AIWF Studio for Windows** from the desktop. The engine API starts by itself.
2. Home: press **Start all engines**. Wait until every engine row is Running (the image engine takes about a minute).
3. Pick the demo project in the title bar (or create "NVIDIA demo" on Projects).
4. Chat: send one short message so the model is in GPU memory (first token then arrives in about 2 s).
5. Create: generate one draft image so the diffusion weights are loaded (the first image after a restart loads about 20 GB from disk).
6. Close any other GPU-heavy app. Keep Home visible.

## Live flow (about 8 minutes)

| Step | Show | Say |
|---|---|---|
| 1. Home | GPU card and the "Who is using video memory" bar | "This is NVML from the driver plus Windows' per-process GPU counters. Each colour is one engine. We see what the hardware is doing, not a guess." |
| 2. Home | Engine rows: Running, VRAM per engine, Stop | "Engines are separate CUDA processes. The app starts them in a Job Object, so closing the app leaves nothing running. Engines started by other tools are shown but never killed." |
| 3. Chat | Send: "In two sentences, what does QLoRA change compared with full fine-tuning?" | "Streaming from llama.cpp on the GPU. The line under the answer is llama.cpp's own measurement: tokens per second and time to first token." |
| 4. Chat | Toggle **Project context**, ask "What has this project done so far?" | "Context is a card of IDs and counts only. It is sent only when you press Send." |
| 5. Create | Prompt: "A red fox standing in fresh snow at sunrise, soft golden light"; Generate | "Qwen Image 2.1, int8 weights in ComfyUI. Watch the VRAM bar on Home while it runs." |
| 6. Datasets | Select the new image, **Catalog** | "The image lands in the dataset engine with the project tag; its prompt becomes the caption." |
| 7. Train | Base models: on this PC / download size / needs key | "Gated models ask for the user's Hugging Face key; downloads ask first and show size, progress and Cancel." |
| 8. Train | Pick a published text package revision, Import, **Check plan** | "A dry run against this exact GPU: gates, dependencies and a VRAM estimate. Training starts in the training engine, never by surprise." |
| 9. Settings | "This window uses ... MB" | "The whole native shell is one process using about 150 MB private memory. The same UI in a browser window costs about 720 MB private across 13 processes." |

## Measured (2026-10-07, this PC)

| What | Result |
|---|---|
| Chat, qwen3-30b-a3b-instruct-2507 Q3_K_XL (llama.cpp CUDA) | 38.0 tokens/s, first token after 1.8 s, 73-token answer |
| Image, Qwen Image 2.1 int8, draft 0.6 MP, 30 steps | 38 s (1056 x 592) and 40 s (800 x 800), weights already loaded |
| Native shell RAM (Release, idle on Home) | 148 MB working set, 126 MB private, 1 process |
| Native shell RAM after visiting all 7 pages | 180 MB working set, 150 MB private |
| Same React UI in an Edge app window (warm profile) | 1,070 MB working set, 718 MB private, 13 processes (renderers alone 431 MB private) |
| Build | `build.ps1`: VS 2022 Build Tools only (MSVC v143, Windows App SDK 1.7 self-contained); exe 1.8 MB, folder 145 MB with the runtime |

Caveats: chat speed depends on the model and quantization; image time excludes the first-run weight load; the browser numbers are a dedicated Edge app window, so a tab in an already-open browser costs less than the full instance (closer to the renderer figure).

## NVIDIA technology on screen

- **CUDA** in every engine: llama.cpp (CUDA 13.3 build), ComfyUI (int8 diffusion kernels), PyTorch for fine-tuning.
- **NVML** for live device telemetry: memory, load, clocks, temperature, power and power limit.
- **RTX VRAM accounting** per engine from Windows GPU process counters, matched to the NVIDIA adapter.
- **QLoRA planning** sized against the actual card's memory before anything runs.

## What is real today and what is next

Real: native shell with seven pages; engine supervisor with Job Objects; NVML telemetry and per-engine VRAM; streaming chat with live speed; Qwen Image 2.1 generation through the shared engine API (also available to the CLI and MCP agents); dataset catalog; model weights with download/key flow; package import and dry-run plan UI. The demo workstation currently has no published Dataset Studio package, so the package import/plan path is verified with isolated synthetic fixtures only.

Next: starting and monitoring training runs in-app; MSIX packaging; automatic discovery of external engine executables; Markdown rendering in chat; image editing; video.

The Train screen reads published text packages from Dataset Studio. When the demo PC has no package, use the synthetic-only Studio Flow fixture in `tests/individual_tests/test_unified_bridge.py` and the Playwright Studio Flow fixture; they mock Dataset Studio and ReTrain in memory and never write to the real catalog. Do not catalog a test image or publish a demo package into the user's Dataset Studio project.

Install the native shell with `powershell -NoProfile -ExecutionPolicy Bypass -File native\install.ps1`. This installs per-user into `native\installed` under the selected AIWF Studio checkout and creates a Desktop shortcut. The shipped manifest has no machine-specific external executable paths; already-running loopback services can be detected, and startup paths for external apps are a per-user override in `%LOCALAPPDATA%\AIWF Studio\engines.json`.

## If something goes wrong

- An engine row shows **Problem**: Settings > that engine > **Open log**. Start it again from Home.
- Chat says the model returned no text: pick a model marked loaded, or wait for the first load.
- The image takes much longer: it is loading weights after a restart; the second image is fast.
- Everything stops when the app closes by design; reopen and Start all.
