# AIWF Studio Second GUI WIP Map

This document tracks the screenshot-driven second GUI shell added for Windows preview work.

## Purpose

The second GUI is a fast visual shell for the newer AIWF Studio layout: left rail, image/video/inpaint/model/data tabs, prompt controls, canvas preview, runtime status, loaded model card, queue, and recent outputs.

It is intentionally honest about unfinished work. A button that does not have a real backend route returns a visible `WIP` toast or JSON payload instead of silently failing.

## Launch

From the repository root:

```bat
second_gui.bat
```

PowerShell:

```powershell
.\scripts\run_second_gui.ps1
```

Run with the main AIWF backend at `http://127.0.0.1:7860` and allow the Second GUI bridge to call the backend:

```powershell
.\scripts\run_second_gui.ps1 -Backend http://127.0.0.1:7860 -Proxy
```

Force a specific generation adapter when testing route parity:

```powershell
.\scripts\run_second_gui.ps1 -Backend http://127.0.0.1:7860 -Proxy -ProxyMode native
.\scripts\run_second_gui.ps1 -Backend http://127.0.0.1:7860 -Proxy -ProxyMode sdapi
```

Equivalent batch form:

```bat
second_gui.bat --backend http://127.0.0.1:7860 --proxy --proxy-mode auto
```

## Local server routes

| Route | Status | Notes |
| --- | --- | --- |
| `/` | Wired | Serves `static/second_gui/index.html`. |
| `/api/runtime/status` | Wired bridge | Probes backend health, progress, memory, options, and optimization status when available. |
| `/api/features` | Wired shell | Lists preview/WIP feature states. |
| `/api/catalog` | Wired bridge | Pulls models/samplers from native `/api/v1/*` and falls back to A1111 `/sdapi/v1/*` routes. |
| `/api/progress` | Wired bridge | Uses native `/api/v1/progress` first, then `/sdapi/v1/progress`. |
| `/api/generate` | WIP by default; wired with `--proxy` | In `auto` mode, tries native `/api/v1/txt2img`, then `/sdapi/v1/txt2img`. |
| `/api/interrupt` | WIP by default; wired with `--proxy` | Tries native `/api/v1/interrupt`, then `/sdapi/v1/interrupt`. |
| `/api/wip` | Wired | Standard WIP response for unfinished UI controls. |

## Current feature mapping

| UI area | Current behavior | Next real route/service |
| --- | --- | --- |
| Image tab | Full visual shell; Generate posts to `/api/generate`; optional backend proxy works when backend is running. | Add richer request controls: LoRA, hires fix, scheduler, refiner, ControlNet. |
| Model dropdown | Wired to `/api/catalog` when backend is available. | Add load/unload/import controls once backend exposes safe model lifecycle operations. |
| Sampler dropdown | Wired to `/api/catalog` when backend is available. | Add scheduler pairing and backend validation hints. |
| Runtime status | Backend reachability, progress, RAM/memory, loaded model, optimization profile where available. | Add true VRAM telemetry once the backend exposes CUDA memory totals. |
| Queue | Uses progress state now. | Add multi-job queue list from `/api/v1/jobs`. |
| Video tab | WIP toast. | Wire to Wan/LTX route selector after backend smoke tests. |
| Inpaint tab | WIP toast. | Reuse existing `/api/v1/inpaint` service request model with canvas image/mask capture. |
| Data tab | WIP toast. | Connect dataset/reference/library views. |
| Batch / Workflows / Logs | Hidden until advanced tabs are enabled; WIP toast. | Mount stable advanced panels one at a time. |
| Recent outputs | Static CSS thumbnails plus generated proxy images when available. | Load real output history from library/history service. |

## Guardrail

Do not make the second GUI claim that Video, Inpaint, Data, Batch, Workflows, Logs, installers, full telemetry, or model lifecycle controls are functional until their backend routes are wired and tested. Keep the UI clickable, but keep unfinished routes visibly marked as WIP.
