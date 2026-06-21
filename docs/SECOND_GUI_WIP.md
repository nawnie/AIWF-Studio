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

Run with the main AIWF backend at `http://127.0.0.1:7860` and attempt to proxy Generate through the existing A1111-style adapter:

```powershell
.\scripts\run_second_gui.ps1 -Backend http://127.0.0.1:7860 -Proxy
```

Equivalent batch form:

```bat
second_gui.bat --backend http://127.0.0.1:7860 --proxy
```

## Local server routes

| Route | Status | Notes |
| --- | --- | --- |
| `/` | Wired | Serves `static/second_gui/index.html`. |
| `/api/runtime/status` | Wired shell | Probes the main AIWF backend and returns runtime placeholders when unavailable. |
| `/api/features` | Wired shell | Lists preview/WIP feature states. |
| `/api/generate` | WIP by default | Returns WIP unless `--proxy` or `AIWF_SECOND_GUI_PROXY=1` is set. |
| `/api/wip` | Wired | Standard WIP response for unfinished UI controls. |

## Current feature mapping

| UI area | Current behavior | Next real route/service |
| --- | --- | --- |
| Image tab | Full visual shell; Generate posts to `/api/generate`. | Route to native `/api/v1` image generation or A1111 `/sdapi/v1/txt2img` adapter. |
| Video tab | WIP toast. | Wire to Wan/LTX route selector after backend smoke tests. |
| Inpaint tab | WIP toast. | Reuse existing inpaint service request model. |
| Models tab | WIP toast. | Connect model catalog scan, load, unload, import helpers. |
| Data tab | WIP toast. | Connect dataset/reference/library views. |
| Batch / Workflows / Logs | Hidden until advanced tabs are enabled; WIP toast. | Mount stable advanced panels one at a time. |
| Runtime status | Backend reachability probe plus environment placeholders. | Replace placeholders with real runtime telemetry endpoint. |
| Recent outputs | Static CSS thumbnails plus generated proxy images when available. | Load real output history from library/history service. |

## Guardrail

Do not make the second GUI claim that Video, Inpaint, Models, Data, Batch, Workflows, Logs, installers, or full telemetry are functional until their backend routes are wired and tested. Keep the UI clickable, but keep unfinished routes visibly marked as WIP.
