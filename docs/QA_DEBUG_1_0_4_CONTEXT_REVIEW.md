# AIWF Studio QA Debug Review - Grok Build 1.0.4 / Memory Layer / sd.cpp

Date: 2026-08-18
Branch: `sdcpp-qa-debug-1.0.4`

## Review sources

- Context7 xAI docs: current official examples center on `grok-4.5` via the xAI SDK, OpenAI-compatible Responses API, Chat Completions, streaming, function tools, web/X/code tools, and long reasoning timeouts.
- Context7 Mnemosyne docs: memory server shape is MCP-friendly, SQLite-backed, and exposes `remember`, `recall`, and graph traversal patterns.
- xAI Grok Build changelog: `xai-org/grok-build` `xai-grok-shell` 1.0.4 and 1.0.5.
- stable-diffusion.cpp docs: sd-cli is active-development CLI/API, supports SD/Flux/Wan/Qwen/Z-Image family routes, CUDA builds through CMake `-DSD_CUDA=ON`, LoRA directories, split asset args, video generation, and backend/offload controls.

## Grok Build 1.0.4 compatibility notes

Grok Build 1.0.4 added or changed several behaviors that matter for AIWF debugging workflows:

- `StopCancelled` hook event reports turns that end without completion.
- Tool commands and MCP servers receive `GROK_SESSION_ID`.
- PreToolUse hooks can rewrite tool input before execution.
- Subagent lifecycle events are preserved even when delivered out of order.
- Finished subagent transcripts are evicted from memory and rebuilt from disk when reopened.
- Windows fixes landed for `grok du` and worktree commands when only `USERPROFILE` is set.

AIWF implication:

- Keep launcher state and test receipts file-backed, not only process-memory-backed.
- Do not assume hook event order during Grok-driven debugging.
- Capture `GROK_SESSION_ID` when present if future AIWF/Grok integration logs agent work.
- Do not rely on hook stdout for all clients; treat handoff/memory retrieval as explicit tool or file operations.

## Grok Build 1.0.5 follow-up notes

1.0.5 adds `GROK_CONFIG` / `GROK_CONFIG_PATH`, worktree cleanup safeguards, clearer hook-block messages, MCP spinner text improvements, reasoning effort passed via ACP, and Windows home-dir fixes for agent skill discovery.

AIWF implication:

- Any future Grok launcher should prefer env/config overrides over editing global config.
- Keep AIWF-generated worktrees separate from user primary checkout.
- Do not hard-code `~/.grok`; resolve through env where available.

## Memory layer requirements

The current memory layer docs favor:

- MCP server configuration via command + args.
- SQLite local storage.
- Hybrid recall: vector, full-text, importance, filters, and temporal weighting.
- Graph traversal from a seed memory.

AIWF implication:

- Store sd.cpp QA results as deterministic receipts first.
- Later, send summaries to memory as short observed facts, not raw logs.
- Preserve artifact paths, backend profile, sd-cli command, model id/path, and failure class so recalls can answer “what broke last time?” without re-reading entire logs.

## stable-diffusion.cpp QA requirements

The sd.cpp lane should remain a backend profile, not a live hot-swap:

```text
Pro UI -> Pro API -> GenerationService -> selected backend at boot
```

Backend choices:

- `diffusers`: broad AIWF model-family support.
- `sdcpp`: sd-cli subprocess fallback/speed lane.
- `onnx`: existing ONNX route.

Must smoke test in this order:

1. Diffusers baseline still launches.
2. SD1.5 txt2img through sd.cpp.
3. SDXL txt2img through sd.cpp.
4. Cancel behavior.
5. Output save/history metadata.
6. Inpaint with image + mask.
7. LoRA through `--lora-model-dir`.
8. Split assets for Flux/Qwen-style models.
9. Video only after image routes are stable.

## UI requirements

- No mint or teal in new sd.cpp UI surfaces.
- Backend setting must appear in the native React Settings dropdown.
- Launch path must run the frontend source updater before Pro starts so users do not manually patch after a pull.
- sd.cpp detailed settings live in `/api/ext/sdcpp-pipeline/ui` until the full React settings panel gets deeper form wiring.

## Local test commands

```powershell
git fetch origin
git checkout sdcpp-qa-debug-1.0.4
git pull origin sdcpp-qa-debug-1.0.4

.\AIWF Studio Pro.bat --terminal
```

Set sd.cpp profile:

```powershell
.\scripts\launch_sdcpp.ps1 `
  -SdCli "F:\tools\stable-diffusion.cpp\bin\sd-cli.exe" `
  -Backend cuda0 `
  -MaxVram 14 `
  -SetDefault `
  -Terminal
```

Open sd.cpp pipeline UI:

```text
http://127.0.0.1:7860/api/ext/sdcpp-pipeline/ui
```

## Known gaps before release packaging

- The connector safety layer blocked committing an executable sd-cli clone/build helper. The local build commands are documented in `docs/SDCPP_LOCAL_BUILD_COMMANDS.md`.
- Runtime sd-cli smoke tests must be run on the Windows/NVIDIA machine.
- Video should remain QA-gated until image + split-asset routes are stable.

Logs first. Panic later.
