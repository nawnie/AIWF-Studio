# AIWF Studio v5 changelog

## Studio-first scope

This root overlay keeps the original Gradio Studio as the production surface. It does not copy or overwrite `frontend/`, `aiwf/web/modern/`, or Pro UI source. Useful interaction ideas from the other GUI tracks were absorbed into Studio without importing their separate narratives or build chains.

## Unified lab interaction

Image Lab, Video Lab, and Audio Lab now share one interaction contract:

1. choose a starting preset or Custom;
2. select the processes to run;
3. reveal settings only for selected processes;
4. display the canonical resolved order;
5. build a machine-readable plan;
6. run and save a reproducible manifest.

Presets are editable starting points, not locked recipes.

## Image and masking

- Added Image Lab Workflow with selected-stage execution.
- Added deterministic denoise, tone, resize, and export stages.
- Connected existing Segment, generation/inpaint, restoration, and upscaling services.
- Reworked Segment presets into complete parameter profiles.
- Added independent edge feathering.

## Video

- Reworked Video Lab around selectable stage cards.
- Replaced boolean-only restoration stages with editable FFmpeg parameters for deinterlace, stabilization, deflicker, denoise, sharpen, resize, audio cleanup, loudness, and export.
- Preserved FFmpeg capability detection, codec fallback, cancellation, atomic publish, logs, and `job.json`.
- Retained chunked RIFE and v4 backend/load fixes.

## Audio

- Added an isolated optional Audio Lab environment.
- Added deterministic signal-chain processing and MIDI metadata inspection.
- Preserved existing audio generation.
- Added a deliberately limited DAW-command planner for measure transposition and unison orchestration examples.

## Inherited backend fixes

The v4 transactional updater remains included because it carries the reviewed bbb1cae GPU ownership, model inventory, Flux/Flux.2 Klein/Z-Image, Wan, prompt-cache, loading, warmup, and smoke-test corrections.
