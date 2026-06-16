# Engine Isolation

AIWF Studio is moving to a two-tier runtime:

1. **Stable core venv**
   - Gradio UI
   - AppContext and typed domain models
   - API routes
   - settings/model catalogs
   - stable Diffusers image reference path
   - ProcessSupervisor and worker orchestration

2. **Optional worker venvs**
   - `engines/wan/.venv` for volatile video runtime packages
   - `engines/kohya/.venv` for LoRA training stacks
   - `engines/ed2/.venv` for full fine-tuning stacks
   - future engines can be added without poisoning the core UI venv

The core app must boot even when every worker venv is missing. Missing engines
are setup/status problems, not app startup failures.

## Worker Contract

Workers are launched as subprocesses:

```text
<engine-python> <engine-worker.py> <request.json>
```

The worker reads the JSON request and emits JSONL events on stdout. See
`docs/WORKER_PROTOCOL.md` for event types.

## Tenant Resolution

`aiwf.services.worker_tenant.WorkerTenantRegistry` resolves:

- engine enabled state from `engines.json`
- engine venv python path
- worker script path
- optional upstream repo/entry script
- readiness messages
- `WorkerCommand` for `ProcessSupervisor`

This keeps venv selection out of UI callbacks and engine-specific services.

## Release Modes

**Stable tutorial build:** core venv only. Worker engines can be absent.

**Developer build:** selected workers enabled in `engines.json` and installed
under `engines/<name>/.venv`.

## Wan Direction

Diffusers remains the reference/default method. Wan-specific experimental
methods move toward explicit backend choices:

- `diffusers_reference`
- `aiwf_fp8`
- `aiwf_gguf`
- `comfy_engine`

Benchmark receipts must identify the method, model pair, text encoder, LoRAs,
resolution, frames, steps, elapsed time, and output path before speed claims are
made.
