# AIWF Studio Architecture

AIWF Studio is a clean-room local AI application built as a stable UI shell with explicit services and isolated heavy engines. The project is not a direct port of legacy Stable Diffusion web UIs. It keeps useful workflows while avoiding implicit global state, callback monkey-patching, and dependency stacks that cannot be inspected.

## System model

AIWF Studio treats local hardware as a constrained runtime system:

- VRAM: the active inference or training tenant owns the current heavy model state.
- RAM: the Gradio app, service layer, session state, catalogs, request models, and workflow metadata stay warm.
- SSD: model weights, outputs, receipts, benchmark logs, prompt libraries, workflows, and cold snapshots live in repo-local folders.

This layout is intentional. Consumer GPUs can run several local AI workflows, but not if every backend imports into one process and assumes it owns VRAM forever.

## Main layers

```text
Gradio UI / FastAPI routes
    -> services
        -> domain models and interfaces
            -> infrastructure backends or subprocess workers
```

Important boundaries:

- `aiwf/bootstrap.py` builds `AppContext`, the dependency container for the app.
- UI code calls services, not raw torch or worker code directly.
- Domain models live under `aiwf/core/domain/`.
- Infrastructure code owns filesystems, torch, diffusers, video, model IO, and engine-specific runtime details.
- Optional engines are launched through workers, not imported at app boot.

## Engine tenants

`EngineTenant` models GPU ownership:

- `IMAGE`
- `VIDEO`
- `CHAT`
- `LORA_TRAINING`
- `FULL_TRAINING`
- `ENHANCE`
- `IDLE`

The core rule is simple: one GPU-heavy tenant at a time. Chat is treated differently because Ollama/GGUF text runtimes manage their own memory lifecycle, but switching from chat to a GPU-heavy tenant must unload the active chat model first.

The choke point for transitions is `EngineSupervisor.request_switch()`.

## ProcessSupervisor

`ProcessSupervisor` manages named subprocess slots for heavy or volatile work:

- no `shell=True`
- one live process per worker slot
- clean process-tree termination with psutil when available
- JSONL stdout event streaming
- worker crashes do not crash the core UI server

This is the core multi-venv isolation pattern. Wan video, Kohya LoRA training, and EveryDream2 full fine-tuning can use their own Python environments without poisoning the main Studio venv.

## Worker protocol

Workers run as subprocesses:

```text
<engine-python> <engine-worker.py> <request.json>
```

They read one JSON request file and write JSONL events to stdout:

- `status`
- `progress`
- `artifact`
- `complete`
- `error`

There is no shared memory contract and no UI import contract. See `docs/WORKER_PROTOCOL.md` for the full schema.

## Test posture

AIWF Studio relies on a broad regression suite rather than hand-waved demos. The current contributor baseline is 715+ collected tests, with 722 collected in this workspace at the time this file was drafted.

Run tests through the repo venv:

```powershell
.\venv\Scripts\python.exe -m pytest --collect-only -q tests
.\venv\Scripts\python.exe -m pytest tests -q
```

## Extension and clean-room rules

AIWF Studio can study public behavior and reimplement useful workflows, but contributors must not copy incompatible source or import abandoned plugins wholesale.

Allowed:

- Public documentation research
- Behavior-compatible clean-room interfaces
- Explicit services and typed models
- Optional worker engines

Not allowed:

- Recreating `shared` global state
- Hidden callback mutation as a plugin mechanism
- Making optional engines mandatory at app boot
- Copying incompatible legacy code

## Further reading

- `docs/ENGINE_ISOLATION.md`
- `docs/WORKER_PROTOCOL.md`
- `docs/FEATURES.md`
- `docs/IMAGE_MATURITY_MATRIX.md`
- `docs/qa/README.md`
