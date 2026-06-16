# Training Engine Roadmap

This document describes how AIWF Studio integrates Kohya SS (LoRA) and EveryDream2 (full fine-tuning) as subprocess-based training engines. No training library is imported at boot time.

ED2 is the mandatory engine for full fine-tune jobs. That does **not** make ED2 a mandatory app boot dependency: AIWF must still start when ED2 is disabled or missing, but any full fine-tune preflight must block until ED2 is enabled, has a usable Python environment, and the EveryDream2 `train.py` entrypoint exists.

ED2 normally uses a dedicated venv. For Studio-dependency experiments, set `"venv_dir": "studio"` (or `"shared"` / `"main"`) in `engines.json`. Shared mode installs only AIWF's ED2 overlay requirements into the main Studio venv and intentionally skips `EveryDream2trainer/requirements.txt`, because the upstream file pins older torch, diffusers, numpy, protobuf, xformers, and compel versions.

---

## Architecture overview

```
Web UI (training.py)
    │
    ├─ DatasetValidator        ← pure stdlib, safe to import at boot
    │
    ├─ KohyaRunner / ED2Runner ← lazy: resolves venv python at first use
    │       │
    │       └─ ProcessSupervisor.start(name, WorkerCommand)
    │               │
    │               └─ subprocess (shell=False, unbuffered stdout)
    │                       │
    │                       └─ engine worker script (kohya / ed2)
    │                               JSONL events → stdout
    │
    └─ EngineSupervisor.request_switch(LORA_TRAINING | FULL_TRAINING)
            └─ GPU tenant lock — one heavy tenant at a time
```

---

## Engine registration

Engines are declared in `engines.json` (sibling to `launch.py`):

```json
{
  "engines": {
    "kohya": {
      "enabled": true,
      "repo_root": "C:/Users/Shawn/Desktop/kohya_ss",
      "venv": "C:/Users/Shawn/Desktop/kohya_ss/venv"
    },
    "ed2": {
      "enabled": true,
      "repo_root": "C:/Users/Shawn/Desktop/EveryDream2trainer",
      "venv": "C:/Users/Shawn/Desktop/EveryDream2trainer/venv"
    }
  }
}
```

`launch.py` reads this file at startup and passes resolved `EngineSpec` objects to `_build_engine_registry()`. The training tab calls `_probe_engines()` to check availability — all via deferred imports inside the function, never at module level.

---

## Key rule: optional engines must never become mandatory boot dependencies

Every import of `launch`, `kohya_runner`, `ed2_runner`, torch, diffusers, or any training library **must** be inside a method body or callback function — never at the top of a module.

Violation examples to avoid:

```python
# BAD — top-level import makes Kohya a hard boot dep
from aiwf.services.training.kohya_runner import KohyaRunner

# GOOD — deferred
def on_start(...):
    from aiwf.services.training.kohya_runner import KohyaRunner
    runner = KohyaRunner()
```

The training tab renders a "not configured" notice and remains interactive even when neither engine is installed.

---

## GPU tenant lock

Training jobs acquire the GPU through `EngineSupervisor.request_switch()` before spawning the worker subprocess:

| Engine      | Tenant enum value      |
|-------------|------------------------|
| Kohya LoRA  | `LORA_TRAINING`        |
| ED2 full FT | `FULL_TRAINING`        |

The tenant is released in both the success path and all exception/stop paths via `_release_tenant()`.

Rules:
- Only one GPU-heavy tenant is active at a time.
- CHAT (Ollama) is NOT GPU-heavy; it does not block training from starting.
- Switching away from CHAT always unloads the active Ollama model first.

---

## Dataset validator

`aiwf/services/training/dataset_validator.py` — zero engine imports, pure stdlib.

Pre-flight checks run **before** the GPU tenant is acquired or any subprocess is launched:

- Dataset directory exists, is not empty, contains image files
- Missing captions: >50% missing → error; ≤50% → warning
- Output directory is writable (created if absent)
- Base model path: local file must exist; HF IDs (`org/repo`) pass through
- Resolution must be a multiple of 64 (Kohya)
- LR sanity bounds (ED2)

HF ID detection heuristic: exactly one `/`, no leading `/` or `./`, no backslash, suffix not in `{.safetensors, .ckpt, .pt, .bin, .pth}`.

---

## Config builders

### Kohya (`kohya_config.py`)

Generates TOML config for `train_network.py` / `sdxl_train_network.py` / `flux_train_network.py`.

Uses a pure-Python minimal TOML serialiser (`_toml_value`, `_toml_section`, `_render_toml`) — no `tomli_w` or `toml` dependency required.

Architecture routing:

| `base_arch` | Training script |
|-------------|----------------|
| `sd1`       | `sd_scripts/train_network.py` |
| `sdxl`      | `sdxl_train_network.py` |
| `flux`      | `flux_train_network.py` |
| (unknown)   | falls back to `sdxl` |

### ED2 (`ed2_config.py`)

Generates `train.json` dict serialised to JSON. Accepts the same dict/object interface as the validator via `_RequestProxy`.

---

## Runners

Both runners (`KohyaRunner`, `ED2Runner`) share the same lifecycle:

1. `_resolve_python_exe()` — deferred, reads engine registry once; cached.
2. Write request JSON to a temp directory.
3. Build `WorkerCommand(args=[python, script, "--config", toml_or_json], cwd=repo_root, env=env)`.
4. Delegate to `ProcessSupervisor.start(name, cmd)` — yields stdout lines.
5. `stop()` → `ProcessSupervisor.stop(name)` with psutil recursive tree-kill.

`KohyaEngineNotReady` / `ED2EngineNotReady` are raised by `_resolve_python_exe()` when the engine is not configured. These propagate to the UI as human-readable error strings — full tracebacks are logger-only.

---

## ProcessSupervisor

`aiwf/services/process_supervisor.py` — named-slot subprocess manager.

- `shell=False` always.
- `CREATE_NEW_PROCESS_GROUP` on Windows, `start_new_session=True` on POSIX.
- Double-start guard: raises `RuntimeError` if the named slot is already live.
- `stop()`: psutil recursive tree-kill → fallback `proc.terminate()` → `proc.kill()`.
- `_resolve_cwd()`: maps `/tmp` → `tempfile.gettempdir()` for Windows compatibility.
- Module-level singleton via `get_process_supervisor()`.

---

## JSONL event protocol

Workers write newline-delimited JSON to stdout:

```json
{"kind": "progress", "job_id": "lora-42", "step": 150, "total": 1500, "loss": 0.042}
{"kind": "artifact",  "job_id": "lora-42", "path": "outputs/kohya/my_lora-step150.safetensors"}
{"kind": "complete",  "job_id": "lora-42", "return_code": 0}
{"kind": "error",     "job_id": "lora-42", "message": "CUDA OOM at step 150"}
```

See `docs/WORKER_PROTOCOL.md` for the full schema.

---

## Adding a new training engine

1. Add an entry to `engines.json` with `repo_root` and `venv`.
2. Create `aiwf/services/training/<engine>_config.py` with a `build_<engine>_config()` function.
3. Create `aiwf/services/training/<engine>_runner.py` following the `KohyaRunner` pattern.
4. Register the engine name in `_probe_engines()` in `training.py`.
5. Add a branch in `on_start()` to import and instantiate the new runner (deferred import inside the callback).
6. Add validation in `DatasetValidator` or create a dedicated validator.
7. Write tests in `tests/test_training_config_builders.py` and `tests/test_dataset_validator.py`.

No boot-time imports. No mandatory dependencies. The tab must render with a "not configured" notice even when the new engine is absent.
