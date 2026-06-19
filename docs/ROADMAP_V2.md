# AIWF Studio â€” v2 Roadmap Implementation Plan

**Branch:** `v2-roadmap`
**Baseline commit:** `983f9e9` (Initial project snapshot)
**Last updated:** 2026-06-16

---

## Repo State (audit snapshot)

### Test baseline
- Full local suite: 765 tests passing on the Studio venv as of the GPU-tenant audit
- **Run command:** `python scripts/run_tests.py --full`

### What already exists (do not re-implement)

| Area | Files | Status |
|------|-------|--------|
| GPU tenant lock | `services/gpu_tenant_lock.py` | âœ… done |
| Engine supervisor | `services/engine_supervisor.py` | âœ… done |
| Engine events domain | `core/domain/engine_events.py` | âœ… done |
| Job status domain | `core/domain/job_status.py` | âœ… done |
| Training domain models | `core/domain/training.py` | âœ… done |
| ED2 subprocess client | `services/ed2_client.py` | âœ… done |
| Kohya subprocess client | `services/kohya_client.py` | âœ… done |
| Engine worker stubs | `engines/{ed2,kohya,generation}/worker.py` | âœ… done |
| Wan I2V pipeline | `infrastructure/wan/pipeline.py` | âœ… done |
| Wan service + GPU lock | `services/wan.py` | âœ… done |
| Video tab UI | `web/tabs/wan_i2v.py` | âœ… done |
| Model download catalog | `services/model_download_catalog.py` | âœ… done |
| GGUF/safetensors header reader | `infrastructure/model_header.py` | âœ… done |
| Model swap optimization | `infrastructure/wan/pipeline.py` (swap_models) | âœ… done |
| Engine isolation arch doc | `docs/architecture/engine_isolation_architecture.md` | âœ… done |
| Video domain models | `core/domain/video.py`, `core/domain/wan.py` | âœ… done |
| RIFE service + tab | `services/rife.py`, `web/tabs/rife.py` | âœ… done |

### What the roadmap calls for that is NOT yet done

| Item | Roadmap section | Priority |
|------|----------------|----------|
| `aiwf/core/domain/engine.py` (EngineTenant enum, switch models) | Phase 1 | High |
| `aiwf/core/domain/worker.py` (WorkerCommand, WorkerResult) | Phase 3 | High |
| `aiwf/services/process_supervisor.py` | Phase 3 | High |
| `aiwf/services/ollama_client.py` | Phase 2 | High |
| `aiwf/web/tabs/chat_workspace.py` | Phase 2 | High |
| `aiwf/services/training/dataset_validator.py` | Phase 4 | Medium |
| `aiwf/services/training/kohya_config.py` | Phase 4 | Medium |
| `aiwf/services/training/kohya_runner.py` | Phase 4 | Medium |
| `aiwf/services/training/ed2_config.py` | Phase 4 | Medium |
| `aiwf/services/training/ed2_runner.py` | Phase 4 | Medium |
| `aiwf/web/tabs/training.py` | Phase 4 | Medium |
| Model info lookup (HF / CivitAI / Ollama API) | Task #36 | Medium |
| `docs/WORKER_PROTOCOL.md` | Phase 3 | Low |
| `docs/OLLAMA_CHAT_TENANT.md` | Phase 2 | Low |
| `docs/TRAINING_ENGINE_ROADMAP.md` | Phase 4 | Low |
| `docs/ACCELERATION_EXPERIMENTS.md` | Phase 6 | Low |
| `docs/BENCHMARK_PROTOCOL.md` | Phase 6 | Low |
| Benchmark harness | Phase 6 | Low |
| CUDA Graphs / torch.compile experiment | Phase 6 | Deferred |
| NVENC export | Phase 6 | Deferred |
| Agentic prompt tools | Phase 7 | Deferred |

---

## Architecture rules (non-negotiable)

1. No global `shared` state. Dependencies flow through `AppContext`.
2. UI callbacks call services only â€” no torch/diffusers in tab files.
3. Services use typed domain models (Pydantic or dataclass).
4. Workers stream structured events back (JSONL or generator).
5. One GPU-heavy tenant at a time (enforced by `gpu_tenant_lock.py`).
6. `subprocess.Popen([...], shell=False)` only â€” never raw shell strings.
7. `pathlib.Path` everywhere. No hardcoded user paths.
8. Kill subprocesses via `psutil` recursive tree termination.
9. Acceleration features behind flags until locally benchmarked.
10. `transformers` stays `>=4.44,<5`.

---

## Sprint A â€” Engine Domain + Process Supervisor + Ollama Chat

**Goal:** Fill the three biggest gaps in one sprint: formal engine domain types, a real process supervisor, and a working chat tab.

### A1 â€” `aiwf/core/domain/engine.py`

Formal types for tenant switching. `EngineSupervisor` now uses these domain types for ownership, waiting, denial, and release decisions.

```python
class EngineTenant(str, Enum):
    IDLE = "idle"
    IMAGE = "image"
    VIDEO = "video"
    CHAT = "chat"
    LORA_TRAINING = "lora_training"
    FULL_TRAINING = "full_training"
    ENHANCE = "enhance"

@dataclass(frozen=True)
class EngineSwitchRequest:
    target: EngineTenant
    reason: str = ""
    job_id: str = ""
    allow_wait: bool = False

@dataclass(frozen=True)
class EngineSwitchResult:
    ok: bool
    active: EngineTenant
    message: str
```

Wire into `engine_supervisor.py` â€” replace its informal string state with `EngineTenant`.

**Test:** `tests/test_engine_domain.py` â€” switch logic, idle/chat/video transitions.

### A2 â€” `aiwf/core/domain/worker.py`

Typed worker contract used by process supervisor and all runners.

```python
@dataclass(frozen=True)
class WorkerCommand:
    args: list[str]
    cwd: Path
    env: dict[str, str]
    name: str
    timeout_seconds: int | None = None

@dataclass(frozen=True)
class WorkerResult:
    job_id: str
    status: str          # "completed" | "failed" | "cancelled"
    return_code: int | None
    output_paths: list[Path]
    logs_path: Path | None
    error_message: str
```

**Test:** pure dataclass â€” import and construct.

### A3 â€” `aiwf/services/process_supervisor.py`

Single place to launch, stream, and kill subprocess workers. Used by ED2Runner, KohyaRunner, and eventually video workers.

Key points:
- `start(worker_id, command) -> Iterator[str]` â€” streams log lines
- `stop(worker_id) -> str` â€” psutil recursive kill
- `CREATE_NEW_PROCESS_GROUP` on Windows, `os.setsid` on POSIX
- Never `shell=True`
- Tracks active processes in `dict[str, Popen]`

**Test:** `tests/test_process_supervisor.py` â€” echo subprocess, stop, double-start guard.

### A4 â€” `aiwf/services/ollama_client.py`

Thin HTTP wrapper around Ollama's REST API. No mandatory install â€” gracefully degrades.

```python
class OllamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11434")
    def healthcheck(self) -> bool
    def list_models(self) -> list[str]
    def unload(self, model: str) -> bool        # keep_alive: 0
    def stream_chat(self, model, messages, options) -> Iterator[str]
```

Dependencies: `httpx` (already in requirements via fastapi chain â€” verify).
Fallback: if `httpx` missing, raise `ImportError` with install hint.

**Test:** `tests/test_ollama_client.py` â€” mock httpx, healthcheck True/False, stream tokens.

### A5 â€” `aiwf/web/tabs/chat_workspace.py`

Chat tab. Registers via `@registry.tab("Chat", order=15)`.

UI layout:
- Ollama status pill (green/red, refresh button)
- Model dropdown (populated from `list_models()`)
- Chat history (`gr.Chatbot`)
- Message input + Send button
- Unload button (explicit â€” does not auto-unload)
- Setup note if Ollama not detected

Wiring rules:
- Tab select â†’ check Ollama health, refresh model list
- Send â†’ acquire GPU lock (CHAT tenant), stream tokens to chatbot
- Unload â†’ release lock, call `client.unload(model)`
- If Ollama not running â†’ show friendly install guidance, no crash

**Does NOT:** auto-download Ollama, auto-load on startup, share VRAM with video/image.

### A6 â€” Wire chat unload into EngineSupervisor

When switching away from CHAT to IMAGE/VIDEO/TRAINING:
- Call `ollama_client.unload(active_model)` before releasing the lock
- Log the switch

### A7 â€” Tests

- `tests/test_engine_domain.py`
- `tests/test_process_supervisor.py`
- `tests/test_ollama_client.py`

---

## Sprint B â€” Training Services

**Goal:** Make the training tab functional end-to-end with subprocess isolation.

### B1 â€” `aiwf/services/training/dataset_validator.py`

Preflight checks before any training run:
- Dataset root exists and is a directory
- At least one image file present (jpg/png/webp)
- Caption files present if captioning mode requires them
- No empty subdirectories (warn, don't block)
- Output directory is writable
- Base checkpoint path exists
- Training engine venv exists at expected path

Returns `ValidationResult(ok: bool, errors: list[str], warnings: list[str])`.

**Test:** `tests/test_dataset_validator.py` â€” tmp_path fixtures.

### B2 â€” `aiwf/services/training/kohya_config.py`

Builds Kohya TOML config from `KohyaLoRARequest` domain model. No subprocess logic here â€” pure config generation.

**Test:** `tests/test_training_config_builders.py` â€” field mapping, path escaping, repeats syntax.

### B3 â€” `aiwf/services/training/kohya_runner.py`

Thin wrapper: takes config path + output dir, builds `WorkerCommand`, delegates to `ProcessSupervisor`.

```python
class KohyaRunner:
    def start(self, config_path: Path, output_dir: Path) -> Iterator[str]
    def stop(self) -> str
```

### B4 â€” `aiwf/services/training/ed2_config.py` + `ed2_runner.py`

Same pattern as Kohya. ED2 runner delegates to existing `ed2_client.py` or directly to ProcessSupervisor.

### B5 â€” `aiwf/web/tabs/training.py`

Training tab (`@registry.tab("Training", order=20)`):
- Dataset path input + Validate button
- Base model selector
- Output name/dir
- Steps, LR, batch size controls
- Start Training / Stop buttons
- Live log stream (`gr.Textbox` streaming)
- Engine lock check â€” block if GPU held by video/chat

---

## Sprint C â€” Model Info Lookup (Task #36)

**Goal:** Fetch metadata from HF API, CivitAI API, and Ollama by model name/URL. Display in Models tab info panel.

### C1 â€” `aiwf/services/metadata.py` (expand existing)

`metadata.py` already exists â€” extend it:
- `fetch_hf_model_info(repo_id) -> ModelInfo | None` â€” HF `/api/models/{repo_id}`
- `fetch_civitai_model_info(model_id_or_url) -> ModelInfo | None` â€” CivitAI `/api/v1/models/{id}`
- `fetch_ollama_model_info(model_name) -> ModelInfo | None` â€” Ollama `/api/show`
- `ModelInfo` dataclass: name, description, tags, license, downloads, size_mb, url

### C2 â€” Wire into Models tab

In `model_manager.py`: add "Model Info" panel â€” enter model name/URL, click Lookup, display info card.

---

## Sprint D â€” Docs + AGENTS.md Addendum

Write the docs the roadmap calls for (these are not optional â€” they keep agents aligned):

- `docs/WORKER_PROTOCOL.md` â€” event types, command/result schemas, streaming contract
- `docs/OLLAMA_CHAT_TENANT.md` â€” chat architecture, unload behavior, GPU lock interaction
- `docs/TRAINING_ENGINE_ROADMAP.md` â€” Kohya and ED2 subprocess design
- `docs/ACCELERATION_EXPERIMENTS.md` â€” what's experimental, what's verified, benchmark protocol
- Append AGENTS.md addendum from roadmap Â§6

---

## Sprint E â€” Performance Experiments (deferred, flagged)

These are experiments, not features. Nothing ships to users unless benchmarked locally.

| Experiment | Flag | Status |
|-----------|------|--------|
| CUDA Graphs for SDXL denoising | `AIWF_CUDA_GRAPHS=1` | Not started |
| `torch.compile` UNet/transformer | `AIWF_TORCH_COMPILE=1` | Not started |
| channels-last image UNet | `AIWF_CHANNELS_LAST=1` | Not started |
| SageAttention 2 / Triton-Windows | `AIWF_SAGE_ATTN=1` | Verify install first |
| torchao native quantization | `AIWF_TORCHAO=1` | Verify API first |
| FP8 Ada Lovelace paths | `AIWF_FP8=1` | Partially done (Wan) |
| NVENC video export | `AIWF_NVENC=1` | Not started |
| RTX VSR research | `AIWF_VSR=1` | Do not claim without measurement |

**Benchmark every change with:** prompt, seed, checkpoint, resolution, steps, sampler, batch size, VRAM peak, first-run time, second-run time, output hash.

---

## Implementation order (recommended)

```
Sprint A  â†’  Sprint B  â†’  Sprint C  â†’  Sprint D  â†’  Sprint E (optional)
   |              |             |
engine.py    training       model
worker.py    services       info
process_sup  dataset_val    lookup
ollama       kohya/ed2
chat_tab     training_tab
```

---

## Definition of done

A task is **not** complete until:

- [ ] Existing tests still pass (`python -m pytest tests/ -q`)
- [ ] New tests added for non-trivial logic
- [ ] No hardcoded `C:\Users\...` paths
- [ ] No `shell=True` subprocess calls
- [ ] No torch imports in UI tab files
- [ ] Clean error messages surfaced to UI; full tracebacks in logs
- [ ] README or AGENTS.md updated if architecture changes
- [ ] No performance claims without local benchmark data

---

## File manifest for Sprint A

```text
NEW  aiwf/core/domain/engine.py
NEW  aiwf/core/domain/worker.py
NEW  aiwf/services/process_supervisor.py
NEW  aiwf/services/ollama_client.py
NEW  aiwf/web/tabs/chat_workspace.py
MOD  aiwf/services/engine_supervisor.py   (use EngineTenant, wire ollama unload)
NEW  tests/test_engine_domain.py
NEW  tests/test_process_supervisor.py
NEW  tests/test_ollama_client.py
NEW  docs/WORKER_PROTOCOL.md
NEW  docs/OLLAMA_CHAT_TENANT.md
MOD  AGENTS.md                            (append roadmap addendum)
```

## File manifest for Sprint B

```text
NEW  aiwf/services/training/__init__.py
NEW  aiwf/services/training/dataset_validator.py
NEW  aiwf/services/training/kohya_config.py
NEW  aiwf/services/training/kohya_runner.py
NEW  aiwf/services/training/ed2_config.py
NEW  aiwf/services/training/ed2_runner.py
NEW  aiwf/web/tabs/training.py
NEW  tests/test_dataset_validator.py
NEW  tests/test_training_config_builders.py
NEW  docs/TRAINING_ENGINE_ROADMAP.md
```

## File manifest for Sprint C

```text
MOD  aiwf/services/metadata.py            (add HF/CivitAI/Ollama lookup)
MOD  aiwf/web/tabs/model_manager.py       (add info panel)
NEW  tests/test_model_info_lookup.py
```
