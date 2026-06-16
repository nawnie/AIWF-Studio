# AGENTS.md

Guidance for human developers and coding agents working on **AIWF Studio** — a clean-room rebuild of the AUTOMATIC1111-style Stable Diffusion web UI.

---

## Project mission

Build a professional, maintainable, local-first image-generation application using modern Python architecture and a Gradio web UI. Preserve useful workflow ideas from older SD web UIs while avoiding inherited technical debt, fragile global state, and uninspectable plugin behavior.

**This is not a direct port.** Reimplement from first principles.

Prioritize clarity, modularity, testability, and long-term expansion over legacy-style shortcuts.

---

## Project identity

The product should feel like a serious creative tool: clean UI, predictable code, typed domain models, and explicit wiring. Useful automation is welcome; mystery behavior is not.

There is no `shared` global. Dependencies flow through `AppContext` (`aiwf/bootstrap.py`).

---

## Architecture (current)

Layered packages under `aiwf/`:

```text
aiwf/
├── app.py                 # CLI entry: flags, launch merge, Gradio/FastAPI serve
├── bootstrap.py           # Composition root — builds AppContext
├── core/
│   ├── config/            # RuntimeFlags, UserSettings, LaunchSettings
│   ├── domain/            # Pydantic models (GenerationRequest, workflows, styles, …)
│   ├── events/            # EventBus + lifecycle event types
│   ├── interfaces/        # InferenceBackend protocol, plugin interfaces
│   ├── bridge.py          # InfotextBridge (PNG Info → Studio)
│   └── infotext.py        # A1111-style parameter parse/format
├── services/              # Application layer — UI/API call these, not torch
├── infrastructure/        # Diffusers backend, storage, segment, enhance, faceswap, …
├── api/v1/                # FastAPI routes (/api/v1 + /sdapi/v1 adapter)
├── plugins/               # PluginRegistry (in-package; discovers `plugins/` at runtime)
└── web/
    ├── app.py             # create_web_ui(), tab registration
    ├── registry.py        # WebRegistry — @registry.tab decorator
    ├── studio.py          # Primary Studio tab (txt2img / img2img / inpaint)
    ├── components/        # Shared Gradio helpers (checkpoints, results, infotext)
    └── tabs/              # Secondary tabs (Models, Settings, History, …)
```

**Request flow:** Gradio callback or API route → `GenerationService` / other service → `InferenceBackend` (`DiffusersBackend`) → `JobQueue` with progress events.

**Heavy work never belongs in UI callbacks** beyond orchestration. Use `submit_streaming()` for generation with live preview yields.

---

## Shipped capabilities (audit snapshot)

Use this as ground truth for what exists today (not aspirational README items).

### Studio (`aiwf/web/studio.py`)

| Area | Status |
|------|--------|
| txt2img / img2img / inpaint modes | Shipped |
| Live latent preview, continuous generation, interrupt | Shipped |
| Hires fix, CFG/steps/sampler, clip skip, VAE select | Shipped |
| Dynamic prompts (`{a\|b}`, `__wildcard__`, prompt files, Compel) | Shipped |
| Style presets (`{prompt}` templates, built-ins in `style_presets.py`, editable in UI) | Shipped |
| Tags, PNG infotext, seed reuse / randomize, before-after compare | Shipped |
| ControlNet (single unit, preprocessors, Studio Advanced panel) | Shipped |
| Inpaint: mask editor, SAM presets dropdown, outpaint canvas extend | Shipped |
| Inpaint session: keep original + mask; source toggle original/last result | Shipped |
| ReActor panel: face swap on result + optional img2img seam blend | Shipped |
| LoRA keyword expansion via model catalog aliases | Shipped |

### UI tabs (`aiwf/web/tabs/`)

| Tab | Module | Notes |
|-----|--------|-------|
| Studio | `studio.py` | Primary workspace |
| Models | `model_manager.py` | Catalog, aliases, **Download** tab (HF/CivitAI/direct URLs) |
| Segment | `segment.py` | SAM + GroundingDINO text masks |
| Enhance | `enhance.py` | Upscale, GFPGAN/CodeFormer, **old photo restore** pipeline |
| Workflows | `workflows.py` | JSON chains (txt2img → upscale → inpaint, etc.) |
| Face Swap | `faceswap.py` | insightface + inswapper (optional dep) |
| Library | `library.py` | Tag search over saved outputs |
| PNG Info | `pnginfo.py` | Parse metadata → InfotextBridge |
| History | `history.py` | Session job gallery, re-run, infotext copy |
| Settings | `settings.py` | `launch.json`, remote URLs, live preview, output paths |

### Backend / API

- **Diffusers** pipelines for SD1.5, SDXL, inpaint variants; architecture auto-detected from weights
- **ControlNet** SD1.5 light rank128 via PEFT loader (`peft` in requirements)
- **Checkpoint scan** skips `Loras/` and lora-shaped weight files misclassified as checkpoints
- **REST:** `/api/v1/` (native) + `/sdapi/v1/` (A1111-compatible adapter)
- **Enhance API:** upscale, restore, ControlNet detect
- **Job queue:** cancel → `JobState.CANCELLED` (UI shows Stopped, not Error)

### Tests

`pytest tests/` — **149 tests** passing (as of last audit). Run before claiming work complete.

---

## Known gaps & agent cautions

Honest limits — do not assume these exist:

| Gap | Notes |
|-----|-------|
| Multi-unit ControlNet | Single unit per generation today |
| Rich bundled ControlNet annotators | Many preprocessors are pass-through or basic; depth/openpose not fully bundled |
| Composable pipeline stages | Hires fix exists; explicit upscale-as-stage in txt2img path is roadmap |
| Plugin ecosystem | `PluginRegistry` works; only `plugins/example_hello` ships. No callback monkey-patching. |
| Documentation | `docs/` contains mainly `ATTRIBUTION.md`. README roadmap lags some shipped features. |
| `studio.py` size | ~1800 lines — new Studio features should extract helpers/components when practical |
| Git | Repo may be local-only; do not assume CI or remote branches |

**transformers ≥5 breaks checkpoint loading** — keep `transformers>=4.44,<5` (enforced in bootstrap with a log warning).

**Gradio** is installed at **6.x** (`requirements.txt` says `>=4.19`; verify with `import gradio; gradio.__version__`). Prefer documented public APIs. Use `buttons=["copy"]` on Textbox where needed. Avoid undocumented internals.

---

## Clean-room rule

**Allowed:** study behavior, read public docs, reimplement cleanly, compatibility-inspired interfaces with original code.

**Not allowed:** copy A1111 or other incompatible-licensed source; import abandoned plugins wholesale; recreate `shared`-style global state.

When in doubt, rebuild the feature.

---

## Dependency baseline

Pinned in `requirements.txt` (torch/torchvision installed by `launch.py` with CUDA wheels, default cu124):

- **Core ML:** `diffusers`, `transformers` (4.x only), `accelerate`, `peft`, `safetensors`, `compel`
- **UI / API:** `gradio`, `fastapi`, `uvicorn`, `pydantic`, `pydantic-settings`
- **Vision / enhance:** `Pillow`, `numpy`, `opencv-python-headless`, `spandrel`, `spandrel-extra-arches`, `facexlib`, `scipy`
- **Segment:** `segment-anything` (+ transformers GroundingDINO-tiny)
- **Face swap (optional):** `insightface`, `onnxruntime` / `onnxruntime-gpu`
- **Dev:** `pytest`, `psutil`

Do not casually bump versions. Changes need a reason, compatibility note, and test impact. Be especially careful with: `torch`, `diffusers`, `transformers`, `gradio`, `opencv-python-headless`, `spandrel`.

---

## Repository layout (runtime)

```text
aiwf/              Main Python application package (see Architecture)
docs/              Project documentation (sparse — expand when adding features)
mcps/              MCP tool descriptors for agent tooling in this workspace
models/            Local model storage (checkpoints, LoRAs, VAE, ControlNet, SAM, …)
outputs/           Generated images and runtime output
plugins/           Runtime plugin folders (`plugin.py` with `setup(ctx)` or `plugin.on_load`)
prompts/           Prompt library text files (`{data_dir}/prompts/`)
static/            Frontend assets (`style.css`, `studio.js` — hotkeys, status pill)
tests/             Pytest suite
venv/              Local virtualenv — do not edit or commit
wildcards/         Wildcard prompt files (`{data_dir}/wildcards/`)
workflows/         Workflow JSON presets (`{data_dir}/workflows/`)

launch.py          Environment bootstrap + venv torch install
webui.bat          Windows launcher
config.json        UserSettings persistence (styles, tags, output prefs, …)
launch.json        LaunchSettings persistence (GPU flags, listen, port, API mode)
```

Default `data_dir` is the project root. Model paths resolve via `RuntimeFlags.resolved_*()` with optional overrides in `launch.json` / CLI.

### Model folder conventions

```text
models/
├── Stable-diffusion/   Checkpoints (.safetensors, .ckpt)
├── Loras/              LoRA weights (not scanned as checkpoints)
├── VAE/
├── ControlNet/
├── sam/                SAM .pth weights
├── RealESRGAN/
├── GFPGAN/
├── Codeformer/
└── insightface/        Face swap models
```

---

## Configuration & launch

| File | Purpose |
|------|---------|
| `launch.json` | Saved GPU/network flags; synced to `webui.settings.bat` on save |
| `config.json` | App settings: styles, tags, output subdirs, live preview interval |
| `RuntimeFlags` | CLI + env (`AIWF_*`) — session flags |
| `LaunchSettings` | Persisted launch profile merged with CLI on startup |

Entry points:

```powershell
python launch.py          # Install deps + start
python -m aiwf.app          # Direct serve (after env ready)
python -m pytest tests/     # Test suite
```

---

## Gradio UI rules

1. Register tabs via `@registry.tab("Name", order=N)` in `aiwf/web/tabs/` or `register_studio()` for Studio.
2. Keep event wiring colocated with the tab that owns the components.
3. Use `tab.select(...)` to refresh stale catalogs when users switch tabs (checkpoints, SAM, network URLs).
4. Studio keyboard shortcuts live in `static/studio.js` (Shift+Enter generate, Escape stop).
5. Prefer `gr.update()` streaming yields for long operations; show progress via `JobQueue` events.
6. Check Gradio version before using version-specific component args.

---

## Domain patterns agents should follow

### Generation

- Build requests with `GenerationRequest` + `GenerationMode` — never ad-hoc dicts in services.
- Prompt resolution: `PromptProcessorService.prepare_prompt()` applies styles, wildcards, prompt files, LoRA keywords.
- Styles: `PromptStyle` with `{prompt}` placeholder; built-ins in `core/domain/style_presets.py`; user edits persist in `config.json`.
- Inpaint masks: `infrastructure/diffusers/mask.py` — session mask reuse when editor hidden; never treat full background as mask.
- Dev traces: use `aiwf.dev.diagnostics` for structured performance notes. When a model run changes speed, record `app_version`, `model_id`, elapsed seconds, and a real throughput field such as `steps_per_second` or `frames_per_second` so later comparisons stay meaningful.

### Models

- Invalidate catalogs after disk changes: `refresh_checkpoint_catalog()`, `refresh_vae_catalog()`, `segment.refresh_models()`.
- Model downloads: `services/model_download.py` + curated `model_download_catalog.py`.

### Workflows

- Definitions in `core/domain/workflow.py`; builtins in `workflow_store.py`; execution in `workflow_executor.py`.

### Plugins

- Drop-in under `plugins/<name>/plugin.py`. Expose `setup(ctx)` or a `plugin` object with `on_load(ctx)`. Failures log and skip — never crash startup.

---

## Adding features (checklist)

1. **Domain model** in `aiwf/core/domain/` if new typed data
2. **Service** in `aiwf/services/` for orchestration
3. **Infrastructure** in `aiwf/infrastructure/` for torch/diffusers/filesystem
4. **UI** in `aiwf/web/tabs/` or Studio components — thin callbacks
5. **Tests** in `tests/test_<area>.py` — required for non-trivial logic
6. **API** (optional) in `aiwf/api/v1/routes.py` + `api/schemas.py`
7. Update this file or README only when the user asks or the change is architectural

---

## Verification before completion

Agents must run relevant tests and not claim success without evidence:

```powershell
python -m pytest tests/ -q
python -m pytest tests/test_<changed_area>.py -q
```

For UI-only changes: `from aiwf.web.app import create_web_ui` / `build_context()` smoke import.

---

## Audit summary (professional)

| Dimension | Assessment |
|-----------|------------|
| **Architecture** | Strong — clear layers, `AppContext`, typed requests, protocol-based backend |
| **Feature completeness** | High for a rebuild — Studio + 9 tabs, workflows, API adapter, downloads |
| **Code health** | Good with hotspots — `studio.py` monolith; otherwise modular services |
| **Test coverage** | Solid for core paths (149 tests); UI mostly integration-smoke level |
| **Docs** | Weak — AGENTS.md + README carry most guidance; `docs/` minimal |
| **Agent readiness** | Good after this file — structure, patterns, and gaps are explicit |

**Top maintenance priorities for future agents:**

1. Extract Studio subpanels (inpaint, styles, controlnet) into `web/components/` as they grow
2. Keep README shipped list aligned with reality when touching features
3. Add tests when fixing mask/style/generation edge cases
4. Do not regress clean-room or global-state rules for short-term parity hacks

---

## Quick reference — key files

| Task | Start here |
|------|------------|
| Studio UI | `aiwf/web/studio.py` |
| New tab | `aiwf/web/tabs/`, register in `aiwf/web/app.py` |
| Generation logic | `aiwf/services/generation.py`, `aiwf/infrastructure/diffusers/backend.py` |
| Prompts / styles | `aiwf/services/prompt_processor.py`, `aiwf/core/domain/style_presets.py` |
| Inpaint / masks | `aiwf/infrastructure/diffusers/mask.py` |
| Model downloads | `aiwf/web/tabs/model_manager.py`, `aiwf/services/model_download_catalog.py` |
| API | `aiwf/api/v1/routes.py` |
| Job tracking | `aiwf/services/engine_supervisor.py`, `aiwf/core/domain/job_status.py` |
| Video (Wan I2V) | `aiwf/services/wan.py`, `aiwf/infrastructure/wan/pipeline.py` |
| RIFE interpolation | `aiwf/services/rife.py`, `aiwf/web/tabs/rife.py` |
| Chat (Ollama) | `aiwf/services/ollama_client.py`, `aiwf/web/tabs/chat_workspace.py` |
| GPU tenant tracking | `aiwf/services/engine_supervisor.py` (`active_tenant`, `request_switch`) |
| Subprocess workers | `aiwf/services/process_supervisor.py`, `aiwf/core/domain/worker.py` |

---

## Sprint A addendum (v2 roadmap)

### New files added in Sprint A

| File | Purpose |
|------|---------|
| `aiwf/core/domain/engine.py` | `EngineTenant` enum, `EngineSwitchRequest/Result`, `EngineStatus` |
| `aiwf/core/domain/worker.py` | `WorkerCommand`, `WorkerResult` -- typed subprocess contract |
| `aiwf/services/process_supervisor.py` | Named-slot subprocess launcher with psutil tree-kill |
| `aiwf/services/ollama_client.py` | Thin `httpx` wrapper around Ollama REST API |
| `aiwf/web/tabs/chat_workspace.py` | Chat tab -- Ollama model picker, streaming chatbot |
| `tests/test_engine_domain.py` | Engine domain tests |
| `tests/test_process_supervisor.py` | ProcessSupervisor tests (echo, stop, double-start guard) |
| `tests/test_ollama_client.py` | OllamaClient tests (mocked httpx) |
| `docs/WORKER_PROTOCOL.md` | Full JSONL event protocol spec |
| `docs/OLLAMA_CHAT_TENANT.md` | Chat GPU tenant architecture |

### Engine tenant rules (non-negotiable)

1. At most **one GPU-heavy tenant** active at a time (`is_gpu_heavy()` covers IMAGE, VIDEO, TRAINING, ENHANCE).
2. CHAT (`is_gpu_heavy() == False`) shares the lock slot but does not block image/video if scheduled carefully.
3. Switching FROM CHAT to any GPU-heavy tenant **always** calls `ollama_client.unload(model)` first.
4. `EngineSupervisor.request_switch()` is the single choke-point for all tenant transitions -- never bypass it.
5. `set_chat_model(name)` must be called when a chat model is loaded so the supervisor can unload it later.
6. Optional engines must never become mandatory boot dependencies.

### ProcessSupervisor rules

- Never `shell=True`.
- Use `CREATE_NEW_PROCESS_GROUP` (Windows) / `start_new_session=True` (POSIX) for clean tree-kill.
- Worker slots are identified by a string name; one live process per name.
- `stop()` uses psutil recursive tree termination if available; falls back to `proc.terminate()`.
- `stop_all()` is called on app shutdown.

### OllamaClient rules

- Gracefully degrades -- `healthcheck()` returns `False` if Ollama is not running (never crashes the app).
- `unload(model)` uses `keep_alive: 0` on `/api/generate` -- the only correct way to evict a model.
- `stream_chat()` uses `httpx.stream()` context manager; never buffers the full response.
- Import is deferred via `_httpx()` helper -- the client can be constructed even without `httpx` installed.

### Chat tab rules

- Direction update: future local chat work should target **llama.cpp with GGUF models**, not Ollama, unless the user explicitly asks to keep Ollama compatibility.
- No torch imports in `chat_workspace.py`.
- Tab select triggers
---

## Sprint B addendum — Training services

### Training engine rule (CRITICAL)

**Optional engines must never become mandatory boot dependencies.**

- `from launch import ...` is only permitted inside method/function bodies, never at module top-level.
- `KohyaRunner._resolve_python_exe()` and `ED2Runner._resolve_python_exe()` defer the `launch` import to first call.
- `_probe_engines()` in `training.py` wraps the entire `launch` import in `try/except Exception: pass`.
- The training tab renders fully (with a "not configured" guidance notice) even when no engine is installed.
- Any new engine added in future MUST follow this pattern.

### Dataset validator rules

- `DatasetValidator` and all helpers in `dataset_validator.py` are pure stdlib — zero engine, torch, or diffusers imports.
- Safe to import at boot time.
- HF ID heuristic: `count("/")==1` AND no leading `/` AND no `\\` AND suffix not in `{.safetensors,.ckpt,.pt,.bin,.pth}`.
  - This correctly handles `stabilityai/stable-diffusion-xl-base-1.0` (suffix `.0` is not a model extension).
  - On Linux `os.sep=='/'` — do NOT use `os.sep in path` to detect local paths; use the explicit heuristic above.
- Caption missing threshold: >50% missing → error; ≤50% → warning.

### Config builder rules

- Pure-Python TOML serialiser (`_toml_value`, `_toml_section`, `_render_toml`) — no `tomli_w`/`toml` dependency.
- `bool` must be checked before `int` in `_toml_value` (Python `bool` is a subclass of `int`).
- All string values must escape `\` as `\\` and `"` as `\"`.
- `_RequestProxy` lets validators and builders accept both `dict` and plain-attribute objects without importing domain types.

### ProcessSupervisor `_resolve_cwd` rule

- `_resolve_cwd(cwd)` maps POSIX `/tmp` → `tempfile.gettempdir()` on Windows so workers can specify `/tmp` cross-platform.
- Raises `FileNotFoundError` if the resolved path does not exist (fail-fast, not silent fallback to cwd).

### Quick-reference: Sprint B files

| File | Purpose |
|------|---------|
| `aiwf/services/training/__init__.py` | Package marker |
| `aiwf/services/training/dataset_validator.py` | Pre-flight checks, pure stdlib |
| `aiwf/services/training/kohya_config.py` | TOML builder for Kohya configs |
| `aiwf/services/training/ed2_config.py` | JSON builder for ED2 `train.json` |
| `aiwf/services/training/kohya_runner.py` | Subprocess runner for Kohya LoRA |
| `aiwf/services/training/ed2_runner.py` | Subprocess runner for ED2 full FT |
| `aiwf/web/tabs/training.py` | Training tab UI |
| `tests/test_dataset_validator.py` | 29 tests |
| `tests/test_training_config_builders.py` | 35 tests |
| `docs/TRAINING_ENGINE_ROADMAP.md` | Architecture + "how to add an engine" guide |

---

## Sprint C addendum — Model info lookup

### ModelInfoLookup rules

- `aiwf/services/model_info_lookup.py` — zero imports at module level beyond stdlib.
- All HTTP is done via `urllib.request` (stdlib only) — no httpx, no requests, no huggingface_hub package required.
- Returns `None` rather than raising on network errors, 404s, or missing tokens.
- Callers must treat the result as advisory — never block the user on a lookup failure.
- Token resolution reads env vars (`HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, `CIVITAI_API_TOKEN`) or the passed argument. Never imports `launch` or any engine module.

### HF lookup notes

- Uses `https://huggingface.co/api/models/{repo_id}` — no `huggingface_hub` package required.
- `tags` list is filtered: entries starting with `license:`, `arxiv:`, `base_model:`, `language:` are stripped from concept tags but parsed into dedicated fields.
- Trigger words come from `cardData.trigger_words`, base model from `cardData.base_model`.
- Token sent as `Authorization: Bearer {token}` header (not query param).

### CivitAI lookup notes

- Token sent as `?token=` query param (CivitAI's documented auth method).
- `description` field contains HTML — always call `_strip_html()` before display.
- `trainedWords` from the version object are the trigger words.
- `_parse_civitai_ref()` handles bare integer IDs, `/models/<id>` URLs, and `?modelVersionId=` query strings.

### Auto-detect routing

```
civitai.com in string  → lookup_civitai
bare integer           → lookup_civitai
org/repo (one slash)   → lookup_hf
anything else          → lookup_ollama → (None if miss)
```

### Model Info tab

- Added as a new tab in `aiwf/web/tabs/model_manager.py` (after ControlNet).
- Reads `ctx.settings.huggingface_token` and `ctx.settings.civitai_token` at call time (not at import time).
- Supports pressing Enter in the query box (`.submit`) as well as button click.

### Quick-reference: Sprint C files

| File | Purpose |
|------|---------|
| `aiwf/services/model_info_lookup.py` | RemoteModelInfo + ModelInfoLookup service |
| `aiwf/web/tabs/model_manager.py` | Model Info tab wired in (new tab after ControlNet) |
| `tests/test_model_info_lookup.py` | 57 tests — all network calls mocked |

---

## Addendum — Sprint A/B/C/D (added 2026-06-14)

### Engine domain (`aiwf/core/domain/engine.py`)

- `EngineTenant` enum: `IDLE`, `IMAGE`, `VIDEO`, `CHAT`, `LORA_TRAINING`, `FULL_TRAINING`, `ENHANCE`
- `EngineSwitchRequest(target, reason, allow_wait)` — frozen dataclass
- `EngineSwitchResult(ok, active, message, log_path)` — frozen dataclass
- `EngineStatus` — mutable live snapshot; call `record_switch()` on each transition
- `CHAT` is **not** GPU-heavy (`is_gpu_heavy() → False`). Ollama manages its own VRAM.

### Worker domain (`aiwf/core/domain/worker.py`)

- `WorkerCommand(args, cwd, env, name, timeout_seconds)` — subprocess spec, frozen
- `WorkerResult(job_id, status, return_code, output_paths, logs_path, error_message)` — frozen
- `WorkerEvent` dataclass hierarchy: `LogLine`, `ProgressEvent`, `DoneEvent`, `ErrorEvent`
- `status` values: `"completed"`, `"failed"`, `"cancelled"`

### Process supervisor (`aiwf/services/process_supervisor.py`)

- `ProcessSupervisor` — singleton-style (one per app). Stores `dict[str, Popen]`.
- `start(worker_id, command) → Iterator[WorkerEvent]` — never `shell=True`; `CREATE_NEW_PROCESS_GROUP` on Windows, `os.setsid` on POSIX
- `stop(worker_id) → str` — psutil recursive tree kill; returns status string
- Guard: raises `RuntimeError` if `worker_id` already active (no silent double-start)
- All log lines are yielded as `LogLine` events before `DoneEvent`

### Ollama client (`aiwf/services/ollama_client.py`)

- `OllamaClient(base_url="http://127.0.0.1:11434")` — no mandatory install
- `healthcheck() → bool` — GET `/api/tags`; returns `False` on any error
- `list_models() → list[str]` — parses `models[].name` from tags response
- `unload(model) → bool` — POST `/api/generate` with `keep_alive: 0`
- `stream_chat(model, messages, options) → Iterator[str]` — streams token strings from `/api/chat`
- Always uses `httpx`. If `httpx` missing, raises `ImportError` with install hint.
- Never imports torch, diffusers, or any GPU module.

### Chat workspace tab (`aiwf/web/tabs/chat_workspace.py`)

- Registered as `@registry.tab("Chat", order=15)`
- Ollama status pill refreshes on tab select; shows install guidance if not detected
- Send → acquires `gpu_tenant_lock` with `CHAT` tenant, streams tokens to `gr.Chatbot`
- Unload button → `client.unload(model)` then releases lock
- Does NOT auto-download Ollama, auto-load on startup, or share VRAM

### Training services (`aiwf/services/training/`)

- `dataset_validator.py`: `DatasetValidator.validate(root, config) → ValidationResult(ok, errors, warnings)`
  - Checks: root exists, images present, captions present (if mode requires), output dir writable, base checkpoint exists, engine venv exists
- `kohya_config.py`: `KohyaConfigBuilder.build(request) → str` — pure TOML string, no subprocess
- `kohya_runner.py`: `KohyaRunner.start(config_path, output_dir) → Iterator[WorkerEvent]` — delegates to `ProcessSupervisor`
- `ed2_config.py` / `ed2_runner.py`: same pattern for EveryDream2
- Training tab: `@registry.tab("Training", order=20)` — blocks if GPU held by video/chat
- Full fine-tune policy: ED2 is mandatory for full model fine-tuning jobs, but ED2 must still remain optional at app boot. Preflight should block full tune until ED2 is enabled, a usable Python environment exists, and `EveryDream2trainer/train.py` is present.
- ED2 may use the main Studio venv only when `engines.json` explicitly sets `"venv_dir": "studio"` (or `"shared"` / `"main"`). Shared mode installs only AIWF's ED2 overlay requirements and must skip `EveryDream2trainer/requirements.txt` because the upstream file pins older core packages.

### Model info lookup (`aiwf/services/model_info_lookup.py`)

- `ModelInfoLookup(hf_token, civitai_token)` — tokens from settings, read at call time
- `lookup(query) → RemoteModelInfo | None` — auto-routes by query shape
- Auto-detect routing: `civitai.com` or bare int → CivitAI; `org/repo` → HF; else → Ollama
- All network errors return `None`, never raise to caller
- `RemoteModelInfo`: name, description, tags, license, downloads, size_mb, url, trigger_words, base_model, source

### Acceleration experiments

All perf experiments are **flag-gated** (`AIWF_*=1`). See `docs/ACCELERATION_EXPERIMENTS.md` for the full list, benchmark protocol, and current status. Nothing is enabled by default. Do not claim performance improvements without a logged benchmark entry.

### Model operations tabs

- Model mixing and conversion/quantization live under `aiwf/services/model_ops.py`, `aiwf/workers/model_ops.py`, and `aiwf/web/tabs/model_manager.py`.
- Heavy operations must preflight first, then run through `ProcessSupervisor`; never load or convert large models directly inside Gradio callbacks.
- Write a receipt for generated model artifacts so users can trace source files, ratios, dtypes/quant choices, warnings, and timestamps.
- NVFP4 is a storage/compression concept on Shawn's RTX 4070 Ti SUPER (Ada Lovelace), not a promised speed path. Native FP4/NVFP4 speed claims require future hardware/runtime validation.
- Optional converters and quant packages must never become mandatory boot dependencies.
- TODO only for now: add AMD, Intel, and CPU quant/acceleration backends after the NVIDIA path has preflight, receipts, and benchmarks.

### Sprint D docs

| Doc | Purpose |
|-----|---------|
| `docs/WORKER_PROTOCOL.md` | WorkerEvent types, streaming contract, error handling |
| `docs/OLLAMA_CHAT_TENANT.md` | Chat architecture, unload behavior, GPU lock interaction |
| `docs/TRAINING_ENGINE_ROADMAP.md` | Kohya and ED2 subprocess design, config generation |
| `docs/ACCELERATION_EXPERIMENTS.md` | Flagged experiments, benchmark protocol, status |
