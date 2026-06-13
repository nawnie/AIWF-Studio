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
| Model downloads | `aiwf/web/tabs/model_manager.py`, `aiwf/services/model_download.py` |
| API | `aiwf/api/v1/routes.py` |
| Settings / launch | `aiwf/web/tabs/settings.py`, `aiwf/core/config/launch.py` |
| Bootstrap wiring | `aiwf/bootstrap.py` |

## Imported Claude Cowork project instructions
