# A1111 Extension Compatibility — AIWF Studio Roadmap

> The honest answer: AIWF Studio can support some A1111 extensions drop-in, but full universal support requires an A1111 compatibility shim layer that emulates the environment A1111 extensions expect.  There is a clear path to real drop-in support — it just has to be treated like a compatibility *runtime*, not simply "load their extension folder."

---

## Why Drop-In Is Hard

A1111 extensions are not plugins with a clean public API. The official extension docs define an extension as a subdirectory inside `extensions/` that may:

- Run `install.py` on first load
- Execute Python files in its `scripts/` directory as user scripts
- Inject JavaScript, CSS, and localization strings
- Use `preload.py` for early hooks
- Define metadata ordering and inter-extension dependencies

### The expected Python world

A1111 scripts assume a specific runtime environment:

| Surface | What extensions import / call |
|---|---|
| `modules.scripts.Script` | Base class — `ui()`, `run()`, `setup()`, `before_process()`, `process()`, `process_batch()`, `postprocess_batch()` |
| `modules.shared` | `opts`, `cmd_opts`, `sd_model`, `state` |
| `modules.script_callbacks` | `on_app_started`, `on_model_loaded`, `on_ui_tabs`, `on_ui_settings`, `before_image_saved`, `image_saved`, `cfg_denoiser_callback`, `on_component`, `on_infotext_pasted`, `before_token_counter`, etc. |
| `modules.processing` | `StableDiffusionProcessing`, `StableDiffusionProcessingTxt2Img`, `process_images()`, `Processed` |
| `modules.images` | `save_image()`, `flatten()` |
| `modules.paths` | Root paths, data paths |
| `modules.sd_models` | `load_model()`, `reload_model_weights()`, `get_closet_checkpoint_match()` |
| `modules.sd_samplers` | Sampler registration, lookup |
| `gradio` component injection | Extensions call `gr.Blocks()`, inject components into specific A1111 UI locations |

So "true drop-in" means AIWF Studio must provide enough of that expected world that an extension believes it is running inside A1111.

---

## The Right Architecture

Build an **A1111 Compatibility Layer** beside the native plugin system. Do not implement it inside core:

```
aiwf/
  compat/
    a1111/
      loader.py               # context-manager that installs/removes shim modules
      scanner.py              # inspect extension directory, classify, report
      shim_modules/
        modules/              # fake `modules` package injected at import time
          scripts.py          # Script base class + ScriptRunner shim
          script_callbacks.py # callback registry mapped to AIWF events
          shared.py           # opts proxy -> UserSettings, cmd_opts -> RuntimeFlags
          processing.py       # StableDiffusionProcessing adapter object
          images.py           # save_image shim -> AIWF image store
          paths.py            # path constants
          extensions.py       # extension list/state shim
          sd_models.py        # model load/reload shim -> GenerationService
          sd_samplers.py      # sampler registry shim -> AIWF samplers
          ui_components.py    # Gradio component injection helpers
      adapters/
        processing_to_generation.py  # translate p.* mutations into GenerationRequest
        gradio_mount.py              # mount extension tabs/panels into AIWF Gradio app
        callbacks_to_events.py       # route A1111 script_callbacks to AIWF event bus
        infotext_bridge.py           # A1111 infotext paste -> AIWF PNG info tags
```

**Goal:** create a fake A1111 runtime boundary that maps compatible extension behaviour into AIWF services without polluting the clean architecture.

### Conceptual loader shape

```python
with A1111CompatRuntime(ctx, extension_path) as runtime:
    runtime.load_preload_py()        # optional early hook
    runtime.load_install_metadata()  # read requirements, skip risky installs
    runtime.import_extension_scripts()
    runtime.mount_supported_ui()     # inject tabs / controls into Gradio app
    runtime.register_supported_callbacks()
```

Inside the context manager, `modules.*` is live in `sys.modules`.  On exit, it is removed so the shim never leaks into core code.

### Complete mapping table

| A1111 surface | AIWF mapping |
|---|---|
| `modules.shared.opts` | `ctx.settings` compatibility proxy |
| `modules.shared.cmd_opts` | `ctx.flags` compatibility proxy |
| `modules.shared.sd_model` | `ctx.generation.current_pipeline` proxy |
| `modules.scripts.Script` | AIWF-compatible base class with lifecycle stubs |
| `modules.script_callbacks.on_ui_tabs` | `WebRegistry.mount()` tab wrapper |
| `modules.script_callbacks.on_image_saved` | AIWF `AfterGenerate` / image-store event |
| `modules.script_callbacks.before_image_saved` | AIWF pre-save hook |
| `modules.script_callbacks.on_model_loaded` | AIWF checkpoint-load event |
| `processing.process_images(p)` | `ctx.generation.submit(GenerationRequest(...))` |
| `StableDiffusionProcessingTxt2Img` | `A1111ProcessingAdapter` wrapping `GenerationRequest` |
| `Processed` return object | Thin wrapper around `GenerationResult` |
| `modules.images.save_image` | AIWF image-store write |
| `modules.sd_models.reload_model_weights` | `GenerationService.load_checkpoint()` |

That is the bridge.

---

## The Scanner (just as important as the loader)

Before loading any extension, scan it and produce a report. The scanner should never execute extension code — only read and analyse it.

### What to inspect

```
extension/
  install.py          # risk: arbitrary pip installs, subprocess calls
  preload.py          # early import hooks
  metadata.ini        # name, version, dependencies
  scripts/*.py        # main extension logic
  javascript/*.js     # client-side; check for A1111 DOM ID selectors
  style.css           # safe to inject
  localizations/*.json  # safe
```

### What to detect (static analysis)

- Imported `modules.*` names
- Which `script_callbacks` are registered
- `scripts.Script` subclasses and which lifecycle methods they define
- Use of `AlwaysVisible` flag
- `modules.shared.opts` key access patterns
- Calls to `processing.process_images`
- Direct `torch` / sampler / UNet hooks or monkey-patches
- `launch.run_pip` or `subprocess` calls
- JavaScript selectors targeting A1111 DOM IDs/classes
- Dependency installs declared in `install.py`
- License file presence and type
- Whether the extension name matches a known curated list

### Scanner output format

```json
{
  "extension": "example-extension",
  "status": "B",
  "compatibility": "shim-compatible",
  "supported": [
    "scripts.Script.ui",
    "on_ui_tabs",
    "style.css"
  ],
  "unsupported": [
    {
      "feature": "modules.sd_hijack",
      "reason": "AIWF does not expose A1111 model hijack internals",
      "suggestion": "Use AIWF native attention optimization flags instead"
    }
  ],
  "risks": [],
  "requires_adapter": false
}
```

---

## Compatibility Levels

Classify every extension before loading it:

| Level | Label | Meaning |
|---|---|---|
| **A** | Drop-in compatible | Loads without code changes through the shim layer |
| **B** | Shim-compatible | Needs a supported shim, no extension rewrite required |
| **C** | Adapter required | Needs a small AIWF adapter file; depends on A1111 internals that must be bridged |
| **D** | Not compatible yet | Uses unsupported hooks, model hijacks, sampler internals, or frontend assumptions |
| **X** | Unsafe / blocked | Runs risky install code, arbitrary system commands, incompatible licenses, or destructive monkey-patches |

This tells users exactly what won't work, why, and what they or the project can do to make it work.

---

## What Can Probably Work First

### Easy / high-value targets (Level A–B)

These map cleanly to AIWF's existing surfaces:

- Extensions that **add tabs** — AIWF already has a Gradio app registry
- Extensions that **add Gradio controls** — standard `gr.Blocks` injection
- Extensions that **use `scripts.Script.ui()`** — shimmed into a generation panel row
- Extensions that **read/write prompts** — hook into prompt pre/post processing
- Extensions that **post-process images** — AIWF image-save pipeline
- Extensions that **add CSS/JS** — Gradio static asset injection
- Extensions that **depend mostly on `/sdapi/v1`** — AIWF has equivalent API routes
- Extensions that **add metadata/infotext behaviour** — AIWF PNG info + tags layer

AIWF already has a Gradio app registry, native/API generation routes, PNG info, image store, and an event-driven `GenerationService` — the right place to bridge extension actions safely.

### Harder targets (Level C)

Possible, but need adapter objects:

- Scripts that expect A1111 `processing` object classes (`StableDiffusionProcessing`, etc.)
- Scripts that mutate `p.prompt`, `p.seed`, `p.steps`, `p.sampler_name`
- Scripts that call `processing.process_images(p)`
- Scripts that expect A1111's `Processed` return object
- Scripts that rely on `modules.shared.opts` settings keys
- Scripts that register `script_callbacks`
- Scripts that inject UI before/after specific A1111 component element IDs

### Probably bad targets (Level D–X)

Mark as incompatible unless explicitly implemented:

- Extensions that monkey-patch A1111 sampler internals
- Extensions that hook into K-diffusion CFG denoiser internals
- Extensions that assume A1111's global `shared.sd_model` singleton
- Extensions that patch model loading, UNet replacement, hijack modules, attention optimizers, or Forge-specific internals
- Extensions that require exact A1111 DOM element IDs/classes in JavaScript
- Extensions that ship `install.py` scripts that can corrupt a clean venv

These can corrupt AIWF's clean architecture if allowed to run raw.

---

## Implementation Phases

### Phase 1 — Scanner + classification (no runtime yet)

1. `scanner.py` walks an `extensions/` directory
2. Reads `install.py`, `scripts/*.py` imports, `metadata.ini`
3. Classifies each extension A–X and emits a JSON report
4. UI shows the report before the user enables anything

### Phase 2 — Shim modules skeleton

5. `modules/shared.py` — opts + cmd_opts proxies
6. `modules/scripts.py` — Script base class + no-op lifecycle stubs
7. `modules/script_callbacks.py` — registry with safe no-ops for unsupported hooks
8. `A1111CompatRuntime` context manager with `sys.modules` injection/cleanup

### Phase 3 — Tab / UI mounting (Level A)

9. `gradio_mount.py` — detect `on_ui_tabs` callbacks, mount into AIWF Gradio app
10. CSS/JS injection via Gradio `js=` and `css=` parameters
11. First passing extensions: tab-only extensions with no generation dependency

### Phase 4 — Processing adapter (Level B–C)

12. `processing_to_generation.py` — translate `StableDiffusionProcessing` mutations into `GenerationRequest`
13. `process_images(p)` shim calls `GenerationService.generate()`
14. `Processed` return adapter wraps `GenerationResult`
15. `before_process` / `postprocess_batch` lifecycle calls through adapter

### Phase 5 — Callbacks + infotext (Level B–C)

16. Map `before_image_saved` / `image_saved` → AIWF image-save events
17. `infotext_bridge.py` — paste infotext parameters via AIWF tag parser
18. `on_model_loaded` → emit after `GenerationService` checkpoint load

### Phase 6 — API compatibility (Level A–B)

19. Verify AIWF `/api/*` routes satisfy `/sdapi/v1/*` surface
20. Add any missing aliases needed by common extensions

---

## Non-Goals

- Full monkey-patch support (sampler internals, UNet hijacking, Forge extensions)
- Running arbitrary `install.py` code into AIWF's venv
- Supporting extensions built for ComfyUI or InvokeAI
- Maintaining full parity with every A1111 version

---

## Open Questions

- What is the sandboxing boundary for extension `install.py`? (Isolated subprocess? Dry-run with dependency list output only?)
- Should the shim `modules` package live at `aiwf.compat.a1111.shim_modules.modules` and be aliased as `modules` only inside the runtime context?
- Which AIWF event bus events map cleanly to A1111 `script_callbacks`?
- Can extension CSS/JS be scoped so it doesn't leak into AIWF's own UI?

---

*Document started by Nawnie. Compiled 2026-06-14.*
