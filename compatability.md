# AIWF Studio A1111 Extension Compatibility Roadmap

> Filename note: this file intentionally keeps the branch-requested name `compatability.md`. The project should use the standard spelling, **compatibility**, in user-facing copy and code names.

## Purpose

AIWF Studio is an original local-first creative AI workspace inspired by AUTOMATIC1111, with familiar launch conventions and A1111-compatible API routes. The goal of this roadmap is to define a realistic path toward **measured A1111 extension compatibility** without turning AIWF Studio into a clone of A1111's internal architecture.

The compatibility goal is not:

- "all A1111 extensions magically work";
- importing A1111 global-state architecture;
- allowing unsafe monkey patches into AIWF core;
- breaking AIWF's service-based design.

The compatibility goal is:

- scan A1111 extensions before loading them;
- classify what can work, what cannot, and why;
- support safe drop-in behavior where practical;
- provide shims for common A1111 extension APIs;
- give users clear reports and migration paths when an extension needs an adapter.

## Current AIWF Position

AIWF Studio already has several compatibility foundations:

- familiar `webui.bat` / `webui-user.bat` launch convention;
- familiar Stable Diffusion model folder layout;
- native `/api/v1` routes;
- A1111-compatible `/sdapi/v1` adapter routes;
- Gradio UI shell;
- service-routed generation layer;
- plugin registry;
- event bus;
- image metadata and PNG Info direction;
- ControlNet, LoRA, VAE, inpaint, face swap, segment, workflow, and video tabs.

This means AIWF can support A1111 compatibility at multiple levels:

1. **Workflow compatibility** — users recognize the concepts and folders.
2. **Asset compatibility** — checkpoints, LoRAs, VAEs, embeddings, ControlNet models.
3. **API compatibility** — A1111-style tools talking to `/sdapi/v1`.
4. **Extension compatibility** — selected A1111 extensions loaded through a compatibility runtime.

This roadmap focuses on layer 4.

## Compatibility Classes

Every scanned A1111 extension should receive a compatibility class.

| Class | Label | Meaning | User Message |
|---|---|---|---|
| **A** | Drop-in compatible | Loads and runs through AIWF's compatibility layer without extension code changes. | "Works in AIWF Studio." |
| **B** | Shim-compatible | Uses known A1111 APIs that AIWF can emulate through supported shims. | "Works through compatibility shims." |
| **C** | Adapter required | Needs a small AIWF adapter because it depends on A1111 internals. | "Needs an AIWF adapter file." |
| **D** | Not compatible yet | Uses unsupported hooks, deep model internals, DOM assumptions, or lifecycle paths. | "Not supported yet; report explains why." |
| **X** | Unsafe / blocked | Runs risky install code, destructive monkey patches, unknown binaries, unsafe commands, or incompatible licenses. | "Blocked for safety." |

## Extension Doctor

The first milestone is an **Extension Doctor** that scans an extension without executing it.

### Scanner Inputs

The scanner should inspect:

```text
extension-root/
  install.py
  preload.py
  metadata.ini
  scripts/*.py
  javascript/*.js
  style.css
  localizations/*.json
  README.md
  requirements.txt
```

### Scanner Signals

The scanner should detect:

- extension name and path;
- metadata dependencies and ordering rules;
- Python imports from `modules.*`;
- use of `modules.scripts.Script`;
- use of `modules.shared`, `shared.opts`, `shared.cmd_opts`, or `shared.sd_model`;
- use of `modules.script_callbacks` hooks;
- use of `modules.processing.process_images`;
- use of `StableDiffusionProcessingTxt2Img` / `StableDiffusionProcessingImg2Img`;
- use of A1111 image helpers such as `modules.images`;
- use of A1111 infotext helpers;
- use of ControlNet extension-specific APIs;
- JavaScript selectors that target A1111 DOM IDs or classes;
- CSS files;
- localization files;
- direct `torch`, `k_diffusion`, sampler, UNet, VAE, hijack, or optimizer imports;
- calls to `launch.run_pip`, `subprocess`, `os.system`, shell scripts, or network installers;
- binary files or unusual executable payloads;
- license and attribution hints.

### Scanner Output

The scanner should emit JSON and a human-readable report.

Example JSON:

```json
{
  "extension": "example-extension",
  "path": "extensions/example-extension",
  "status": "B",
  "label": "shim-compatible",
  "summary": "Uses Script.ui and on_ui_tabs; no unsafe install behavior detected.",
  "supported": [
    "scripts.Script.ui",
    "script_callbacks.on_ui_tabs",
    "style.css"
  ],
  "unsupported": [
    {
      "feature": "modules.processing.process_images",
      "reason": "AIWF processing-object adapter is not implemented yet.",
      "suggested_fix": "Target Phase 4 processing adapter or create an AIWF-native plugin wrapper."
    }
  ],
  "risk": "low",
  "can_attempt_load": true
}
```

Example user report:

```text
Extension: example-extension
Status: B - Shim-compatible
Risk: low

Can load:
- CSS assets
- JavaScript assets
- UI tab callback

Needs shim support:
- modules.processing.process_images

Why:
- The extension expects A1111 processing objects. AIWF uses GenerationRequest.

Suggested fix:
- Enable the processing adapter milestone or write a small AIWF adapter.
```

## Compatibility Runtime Design

AIWF should add a compatibility runtime under a contained namespace:

```text
aiwf/compat/a1111/
  __init__.py
  loader.py
  scanner.py
  report.py
  classes.py
  runtime.py
  safety.py
  shim_modules/
    modules/
      __init__.py
      scripts.py
      shared.py
      script_callbacks.py
      processing.py
      images.py
      paths.py
      extensions.py
      sd_models.py
      sd_samplers.py
      ui_components.py
  adapters/
    processing_to_generation.py
    gradio_mount.py
    callbacks_to_events.py
    infotext_bridge.py
```

The runtime should load extensions inside a controlled boundary:

```text
AIWF AppContext
-> A1111CompatRuntime
-> temporary modules.* shim tree
-> extension scanner report
-> safe loader
-> supported UI/callback mounting
-> AIWF service calls
```

The compatibility layer must not become AIWF core. It should be an adapter boundary.

## Shim Targets

### Phase 1 Shim: `modules.shared`

Expose a compatibility proxy for common A1111 globals.

Map:

```text
modules.shared.opts      -> ctx.settings compatibility proxy
modules.shared.cmd_opts  -> ctx.flags compatibility proxy
modules.shared.state     -> AIWF job/progress state proxy
modules.shared.sd_model  -> blocked or read-only model proxy initially
```

Rules:

- `opts` may read/write safe settings only.
- `cmd_opts` should be mostly read-only.
- `sd_model` should not expose direct mutation at first.
- unsupported attributes should report clear compatibility warnings.

### Phase 2 Shim: `modules.scripts`

Support common script extension patterns.

Target support:

- `scripts.Script` base class;
- `title()`;
- `show()`;
- `ui()`;
- `run()` where possible;
- `AlwaysVisible`;
- basic `ScriptRunner` lifecycle facade.

Initial behavior:

- UI-only scripts can mount into an "A1111 Extensions" panel.
- unsupported run hooks produce clear reports rather than crashing.

### Phase 3 Shim: `modules.script_callbacks`

Support safe callback hooks first.

Early supported callbacks:

- `on_app_started`;
- `on_ui_tabs`;
- `on_ui_settings`;
- `on_before_image_saved`;
- `on_image_saved`;
- `on_infotext_pasted`.

Delayed / advanced callbacks:

- `on_cfg_denoiser`;
- `on_cfg_denoised`;
- `on_cfg_after_cfg`;
- `on_list_optimizers`;
- `on_list_unets`;
- sampler or model hijack callbacks.

Reason:

These advanced callbacks touch model internals and sampler execution. They should not be enabled until AIWF has explicit backend adapter contracts.

### Phase 4 Shim: `modules.processing`

This is the most important deep-compatibility adapter.

Provide compatibility wrappers:

```text
StableDiffusionProcessing
StableDiffusionProcessingTxt2Img
StableDiffusionProcessingImg2Img
Processed
process_images(p)
```

Map A1111 processing fields to AIWF:

```text
p.prompt              -> GenerationRequest.prompt
p.negative_prompt     -> GenerationRequest.negative_prompt
p.steps               -> GenerationRequest.steps
p.cfg_scale           -> GenerationRequest.cfg_scale
p.width / p.height    -> GenerationRequest.width / height
p.seed                -> GenerationRequest.seed
p.sampler_name        -> GenerationRequest.sampler
p.batch_size          -> GenerationRequest.batch_size
p.n_iter              -> GenerationRequest.batch_count
p.denoising_strength  -> GenerationRequest.denoising_strength
p.init_images         -> init_images
p.mask                -> mask_images
```

Map AIWF results back to A1111-shaped results:

```text
GenerationResult.images     -> Processed.images
GenerationResult.seeds      -> Processed.all_seeds
GenerationResult.infotexts  -> Processed.infotexts
GenerationResult.artifacts  -> saved paths / metadata
```

### Phase 5 Shim: Assets

Support non-Python extension assets:

- `style.css`;
- `javascript/*.js`;
- localizations;
- metadata display;
- README display;
- license display.

Caution:

A1111 JavaScript often targets specific DOM IDs/classes. AIWF should load JS only if the scanner marks selectors as low risk or the user explicitly enables it.

### Phase 6 Shim: ControlNet Extension Bridge

Many A1111 extensions depend on ControlNet extension behavior.

AIWF should provide:

- ControlNet model list adapter;
- ControlNet module list adapter;
- detect/preprocess adapter;
- alwayson script payload parsing;
- compatibility reports for unsupported multi-unit or advanced options.

AIWF already has a ControlNet service and `/sdapi/v1/controlnet/*` direction, so this is a high-value bridge.

## Safety Policy

Drop-in extension support is risky because extensions can execute arbitrary Python.

AIWF should default to **scan-first, execute-later**.

### Block by Default

Block or warn heavily on:

- `install.py` that calls pip automatically;
- `subprocess` shell commands;
- `os.system`;
- remote downloads at import time;
- binary payloads;
- monkey patches to `torch`, `diffusers`, `gradio`, or AIWF internals;
- writes outside the extension folder or AIWF data folders;
- model hijack patches;
- sampler replacement patches;
- license conflicts.

### Allow by Default

Allow when scanner risk is low:

- metadata;
- README/license reading;
- CSS with no dangerous imports;
- localization files;
- static JS when selectors are scoped or user-approved;
- Python UI scripts using known safe APIs.

### User Controls

Expose these controls:

- Scan extension;
- View report;
- Load once;
- Trust this extension;
- Disable extension;
- Open logs;
- Create adapter stub;
- Mark as incompatible.

## Extension Manager UI

Add an Extensions tab or subsection.

Recommended table:

| Extension | Status | Risk | Load | Reason |
|---|---|---|---|---|
| prompt-tools | A | Low | Enabled | UI script + prompt helpers only |
| example-controlnet-tool | B | Medium | Manual | Needs ControlNet shim |
| sampler-hack | D | High | Disabled | Unsupported CFG denoiser hook |
| model-hijack-old | X | Critical | Blocked | Monkey-patches model globals |

Per-extension detail view:

- Summary;
- Supported APIs;
- Unsupported APIs;
- unsafe patterns;
- suggested adapter path;
- logs;
- file list;
- README/license panel.

## Adapter Stubs

For Class C extensions, AIWF should generate an adapter stub.

Example:

```python
# extensions/example-extension/aiwf_adapter.py
from aiwf.plugins import AIWFPlugin

class ExampleExtensionAdapter(AIWFPlugin):
    name = "example-extension-adapter"

    def register(self, ctx):
        # Map extension behavior to AIWF services here.
        pass
```

The adapter report should tell the user exactly what to implement.

## Development Phases

### Phase 0 — Branch and Roadmap

Status: started on `feature/a1111-extension-compat`.

Deliverables:

- `compatability.md` roadmap;
- README note linking to compatibility roadmap;
- issue checklist or project tasks.

### Phase 1 — Static Extension Doctor

Deliverables:

- `aiwf/compat/a1111/scanner.py`;
- dataclasses/enums for compatibility class, risk level, findings;
- CLI command or dev function to scan an extension folder;
- JSON report output;
- Markdown/text report output;
- tests with fake extension fixtures.

Acceptance criteria:

- scanning never executes extension Python;
- detects key imports and risky commands;
- reports A/B/C/D/X status;
- unit tests cover at least five fake extension types.

### Phase 2 — Extension Manager UI Skeleton

Deliverables:

- Extensions tab;
- scan folder button;
- compatibility table;
- report viewer;
- enable/disable state stored in config;
- no Python execution yet.

Acceptance criteria:

- user can scan installed extensions;
- UI explains why an extension will or will not load;
- no unsafe code runs.

### Phase 3 — Static Assets Support

Deliverables:

- CSS loader for trusted extensions;
- JS loader with warning and selector/risk report;
- localization file discovery;
- metadata display;
- README/license display.

Acceptance criteria:

- low-risk CSS can load into the Gradio app;
- JS requires explicit trust unless known safe;
- assets are isolated and traceable to extension name.

### Phase 4 — UI Script Shim

Deliverables:

- `modules.scripts.Script` shim;
- controlled import environment;
- `ui()` mounting into AIWF Extension panel;
- `AlwaysVisible` support;
- basic Gradio component passthrough.

Acceptance criteria:

- simple A1111 UI scripts can render controls;
- unsupported script methods produce warnings, not crashes;
- scanner and loader agree on compatibility status.

### Phase 5 — Callback Bridge

Deliverables:

- `modules.script_callbacks` shim;
- support safe callbacks;
- map image-save callbacks to AIWF generation/storage events;
- map UI tab/settings callbacks into AIWF WebRegistry.

Acceptance criteria:

- simple callback-based extensions can register tabs/settings;
- image-save callbacks receive useful metadata;
- advanced sampler/model callbacks are blocked with clear reasons.

### Phase 6 — Processing Adapter

Deliverables:

- A1111 processing object wrappers;
- `process_images(p)` adapter;
- mapping to `GenerationRequest`;
- `Processed` wrapper;
- support txt2img/img2img/inpaint basics.

Acceptance criteria:

- simple `run(p, *args)` extensions can generate images through AIWF;
- prompt, seed, sampler, steps, CFG, size, batch, denoise, init image, and mask map correctly;
- unsupported fields are reported.

### Phase 7 — ControlNet Compatibility

Deliverables:

- ControlNet payload adapter;
- model/module list compatibility;
- alwayson script parsing;
- detector/preprocessor bridge;
- compatibility report for unsupported units/options.

Acceptance criteria:

- common ControlNet extension payloads route to AIWF ControlNet service;
- unsupported ControlNet options are visible in reports.

### Phase 8 — Real Extension Test Matrix

Deliverables:

- curated list of target A1111 extensions;
- scan reports committed under `docs/compatibility/extensions/`;
- status table;
- adapter notes.

Suggested first targets:

- simple prompt helper extension;
- simple UI tab extension;
- PNG/metadata helper;
- API-only tool;
- ControlNet-adjacent tool;
- one known incompatible sampler/model hijack extension for negative testing.

Acceptance criteria:

- at least 10 extensions scanned;
- at least 3 low-risk extensions load or partially load;
- every failure has a clear reason and next step.

## Code Quality Rules

- AIWF core must not import A1111 shim modules.
- Compatibility code lives under `aiwf/compat/a1111/`.
- Unsupported APIs must fail soft with reports, not random tracebacks.
- Scanner must run without importing extension code.
- Loader must never run untrusted install scripts automatically.
- Every loaded extension must be traceable in logs.
- User must be able to disable compatibility loading entirely.

## Public Messaging

Recommended README language:

> AIWF Studio is designed for A1111-style workflow compatibility, including familiar launch conventions, familiar model folders, `/sdapi/v1` routes, and a planned extension compatibility layer. Extensions will be scanned and classified before loading. Safe UI/API extensions should be bridgeable; extensions that depend on A1111 internals will receive compatibility reports and may require AIWF adapters.

Avoid saying:

- "all A1111 extensions work";
- "drop-in support for everything";
- "A1111 clone";
- "full extension compatibility" before the scanner and loader prove it.

## First Implementation Checklist

- [ ] Add `aiwf/compat/a1111/classes.py` with compatibility enums and report models.
- [ ] Add `aiwf/compat/a1111/scanner.py` with static file scanning.
- [ ] Add fake extension fixtures under tests.
- [ ] Add scanner tests for Class A/B/C/D/X examples.
- [ ] Add CLI/dev command to scan an extension path.
- [ ] Add Extensions tab skeleton.
- [ ] Add JSON + Markdown report output.
- [ ] Add README link to this roadmap.
- [ ] Build initial `modules.shared` and `modules.scripts` shims only after scanner is stable.

## Success Definition

This feature succeeds when a user can drop an A1111 extension folder into AIWF Studio and get one of these outcomes:

1. it loads safely;
2. it partially loads with an accurate warning;
3. it is blocked with a clear reason;
4. it gets an adapter stub explaining how to make it work.

No silent failure. No mystery crashes. No unsafe extension execution by default.

That is the AIWF version of drop-in support: **measured compatibility with honest diagnostics.**
