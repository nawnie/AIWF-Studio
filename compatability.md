# AIWF Studio A1111 Extension Compatibility Roadmap

> Filename note: this file intentionally keeps the branch-requested name `compatability.md`. User-facing copy and code should use the standard spelling: **compatibility**.

## Living Roadmap Policy

This file is the living roadmap for the `feature/a1111-extension-compat` branch.

Every substantial design pass should update this file with:

1. the new decision;
2. the reason for the decision;
3. the exact implementation impact;
4. the test or acceptance criteria needed to prove it works.

Do not treat compatibility as a vague promise. The goal is measurable compatibility with clear diagnostics.

---

# Pass 001 — Implementation-Grade Compatibility Plan

## Core Strategy

AIWF Studio should not pretend to be AUTOMATIC1111 internally. AIWF should emulate the A1111 extension contract at the boundary, classify compatibility before execution, and give users a precise failure report or adapter stub whenever an extension crosses into unsupported internals.

The goal is not to blindly run every A1111 extension.

The goal is to build a compatibility system where a user can drop an A1111 extension folder into AIWF Studio and get one of four outcomes:

1. **It works.**
2. **It partially works and AIWF explains what is missing.**
3. **It needs an adapter and AIWF tells the developer exactly what to write.**
4. **It is blocked because it is unsafe or depends on internals AIWF intentionally does not expose.**

AIWF's position:

> AIWF Studio supports A1111-style workflows, assets, launch conventions, and API compatibility. Native A1111 extension support is handled through a scan-first compatibility layer, not by copying A1111 internals into AIWF core.

The compatibility layer must preserve AIWF's architecture:

```text
AIWF core stays clean.
A1111 compatibility lives behind an adapter boundary.
Unsafe extension behavior is scanned and reported before execution.
```

---

# 1. Definitions

## 1.1 A1111 Extension

For AIWF purposes, an A1111 extension is a folder with one or more of these:

```text
extension-root/
  install.py
  preload.py
  metadata.ini
  scripts/
    *.py
  javascript/
    *.js
  localizations/
    *.json
  style.css
  README.md
  requirements.txt
```

## 1.2 AIWF Native Plugin

An AIWF plugin is clean plugin code written for AIWF's own services, event bus, UI registry, and app context.

## 1.3 A1111 Compatibility Extension

An A1111 extension loaded through a fake `modules.*` compatibility environment.

## 1.4 Extension Doctor

A static scanner that reads an extension folder and produces:

```text
compatibility class
risk class
supported features
unsupported features
dangerous patterns
load recommendation
developer fix instructions
```

The scanner must **not execute extension Python**.

---

# 2. Compatibility Classes

Every extension gets exactly one compatibility class.

| Class | Name | Meaning |
|---|---|---|
| **A** | Drop-in compatible | Can load and run through current AIWF compatibility shims. |
| **B** | Shim-compatible | Uses known A1111 APIs that AIWF can emulate, but needs a shim not yet complete or not yet enabled. |
| **C** | Adapter required | Depends on A1111 internals enough that a small AIWF adapter is needed. |
| **D** | Unsupported yet | Uses deep hooks or unsupported features that AIWF should not emulate until a future milestone. |
| **X** | Blocked / unsafe | Uses dangerous install/runtime behavior, destructive monkey patches, unknown binaries, or incompatible assumptions. |

User-facing wording:

```text
A — Works in AIWF Studio.
B — Should work once the listed shim is enabled.
C — Needs an AIWF adapter.
D — Not supported yet; reason shown.
X — Blocked for safety.
```

---

# 3. Risk Classes

Compatibility is not the same as safety.

An extension can be technically compatible but unsafe.

| Risk | Meaning |
|---|---|
| **Low** | Static assets, UI only, no install script, no shell execution. |
| **Medium** | Uses Python scripts and known A1111 APIs, but no obvious system-level behavior. |
| **High** | Uses subprocess, install scripts, monkey patches, direct model internals, or remote downloads. |
| **Critical** | Runs unknown binaries, modifies files outside allowed folders, patches Torch/Diffusers/AIWF globally, or hides execution. |

Rule:

```text
Risk Low      -> can offer safe load
Risk Medium   -> require user approval
Risk High     -> default disabled
Risk Critical -> block
```

---

# 4. Legacy AUTOMATIC1111 Theme / Layout Track

## Purpose

AIWF Studio can honor A1111 without being a clone.

The original WebUI can be downloaded locally as a reference for behavior, layout vocabulary, and user comfort. The legacy layout should be implemented as an AIWF-native theme/layout option, not by copying incompatible source code.

## Positioning

AIWF Studio should say:

> AIWF Studio is inspired by AUTOMATIC1111 and includes an optional legacy-style layout for users who prefer the original WebUI workflow. The app remains original clean-room code with AIWF's own architecture and services.

## Implementation Rule

The legacy layout should be a theme and layout preset:

```text
AIWF current architecture stays intact.
Legacy layout changes arrangement, labels, density, and visual style.
It does not replace AIWF services with A1111 internals.
```

## Reference Workflow

The developer may keep a local checkout of the A1111 repository outside AIWF Studio:

```powershell
git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git .reference\stable-diffusion-webui
```

The `.reference/` folder must not be committed.

Add to `.gitignore` if needed:

```gitignore
.reference/
reference/
upstream/
```

## Theme Target Files

Recommended AIWF-native files:

```text
aiwf/web/layouts/
  __init__.py
  registry.py
  classic_a1111.py

aiwf/web/themes/
  classic_a1111.css
  classic_a1111.py

static/
  classic-a1111.css
```

## Settings Model

Add a user setting:

```python
ui_layout: Literal["studio", "classic_a1111"] = "studio"
```

Optional related settings:

```python
ui_density: Literal["comfortable", "compact"] = "comfortable"
show_legacy_labels: bool = False
show_advanced_first: bool = False
```

## UI Behavior

Classic A1111 layout should prioritize:

- prompt and negative prompt at top;
- generation controls visible in a familiar block;
- txt2img/img2img/inpaint as familiar modes;
- output gallery under controls;
- compact accordions;
- familiar labels such as Steps, Sampler, CFG Scale, Seed, Size, Hires Fix;
- optional A1111-style send-to buttons;
- familiar PNG Info restore flow;
- extension controls grouped in an A1111-like panel.

## What Not To Do

Do not:

- copy A1111 source code into AIWF;
- import A1111 Python modules directly;
- reproduce license-incompatible assets;
- make classic layout the default identity;
- break the current AIWF Studio layout.

## Acceptance Criteria

Classic layout is done when:

- user can switch between Studio layout and Classic A1111-inspired layout in Settings;
- both layouts call the same AIWF services;
- current image generation still works;
- PNG Info and send-to flows work in both layouts;
- Extension Doctor panel still works;
- README clearly describes it as "inspired by" and "familiar layout," not a clone.

---

# 5. Proposed File Layout

Create this structure:

```text
aiwf/
  compat/
    __init__.py
    a1111/
      __init__.py
      classes.py
      scanner.py
      report.py
      safety.py
      loader.py
      runtime.py
      metadata.py
      registry.py
      paths.py
      adapter_stub.py

      adapters/
        __init__.py
        processing_to_generation.py
        callbacks_to_events.py
        gradio_mount.py
        infotext_bridge.py
        controlnet_bridge.py

      shim_modules/
        modules/
          __init__.py
          scripts.py
          shared.py
          script_callbacks.py
          processing.py
          images.py
          paths.py
          sd_models.py
          sd_samplers.py
          extensions.py
          errors.py
          ui_components.py
          generation_parameters_copypaste.py
          infotext_utils.py

aiwf/web/tabs/
  extensions.py

aiwf/web/layouts/
  __init__.py
  registry.py
  classic_a1111.py

aiwf/web/themes/
  classic_a1111.py

static/
  classic-a1111.css

docs/
  compatibility/
    README.md
    a1111-extension-status.md
    extension-report-schema.json
    examples/
      class-a-ui-only.md
      class-b-processing-needed.md
      class-c-adapter-needed.md
      class-x-blocked.md

tests/
  compat/
    a1111/
      fixtures/
        ui_only_extension/
        css_js_extension/
        processing_extension/
        callback_extension/
        unsafe_install_extension/
        model_hijack_extension/
      test_scanner.py
      test_reports.py
      test_metadata.py
      test_safety.py
      test_processing_adapter.py
      test_callback_bridge.py
```

---

# 6. Phase 1 — Data Models

## File

```text
aiwf/compat/a1111/classes.py
```

## Required Enums

```python
from enum import Enum


class CompatClass(str, Enum):
    DROP_IN = "A"
    SHIM_COMPATIBLE = "B"
    ADAPTER_REQUIRED = "C"
    UNSUPPORTED = "D"
    BLOCKED = "X"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FindingSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    BLOCKER = "blocker"


class FeatureStatus(str, Enum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    BLOCKED = "blocked"
```

## Required Models

Use Pydantic because AIWF already uses Pydantic domain models.

```python
from pydantic import BaseModel, Field


class ExtensionFinding(BaseModel):
    code: str
    severity: FindingSeverity
    message: str
    path: str | None = None
    line: int | None = None
    feature: str | None = None
    suggested_fix: str | None = None


class ExtensionFeature(BaseModel):
    name: str
    status: FeatureStatus
    details: str = ""
    required_phase: str | None = None


class ExtensionReport(BaseModel):
    name: str
    path: str
    compat_class: CompatClass
    risk_level: RiskLevel
    summary: str

    detected_files: list[str] = Field(default_factory=list)
    imports: list[str] = Field(default_factory=list)
    callbacks: list[str] = Field(default_factory=list)
    script_classes: list[str] = Field(default_factory=list)

    supported_features: list[ExtensionFeature] = Field(default_factory=list)
    partial_features: list[ExtensionFeature] = Field(default_factory=list)
    unsupported_features: list[ExtensionFeature] = Field(default_factory=list)
    blocked_features: list[ExtensionFeature] = Field(default_factory=list)

    findings: list[ExtensionFinding] = Field(default_factory=list)

    can_attempt_load: bool = False
    requires_user_trust: bool = False
    requires_adapter: bool = False
```

## Acceptance Criteria

Run:

```powershell
python -m pytest tests/compat/a1111/test_reports.py -q
```

Verify:

- every report serializes to JSON;
- every report can be rendered to Markdown;
- blocked reports set `can_attempt_load = False`;
- adapter-required reports set `requires_adapter = True`.

---

# 7. Phase 2 — Static Scanner

## File

```text
aiwf/compat/a1111/scanner.py
```

## Scanner API

```python
from pathlib import Path
from aiwf.compat.a1111.classes import ExtensionReport


def scan_extension(path: Path) -> ExtensionReport:
    ...
```

## Scanner Must Inspect

```text
install.py
preload.py
metadata.ini
requirements.txt
scripts/*.py
javascript/*.js
style.css
localizations/*.json
README.md
LICENSE
```

## Scanner Must Not Execute

Do not use:

```python
import extension_code
exec(extension_code)
runpy.run_path(...)
subprocess.run(...)
pip install ...
```

## Python AST Rules

Use Python `ast` to inspect `.py` files.

Detect imports:

```python
from modules import scripts
from modules import shared
from modules import script_callbacks
from modules.processing import process_images
from modules.processing import StableDiffusionProcessingTxt2Img
from modules.processing import StableDiffusionProcessingImg2Img
from modules import sd_hijack
from modules import sd_models
from modules import images
from modules import paths
```

Detect risky imports:

```python
import subprocess
import os
import socket
import urllib
import requests
import torch
import k_diffusion
import launch
```

Detect risky calls:

```python
subprocess.run(...)
subprocess.call(...)
subprocess.Popen(...)
os.system(...)
os.remove(...)
os.rmdir(...)
shutil.rmtree(...)
launch.run_pip(...)
pip.main(...)
requests.get(...)
urllib.request.urlretrieve(...)
```

Do not automatically classify `requests.get` as blocked. It may be legitimate. Classify as high risk unless it is clearly only used after user action.

## Script Class Detection

Detect classes that inherit from:

```python
scripts.Script
modules.scripts.Script
Script
```

Inspect methods:

```text
title
show
ui
run
setup
before_process
process
before_process_batch
process_batch
postprocess_batch
postprocess_batch_list
postprocess_image
postprocess
before_component
after_component
before_hr
on_mask_blend
post_sample
```

## Callback Detection

Detect calls like:

```python
script_callbacks.on_app_started(...)
script_callbacks.on_ui_tabs(...)
script_callbacks.on_ui_settings(...)
script_callbacks.on_before_image_saved(...)
script_callbacks.on_image_saved(...)
script_callbacks.on_infotext_pasted(...)
script_callbacks.on_cfg_denoiser(...)
script_callbacks.on_cfg_denoised(...)
script_callbacks.on_cfg_after_cfg(...)
script_callbacks.on_list_optimizers(...)
script_callbacks.on_list_unets(...)
```

## Metadata Scan

Parse:

```text
metadata.ini
```

Detect:

```text
[Extension]
Name =
Requires =

[scripts]
Requires =
Before =
After =

[scripts/file.py]
Requires =
Before =
After =

[callbacks/...]
Before =
After =
```

## Asset Scan

Detect:

```text
javascript/*.js
style.css
localizations/*.json
```

For JavaScript, detect A1111 DOM assumptions:

```text
txt2img_prompt
img2img_prompt
txt2img_gallery
img2img_gallery
setting_
quicksettings
tabs
script_
```

Classify:

```text
JS only + no A1111 DOM selectors -> low risk
JS with A1111 DOM selectors -> medium/high, likely adapter required
JS that calls fetch('/sdapi/v1/...') -> likely bridgeable
```

## Scanner Classification Logic

### Class A

Only uses:

```text
style.css
localizations
README/license
simple javascript
Script.ui
Script.title
Script.show
on_ui_tabs
on_ui_settings
```

No risky imports.

### Class B

Uses:

```text
modules.shared.opts
modules.shared.cmd_opts
modules.processing.process_images
Script.run
before_image_saved
image_saved
infotext_pasted
```

But does not monkey-patch model/sampler internals.

### Class C

Uses:

```text
direct DOM assumptions
custom JS tied to A1111 UI IDs
custom processing object fields not mapped yet
ControlNet alwayson scripts beyond current support
A1111 image helpers that need mapping
```

### Class D

Uses:

```text
cfg denoiser hooks
latent hooks
sampler internals
post_sample tensor hooks
model_loaded hook requiring real sd_model
optimizer/unet list hooks
sd_hijack
```

### Class X

Uses:

```text
unsafe install behavior
unknown binary execution
destructive file operations
hidden network execution
global monkey patches to torch/diffusers/gradio/AIWF
license conflict
```

## Test Fixtures

Create:

```text
tests/compat/a1111/fixtures/ui_only_extension/
tests/compat/a1111/fixtures/css_js_extension/
tests/compat/a1111/fixtures/processing_extension/
tests/compat/a1111/fixtures/callback_extension/
tests/compat/a1111/fixtures/unsafe_install_extension/
tests/compat/a1111/fixtures/model_hijack_extension/
```

Example `ui_only_extension/scripts/ui_tool.py`:

```python
import gradio as gr
from modules import scripts

class UiTool(scripts.Script):
    def title(self):
        return "UI Tool"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        enabled = gr.Checkbox(label="Enable UI Tool", value=False)
        strength = gr.Slider(0, 1, value=0.5, label="Strength")
        return [enabled, strength]
```

Expected:

```text
Class A or B depending current shim level
Risk low
No blockers
```

Example unsafe:

```python
# install.py
import launch
launch.run_pip("install random-package", "extension dependency")
```

Expected:

```text
Class X until user explicitly allows install scripts
Risk high or critical
can_attempt_load false
```

---

# 8. Phase 3 — Report Renderer

## File

```text
aiwf/compat/a1111/report.py
```

## Functions

```python
def render_markdown(report: ExtensionReport) -> str:
    ...


def render_text(report: ExtensionReport) -> str:
    ...


def write_report(report: ExtensionReport, output_dir: Path) -> None:
    ...
```

## Markdown Format

```markdown
# Extension Compatibility Report: <name>

## Summary

Status: B — Shim-compatible  
Risk: medium  
Can attempt load: yes  
Requires adapter: no  

## Supported

- scripts.Script.ui
- script_callbacks.on_ui_tabs

## Unsupported

### modules.processing.process_images

Reason: AIWF processing adapter is not implemented yet.

Suggested fix:
Implement Phase 6 processing adapter or create an AIWF adapter.

## Safety Findings

- No install.py found.
- No subprocess usage found.
```

## Acceptance Criteria

Running:

```powershell
python -m aiwf.compat.a1111 scan .\extensions\some-extension --markdown
```

prints a human-readable report.

Running:

```powershell
python -m aiwf.compat.a1111 scan .\extensions\some-extension --json
```

prints machine-readable JSON.

---

# 9. Phase 4 — CLI / Dev Command

## File

```text
aiwf/compat/a1111/__main__.py
```

## Usage

```powershell
python -m aiwf.compat.a1111 scan .\extensions\sd-webui-example
python -m aiwf.compat.a1111 scan-all .\extensions
python -m aiwf.compat.a1111 scan .\extensions\sd-webui-example --json
python -m aiwf.compat.a1111 scan .\extensions\sd-webui-example --markdown
python -m aiwf.compat.a1111 scan-all .\extensions --out .\reports
```

## Acceptance Criteria

A developer can run:

```powershell
python -m aiwf.compat.a1111 scan tests\compat\a1111\fixtures\ui_only_extension
```

and see a report without launching Gradio or loading Torch.

---

# 10. Phase 5 — Extension Manager UI Skeleton

## File

```text
aiwf/web/tabs/extensions.py
```

Register it in:

```text
aiwf/web/app.py
```

Add:

```python
from aiwf.web.tabs.extensions import register_extensions
```

Then mount:

```python
register_extensions(registry)
```

## UI Layout

Tab name:

```text
Extensions
```

Sections:

```text
A1111 Extension Doctor
Installed extension folders
Compatibility report
Risk findings
Actions
```

## UI Controls

```text
Extension root path textbox
Scan extension button
Scan all button
Report table
Report markdown viewer
Trust extension checkbox
Enable extension checkbox
Create adapter stub button
Open extension folder button
```

## UI Table Columns

```text
Name
Class
Risk
Can Load
Needs Adapter
Blocked Reason
Path
```

## Acceptance Criteria

No extension Python executes in this phase.

The tab only scans and reports.

---

# 11. Phase 6 — Safe Asset Loading

## Goal

Support extensions that only add static assets:

```text
style.css
javascript/*.js
localizations/*.json
```

## Files

```text
aiwf/compat/a1111/assets.py
aiwf/compat/a1111/safety.py
```

## CSS Loading Rules

```text
Only load CSS after scan.
Only load CSS from extension folder.
Log extension name and CSS path.
Allow disabling per extension.
```

## JavaScript Loading Rules

```text
JS never autoloads until user trusts extension.
JS report must list likely DOM selectors.
JS should be wrapped/scoped if possible.
```

## Localizations

Store discovered localization files but do not merge into settings until AIWF has a localization system.

Classify as:

```text
detected, safe, not active yet
```

---

# 12. Phase 7 — Fake `modules` Import Runtime

## Goal

Many A1111 extensions import:

```python
from modules import scripts, shared, script_callbacks
```

AIWF must provide a temporary fake `modules` package only while loading A1111 extensions.

## Rule

Do not create uncontrolled permanent pollution.

Better pattern:

```python
with A1111CompatRuntime(ctx, extension_path):
    load_extension_scripts()
```

## File

```text
aiwf/compat/a1111/runtime.py
```

## Runtime Behavior

```python
class A1111CompatRuntime:
    def __init__(self, ctx, extension_path: Path, report: ExtensionReport):
        ...

    def __enter__(self):
        # install temporary modules.* shims into sys.modules
        # prepend extension path to sys.path
        # set current extension basedir
        ...

    def __exit__(self, exc_type, exc, tb):
        # remove temporary shims if safe
        # restore sys.path
        # clear basedir
        ...
```

## Safety Rule

Do not run runtime if:

```text
report.compat_class == X
report.risk_level == critical
report.can_attempt_load == false
```

unless developer mode explicitly overrides.

---

# 13. Phase 8 — `modules.shared` Shim

## File

```text
aiwf/compat/a1111/shim_modules/modules/shared.py
```

## Required Objects

```python
opts
cmd_opts
state
sd_model
```

## `opts` Proxy

```python
class OptionsProxy:
    def __init__(self, ctx):
        self.ctx = ctx

    def __getattr__(self, name):
        # map known A1111 option names to AIWF settings
        # unknown returns None or raises CompatUnsupportedOption
        ...

    def __setattr__(self, name, value):
        # allow safe mapped settings only
        ...
```

Suggested mappings:

```text
samples_save              -> ctx.settings.save_images
grid_save                 -> ctx.settings.save_grid
CLIP_stop_at_last_layers  -> ctx.settings.default_clip_skip or request.clip_skip
sd_model_checkpoint       -> ctx.settings.last_checkpoint_id
sd_vae                    -> selected VAE / request VAE
```

## `cmd_opts` Proxy

Read-only proxy to `ctx.flags`.

Mappings:

```text
listen
share
port
theme
xformers
medvram
lowvram
api
nowebui
```

## `state` Proxy

Map to active AIWF job:

```text
state.job
state.job_count
state.sampling_step
state.sampling_steps
state.interrupted
state.skipped
```

## `sd_model`

Start as read-only or `None`.

Do not expose raw Diffusers pipeline as `sd_model` yet.

Reason:

```text
A1111 extensions may mutate sd_model expecting CompVis/K-diffusion internals.
AIWF backend is Diffusers-native.
```

Classification:

```text
read shared.sd_model  -> Class D or C
write shared.sd_model -> Class X or D
```

---

# 14. Phase 9 — `modules.scripts` Shim

## File

```text
aiwf/compat/a1111/shim_modules/modules/scripts.py
```

## Required Constants / Classes

```python
AlwaysVisible = object()

class Script:
    ...
```

## Base Methods

```python
class Script:
    name = None
    section = None
    filename = None
    args_from = None
    args_to = None
    alwayson = False
    is_txt2img = False
    is_img2img = False
    tabname = None
    controls = None
    infotext_fields = None
    paste_field_names = None

    def title(self):
        raise NotImplementedError()

    def ui(self, is_img2img):
        return []

    def show(self, is_img2img):
        return True

    def run(self, p, *args):
        return None

    def setup(self, p, *args):
        pass

    def before_process(self, p, *args):
        pass

    def process(self, p, *args):
        pass

    def before_process_batch(self, p, *args, **kwargs):
        pass

    def process_batch(self, p, *args, **kwargs):
        pass

    def postprocess_batch(self, p, *args, **kwargs):
        pass

    def postprocess_image(self, p, pp, *args):
        pass

    def postprocess(self, p, processed, *args):
        pass
```

## `basedir()`

```python
_current_basedir: Path | None = None

def basedir():
    return str(_current_basedir or project_root)
```

## UI Mounting

For Class A/B UI scripts:

```text
Call script.title()
Call script.show(is_img2img=False/True)
Call script.ui(is_img2img=False/True)
Mount returned Gradio components in AIWF Extensions tab
```

Initial compromise:

```text
Do not inject into main txt2img/img2img UI yet.
Mount in a contained “A1111 Extension Controls” accordion.
```

---

# 15. Phase 10 — `script_callbacks` Shim

## File

```text
aiwf/compat/a1111/shim_modules/modules/script_callbacks.py
```

## Callback Map

```python
callback_map = {
    "app_started": [],
    "ui_tabs": [],
    "ui_settings": [],
    "before_image_saved": [],
    "image_saved": [],
    "infotext_pasted": [],
    "before_component": [],
    "after_component": [],

    # advanced / initially blocked
    "model_loaded": [],
    "cfg_denoiser": [],
    "cfg_denoised": [],
    "cfg_after_cfg": [],
    "list_optimizers": [],
    "list_unets": [],
}
```

## Safe Registration Functions

```python
def on_app_started(callback, *, name=None): ...
def on_ui_tabs(callback, *, name=None): ...
def on_ui_settings(callback, *, name=None): ...
def on_before_image_saved(callback, *, name=None): ...
def on_image_saved(callback, *, name=None): ...
def on_infotext_pasted(callback, *, name=None): ...
def on_before_component(callback, *, name=None): ...
def on_after_component(callback, *, name=None): ...
```

## Blocked / Delayed Functions

```python
def on_cfg_denoiser(callback, *, name=None):
    register_blocked("cfg_denoiser", callback)

def on_cfg_denoised(callback, *, name=None):
    register_blocked("cfg_denoised", callback)

def on_cfg_after_cfg(callback, *, name=None):
    register_blocked("cfg_after_cfg", callback)

def on_list_optimizers(callback, *, name=None):
    register_blocked("list_optimizers", callback)

def on_list_unets(callback, *, name=None):
    register_blocked("list_unets", callback)
```

## Bridge Callbacks To AIWF Events

File:

```text
aiwf/compat/a1111/adapters/callbacks_to_events.py
```

Map:

```text
on_app_started        -> after Gradio app launch
on_ui_tabs            -> AIWF WebRegistry / Extension tab
on_ui_settings        -> Settings tab compatibility section
on_before_image_saved -> before ImageStore save
on_image_saved        -> after ImageStore save / AfterGenerate event
on_infotext_pasted    -> InfotextBridge
```

## Advanced Callback Policy

Do not implement these first:

```text
cfg_denoiser
cfg_denoised
cfg_after_cfg
list_optimizers
list_unets
model_loaded with mutable model
```

Reason:

```text
They depend on A1111 sampler/model internals. AIWF uses Diffusers pipelines and service boundaries.
```

---

# 16. Phase 11 — Processing Adapter

This is the hardest and most valuable part.

## Files

```text
aiwf/compat/a1111/shim_modules/modules/processing.py
aiwf/compat/a1111/adapters/processing_to_generation.py
```

## Classes To Implement

```python
class StableDiffusionProcessing:
    ...

class StableDiffusionProcessingTxt2Img(StableDiffusionProcessing):
    ...

class StableDiffusionProcessingImg2Img(StableDiffusionProcessing):
    ...

class Processed:
    ...
```

## Common Processing Fields

```python
prompt: str
negative_prompt: str
styles: list[str]
seed: int
subseed: int
subseed_strength: float
seed_resize_from_h: int
seed_resize_from_w: int

sampler_name: str
steps: int
cfg_scale: float
width: int
height: int

batch_size: int
n_iter: int

restore_faces: bool
tiling: bool
do_not_save_samples: bool
do_not_save_grid: bool

init_images: list
mask
denoising_strength: float
inpaint_full_res: bool
inpaint_full_res_padding: int
inpainting_mask_invert: int

extra_generation_params: dict
scripts
script_args
```

Unknown fields should exist but may be no-op.

## Mapping To AIWF

```text
p.prompt              -> GenerationRequest.prompt
p.negative_prompt     -> GenerationRequest.negative_prompt
p.steps               -> GenerationRequest.steps
p.cfg_scale           -> GenerationRequest.cfg_scale
p.width               -> GenerationRequest.width
p.height              -> GenerationRequest.height
p.seed                -> GenerationRequest.seed
p.sampler_name        -> GenerationRequest.sampler
p.batch_size          -> GenerationRequest.batch_size
p.n_iter              -> GenerationRequest.batch_count
p.denoising_strength  -> GenerationRequest.denoising_strength
p.init_images         -> init_images
p.mask                -> mask_images
p.do_not_save_samples -> save_images=False
```

## `process_images(p)`

```python
def process_images(p):
    request = generation_request_from_processing(p)
    job = ctx.generation.submit(
        request,
        init_images=p.init_images,
        mask_images=[p.mask] if p.mask else None,
    )
    return processed_from_job(job, p)
```

## `Processed`

```python
class Processed:
    def __init__(
        self,
        p,
        images_list,
        seed=-1,
        info="",
        comments="",
        all_seeds=None,
        all_prompts=None,
        infotexts=None,
    ):
        self.p = p
        self.images = images_list
        self.seed = seed
        self.info = info
        self.comments = comments
        self.all_seeds = all_seeds or []
        self.all_prompts = all_prompts or []
        self.infotexts = infotexts or []
```

## Script Lifecycle With Processing

Initial support:

```text
setup              -> supported
before_process     -> supported
process            -> supported
postprocess_image  -> supported
postprocess        -> supported
batch hooks        -> partial
latent hooks       -> unsupported
```

## Acceptance Criteria

A fake extension like this should work:

```python
from modules import scripts, processing

class SimpleRun(scripts.Script):
    def title(self):
        return "Simple Run"

    def show(self, is_img2img):
        return True

    def run(self, p, *args):
        p.prompt = p.prompt + ", cinematic lighting"
        return processing.process_images(p)
```

Expected:

```text
AIWF generates image using modified prompt.
Report says Class B or A depending enabled shims.
No crash.
```

---

# 17. Phase 12 — Image Helpers

## File

```text
aiwf/compat/a1111/shim_modules/modules/images.py
```

## Common Helpers

```python
def save_image(...):
    ...

def image_grid(imgs, batch_size=1, rows=None):
    ...

def resize_image(resize_mode, im, width, height, upscaler_name=None):
    ...
```

## Map To AIWF

```text
save_image   -> FilesystemImageStore
image_grid   -> AIWF grid helper or new utility
resize_image -> PIL resize helper
```

---

# 18. Phase 13 — Infotext / PNG Info Bridge

## Files

```text
aiwf/compat/a1111/shim_modules/modules/generation_parameters_copypaste.py
aiwf/compat/a1111/adapters/infotext_bridge.py
```

## Support

```text
parse_generation_parameters
create_override_settings_dict
infotext paste callback
send-to txt2img/img2img style fields
```

## AIWF Bridge

Map to existing AIWF concepts:

```text
parse_infotext
infotext_to_request_updates
InfotextBridge
PNG Info tab
```

---

# 19. Phase 14 — ControlNet Bridge

## Files

```text
aiwf/compat/a1111/adapters/controlnet_bridge.py
```

## Support

```text
/sdapi/v1/controlnet/model_list
/sdapi/v1/controlnet/module_list
/sdapi/v1/controlnet/detect
alwayson_scripts.controlnet.args
```

## Field Mapping

```text
input_image     -> ControlNetUnit.image
module          -> preprocessor
model           -> model_id
weight          -> weight
resize_mode     -> resize_mode
processor_res   -> processor_res
threshold_a     -> threshold_a
threshold_b     -> threshold_b
guidance_start  -> start_percent
guidance_end    -> end_percent
control_mode    -> control_mode
pixel_perfect   -> pixel_perfect
```

## Compatibility Classification

```text
Single ControlNet unit -> likely supported
Multiple units -> partial until AIWF supports multi-unit fully
Advanced reference/IPAdapter/t2ia variants -> adapter required or unsupported
Forge-specific ControlNet APIs -> unsupported or adapter required
```

---

# 20. Phase 15 — Extension Load Order

## File

```text
aiwf/compat/a1111/metadata.py
```

## Parser

Parse:

```text
Requires
Before
After
```

From:

```text
[Extension]
[scripts]
[scripts/file.py]
[callbacks/...]
[javascript]
[localizations]
```

## Initial Behavior

Phase 1:

```text
Read and report only.
```

Phase 2:

```text
Sort load order with topological sort.
```

Phase 3:

```text
Honor callback ordering.
```

## Acceptance Criteria

If extension B requires extension A and A is missing:

```text
Report: Class D
Finding: missing dependency
can_attempt_load: false unless user override
```

---

# 21. Phase 16 — Install Script Policy

A1111 may execute `install.py` before the web UI starts. AIWF should not copy that behavior by default.

## Rule

```text
install.py is never executed automatically.
```

## Scanner Report

If `install.py` exists:

```text
Risk: medium/high
Finding: install script present
Action: user must review
```

If it imports `launch` and calls `launch.run_pip`:

```text
Risk: high
Class: B/C/X depending package behavior
Action: generate dependency recommendation instead of running
```

## Safe Dependency Output

Instead of running install:

```text
This extension requests:
pip install aitextgen==0.6.0

AIWF recommendation:
Create an isolated extension venv or adapter-specific dependency group.
```

## Future Option

Add:

```text
Run install.py in isolated compatibility venv
```

But not in v1.

---

# 22. Phase 17 — Adapter Stub Generator

## File

```text
aiwf/compat/a1111/adapter_stub.py
```

## User Action

In Extension Manager:

```text
Create adapter stub
```

## Output Path

Prefer:

```text
aiwf_adapters/<extension-name>.py
```

Do not modify third-party extension folders by default.

## Stub Template

```python
from __future__ import annotations

from aiwf.bootstrap import AppContext


class AIWFExtensionAdapter:
    name = "example-extension"
    source_extension = "example-extension"

    def register(self, ctx: AppContext) -> None:
        """Register AIWF-native behavior for this A1111 extension.

        TODO:
        - Map unsupported A1111 APIs listed in the compatibility report.
        - Use ctx.generation, ctx.models, ctx.workflows, ctx.events, or ctx.settings.
        """
        pass
```

## Include Report Hints

```python
# Unsupported feature: modules.sd_hijack
# Reason: AIWF does not expose A1111 model hijack internals.
# Suggested AIWF target: aiwf.infrastructure.diffusers.backend extension ABI.
```

---

# 23. Phase 18 — Extension Manager Storage

## File

```text
aiwf/compat/a1111/registry.py
```

## Config Path

Store inside the AIWF data dir:

```text
<data-dir>/compat/a1111/extensions.json
```

## Schema

```json
{
  "extension_roots": [
    "extensions"
  ],
  "extensions": {
    "example-extension": {
      "path": "extensions/example-extension",
      "enabled": false,
      "trusted": false,
      "last_scan_report": "compat/reports/example-extension.json",
      "compat_class": "B",
      "risk_level": "medium",
      "notes": ""
    }
  }
}
```

## Acceptance Criteria

User can:

```text
scan
trust
enable
disable
rescan
view report
```

without editing JSON manually.

---

# 24. Phase 19 — Real Extension Test Matrix

## File

```text
docs/compatibility/a1111-extension-test-matrix.md
```

## Columns

```text
Extension
Repo URL
Category
Scan Class
Risk
Load Status
Supported Features
Unsupported Features
Adapter Needed
Notes
```

## Target Categories

Test at least:

```text
UI-only helper
prompt helper
metadata/PNG helper
CSS/JS-only extension
ControlNet-adjacent extension
API client tool
postprocess image extension
wildcard/prompt extension
sampler/model internal extension
Forge-specific extension
```

## Public Compatibility Threshold

Before claiming public compatibility:

```text
10 extensions scanned
5 extensions loaded partially or fully
3 extension reports published
0 unsafe scripts auto-executed
```

---

# 25. Phase 20 — Backend Extension ABI

Some extensions cannot be A1111-compatible because they patch internals. AIWF needs its own safe backend extension ABI.

## Future File

```text
aiwf/core/interfaces/backend_extension.py
```

## Interfaces

```python
class BackendExtension:
    name: str

    def before_load_model(self, request):
        ...

    def after_load_model(self, pipe):
        ...

    def before_generate(self, request, pipe):
        ...

    def after_generate(self, request, result):
        ...

    def modify_scheduler(self, scheduler, request):
        ...

    def modify_prompt_kwargs(self, prompt_kwargs, request):
        ...
```

## Why

This gives adapter authors a safe target instead of monkey-patching:

```text
shared.sd_model
sd_hijack
k_diffusion
unet replacement
sampler replacement
```

## Classification

Extensions using A1111 deep internals should get:

```text
Class C if an AIWF backend adapter could solve it.
Class D if no backend extension ABI exists yet.
Class X if it monkey-patches unsafely.
```

---

# 26. Build Order For A 1-Year Python Developer

## Step 1 — Create Folders

```powershell
mkdir aiwf\compat
mkdir aiwf\compat\a1111
mkdir aiwf\compat\a1111\adapters
mkdir aiwf\compat\a1111\shim_modules
mkdir aiwf\compat\a1111\shim_modules\modules
mkdir tests\compat
mkdir tests\compat\a1111
mkdir tests\compat\a1111\fixtures
```

Add empty init files:

```powershell
ni aiwf\compat\__init__.py
ni aiwf\compat\a1111\__init__.py
ni aiwf\compat\a1111\adapters\__init__.py
ni aiwf\compat\a1111\shim_modules\modules\__init__.py
```

## Step 2 — Implement Data Models

Implement:

```text
aiwf/compat/a1111/classes.py
```

Run tests.

## Step 3 — Implement Scanner Filesystem Walk

No AST yet.

Detect files only.

Run tests.

## Step 4 — Add AST Import Detection

Detect `modules.*` imports.

Run tests.

## Step 5 — Add Risky Call Detection

Detect subprocess, shell, install, destructive file, and network patterns.

Run tests.

## Step 6 — Add Metadata Parser

Parse `metadata.ini`.

Run tests.

## Step 7 — Add Classification Logic

Assign A/B/C/D/X and risk level.

Run tests.

## Step 8 — Add Report Renderer

JSON, text, and Markdown output.

Run tests.

## Step 9 — Add CLI

Run:

```powershell
python -m aiwf.compat.a1111 scan tests\compat\a1111\fixtures\ui_only_extension
```

## Step 10 — Add Extensions UI Tab

Only scan. Do not load extension Python yet.

## Step 11 — Add `modules.scripts` Shim

Test fake UI extension.

## Step 12 — Add `modules.shared` Shim

Test fake settings extension.

## Step 13 — Add `script_callbacks` Shim

Test fake callback extension.

## Step 14 — Add Asset Loading

Test CSS-only extension.

## Step 15 — Add Processing Adapter

Test fake `process_images` extension.

## Step 16 — Add Classic A1111-Inspired Layout

Use the downloaded A1111 repo as behavior/layout reference only.

Do not copy source.

Run:

```powershell
python -m pytest tests/ -q
python -m aiwf.app
```

Manually verify:

```text
Settings -> UI Layout -> Studio
Settings -> UI Layout -> Classic A1111-inspired
Generate works in both layouts.
```

## Step 17 — Test Real Extensions

Only after scanner and reports are stable.

---

# 27. Definition Of Done

## Scanner v1 Done When

```text
- Scans extension folders without executing Python.
- Detects files, imports, callbacks, script classes, metadata, and risky calls.
- Produces JSON and Markdown reports.
- Assigns A/B/C/D/X class.
- Assigns risk level.
- Has tests for at least 6 fake extension types.
```

## UI v1 Done When

```text
- Extensions tab exists.
- User can scan one extension folder.
- User can scan all extension folders.
- User can read report.
- User can see blocked reason.
- No extension Python executes.
```

## Loader v1 Done When

```text
- Can load Class A UI-only extension.
- Can mount simple Gradio controls.
- Can load CSS after user trust.
- Can reject unsafe extension.
- Can report unsupported methods instead of crashing.
```

## Processing v1 Done When

```text
- Simple Script.run extension can call process_images.
- Txt2img maps correctly.
- Img2img maps correctly.
- Inpaint maps basic image + mask.
- Processed object returns images, seeds, infotexts.
```

## Classic Layout v1 Done When

```text
- User can switch between Studio and Classic A1111-inspired layout.
- Both layouts use the same AIWF services.
- Classic layout is clearly described as inspired by, not copied from, A1111.
- Legacy launch and model-folder familiarity remain intact.
```

## Public Claim Allowed

```text
AIWF Studio includes experimental A1111 extension compatibility with scan-first diagnostics.
```

## Public Claim Not Allowed Yet

```text
AIWF supports all A1111 extensions.
```

---

# 28. First Commit Checklist For Implementation

- [ ] Add `aiwf/compat/a1111/classes.py`.
- [ ] Add scanner skeleton.
- [ ] Add report renderer skeleton.
- [ ] Add CLI skeleton.
- [ ] Add tests for report models.
- [ ] Add fake fixture: UI-only extension.
- [ ] Add fake fixture: unsafe install extension.
- [ ] Add `docs/compatibility/a1111-extension-test-matrix.md`.
- [ ] Add README note linking to `compatability.md` after scanner exists.

---

# 29. Open Questions

These should be answered during implementation, not guessed up front.

1. Should AIWF create a top-level `extensions/` folder, or should it scan user-selected folders only?
2. Should trusted extension state live in `config.json` or separate `<data-dir>/compat/a1111/extensions.json`?
3. Should install scripts ever run automatically in a separate venv, or always remain manual?
4. Should Classic A1111-inspired layout ship before extension loading, or after scanner v1?
5. Should A1111 JavaScript be allowed at all, or only CSS/theme compatibility at first?
6. Should deep backend hooks target a new AIWF backend extension ABI instead of A1111 shims?

---

# 30. North Star

No silent failure. No mystery crashes. No unsafe extension execution by default.

The user should always know:

```text
what worked
what did not work
why it did not work
what shim or adapter would make it work
whether loading it is safe
```

That is the AIWF version of drop-in support: **measured compatibility with honest diagnostics.**
