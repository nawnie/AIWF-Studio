# AIWF Studio Engine Isolation Architecture Handoff

**Document purpose:** Project-manager handoff for the AIWF Studio runtime architecture decision.  
**Project:** AIWF Studio  
**Audience:** Project manager, implementation agents, contributors, and future maintainers.  
**Status:** Architecture decision draft for implementation planning.  
**Core decision:** AIWF Studio should remain one user-facing Gradio application while running heavy AI backends as isolated engine workers in separate Python environments.

---

## 1. Executive Summary

AIWF Studio should be built as a **single stable UI shell** with **isolated backend engines**. The user launches one application and interacts with one Gradio interface. Under the hood, AIWF Studio supervises separate runtime workers for generation, training, and chat instead of importing every backend into the same Python process.

This is the key architectural difference from extension-heavy systems such as Auto1111 WebUI and ComfyUI, where many extensions or custom nodes share the same Python environment, dependency stack, import system, and CUDA runtime. That approach is powerful, but it pushes dependency conflicts and complex tracebacks onto users.

AIWF Studio should move the boundary outward:

```text
AIWF Studio UI = stable supervisor
Heavy backends = isolated engines
User experience = one program
Dependency risk = contained behind process boundaries
```

The goal is to make the hard part live in the architecture, not in the user's lap.

---

## 2. Final Recommended Architecture

Use one Gradio UI process and multiple isolated backend engines.

```text
AIWF Studio UI
Python 3.10 main environment
Gradio / routing / config / logs / model browser / phone companion
        │
        ├── Generation Engine
        │       image generation / Wan / LTX / VAE decode / RIFE / NVENC
        │       separate generation venv
        │
        ├── Ollama Chat Tenant
        │       external local service via HTTP API
        │
        ├── Kohya LoRA Engine
        │       separate training venv
        │
        └── ED2 Full-Training Engine
                separate Python 3.10 training venv
```

### Initial grouping recommendation

For the first maintainable version, do **not** split image generation, Wan, and LTX into separate environments immediately. Start with one generation engine venv.

```text
Main AIWF UI venv
    └── Gradio, routing, config models, logs, API routes, process supervision

Generation engine venv
    └── image generation, Wan, LTX, VAE decode, RIFE, FFmpeg/NVENC calls

Kohya engine venv
    └── LoRA training

ED2 engine venv
    └── full model training

Ollama
    └── independent local service
```

This gives the best balance between maintainability and isolation. If Wan later proves to require an incompatible dependency stack, the generation engine can be split into dedicated image, Wan, and LTX workers.

---

## 3. Architecture Decision: Do Not Reload Gradio Between Venvs

AIWF Studio should **not** switch or reload Gradio apps when the user changes tabs.

Avoid this design:

```text
Main UI venv
    ↓ user clicks Video tab
Reload into video Gradio app from video venv
    ↓ user clicks Training tab
Reload into training Gradio app from training venv
```

That design appears simple but creates long-term problems:

- Tab changes become process changes.
- UI state can disappear.
- Browser refresh behavior becomes confusing.
- Logs and progress streams are harder to keep consistent.
- API routes become unstable for automation tools.
- Phone companion support becomes harder.
- Multiple Gradio apps duplicate layout code.
- GPU ownership becomes ambiguous.
- Error handling becomes harder to centralize.

The correct design is:

```text
Gradio never moves.
Engines move underneath it.
```

The UI should stay boring, stable, and predictable. The heavy engines can be isolated, restarted, killed, upgraded, or replaced underneath the UI.

---

## 4. Why This Works Compared to Auto1111 WebUI and ComfyUI

Auto1111 WebUI and ComfyUI are powerful because extensions or custom nodes can directly plug into the application. The tradeoff is that those extensions usually share the same runtime surface:

```text
same Python environment
same torch install
same numpy install
same xformers install
same CUDA wheel stack
same import system
same process memory
same GPU runtime
```

A typical failure mode looks like this:

```text
Extension A requires:
    numpy < 2
    torch 2.1
    xformers version A

Extension B requires:
    numpy >= 2
    torch 2.6
    xformers version B

User result:
    broken imports
    dependency downgrade loops
    CUDA mismatch errors
    tracebacks normal users cannot resolve
```

AIWF Studio avoids this by making the UI an orchestrator instead of a plugin bucket.

```text
AIWF does not import ED2.
AIWF launches ED2.

AIWF does not import Kohya internals.
AIWF launches Kohya.

AIWF does not need every generation backend dependency in the main UI.
AIWF talks to a generation worker.
```

This moves the dependency boundary from **inside the same Python app** to **outside the app at the process level**.

---

## 5. Core Design Principles

These rules should be treated as architectural guardrails.

### 5.1 One stable UI shell

The AIWF Studio UI runs in one main Python 3.10 environment.

The main UI owns:

- Gradio layout
- tab routing
- typed request construction
- form validation
- model/config browsing
- logs
- progress displays
- output file references
- process supervision
- phone companion API surface
- global GPU tenant coordination

The main UI should avoid importing heavy backend code directly.

### 5.2 Heavy engines run out-of-process

Heavy engines run as subprocess workers or local services.

Examples:

- generation engine worker
- Kohya LoRA worker
- ED2 full-training worker
- Ollama chat service

Each engine can own its own Python environment, dependency stack, and runtime assumptions.

### 5.3 UI actions call services, not backend internals

A Gradio button should not directly call Torch, ED2, Kohya, Wan internals, or custom backend loaders.

Correct flow:

```text
Gradio event
    ↓
Typed request model
    ↓
Service client
    ↓
Engine supervisor
    ↓
Subprocess worker
    ↓
Progress/log/result events
    ↓
UI update
```

Incorrect flow:

```text
Gradio event
    ↓
import torch
    ↓
load model directly inside UI process
    ↓
crash takes UI down
```

### 5.4 No global shared backend state

The main UI should not rely on hidden global backend objects. Engine state must be explicit and observable through a supervisor.

Examples of explicit state:

- active engine name
- process ID
- job ID
- GPU tenant lock owner
- current log file path
- output directory
- current status
- last heartbeat

### 5.5 Single-tenant GPU policy

The 16 GB GPU must be treated as a single-tenant workspace. Only one heavy GPU tenant should run at a time.

Potential tenants:

- image/video generation
- Wan
- LTX
- Kohya training
- ED2 full training
- Ollama model loaded into GPU

Policy:

```text
If ED2 training is running:
    block Wan generation
    block LTX generation
    block Kohya training
    optionally unload Ollama

If Wan generation is running:
    block ED2 training
    block Kohya training
    block LTX generation
    unload Ollama if necessary

If LTX generation is running:
    block ED2 training
    block Kohya training
    block Wan generation
    unload Ollama if necessary
```

---

## 6. Proposed Repository Layout

```text
AIWF_Studio/
├── launch_aiwf.ps1
├── README.md
├── docs/
│   ├── architecture/
│   │   └── engine_isolation_architecture.md
│   └── permissions/
│       ├── everydream2_permission_note.md
│       └── screenshots/
│           └── ed2_fork_permission_discord.png
│
├── app/
│   ├── webui.py
│   ├── ui/
│   │   ├── image_tab.py
│   │   ├── video_tab.py
│   │   ├── chat_tab.py
│   │   ├── training_tab.py
│   │   └── settings_tab.py
│   │
│   ├── services/
│   │   ├── gpu_tenant_lock.py
│   │   ├── engine_supervisor.py
│   │   ├── generation_client.py
│   │   ├── kohya_client.py
│   │   ├── ed2_client.py
│   │   └── ollama_client.py
│   │
│   ├── domain/
│   │   ├── generation_requests.py
│   │   ├── training_requests.py
│   │   ├── engine_events.py
│   │   ├── job_status.py
│   │   └── paths.py
│   │
│   └── utils/
│       ├── log_tail.py
│       ├── process_tree.py
│       └── path_safety.py
│
├── engines/
│   ├── generation/
│   │   ├── .venv/
│   │   ├── worker.py
│   │   ├── requirements.txt
│   │   └── engine_config.example.toml
│   │
│   ├── kohya/
│   │   ├── .venv/
│   │   ├── kohya_ss/
│   │   ├── worker.py
│   │   ├── requirements.txt
│   │   └── engine_config.example.toml
│   │
│   └── ed2/
│       ├── .venv/
│       ├── EveryDream2trainer/
│       ├── worker.py
│       ├── requirements.txt
│       └── engine_config.example.toml
│
├── models/
│   ├── Stable-diffusion/
│   ├── Lora/
│   ├── VAE/
│   ├── text_encoders/
│   ├── wan/
│   └── ltx/
│
├── datasets/
│   ├── raw/
│   ├── prepared/
│   └── validation_reports/
│
├── outputs/
│   ├── images/
│   ├── videos/
│   ├── training/
│   ├── logs/
│   └── jobs/
│
└── configs/
    ├── aiwf_studio.toml
    ├── engines.toml
    └── gpu_policy.toml
```

---

## 7. Engine Responsibilities

### 7.1 Main AIWF UI Environment

The main UI environment should remain lightweight and stable.

Owns:

- Gradio UI
- page/tab layout
- request validation
- typed request models
- engine startup/shutdown commands
- log streaming
- progress display
- status polling
- output file registration
- global settings
- phone companion routing
- GPU tenant lock

Should avoid owning:

- direct model loading
- direct Torch calls for heavy inference
- ED2 imports
- Kohya imports
- backend-specific CUDA extensions
- backend-specific training dependencies

### 7.2 Generation Engine

Initial scope:

- image generation
- Wan video generation
- LTX video generation
- VAE decode
- RIFE/RIF interpolation
- FFmpeg/NVENC compilation
- generation-specific model lifecycle management

Why group these first:

- They are all generation-side workloads.
- They likely share Torch/CUDA/diffusion dependencies.
- It is easier to maintain one generation engine early.
- It avoids premature over-splitting.

Split later only if needed:

```text
engines/generation_image/
engines/generation_wan/
engines/generation_ltx/
```

Trigger for splitting:

- Wan requires incompatible Torch/CUDA/SageAttention stack.
- LTX requires incompatible dependency stack.
- Image generation stability is harmed by video dependencies.
- generation worker becomes too large to maintain.

### 7.3 Kohya LoRA Engine

Scope:

- simple LoRA training
- advanced LoRA training
- Kohya config generation
- dataset validation result consumption
- subprocess launch
- log streaming
- abort/cleanup
- output LoRA registration

Should run in its own venv because training dependencies are high-risk.

### 7.4 ED2 Full-Training Engine

Scope:

- full model training
- ED2-compatible config generation
- dataset validation result consumption
- subprocess launch
- log streaming
- abort/cleanup
- checkpoint output registration

Important permission note:

- The ED2 creator has explicitly indicated the project is archived and may be forked.
- AIWF should still respect ED2 license terms, third-party dependency licenses, model licenses, and dataset licenses.
- ED2 should be treated as an optional isolated backend, not merged into the main app process.

### 7.5 Ollama Chat Tenant

Scope:

- local AI chat
- prompt help
- workflow assistant
- model explanation
- local API calls

Ollama is already its own service. AIWF should communicate with it through local HTTP calls.

AIWF should unload Ollama when a heavy GPU tenant needs the card:

```text
/api/generate or /api/chat with keep_alive: 0
```

---

## 8. Process Communication Model

The first implementation can use subprocesses with stdout/stderr streaming. A future version can move to local HTTP or JSON-RPC if needed.

### 8.1 Minimum viable protocol

The UI launches an engine worker with:

```text
engine_python.exe worker.py --job-file path/to/job.json
```

The worker writes structured lines to stdout:

```json
{"event":"status","job_id":"job_001","message":"loading model"}
{"event":"progress","job_id":"job_001","step":4,"total":30}
{"event":"artifact","job_id":"job_001","path":"outputs/videos/job_001.mp4"}
{"event":"complete","job_id":"job_001"}
```

The UI parses these events and updates Gradio.

### 8.2 Why JSONL stdout is a good first choice

- Easy to implement.
- Works across venv boundaries.
- Does not require a server per engine.
- Easy to log.
- Easy to debug.
- Easy to replace later.

### 8.3 Future communication upgrade

If needed, each engine can become a local service:

```text
127.0.0.1:78xx/generate
127.0.0.1:78xx/train
127.0.0.1:78xx/status/{job_id}
```

Do not start with this unless subprocess communication becomes limiting.

---

## 9. GPU Tenant Lock

The GPU tenant lock is a central service in the main UI process.

It should answer:

- Is the GPU free?
- Which engine owns the GPU?
- Which job owns the GPU?
- Can this new task start?
- Should Ollama be unloaded first?
- What should the UI show if blocked?

Example policy:

```text
request: start ED2 training
if active_tenant is None:
    unload Ollama if needed
    grant tenant = ed2
elif active_tenant == ed2:
    reject: ED2 already running
else:
    reject: GPU is busy with active_tenant
```

### Required UI behavior

If the GPU is busy, do not crash or silently queue dangerous work. Show a normal message:

```text
GPU is currently owned by: Wan Video Generation
Current job: wan_2026_06_13_001
Action blocked: ED2 Full Training
Options: wait, stop current job, or cancel.
```

---

## 10. Job Lifecycle

Every heavy operation should become a job.

```text
created
    ↓
validated
    ↓
queued or blocked
    ↓
running
    ↓
completed / failed / cancelled
    ↓
artifacts registered
    ↓
logs retained
```

### Job directory structure

```text
outputs/jobs/job_001/
├── request.json
├── resolved_config.toml
├── stdout.jsonl
├── stderr.log
├── status.json
└── artifacts/
    ├── output.mp4
    ├── output.safetensors
    └── validation_report.md
```

This makes debugging, project handoff, phone companion status, and future reproducibility much easier.

---

## 11. Training Tab Architecture

The training tab should provide one user-facing training workspace with multiple isolated engines behind it.

```text
Training Tab
├── Dataset Preflight
│   ├── image count
│   ├── caption pairing
│   ├── repeats_concept folder validation
│   ├── resolution/bucket sanity checks
│   └── optional VLM auto-captioning
│
├── Simple LoRA Training
│   └── Kohya backend
│
├── Advanced LoRA Training
│   └── Kohya backend with expanded parameters
│
└── Full Model Training
    └── ED2-compatible backend
```

### Training tab responsibilities

AIWF owns:

- dataset selection
- preflight validation
- caption checks
- config generation
- launch button
- stop button
- log streaming
- output registration
- cleanup

Backends own:

- actual training math
- backend-specific dependency stack
- backend-specific execution details

---

## 12. Wan/LTX/Image Generation Architecture

The generation tab should call the generation service, not import backend code directly.

```text
Video/Image Tab
    ↓
GenerationRequest
    ↓
GenerationClient
    ↓
EngineSupervisor
    ↓
Generation Worker
    ↓
JSONL status/progress/artifact events
    ↓
Gradio UI updates
```

### Important note for Wan

Isolating Wan as a worker will solve some problems but not all of them.

It helps with:

- dependency conflicts
- crash containment
- cleaner CUDA context release when the worker exits
- keeping the UI alive after backend failure
- enforcing GPU tenant ownership

It does not automatically solve:

- Wan model size
- attention scaling
- bad quant choices
- Windows shared-memory fallback
- improper FP8/GGUF loading
- VAE memory spikes
- incorrect frame settings

Wan still needs its own memory-management strategy inside the generation engine.

---

## 13. External API Route Stability

For automation tools and future phone companion support, the UI should expose stable routes.

Rules:

- API route names should be hardcoded and versioned.
- Inputs should be stateless typed payloads.
- Do not depend on browser-only state for engine execution.
- Every route should create or operate on a job.

Example routes:

```text
/api/v1/generate/image
/api/v1/generate/video
/api/v1/train/lora
/api/v1/train/full
/api/v1/jobs/status
/api/v1/jobs/cancel
/api/v1/models/list
```

In Gradio, this means button/event functions should use stable `api_name` values where applicable.

---

## 14. Maintainability Benefits

This architecture is easier to maintain because new engines become isolated additions instead of invasive app rewrites.

To add a backend:

```text
1. Add engine folder.
2. Add venv setup script.
3. Add request model.
4. Add config generator.
5. Add service client.
6. Add engine supervisor registration.
7. Add UI controls.
8. Add GPU tenant policy.
9. Add logging/artifact registration.
```

The main UI stays stable.

---

## 15. Implementation Phases

### Phase 1 — Lock the architecture boundary

Deliverables:

- main Gradio shell remains single process
- no backend imports in UI tabs
- service-client pattern established
- engine supervisor skeleton created
- job directory structure created
- GPU tenant lock skeleton created

Acceptance criteria:

- UI launches without importing ED2, Kohya, or heavy generation internals.
- UI can create a fake job and stream fake JSONL events.
- job status appears in the UI.

### Phase 2 — Generation engine worker

Deliverables:

- generation worker venv
- generation worker launch script
- image generation route integrated
- video generation route integrated later
- JSONL event streaming
- output artifact registration

Acceptance criteria:

- UI can launch generation worker.
- UI stays alive if worker crashes.
- UI can display worker logs.
- UI can register output paths.

### Phase 3 — Training preflight

Deliverables:

- dataset validator
- caption-pair checker
- repeats_concept checker
- validation report writer
- Gradio validation panel

Acceptance criteria:

- user can select dataset
- AIWF reports missing captions, folder issues, and image counts
- validation report is saved per job

### Phase 4 — Kohya LoRA engine

Deliverables:

- Kohya venv integration
- config generation
- subprocess launch
- log streaming
- stop/cleanup
- output LoRA registration

Acceptance criteria:

- UI starts Kohya through its own Python executable.
- UI does not import Kohya directly.
- Stop button terminates process tree.

### Phase 5 — ED2 full-training engine

Deliverables:

- ED2 venv integration
- ED2-compatible config generation
- subprocess launch
- log streaming
- stop/cleanup
- output checkpoint registration
- permission note added to docs

Acceptance criteria:

- UI starts ED2 through its own Python 3.10 environment.
- UI does not import ED2 directly.
- ED2 logs stream into AIWF Studio.
- Stop button terminates ED2 process tree.

### Phase 6 — Phone companion support

Deliverables:

- job status endpoint
- running job logs endpoint
- artifact listing endpoint
- safe stop endpoint

Acceptance criteria:

- phone companion can view active jobs
- phone companion can view status/logs
- phone companion can request cancellation

---

## 16. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Too many venvs too early | High maintenance burden | Start with one generation venv, one Kohya venv, one ED2 venv |
| Gradio duplicated across engines | State loss, brittle UX | Keep one Gradio shell only |
| Worker hangs | GPU locked forever | Heartbeats, timeout policy, process-tree kill |
| CUDA memory not released | Future jobs fail | Kill worker process to release CUDA context |
| Ollama keeps VRAM | Generation/training OOM | Explicit unload call before heavy GPU jobs |
| ED2/Kohya dependency conflict | Broken training | Separate venvs |
| Wan dependency conflict with image generation | Broken generation engine | Split Wan into dedicated worker only if proven necessary |
| Users see raw tracebacks | Bad UX | Convert worker errors into clean UI messages while preserving full logs |
| Permission/licensing confusion | Project risk | Document ED2 permission note and respect all third-party licenses |

---

## 17. Project Manager Task List

### Architecture tasks

- [ ] Approve one-Gradio-shell architecture.
- [ ] Reject tab-based venv switching.
- [ ] Approve isolated engine-worker pattern.
- [ ] Define initial engine grouping.
- [ ] Define GPU tenant lock policy.

### Repository tasks

- [ ] Create `app/services/engine_supervisor.py`.
- [ ] Create `app/services/gpu_tenant_lock.py`.
- [ ] Create `app/domain/engine_events.py`.
- [ ] Create `app/domain/generation_requests.py`.
- [ ] Create `app/domain/training_requests.py`.
- [ ] Create `engines/generation/worker.py`.
- [ ] Create `engines/kohya/worker.py`.
- [ ] Create `engines/ed2/worker.py`.
- [ ] Create `outputs/jobs/` structure.

### UI tasks

- [ ] Ensure UI tabs call service clients only.
- [ ] Add job status display component.
- [ ] Add log stream component.
- [ ] Add stop/cancel button connected to supervisor.
- [ ] Add blocked-GPU message state.

### Training tasks

- [ ] Add dataset preflight validator.
- [ ] Add simple LoRA config builder.
- [ ] Add Kohya subprocess runner.
- [ ] Add ED2 subprocess runner.
- [ ] Add recursive process cleanup.
- [ ] Add output model registration.

### Documentation tasks

- [ ] Add this architecture document under `docs/architecture/`.
- [ ] Add ED2 permission note under `docs/permissions/`.
- [ ] Add setup docs for each engine venv.
- [ ] Add user-facing explanation: “one app, isolated engines.”

---

## 18. Recommended Language for Public/Contributor Explanation

Use this phrasing:

```text
AIWF Studio is designed as one stable user-facing application with isolated backend engines. The Gradio UI does not import every training or generation backend into one shared Python runtime. Instead, it supervises separate workers with their own environments, streams logs and progress back to the interface, and manages GPU ownership centrally.
```

Short version:

```text
AIWF Studio is an orchestrator, not a plugin bucket.
```

Engineer-to-engineer version:

```text
The main design difference is process-level dependency isolation. Heavy backends run behind subprocess or service boundaries, so Torch/CUDA/xformers/NumPy conflicts do not poison the main UI runtime.
```

User-facing version:

```text
AIWF Studio keeps the app simple for users by doing the hard environment management underneath the interface.
```

---

## 19. Final Decision

The project should proceed with this architecture:

```text
One Gradio UI.
One main AIWF Python 3.10 environment.
One generation engine venv for image/Wan/LTX at first.
One Kohya LoRA training venv.
One ED2 full-training venv.
Ollama as an external local service.
No Gradio reloads between venvs.
No tab-owned Python environments.
Heavy engines are isolated subprocess workers.
The UI supervises workers through typed requests, logs, events, and job artifacts.
```

This preserves the “one program” experience while avoiding the dependency-conflict model that makes normal users deal with broken environments and unreadable traces.
