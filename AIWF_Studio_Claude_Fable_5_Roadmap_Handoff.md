# AIWF Studio — Claude Fable 5 Roadmap, Architecture Plan, and Agent Handoff

**Purpose:** Give Claude Fable 5, project-manager agents, and coding agents a coherent implementation plan for AIWF Studio based on the current repo direction plus the uploaded architecture notes.

**Primary rule:** Do **not** remove existing working functionality. Add new layers, wrappers, docs, tests, and feature flags. Only remove duplication or dead code when the replacement is already proven and covered by tests.

**Current product identity:** AIWF Studio is a local-first creative AI workspace with a stable Gradio shell, clean-room architecture, typed domain models, explicit service boundaries, and isolated heavy engines. It began as a clean-room rebuild of an AUTOMATIC1111-style Stable Diffusion web UI, but the roadmap now expands toward image generation, video generation, local chat, LoRA training, full-model training, agentic prompt tools, and hardware-aware acceleration.

---

## 1. Executive Summary

AIWF Studio should evolve as **one user-facing program** with **isolated backend engines**.

The UI must remain stable and boring. The engines can be heavy, fragile, experimental, dependency-sensitive, or GPU-hungry, but they must sit behind process/service boundaries.

```text
AIWF Studio UI
Python 3.10 main environment
Gradio / routing / config / logs / API / model browser / phone companion
        |
        |-- Generation engine
        |       image generation first
        |       Wan / LTX video later
        |       separate worker/venv when needed
        |
        |-- Ollama chat engine
        |       external local service / controlled GPU tenant
        |
        |-- Kohya LoRA training engine
        |       separate training venv
        |
        `-- EveryDream2-compatible full-training engine
                separate Python 3.10 training venv
```

This preserves the key AIWF principle:

```text
Gradio never changes environments.
The engines move underneath it.
```

This is the core difference from extension-bucket architectures:

```text
AIWF Studio is an orchestrator, not a plugin soup.
```

---

## 2. Canonical Repo Context for Agents

### 2.1 Current architectural rules

Agents must keep the existing AIWF Studio rules intact:

- No global `shared` state.
- No mystery callbacks.
- No monkey-patched extension hooks.
- UI callbacks call services, not Torch/diffusers/training internals directly.
- Requests use typed domain models.
- Heavy work belongs in services, infrastructure, workers, or subprocess engines.
- Keep `AppContext` / composition-root wiring clean.
- Update docs when architecture changes.
- Add tests for non-trivial behavior.

### 2.2 Current usable core

Treat the current image workspace as the first working core. Do not destabilize it while adding future engines.

Current shipped/known areas include:

- txt2img / img2img / inpaint
- live preview, continuous generation, interrupt
- hires fix, sampler, CFG, steps, clip skip, VAE selection
- dynamic prompts, wildcards, prompt files, Compel support
- style presets
- ControlNet single-unit support
- SAM-assisted masking
- ReActor-style result face swap
- model manager, segment, enhance, workflows, library, PNG info, history, settings
- native `/api/v1`
- A1111-style `/sdapi/v1` adapter

### 2.3 Current technical baseline

- Python 3.10+
- Windows-first local development
- repo-local `venv/`
- Torch CUDA installed by `launch.py`
- Gradio UI
- FastAPI API surface
- current requirements should not be casually bumped
- `transformers` must remain `<5` unless a controlled migration proves otherwise

---

## 3. Roadmap Overview

### Phase 0 — Audit, Guardrails, and Branch Discipline

**Goal:** Make sure Claude starts from current repo truth, not from stale notes.

**Tasks:**

1. Read:
   - `README.md`
   - `AGENTS.md`
   - `pyproject.toml`
   - `requirements.txt`
   - `launch.py`
   - `aiwf/app.py`
   - `aiwf/bootstrap.py`
   - `aiwf/web/app.py`
   - `aiwf/services/generation.py`
   - `aiwf/infrastructure/diffusers/backend.py`
2. Run:
   ```powershell
   python -m pytest tests/ -q
   ```
3. Do not start by editing the generation backend.
4. Add docs and architecture boundaries first.
5. Create a short implementation checklist before code changes.

**Acceptance criteria:**

- Claude reports current test count/result.
- Claude lists files it intends to add or modify.
- Claude identifies whether any planned dependency changes are required.

---

### Phase 1 — Engine-Supervisor Foundation

**Goal:** Add an orchestration layer that can manage heavy backend tenants without moving Gradio between venvs.

**Additive files to create:**

```text
aiwf/core/domain/engine.py
aiwf/core/orchestration/tenant_manager.py
aiwf/services/engine_supervisor.py
tests/test_engine_supervisor.py
docs/ENGINE_ISOLATION_ARCHITECTURE.md
```

**Responsibilities:**

- Track active engine tenant.
- Provide a lock so only one GPU-heavy tenant owns the card.
- Expose friendly error messages if another tenant is running.
- Kill subprocess trees cleanly.
- Flush PyTorch CUDA memory only from processes that import Torch.
- Avoid importing Torch in UI modules unless already established by current architecture.
- Keep engine state in service objects, not global mutable UI state.

**Tenant categories:**

```text
image
video
chat
lora_training
full_training
enhance
idle
```

**Design note:** A singleton-like supervisor may be created through `AppContext`, but avoid raw module-level mutable global state. If a tab needs the supervisor, it should receive it from context/service wiring.

---

### Phase 2 — Ollama Chat MVP

**Goal:** Add local chat as an isolated tenant using Ollama first, not vLLM.

**Why Ollama first:**

- It is easier on Windows.
- It runs as an external service.
- It avoids making the main UI manage LLM weights directly.
- It can be unloaded with `keep_alive: 0`.
- It is a better MVP than vLLM for this repo state.

**Additive files to create:**

```text
aiwf/services/ollama_client.py
aiwf/web/tabs/chat_workspace.py
tests/test_ollama_client.py
docs/OLLAMA_CHAT_TENANT.md
```

**Minimum features:**

- Endpoint config, default `http://127.0.0.1:11434`
- model name field
- streaming chat UI
- explicit unload call before video/training generation
- graceful error if Ollama is not installed/running
- no mandatory auto-download on first pass
- optional bootstrap helper later

**Do not do yet:**

- Do not force-download Ollama from a vague URL.
- Do not hardcode `C:\Users\Shawn\...`.
- Do not assume `http://127.0.0` is valid.
- Do not make chat and Wan coexist in VRAM.

---

### Phase 3 — Process Supervisor and Worker Protocol

**Goal:** Define a general worker contract before implementing Wan, LTX, Kohya, or ED2 workers.

**Additive files to create:**

```text
aiwf/core/domain/worker.py
aiwf/services/process_supervisor.py
aiwf/services/worker_events.py
tests/test_process_supervisor.py
docs/WORKER_PROTOCOL.md
```

**Worker event model:**

```text
started
stdout
stderr
progress
metric
warning
error
cancelled
completed
```

**Worker request model should include:**

```text
job_id
engine_type
command
cwd
env
created_at
timeout_policy
gpu_lock_required
output_dir
```

**Worker result model should include:**

```text
job_id
status
return_code
output_paths
logs_path
metrics
error_message
```

**Rules:**

- Never execute user-supplied raw shell strings.
- Use `subprocess.Popen([...], shell=False)`.
- Sanitize all file paths.
- Prefer `pathlib.Path`.
- Use `psutil` for recursive process cleanup.
- On Windows, support process groups.
- On POSIX, support process groups/signals.
- Always stream logs back to Gradio through service-layer generators.

---

### Phase 4 — Training Engines: Kohya and ED2

**Goal:** Add training without importing training frameworks into the main AIWF UI process.

**Architecture:**

```text
AIWF Studio main venv
        |
        |-- launches
        v
engines/kohya/.venv/
engines/ed2/.venv/
```

**Kohya path:**

```text
aiwf/web/tabs/training.py
aiwf/services/training/dataset_validator.py
aiwf/services/training/kohya_config.py
aiwf/services/training/kohya_runner.py
docs/TRAINING_KOHYA_ROADMAP.md
```

**ED2 path:**

```text
aiwf/services/training/ed2_config.py
aiwf/services/training/ed2_runner.py
docs/TRAINING_ED2_ROADMAP.md
```

**Dataset preflight checks:**

- Dataset root exists.
- Expected image formats exist.
- Captions paired where required.
- Empty directories flagged.
- Suspicious filenames flagged.
- Repeats/concept folder conventions validated.
- Output folder is writable.
- Base checkpoint path exists.
- Training engine venv exists or setup is guided.

**ED2 design rule:**

AIWF Studio owns the user experience:

- UI
- config generation
- dataset validation
- launch
- logs
- stop/cleanup
- output registration

ED2 owns full-model training internals.

Do **not** import ED2 internals directly into the AIWF UI.

---

### Phase 5 — Video Engine Roadmap: Wan / LTX

**Goal:** Prepare video support behind a safe worker boundary. Do not destabilize image generation.

**Additive docs first:**

```text
docs/VIDEO_ENGINE_ROADMAP.md
docs/WAN_LTX_MEMORY_STRATEGY.md
docs/VIDEO_ENGINE_EXPERIMENT_FLAGS.md
```

**Future additive files:**

```text
engines/generation/
engines/generation/worker.py
aiwf/services/video_generation_client.py
aiwf/core/domain/video_generation.py
aiwf/web/tabs/video_workspace.py
```

**Required concepts:**

- Single-tenant GPU lock.
- Chat unload before video.
- Training blocked while video is running.
- Worker crash must not kill UI.
- Result path returned to Gradio, not raw video arrays.
- NVENC export after frames are materialized.
- VAE decode must be explicit and memory-budgeted.
- LTX frame constraints must be validated before launch.
- Wan/LTX acceleration features are behind flags until benchmarked.

---

### Phase 6 — Performance Experiments

**Goal:** Create an experimental acceleration lane without pretending every optimization is already safe.

**Additive docs:**

```text
docs/ACCELERATION_EXPERIMENTS.md
docs/BENCHMARK_PROTOCOL.md
```

**Experiment candidates from the uploaded notes:**

1. CUDA Graphs for SDXL image denoising.
2. `torch.compile` for inner transformer/UNet blocks.
3. channels-last layouts for image UNet paths.
4. SageAttention 2 / Triton-Windows experiments.
5. torchao native quantization experiments.
6. FP8 execution paths for Ada Lovelace hardware.
7. NVENC video export.
8. RIFE/RIF interpolation.
9. RTX VSR research track.

**Hard rule:** These are experiments until proven by benchmark logs.

**Benchmark every change with:**

```text
prompt
seed
checkpoint
resolution
steps
sampler
batch size
ControlNet/LoRA on/off
VRAM peak
shared GPU memory usage
first-run compile time
second-run latency
output image hash or artifact sample
```

**Do not claim:**

- “beats ComfyUI”
- “30x faster”
- “2x speedup”
- “native VSR active”
- “works on all GPUs”

unless a local benchmark supports it.

---

### Phase 7 — Agentic Prompt/Tool Workspace

**Goal:** Add an internal assistant/tooling layer only after engine boundaries are stable.

**Additive docs:**

```text
docs/AGENTIC_ASSISTANT_ROADMAP.md
docs/LOCAL_TOOL_SECURITY.md
```

**Possible tools:**

- list local checkpoints
- list local LoRAs
- read safetensors metadata where safe
- inspect prompt/style libraries
- build prompt draft
- recommend settings
- generate workflow JSON
- send request to image generation API
- report output path

**Security rules:**

- No raw shell command tool.
- No unconstrained file write.
- No deleting files.
- No arbitrary Python execution.
- No agent-triggered training without confirmation.
- No agent-triggered video generation if GPU lock is held.

---

## 4. Review Notes: What to Keep, What to Defer, What to Fix

### 4.1 Keep as architectural direction

- Process-isolated tenant supervisor.
- One stable Gradio UI.
- Isolated engines behind services/subprocesses.
- Ollama as first chat MVP.
- GPU tenant lock.
- Dataset validator before training.
- ED2 and Kohya as subprocess engines.
- CUDA Graphs as an image-speed experiment.
- NVENC export as a video-output path.
- Benchmark-first claims.

### 4.2 Defer until verified

These ideas may be useful, but should not be implemented as hard facts without verifying APIs, package status, and local hardware behavior:

- vLLM on Windows as the first chat backend.
- torchao APIs such as `change_linear_weights_to_float8_e4m3fn`.
- SageAttention 2.2.0 + Triton-Windows install automation.
- Any direct claim that PyTorch bicubic interpolation triggers RTX VSR.
- Any assumption that `torchaudio.io.StreamReader/StreamWriter` CUDA video support is available in the user’s installed build.
- Full Wan 14B native FP8 residency in 16 GB VRAM without real local benchmarks.
- TensorRT/VSR/RIFE speed claims without local measurements.

### 4.3 Fix obvious defects from the notes before implementation

- Replace hardcoded `C:/Users/Shawn/Desktop/AIWF-Studio` with `Path` values from runtime config.
- Replace invalid `http://127.0.0` with `http://127.0.0.1:11434`.
- Use `shell=False` subprocess lists.
- Make Ollama auto-bootstrap optional, not default.
- Do not run tab switching as an automatic heavy model load until user chooses it.
- Do not use raw module-level `supervisor = GPUWorkloadSupervisor()` unless existing app context cannot inject it.
- Do not put `torch` imports into UI-only modules if avoidable.
- Handle Windows and POSIX process termination separately.
- Make acceleration packages optional extras, not mandatory baseline requirements.

---

## 5. Recommended File Additions

### 5.1 Documentation

```text
docs/ROADMAP.md
docs/ENGINE_ISOLATION_ARCHITECTURE.md
docs/WORKER_PROTOCOL.md
docs/OLLAMA_CHAT_TENANT.md
docs/TRAINING_ENGINE_ROADMAP.md
docs/VIDEO_ENGINE_ROADMAP.md
docs/ACCELERATION_EXPERIMENTS.md
docs/BENCHMARK_PROTOCOL.md
docs/LOCAL_TOOL_SECURITY.md
docs/CLAUDE_FABLE_5_HANDOFF.md
```

### 5.2 Core/domain/service layers

```text
aiwf/core/domain/engine.py
aiwf/core/domain/worker.py
aiwf/core/domain/chat.py
aiwf/core/domain/training.py
aiwf/core/domain/video_generation.py

aiwf/services/engine_supervisor.py
aiwf/services/process_supervisor.py
aiwf/services/ollama_client.py
aiwf/services/training/dataset_validator.py
aiwf/services/training/kohya_config.py
aiwf/services/training/kohya_runner.py
aiwf/services/training/ed2_config.py
aiwf/services/training/ed2_runner.py
```

### 5.3 UI tabs

```text
aiwf/web/tabs/chat_workspace.py
aiwf/web/tabs/training.py
aiwf/web/tabs/video_workspace.py
```

### 5.4 Future workers

```text
engines/generation/README.md
engines/kohya/README.md
engines/ed2/README.md
```

### 5.5 Tests

```text
tests/test_engine_supervisor.py
tests/test_process_supervisor.py
tests/test_ollama_client.py
tests/test_dataset_validator.py
tests/test_training_config_builders.py
tests/test_worker_protocol.py
```

---

## 6. AGENTS.md Addendum to Append

Append the following to the existing `AGENTS.md`. Do not replace the existing file.

```md
---

## Additive roadmap: isolated engines and GPU tenant supervision

AIWF Studio is moving toward a one-shell / many-engine architecture.

The Gradio UI must remain in the main Python 3.10 environment. Heavy engines must be supervised through services, typed domain models, and worker/process boundaries.

### Current direction

```text
AIWF Studio main UI
    Gradio / API / settings / logs / model browser
        |
        |-- Generation engine: image first, Wan/LTX later
        |-- Chat engine: Ollama first, vLLM optional/research
        |-- Kohya LoRA training engine: separate venv
        `-- EveryDream2-compatible full-training engine: separate Python 3.10 venv
```

### Non-negotiable engine rules

1. Do not import Kohya, EveryDream2, vLLM, Wan video stacks, or other dependency-heavy engines directly into UI callbacks.
2. UI callbacks must call services.
3. Services must use typed request/response models.
4. Workers must stream structured events back to the UI.
5. Only one GPU-heavy tenant may run at a time unless a specific feature proves safe concurrency.
6. Killing a worker must clean its child processes.
7. Training and video engines must never silently share the GPU with chat.
8. Do not hide raw tracebacks from logs, but show friendly error messages in the UI.
9. Acceleration paths must be feature-flagged until benchmarked.
10. Do not claim performance wins without reproducible benchmark logs.

### Preferred first implementation order

1. Documentation and tests.
2. Engine tenant domain models.
3. Process supervisor.
4. Ollama chat tenant MVP.
5. Dataset validator.
6. Kohya config/runner.
7. ED2 config/runner.
8. Video engine worker contract.
9. Acceleration experiments.

### Risky ideas requiring verification

- vLLM on Windows.
- SageAttention 2 / Triton-Windows install behavior.
- torchao FP8 APIs.
- RTX VSR from PyTorch interpolation.
- torchaudio CUDA video IO availability.
- Full 14B Wan FP8 residency in 16 GB VRAM.

Treat these as experiments until proven locally.
```

---

## 7. Consolidated Implementation Skeletons

These are starting-point patterns for Claude. They are not final code. Claude must adapt them to the actual repo structure and test them.

### 7.1 Engine domain model

```python
# aiwf/core/domain/engine.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


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
    allow_wait: bool = False


@dataclass(frozen=True)
class EngineSwitchResult:
    ok: bool
    active: EngineTenant
    message: str
    log_path: Optional[Path] = None
```

### 7.2 Process supervisor skeleton

```python
# aiwf/services/process_supervisor.py
from __future__ import annotations

import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import psutil


@dataclass(frozen=True)
class WorkerCommand:
    args: List[str]
    cwd: Path
    env: Dict[str, str]
    name: str


class ProcessSupervisor:
    def __init__(self) -> None:
        self._processes: Dict[str, subprocess.Popen[str]] = {}

    def start(self, worker_id: str, command: WorkerCommand) -> Iterator[str]:
        if worker_id in self._processes and self._processes[worker_id].poll() is None:
            yield f"{command.name} is already running."
            return

        creationflags = 0
        preexec_fn = None

        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            preexec_fn = os.setsid

        process = subprocess.Popen(
            command.args,
            cwd=str(command.cwd),
            env=command.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            shell=False,
            creationflags=creationflags,
            preexec_fn=preexec_fn,
        )
        self._processes[worker_id] = process

        yield f"{command.name} started."

        if process.stdout is not None:
            for line in process.stdout:
                clean = line.rstrip()
                if clean:
                    yield clean

        code = process.wait()
        yield f"{command.name} exited with code {code}."

    def stop(self, worker_id: str) -> str:
        process = self._processes.get(worker_id)
        if process is None or process.poll() is not None:
            return "No running process found."

        try:
            parent = psutil.Process(process.pid)
            children = parent.children(recursive=True)

            for child in children:
                child.terminate()
            parent.terminate()

            gone, alive = psutil.wait_procs([parent, *children], timeout=5)
            for proc in alive:
                proc.kill()

            return "Process tree stopped."
        except psutil.Error as exc:
            return f"Process cleanup warning: {exc}"
```

### 7.3 Ollama client skeleton

```python
# aiwf/services/ollama_client.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional

import httpx


@dataclass(frozen=True)
class OllamaChatMessage:
    role: str
    content: str


class OllamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11434") -> None:
        self.base_url = base_url.rstrip("/")

    def healthcheck(self) -> bool:
        try:
            with httpx.Client(timeout=2.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def unload(self, model: str) -> bool:
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.post(
                    f"{self.base_url}/api/generate",
                    json={"model": model, "keep_alive": 0},
                )
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def stream_chat(
        self,
        model: str,
        messages: List[OllamaChatMessage],
        options: Optional[Dict[str, object]] = None,
    ) -> Iterator[str]:
        payload = {
            "model": model,
            "messages": [{"role": item.role, "content": item.content} for item in messages],
            "stream": True,
        }
        if options:
            payload["options"] = options

        with httpx.stream("POST", f"{self.base_url}/api/chat", json=payload, timeout=60.0) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                token = data.get("message", {}).get("content", "")
                if token:
                    yield token
```

### 7.4 Engine supervisor skeleton

```python
# aiwf/services/engine_supervisor.py
from __future__ import annotations

import gc
import threading
from typing import Optional

from aiwf.core.domain.engine import EngineSwitchRequest, EngineSwitchResult, EngineTenant
from aiwf.services.ollama_client import OllamaClient


class EngineSupervisor:
    def __init__(self, ollama_client: Optional[OllamaClient] = None) -> None:
        self._lock = threading.RLock()
        self._active = EngineTenant.IDLE
        self._ollama = ollama_client or OllamaClient()
        self._default_chat_model = "llama3:8b"

    @property
    def active(self) -> EngineTenant:
        with self._lock:
            return self._active

    def request_switch(self, request: EngineSwitchRequest) -> EngineSwitchResult:
        with self._lock:
            if self._active == request.target:
                return EngineSwitchResult(True, self._active, "Engine already active.")

            if self._active == EngineTenant.CHAT:
                self._ollama.unload(self._default_chat_model)

            if request.target in {EngineTenant.IMAGE, EngineTenant.VIDEO, EngineTenant.LORA_TRAINING, EngineTenant.FULL_TRAINING}:
                self._flush_cuda_if_available()

            self._active = request.target
            return EngineSwitchResult(True, self._active, f"Switched to {request.target.value}.")

    def _flush_cuda_if_available(self) -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            # Keep the UI/service layer alive even when torch is unavailable.
            return
```

### 7.5 Training runner skeleton

```python
# aiwf/services/training/ed2_runner.py
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from aiwf.services.process_supervisor import ProcessSupervisor, WorkerCommand


class ED2Runner:
    def __init__(self, supervisor: ProcessSupervisor, ed2_python: Path, ed2_script: Path) -> None:
        self.supervisor = supervisor
        self.ed2_python = ed2_python
        self.ed2_script = ed2_script

    def start(self, config_path: Path, output_dir: Path) -> Iterator[str]:
        command = WorkerCommand(
            name="EveryDream2-compatible training worker",
            args=[
                str(self.ed2_python),
                str(self.ed2_script),
                "--config",
                str(config_path),
                "--output_dir",
                str(output_dir),
            ],
            cwd=self.ed2_script.parent,
            env={},
        )
        yield from self.supervisor.start("ed2", command)

    def stop(self) -> str:
        return self.supervisor.stop("ed2")
```

---

## 8. Claude Fable 5 Prompt

Use this as the prompt to hand to Claude Fable 5.

```text
You are Claude Fable 5 acting as Principal AI Systems Architect, Staff Python Engineer, and cautious project manager for AIWF Studio.

Repository: https://github.com/nawnie/AIWF-Studio

Mission:
Turn AIWF Studio into a local-first creative AI console with one stable Gradio UI and isolated heavy engines. The current working image-generation workspace must remain stable. Add architecture, docs, tests, and service boundaries first. Do not rip out working code.

Hard rules:
1. Do not remove existing functionality.
2. Do not rewrite large working areas unless there is a clear bug and tests.
3. Only remove duplication or dead code when the replacement is already added and validated.
4. Do not import dependency-heavy engines such as Kohya, EveryDream2, vLLM, Wan video stacks, or LTX directly into Gradio callbacks.
5. UI callbacks call services.
6. Services use typed domain models.
7. Infrastructure and worker layers handle Torch, subprocesses, filesystem, and external services.
8. Keep clean-room rules intact.
9. Keep AGENTS.md as canonical guidance; append addendums instead of replacing it.
10. Do not claim performance wins without benchmark logs.

Current architectural direction:
- One main AIWF Studio Python 3.10 environment.
- One stable Gradio UI shell.
- One generation engine path for image now; Wan/LTX later behind worker boundaries.
- Ollama as the first local chat tenant.
- Kohya LoRA training as a separate venv/subprocess engine.
- EveryDream2-compatible full training as a separate Python 3.10 venv/subprocess engine.
- Optional experimental acceleration paths behind flags only.

Start by reading:
- README.md
- AGENTS.md
- pyproject.toml
- requirements.txt
- launch.py
- aiwf/app.py
- aiwf/bootstrap.py
- aiwf/web/app.py
- aiwf/services/generation.py
- aiwf/infrastructure/diffusers/backend.py
- tests/

Before coding:
1. Summarize current repo state.
2. Run or specify the test command: python -m pytest tests/ -q.
3. Identify which files you will add.
4. Identify any risky dependency or API assumptions.
5. Propose a small first patch.

First patch target:
Add documentation and foundation types for the isolated-engine architecture. Do not modify the generation backend yet.

Recommended first files:
- docs/ENGINE_ISOLATION_ARCHITECTURE.md
- docs/WORKER_PROTOCOL.md
- docs/ACCELERATION_EXPERIMENTS.md
- aiwf/core/domain/engine.py
- aiwf/core/domain/worker.py
- aiwf/services/process_supervisor.py
- aiwf/services/engine_supervisor.py
- tests/test_process_supervisor.py
- tests/test_engine_supervisor.py

Implementation style:
- Python 3.10-compatible typing.
- Prefer dataclasses or existing Pydantic conventions where the repo already uses them.
- Use pathlib.Path.
- Use shell=False subprocess lists.
- No hardcoded C:\Users\Shawn paths.
- No raw user shell execution.
- Friendly UI errors, detailed logs.
- Feature flags for experiments.
- Tests for process cleanup, switch behavior, path validation, and config generation.

Important corrections from architecture notes:
- Use Ollama first for chat; vLLM is optional/research, not MVP.
- Do not auto-download Ollama by default. Provide detection and setup guidance first.
- Treat RTX VSR, torchao FP8 APIs, SageAttention 2/Triton-Windows, and torchaudio CUDA video IO as experimental until verified.
- Do not assert ComfyUI-beating performance without benchmark evidence.
- Do not make tab selection automatically load huge models until explicitly confirmed by the user.

Deliverables:
1. A short implementation plan.
2. The patch.
3. Tests or test notes.
4. A summary of what changed.
5. A clear list of follow-up tasks.

Tone:
Be direct, practical, and careful. This project is meant to become a serious creative tool, not a pile of clever hacks.
```

---

## 9. Project Manager Roadmap Checklist

### Sprint A — Docs and Guardrails

- [ ] Add `docs/ENGINE_ISOLATION_ARCHITECTURE.md`
- [ ] Add `docs/WORKER_PROTOCOL.md`
- [ ] Add `docs/ACCELERATION_EXPERIMENTS.md`
- [ ] Append AGENTS.md addendum
- [ ] Confirm README and AGENTS agree
- [ ] Run tests

### Sprint B — Engine Supervisor Foundation

- [ ] Add engine domain types
- [ ] Add process supervisor
- [ ] Add engine supervisor
- [ ] Add tests
- [ ] Ensure no UI imports heavy engines

### Sprint C — Ollama Chat MVP

- [ ] Add Ollama client
- [ ] Add Chat Workspace tab
- [ ] Add model unload behavior
- [ ] Add friendly setup warning if Ollama unavailable
- [ ] Add tests/mocks

### Sprint D — Training Services

- [ ] Add dataset validator
- [ ] Add training config builders
- [ ] Add Kohya subprocess runner
- [ ] Add ED2 subprocess runner
- [ ] Add stop/cleanup
- [ ] Add output registration

### Sprint E — Video Worker Design

- [ ] Add video request domain models
- [ ] Add video worker protocol
- [ ] Add result-path return behavior
- [ ] Add GPU lock behavior
- [ ] Keep Wan/LTX under feature flags

### Sprint F — Performance Experiments

- [ ] Add benchmark harness
- [ ] Test CUDA Graphs for SDXL
- [ ] Test `torch.compile` feature flag
- [ ] Test NVENC export
- [ ] Research VSR path without overclaiming
- [ ] Record results before README claims

---

## 10. Definition of Done

A task is not complete until:

- It preserves existing features.
- It follows the service-boundary architecture.
- It adds or updates tests when logic changes.
- It has user-facing docs if it changes workflow.
- It avoids hardcoded local paths.
- It avoids raw shell execution.
- It has clean error handling.
- It does not claim benchmark gains without data.
- It keeps README, AGENTS.md, and docs aligned.

---

## 11. Final Guidance for Claude

Build the boring foundation first.

The exciting parts — Wan, LTX, CUDA Graphs, torchao FP8, SageAttention, RTX VSR, ED2, Kohya, tool agents — all become much easier if AIWF Studio first has a trustworthy supervisor, worker protocol, and GPU tenant lock.

Do not chase raw speed before containment.

Containment first. Then acceleration.
