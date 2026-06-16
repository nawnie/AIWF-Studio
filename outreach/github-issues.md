# GitHub Issue Drafts

Use these as starter issues before public outreach. Apply labels exactly where available:

- `help wanted`
- `good first issue`

## Issue 1: Add worker protocol validation helpers

Labels: `help wanted`, `good first issue`

Title:

```text
Add validation helpers for JSONL worker events
```

Body:

```md
AIWF workers communicate with the core app through JSONL events on stdout. The protocol is documented in `docs/WORKER_PROTOCOL.md`.

Goal:
- Add a small validation helper for worker event dictionaries.
- Cover required fields for `status`, `progress`, `artifact`, `complete`, and `error`.
- Add focused tests around valid and invalid event payloads.

Constraints:
- Do not import optional engine packages.
- Keep the helper pure Python and safe to import at app boot.
- Do not change the wire protocol without discussing it first.

Good starting files:
- `docs/WORKER_PROTOCOL.md`
- `aiwf/core/domain/worker.py`
- `tests/test_process_supervisor.py`
```

## Issue 2: Improve optional engine readiness messages

Labels: `help wanted`, `good first issue`

Title:

```text
Make optional engine readiness errors more actionable
```

Body:

```md
Optional engines such as Wan, Kohya, and ED2 should never block core Studio boot. When they are missing, the UI should show clear setup guidance.

Goal:
- Audit current readiness messages for disabled/missing engine venvs.
- Improve wording so a new developer knows which command or config file to use next.
- Add or update tests for the readiness messages.

Constraints:
- Optional engines must remain optional at import time.
- Do not add top-level imports of `launch`, torch, diffusers, Kohya, or ED2 packages.

Good starting files:
- `aiwf/services/worker_tenant.py`
- `aiwf/services/training/engine_status.py`
- `tests/test_worker_tenant.py`
- `tests/test_training_engine_status.py`
```

## Issue 3: Add docs smoke test for contributor setup commands

Labels: `help wanted`, `good first issue`

Title:

```text
Add a smoke test that contributor docs reference valid local paths
```

Body:

```md
The public setup path depends on `CONTRIBUTING.md`, `ARCHITECTURE.md`, and docs links staying accurate.

Goal:
- Add a lightweight test that checks contributor docs reference existing repo files and scripts.
- Validate paths such as `scripts/bootstrap_engine.ps1`, `docs/WORKER_PROTOCOL.md`, and `docs/ENGINE_ISOLATION.md`.

Constraints:
- The test should not run installs, bootstrap venvs, or require network.
- Keep it fast enough for the normal pytest suite.

Good starting files:
- `CONTRIBUTING.md`
- `ARCHITECTURE.md`
- `tests/`
```

## Issue 4: Record benchmark receipt fields for engine methods

Labels: `help wanted`

Title:

```text
Ensure benchmark receipts identify engine method and model family
```

Body:

```md
AIWF should not claim speed or memory improvements without receipts. Benchmark records need enough metadata to compare methods honestly.

Goal:
- Ensure benchmark receipts include method/backend, model family, text encoder family, LoRA choices, resolution, frames, steps, elapsed time, and output path where applicable.
- Add tests around receipt shape.

Constraints:
- Do not claim a performance improvement in docs unless a receipt supports it.
- Keep optional accelerator packages optional.

Good starting files:
- `aiwf/workers/pipeline_benchmark.py`
- `docs/benchmark_log.jsonl`
- `tests/test_pipeline_benchmark.py`
```

## Issue 5: Improve first-run troubleshooting notes

Labels: `help wanted`, `good first issue`

Title:

```text
Add first-run troubleshooting notes for common Windows setup failures
```

Body:

```md
New contributors need a short troubleshooting path when bootstrap or tests fail.

Goal:
- Add a concise troubleshooting section to `CONTRIBUTING.md`.
- Cover using `.\venv\Scripts\python.exe` instead of system Python.
- Cover missing Python 3.10, CUDA Torch install issues, and optional engine confusion.

Constraints:
- Do not recommend committing generated files or local config.
- Keep the guide focused on contributor setup, not model download walkthroughs.

Good starting files:
- `CONTRIBUTING.md`
- `launch.py`
- `scripts/bootstrap_engine.ps1`
```
