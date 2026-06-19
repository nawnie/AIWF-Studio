# Worker Protocol

This document describes the contract between `ProcessSupervisor` / `EngineSupervisor` and subprocess worker scripts.

---

## Overview

Workers are standalone Python scripts that receive a JSON request file as their first CLI argument, do their work, and communicate back to the supervisor exclusively through **structured JSONL events on stdout**.

```
supervisor → spawns → worker.py <request.json>
supervisor ← reads stdout ← worker (JSONL events)
```

No shared memory. No sockets. No direct imports of UI code.

---

## Command line

```
<engine_python> <worker_script> <request_json_path>
```

- `<engine_python>` — absolute path to the engine's venv Python binary
- `<worker_script>` — absolute path to the worker entry point (`engines/<name>/worker.py`)
- `<request_json_path>` — absolute path to the JSON file written by the supervisor

---

## Request file format

The supervisor writes a JSON file containing the original request plus injected fields:

```json
{
  "_job_id":       "wan_a1b2c3d4",
  "_engine":       "kohya",
  "_repo_dir":     "/path/to/engine/repo",
  "_outputs_root": "/path/to/outputs",
  "...":           "caller-provided fields"
}
```

Workers read this file at startup; they do **not** read stdin.

---

## Event protocol (stdout JSONL)

Every line the worker writes to stdout that begins with `{` and contains a `"kind"` field is treated as a structured event.  All other lines are captured to the crash log as raw text.

### Event types

#### `progress`
Emitted during long operations (e.g., training steps, frame generation).

```json
{"kind": "progress", "job_id": "wan_a1b2", "step": 10, "total": 100, "message": "Step 10/100"}
```

| Field   | Type   | Required | Description |
|---------|--------|----------|-------------|
| kind    | string | ✓ | `"progress"` |
| job_id  | string | ✓ | Matches `_job_id` from the request file |
| step    | int    | ✓ | Current step (0-indexed) |
| total   | int    | ✓ | Total steps |
| message | string |   | Human-readable status |

#### `artifact`
Emitted when an output file is ready.

```json
{"kind": "artifact", "job_id": "wan_a1b2", "path": "/outputs/video.mp4"}
```

| Field  | Type   | Required | Description |
|--------|--------|----------|-------------|
| kind   | string | ✓ | `"artifact"` |
| job_id | string | ✓ | |
| path   | string | ✓ | Absolute path to the produced file |

#### `status`
Emitted for arbitrary human-readable status messages.

```json
{"kind": "status", "job_id": "wan_a1b2", "message": "Loading VAE..."}
```

#### `complete`
**Terminal event.** Emitted once on successful completion. The supervisor sets the job to `completed` and stops reading.

```json
{"kind": "complete", "job_id": "wan_a1b2", "message": "Done — 3 frames rendered"}
```

#### `error`
**Terminal event.** Emitted on failure. The supervisor sets the job to `failed` and stops reading.

```json
{"kind": "error", "job_id": "wan_a1b2", "detail": "CUDA OOM at step 5", "message": "Out of memory"}
```

| Field   | Type   | Required | Description |
|---------|--------|----------|-------------|
| kind    | string | ✓ | `"error"` |
| job_id  | string | ✓ | |
| detail  | string | ✓ | Full technical detail (stack trace, etc.) |
| message | string |   | Short user-facing description |

---

## Worker lifecycle

1. Worker starts, reads `request.json`.
2. Emits zero or more `progress` / `status` / `artifact` events.
3. On success: emits exactly one `complete` event, then exits with code 0.
4. On failure: emits exactly one `error` event, then exits with non-zero code.
5. If the worker is killed (SIGTERM): no terminal event is required. The supervisor detects the unexpected exit and calls `fail_job()` automatically.

---

## Heartbeat (optional)

Workers that run long operations (> 30 seconds between output lines) should periodically emit a `status` event so the supervisor's heartbeat monitor does not declare them hung.

```python
# Every 30 seconds during a long loop:
emit(EngineEvent.status(job_id, f"Training epoch {epoch}..."))
```

The supervisor timeout (`SUBPROCESS_HEARTBEAT_TIMEOUT`) defaults to 120 seconds.

---

## WorkerCommand schema

Defined in `aiwf/core/domain/worker.py`.

```python
@dataclass(frozen=True)
class WorkerCommand:
    args: list[str]       # Full argv for Popen — no shell interpolation
    cwd: Path             # Working directory
    env: dict[str, str]   # Merged into os.environ for the child
    name: str             # Slot name in ProcessSupervisor
    timeout_seconds: int | None = None
```

---

## WorkerResult schema

```python
@dataclass(frozen=True)
class WorkerResult:
    job_id: str
    status: str             # "completed" | "failed" | "cancelled" | "timeout"
    return_code: int | None
    output_paths: list[Path]
    logs_path: Path | None
    error_message: str
```

Convenience constructors: `WorkerResult.success(...)`, `WorkerResult.failure(...)`, `WorkerResult.cancelled_result(...)`.

---

## Adding a new worker

1. Create `engines/<name>/worker.py`.
2. Read `sys.argv[1]` as the request JSON path.
3. Import `from aiwf.core.domain.engine_events import emit, EngineEvent` (or write the JSON manually).
4. Emit events to stdout.
5. Exit 0 on success, non-zero on failure.
6. Register the engine in `launch.py` `EngineSpec` registry.
