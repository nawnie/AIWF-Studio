"""EveryDream2 full fine-tuning engine worker entry point.

Runs in engines/ed2/.venv/ — a separate venv from the main AIWF UI.

What this worker does:
  1. Reads ED2TrainingRequest fields from request.json (written by ed2_client.py).
  2. Builds a train.json config from the request (via ED2TrainingRequest.to_ed2_config()).
  3. Calls EveryDream2trainer/train.py via subprocess.
  4. Streams JSONL progress events to stdout so the supervisor can update the UI.

ED2 does not support Textual Inversion by default — no exclusion logic needed.

Invocation (by EngineSupervisor):
    engines/ed2/.venv/Scripts/python.exe engines/ed2/worker.py <path/to/request.json>

Required request.json fields (ED2TrainingRequest):
    _job_id, _engine="ed2", _repo_dir, job_name, base_model_path, dataset_dir, ...
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

_WORKER_DIR = Path(__file__).resolve().parent
_ENGINES_DIR = _WORKER_DIR.parent
_ROOT = _ENGINES_DIR.parent
sys.path.insert(0, str(_ROOT))

try:
    from aiwf.engine_workers.base import (
        WorkerContext, emit_artifact, emit_complete, emit_error,
        emit_heartbeat, emit_progress, emit_status, run_worker,
    )
except ImportError:
    # Fallback inline helpers — same pattern as kohya/worker.py
    import json as _json
    from datetime import datetime, timezone

    def _ts():
        return datetime.now(timezone.utc).isoformat()

    def _emit(obj):
        print(_json.dumps(obj, ensure_ascii=False), flush=True)

    def emit_status(job_id, message):
        _emit({"kind": "status", "job_id": job_id, "message": message, "ts": _ts()})

    def emit_progress(job_id, *, step, total, message=""):
        _emit({"kind": "progress", "job_id": job_id, "step": step,
               "total": total, "message": message, "ts": _ts()})

    def emit_artifact(job_id, *, path):
        _emit({"kind": "artifact", "job_id": job_id, "path": path, "ts": _ts()})

    def emit_complete(job_id, message=""):
        _emit({"kind": "complete", "job_id": job_id, "message": message, "ts": _ts()})

    def emit_error(job_id, *, detail, message="job failed"):
        _emit({"kind": "error", "job_id": job_id, "detail": detail,
               "message": message, "ts": _ts()})

    def emit_heartbeat(job_id):
        _emit({"kind": "heartbeat", "job_id": job_id, "ts": _ts()})

    class WorkerContext:
        def __init__(self, job_id, request, request_file):
            self.job_id = job_id
            self.request = request
            self.request_file = request_file
        @classmethod
        def from_argv(cls):
            if len(sys.argv) < 2:
                sys.exit(2)
            f = Path(sys.argv[1])
            raw = _json.loads(f.read_text(encoding="utf-8"))
            return cls(raw.get("_job_id", "unknown"), raw, f)
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def run_worker(fn):
        ctx = WorkerContext.from_argv()
        with ctx:
            fn(ctx)


# ED2 epoch/step log patterns
_EPOCH_RE = re.compile(r"Epoch\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)
_STEP_RE  = re.compile(r"step[:\s]+(\d+)\s*/\s*(\d+)", re.IGNORECASE)
_LOSS_RE  = re.compile(r"loss[:\s=]+([0-9.eE+\-]+)", re.IGNORECASE)


def _build_ed2_config(req: dict, job_dir: Path, repo_dir: Path) -> Path:
    """Write a train.json that ED2's train.py will accept."""
    output_dir = Path(req.get("output_dir", "outputs/training/ed2"))
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg: dict = {
        "model": req["base_model_path"],
        "train_data_dir": req["dataset_dir"],
        "output_dir": str(output_dir),
        "log_dir": req.get("log_dir", str(job_dir / "logs")),
        "resolution": req.get("resolution", 512),
        "flip_p": req.get("flip_p", 0.0),
        "max_epochs": req.get("max_epochs", 20),
        "batch_size": req.get("batch_size", 4),
        "lr": req.get("lr", 1.5e-6),
        "lr_scheduler": req.get("lr_scheduler", "constant"),
        "lr_warmup_steps": req.get("lr_warmup_steps", 0),
        "optimizer": req.get("optimizer", "adamw"),
        "mixed_precision": req.get("mixed_precision", "bf16"),
        "gradient_checkpointing": req.get("gradient_checkpointing", True),
        "clip_skip": req.get("clip_skip", 2),
        "seed": req.get("seed", 42),
        "save_every_n_epochs": req.get("save_every_n_epochs", 1),
        "save_last_n_epochs": req.get("save_last_n_epochs", 3),
        "ckpt_type": req.get("ckpt_type", "safetensors"),
        "project_name": req.get("job_name", "ed2_training"),
    }

    if req.get("vae_path"):
        cfg["vae"] = req["vae_path"]

    sample_steps = req.get("sample_steps", 0)
    sample_prompts = req.get("sample_prompts", [])
    if sample_steps > 0 and sample_prompts:
        cfg["sample_steps"] = sample_steps
        cfg["sample_prompts"] = sample_prompts

    config_path = job_dir / "train.json"
    config_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    return config_path


def main(ctx: WorkerContext) -> None:
    req = ctx.request
    job_id = ctx.job_id

    repo_dir = Path(req.get("_repo_dir", str(_WORKER_DIR / "EveryDream2trainer")))
    if not repo_dir.exists():
        raise RuntimeError(
            f"EveryDream2trainer repository not found at {repo_dir}. "
            "Clone it and configure repo_dir in engines.json."
        )

    train_script = repo_dir / "train.py"
    if not train_script.exists():
        raise FileNotFoundError(
            f"ED2 train.py not found at {train_script}. "
            "Ensure the EveryDream2trainer repository is properly cloned."
        )

    job_dir = ctx.request_file.parent
    (job_dir / "logs").mkdir(parents=True, exist_ok=True)

    job_name = req.get("job_name", "ed2_job")
    max_epochs = req.get("max_epochs", 20)

    emit_status(job_id, f"Building ED2 config for {job_name!r} ({max_epochs} epochs)")
    config_path = _build_ed2_config(req, job_dir, repo_dir)
    emit_status(job_id, f"Config written to {config_path}")

    cmd = [
        sys.executable,
        str(train_script),
        f"--config={config_path}",
    ]

    emit_status(job_id, f"Launching ED2: {' '.join(str(c) for c in cmd)}")

    # Heartbeat thread
    _stop_heartbeat = threading.Event()

    def _heartbeat_loop():
        while not _stop_heartbeat.wait(30):
            emit_heartbeat(job_id)

    hb_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
    hb_thread.start()

    current_epoch = 0
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONPATH": str(repo_dir)},
        )

        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue

            m_epoch = _EPOCH_RE.search(line)
            m_step  = _STEP_RE.search(line)
            m_loss  = _LOSS_RE.search(line)

            if m_epoch:
                current_epoch = int(m_epoch.group(1))
                total_epochs = int(m_epoch.group(2))
                loss_str = f" | loss: {m_loss.group(1)}" if m_loss else ""
                emit_progress(job_id, step=current_epoch, total=total_epochs,
                              message=f"Epoch {current_epoch}/{total_epochs}{loss_str}")
            elif m_step:
                step = int(m_step.group(1))
                total = int(m_step.group(2))
                emit_progress(job_id, step=step, total=total, message=f"Step {step}/{total}")
            else:
                emit_status(job_id, line[:200])

        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"ED2 training exited with code {proc.returncode}")

    finally:
        _stop_heartbeat.set()

    # Collect output checkpoints
    output_dir = Path(req.get("output_dir", "outputs/training/ed2"))
    for ckpt in sorted(output_dir.glob("*.safetensors")):
        emit_artifact(job_id, path=str(ckpt))
    for ckpt in sorted(output_dir.glob("*.ckpt")):
        emit_artifact(job_id, path=str(ckpt))

    emit_complete(job_id, message=f"ED2 training complete — {current_epoch} epochs.")


if __name__ == "__main__":
    run_worker(main)
