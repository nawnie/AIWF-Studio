"""Kohya LoRA training engine worker entry point.

Runs in engines/kohya/.venv/ — a separate venv from the main AIWF UI.

What this worker does:
  1. Reads KohyaLoraRequest fields from request.json (written by kohya_client.py).
  2. Generates a Kohya TOML config from the request.
  3. Calls the appropriate Kohya training script via subprocess.
  4. Streams JSONL progress events to stdout so the supervisor can update the UI.

Textual Inversion is explicitly blocked — requesting it raises an error.

Invocation (by EngineSupervisor):
    engines/kohya/.venv/Scripts/python.exe engines/kohya/worker.py <path/to/request.json>

Required request.json fields (KohyaLoraRequest):
    _job_id, _engine="kohya", _repo_dir, job_name, base_model_path,
    base_arch, dataset_dir, ...
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

# This worker runs in its own venv — aiwf may or may not be importable.
# We use the self-contained base helpers instead of the full aiwf package.
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
    # Fallback: aiwf not importable from this venv — use inline helpers.
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


# ---------------------------------------------------------------------------
# TI block-list — these script names are never allowed through AIWF
# ---------------------------------------------------------------------------
_BLOCKED_SCRIPTS = {
    "train_textual_inversion.py",
    "train_textual_inversion_XL.py",
    "sdxl_train_textual_inversion.py",
    "train_ti.py",
}

_STEP_RE = re.compile(r"(\d+)/(\d+)\s+\[")  # Kohya tqdm pattern "  4/100 ["


def _build_toml(req: dict, job_dir: Path, repo_dir: Path) -> Path:
    """Generate a Kohya TOML config from the request dict."""
    import toml  # available in kohya venv

    job_name: str = req["job_name"]
    output_dir: str = req.get("output_dir", "outputs/training/kohya")
    output_name: str = req.get("output_name", job_name)

    # Kohya TOML structure (accelerate + train_network compatible)
    config = {
        "model_arguments": {
            "pretrained_model_name_or_path": req["base_model_path"],
            "v2": False,
            "v_parameterization": False,
        },
        "dataset_arguments": {
            "train_data_dir": req["dataset_dir"],
            "resolution": str(req.get("resolution", 1024)),
            "enable_bucket": True,
            "caption_extension": req.get("caption_extension", ".txt"),
            "shuffle_caption": True,
            "keep_tokens": 1,
        },
        "training_arguments": {
            "output_dir": str(Path(output_dir)),
            "output_name": output_name,
            "save_every_n_steps": req.get("save_every_n_steps", 500),
            "save_last_n_steps": req.get("save_last_n_steps", 5),
            "max_train_steps": req.get("max_train_steps", 1500),
            "learning_rate": req.get("learning_rate", 1e-4),
            "unet_lr": req.get("unet_lr", req.get("learning_rate", 1e-4)),
            "text_encoder_lr": req.get("text_encoder_lr", 5e-5),
            "lr_scheduler": req.get("lr_scheduler", "cosine_with_restarts"),
            "lr_warmup_steps": req.get("lr_warmup_steps", 100),
            "optimizer_type": req.get("optimizer", "AdamW8bit"),
            "train_batch_size": req.get("batch_size", 1),
            "mixed_precision": req.get("mixed_precision", "bf16"),
            "save_precision": req.get("mixed_precision", "bf16"),
            "gradient_checkpointing": req.get("gradient_checkpointing", True),
            "clip_grad_norm": req.get("clip_grad_norm", 1.0),
            "seed": req.get("seed", 42),
            "logging_dir": str(job_dir / "logs"),
        },
        "network_arguments": {
            "network_module": req.get("network_module", "networks.lora"),
            "network_dim": req.get("network_dim", 32),
            "network_alpha": req.get("network_alpha", 16.0),
        },
    }

    toml_path = job_dir / "kohya_config.toml"
    toml_path.write_text(toml.dumps(config), encoding="utf-8")
    return toml_path


def main(ctx: WorkerContext) -> None:
    req = ctx.request
    job_id = ctx.job_id

    repo_dir = Path(req.get("_repo_dir", str(_WORKER_DIR / "kohya_ss")))
    if not repo_dir.exists():
        raise RuntimeError(
            f"Kohya repository not found at {repo_dir}. "
            "Clone kohya_ss and configure repo_dir in engines.json."
        )

    base_arch = req.get("base_arch", "sdxl")
    training_script_rel = {
        "sd1": "sd_scripts/train_network.py",
        "sdxl": "sd_scripts/sdxl_train_network.py",
        "flux": "flux_train_network.py",
    }.get(base_arch, "sd_scripts/sdxl_train_network.py")

    # Safety: block TI scripts
    script_name = Path(training_script_rel).name
    if script_name in _BLOCKED_SCRIPTS:
        raise ValueError(
            f"Training script {script_name!r} is not allowed through AIWF Studio. "
            "Textual Inversion is excluded by design — use LoRA instead."
        )

    training_script = repo_dir / training_script_rel
    if not training_script.exists():
        raise FileNotFoundError(
            f"Kohya training script not found: {training_script}. "
            "Ensure the kohya_ss repository is properly cloned."
        )

    # Create output and log directories
    job_dir = ctx.request_file.parent
    output_dir = Path(req.get("output_dir", "outputs/training/kohya"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "logs").mkdir(parents=True, exist_ok=True)

    emit_status(job_id, f"Building Kohya config for {req.get('job_name', 'unnamed')} ({base_arch} LoRA)")

    toml_path = _build_toml(req, job_dir, repo_dir)
    emit_status(job_id, f"Config written to {toml_path}")

    max_steps = req.get("max_train_steps", 1500)

    cmd = [
        sys.executable,
        "accelerate",
        "launch",
        "--num_cpu_threads_per_process=1",
        str(training_script),
        f"--config_file={toml_path}",
    ]
    # For Flux, accelerate launch isn't always needed; invoke directly
    if base_arch == "flux":
        cmd = [sys.executable, str(training_script), f"--config_file={toml_path}"]

    emit_status(job_id, f"Launching: {' '.join(str(c) for c in cmd)}")

    # Heartbeat thread — sends a heartbeat every 30s so the supervisor knows we're alive
    _stop_heartbeat = threading.Event()

    def _heartbeat_loop():
        while not _stop_heartbeat.wait(30):
            emit_heartbeat(job_id)

    hb_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
    hb_thread.start()

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

        step = 0
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            # Detect tqdm progress lines
            m = _STEP_RE.search(line)
            if m:
                step = int(m.group(1))
                total = int(m.group(2))
                emit_progress(job_id, step=step, total=total,
                              message=f"Training step {step}/{total}")
            else:
                # Forward log lines as status messages
                emit_status(job_id, line[:200])

        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"Kohya training exited with code {proc.returncode}")

    finally:
        _stop_heartbeat.set()

    # Collect output checkpoints
    output_dir = Path(req.get("output_dir", "outputs/training/kohya"))
    for ckpt in sorted(output_dir.glob("*.safetensors")):
        emit_artifact(job_id, path=str(ckpt))
    for ckpt in sorted(output_dir.glob("*.pt")):
        emit_artifact(job_id, path=str(ckpt))

    emit_complete(job_id, message=f"Kohya LoRA training complete — {step} steps.")


if __name__ == "__main__":
    run_worker(main)
