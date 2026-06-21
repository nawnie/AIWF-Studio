"""AI bot text-model training engine worker entry point.

Runs in engines/llm/.venv and trains Causal LM models with TRL SFT.

Supported methods:
  - lora: train a PEFT LoRA adapter on an unquantized base model
  - qlora: load the base model in 4-bit and train a PEFT LoRA adapter
  - full: train all model weights
"""
from __future__ import annotations

import inspect
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

_WORKER_DIR = Path(__file__).resolve().parent
_ROOT = _WORKER_DIR.parent.parent
sys.path.insert(0, str(_ROOT))

try:
    from aiwf.engine_workers.base import (
        WorkerContext, emit_artifact, emit_complete, emit_heartbeat,
        emit_progress, emit_status, run_worker,
    )
except ImportError:
    import json as _json
    from datetime import datetime, timezone

    def _ts():
        return datetime.now(timezone.utc).isoformat()

    def _emit(obj):
        print(_json.dumps(obj, ensure_ascii=False), flush=True)

    def emit_status(job_id, message):
        _emit({"kind": "status", "job_id": job_id, "message": message, "ts": _ts()})

    def emit_progress(job_id, *, step, total, message=""):
        _emit({"kind": "progress", "job_id": job_id, "step": step, "total": total, "message": message, "ts": _ts()})

    def emit_artifact(job_id, *, path):
        _emit({"kind": "artifact", "job_id": job_id, "path": path, "ts": _ts()})

    def emit_complete(job_id, message=""):
        _emit({"kind": "complete", "job_id": job_id, "message": message, "ts": _ts()})

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
            request_file = Path(sys.argv[1])
            raw = _json.loads(request_file.read_text(encoding="utf-8"))
            return cls(raw.get("_job_id", "unknown"), raw, request_file)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def run_worker(fn):
        ctx = WorkerContext.from_argv()
        with ctx:
            fn(ctx)


def main(ctx: WorkerContext) -> None:
    req = dict(ctx.request)
    job_id = ctx.job_id

    if req.get("mode") == "probe":
        emit_status(job_id, "LLM trainer worker probe started")
        emit_complete(job_id, "LLM trainer worker probe complete.")
        return

    from aiwf.services.training.llm_config import write_llm_training_config

    job_dir = ctx.request_file.parent
    config_path = write_llm_training_config(req, job_dir / "llm_training_config.json")
    emit_artifact(job_id, path=str(config_path))

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    output_dir = Path(cfg["job"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if bool(req.get("preview_only", False)):
        emit_complete(job_id, f"LLM training config preview written to {config_path}")
        return

    emit_status(job_id, f"Preparing {cfg['job']['method']} training for {cfg['job']['job_name']}")

    # Heavy ML imports are intentionally deferred until a real training run.
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainerCallback,
        set_seed,
    )
    from trl import SFTConfig, SFTTrainer

    set_seed(int(cfg["job"]["seed"]))

    _start_heartbeat(job_id)

    emit_status(job_id, "Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["base_model_path"],
        trust_remote_code=cfg["model"]["trust_remote_code"],
        local_files_only=cfg["model"]["local_files_only"],
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    emit_status(job_id, "Loading dataset")
    train_dataset = _load_and_prepare_dataset(load_dataset, tokenizer, cfg["dataset"])
    total_rows = len(train_dataset)
    emit_status(job_id, f"Loaded {total_rows} training rows")

    emit_status(job_id, "Loading base model")
    method = str(cfg["job"]["method"])
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": cfg["model"]["trust_remote_code"],
        "local_files_only": cfg["model"]["local_files_only"],
    }
    dtype = _torch_dtype(torch, cfg["training_args"])
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    if method == "qlora":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=cfg["quantization"]["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=cfg["quantization"]["bnb_4bit_use_double_quant"],
            bnb_4bit_compute_dtype=dtype or torch.float16,
        )
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(cfg["model"]["base_model_path"], **model_kwargs)
    if bool(cfg["training_args"]["gradient_checkpointing"]):
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    if method == "qlora":
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(cfg["training_args"]["gradient_checkpointing"]),
        )
    elif torch.cuda.is_available():
        model.to("cuda")

    peft_config = None
    if bool(cfg["peft"]["enabled"]):
        peft_config = LoraConfig(
            r=int(cfg["peft"]["r"]),
            lora_alpha=float(cfg["peft"]["lora_alpha"]),
            lora_dropout=float(cfg["peft"]["lora_dropout"]),
            target_modules=cfg["peft"]["target_modules"],
            task_type="CAUSAL_LM",
        )

    emit_status(job_id, "Starting TRL SFT training")
    sft_args = _build_sft_config(SFTConfig, cfg, output_dir)
    callback = _make_progress_callback(TrainerCallback, job_id, int(cfg["training_args"]["max_steps"]))
    trainer_kwargs = {
        "model": model,
        "args": sft_args,
        "train_dataset": train_dataset,
        "callbacks": [callback],
    }
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config
    _add_tokenizer_arg(SFTTrainer, trainer_kwargs, tokenizer)

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()

    emit_status(job_id, f"Saving trained output to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    for artifact in sorted(output_dir.rglob("*")):
        if artifact.is_file() and artifact.suffix.lower() in {".json", ".safetensors", ".bin", ".model"}:
            emit_artifact(job_id, path=str(artifact))

    emit_complete(job_id, f"LLM {method} training complete. Rows: {total_rows}. Output: {output_dir}")


def _start_heartbeat(job_id: str) -> None:
    stop = threading.Event()

    def _loop() -> None:
        while not stop.wait(30):
            emit_heartbeat(job_id)

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()


def _load_and_prepare_dataset(load_dataset_fn, tokenizer, dataset_cfg: dict) -> Any:
    dataset_path = Path(str(dataset_cfg["dataset_path"]))
    data_files = _dataset_files(dataset_path)
    loaded = load_dataset_fn("json", data_files=data_files, split="train")
    dataset_format = str(dataset_cfg.get("dataset_format") or "auto")

    def _map(row):
        return {"text": _row_to_text(row, tokenizer, dataset_format)}

    remove_columns = list(getattr(loaded, "column_names", []) or [])
    return loaded.map(_map, remove_columns=remove_columns)


def _dataset_files(path: Path):
    if path.is_file():
        return str(path)
    files = sorted(
        str(file)
        for file in path.rglob("*")
        if file.is_file() and file.suffix.lower() in {".json", ".jsonl"}
    )
    if not files:
        raise FileNotFoundError(f"No JSON/JSONL dataset files found in {path}")
    return files


def _row_to_text(row: dict, tokenizer, dataset_format: str) -> str:
    fmt = dataset_format.lower()
    if fmt in {"auto", "messages"} and isinstance(row.get("messages"), list):
        return _messages_to_text(row["messages"], tokenizer)
    if fmt in {"auto", "prompt_completion"}:
        prompt = row.get("prompt") or row.get("instruction") or row.get("input")
        completion = row.get("completion") or row.get("response") or row.get("output")
        if isinstance(prompt, str) and isinstance(completion, str):
            return _prompt_completion_to_text(prompt, completion, tokenizer)
    if fmt in {"auto", "text"} and isinstance(row.get("text"), str):
        return row["text"]
    raise ValueError("Dataset row does not match messages, prompt/completion, or text format.")


def _messages_to_text(messages: list[dict], tokenizer) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)


def _prompt_completion_to_text(prompt: str, completion: str, tokenizer) -> str:
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    return f"User: {prompt}\nAssistant: {completion}"


def _torch_dtype(torch, training_args: dict):
    if bool(training_args.get("bf16")):
        return torch.bfloat16
    if bool(training_args.get("fp16")):
        return torch.float16
    return None


def _build_sft_config(SFTConfig, cfg: dict, output_dir: Path):
    training_args = dict(cfg["training_args"])
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        **training_args,
    }
    signature = inspect.signature(SFTConfig)
    params = signature.parameters
    if "dataset_text_field" in params:
        kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in params:
        kwargs["max_seq_length"] = int(cfg["dataset"]["max_seq_length"])
    elif "max_length" in params:
        kwargs["max_length"] = int(cfg["dataset"]["max_seq_length"])
    if "packing" in params:
        kwargs["packing"] = bool(cfg["dataset"]["packing"])
    return SFTConfig(**{key: value for key, value in kwargs.items() if key in params})


def _add_tokenizer_arg(SFTTrainer, kwargs: dict, tokenizer) -> None:
    params = inspect.signature(SFTTrainer).parameters
    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer


def _make_progress_callback(base_cls, job_id: str, total_steps: int):
    class ProgressCallback(base_cls):
        def __init__(self) -> None:
            self.job_id = job_id
            self.total_steps = max(1, total_steps)

        def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
            logs = logs or {}
            step = int(getattr(state, "global_step", 0) or 0)
            total = int(getattr(state, "max_steps", 0) or self.total_steps)
            if "loss" in logs:
                message = f"Step {step}/{total} loss={logs['loss']}"
            else:
                message = f"Step {step}/{total}"
            emit_progress(self.job_id, step=step, total=max(1, total), message=message)

        def on_save(self, args, state, control, **kwargs):  # noqa: ANN001
            emit_status(self.job_id, f"Checkpoint saved at step {int(getattr(state, 'global_step', 0) or 0)}")

    return ProgressCallback()


if __name__ == "__main__":
    run_worker(main)
