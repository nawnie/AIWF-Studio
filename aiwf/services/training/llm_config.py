"""Pure config builder for AI bot text-model training.

No torch, transformers, datasets, peft, or trl imports live here. This module
is safe for app startup, tests, and UI previews.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_llm_training_config(request) -> dict[str, Any]:
    """Build a serialisable training config from an LLM request or dict."""
    req = _proxy(request)
    method = str(req("method", "qlora")).lower()
    optimizer = str(req("optimizer", "") or _default_optimizer(method))
    output_dir = Path(str(req("output_dir", "outputs/training/llm"))) / str(req("job_name", "llm_job"))

    cfg: dict[str, Any] = {
        "job": {
            "job_name": req("job_name", "llm_job"),
            "method": method,
            "output_dir": str(output_dir),
            "seed": int(req("seed", 42)),
        },
        "model": {
            "base_model_path": req("base_model_path", ""),
            "trust_remote_code": bool(req("trust_remote_code", True)),
            "local_files_only": bool(req("local_files_only", True)),
        },
        "dataset": {
            "dataset_path": req("dataset_path", ""),
            "dataset_format": req("dataset_format", "auto"),
            "max_seq_length": int(req("max_seq_length", 1024)),
            "packing": bool(req("packing", False)),
        },
        "training_args": {
            "max_steps": int(req("max_steps", 100)),
            "num_train_epochs": float(req("num_train_epochs", 1.0)),
            "per_device_train_batch_size": int(req("batch_size", 1)),
            "gradient_accumulation_steps": int(req("gradient_accumulation_steps", 8)),
            "learning_rate": float(req("learning_rate", 2e-5)),
            "optim": optimizer,
            "logging_steps": int(req("logging_steps", 10)),
            "save_steps": int(req("save_steps", 100)),
            "save_total_limit": int(req("save_total_limit", 2)),
            "gradient_checkpointing": bool(req("gradient_checkpointing", True)),
            "bf16": str(req("mixed_precision", "bf16")) == "bf16",
            "fp16": str(req("mixed_precision", "bf16")) == "fp16",
            "report_to": "none",
        },
        "peft": {
            "enabled": method in {"lora", "qlora"},
            "r": int(req("lora_rank", 16)),
            "lora_alpha": float(req("lora_alpha", 32.0)),
            "lora_dropout": float(req("lora_dropout", 0.05)),
            "target_modules": req("target_modules", "all-linear"),
        },
        "quantization": {
            "enabled": method == "qlora",
            "load_in_4bit": method == "qlora",
            "bnb_4bit_quant_type": req("bnb_4bit_quant_type", "nf4"),
            "bnb_4bit_use_double_quant": bool(req("bnb_4bit_use_double_quant", True)),
        },
    }
    return cfg


def write_llm_training_config(request, dest: Path) -> Path:
    """Write the built config to *dest* and return the path."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(build_llm_training_config(request), indent=2), encoding="utf-8")
    return dest


def _default_optimizer(method: str) -> str:
    return {
        "lora": "adamw_bnb_8bit",
        "qlora": "paged_adamw_8bit",
        "full": "adamw_torch",
    }.get(method, "paged_adamw_8bit")


class _Proxy:
    def __init__(self, request) -> None:
        self._request = request

    def __call__(self, key: str, default=None):
        if isinstance(self._request, dict):
            return self._request.get(key, default)
        return getattr(self._request, key, default)


def _proxy(request) -> _Proxy:
    return _Proxy(request)
