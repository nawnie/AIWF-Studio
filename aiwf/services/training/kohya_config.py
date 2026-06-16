"""
aiwf/services/training/kohya_config.py

Pure config builder: KohyaLoraRequest → Kohya TOML config string/file.

No subprocess logic.  No engine imports.  No torch.
Safe to import at boot time and in tests.

The actual TOML is written using a minimal pure-Python serialiser so there
is no dependency on the 'toml', 'tomllib', or 'tomli_w' packages from the
main venv.  The kohya venv has 'toml' available for the worker.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Minimal pure-Python TOML serialiser
# (subset: handles str, int, float, bool, dict/table — sufficient for Kohya)
# ---------------------------------------------------------------------------

def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        # Use repr to avoid precision loss, then strip trailing zeros
        return repr(v)
    if isinstance(v, str):
        # Escape backslashes and double-quotes
        escaped = v.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    raise TypeError(f"_toml_value: unsupported type {type(v).__name__} for {v!r}")


def _toml_section(name: str, entries: dict[str, Any]) -> str:
    lines = [f"[{name}]"]
    for k, v in entries.items():
        lines.append(f"{k} = {_toml_value(v)}")
    return "\n".join(lines)


def _render_toml(sections: dict[str, dict[str, Any]]) -> str:
    parts = []
    for section_name, entries in sections.items():
        parts.append(_toml_section(section_name, entries))
    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------

def build_kohya_toml(request) -> str:
    """Build a Kohya TOML config string from a KohyaLoraRequest (or dict).

    Accepts a KohyaLoraRequest Pydantic model or a plain dict so this
    function can be used without importing the domain type.

    Returns
    -------
    str
        TOML config text suitable for writing to a .toml file.
    """
    req = _proxy(request)

    job_name: str    = req("job_name", "lora_job")
    base_arch: str   = req("base_arch", "sdxl")
    output_dir: str  = req("output_dir", "outputs/training/kohya")
    output_name: str = req("output_name", "") or job_name
    log_dir: str     = str(Path(output_dir) / "logs")

    sections: dict[str, dict[str, Any]] = {
        "model_arguments": {
            "pretrained_model_name_or_path": req("base_model_path", ""),
            "v2": False,
            "v_parameterization": False,
        },
        "dataset_arguments": {
            "train_data_dir":   req("dataset_dir", ""),
            "resolution":       str(req("resolution", 1024)),
            "enable_bucket":    True,
            "caption_extension": req("caption_extension", ".txt"),
            "shuffle_caption":  True,
            "keep_tokens":      1,
        },
        "training_arguments": {
            "output_dir":            str(Path(output_dir)),
            "output_name":           output_name,
            "logging_dir":           log_dir,
            "save_every_n_steps":    req("save_every_n_steps", 500),
            "save_last_n_steps":     req("save_last_n_steps", 5),
            "max_train_steps":       req("max_train_steps", 1500),
            "learning_rate":         req("learning_rate", 1e-4),
            "unet_lr":               _unet_lr(req),
            "text_encoder_lr":       _te_lr(req),
            "lr_scheduler":          req("lr_scheduler", "cosine_with_restarts"),
            "lr_warmup_steps":       req("lr_warmup_steps", 100),
            "optimizer_type":        req("optimizer", "AdamW8bit"),
            "train_batch_size":      req("batch_size", 1),
            "mixed_precision":       req("mixed_precision", "bf16"),
            "save_precision":        req("mixed_precision", "bf16"),
            "gradient_checkpointing": req("gradient_checkpointing", True),
            "clip_grad_norm":        req("clip_grad_norm", 1.0),
            "seed":                  req("seed", 42),
        },
        "network_arguments": {
            "network_module": req("network_module", "networks.lora"),
            "network_dim":    req("network_dim", 32),
            "network_alpha":  float(req("network_alpha", 16.0)),
        },
    }

    return _render_toml(sections)


def write_kohya_toml(request, dest: Path) -> Path:
    """Write the TOML config to *dest* and return the path.

    Creates parent directories as needed.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(build_kohya_toml(request), encoding="utf-8")
    return dest


def training_script_for_arch(arch: str) -> str:
    """Return the relative path to the Kohya training script for *arch*.

    Paths are relative to the kohya_ss repository root.
    """
    return {
        "sd1":  "sd_scripts/train_network.py",
        "sdxl": "sd_scripts/sdxl_train_network.py",
        "flux": "flux_train_network.py",
    }.get(arch, "sd_scripts/sdxl_train_network.py")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _unet_lr(req) -> float:
    v = req("unet_lr", None)
    if v is not None:
        return float(v)
    return float(req("learning_rate", 1e-4))


def _te_lr(req) -> float:
    v = req("text_encoder_lr", None)
    if v is not None:
        return float(v)
    return float(req("learning_rate", 1e-4)) / 2.0


class _Proxy:
    def __init__(self, r) -> None:
        self._r = r

    def __call__(self, key: str, default=None):
        if isinstance(self._r, dict):
            return self._r.get(key, default)
        return getattr(self._r, key, default)


def _proxy(r) -> "_Proxy":
    return _Proxy(r)
