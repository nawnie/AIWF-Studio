"""
aiwf/services/training/ed2_config.py

Pure config builder: ED2TrainingRequest → ED2 train.json dict / file.

No subprocess logic.  No engine imports.  No torch.
Safe to import at boot time and in tests.

ED2 (EveryDream2trainer) reads a JSON config file passed as --config=<path>.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_ed2_config(request) -> dict:
    """Build a train.json dict from an ED2TrainingRequest (or dict).

    Accepts a KohyaLoraRequest Pydantic model or a plain dict so this
    function can be called without importing the domain type.

    Returns
    -------
    dict
        Config dict suitable for json.dumps() and writing to train.json.
    """
    req = _proxy(request)

    job_name:     str  = req("job_name", "ed2_job")
    output_dir:   str  = req("output_dir", "outputs/training/ed2")
    log_dir:      str  = req("log_dir", str(Path(output_dir) / "logs"))

    cfg: dict[str, Any] = {
        "project_name":          job_name,
        "resume_ckpt":           req("base_model_path", ""),
        "data_root":             req("dataset_dir", ""),
        "save_ckpt_dir":         str(Path(output_dir)),
        "logdir":                log_dir,
        "resolution":            req("resolution", 512),
        "flip_p":                req("flip_p", 0.0),
        "max_epochs":            req("max_epochs", 20),
        "batch_size":            req("batch_size", 4),
        "lr":                    req("lr", 1.5e-6),
        "lr_scheduler":          req("lr_scheduler", "constant"),
        "lr_warmup_steps":       req("lr_warmup_steps", 0),
        "optimizer":             req("optimizer", "adamw"),
        "mixed_precision":       req("mixed_precision", "bf16"),
        "gradient_checkpointing": req("gradient_checkpointing", True),
        "clip_skip":             req("clip_skip", 2),
        "seed":                  req("seed", 42),
        "save_every_n_epochs":   req("save_every_n_epochs", 1),
        "sample_steps":          req("sample_steps", 0),
    }

    # Optional fields — only include when set
    vae_path = req("vae_path", "")
    if vae_path:
        cfg["vae"] = str(vae_path)

    sample_prompts = req("sample_prompts", [])
    if cfg["sample_steps"]:
        if isinstance(sample_prompts, str) and sample_prompts:
            cfg["sample_prompts"] = sample_prompts

    return cfg


def write_ed2_config(request, dest: Path) -> Path:
    """Write the train.json to *dest* and return the path.

    Creates parent directories as needed.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(build_ed2_config(request), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return dest


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _Proxy:
    def __init__(self, r) -> None:
        self._r = r

    def __call__(self, key: str, default=None):
        if isinstance(self._r, dict):
            return self._r.get(key, default)
        return getattr(self._r, key, default)


def _proxy(r) -> "_Proxy":
    return _Proxy(r)
