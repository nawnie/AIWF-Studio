"""
aiwf/services/training/dataset_validator.py

Preflight checks for training datasets and job parameters.

Zero engine dependencies — pure stdlib only.  Safe to import at boot time.

Usage
-----
    from aiwf.services.training.dataset_validator import DatasetValidator

    v = DatasetValidator()
    result = v.validate_kohya(request)
    if not result.ok:
        for err in result.errors:
            print("ERROR:", err)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

# Image extensions we consider valid training images
_IMAGE_EXTS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
)

# Caption extensions we consider valid
_CAPTION_EXTS: frozenset[str] = frozenset({".txt", ".caption"})

_TEXT_DATA_EXTS: frozenset[str] = frozenset({".jsonl", ".json"})


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Outcome of a dataset / job preflight check.

    Attributes
    ----------
    ok:       True iff there are zero errors.
    errors:   Blocking problems — the job must not start.
    warnings: Non-blocking advisories — the job can start but may fail.
    """

    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def passed(cls, warnings: list[str] | None = None) -> "ValidationResult":
        return cls(ok=True, warnings=warnings or [])

    @classmethod
    def failed(cls, errors: list[str], warnings: list[str] | None = None) -> "ValidationResult":
        return cls(ok=False, errors=errors, warnings=warnings or [])

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """Merge another result into this one (in-place) and return self."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.ok = len(self.errors) == 0
        return self


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class DatasetValidator:
    """Validate training requests before spawning a subprocess worker.

    All checks are pure filesystem operations.  No torch, no diffusers,
    no engine imports of any kind.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_kohya(self, request) -> ValidationResult:
        """Validate a KohyaLoraRequest.

        Accepts a KohyaLoraRequest instance or any object/dict with the
        same attribute names so the validator doesn't need to import
        aiwf.core.domain.training at class-load time.
        """
        req = _RequestProxy(request)
        errors: list[str] = []
        warnings: list[str] = []

        # Dataset directory
        errors.extend(_check_dataset_dir(
            req.get("dataset_dir"),
            caption_extension=req.get("caption_extension", ".txt"),
            require_captions=True,
            warnings=warnings,
        ))

        # Output directory writability
        errors.extend(_check_output_dir(req.get("output_dir", "outputs/training/kohya"), warnings))

        # Base model
        errors.extend(_check_base_model(req.get("base_model_path", ""), allow_hf_id=True))

        # Resolution sanity
        res = req.get("resolution", 1024)
        if res % 64 != 0:
            warnings.append(f"Resolution {res} is not a multiple of 64 — Kohya may pad or reject it.")

        # Step count sanity
        steps = req.get("max_train_steps", 1500)
        if steps < 100:
            warnings.append(f"max_train_steps={steps} is very low; training may underfit.")
        if steps > 10_000:
            warnings.append(f"max_train_steps={steps} is very high; consider checkpointing often.")

        ok = len(errors) == 0
        return ValidationResult(ok=ok, errors=errors, warnings=warnings)

    def validate_ed2(self, request) -> ValidationResult:
        """Validate an ED2TrainingRequest."""
        req = _RequestProxy(request)
        errors: list[str] = []
        warnings: list[str] = []

        # Dataset directory (ED2 uses caption .txt files)
        errors.extend(_check_dataset_dir(
            req.get("dataset_dir"),
            caption_extension=".txt",
            require_captions=True,
            warnings=warnings,
        ))

        # Output directory
        errors.extend(_check_output_dir(req.get("output_dir", "outputs/training/ed2"), warnings))

        # Base model (ED2 requires a local path or HF ID)
        errors.extend(_check_base_model(req.get("base_model_path", ""), allow_hf_id=True))

        # Optional VAE path check
        vae_path = req.get("vae_path", "")
        if vae_path:
            vp = Path(vae_path)
            if not vp.exists():
                warnings.append(f"VAE path not found: {vae_path} — ED2 may fall back to model VAE.")

        # Epoch count sanity
        max_epochs = req.get("max_epochs", 20)
        if max_epochs < 1:
            errors.append("max_epochs must be at least 1.")
        if max_epochs > 500:
            warnings.append(f"max_epochs={max_epochs} is very high; training will take a long time.")

        # LR sanity
        lr = req.get("lr", 1.5e-6)
        if lr > 1e-4:
            warnings.append(
                f"lr={lr:.2e} seems high for full fine-tuning. Typical range: 1e-6 to 1e-5."
            )

        ok = len(errors) == 0
        return ValidationResult(ok=ok, errors=errors, warnings=warnings)

    def validate_llm(self, request) -> ValidationResult:
        """Validate an LLMBotTrainingRequest."""
        req = _RequestProxy(request)
        errors: list[str] = []
        warnings: list[str] = []

        errors.extend(_check_llm_dataset(
            req.get("dataset_path"),
            dataset_format=req.get("dataset_format", "auto"),
            warnings=warnings,
        ))
        errors.extend(_check_output_dir(req.get("output_dir", "outputs/training/llm"), warnings))
        errors.extend(_check_base_model(req.get("base_model_path", ""), allow_hf_id=True))

        method = str(req.get("method", "qlora")).lower()
        if method == "full":
            warnings.append(
                "Full fine-tuning updates every model weight. On a 16GB card, keep models small, "
                "batch size at 1, sequence length modest, and gradient checkpointing on."
            )
        elif method == "qlora":
            warnings.append(
                "QLoRA loads the base model in 4-bit and trains adapters. It is usually the safest first run."
            )

        seq_len = int(req.get("max_seq_length", 1024) or 1024)
        if seq_len > 2048:
            warnings.append(
                f"max_seq_length={seq_len} may raise VRAM sharply; start at 1024 or 2048 for local tests."
            )
        if not bool(req.get("gradient_checkpointing", True)):
            warnings.append("Gradient checkpointing is off; VRAM use will be higher.")

        ok = len(errors) == 0
        return ValidationResult(ok=ok, errors=errors, warnings=warnings)

    def validate_dataset_dir(self, dataset_dir: str | Path) -> ValidationResult:
        """Standalone dataset directory check (used by UI Validate button)."""
        errors: list[str] = []
        warnings: list[str] = []
        errors.extend(_check_dataset_dir(str(dataset_dir), warnings=warnings))
        return ValidationResult(ok=len(errors) == 0, errors=errors, warnings=warnings)


# ---------------------------------------------------------------------------
# Internal helpers — pure filesystem, no engine imports
# ---------------------------------------------------------------------------

def _check_dataset_dir(
    dataset_dir: str | None,
    *,
    caption_extension: str = ".txt",
    require_captions: bool = False,
    warnings: list[str],
) -> list[str]:
    errors: list[str] = []

    if not dataset_dir:
        errors.append("dataset_dir is required.")
        return errors

    p = Path(dataset_dir)
    if not p.exists():
        errors.append(f"Dataset directory does not exist: {p}")
        return errors
    if not p.is_dir():
        errors.append(f"dataset_dir is not a directory: {p}")
        return errors

    # Collect all image files (recursively)
    images = [f for f in p.rglob("*") if f.suffix.lower() in _IMAGE_EXTS and f.is_file()]
    if not images:
        errors.append(
            f"No image files found in {p}. "
            f"Supported formats: {', '.join(sorted(_IMAGE_EXTS))}"
        )
        return errors

    # Check captions
    if require_captions:
        missing_captions = [
            img for img in images
            if not (img.with_suffix(caption_extension)).exists()
        ]
        if missing_captions:
            pct = 100 * len(missing_captions) / len(images)
            if pct > 50:
                errors.append(
                    f"{len(missing_captions)}/{len(images)} images are missing "
                    f"{caption_extension} caption files."
                )
            else:
                warnings.append(
                    f"{len(missing_captions)}/{len(images)} images are missing "
                    f"{caption_extension} caption files (non-blocking)."
                )

    # Warn about empty subdirectories
    empty_subdirs = [
        d for d in p.iterdir()
        if d.is_dir() and not any(True for _ in d.iterdir())
    ]
    if empty_subdirs:
        names = ", ".join(d.name for d in empty_subdirs[:3])
        warnings.append(f"Empty subdirectories found: {names}{' …' if len(empty_subdirs) > 3 else ''}")

    return errors


def _check_llm_dataset(
    dataset_path: str | None,
    *,
    dataset_format: str = "auto",
    warnings: list[str],
) -> list[str]:
    errors: list[str] = []
    if not dataset_path:
        errors.append("dataset_path is required.")
        return errors

    p = Path(dataset_path)
    if not p.exists():
        errors.append(f"Dataset path does not exist: {p}")
        return errors

    files: list[Path]
    if p.is_dir():
        files = sorted(
            f for f in p.rglob("*")
            if f.is_file() and f.suffix.lower() in _TEXT_DATA_EXTS
        )
        if not files:
            errors.append(
                f"No JSON/JSONL dataset files found in {p}. "
                f"Supported formats: {', '.join(sorted(_TEXT_DATA_EXTS))}"
            )
            return errors
    elif p.is_file():
        if p.suffix.lower() not in _TEXT_DATA_EXTS:
            errors.append(
                f"Dataset file must be JSON or JSONL, got: {p.name}"
            )
            return errors
        files = [p]
    else:
        errors.append(f"Dataset path is not a file or directory: {p}")
        return errors

    samples: list[dict] = []
    for file in files[:5]:
        try:
            samples.extend(_sample_llm_rows(file, limit=5 - len(samples)))
        except Exception as exc:
            errors.append(f"Could not read {file}: {exc}")
            return errors
        if len(samples) >= 5:
            break

    if not samples:
        errors.append(f"No training rows found in dataset path: {p}")
        return errors

    supported_count = sum(1 for row in samples if _row_matches_format(row, dataset_format))
    if supported_count == 0:
        errors.append(
            "Sampled rows do not match a supported LLM dataset shape. "
            "Use messages, prompt+completion, or text fields."
        )
    elif supported_count < len(samples):
        warnings.append(
            f"{len(samples) - supported_count}/{len(samples)} sampled rows did not match the selected dataset format."
        )

    return errors


def _sample_llm_rows(path: Path, *, limit: int) -> list[dict]:
    if limit <= 0:
        return []
    if path.suffix.lower() == ".jsonl":
        rows: list[dict] = []
        with path.open("r", encoding="utf-8-sig") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
                if len(rows) >= limit:
                    break
        return rows

    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(raw, list):
        return [row for row in raw[:limit] if isinstance(row, dict)]
    if isinstance(raw, dict):
        for key in ("train", "data", "examples", "rows"):
            value = raw.get(key)
            if isinstance(value, list):
                return [row for row in value[:limit] if isinstance(row, dict)]
        return [raw]
    return []


def _row_matches_format(row: dict, dataset_format: str) -> bool:
    fmt = (dataset_format or "auto").lower()
    if fmt in {"auto", "messages"} and _has_messages(row):
        return True
    if fmt in {"auto", "prompt_completion"} and _has_prompt_completion(row):
        return True
    if fmt in {"auto", "text"} and isinstance(row.get("text"), str) and row.get("text", "").strip():
        return True
    return False


def _has_messages(row: dict) -> bool:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return False
    for message in messages:
        if not isinstance(message, dict):
            return False
        if not isinstance(message.get("role"), str) or not isinstance(message.get("content"), str):
            return False
    return True


def _has_prompt_completion(row: dict) -> bool:
    prompt = row.get("prompt") or row.get("instruction") or row.get("input")
    completion = row.get("completion") or row.get("response") or row.get("output")
    return isinstance(prompt, str) and bool(prompt.strip()) and isinstance(completion, str) and bool(completion.strip())


def _check_output_dir(output_dir: str, warnings: list[str]) -> list[str]:
    errors: list[str] = []
    if not output_dir:
        return errors  # will be defaulted elsewhere

    p = Path(output_dir)
    if p.exists():
        if not p.is_dir():
            errors.append(f"output_dir path exists but is not a directory: {p}")
            return errors
        if not os.access(p, os.W_OK):
            errors.append(f"output_dir is not writable: {p}")
    else:
        # Try creating it
        try:
            p.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            errors.append(f"Cannot create output_dir {p}: {exc}")
    return errors


def _check_base_model(base_model_path: str, *, allow_hf_id: bool = True) -> list[str]:
    errors: list[str] = []
    if not base_model_path:
        errors.append("base_model_path is required.")
        return errors

    p = Path(base_model_path)

    # If it looks like a local path (has a separator or .safetensors/.ckpt suffix), check it exists
    # A HuggingFace ID looks like "org/repo" — one slash, no extension, no
    # directory separators, no drive letter.  Everything else is treated as
    # a local filesystem path.
    _hf_id = (
        "/" in base_model_path
        and not base_model_path.startswith("/")
        and not base_model_path.startswith("./")
        and not base_model_path.startswith("../")
        and "\\" not in base_model_path
        and p.suffix.lower() not in {".safetensors", ".ckpt", ".pt", ".bin", ".pth"}
        and base_model_path.count("/") == 1        # exactly "org/repo"
    )

    is_local = not _hf_id and (
        p.suffix.lower() in {".safetensors", ".ckpt", ".pt", ".bin", ".pth"}
        or base_model_path.startswith("/")
        or base_model_path.startswith("./")
        or base_model_path.startswith("../")
        or "\\" in base_model_path
        or os.sep in base_model_path
        or (os.altsep and os.altsep in base_model_path)
    )

    if is_local and not p.exists():
        errors.append(f"Base model not found: {p}")
    elif not is_local and not allow_hf_id:
        errors.append(
            f"base_model_path {base_model_path!r} does not look like a local file. "
            "Provide an absolute path."
        )

    return errors


# ---------------------------------------------------------------------------
# Internal proxy -- lets us accept dict, Pydantic model, or plain object
# ---------------------------------------------------------------------------

class _RequestProxy:
    """Thin wrapper to access request fields without importing the domain type."""

    def __init__(self, request) -> None:
        self._req = request

    def get(self, key: str, default=None):
        if isinstance(self._req, dict):
            return self._req.get(key, default)
        return getattr(self._req, key, default)
