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
