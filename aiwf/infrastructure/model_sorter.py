"""Startup auto-sort for the "models to sort" inbox.

Users drop new checkpoints, transformers, LoRAs, VAEs and text encoders into
``models/models to sort``. On startup we read each file's header (no weights
loaded) to classify it, then move confidently-identified files into the folder
the rest of the app already expects them in.

Classification and destination mapping are delegated to
``aiwf.infrastructure.model_inventory`` so the inbox sorter and the model
library always agree on where a given file "belongs".

Anything that cannot be confidently identified is left in place and logged, so
the user can deal with it manually — we never guess-move an ambiguous file and
we never overwrite an existing model.
"""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from aiwf.core.config.settings import RuntimeFlags
from aiwf.infrastructure.model_inventory import (
    MODEL_EXTENSIONS,
    ModelInventoryRecord,
    classify_model_file,
    invalidate_model_inventory_cache,
    model_inventory_roots,
)

logger = logging.getLogger(__name__)

# Folder, relative to the models dir, that users drop new models into.
SORT_INBOX_DIRNAME = "models to sort"

# Destinations we never auto-move into (no confident mapping).
_UNROUTABLE_SUBDIRS = {"", "misc", SORT_INBOX_DIRNAME.lower()}

# header_identifiers signatures that mean "only matched by bare file extension".
# classify_model_file falls back to family=="checkpoint" for any loose
# .safetensors/.ckpt/.pt; we treat that alone as NOT confident so unknown files
# are left in the inbox instead of being dumped into Stable-diffusion.
_WEAK_ONLY_MARKERS = {"fallback_marker"}


@dataclass
class SortAction:
    """One decision the sorter made about one inbox file."""

    filename: str
    source: str
    family: str
    architecture: str
    dest_subdir: str
    status: str  # "moved" | "left" | "conflict" | "error"
    reason: str

    @property
    def moved(self) -> bool:
        return self.status == "moved"


def _is_confident(record: ModelInventoryRecord) -> tuple[bool, str]:
    """Decide whether a classified file is safe to auto-move.

    Confident == identified by a real signal (header arch, tensor markers,
    model_index, etc.), mapped to a specific destination. Bare extension
    fallbacks and unknown/misc destinations are left for the user.
    """
    if record.family == "unknown":
        return False, "header did not identify a known model type"
    if record.recommended_subdir.lower() in _UNROUTABLE_SUBDIRS:
        return False, "no specific destination for this model type"
    markers = set(record.header_identifiers)
    if markers and markers <= _WEAK_ONLY_MARKERS:
        return False, "only matched by file extension, not by header content"
    return True, ""


def plan_inbox_sort(flags: RuntimeFlags) -> list[SortAction]:
    """Classify every file in the inbox without moving anything (dry run)."""
    return _run(flags, apply=False)


def sort_inbox_models(flags: RuntimeFlags) -> list[SortAction]:
    """Move confidently-identified inbox files into their correct folders."""
    return _run(flags, apply=True)


def plan_model_reorganize(flags: RuntimeFlags) -> list[SortAction]:
    """Classify files under the main models directory without moving them."""
    return _run_all_models(flags, apply=False)


def reorganize_models(flags: RuntimeFlags) -> list[SortAction]:
    """Move confidently-identified files under the main models directory."""
    return _run_all_models(flags, apply=True)


def _run(flags: RuntimeFlags, *, apply: bool) -> list[SortAction]:
    models_dir = flags.resolved_models_dir()
    inbox = models_dir / SORT_INBOX_DIRNAME
    if not inbox.is_dir():
        return []

    roots = model_inventory_roots(flags)
    actions: list[SortAction] = []

    for path in sorted(inbox.rglob("*"), key=lambda p: str(p).lower()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in MODEL_EXTENSIONS:
            continue  # skip receipts, .txt placeholders, etc.

        try:
            record = classify_model_file(path, roots)
        except Exception:
            logger.warning("model_sorter: could not classify %s — leaving in place", path.name, exc_info=True)
            actions.append(SortAction(path.name, str(path), "unknown", "unknown", "", "error", "header read failed"))
            continue

        if record is None:
            actions.append(SortAction(path.name, str(path), "unknown", "unknown", "", "left", "unrecognized file"))
            continue

        confident, reason = _is_confident(record)
        dest_subdir = record.recommended_subdir
        if not confident:
            actions.append(
                SortAction(path.name, str(path), record.family, record.architecture, dest_subdir, "left", reason)
            )
            logger.info("model_sorter: leaving %s in inbox (%s)", path.name, reason)
            continue

        dest_dir = models_dir / dest_subdir
        dest = dest_dir / path.name

        if dest.exists():
            actions.append(
                SortAction(
                    path.name, str(path), record.family, record.architecture, dest_subdir,
                    "conflict", f"{dest_subdir}/{path.name} already exists",
                )
            )
            logger.warning(
                "model_sorter: %s already has %s — leaving inbox copy untouched", dest_subdir, path.name
            )
            continue

        if apply:
            try:
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(dest))
            except OSError:
                logger.warning("model_sorter: failed to move %s -> %s", path.name, dest_subdir, exc_info=True)
                actions.append(
                    SortAction(path.name, str(path), record.family, record.architecture, dest_subdir, "error", "move failed")
                )
                continue

        actions.append(
            SortAction(path.name, str(path), record.family, record.architecture, dest_subdir, "moved", "")
        )
        logger.info(
            "model_sorter: %s %s -> %s  [%s / %s]",
            "moved" if apply else "would move",
            path.name,
            dest_subdir,
            record.family,
            record.architecture,
        )

    if apply and any(action.moved for action in actions):
        invalidate_model_inventory_cache()
    return actions


def _run_all_models(flags: RuntimeFlags, *, apply: bool) -> list[SortAction]:
    models_dir = flags.resolved_models_dir().resolve()
    if not models_dir.is_dir():
        return []

    roots = model_inventory_roots(flags)
    actions: list[SortAction] = []

    for path in sorted(models_dir.rglob("*"), key=lambda p: str(p).lower()):
        if not path.is_file() or path.suffix.lower() not in MODEL_EXTENSIONS:
            continue
        try:
            resolved = path.resolve()
            resolved.relative_to(models_dir)
        except (OSError, ValueError):
            continue

        try:
            record = classify_model_file(resolved, roots)
        except Exception:
            logger.warning("model_sorter: could not classify %s - leaving in place", resolved.name, exc_info=True)
            actions.append(SortAction(resolved.name, str(resolved), "unknown", "unknown", "", "error", "header read failed"))
            continue

        if record is None:
            continue
        if not record.should_move:
            continue

        confident, reason = _is_confident(record)
        dest_subdir = record.recommended_subdir
        if not confident:
            actions.append(
                SortAction(resolved.name, str(resolved), record.family, record.architecture, dest_subdir, "left", reason)
            )
            continue

        dest_dir = models_dir / dest_subdir
        dest = dest_dir / resolved.name
        try:
            if dest.resolve() == resolved:
                continue
        except OSError:
            pass

        if dest.exists():
            actions.append(
                SortAction(
                    resolved.name,
                    str(resolved),
                    record.family,
                    record.architecture,
                    dest_subdir,
                    "conflict",
                    f"{dest_subdir}/{resolved.name} already exists",
                )
            )
            continue

        if apply:
            try:
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(resolved), str(dest))
            except OSError:
                logger.warning("model_sorter: failed to move %s -> %s", resolved.name, dest_subdir, exc_info=True)
                actions.append(
                    SortAction(
                        resolved.name,
                        str(resolved),
                        record.family,
                        record.architecture,
                        dest_subdir,
                        "error",
                        "move failed",
                    )
                )
                continue

        actions.append(
            SortAction(resolved.name, str(resolved), record.family, record.architecture, dest_subdir, "moved", "")
        )

    if apply and any(action.moved for action in actions):
        invalidate_model_inventory_cache()
    return actions


def sort_inbox_on_startup(flags: RuntimeFlags) -> list[SortAction]:
    """Entry point called during app startup. Never raises."""
    try:
        actions = sort_inbox_models(flags)
    except Exception:
        logger.warning("model_sorter: inbox sort failed", exc_info=True)
        return []

    moved = [a for a in actions if a.moved]
    left = [a for a in actions if a.status in {"left", "conflict", "error"}]
    if moved:
        logger.info("model_sorter: sorted %d model(s) from '%s' into place", len(moved), SORT_INBOX_DIRNAME)
    if left:
        logger.info(
            "model_sorter: left %d file(s) in '%s' for manual review", len(left), SORT_INBOX_DIRNAME
        )
    return actions
