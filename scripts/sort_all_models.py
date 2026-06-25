#!/usr/bin/env python
"""Utility script to sort all models into their correct paths using Safetensors headers.

This script scans the entire models directory, classifies each file using the
existing model_inventory classification rules, and moves confidently-identified
models into their recommended subfolders (e.g., Stable-diffusion, Loras/Flux, etc.).
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

# Add workspace directory to python path
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from aiwf.core.config.settings import RuntimeFlags
from aiwf.infrastructure.model_inventory import (
    get_model_inventory,
    invalidate_model_inventory_cache,
    scan_and_write_model_inventory,
    ModelInventoryRecord,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("sort_all_models")

_UNROUTABLE_SUBDIRS = {"", "misc", "models to sort"}
_WEAK_ONLY_MARKERS = {"fallback_marker"}


def is_confident(record: ModelInventoryRecord) -> tuple[bool, str]:
    """Decide whether a classified file is safe to move.

    Checks if type/family is known, destination is not 'misc' or inbox, and
    classification is not solely based on file extension fallback.
    """
    if record.family == "unknown":
        return False, "header did not identify a known model type"
    if record.recommended_subdir.lower() in _UNROUTABLE_SUBDIRS:
        return False, "no specific destination for this model type"
    markers = set(record.header_identifiers)
    if markers and markers <= _WEAK_ONLY_MARKERS:
        return False, "only matched by file extension, not by header content"
    return True, ""


def clean_empty_directories(root_dir: Path):
    """Recursively delete empty subdirectories bottom-up."""
    logger.info("Cleaning up empty folders...")
    removed_count = 0
    for dirpath, _, _ in os.walk(root_dir, topdown=False):
        path = Path(dirpath)
        if path == root_dir:
            continue
        try:
            # os.rmdir only succeeds if the folder is completely empty
            if not os.listdir(path):
                path.rmdir()
                logger.info(f"Removed empty folder: {path.relative_to(root_dir)}")
                removed_count += 1
        except OSError:
            pass
    logger.info(f"Cleaned up {removed_count} empty folder(s).")


def main():
    parser = argparse.ArgumentParser(description="Sort model files into their correct folders.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without moving files.")
    args = parser.parse_args()

    flags = RuntimeFlags()
    models_dir = flags.resolved_models_dir()
    logger.info(f"Scanning models under: {models_dir}")

    # Force a full rescan to get fresh path positions
    records = get_model_inventory(flags, force_rescan=True)
    to_move = [r for r in records if r.should_move]

    if not to_move:
        logger.info("All models are already in their correct paths.")
        return

    logger.info(f"Found {len(to_move)} model file(s) in non-standard paths.")

    moved_count = 0
    skipped_count = 0
    conflict_count = 0

    for record in to_move:
        src_path = Path(record.path)
        if not src_path.is_file():
            logger.warning(f"File not found on disk: {src_path}")
            continue

        confident, reason = is_confident(record)
        if not confident:
            logger.info(f"Skipping {src_path.name} ({reason})")
            skipped_count += 1
            continue

        dest_dir = models_dir / record.recommended_subdir
        dest_path = dest_dir / src_path.name

        # Check for destination conflicts
        if dest_path.exists():
            # If it's the exact same file (path-wise case difference or similar resolution)
            if dest_path.resolve() == src_path.resolve():
                continue
            logger.warning(f"Conflict: Destination already exists. Skipping: {src_path.name} -> {record.recommended_subdir}")
            conflict_count += 1
            continue

        if args.dry_run:
            logger.info(f"[DRY-RUN] Would move: {src_path.relative_to(models_dir)} -> {record.recommended_subdir}/{src_path.name}")
            moved_count += 1
        else:
            try:
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src_path), str(dest_path))
                logger.info(f"Moved: {src_path.relative_to(models_dir)} -> {record.recommended_subdir}/{src_path.name}")
                moved_count += 1
            except Exception as e:
                logger.error(f"Failed to move {src_path.name}: {e}")
                skipped_count += 1

    if args.dry_run:
        logger.info(f"[DRY-RUN] Summary: Would move {moved_count}, skipped {skipped_count}, conflicts {conflict_count}")
    else:
        logger.info(f"Summary: Successfully moved {moved_count}, skipped {skipped_count}, conflicts {conflict_count}")
        # Clean up empty directories bottom-up
        clean_empty_directories(models_dir)
        # Clear and rebuild cache
        logger.info("Invalidating and rebuilding model inventory cache...")
        invalidate_model_inventory_cache()
        scan_and_write_model_inventory(flags)
        logger.info("Model inventory cache updated successfully.")


if __name__ == "__main__":
    main()
