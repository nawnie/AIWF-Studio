from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

from aiwf.core.config.settings import RuntimeFlags
from aiwf.core.domain.models import Checkpoint
from aiwf.infrastructure.diffusers.model_arch import (
    architecture_label,
    detect_checkpoint_architecture,
    is_inpaint_architecture,
    looks_like_lora_weights,
)
from aiwf.infrastructure.model_inventory import ModelInventoryRecord, scan_and_write_model_inventory

logger = logging.getLogger(__name__)

CHECKPOINT_EXTENSIONS = {".ckpt", ".safetensors", ".pt"}
VAE_SUFFIXES = (".vae.safetensors", ".vae.ckpt", ".vae.pt")
SKIP_DIR_NAMES = {
    "vae",
    "vae-approx",
    "lora",
    "loras",
    "embeddings",
    "hypernetworks",
    "codeformer",
    "gfpgan",
    "esrgan",
    "realesrgan",
    "deepbooru",
    "diffusion_models",
    "karlo",
    "controlnet",
    "clip",
    "clip_vision",
    "sam",
    "textencoder",
    "text_encoder",
    "text-encoder",
    "wan",
}


def resolve_search_roots(flags: RuntimeFlags) -> list[Path]:
    """All directories scanned for checkpoints, in priority order.

    This is the local model discovery boundary for image generation. Keep it
    rooted in configured AIWF/model directories; broad drive scans belong in a
    user-triggered import/index task, not in normal app startup.
    """
    models_dir = flags.resolved_models_dir()
    ckpt_dir = flags.resolved_ckpt_dir()
    roots: list[Path] = []

    seen_keys: set[str] = set()

    def add_root(path: Path) -> None:
        resolved = path.resolve()
        if resolved.name.lower() in SKIP_DIR_NAMES:
            return
        key = os.path.normcase(str(resolved))
        if resolved.exists() and key not in seen_keys:
            seen_keys.add(key)
            roots.append(resolved)

    for candidate in (ckpt_dir, models_dir / "Stable-diffusion", models_dir / "stable-diffusion", models_dir):
        add_root(candidate)
    for extra_ckpt_dir in flags.resolved_extra_ckpt_dirs():
        add_root(extra_ckpt_dir)
    for extra_models_dir in flags.resolved_extra_model_dirs():
        for candidate in (
            extra_models_dir / "Stable-diffusion",
            extra_models_dir / "stable-diffusion",
            extra_models_dir,
        ):
            add_root(candidate)

    return roots


def _fast_fingerprint(path: Path) -> str | None:
    """Quick id for UI labels — avoids reading multi-GB files on every scan."""
    try:
        stat = path.stat()
        digest = hashlib.sha256()
        digest.update(str(stat.st_size).encode())
        digest.update(str(int(stat.st_mtime)).encode())
        with path.open("rb") as handle:
            digest.update(handle.read(1024 * 1024))
            if stat.st_size > 1024 * 1024:
                handle.seek(-1024 * 1024, 2)
                digest.update(handle.read(1024 * 1024))
        return digest.hexdigest()[:10]
    except OSError:
        return None


def _is_checkpoint_file(path: Path) -> bool:
    if not path.is_file():
        return False
    lower_name = path.name.lower()
    if path.suffix.lower() not in CHECKPOINT_EXTENSIONS:
        return False
    return not any(lower_name.endswith(suffix) for suffix in VAE_SUFFIXES)


def _iter_checkpoint_files(root: Path) -> list[Path]:
    found: list[Path] = []

    def walk(directory: Path) -> None:
        try:
            entries = sorted(directory.iterdir(), key=lambda p: p.name.lower())
        except OSError:
            return

        for entry in entries:
            if entry.is_file() and _is_checkpoint_file(entry):
                found.append(entry)
                continue
            if not entry.is_dir():
                continue
            # Avoid descending into sibling asset families whose files often
            # share .safetensors/.pt suffixes but are not selectable checkpoints.
            if entry.name.lower() in SKIP_DIR_NAMES:
                continue
            walk(entry)

    walk(root)
    return found


def scan_checkpoints(roots: list[Path]) -> list[Checkpoint]:
    seen_paths: set[str] = set()
    results: list[Checkpoint] = []

    for root in roots:
        if not root.exists():
            logger.debug("Checkpoint root does not exist: %s", root)
            continue

        for path in _iter_checkpoint_files(root):
            resolved = str(path.resolve())
            dedup_key = os.path.normcase(resolved)
            if dedup_key in seen_paths:
                continue
            if looks_like_lora_weights(path):
                logger.debug("Skipping LoRA weights in checkpoint scan: %s", path)
                continue
            seen_paths.add(dedup_key)

            short_hash = _fast_fingerprint(path)
            checkpoint_id = path.stem
            architecture = detect_checkpoint_architecture(path)
            arch_tag = architecture_label(architecture)
            title = f"{checkpoint_id} [{arch_tag}]"
            if short_hash:
                title = f"{title} [{short_hash}]"
            results.append(
                Checkpoint(
                    id=checkpoint_id,
                    title=title,
                    filename=path.name,
                    path=resolved,
                    hash=short_hash,
                    kind="inpaint" if is_inpaint_architecture(architecture) else "checkpoint",
                    architecture=architecture,
                )
            )

    def sort_key(item: Checkpoint) -> tuple[int, str]:
        name = item.filename.lower()
        is_inpaint = "inpaint" in name
        return (1 if is_inpaint else 0, item.title.lower())

    results.sort(key=sort_key)
    return results


def scan_from_flags(flags: RuntimeFlags) -> list[Checkpoint]:
    roots = resolve_search_roots(flags)
    logger.info("Scanning for checkpoints in: %s", ", ".join(str(r) for r in roots))
    inventory = scan_and_write_model_inventory(flags)
    checkpoints = [
        _checkpoint_from_inventory(record)
        for record in inventory
        if record.family == "checkpoint"
    ]
    checkpoints.sort(key=lambda item: (1 if item.kind == "inpaint" else 0, item.title.lower()))

    logger.info("Found %d checkpoint(s)", len(checkpoints))
    if not checkpoints:
        logger.warning(
            "No checkpoints found. Place .safetensors or .ckpt files in: %s",
            flags.resolved_ckpt_dir(),
        )
    return checkpoints


def _checkpoint_from_inventory(record: ModelInventoryRecord) -> Checkpoint:
    path = Path(record.path)
    architecture = record.architecture or detect_checkpoint_architecture(path)
    short_hash = _fast_fingerprint(path)
    checkpoint_id = path.stem
    title = f"{checkpoint_id} [{architecture_label(architecture)}]"
    if short_hash:
        title = f"{title} [{short_hash}]"
    return Checkpoint(
        id=checkpoint_id,
        title=title,
        filename=path.name,
        path=str(path.resolve()),
        hash=short_hash,
        kind="inpaint" if is_inpaint_architecture(architecture) else "checkpoint",
        architecture=architecture,
    )
