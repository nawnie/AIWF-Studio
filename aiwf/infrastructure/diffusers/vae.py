from __future__ import annotations

import logging
from pathlib import Path

from aiwf.core.config.settings import RuntimeFlags
from aiwf.core.domain.models import VaeInfo
from aiwf.infrastructure.model_asset_summary import (
    asset_file_count,
    asset_shape_label,
    asset_size_bytes,
    precision_from_text,
    safetensors_precision,
)

logger = logging.getLogger(__name__)

VAE_EXTENSIONS = {".safetensors", ".pt", ".ckpt"}


def resolve_vae_roots(flags: RuntimeFlags) -> list[Path]:
    """Only dedicated VAE folders - scanning the models root would list
    checkpoints with "VAE" in their filename as VAEs."""
    import os

    roots: list[Path] = []
    seen: set[str] = set()
    model_roots = [flags.resolved_models_dir(), *flags.resolved_extra_model_dirs()]
    for models_dir in model_roots:
        candidates = []
        if models_dir.name.lower() in {"vae", "vae-approx"}:
            candidates.append(models_dir)
        candidates.extend(models_dir / name for name in ("VAE", "vae", "vae-approx"))
        for candidate in candidates:
            resolved = candidate.resolve()
            key = os.path.normcase(str(resolved))
            if resolved.exists() and key not in seen:
                seen.add(key)
                roots.append(resolved)
    return roots


def scan_vaes(flags: RuntimeFlags) -> list[VaeInfo]:
    seen: set[str] = set()
    results: list[VaeInfo] = []

    for root in resolve_vae_roots(flags):
        try:
            paths = sorted(root.rglob("*"))
        except OSError:
            continue
        for path in paths:
            if not path.is_file() or path.suffix.lower() not in VAE_EXTENSIONS:
                continue
            if root.name.lower() not in {"vae", "vae-approx", "lora", "loras"} and path.parent == root:
                continue
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            vae_id = path.stem
            size_bytes = asset_size_bytes(path)
            file_count = asset_file_count(path)
            summary = asset_shape_label(path, size_bytes=size_bytes, file_count=file_count)
            precision = safetensors_precision(path) or precision_from_text(path.name)
            precision_suffix = f", {precision}" if precision else ""
            results.append(
                VaeInfo(
                    id=vae_id,
                    title=f"{vae_id} [{summary}{precision_suffix}]",
                    filename=path.name,
                    path=resolved,
                    size_bytes=size_bytes,
                    file_count=file_count,
                    asset_summary=summary,
                    precision=precision,
                )
            )

    results.sort(key=lambda item: item.title.lower())
    logger.info("Found %d VAE(s)", len(results))
    return results


def resolve_vae(vaes: list[VaeInfo], vae_id: str | None) -> VaeInfo | None:
    if not vae_id:
        return None
    lowered = vae_id.lower()
    for vae in vaes:
        if vae.id.lower() == lowered or vae.title.lower() == lowered or vae.filename.lower() == lowered:
            return vae
    return None
