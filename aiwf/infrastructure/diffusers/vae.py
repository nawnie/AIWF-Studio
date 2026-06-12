from __future__ import annotations

import logging
from pathlib import Path

from aiwf.core.config.settings import RuntimeFlags
from aiwf.core.domain.models import VaeInfo

logger = logging.getLogger(__name__)

VAE_EXTENSIONS = {".safetensors", ".pt", ".ckpt"}


def resolve_vae_roots(flags: RuntimeFlags) -> list[Path]:
    """Only dedicated VAE folders — scanning the models root would list
    checkpoints with "VAE" in their filename as VAEs."""
    import os

    models_dir = flags.resolved_models_dir()
    roots: list[Path] = []
    seen: set[str] = set()
    for candidate in (models_dir / "VAE", models_dir / "vae", models_dir / "vae-approx"):
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
            results.append(
                VaeInfo(
                    id=vae_id,
                    title=vae_id,
                    filename=path.name,
                    path=resolved,
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
