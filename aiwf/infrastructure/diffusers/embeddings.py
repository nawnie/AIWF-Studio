from __future__ import annotations

import logging
import os
from pathlib import Path

from aiwf.core.config.settings import RuntimeFlags
from aiwf.core.domain.models import EmbeddingInfo

logger = logging.getLogger(__name__)

EMBEDDING_EXTENSIONS = {".pt", ".safetensors", ".bin"}


def resolve_embedding_roots(flags: RuntimeFlags) -> list[Path]:
    """Only the dedicated embeddings folder (subfolders included)."""
    models_dir = flags.resolved_models_dir()
    roots: list[Path] = []
    seen: set[str] = set()
    for candidate in (models_dir / "embeddings", models_dir / "Embeddings"):
        resolved = candidate.resolve()
        key = os.path.normcase(str(resolved))
        if resolved.exists() and key not in seen:
            seen.add(key)
            roots.append(resolved)
    return roots


def scan_embeddings(flags: RuntimeFlags) -> list[EmbeddingInfo]:
    seen: set[str] = set()
    results: list[EmbeddingInfo] = []

    for root in resolve_embedding_roots(flags):
        try:
            paths = sorted(root.rglob("*"))
        except OSError:
            continue
        for path in paths:
            if not path.is_file() or path.suffix.lower() not in EMBEDDING_EXTENSIONS:
                continue
            resolved = str(path.resolve())
            dedup_key = os.path.normcase(resolved)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            results.append(
                EmbeddingInfo(
                    id=path.stem,
                    title=path.stem,
                    filename=path.name,
                    path=resolved,
                )
            )

    results.sort(key=lambda item: item.title.lower())
    logger.info("Found %d embedding(s)", len(results))
    return results
