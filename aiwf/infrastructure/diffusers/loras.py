from __future__ import annotations

import logging
from pathlib import Path

from aiwf.core.config.settings import RuntimeFlags
from aiwf.core.domain.models import LoraInfo
from aiwf.infrastructure.diffusers.checkpoints import resolve_search_roots

logger = logging.getLogger(__name__)

LORA_EXTENSIONS = {".safetensors", ".pt", ".ckpt"}


def resolve_lora_roots(flags: RuntimeFlags) -> list[Path]:
    """Only dedicated LoRA folders — scanning the models root would list
    checkpoints, ControlNets, and VAEs as LoRAs."""
    import os

    models_dir = flags.resolved_models_dir()
    roots: list[Path] = []
    seen: set[str] = set()
    for candidate in (models_dir / "Lora", models_dir / "lora", models_dir / "Loras", models_dir / "loras"):
        resolved = candidate.resolve()
        key = os.path.normcase(str(resolved))
        if resolved.exists() and key not in seen:
            seen.add(key)
            roots.append(resolved)
    return roots


def scan_loras(flags: RuntimeFlags) -> list[LoraInfo]:
    seen: set[str] = set()
    results: list[LoraInfo] = []

    for root in resolve_lora_roots(flags):
        try:
            paths = sorted(root.rglob("*"))
        except OSError:
            continue
        for path in paths:
            if not path.is_file() or path.suffix.lower() not in LORA_EXTENSIONS:
                continue
            import os

            resolved = str(path.resolve())
            dedup_key = os.path.normcase(resolved)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            lora_id = path.stem
            results.append(
                LoraInfo(
                    id=lora_id,
                    title=lora_id,
                    filename=path.name,
                    path=resolved,
                )
            )

    results.sort(key=lambda item: item.title.lower())
    logger.info("Found %d LoRA(s)", len(results))
    return results


def resolve_lora(loras: list[LoraInfo], name: str) -> LoraInfo | None:
    lowered = name.lower()
    for lora in loras:
        if lora.id.lower() == lowered or lora.filename.lower() == lowered:
            return lora
        if lora.id.lower().replace(" ", "_") == lowered.replace(" ", "_"):
            return lora
    return None
