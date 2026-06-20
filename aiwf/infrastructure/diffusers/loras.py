from __future__ import annotations

import logging
from pathlib import Path

from aiwf.core.config.settings import RuntimeFlags
from aiwf.core.domain.models import LoraInfo
from aiwf.infrastructure.model_inventory import scan_and_write_model_inventory

logger = logging.getLogger(__name__)

LORA_EXTENSIONS = {".safetensors", ".pt", ".ckpt"}


def resolve_lora_roots(flags: RuntimeFlags) -> list[Path]:
    """Only dedicated LoRA folders - scanning the models root would list
    checkpoints, ControlNets, and VAEs as LoRAs."""
    import os

    roots: list[Path] = []
    seen: set[str] = set()
    model_roots = [flags.resolved_models_dir(), *flags.resolved_extra_model_dirs()]
    for models_dir in model_roots:
        candidates = []
        if models_dir.name.lower() in {"lora", "loras"}:
            candidates.append(models_dir)
        candidates.extend(
            models_dir / name for name in ("Lora", "lora", "Loras", "loras")
        )
        for candidate in candidates:
            resolved = candidate.resolve()
            key = os.path.normcase(str(resolved))
            if resolved.exists() and key not in seen:
                seen.add(key)
                roots.append(resolved)
    return roots


def scan_loras(flags: RuntimeFlags) -> list[LoraInfo]:
    seen: set[str] = set()
    results: list[LoraInfo] = []
    inventory = scan_and_write_model_inventory(flags)

    for record in inventory:
        if record.family != "lora":
            continue
        if record.architecture == "wan":
            continue
        path = Path(record.path)
        if path.suffix.lower() not in LORA_EXTENSIONS:
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
                architecture=record.architecture,
                recommended_subdir=record.recommended_subdir,
                metadata=record.metadata,
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
