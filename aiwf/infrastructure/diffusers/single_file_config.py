"""Required local config files for supported single-file Diffusers pipelines."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


SINGLE_FILE_CONFIG_REPOSITORIES = {
    "sd15": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "sd15_inpaint": "stable-diffusion-v1-5/stable-diffusion-inpainting",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "sdxl_inpaint": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    "sdxl_refiner": "stabilityai/stable-diffusion-xl-refiner-1.0",
    "sd35": "stabilityai/stable-diffusion-3.5-medium",
}

_SD15_FILES = (
    "model_index.json", "scheduler/scheduler_config.json", "text_encoder/config.json",
    "tokenizer/tokenizer_config.json", "tokenizer/vocab.json", "tokenizer/merges.txt",
    "unet/config.json", "vae/config.json",
)
_SDXL_FILES = (
    "model_index.json", "scheduler/scheduler_config.json", "text_encoder/config.json",
    "text_encoder_2/config.json", "tokenizer/tokenizer_config.json", "tokenizer/vocab.json",
    "tokenizer/merges.txt", "tokenizer_2/tokenizer_config.json", "tokenizer_2/vocab.json",
    "tokenizer_2/merges.txt", "unet/config.json", "vae/config.json",
)
_SDXL_REFINER_FILES = (
    "model_index.json", "scheduler/scheduler_config.json", "text_encoder_2/config.json",
    "tokenizer_2/tokenizer_config.json", "tokenizer_2/vocab.json", "tokenizer_2/merges.txt",
    "unet/config.json", "vae/config.json",
)
_SD35_FILES = (
    "model_index.json", "scheduler/scheduler_config.json", "text_encoder/config.json",
    "text_encoder_2/config.json", "text_encoder_3/config.json", "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json", "tokenizer/merges.txt", "tokenizer_2/tokenizer_config.json",
    "tokenizer_2/vocab.json", "tokenizer_2/merges.txt", "tokenizer_3/tokenizer_config.json",
    "tokenizer_3/spiece.model", "transformer/config.json", "vae/config.json",
)

SINGLE_FILE_CONFIG_REQUIRED_FILES = {
    "sd15": _SD15_FILES,
    "sd15_inpaint": _SD15_FILES,
    "sdxl": _SDXL_FILES,
    "sdxl_inpaint": _SDXL_FILES,
    "sdxl_refiner": _SDXL_REFINER_FILES,
    "sd35": _SD35_FILES,
}

_EXPECTED_PIPELINE_PREFIX = {
    "sd15": "StableDiffusionPipeline",
    "sd15_inpaint": "StableDiffusionInpaintPipeline",
    "sdxl": "StableDiffusionXLPipeline",
    "sdxl_inpaint": "StableDiffusionXLInpaintPipeline",
    "sdxl_refiner": "StableDiffusionXL",
    "sd35": "StableDiffusion3Pipeline",
}
_REQUIRED_COMPONENTS = {
    "sd15": ("scheduler", "text_encoder", "tokenizer", "unet", "vae"),
    "sd15_inpaint": ("scheduler", "text_encoder", "tokenizer", "unet", "vae"),
    "sdxl": ("scheduler", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "unet", "vae"),
    "sdxl_inpaint": ("scheduler", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "unet", "vae"),
    "sdxl_refiner": ("scheduler", "text_encoder_2", "tokenizer_2", "unet", "vae"),
    "sd35": ("scheduler", "text_encoder", "text_encoder_2", "text_encoder_3", "tokenizer", "tokenizer_2", "tokenizer_3", "transformer", "vae"),
}


def _snapshot_signature(path: Path, family: str) -> tuple[tuple[str, int, int], ...] | None:
    signature: list[tuple[str, int, int]] = []
    for relative in SINGLE_FILE_CONFIG_REQUIRED_FILES[family]:
        candidate = path / relative
        try:
            stat = candidate.stat()
            if not candidate.is_file() or stat.st_size <= 0:
                return None
            signature.append((relative, stat.st_mtime_ns, stat.st_size))
        except OSError:
            return None
    return tuple(signature)


@lru_cache(maxsize=32)
def _snapshot_json_is_compatible(
    root: str,
    family: str,
    signature: tuple[tuple[str, int, int], ...],
) -> bool:
    path = Path(root)
    parsed: dict[str, object] = {}
    try:
        for relative, _, _ in signature:
            if not relative.endswith(".json"):
                continue
            value = json.loads((path / relative).read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                return False
            parsed[relative] = value
    except (OSError, UnicodeError, ValueError):
        return False

    model_index = parsed.get("model_index.json")
    if not isinstance(model_index, dict):
        return False
    class_name = model_index.get("_class_name")
    expected_prefix = _EXPECTED_PIPELINE_PREFIX[family]
    if not isinstance(class_name, str) or not class_name.startswith(expected_prefix):
        return False
    for component in _REQUIRED_COMPONENTS[family]:
        declaration = model_index.get(component)
        if not isinstance(declaration, list) or len(declaration) < 2 or not all(declaration[:2]):
            return False
        component_config = parsed.get(f"{component}/config.json")
        if component in {"text_encoder", "text_encoder_2", "text_encoder_3", "unet", "transformer", "vae"}:
            if not isinstance(component_config, dict):
                return False
    return True


def missing_single_file_config_files(path: Path, family: str) -> list[Path]:
    """Return missing or invalid config files required by a single-file pipeline."""
    required = SINGLE_FILE_CONFIG_REQUIRED_FILES.get(family)
    if required is None:
        return [Path(path) / "model_index.json"]
    root = Path(path)
    missing: list[Path] = []
    for relative in required:
        candidate = root / relative
        try:
            if not candidate.is_file() or candidate.stat().st_size <= 0:
                missing.append(candidate)
        except OSError:
            missing.append(candidate)
    if not missing:
        signature = _snapshot_signature(root, family)
        try:
            resolved_root = str(root.resolve(strict=True))
        except (OSError, RuntimeError):
            return [root / "model_index.json"]
        if signature is None or not _snapshot_json_is_compatible(resolved_root, family, signature):
            missing.append(root / "model_index.json")
    return missing
