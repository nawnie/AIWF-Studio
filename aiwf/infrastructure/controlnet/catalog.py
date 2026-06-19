from __future__ import annotations

import os
from pathlib import Path

from aiwf.core.config.settings import RuntimeFlags

CONTROLNET_EXTENSIONS = {".safetensors", ".ckpt", ".pt", ".pth", ".bin"}
CONTROLNET_SKIP_DIR_NAMES = {
    "annotator",
    "annotators",
    "image",
    "images",
    "preprocessor",
    "preprocessors",
}

_UNSUPPORTED_FILE_MARKERS = (
    "ip-adapter",
    "ip_adapter",
    "t2i-adapter",
    "t2i_adapter",
    "controllllite",
)


def resolve_controlnet_roots(flags: RuntimeFlags) -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()
    model_roots = [flags.resolved_models_dir(), *flags.resolved_extra_model_dirs()]

    def add(path: Path) -> None:
        resolved = path.resolve()
        key = os.path.normcase(str(resolved))
        if resolved.exists() and key not in seen:
            seen.add(key)
            roots.append(resolved)

    for models_dir in model_roots:
        if models_dir.name.lower() == "controlnet":
            add(models_dir)
        for candidate in (models_dir / "ControlNet", models_dir / "controlnet", models_dir / "control_net"):
            add(candidate)

    return roots


def is_controlnet_auxiliary_path(path: Path) -> bool:
    return any(part.lower() in CONTROLNET_SKIP_DIR_NAMES for part in path.parts)


def is_diffusers_controlnet_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / "config.json").is_file():
        return False
    return any(path.glob("diffusion_pytorch_model*.safetensors")) or any(path.glob("diffusion_pytorch_model*.bin"))


def is_supported_controlnet_file(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() not in CONTROLNET_EXTENSIONS:
        return False
    if is_controlnet_auxiliary_path(path):
        return False
    name = path.name.lower()
    if any(marker in name for marker in _UNSUPPORTED_FILE_MARKERS):
        return False
    if "lora" in name and "control_lora" not in name:
        return False
    if is_diffusers_controlnet_dir(path.parent):
        return False
    return True


def iter_controlnet_model_paths(flags: RuntimeFlags) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        resolved = path.resolve()
        key = os.path.normcase(str(resolved))
        if key in seen:
            return
        seen.add(key)
        paths.append(path)

    for root in resolve_controlnet_roots(flags):
        if is_diffusers_controlnet_dir(root):
            add(root)
        for path in root.rglob("*"):
            if is_controlnet_auxiliary_path(path):
                continue
            if is_diffusers_controlnet_dir(path):
                add(path)
                continue
            if is_supported_controlnet_file(path):
                add(path)

    return sorted(paths, key=lambda item: str(item).lower())
