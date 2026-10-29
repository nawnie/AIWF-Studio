from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImportedModelPaths:
    source: str
    root: Path
    extra_model_dirs: list[Path]
    extra_ckpt_dirs: list[Path]
    summary: str


def _unique_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    results: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        key = str(resolved).lower()
        if key in seen or not resolved.exists():
            continue
        seen.add(key)
        results.append(resolved)
    return results


def _path_lines(value: str) -> list[Path]:
    return [Path(line.strip()).resolve() for line in (value or "").splitlines() if line.strip()]


def _merge_path_text(existing: str, additions: list[Path]) -> str:
    merged = _unique_paths([*_path_lines(existing), *additions])
    return "\n".join(str(path) for path in merged)


def _resolve_models_root(root: Path) -> Path:
    if root.name.lower() == "models":
        return root
    models_root = root / "models"
    return models_root if models_root.exists() else root


def _resolve_a1111_root(root: Path) -> tuple[Path, Path]:
    resolved = root.resolve()
    if resolved.name.lower() == "models":
        return resolved.parent, resolved
    return resolved, _resolve_models_root(resolved)


def _resolve_comfy_root(root: Path) -> tuple[Path, Path]:
    resolved = root.resolve()
    if resolved.name.lower() == "models":
        return resolved.parent, resolved
    return resolved, _resolve_models_root(resolved)


def import_automatic1111_paths(root: str) -> ImportedModelPaths:
    source_root, models_root = _resolve_a1111_root(Path(root).expanduser())
    if not source_root.exists():
        raise ValueError(f"AUTOMATIC1111 folder not found: {source_root}")
    if not models_root.exists():
        raise ValueError(f"No models folder found under: {source_root}")

    extra_model_dirs = _unique_paths(
        [
            models_root,
            source_root / "embeddings",
        ]
    )
    extra_ckpt_dirs = _unique_paths([models_root / "Stable-diffusion"])
    return ImportedModelPaths(
        source="AUTOMATIC1111",
        root=source_root,
        extra_model_dirs=extra_model_dirs,
        extra_ckpt_dirs=extra_ckpt_dirs,
        summary=(
            f"Imported `{source_root}`. "
            "This links its models library and embeddings folder for the next restart."
        ),
    )


def import_comfyui_paths(root: str) -> ImportedModelPaths:
    source_root, models_root = _resolve_comfy_root(Path(root).expanduser())
    if not source_root.exists():
        raise ValueError(f"ComfyUI folder not found: {source_root}")
    if not models_root.exists():
        raise ValueError(f"No models folder found under: {source_root}")

    extra_model_dirs = _unique_paths([models_root])
    extra_ckpt_dirs = _unique_paths([models_root / "checkpoints"])
    return ImportedModelPaths(
        source="ComfyUI",
        root=source_root,
        extra_model_dirs=extra_model_dirs,
        extra_ckpt_dirs=extra_ckpt_dirs,
        summary=(
            f"Imported `{source_root}`. "
            "This links its shared models library for the next restart."
        ),
    )


def merge_imported_path_text(
    existing_model_dirs: str,
    existing_ckpt_dirs: str,
    imported: ImportedModelPaths,
) -> tuple[str, str]:
    return (
        _merge_path_text(existing_model_dirs, imported.extra_model_dirs),
        _merge_path_text(existing_ckpt_dirs, imported.extra_ckpt_dirs),
    )
