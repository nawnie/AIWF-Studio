"""Small, fail-closed helpers for installed model-file layouts."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Callable, Iterable


def configured_model_roots(flags) -> list[Path]:  # noqa: ANN001
    """Return the primary and explicitly configured shared model roots once each."""
    roots = [flags.resolved_models_dir(), *flags.resolved_extra_model_dirs()]
    result: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            resolved = Path(root).resolve()
        except (OSError, RuntimeError):
            continue
        key = str(resolved).casefold()
        if key not in seen:
            seen.add(key)
            result.append(resolved)
    return result


def resolve_model_asset(
    flags,
    relative_candidates: Iterable[str | Path],
    *,
    predicate: Callable[[Path], bool] | None = None,
    fallback: Path | None = None,
) -> Path:  # noqa: ANN001
    """Resolve an installed family asset from primary then configured shared roots.

    Candidate layouts are supplied by the route, keeping family and variant
    requirements explicit. Existing candidates must remain inside their
    configured root; linked paths escaping that root are ignored.
    """
    check = predicate or _nonempty_file
    candidates = tuple(Path(value) for value in relative_candidates)
    for root in configured_model_roots(flags):
        for relative in candidates:
            if relative.is_absolute() or ".." in relative.parts:
                continue
            candidate = root / relative
            try:
                resolved = candidate.resolve(strict=False)
                resolved.relative_to(root)
                if check(resolved):
                    return resolved
            except (OSError, RuntimeError, ValueError):
                continue
    if fallback is not None:
        return Path(fallback)
    primary = flags.resolved_models_dir()
    return primary / (candidates[0] if candidates else Path())


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def indexed_safetensors_shards_ready(component_dir: Path, index_path: Path) -> bool:
    """Confirm every indexed shard is a nonempty file confined to its component.

    Hugging Face indexes are local data and can be incomplete or malformed. The
    resolver must not follow absolute, traversal, drive, UNC, or escaping symlink
    paths when deciding whether a model is installed.
    """
    return indexed_weight_shards_ready(component_dir, index_path, allowed_extensions={".safetensors"})


def indexed_weight_shards_ready(
    component_dir: Path,
    index_path: Path,
    *,
    allowed_extensions: set[str] | frozenset[str] = frozenset({".safetensors", ".bin", ".pt", ".onnx"}),
) -> bool:
    """Validate an indexed component's weight shards without escaping its folder."""
    return not indexed_weight_shard_issues(
        component_dir, index_path, allowed_extensions=allowed_extensions,
    )


def indexed_weight_shard_issues(
    component_dir: Path,
    index_path: Path,
    *,
    allowed_extensions: set[str] | frozenset[str] = frozenset({".safetensors", ".bin", ".pt", ".onnx"}),
) -> list[Path]:
    """Return missing shard paths or the index path for malformed/unsafe indexes."""
    try:
        root = Path(component_dir).resolve()
        index = Path(index_path).resolve(strict=True)
        index.relative_to(root)
        if not index.is_file() or index.stat().st_size <= 0:
            return [Path(index_path)]
        payload = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, ValueError, RuntimeError):
        return [Path(index_path)]
    weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        return [Path(index_path)]
    if not all(isinstance(key, str) and key.strip() for key in weight_map):
        return [Path(index_path)]
    raw_shards = list(weight_map.values())
    if not raw_shards or not all(isinstance(name, str) and name.strip() for name in raw_shards):
        return [Path(index_path)]
    shards = set(raw_shards)
    issues: list[Path] = []
    for name in shards:
        normalized = name.replace("\\", "/")
        posix_path = PurePosixPath(normalized)
        windows_path = PureWindowsPath(name)
        if (
            posix_path.is_absolute()
            or windows_path.is_absolute()
            or windows_path.drive
            or ".." in posix_path.parts
            or not posix_path.parts
            or posix_path.suffix.casefold() not in {suffix.casefold() for suffix in allowed_extensions}
        ):
            return [Path(index_path)]
        try:
            unresolved_candidate = root / Path(*posix_path.parts)
            candidate = unresolved_candidate.resolve(strict=False)
            candidate.relative_to(root)
            if not candidate.is_file() or candidate.stat().st_size <= 0:
                issues.append(unresolved_candidate)
        except (OSError, ValueError, RuntimeError):
            # Do not surface or follow a symlink target outside the component.
            issues.append(Path(index_path))
    return issues


def nonempty_json_object(path: Path) -> bool:
    """Require a nonempty JSON object for Diffusers component metadata."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(payload, dict) and bool(payload)
