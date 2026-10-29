"""Startup auto-sort for the "models to sort" inbox.

Users drop new checkpoints, transformers, LoRAs, VAEs, text encoders, and
Diffusers pipeline folders into ``models/models to sort``. On startup we read
headers and model indexes (no weights loaded) to classify them, then move
confidently-identified assets into the folder the rest of the app expects.

Classification and destination mapping are delegated to
``aiwf.infrastructure.model_inventory`` so the inbox sorter and the model
library always agree on where a given file "belongs".

Anything that cannot be confidently identified is left in place and logged, so
the user can deal with it manually — we never guess-move an ambiguous file and
we never overwrite an existing model.
"""
from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from aiwf.core.config.settings import RuntimeFlags
from aiwf.infrastructure.model_inventory import (
    MODEL_EXTENSIONS,
    ModelInventoryRecord,
    classify_model_dir,
    classify_model_file,
    invalidate_model_inventory_cache,
    model_asset_placement_confidence,
    model_inventory_roots,
)
from aiwf.infrastructure.diffusers.checkpoints import diffusers_dir_has_required_local_files

logger = logging.getLogger(__name__)

# Folder, relative to the models dir, that users drop new models into.
SORT_INBOX_DIRNAME = "models to sort"

# Only complete, locally supported pipeline families receive automatic folder
# placement. Model indexes on component folders alone are not enough to infer a
# safe destination.
_DIFFUSERS_PIPELINE_DESTINATIONS = {
    # The Flux Kontext GGUF resolver searches Components/Diffusers locations,
    # not the legacy `flux/Kontext` inbox used by the sorter previously.
    "FluxKontextPipeline": ("flux_kontext", "flux/Components/FLUX.1-Kontext-dev"),
    "Flux2KleinPipeline": ("flux2_klein", "flux2/Components"),
    "Krea2Pipeline": ("krea2", "krea2/Diffusers"),
    "QwenImagePipeline": ("qwen_image", "qwen-image/Diffusers"),
    "QwenImage21Pipeline": ("qwen_image", "qwen-image/Diffusers"),
    "SanaPipeline": ("sana", "sana/Diffusers"),
    "SanaSprintPipeline": ("sana", "sana/Diffusers"),
    "SanaImageToVideoPipeline": ("sana_video", "sana-video/Diffusers"),
    "SanaVideoPipeline": ("sana_video", "sana-video/Diffusers"),
    "StableDiffusionPipeline": ("sd15", "Stable-diffusion"),
    "StableDiffusionInpaintPipeline": ("inpaint", "Stable-diffusion"),
    "StableDiffusion3Pipeline": ("sd35", "Stable-diffusion"),
    "StableDiffusionXLPipeline": ("sdxl", "Stable-diffusion"),
    "StableDiffusionXLInpaintPipeline": ("sdxl", "Stable-diffusion"),
    "WanPipeline": ("wan", "wan/Diffusers"),
    "WanImageToVideoPipeline": ("wan", "wan/Diffusers"),
    "WanVideoPipeline": ("wan", "wan/Diffusers"),
    "ZImagePipeline": ("z_image", "z-image/Components"),
}


@dataclass
class SortAction:
    """One decision the sorter made about one inbox file."""

    filename: str
    source: str
    family: str
    architecture: str
    dest_subdir: str
    status: str  # "moved" | "left" | "conflict" | "error"
    reason: str

    @property
    def moved(self) -> bool:
        return self.status == "moved"


def _is_confident(record: ModelInventoryRecord) -> tuple[bool, str]:
    """Decide whether a classified file is safe to auto-move.

    Confident == identified by a real signal (header arch, tensor markers,
    model_index, etc.), mapped to a specific destination. Bare extension
    fallbacks and unknown/misc destinations are left for the user.
    """
    return model_asset_placement_confidence(record)


def plan_inbox_sort(flags: RuntimeFlags) -> list[SortAction]:
    """Classify inbox files and supported Diffusers folders without moving them."""
    return _run(flags, apply=False)


def sort_inbox_models(flags: RuntimeFlags) -> list[SortAction]:
    """Move confidently-identified files and supported Diffusers folders."""
    return _run(flags, apply=True)


def plan_model_reorganize(flags: RuntimeFlags) -> list[SortAction]:
    """Classify files under the main models directory without moving them."""
    return _run_all_models(flags, apply=False)


def reorganize_models(
    flags: RuntimeFlags,
    *,
    approved_moves: set[tuple[str, str, str, str]] | None = None,
) -> list[SortAction]:
    """Move confidently-identified files, optionally limited to reviewed actions."""
    return _run_all_models(flags, apply=True, approved_moves=approved_moves)


def _run(flags: RuntimeFlags, *, apply: bool) -> list[SortAction]:
    models_dir = flags.resolved_models_dir().resolve()
    inbox = models_dir / SORT_INBOX_DIRNAME
    if inbox.is_symlink():
        return [SortAction(inbox.name, str(inbox), "unknown", "unknown", "", "left", "sort inbox must not be a linked directory")]
    if not inbox.is_dir():
        return []
    try:
        inbox_root = inbox.resolve(strict=True)
        inbox_root.relative_to(models_dir)
    except (OSError, RuntimeError, ValueError):
        return [SortAction(inbox.name, str(inbox), "unknown", "unknown", "", "left", "sort inbox escapes the configured models root")]

    roots = model_inventory_roots(flags)
    snapshot_dirs = _snapshot_dirs(inbox)
    actions: list[SortAction] = []

    for path in sorted(inbox.rglob("*"), key=lambda p: str(p).lower()):
        if path.is_symlink():
            actions.append(SortAction(path.name, str(path), "unknown", "unknown", "", "left", "linked inbox entries are not sorted"))
            continue
        if not path.is_file():
            continue
        try:
            source = path.resolve(strict=True)
            source.relative_to(inbox_root)
            source.relative_to(models_dir)
        except (OSError, RuntimeError, ValueError):
            actions.append(SortAction(path.name, str(path), "unknown", "unknown", "", "left", "source escapes the configured models inbox"))
            continue
        if _is_inside_any(path, snapshot_dirs):
            continue
        if path.suffix.lower() not in MODEL_EXTENSIONS:
            continue  # skip receipts, .txt placeholders, etc.

        try:
            record = classify_model_file(source, roots)
        except Exception:
            logger.warning("model_sorter: could not classify %s — leaving in place", path.name, exc_info=True)
            actions.append(SortAction(path.name, str(path), "unknown", "unknown", "", "error", "header read failed"))
            continue

        if record is None:
            actions.append(SortAction(path.name, str(path), "unknown", "unknown", "", "left", "unrecognized file"))
            continue

        confident, reason = _is_confident(record)
        dest_subdir = record.recommended_subdir
        if not confident:
            actions.append(
                SortAction(path.name, str(path), record.family, record.architecture, dest_subdir, "left", reason)
            )
            logger.info("model_sorter: leaving %s in inbox (%s)", path.name, reason)
            continue

        dest_dir = models_dir / dest_subdir
        dest = dest_dir / path.name
        try:
            dest_dir.resolve(strict=False).relative_to(models_dir)
            dest.resolve(strict=False).relative_to(models_dir)
        except (OSError, RuntimeError, ValueError):
            actions.append(
                SortAction(path.name, str(path), record.family, record.architecture, dest_subdir, "left", "destination escapes the configured models root")
            )
            continue

        if dest.exists() or dest.is_symlink():
            actions.append(
                SortAction(
                    path.name, str(path), record.family, record.architecture, dest_subdir,
                    "conflict", f"{dest_subdir}/{path.name} already exists",
                )
            )
            logger.warning(
                "model_sorter: %s already has %s — leaving inbox copy untouched", dest_subdir, path.name
            )
            continue

        if apply:
            try:
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_dir.resolve(strict=True).relative_to(models_dir)
                dest.resolve(strict=False).relative_to(models_dir)
                shutil.move(str(source), str(dest))
            except (OSError, RuntimeError, ValueError):
                logger.warning("model_sorter: failed to move %s -> %s", path.name, dest_subdir, exc_info=True)
                actions.append(
                    SortAction(path.name, str(path), record.family, record.architecture, dest_subdir, "error", "move failed")
                )
                continue

        actions.append(
            SortAction(path.name, str(path), record.family, record.architecture, dest_subdir, "moved", "")
        )
        logger.info(
            "model_sorter: %s %s -> %s  [%s / %s]",
            "moved" if apply else "would move",
            path.name,
            dest_subdir,
            record.family,
            record.architecture,
        )

    actions.extend(_sort_diffusers_dirs(models_dir, inbox, roots, snapshot_dirs, apply=apply))

    if apply and any(action.moved for action in actions):
        invalidate_model_inventory_cache()
    return actions


def _run_all_models(
    flags: RuntimeFlags,
    *,
    apply: bool,
    approved_moves: set[tuple[str, str, str, str]] | None = None,
) -> list[SortAction]:
    models_dir = flags.resolved_models_dir().resolve()
    if not models_dir.is_dir():
        return []

    roots = model_inventory_roots(flags)
    snapshot_dirs = _snapshot_dirs(models_dir)
    actions: list[SortAction] = []

    for path in sorted(models_dir.rglob("*"), key=lambda p: str(p).lower()):
        if not path.is_file() or path.suffix.lower() not in MODEL_EXTENSIONS:
            continue
        # Never resolve and move through a file symlink. Resolving first can
        # move the symlink target and leave a broken link in the user's tree.
        if path.is_symlink():
            actions.append(
                SortAction(path.name, str(path), "unknown", "unknown", "", "left", "linked model files are not reorganized")
            )
            continue
        if _is_inside_any(path, snapshot_dirs):
            continue
        try:
            resolved = path.resolve()
            resolved.relative_to(models_dir)
        except (OSError, ValueError):
            continue

        try:
            record = classify_model_file(resolved, roots)
        except Exception:
            logger.warning("model_sorter: could not classify %s - leaving in place", resolved.name, exc_info=True)
            actions.append(SortAction(resolved.name, str(resolved), "unknown", "unknown", "", "error", "header read failed"))
            continue

        if record is None:
            continue
        if not record.should_move:
            continue

        confident, reason = _is_confident(record)
        dest_subdir = record.recommended_subdir
        if not confident:
            actions.append(
                SortAction(resolved.name, str(resolved), record.family, record.architecture, dest_subdir, "left", reason)
            )
            continue

        dest_dir = models_dir / dest_subdir
        dest = dest_dir / resolved.name
        if not _destination_inside_models_root(models_dir, dest_dir, dest):
            actions.append(
                SortAction(
                    resolved.name,
                    str(resolved),
                    record.family,
                    record.architecture,
                    dest_subdir,
                    "left",
                    "destination escapes the configured models root",
                )
            )
            continue
        try:
            if dest.resolve() == resolved:
                continue
        except OSError:
            pass

        if dest.exists():
            actions.append(
                SortAction(
                    resolved.name,
                    str(resolved),
                    record.family,
                    record.architecture,
                    dest_subdir,
                    "conflict",
                    f"{dest_subdir}/{resolved.name} already exists",
                )
            )
            continue

        if apply and approved_moves is not None and _approved_move_key(
            resolved, record.family, record.architecture, dest_subdir
        ) not in approved_moves:
            actions.append(
                SortAction(
                    resolved.name,
                    str(resolved),
                    record.family,
                    record.architecture,
                    dest_subdir,
                    "left",
                    "model was not included in the reviewed placement plan",
                )
            )
            continue

        if apply:
            try:
                dest_dir.mkdir(parents=True, exist_ok=True)
                if not _destination_inside_models_root(models_dir, dest_dir, dest):
                    raise ValueError("destination escapes the configured models root")
                shutil.move(str(resolved), str(dest))
            except (OSError, RuntimeError, ValueError):
                logger.warning("model_sorter: failed to move %s -> %s", resolved.name, dest_subdir, exc_info=True)
                actions.append(
                    SortAction(
                        resolved.name,
                        str(resolved),
                        record.family,
                        record.architecture,
                        dest_subdir,
                        "error",
                        "move failed",
                    )
                )
                continue

        actions.append(
            SortAction(resolved.name, str(resolved), record.family, record.architecture, dest_subdir, "moved", "")
        )

    actions.extend(
        _sort_diffusers_dirs(
            models_dir,
            models_dir,
            roots,
            snapshot_dirs,
            apply=apply,
            approved_moves=approved_moves,
        )
    )

    if apply and any(action.moved for action in actions):
        invalidate_model_inventory_cache()
    return actions


def _snapshot_dirs(scan_root: Path) -> list[Path]:
    """Return outermost model-index folders without following directory links."""
    if not scan_root.is_dir():
        return []
    result: list[Path] = []
    for index_path in sorted(
        scan_root.rglob("model_index.json"),
        key=lambda p: (len(p.parts), str(p).casefold()),
    ):
        folder = index_path.parent
        if folder.is_symlink():
            continue
        try:
            folder.resolve(strict=True).relative_to(scan_root.resolve(strict=True))
        except (OSError, RuntimeError, ValueError):
            continue
        if not _is_inside_any(folder, result):
            result.append(folder)
    return result


def _is_inside_any(path: Path, roots: list[Path]) -> bool:
    for root in roots:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _approved_move_key(
    source: Path,
    family: str,
    architecture: str,
    dest_subdir: str,
) -> tuple[str, str, str, str]:
    return (
        str(source.resolve(strict=False)).casefold(),
        str(family or "").casefold(),
        str(architecture or "").casefold(),
        str(dest_subdir or "").replace("\\", "/").casefold(),
    )


def _destination_inside_models_root(models_root: Path, *candidates: Path) -> bool:
    try:
        resolved_root = models_root.resolve(strict=True)
        for candidate in candidates:
            candidate.resolve(strict=False).relative_to(resolved_root)
        return True
    except (OSError, RuntimeError, ValueError):
        return False


def _sort_diffusers_dirs(
    models_dir: Path,
    scan_root: Path,
    roots: list[Path],
    snapshot_dirs: list[Path],
    *,
    apply: bool,
    approved_moves: set[tuple[str, str, str, str]] | None = None,
) -> list[SortAction]:
    actions: list[SortAction] = []
    models_root = models_dir.resolve()
    scan_path = scan_root.resolve()
    for source in snapshot_dirs:
        try:
            source_resolved = source.resolve(strict=True)
            source_resolved.relative_to(scan_path)
            source_resolved.relative_to(models_root)
        except (OSError, RuntimeError, ValueError):
            continue

        record = classify_model_dir(source_resolved, roots)
        if record is None:
            continue
        try:
            model_index = json.loads(
                (source_resolved / "model_index.json").read_text(encoding="utf-8")
            )
            class_name = str(model_index.get("_class_name") or "")
        except (OSError, ValueError, AttributeError):
            class_name = ""
        pipeline_route = _DIFFUSERS_PIPELINE_DESTINATIONS.get(class_name)
        if not pipeline_route:
            if class_name in {"QwenImageEditPipeline", "QwenImageEditPlusPipeline"}:
                actions.append(SortAction(
                    source_resolved.name,
                    str(source_resolved),
                    record.family,
                    record.architecture,
                    "",
                    "left",
                    f"{class_name} is not supported by the current Qwen Image generation route; folder left in the sort inbox.",
                ))
            continue
        if record.family == "controlnet":
            continue
        expected_architecture, dest_subdir = pipeline_route
        if record.architecture != expected_architecture:
            continue
        family_readiness_checked = False
        family_readiness_error = ""
        if class_name == "Flux2KleinPipeline":
            from aiwf.infrastructure.diffusers.checkpoints import (
                flux2_klein_components_missing_local_files,
                flux2_klein_missing_local_files,
            )

            full_ready = not flux2_klein_missing_local_files(source_resolved)
            components_ready = not flux2_klein_components_missing_local_files(source_resolved)
            family_readiness_checked = True
            if full_ready:
                dest_subdir = "flux2/Diffusers"
            elif components_ready and not any(part.casefold() == "diffusers" for part in source_resolved.parts):
                dest_subdir = "flux2/Components"
            else:
                family_readiness_error = "Flux.2 Klein snapshot is incomplete for both full-pipeline and support-component use"
        elif class_name == "ZImagePipeline":
            from aiwf.infrastructure.diffusers.checkpoints import (
                z_image_components_missing_local_files,
                z_image_missing_local_files,
            )

            full_ready = not z_image_missing_local_files(source_resolved)
            components_ready = not z_image_components_missing_local_files(source_resolved)
            family_readiness_checked = True
            if full_ready:
                dest_subdir = "z-image/Diffusers"
            elif components_ready and not any(part.casefold() == "diffusers" for part in source_resolved.parts):
                dest_subdir = "z-image/Components"
            else:
                family_readiness_error = "Z-Image snapshot is incomplete for both full-pipeline and support-component use"

        if family_readiness_error or (
            not family_readiness_checked and not diffusers_dir_has_required_local_files(source_resolved)
        ):
            actions.append(
                SortAction(
                    source_resolved.name,
                    str(source_resolved),
                    record.family,
                    record.architecture,
                    dest_subdir,
                    "left",
                    family_readiness_error or "Diffusers snapshot is incomplete",
                )
            )
            continue

        dest_dir = models_root / dest_subdir
        dest = dest_dir / source_resolved.name
        if not _destination_inside_models_root(models_root, dest_dir, dest):
            actions.append(
                SortAction(
                    source_resolved.name,
                    str(source_resolved),
                    record.family,
                    record.architecture,
                    dest_subdir,
                    "left",
                    "destination escapes the configured models root",
                )
            )
            continue
        if dest.resolve(strict=False) == source_resolved:
            continue
        if dest.exists():
            actions.append(
                SortAction(
                    source_resolved.name,
                    str(source_resolved),
                    record.family,
                    record.architecture,
                    dest_subdir,
                    "conflict",
                    f"{dest_subdir}/{source_resolved.name} already exists",
                )
            )
            continue

        if apply and approved_moves is not None and _approved_move_key(
            source_resolved, record.family, record.architecture, dest_subdir
        ) not in approved_moves:
            actions.append(
                SortAction(
                    source_resolved.name,
                    str(source_resolved),
                    record.family,
                    record.architecture,
                    dest_subdir,
                    "left",
                    "model folder was not included in the reviewed placement plan",
                )
            )
            continue

        if apply:
            try:
                dest_dir.mkdir(parents=True, exist_ok=True)
                if not _destination_inside_models_root(models_root, dest_dir, dest):
                    raise ValueError("destination escapes the configured models root")
                source_resolved.rename(dest)
            except (OSError, RuntimeError, ValueError):
                logger.warning(
                    "model_sorter: failed to move Diffusers folder %s -> %s",
                    source_resolved,
                    dest,
                    exc_info=True,
                )
                actions.append(
                    SortAction(
                        source_resolved.name,
                        str(source_resolved),
                        record.family,
                        record.architecture,
                        dest_subdir,
                        "error",
                        "move failed",
                    )
                )
                continue
        actions.append(
            SortAction(
                source_resolved.name,
                str(source_resolved),
                record.family,
                record.architecture,
                dest_subdir,
                "moved",
                "",
            )
        )
        logger.info(
            "model_sorter: %s Diffusers folder %s -> %s [%s / %s]",
            "moved" if apply else "would move",
            source_resolved.name,
            dest_subdir,
            record.family,
            record.architecture,
        )
    return actions


def sort_inbox_on_startup(flags: RuntimeFlags) -> list[SortAction]:
    """Entry point called during app startup. Never raises."""
    try:
        actions = sort_inbox_models(flags)
    except Exception:
        logger.warning("model_sorter: inbox sort failed", exc_info=True)
        return []

    moved = [a for a in actions if a.moved]
    left = [a for a in actions if a.status in {"left", "conflict", "error"}]
    if moved:
        logger.info("model_sorter: sorted %d model(s) from '%s' into place", len(moved), SORT_INBOX_DIRNAME)
    if left:
        logger.info(
            "model_sorter: left %d file(s) in '%s' for manual review", len(left), SORT_INBOX_DIRNAME
        )
    return actions
