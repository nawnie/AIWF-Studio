from __future__ import annotations

import hashlib
import json
import struct
import logging
import os
from pathlib import Path

from aiwf.core.config.settings import RuntimeFlags
from aiwf.core.domain.models import Checkpoint
from aiwf.infrastructure.diffusers.model_arch import (
    ARCH_FLUX,
    ARCH_FLUX_FILL,
    ARCH_FLUX_KONTEXT,
    ARCH_FLUX2_KLEIN,
    ARCH_ANIMA,
    ARCH_KREA2,
    ARCH_QWEN_IMAGE,
    ARCH_QWEN_IMAGE_NUNCHAKU,
    ARCH_SANA,
    ARCH_SANA_VIDEO,
    ARCH_Z_IMAGE,
    architecture_label,
    detect_checkpoint_architecture,
    is_inpaint_architecture,
    looks_like_controlnet_weights,
    looks_like_lora_weights,
)
from aiwf.infrastructure.diffusers.model_blocks import (
    is_blocked_selectable_image_asset,
    is_non_selectable_image_asset_path,
)
from aiwf.infrastructure.model_asset_summary import asset_file_count, asset_shape_label, asset_size_bytes
from aiwf.infrastructure.model_inventory import ModelInventoryRecord, get_model_inventory
from aiwf.infrastructure.safetensors_metadata import safetensors_file_is_structurally_valid
from aiwf.services.model_files import indexed_weight_shard_issues

logger = logging.getLogger(__name__)

CHECKPOINT_EXTENSIONS = {".ckpt", ".safetensors", ".pt"}
VAE_SUFFIXES = (".vae.safetensors", ".vae.ckpt", ".vae.pt")
SKIP_DIR_NAMES = {
    "vae",
    "vae-approx",
    "lora",
    "loras",
    "embeddings",
    "hypernetworks",
    "codeformer",
    "gfpgan",
    "esrgan",
    "realesrgan",
    "upscale_models",
    "deepbooru",
    "diffusion_models",
    "karlo",
    "controlnet",
    "clip",
    "clip_vision",
    "sam",
    "textencoder",
    "text_encoder",
    "text-encoder",
    "ultralytics",
    "wan",
}


def resolve_search_roots(flags: RuntimeFlags) -> list[Path]:
    """All directories scanned for checkpoints, in priority order.

    This is the local model discovery boundary for image generation. Keep it
    rooted in configured AIWF/model directories; broad drive scans belong in a
    user-triggered import/index task, not in normal app startup.
    """
    models_dir = flags.resolved_models_dir()
    ckpt_dir = flags.resolved_ckpt_dir()
    roots: list[Path] = []

    seen_keys: set[str] = set()

    def add_root(path: Path) -> None:
        resolved = path.resolve()
        if resolved.name.lower() in SKIP_DIR_NAMES:
            return
        key = os.path.normcase(str(resolved))
        if resolved.exists() and key not in seen_keys:
            seen_keys.add(key)
            roots.append(resolved)

    for candidate in (ckpt_dir, models_dir / "Stable-diffusion", models_dir / "stable-diffusion", models_dir):
        add_root(candidate)
    for extra_ckpt_dir in flags.resolved_extra_ckpt_dirs():
        add_root(extra_ckpt_dir)
    for extra_models_dir in flags.resolved_extra_model_dirs():
        for candidate in (
            extra_models_dir / "Stable-diffusion",
            extra_models_dir / "stable-diffusion",
            extra_models_dir,
        ):
            add_root(candidate)

    return roots


def _fast_fingerprint(path: Path) -> str | None:
    """Quick id for UI labels — avoids reading multi-GB files on every scan."""
    try:
        if path.is_dir():
            marker = path / "model_index.json"
            if not marker.is_file():
                return None
            stat = marker.stat()
            digest = hashlib.sha256()
            digest.update(str(path.resolve()).encode())
            digest.update(str(stat.st_size).encode())
            digest.update(str(int(stat.st_mtime)).encode())
            digest.update(marker.read_bytes()[:1024 * 1024])
            return digest.hexdigest()[:10]
        stat = path.stat()
        digest = hashlib.sha256()
        digest.update(str(stat.st_size).encode())
        digest.update(str(int(stat.st_mtime)).encode())
        with path.open("rb") as handle:
            digest.update(handle.read(1024 * 1024))
            if stat.st_size > 1024 * 1024:
                handle.seek(-1024 * 1024, 2)
                digest.update(handle.read(1024 * 1024))
        return digest.hexdigest()[:10]
    except OSError:
        return None


def _is_checkpoint_file(path: Path) -> bool:
    if not path.is_file():
        return False
    lower_name = path.name.lower()
    if path.suffix.lower() not in CHECKPOINT_EXTENSIONS:
        return False
    return not any(lower_name.endswith(suffix) for suffix in VAE_SUFFIXES)


def _iter_checkpoint_files(root: Path) -> list[Path]:
    found: list[Path] = []

    def walk(directory: Path) -> None:
        try:
            entries = sorted(directory.iterdir(), key=lambda p: p.name.lower())
        except OSError:
            return

        for entry in entries:
            if entry.is_file() and _is_checkpoint_file(entry):
                found.append(entry)
                continue
            if not entry.is_dir():
                continue
            # Avoid descending into sibling asset families whose files often
            # share .safetensors/.pt suffixes but are not selectable checkpoints.
            if entry.name.lower() in SKIP_DIR_NAMES:
                continue
            walk(entry)

    walk(root)
    return found


def scan_checkpoints(roots: list[Path]) -> list[Checkpoint]:
    seen_paths: set[str] = set()
    results: list[Checkpoint] = []

    for root in roots:
        if not root.exists():
            logger.debug("Checkpoint root does not exist: %s", root)
            continue

        for path in _iter_checkpoint_files(root):
            if is_blocked_selectable_image_asset(path):
                logger.debug("Skipping blocked/non-selectable image asset in checkpoint scan: %s", path)
                continue
            resolved = str(path.resolve())
            dedup_key = os.path.normcase(resolved)
            if dedup_key in seen_paths:
                continue
            if looks_like_lora_weights(path):
                logger.debug("Skipping LoRA weights in checkpoint scan: %s", path)
                continue
            if looks_like_controlnet_weights(path):
                logger.debug("Skipping ControlNet weights in checkpoint scan: %s", path)
                continue
            seen_paths.add(dedup_key)

            short_hash = _fast_fingerprint(path)
            checkpoint_id = path.stem
            architecture = detect_checkpoint_architecture(path)
            arch_tag = architecture_label(architecture)
            size_bytes = asset_size_bytes(path)
            file_count = asset_file_count(path)
            summary = asset_shape_label(path, size_bytes=size_bytes, file_count=file_count)
            title = f"{checkpoint_id} [{arch_tag}] [{summary}]"
            results.append(
                Checkpoint(
                    id=checkpoint_id,
                    title=title,
                    filename=path.name,
                    path=resolved,
                    hash=short_hash,
                    kind="inpaint" if is_inpaint_architecture(architecture) else "checkpoint",
                    architecture=architecture,
                    size_bytes=size_bytes,
                    file_count=file_count,
                    asset_summary=summary,
                )
            )

    results = _disambiguate_duplicate_checkpoint_ids(results)

    def sort_key(item: Checkpoint) -> tuple[int, str]:
        name = item.filename.lower()
        is_inpaint = "inpaint" in name
        return (1 if is_inpaint else 0, item.title.lower())

    results.sort(key=sort_key)
    return results


def scan_from_flags(flags: RuntimeFlags) -> list[Checkpoint]:
    roots = resolve_search_roots(flags)
    logger.info("Scanning for checkpoints in: %s", ", ".join(str(r) for r in roots))
    inventory = get_model_inventory(flags)
    selectable_runtime_arches = {
        ARCH_FLUX,
        ARCH_FLUX_FILL,
        ARCH_FLUX_KONTEXT,
        ARCH_FLUX2_KLEIN,
        ARCH_Z_IMAGE,
        ARCH_KREA2,
        ARCH_QWEN_IMAGE,
        ARCH_SANA,
    }
    non_image_runtime_arches = {ARCH_SANA_VIDEO}
    checkpoints = [
        _checkpoint_from_inventory(record)
        for record in inventory
        if _is_selectable_inventory_checkpoint(record, selectable_runtime_arches, non_image_runtime_arches)
    ]
    checkpoints = _disambiguate_duplicate_checkpoint_ids(checkpoints)
    checkpoints.sort(key=lambda item: (1 if item.kind == "inpaint" else 0, item.title.lower()))

    logger.info("Found %d checkpoint(s)", len(checkpoints))
    if not checkpoints:
        logger.warning(
            "No checkpoints found. Place .safetensors or .ckpt files in: %s",
            flags.resolved_ckpt_dir(),
        )
    return checkpoints


def _disambiguate_duplicate_checkpoint_ids(
    checkpoints: list[Checkpoint],
) -> list[Checkpoint]:
    """Give same-stem files distinct, path-stable IDs and readable picker titles."""
    counts: dict[str, int] = {}
    for item in checkpoints:
        counts[item.id] = counts.get(item.id, 0) + 1
    duplicate_ids = {model_id for model_id, count in counts.items() if count > 1}
    used_ids = {item.id for item in checkpoints if item.id not in duplicate_ids}
    for item in sorted(
        (entry for entry in checkpoints if entry.id in duplicate_ids),
        key=lambda entry: os.path.normcase(entry.path),
    ):
        stem_id = item.id
        path_key = os.path.normcase(str(Path(item.path).resolve()))
        digest = hashlib.sha256(path_key.encode("utf-8", errors="replace")).hexdigest()[:10]
        unique_id = f"{stem_id}@{digest}"
        suffix = 1
        while unique_id in used_ids:
            suffix += 1
            unique_id = f"{stem_id}@{digest}-{suffix}"
        used_ids.add(unique_id)
        item.id = unique_id
        item.title = f"{item.title} · {Path(item.path).parent.name} ({digest})"
    return checkpoints


def _is_selectable_inventory_checkpoint(
    record: ModelInventoryRecord,
    selectable_runtime_arches: set[str],
    non_image_runtime_arches: set[str],
) -> bool:
    path = Path(record.path)
    if is_blocked_selectable_image_asset(path):
        logger.debug("Skipping blocked/non-selectable inventory checkpoint: %s", path)
        return False
    if is_non_selectable_image_asset_path(path):
        return False
    if record.family == "runtime_asset" and record.architecture in selectable_runtime_arches:
        if path.is_dir() and not diffusers_dir_has_required_local_files(path):
            logger.info("Skipping incomplete Diffusers runtime folder in checkpoint scan: %s", path)
            return False
    if record.family == "checkpoint" and record.architecture not in non_image_runtime_arches:
        if record.architecture == "unknown":
            logger.info("Skipping unknown checkpoint architecture in checkpoint scan: %s", path)
            return False
        if looks_like_lora_weights(path):
            logger.debug("Skipping LoRA inventory checkpoint: %s", path)
            return False
        if looks_like_controlnet_weights(path):
            logger.debug("Skipping ControlNet inventory checkpoint: %s", path)
            return False
        return True
    return record.family == "runtime_asset" and record.architecture in selectable_runtime_arches


def diffusers_dir_has_required_local_files(path: Path) -> bool:
    """Best-effort offline check for sharded Diffusers folders.

    A folder can have model_index.json while one of its component shard files is
    missing. Diffusers catches that only at load time; checkpoint discovery
    should avoid offering those folders as selectable routes.
    """
    model_index = Path(path) / "model_index.json"
    if not model_index.is_file():
        return False
    try:
        payload = json.loads(model_index.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if isinstance(payload, dict) and payload.get("_class_name") == "FluxKontextPipeline":
        return not flux_kontext_missing_local_files(path, limit=1)
    if missing_diffusers_local_files(path, limit=1):
        return False
    return _diffusers_support_components_ready(Path(path), payload)


def _diffusers_support_components_ready(root: Path, model_index: dict) -> bool:
    """Check referenced schedulers, tokenizers, and weighted components."""
    weighted_components = {
        "transformer", "unet", "text_encoder", "text_encoder_2", "vae",
        "image_encoder", "controlnet", "prior",
    }
    found_weighted_component = False
    for component, reference in model_index.items():
        if not isinstance(reference, list) or len(reference) < 2 or not reference[1]:
            continue
        component_name = str(component).casefold()
        component_root = root / str(component)
        if component_name in weighted_components:
            found_weighted_component = True
            if not _nonempty_file_confined_to(root, component_root / "config.json"):
                return False
            try:
                has_weight_file = any(
                    child.is_file()
                    and child.suffix.casefold() in {".safetensors", ".bin", ".pt", ".onnx"}
                    and _nonempty_file_confined_to(root, child)
                    for child in component_root.iterdir()
                )
                has_index = any(
                    child.is_file()
                    and child.name.endswith(".index.json")
                    and _nonempty_file_confined_to(root, child)
                    for child in component_root.iterdir()
                )
            except OSError:
                return False
            if not has_weight_file and not has_index:
                return False
        if component_name == "scheduler":
            if not _nonempty_file_confined_to(root, component_root / "scheduler_config.json"):
                return False
        elif component_name.startswith("tokenizer"):
            tokenizer_files = (
                "tokenizer.json", "spiece.model", "tokenizer.model", "vocab.txt",
            )
            has_core_file = any(
                _nonempty_file_confined_to(root, component_root / name)
                for name in tokenizer_files
            )
            has_bpe_pair = all(
                _nonempty_file_confined_to(root, component_root / name)
                for name in ("vocab.json", "merges.txt")
            )
            if not has_core_file and not has_bpe_pair:
                return False
    return found_weighted_component


def missing_diffusers_local_files(path: Path, *, limit: int = 20) -> list[Path]:
    """Return indexes with incomplete or unsafe local Diffusers shard references."""
    missing: list[Path] = []
    root = Path(path).resolve()
    for index_path in root.rglob("*.index.json"):
        try:
            index_path.resolve(strict=True).relative_to(root)
        except (OSError, ValueError):
            missing.append(index_path)
            if len(missing) >= limit:
                return missing
            continue
        issues = indexed_weight_shard_issues(index_path.parent, index_path)
        if issues:
            logger.debug("Diffusers shard index is incomplete or unsafe: %s", index_path)
            missing.extend(issues[: max(1, limit - len(missing))])
            if len(missing) >= limit:
                return missing[:limit]
    return missing


def flux_kontext_missing_local_files(
    path: Path, *, limit: int = 20, require_transformer_weights: bool = True
) -> list[Path]:
    """Validate the concrete files used by a full Flux Kontext Diffusers pipeline.

    Shard indexes alone are insufficient: an unsharded folder with empty
    component directories otherwise appears ready even though ``from_pretrained``
    cannot load it.
    """
    root = Path(path).resolve()
    required_json = (
        Path("model_index.json"),
        Path("scheduler") / "scheduler_config.json",
        Path("text_encoder") / "config.json",
        Path("text_encoder_2") / "config.json",
        Path("tokenizer") / "tokenizer_config.json",
        Path("tokenizer_2") / "tokenizer_config.json",
        Path("transformer") / "config.json",
        Path("vae") / "config.json",
    )
    missing = [root / item for item in required_json if not _nonempty_file_confined_to(root, root / item)]
    for relative in required_json:
        candidate = root / relative
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict) and candidate not in missing:
            missing.append(candidate)

    expected_components = {
        "scheduler": ("diffusers", "FlowMatchEulerDiscreteScheduler"),
        "text_encoder": ("transformers", "CLIPTextModel"),
        "text_encoder_2": ("transformers", "T5EncoderModel"),
        "tokenizer": ("transformers", "CLIPTokenizer"),
        "tokenizer_2": ("transformers", "T5TokenizerFast"),
        "transformer": ("diffusers", "FluxTransformer2DModel"),
        "vae": ("diffusers", "AutoencoderKL"),
    }
    try:
        index = json.loads((root / "model_index.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        index = None
    if not isinstance(index, dict) or index.get("_class_name") != "FluxKontextPipeline" or any(
        not isinstance(index.get(name), list) or tuple(index[name]) != expected
        for name, expected in expected_components.items()
    ):
        if root / "model_index.json" not in missing:
            missing.append(root / "model_index.json")

    weight_names = {
        "text_encoder": ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json"),
        "text_encoder_2": ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json"),
        "vae": ("diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin", "diffusion_pytorch_model.safetensors.index.json", "diffusion_pytorch_model.bin.index.json"),
    }
    if require_transformer_weights:
        weight_names["transformer"] = (
            "diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin",
            "diffusion_pytorch_model.safetensors.index.json", "diffusion_pytorch_model.bin.index.json",
        )
    for component, names in weight_names.items():
        candidates = [root / component / name for name in names]
        if not any(
            _nonempty_file_confined_to(root, candidate)
            and (not candidate.name.endswith(".safetensors") or candidate.name.endswith(".index.json") or _safetensors_header_is_valid(candidate))
            for candidate in candidates
        ):
            missing.append(candidates[0])

    tokenizer_assets = (
        ((root / "tokenizer" / "tokenizer.json",), (root / "tokenizer" / "vocab.json", root / "tokenizer" / "merges.txt")),
        ((root / "tokenizer_2" / "tokenizer.json",), (root / "tokenizer_2" / "spiece.model",)),
    )
    for alternatives in tokenizer_assets:
        if not any(all(_nonempty_file_confined_to(root, item) for item in option) for option in alternatives):
            missing.append(alternatives[0][0])

    try:
        missing.extend(missing_diffusers_local_files(root, limit=limit))
    except (OSError, RuntimeError):
        missing.append(root / "model_index.json")
    return list(dict.fromkeys(missing[:limit]))


def sana_video_missing_local_files(path: Path) -> list[Path]:
    """Validate the concrete local components required by AIWF's Sana Video route.

    A bare model_index.json is not sufficient evidence that a Diffusers
    snapshot can load. Require each model component's config and weights,
    scheduler config, and tokenizer files before reporting the route ready.
    """
    root = Path(path).resolve()
    required_files = (
        Path("model_index.json"),
        Path("scheduler") / "scheduler_config.json",
        Path("text_encoder") / "config.json",
        Path("tokenizer") / "tokenizer.json",
        Path("tokenizer") / "tokenizer_config.json",
        Path("transformer") / "config.json",
        Path("vae") / "config.json",
    )
    missing: list[Path] = []
    for relative in required_files:
        candidate = root / relative
        if not _nonempty_file_confined_to(root, candidate):
            missing.append(candidate)

    model_index = root / "model_index.json"
    try:
        payload = json.loads(model_index.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        payload = None
    if not isinstance(payload, dict) or payload.get("_class_name") != "SanaVideoPipeline":
        if model_index not in missing:
            missing.append(model_index)
    else:
        expected_components = {
            "scheduler": "DPMSolverMultistepScheduler",
            "text_encoder": "Gemma2Model",
            "tokenizer": "GemmaTokenizerFast",
            "transformer": "SanaVideoTransformer3DModel",
            "vae": "AutoencoderKLWan",
        }
        for component, expected_class in expected_components.items():
            declaration = payload.get(component)
            if not isinstance(declaration, list) or len(declaration) != 2 or declaration[1] != expected_class:
                missing.append(model_index)
                break

    # Nonempty files alone can be truncated or malformed. Parse the route's
    # JSON metadata as objects before advertising the snapshot as structurally ready.
    for relative in required_files:
        if relative.name == "tokenizer.json":
            continue
        candidate = root / relative
        try:
            metadata = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            if candidate not in missing:
                missing.append(candidate)
            continue
        if not isinstance(metadata, dict) and candidate not in missing:
            missing.append(candidate)
    tokenizer_json = root / "tokenizer" / "tokenizer.json"
    try:
        tokenizer_payload = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        tokenizer_payload = None
    if not isinstance(tokenizer_payload, dict) and tokenizer_json not in missing:
        missing.append(tokenizer_json)

    weight_groups = (
        (
            root / "text_encoder",
            ("model.safetensors.index.json", "model.safetensors", "pytorch_model.bin"),
        ),
        (
            root / "transformer",
            (
                "diffusion_pytorch_model.safetensors.index.json",
                "diffusion_pytorch_model.safetensors",
                "diffusion_pytorch_model.bin",
            ),
        ),
        (
            root / "vae",
            ("diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin"),
        ),
    )
    for component, names in weight_groups:
        candidates = [component / name for name in names]
        if not any(_nonempty_file_confined_to(root, candidate) for candidate in candidates):
            missing.append(candidates[0])

    # Validate all present shard indexes, including ones beyond the required
    # text encoder index, so malformed or escaping shard references fail closed.
    missing.extend(missing_diffusers_local_files(root, limit=20))
    return list(dict.fromkeys(missing))


def z_image_missing_local_files(path: Path) -> list[Path]:
    """Validate all files used by a full Z-Image Diffusers snapshot."""
    return _z_image_missing_local_files(path, require_transformer=True)


def z_image_components_missing_local_files(path: Path) -> list[Path]:
    """Validate Z-Image support components paired with a separate transformer."""
    return _z_image_missing_local_files(path, require_transformer=False)


def flux2_klein_components_missing_local_files(path: Path) -> list[Path]:
    """Validate the support components paired with a Flux.2 Klein transformer."""
    return _flux2_klein_missing_local_files(path, require_transformer=False)


def flux2_klein_missing_local_files(path: Path) -> list[Path]:
    """Validate a complete Flux.2 Klein Diffusers pipeline, including transformer weights."""
    return _flux2_klein_missing_local_files(path, require_transformer=True)


def _flux2_klein_missing_local_files(path: Path, *, require_transformer: bool) -> list[Path]:
    root = Path(path).resolve()
    required = [
        Path("model_index.json"),
        Path("scheduler") / "scheduler_config.json",
        Path("text_encoder") / "config.json",
        Path("vae") / "config.json",
        Path("tokenizer") / "tokenizer_config.json",
    ]
    if require_transformer:
        required.append(Path("transformer") / "config.json")
    missing = [root / item for item in required if not _nonempty_file_confined_to(root, root / item)]

    identities = {
        Path("scheduler") / "scheduler_config.json": ("_class_name", "FlowMatchEulerDiscreteScheduler", ("num_train_timesteps",)),
        Path("text_encoder") / "config.json": ("model_type", "qwen3", ("hidden_size", "num_hidden_layers", "vocab_size")),
        Path("vae") / "config.json": ("_class_name", "AutoencoderKLFlux2", ("in_channels", "out_channels", "latent_channels")),
        Path("transformer") / "config.json": ("_class_name", "Flux2Transformer2DModel", ("in_channels", "num_layers")),
    }
    for relative in required:
        candidate = root / relative
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            if candidate not in missing:
                missing.append(candidate)
            continue
        expected = identities.get(relative)
        if expected and (
            payload.get(expected[0]) != expected[1]
            or any(payload.get(field) in (None, "") for field in expected[2])
        ):
            missing.append(candidate)
        if relative == Path("tokenizer") / "tokenizer_config.json" and payload.get("tokenizer_class") not in {
            "Qwen2Tokenizer", "Qwen2TokenizerFast"
        }:
            missing.append(candidate)

    model_index = root / "model_index.json"
    try:
        payload = json.loads(model_index.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        payload = None
    expected_components = {
        "scheduler": ("diffusers", "FlowMatchEulerDiscreteScheduler"),
        "text_encoder": ("transformers", "Qwen3ForCausalLM"),
        "tokenizer": ("transformers", "Qwen2TokenizerFast"),
        "transformer": ("diffusers", "Flux2Transformer2DModel"),
        "vae": ("diffusers", "AutoencoderKLFlux2"),
    }
    if not isinstance(payload, dict) or payload.get("_class_name") != "Flux2KleinPipeline" or any(
        not isinstance(payload.get(name), list) or tuple(payload[name]) != expected
        for name, expected in expected_components.items()
    ):
        if model_index not in missing:
            missing.append(model_index)

    weights = {
        "text_encoder": ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json"),
        "vae": ("diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin", "diffusion_pytorch_model.safetensors.index.json", "diffusion_pytorch_model.bin.index.json"),
    }
    if require_transformer:
        weights["transformer"] = (
            "diffusion_pytorch_model.safetensors",
            "diffusion_pytorch_model.bin",
            "diffusion_pytorch_model.safetensors.index.json",
            "diffusion_pytorch_model.bin.index.json",
        )
    for component, names in weights.items():
        candidates = [root / component / name for name in names]
        if not any(
            _nonempty_file_confined_to(root, candidate)
            and (not candidate.name.endswith(".safetensors") or candidate.name.endswith(".index.json") or _safetensors_header_is_valid(candidate))
            for candidate in candidates
        ):
            missing.append(candidates[0])

    tokenizer = root / "tokenizer"
    if not _qwen_tokenizer_files_ready(root):
        missing.append(tokenizer / "tokenizer.json")
    missing.extend(missing_diffusers_local_files(root, limit=20))
    return list(dict.fromkeys(missing))


def _z_image_missing_local_files(path: Path, *, require_transformer: bool) -> list[Path]:
    """Validate the shared Z-Image contract, optionally without transformer weights."""
    root = Path(path).resolve()
    required_json = [
        Path("model_index.json"),
        Path("scheduler") / "scheduler_config.json",
        Path("text_encoder") / "config.json",
        Path("vae") / "config.json",
        Path("tokenizer") / "tokenizer_config.json",
    ]
    if require_transformer:
        required_json.append(Path("transformer") / "config.json")
    missing = [root / relative for relative in required_json if not _nonempty_file_confined_to(root, root / relative)]
    for relative in required_json:
        candidate = root / relative
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict) and candidate not in missing:
            missing.append(candidate)
        elif isinstance(payload, dict):
            required_fields = {
                Path("scheduler") / "scheduler_config.json": ("_class_name", "num_train_timesteps"),
                Path("text_encoder") / "config.json": ("model_type", "hidden_size", "num_hidden_layers", "vocab_size"),
                Path("transformer") / "config.json": ("_class_name", "in_channels", "dim", "n_layers"),
                Path("vae") / "config.json": ("_class_name", "in_channels", "out_channels", "latent_channels"),
                Path("tokenizer") / "tokenizer_config.json": ("tokenizer_class",),
            }.get(relative, ())
            if any(field not in payload or payload[field] in (None, "") for field in required_fields):
                missing.append(candidate)
            expected_identity = {
                Path("scheduler") / "scheduler_config.json": ("_class_name", "FlowMatchEulerDiscreteScheduler"),
                Path("text_encoder") / "config.json": ("model_type", "qwen3"),
                Path("transformer") / "config.json": ("_class_name", "ZImageTransformer2DModel"),
                Path("vae") / "config.json": ("_class_name", "AutoencoderKL"),
                Path("tokenizer") / "tokenizer_config.json": ("tokenizer_class", "Qwen2Tokenizer"),
            }.get(relative)
            if expected_identity and payload.get(expected_identity[0]) != expected_identity[1] and candidate not in missing:
                missing.append(candidate)

    model_index = root / "model_index.json"
    try:
        payload = json.loads(model_index.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        payload = None
    if not isinstance(payload, dict) or payload.get("_class_name") != "ZImagePipeline":
        if model_index not in missing:
            missing.append(model_index)
    else:
        expected_components = {
            "scheduler": ("diffusers", "FlowMatchEulerDiscreteScheduler"),
            "text_encoder": ("transformers", "Qwen3Model"),
            "tokenizer": ("transformers", "Qwen2Tokenizer"),
            "transformer": ("diffusers", "ZImageTransformer2DModel"),
            "vae": ("diffusers", "AutoencoderKL"),
        }
        for component, expected in expected_components.items():
            declaration = payload.get(component)
            if not isinstance(declaration, list) or tuple(declaration) != expected:
                if model_index not in missing:
                    missing.append(model_index)
                break

    recognized_weight_names = {
        "text_encoder": (
            "model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json",
        ),
        "vae": (
            "diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin",
            "diffusion_pytorch_model.safetensors.index.json", "diffusion_pytorch_model.bin.index.json",
        ),
    }
    if require_transformer:
        recognized_weight_names["transformer"] = (
            "diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.bin",
            "diffusion_pytorch_model.safetensors.index.json", "diffusion_pytorch_model.bin.index.json",
        )
    for component, names in recognized_weight_names.items():
        component_root = root / component
        valid = False
        for name in names:
            candidate = component_root / name
            if not _nonempty_file_confined_to(root, candidate):
                continue
            if name.endswith(".safetensors") and not name.endswith(".index.json"):
                if not _safetensors_header_is_valid(candidate):
                    continue
            valid = True
            break
        if not valid:
            missing.append(component_root / names[0])

    tokenizer_root = root / "tokenizer"
    if not _qwen_tokenizer_files_ready(root):
        missing.append(tokenizer_root / "tokenizer.json")

    try:
        missing.extend(missing_diffusers_local_files(root, limit=20))
    except (OSError, RuntimeError):
        missing.append(root / "model_index.json")
    return list(dict.fromkeys(missing))


def _qwen_tokenizer_files_ready(root: Path) -> bool:
    tokenizer_root = root / "tokenizer"
    tokenizer_json = tokenizer_root / "tokenizer.json"
    if _nonempty_file_confined_to(root, tokenizer_json):
        try:
            payload = json.loads(tokenizer_json.read_text(encoding="utf-8"))
            model = payload.get("model") if isinstance(payload, dict) else None
            if (
                isinstance(model, dict)
                and isinstance(model.get("type"), str)
                and isinstance(model.get("vocab"), dict)
            ):
                return True
        except (OSError, ValueError):
            pass
    return all(
        _nonempty_file_confined_to(root, tokenizer_root / name)
        for name in ("vocab.json", "merges.txt")
    )


def _safetensors_header_is_valid(path: Path) -> bool:
    """Check the safe-tensors framing and tensor metadata without loading tensors."""
    return safetensors_file_is_structurally_valid(path)


def sana_video_dir_has_required_local_files(path: Path) -> bool:
    return not sana_video_missing_local_files(path)


def _nonempty_file_confined_to(root: Path, path: Path) -> bool:
    try:
        resolved_root = root.resolve()
        resolved = path.resolve(strict=True)
        resolved.relative_to(resolved_root)
        return resolved.is_file() and resolved.stat().st_size > 0
    except (OSError, ValueError):
        return False


_diffusers_dir_has_required_local_files = diffusers_dir_has_required_local_files


def _checkpoint_from_inventory(record: ModelInventoryRecord) -> Checkpoint:
    path = Path(record.path)
    architecture = record.architecture or detect_checkpoint_architecture(path)
    short_hash = _fast_fingerprint(path)
    checkpoint_id = path.name if path.is_dir() else path.stem
    size_bytes = asset_size_bytes(path)
    file_count = asset_file_count(path)
    summary = asset_shape_label(path, size_bytes=size_bytes, file_count=file_count)
    title = f"{checkpoint_id} [{architecture_label(architecture)}] [{summary}]"
    is_runtime_asset = record.family == "runtime_asset"
    if is_runtime_asset and architecture == ARCH_FLUX:
        kind = "flux"
    elif is_runtime_asset and architecture == ARCH_FLUX_FILL:
        kind = "flux-fill"
    elif is_runtime_asset and architecture == ARCH_FLUX_KONTEXT:
        kind = "flux-kontext"
    elif is_runtime_asset and architecture == ARCH_FLUX2_KLEIN:
        kind = "flux2"
    elif is_runtime_asset and architecture == ARCH_Z_IMAGE:
        kind = "z-image"
    elif is_runtime_asset and architecture == ARCH_KREA2:
        kind = "krea2"
    elif is_runtime_asset and architecture == ARCH_ANIMA:
        kind = "anima"
    elif is_runtime_asset and architecture == ARCH_QWEN_IMAGE_NUNCHAKU:
        kind = "qwen-nunchaku"
    elif is_runtime_asset and architecture == ARCH_QWEN_IMAGE:
        kind = "qwen-image"
    elif is_runtime_asset and architecture == ARCH_SANA:
        kind = "sana"
    else:
        kind = "inpaint" if is_inpaint_architecture(architecture) else "checkpoint"
    return Checkpoint(
        id=checkpoint_id,
        title=title,
        filename=path.name,
        path=str(path.resolve()),
        hash=short_hash,
        kind=kind,
        architecture=architecture,
        size_bytes=size_bytes,
        file_count=file_count,
        asset_summary=summary,
    )
