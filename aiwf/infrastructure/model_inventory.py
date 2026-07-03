from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from aiwf.core.config.settings import RuntimeFlags
from aiwf.infrastructure.diffusers.model_arch import (
    ARCH_FLUX,
    ARCH_FLUX_KONTEXT,
    ARCH_FLUX2_KLEIN,
    ARCH_INPAINT,
    ARCH_QWEN_IMAGE,
    ARCH_QWEN_IMAGE_NUNCHAKU,
    ARCH_SANA,
    ARCH_SANA_VIDEO,
    ARCH_SD15,
    ARCH_SD35,
    ARCH_SDXL,
    ARCH_SDXL_INPAINT,
    ARCH_Z_IMAGE,
    UNET_INPUT_KEY,
    detect_checkpoint_architecture,
    infer_architecture_from_shapes,
    looks_like_lora_weights,
    _safetensors_tensor_shapes,
)
from aiwf.infrastructure.model_header import (
    ARCH_CLIP,
    ARCH_FLUX2_KLEIN_TRANSFORMER,
    ARCH_LTX_AUDIO_VAE,
    ARCH_LTX_LORA,
    ARCH_LTX_TRANSFORMER,
    ARCH_LTX_VAE,
    ARCH_FLUX_LORA,
    ARCH_FLUX_TRANSFORMER,
    ARCH_FLUX_VAE,
    ARCH_T5XXL_ENCODER,
    ARCH_UMT5_ENCODER,
    ARCH_WAN_LORA,
    ARCH_WAN_TRANSFORMER,
    ARCH_WAN_TRANSFORMER_FP8,
    ARCH_WAN_VAE,
    ARCH_Z_IMAGE_TRANSFORMER,
    ROLE_LORA,
    ROLE_TEXT_ENCODER,
    ROLE_VAE,
    read_model_info,
)
from aiwf.infrastructure.safetensors_metadata import read_safetensors_metadata

logger = logging.getLogger(__name__)

MODEL_EXTENSIONS = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx"}
MODEL_INVENTORY_VERSION = 1


@dataclass(frozen=True)
class ModelInventoryRecord:
    path: str
    filename: str
    family: str
    architecture: str
    current_subdir: str
    recommended_subdir: str
    should_move: bool
    header_identifiers: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)


def inventory_path(flags: RuntimeFlags) -> Path:
    # Keep the cache file OUT of the scanned models dir: writing it there bumps
    # the models-dir mtime, which changes the roots signature and self-
    # invalidates the very cache we just wrote, forcing a full rescan on the
    # next call (the cause of repeated multi-second "Indexed N assets" stalls
    # before each generation).
    return flags.data_dir / "cache" / "model_inventory.json"


def model_inventory_roots(flags: RuntimeFlags) -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()
    candidates = [
        flags.resolved_models_dir(),
        flags.resolved_ckpt_dir(),
        *flags.resolved_extra_model_dirs(),
        *flags.resolved_extra_ckpt_dirs(),
    ]
    for candidate in candidates:
        resolved = candidate.resolve()
        key = os.path.normcase(str(resolved))
        if not resolved.exists() or key in seen:
            continue
        if any(_is_relative_to(resolved, root) for root in roots):
            continue
        seen.add(key)
        roots.append(resolved)
    return roots


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _relative_subdir(path: Path, roots: list[Path]) -> str:
    parent = path.parent.resolve()
    for root in sorted((root.resolve() for root in roots), key=lambda p: len(str(p)), reverse=True):
        try:
            rel = parent.relative_to(root)
        except ValueError:
            continue
        return "" if str(rel) == "." else rel.as_posix()
    return parent.as_posix()


def _relative_path_text(path: Path, roots: list[Path]) -> str:
    resolved = path.resolve()
    for root in sorted((root.resolve() for root in roots), key=lambda p: len(str(p)), reverse=True):
        try:
            return resolved.relative_to(root).as_posix()
        except ValueError:
            continue
    return path.name


def _metadata_text(metadata: dict[str, str]) -> str:
    return " ".join(f"{key} {value}" for key, value in metadata.items()).lower()


def _architecture_from_text(text: str) -> str:
    normalized = text.lower().replace("_", " ").replace("-", " ")
    compact = normalized.replace(" ", "")
    lowered = text.lower().replace("_", "-")
    if (
        ("qwenimagepipeline" in compact or "qwen image" in normalized or "qwen-image" in lowered or "qwen2.0" in lowered)
        and any(marker in lowered for marker in ("nunchaku", "svdq-int4", "lightningv", "4steps"))
    ):
        return ARCH_QWEN_IMAGE_NUNCHAKU
    if "qwenimagepipeline" in compact or "qwen image" in normalized or "qwen-image" in lowered or "qwen2.0" in lowered:
        return ARCH_QWEN_IMAGE
    if (
        "sanavideopipeline" in compact
        or "sanaimagetovideopipeline" in compact
        or "sana-video" in lowered
        or "sana video" in normalized
    ):
        return ARCH_SANA_VIDEO
    if "sanapipeline" in compact or "sanasprintpipeline" in compact or "sana" in normalized:
        return ARCH_SANA
    if "z-image" in lowered or "zimage" in compact:
        return ARCH_Z_IMAGE
    if "flux.2" in lowered or "flux2" in compact or "klein" in normalized:
        return ARCH_FLUX2_KLEIN
    if (
        "stable diffusion 3.5" in normalized
        or "sd3.5" in text.lower()
        or "sd35" in normalized
        or "sd 3.5" in normalized
        or "sd3 large" in normalized
        or "sd3 medium" in normalized
    ):
        return ARCH_SD35
    if "sdxl" in normalized or "sd xl" in normalized or "xl base" in normalized:
        return ARCH_SDXL
    if "fluxkontextpipeline" in compact or "kontext" in normalized:
        return ARCH_FLUX_KONTEXT
    if "flux" in normalized:
        return ARCH_FLUX
    if "ltx" in normalized or "lightricks" in normalized:
        return "ltx"
    if "gemma" in normalized or "llm" in normalized:
        return "llm"
    if "wan" in normalized:
        return "wan"
    if "sd 1" in normalized or "sd1" in normalized or "1.5" in normalized or "v1 5" in normalized:
        return ARCH_SD15
    return "unknown"


def _metadata_architecture(metadata: dict[str, str], path: Path) -> str:
    text = " ".join(
        value
        for value in (
            metadata.get("modelspec.architecture", ""),
            metadata.get("modelspec.implementation", ""),
            metadata.get("ss_base_model_version", ""),
            metadata.get("ss_sd_model_name", ""),
            path.as_posix(),
        )
        if value
    )
    return _architecture_from_text(text)


def _recommended_subdir(family: str, architecture: str, filename: str = "") -> str:
    if family == "lora":
        if architecture == ARCH_SD35:
            return "Loras/SD3.5"
        if architecture == ARCH_SDXL:
            return "Loras/SDXL"
        if architecture == ARCH_FLUX:
            return "Loras/Flux"
        if architecture == ARCH_FLUX2_KLEIN:
            return "Loras/Flux2"
        if architecture == ARCH_Z_IMAGE:
            return "Loras/Z-Image"
        if architecture == "ltx":
            return "ltx/loras"
        if architecture == "wan":
            return "Loras/Wan"
        if architecture in {ARCH_SD15, ARCH_INPAINT}:
            return "Loras/SD15"
        return "Loras"
    if family == "runtime_asset":
        if architecture == ARCH_FLUX:
            suffix = Path(filename).suffix.lower()
            return "flux/GGUF" if suffix == ".gguf" else "flux/UNet"
        if architecture == ARCH_FLUX2_KLEIN:
            suffix = Path(filename).suffix.lower()
            return "flux2/GGUF" if suffix == ".gguf" else "flux2/UNet"
        if architecture == ARCH_Z_IMAGE:
            suffix = Path(filename).suffix.lower()
            return "z-image/GGUF" if suffix == ".gguf" else "z-image/UNet"
        if architecture == ARCH_QWEN_IMAGE_NUNCHAKU:
            return "qwen-image/Nunchaku"
        if architecture == ARCH_QWEN_IMAGE:
            return "qwen-image/Diffusers"
        if architecture == ARCH_SANA:
            return "sana/Diffusers"
        if architecture == ARCH_SANA_VIDEO:
            return "sana-video/Diffusers"
        if architecture == "ltx":
            lowered = filename.lower()
            if Path(filename).suffix.lower() == ".gguf":
                return "ltx/GGUF"
            if "upscaler" in lowered:
                return "ltx/upscalers"
            return "ltx/checkpoints"
        return "misc"
    if family == "checkpoint":
        if architecture == "unknown":
            return "models to sort"
        return "Stable-diffusion"
    if family == "vae":
        if architecture == ARCH_FLUX:
            return "flux/VAE"
        if architecture == "ltx":
            return "ltx/audio_vae" if "audio" in filename.lower() else "ltx/vae"
        return "VAE"
    if family == "embedding":
        return "embeddings"
    if family == "hypernetwork":
        return "hypernetworks"
    if family == "controlnet":
        return "controlnet"
    if family == "text_encoder":
        if architecture == ARCH_FLUX:
            return "flux/Textencoder"
        if architecture == ARCH_FLUX2_KLEIN:
            return "flux2/Components"
        if architecture == ARCH_Z_IMAGE:
            return "z-image/Components"
        if architecture == "ltx":
            return "ltx/text_encoder"
        return "Textencoder"
    if family == "face_embedding":
        return "reactor/faces"
    if family == "wan":
        return "wan/Safetensor"
    if family == "ltx":
        lowered = filename.lower()
        if Path(filename).suffix.lower() == ".gguf":
            return "ltx/GGUF"
        if "upscaler" in lowered:
            return "ltx/upscalers"
        if "lora" in lowered:
            return "ltx/loras"
        if "audio" in lowered and "vae" in lowered:
            return "ltx/audio_vae"
        if "vae" in lowered:
            return "ltx/vae"
        return "ltx/checkpoints"
    if family == "llm":
        return "LLM/GGUF" if Path(filename).suffix.lower() == ".gguf" else "LLM"
    return "misc"


CHECKPOINT_ARCHITECTURES = {ARCH_SD15, ARCH_INPAINT, ARCH_SDXL, ARCH_SDXL_INPAINT, ARCH_SD35}


def _header_family_architecture(path: Path) -> tuple[str, str, dict[str, str]] | None:
    try:
        info = read_model_info(path)
    except Exception:
        logger.debug("Could not inspect model header for %s", path, exc_info=True)
        return None

    identifiers: dict[str, str] = {
        "header_arch": info.arch,
        "header_role": info.role,
    }
    if info.precision:
        identifiers["header_precision"] = info.precision

    if info.role == ROLE_VAE:
        architecture = _architecture_from_text(f"{path.as_posix()} {info.display_name} {' '.join(info.raw_meta.values())}")
        return "vae", architecture, identifiers
    if info.arch in {ARCH_FLUX_TRANSFORMER}:
        return "runtime_asset", ARCH_FLUX, identifiers
    if info.arch == ARCH_FLUX2_KLEIN_TRANSFORMER:
        return "runtime_asset", ARCH_FLUX2_KLEIN, identifiers
    if info.arch == ARCH_Z_IMAGE_TRANSFORMER:
        return "runtime_asset", ARCH_Z_IMAGE, identifiers
    if info.arch in {ARCH_FLUX_LORA}:
        architecture = _architecture_from_text(f"{path.as_posix()} {info.display_name} {' '.join(info.raw_meta.values())}")
        return "lora", architecture if architecture != "unknown" else ARCH_FLUX, identifiers
    if info.arch in {ARCH_FLUX_VAE}:
        return "vae", ARCH_FLUX, identifiers
    if info.arch == ARCH_LTX_TRANSFORMER:
        return "runtime_asset", "ltx", identifiers
    if info.arch == ARCH_LTX_LORA:
        return "lora", "ltx", identifiers
    if info.arch in {ARCH_LTX_VAE, ARCH_LTX_AUDIO_VAE}:
        return "vae", "ltx", identifiers
    if info.arch == ARCH_T5XXL_ENCODER:
        return "text_encoder", ARCH_FLUX, identifiers
    if info.arch == ARCH_CLIP and _architecture_from_text(path.as_posix()) == ARCH_FLUX:
        return "text_encoder", ARCH_FLUX, identifiers

    if info.arch in {ARCH_WAN_TRANSFORMER, ARCH_WAN_TRANSFORMER_FP8}:
        return "wan", "wan", identifiers
    if info.arch == ARCH_WAN_LORA:
        return "lora", "wan", identifiers
    if info.arch == ARCH_WAN_VAE:
        return "vae", "wan", identifiers
    if info.arch == ARCH_UMT5_ENCODER:
        return "text_encoder", "wan", identifiers

    if info.role == ROLE_LORA:
        architecture = _architecture_from_text(f"{path.as_posix()} {info.display_name} {' '.join(info.raw_meta.values())}")
        return "lora", architecture, identifiers
    if info.role == ROLE_TEXT_ENCODER:
        architecture = _architecture_from_text(f"{path.as_posix()} {info.display_name} {' '.join(info.raw_meta.values())}")
        return "text_encoder", architecture, identifiers
    return None


def _matching_path_family(path: Path) -> str | None:
    parent_parts = [part.lower() for part in path.parts[:-1]]
    name = path.name.lower()
    if "reactor" in parent_parts and "faces" in parent_parts:
        return "face_embedding"
    if (
        any(
            part in {"controlnet", "controlnets", "control_net", "control-net", "sd_control_collection"}
            or part.startswith("controlnet-")
            for part in parent_parts
        )
        or name.startswith("control_")
        or "controlnet" in name
    ):
        return "controlnet"
    if any(part in {"textencoder", "text_encoder", "text-encoder", "clip", "clip_vision"} for part in parent_parts):
        return "text_encoder"
    if name.startswith(("clip_g", "clip_l", "t5xxl")):
        return "text_encoder"
    if any(part in {"embedding", "embeddings"} for part in parent_parts):
        return "embedding"
    if any(part in {"hypernetwork", "hypernetworks"} for part in parent_parts):
        return "hypernetwork"
    if any(part in {"diffusion_models", "unet", "transformer"} for part in parent_parts):
        return "runtime_asset"
    if any(part in {"vae", "vae-approx"} for part in parent_parts) or name.endswith((".vae.safetensors", ".vae.ckpt", ".vae.pt")):
        return "vae"
    if any(part in {"llm", "llms"} for part in parent_parts):
        return "llm"
    if any(part == "wan" or part.startswith("wan_") or part.startswith("wan-") for part in parent_parts) or "wan" in name:
        return "wan"
    if any(part == "ltx" or part.startswith("ltx_") or part.startswith("ltx-") for part in parent_parts) or "ltx" in name:
        return "ltx"
    if any(part in {"lora", "loras"} for part in parent_parts):
        return "lora"
    return None


def _read_model_index(path: Path) -> dict:
    try:
        return json.loads((path / "model_index.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def classify_model_dir(path: Path, roots: list[Path]) -> ModelInventoryRecord | None:
    if not path.is_dir() or not (path / "model_index.json").is_file():
        return None

    model_index = _read_model_index(path)
    class_name = str(model_index.get("_class_name") or "")
    text = f"{path.name} {_relative_path_text(path, roots)} {class_name}"
    family = "checkpoint"
    architecture = _architecture_from_text(text)
    lowered = text.lower()
    path_parts = {part.lower() for part in path.parts}
    identifiers = {"model_index": class_name or "model_index.json"}

    # A full pipeline export can still be a ControlNet/adapter pipeline (e.g.
    # "StableDiffusionXLControlNetPipeline") rather than a plain txt2img checkpoint.
    # These have model_index.json but the wrong shape for the generic checkpoint
    # loader, so they must be tagged "controlnet" here - not just caught later as
    # a load-time crash. See aiwf/infrastructure/diffusers/backend.py load guard.
    if "controlnet" in class_name.lower() or any(
        part in {"controlnet", "controlnets", "control_net", "control-net", "sd_control_collection"}
        or part.startswith("controlnet-")
        for part in path_parts
    ):
        return ModelInventoryRecord(
            path=str(path.resolve()),
            filename=path.name,
            family="controlnet",
            architecture=architecture,
            current_subdir=_relative_subdir(path, roots),
            recommended_subdir=_recommended_subdir("controlnet", architecture, path.name),
            should_move=False,
            header_identifiers={**identifiers, "pipeline_marker": "controlnet pipeline"},
            metadata={},
        )

    if "wan" in lowered:
        family = "wan"
        architecture = "wan"
    elif "ltx" in lowered:
        family = "runtime_asset"
        architecture = "ltx"
    elif "components" in path_parts and architecture in {ARCH_FLUX2_KLEIN, ARCH_Z_IMAGE}:
        family = "text_encoder"
    elif architecture in {
        ARCH_FLUX2_KLEIN,
        ARCH_Z_IMAGE,
        ARCH_FLUX_KONTEXT,
        ARCH_QWEN_IMAGE,
        ARCH_QWEN_IMAGE_NUNCHAKU,
        ARCH_SANA,
        ARCH_SANA_VIDEO,
    }:
        family = "runtime_asset"
    elif "flux" in lowered:
        family = "runtime_asset"
        architecture = ARCH_FLUX
    elif "stablediffusionxl" in class_name.lower() or "stable-diffusion-xl" in lowered:
        architecture = ARCH_SDXL
    elif "stablediffusion3" in class_name.lower() or architecture == ARCH_SD35:
        architecture = ARCH_SD35
    elif "stablediffusioninpaint" in class_name.lower() or "inpaint" in lowered:
        architecture = ARCH_INPAINT
    elif "stablediffusion" in class_name.lower():
        architecture = ARCH_SD15
    else:
        family = "runtime_asset"

    current_subdir = _relative_subdir(path, roots)
    recommended = _recommended_subdir(family, architecture, path.name)
    return ModelInventoryRecord(
        path=str(path.resolve()),
        filename=path.name,
        family=family,
        architecture=architecture,
        current_subdir=current_subdir,
        recommended_subdir=recommended,
        should_move=current_subdir.replace("\\", "/").lower() != recommended.lower(),
        header_identifiers=identifiers,
        metadata={},
    )


def classify_model_file(path: Path, roots: list[Path]) -> ModelInventoryRecord | None:
    if not path.is_file() or path.suffix.lower() not in MODEL_EXTENSIONS:
        return None

    metadata = read_safetensors_metadata(path)
    metadata_text = _metadata_text(metadata)
    path_family = _matching_path_family(path)
    family = path_family or "unknown"
    architecture = _metadata_architecture(metadata, path)
    header_match = _header_family_architecture(path)
    identifiers: dict[str, str] = {}
    shapes: dict[str, list[int]] = {}

    if path.suffix.lower() == ".safetensors":
        try:
            shapes = _safetensors_tensor_shapes(path)
        except Exception:
            logger.debug("Could not inspect safetensors tensor header for %s", path, exc_info=True)

    if shapes:
        keys = set(shapes)
        if {"embedding", "bbox", "kps"} & keys and len(shapes) <= 12:
            family = "face_embedding"
            identifiers["tensor_marker"] = "face embedding keys"
        elif looks_like_lora_weights(path):
            if family == "controlnet":
                identifiers["tensor_marker"] = "controlnet lora"
            else:
                family = "lora"
                identifiers["tensor_marker"] = "lora_up/lora_down"
        elif UNET_INPUT_KEY in shapes or "conditioner.embedders.1.model.ln_final.weight" in keys:
            family = "checkpoint"
            architecture = infer_architecture_from_shapes(shapes, filename=path.name)
            identifiers["tensor_marker"] = "diffusion checkpoint"

    if header_match and family not in {"controlnet", "face_embedding"}:
        header_family, header_architecture, header_identifiers = header_match
        if header_family in {"runtime_asset", "text_encoder", "vae", "wan"} or family in {
            "unknown",
            "lora",
            "vae",
            "text_encoder",
            "runtime_asset",
            "wan",
        }:
            family = header_family
            if header_architecture and header_architecture != "unknown":
                architecture = header_architecture
            identifiers.update(header_identifiers)

    if family == "unknown" and architecture in CHECKPOINT_ARCHITECTURES and path.suffix.lower() in {
        ".ckpt",
        ".pt",
        ".safetensors",
    }:
        family = "checkpoint"
        identifiers["metadata_marker"] = "checkpoint architecture"

    if family == "unknown":
        if metadata.get("ss_network_module") or "lora" in metadata_text:
            family = "lora"
            identifiers["metadata_marker"] = "ss_network_module/lora"
        elif "controlnet" in metadata_text:
            family = "controlnet"
            identifiers["metadata_marker"] = "controlnet"
        elif "vae" in metadata_text:
            family = "vae"
            identifiers["metadata_marker"] = "vae"
        elif architecture == ARCH_QWEN_IMAGE_NUNCHAKU and path.suffix.lower() == ".safetensors":
            family = "runtime_asset"
            identifiers["filename_marker"] = "qwen nunchaku transformer"
        elif path.suffix.lower() == ".gguf" and "wan" in path.name.lower():
            family = "wan"
            architecture = "wan"
            identifiers["filename_marker"] = "wan gguf"
        elif path.suffix.lower() == ".gguf" and architecture in {ARCH_FLUX, ARCH_FLUX2_KLEIN, ARCH_Z_IMAGE}:
            family = "runtime_asset"
            identifiers["filename_marker"] = f"{architecture} gguf"
        elif architecture == "ltx" and path.suffix.lower() == ".safetensors":
            family = "ltx"
            identifiers["filename_marker"] = "ltx safetensors"

    if family == "unknown" and path.suffix.lower() in {".ckpt", ".pt", ".safetensors"}:
        family = "checkpoint"
        architecture = detect_checkpoint_architecture(path)
        identifiers["fallback_marker"] = "checkpoint extension"

    if architecture == "unknown" and family == "checkpoint":
        architecture = detect_checkpoint_architecture(path)
    if architecture == "unknown" and family == "lora":
        architecture = _architecture_from_text(f"{path.name} {metadata_text}")

    current_subdir = _relative_subdir(path, roots)
    recommended = _recommended_subdir(family, architecture, path.name)
    important_metadata = {
        key: value
        for key, value in metadata.items()
        if key.startswith("ss_") or key.startswith("modelspec.")
    }
    return ModelInventoryRecord(
        path=str(path.resolve()),
        filename=path.name,
        family=family,
        architecture=architecture,
        current_subdir=current_subdir,
        recommended_subdir=recommended,
        should_move=current_subdir.replace("\\", "/").lower() != recommended.lower(),
        header_identifiers=identifiers,
        metadata=important_metadata,
    )


def scan_model_inventory(flags: RuntimeFlags) -> list[ModelInventoryRecord]:
    roots = model_inventory_roots(flags)
    seen: set[str] = set()
    records: list[ModelInventoryRecord] = []
    for root in roots:
        for current, dir_names, file_names in os.walk(root):
            dir_names.sort(key=str.lower)
            file_names.sort(key=str.lower)
            path = Path(current)
            try:
                key = os.path.normcase(str(path.resolve()))
            except OSError:
                continue
            if key in seen:
                continue
            record = classify_model_dir(path, roots)
            if record is not None:
                seen.add(key)
                records.append(record)
                dir_names[:] = []
                continue
            for filename in file_names:
                file_path = path / filename
                try:
                    file_key = os.path.normcase(str(file_path.resolve()))
                except OSError:
                    continue
                if file_key in seen:
                    continue
                file_record = classify_model_file(file_path, roots)
                if file_record is None:
                    continue
                seen.add(file_key)
                records.append(file_record)
    records.sort(key=lambda item: (item.family, item.architecture, item.filename.lower()))
    return records


def write_model_inventory(flags: RuntimeFlags, records: list[ModelInventoryRecord]) -> Path | None:
    path = inventory_path(flags)
    payload = {
        "schema_version": MODEL_INVENTORY_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "roots": [str(root) for root in model_inventory_roots(flags)],
        "assets": [asdict(record) for record in records],
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
        return path
    except OSError:
        logger.debug("Could not write model inventory to %s", path, exc_info=True)
        return None


_SESSION_INVENTORY: dict[str, list[ModelInventoryRecord]] = {}


def _fast_roots_signature(flags: RuntimeFlags) -> str:
    parts: list[str] = []
    for root in model_inventory_roots(flags):
        try:
            stat = root.stat()
            parts.append(f"{os.path.normcase(str(root))}:{stat.st_mtime_ns}")
        except OSError:
            parts.append(os.path.normcase(str(root)))
    return "|".join(sorted(parts))


def _records_from_payload(payload: dict) -> list[ModelInventoryRecord]:
    records: list[ModelInventoryRecord] = []
    for item in payload.get("assets") or []:
        if not isinstance(item, dict):
            continue
        records.append(ModelInventoryRecord(**item))
    return records


def load_model_inventory(flags: RuntimeFlags) -> list[ModelInventoryRecord] | None:
    path = inventory_path(flags)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("schema_version") != MODEL_INVENTORY_VERSION:
        return None
    stored_roots = [str(root) for root in payload.get("roots") or []]
    current_roots = [str(root) for root in model_inventory_roots(flags)]
    if sorted(stored_roots) != sorted(current_roots):
        return None
    return _records_from_payload(payload)


def invalidate_model_inventory_cache() -> None:
    _SESSION_INVENTORY.clear()


def get_model_inventory(flags: RuntimeFlags, *, force_rescan: bool = False) -> list[ModelInventoryRecord]:
    signature = _fast_roots_signature(flags)
    if not force_rescan:
        session_cached = _SESSION_INVENTORY.get(signature)
        if session_cached is not None:
            return session_cached
        disk_cached = load_model_inventory(flags)
        if disk_cached is not None:
            _SESSION_INVENTORY[signature] = disk_cached
            return disk_cached

    records = scan_model_inventory(flags)
    write_model_inventory(flags, records)
    # Writing model_inventory.json lives inside the models dir, which bumps that
    # dir's mtime and would change the roots signature — instantly invalidating
    # the entry we just made and forcing a full rescan on the next call. Cache
    # the result under the post-write signature too so subsequent calls hit.
    _SESSION_INVENTORY[signature] = records
    _SESSION_INVENTORY[_fast_roots_signature(flags)] = records
    logger.info("Indexed %d local model asset(s)", len(records))
    return records


def scan_and_write_model_inventory(flags: RuntimeFlags) -> list[ModelInventoryRecord]:
    return get_model_inventory(flags, force_rescan=True)
