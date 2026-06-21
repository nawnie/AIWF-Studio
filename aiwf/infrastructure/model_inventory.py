from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from aiwf.core.config.settings import RuntimeFlags
from aiwf.infrastructure.diffusers.model_arch import (
    ARCH_INPAINT,
    ARCH_SD15,
    ARCH_SD35,
    ARCH_SDXL,
    ARCH_SDXL_INPAINT,
    UNET_INPUT_KEY,
    detect_checkpoint_architecture,
    infer_architecture_from_shapes,
    looks_like_lora_weights,
    _safetensors_tensor_shapes,
)
from aiwf.infrastructure.model_header import (
    ARCH_CLIP,
    ARCH_FLUX_LORA,
    ARCH_FLUX_TRANSFORMER,
    ARCH_FLUX_VAE,
    ARCH_T5XXL_ENCODER,
    ARCH_UMT5_ENCODER,
    ARCH_WAN_LORA,
    ARCH_WAN_TRANSFORMER,
    ARCH_WAN_TRANSFORMER_FP8,
    ARCH_WAN_VAE,
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
    return flags.resolved_models_dir() / "model_inventory.json"


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


def _metadata_text(metadata: dict[str, str]) -> str:
    return " ".join(f"{key} {value}" for key, value in metadata.items()).lower()


def _architecture_from_text(text: str) -> str:
    normalized = text.lower().replace("_", " ").replace("-", " ")
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
    if "flux" in normalized:
        return "flux"
    if "ltx" in normalized or "lightricks" in normalized:
        return "ltx"
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
        if architecture == "flux":
            return "Loras/Flux"
        if architecture == "ltx":
            return "ltx/loras"
        if architecture == "wan":
            return "Loras/Wan"
        if architecture in {ARCH_SD15, ARCH_INPAINT}:
            return "Loras/SD15"
        return "Loras"
    if family == "runtime_asset":
        if architecture == "flux":
            suffix = Path(filename).suffix.lower()
            return "flux/GGUF" if suffix == ".gguf" else "flux/UNet"
        if architecture == "ltx":
            lowered = filename.lower()
            if "upscaler" in lowered:
                return "ltx/upscalers"
            return "ltx/checkpoints"
        return "misc"
    if family == "checkpoint":
        return "Stable-diffusion"
    if family == "vae":
        if architecture == "flux":
            return "flux/VAE"
        return "VAE"
    if family == "embedding":
        return "embeddings"
    if family == "hypernetwork":
        return "hypernetworks"
    if family == "controlnet":
        return "controlnet"
    if family == "text_encoder":
        if architecture == "flux":
            return "flux/Textencoder"
        if architecture == "ltx":
            return "ltx/text_encoder"
        return "Textencoder"
    if family == "face_embedding":
        return "reactor/faces"
    if family == "wan":
        return "wan/Safetensor"
    if family == "ltx":
        lowered = filename.lower()
        if "upscaler" in lowered:
            return "ltx/upscalers"
        if "lora" in lowered:
            return "ltx/loras"
        return "ltx/checkpoints"
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

    if info.arch in {ARCH_FLUX_TRANSFORMER}:
        return "runtime_asset", "flux", identifiers
    if info.arch in {ARCH_FLUX_LORA}:
        return "lora", "flux", identifiers
    if info.arch in {ARCH_FLUX_VAE}:
        return "vae", "flux", identifiers
    if info.arch == ARCH_T5XXL_ENCODER:
        return "text_encoder", "flux", identifiers
    if info.arch == ARCH_CLIP and _architecture_from_text(path.as_posix()) == "flux":
        return "text_encoder", "flux", identifiers

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
    if info.role == ROLE_VAE:
        architecture = _architecture_from_text(f"{path.as_posix()} {info.display_name} {' '.join(info.raw_meta.values())}")
        return "vae", architecture, identifiers
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
    text = f"{path.name} {' '.join(path.parts)} {class_name}"
    family = "checkpoint"
    architecture = _architecture_from_text(text)
    lowered = text.lower()
    identifiers = {"model_index": class_name or "model_index.json"}

    if "wan" in lowered:
        family = "wan"
        architecture = "wan"
    elif "ltx" in lowered:
        family = "runtime_asset"
        architecture = "ltx"
    elif "flux" in lowered:
        family = "runtime_asset"
        architecture = "flux"
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
        elif path.suffix.lower() == ".gguf" and "wan" in path.name.lower():
            family = "wan"
            architecture = "wan"
            identifiers["filename_marker"] = "wan gguf"
        elif path.suffix.lower() == ".gguf" and architecture == "flux":
            family = "runtime_asset"
            identifiers["filename_marker"] = "flux gguf"
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
        try:
            paths = [root, *sorted(root.rglob("*"), key=lambda p: str(p).lower())]
        except OSError:
            continue
        for path in paths:
            key = os.path.normcase(str(path.resolve()))
            if key in seen:
                continue
            record = classify_model_dir(path, roots) if path.is_dir() else classify_model_file(path, roots)
            if record is None:
                continue
            seen.add(key)
            records.append(record)
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


def scan_and_write_model_inventory(flags: RuntimeFlags) -> list[ModelInventoryRecord]:
    records = scan_model_inventory(flags)
    write_model_inventory(flags, records)
    logger.info("Indexed %d local model asset(s)", len(records))
    return records
