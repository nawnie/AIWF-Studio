from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from aiwf.core.config.settings import RuntimeFlags
from aiwf.infrastructure.diffusers.model_arch import (
    ARCH_FLUX,
    ARCH_FLUX_FILL,
    ARCH_FLUX_KONTEXT,
    ARCH_FLUX2,
    ARCH_FLUX2_KLEIN,
    ARCH_LONGCAT_IMAGE,
    ARCH_ANIMA,
    ARCH_INPAINT,
    ARCH_KREA2,
    ARCH_QWEN_IMAGE,
    ARCH_QWEN_IMAGE_EDIT,
    ARCH_QWEN_IMAGE_EDIT_PLUS,
    ARCH_QWEN_IMAGE_NUNCHAKU,
    ARCH_SANA,
    ARCH_SANA_VIDEO,
    ARCH_SD15,
    ARCH_SD35,
    ARCH_SDXL,
    ARCH_SDXL_INPAINT,
    ARCH_Z_IMAGE,
    ARCH_UNKNOWN,
    UNET_INPUT_KEY,
    detect_checkpoint_architecture,
    infer_architecture_from_shapes,
    is_torchscript_archive,
    looks_like_lora_weights,
    _safetensors_tensor_shapes,
)
from aiwf.infrastructure.model_header import (
    ARCH_CLIP,
    ARCH_FLUX2_KLEIN_LORA,
    ARCH_FLUX2_KLEIN_TRANSFORMER,
    ARCH_FLUX2_LORA,
    ARCH_FLUX2_TRANSFORMER,
    ARCH_FLUX_KONTEXT_LORA,
    ARCH_FLUX_KONTEXT_TRANSFORMER,
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
    _has_flux2_klein_marker,
    ROLE_LORA,
    ROLE_TEXT_ENCODER,
    ROLE_VAE,
    read_model_info,
)
from aiwf.infrastructure.safetensors_metadata import (
    read_safetensors_metadata,
    safetensors_file_is_structurally_valid,
)

logger = logging.getLogger(__name__)

MODEL_EXTENSIONS = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx"}
# Bump this whenever architecture classification logic changes: the disk
# cache stores classified records, so stale caches would otherwise keep
# serving old (wrong) architectures to every picker after an update.
MODEL_INVENTORY_VERSION = 14
_WEAK_ONLY_PLACEMENT_MARKERS = {"fallback_marker"}


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


def model_asset_placement_confidence(record: ModelInventoryRecord) -> tuple[bool, str]:
    """Share the sorter's confidence gate with non-mutating placement previews."""
    if record.family == "unknown":
        return False, "header did not identify a known model type"
    if record.recommended_subdir.lower() in {"", "misc", "models to sort"}:
        return False, "no specific destination for this model type"
    markers = set(record.header_identifiers)
    if markers and markers <= _WEAK_ONLY_PLACEMENT_MARKERS:
        return False, "only matched by file extension, not by header content"
    return True, ""


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
    # Preserve the model-root-relative spelling through intentional junctions
    # (for example, models/LLM -> a shared LLM library). resolve() remains the
    # authority for file access and placement safety, but resolving first
    # erases the in-root modality marker needed for accurate labels.
    for root in sorted((root.absolute() for root in roots), key=lambda p: len(str(p)), reverse=True):
        try:
            rel = path.absolute().parent.relative_to(root)
        except ValueError:
            continue
        return "" if str(rel) == "." else rel.as_posix()
    parent = path.parent.resolve()
    for root in sorted((root.resolve() for root in roots), key=lambda p: len(str(p)), reverse=True):
        try:
            rel = parent.relative_to(root)
        except ValueError:
            continue
        return "" if str(rel) == "." else rel.as_posix()
    return parent.as_posix()


def _relative_path_text(path: Path, roots: list[Path]) -> str:
    for root in sorted((root.absolute() for root in roots), key=lambda p: len(str(p)), reverse=True):
        try:
            return path.absolute().relative_to(root).as_posix()
        except ValueError:
            continue
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
    if "krea-2" in lowered or "krea2" in compact or "krea 2" in normalized or "krea2pipeline" in compact:
        return ARCH_KREA2
    if (
        "circlestone-labs/anima" in text.lower().replace("\\", "/")
        or "animapipeline" in compact
        or re.search(r"(^|[\s/._-])anima($|[\s/._-])", text.lower())
    ):
        return ARCH_ANIMA
    if (
        ("qwenimagepipeline" in compact or "qwen image" in normalized or "qwen-image" in lowered or "qwen2.0" in lowered)
        and any(marker in lowered for marker in ("nunchaku", "svdq-int4", "lightningv", "4steps"))
    ):
        return ARCH_QWEN_IMAGE_NUNCHAKU
    if (
        "qwenimageeditpluspipeline" in compact
        or "qwen-image-edit-plus" in lowered
    ):
        return ARCH_QWEN_IMAGE_EDIT_PLUS
    if (
        "qwenimageeditpipeline" in compact
        or "qwen-image-edit" in lowered
    ):
        return ARCH_QWEN_IMAGE_EDIT
    if (
        "qwenimagepipeline" in compact
        or "qwenimage21pipeline" in compact
        or "qwen image" in normalized
        or "qwen-image" in lowered
        or "qwen2.0" in lowered
    ):
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
    if "longcat" in lowered:
        return ARCH_LONGCAT_IMAGE
    if "z-image" in lowered or "zimage" in compact:
        return ARCH_Z_IMAGE
    f2k_tokens = [Path(token.replace("\\", "/")).name for token in re.split(r"\s+", text) if token]
    has_f2k_marker = any(_has_flux2_klein_marker(token) for token in f2k_tokens)
    if "klein" in normalized or has_f2k_marker:
        return ARCH_FLUX2_KLEIN
    if "fluxkontextpipeline" in compact or "kontext" in normalized:
        return ARCH_FLUX_KONTEXT
    if "flux.2" in lowered or "flux2" in compact:
        return ARCH_FLUX2
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
    if "flux" in normalized and "fill" in normalized:
        return ARCH_FLUX_FILL
    if "flux" in normalized:
        return ARCH_FLUX
    if compact.startswith("wan") and compact.endswith("pipeline"):
        return "wan"
    if "ltx" in normalized or "lightricks" in normalized:
        return "ltx"
    if "gemma" in normalized or "llm" in normalized:
        return "llm"
    # Require a Wan token/prefix (including common versioned filenames such
    # as wan2.2_*). A substring match labels unrelated names like
    # "want_to_test.safetensors" as Wan and can send them through auto-sort.
    if re.search(r"(?<![a-z0-9])wan(?:[._ -]?\d+(?:\.\d+)*)?(?![a-z0-9])", normalized):
        return "wan"
    if "sd 1" in normalized or "sd1" in normalized or "1.5" in normalized or "v1 5" in normalized:
        return ARCH_SD15
    return "unknown"


def _has_flux_component_layout(model_layout_context: str) -> bool:
    """Recognize the canonical Flux component directory, not incidental names."""
    return any(
        segment.strip().casefold() == "flux"
        for segment in re.split(r"[\\/]+", str(model_layout_context or ""))
    )


def _metadata_architecture(
    metadata: dict[str, str],
    path: Path,
    *,
    model_layout_context: str = "",
) -> str:
    text = " ".join(
        value
        for value in (
            metadata.get("modelspec.architecture", ""),
            metadata.get("modelspec.implementation", ""),
            metadata.get("ss_base_model_version", ""),
            metadata.get("ss_sd_model_name", ""),
            path.name,
            model_layout_context,
        )
        if value
    )
    return _architecture_from_text(text)


def _recommended_subdir(family: str, architecture: str, filename: str = "") -> str:
    if family == "invalid_asset":
        return "models to sort"
    if family == "preprocessor":
        return "ControlNet/Annotators"
    if family == "ip_adapter":
        return "ipadapter"
    if family == "lora":
        if architecture == ARCH_SD35:
            return "Loras/SD3.5"
        if architecture == ARCH_SDXL:
            return "Loras/SDXL"
        if architecture == ARCH_FLUX:
            return "Loras/Flux"
        if architecture == ARCH_FLUX_KONTEXT:
            return "Loras/FluxKontext"
        if architecture == ARCH_FLUX2_KLEIN:
            return "Loras/Flux2"
        if architecture == ARCH_Z_IMAGE:
            return "Loras/Z-Image"
        if architecture == ARCH_KREA2:
            return "Loras/Krea2"
        if architecture == ARCH_ANIMA:
            return "Loras/Anima"
        if architecture == "ltx":
            return "ltx/loras"
        if architecture == "wan":
            return "Loras/Wan"
        if architecture in {ARCH_SD15, ARCH_INPAINT}:
            return "Loras/SD15"
        return "Loras"
    if family == "runtime_asset":
        if architecture == "rife":
            return "frame_interpolation"
        if architecture == ARCH_FLUX:
            suffix = Path(filename).suffix.lower()
            return "flux/GGUF" if suffix == ".gguf" else "flux/UNet"
        if architecture == ARCH_FLUX_KONTEXT:
            suffix = Path(filename).suffix.lower()
            return "flux/GGUF" if suffix == ".gguf" else "flux/UNet"
        if architecture == ARCH_FLUX2_KLEIN:
            suffix = Path(filename).suffix.lower()
            return "flux2/GGUF" if suffix == ".gguf" else "flux2/UNet"
        if architecture == ARCH_FLUX2:
            return "models to sort"
        if architecture == ARCH_LONGCAT_IMAGE:
            return "longcat"
        if architecture == ARCH_Z_IMAGE:
            suffix = Path(filename).suffix.lower()
            return "z-image/GGUF" if suffix == ".gguf" else "z-image/UNet"
        if architecture == ARCH_KREA2:
            return "krea2/UNet" if Path(filename).suffix.lower() else "krea2/Diffusers"
        if architecture == ARCH_ANIMA:
            return "anima/UNet" if Path(filename).suffix.lower() else "anima/Diffusers"
        if architecture == ARCH_QWEN_IMAGE_NUNCHAKU:
            return "qwen-image/Nunchaku"
        if architecture == ARCH_QWEN_IMAGE:
            return "qwen-image/Diffusers"
        if architecture in {ARCH_QWEN_IMAGE_EDIT, ARCH_QWEN_IMAGE_EDIT_PLUS}:
            return "models to sort"
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
        if architecture == ARCH_KREA2:
            return "krea2/Textencoder"
        if architecture == ARCH_ANIMA:
            return "anima/Textencoder"
        if architecture == "ltx":
            return "ltx/text_encoder"
        return "Textencoder"
    if family == "face_embedding":
        return "reactor/faces"
    if family == "wan":
        return "wan/Safetensor"
    if family == "upscaler":
        return "upscale_models"
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


def _recommended_diffusers_subdir(architecture: str, path: Path, roots: list[Path]) -> str:
    if architecture in {ARCH_FLUX2_KLEIN, ARCH_Z_IMAGE}:
        # Keep a pipeline already stored in a Diffusers tree in that tree even
        # when its transformer download is incomplete. Transformer absence is
        # expected for a support-components bundle, but is also common for an
        # interrupted full-pipeline download.
        layout_parts = {
            part.casefold()
            for part in _relative_path_text(path, roots).replace("\\", "/").split("/")
        }
        if "diffusers" in layout_parts:
            return "flux2/Diffusers" if architecture == ARCH_FLUX2_KLEIN else "z-image/Diffusers"
        if "components" in layout_parts:
            return "flux2/Components" if architecture == ARCH_FLUX2_KLEIN else "z-image/Components"
        transformer = path / "transformer"
        patterns = (
            "diffusion_pytorch_model.safetensors",
            "diffusion_pytorch_model*.safetensors",
            "diffusion_pytorch_model.bin",
            "diffusion_pytorch_model*.bin",
        )
        has_transformer_weights = False
        for pattern in patterns:
            for candidate in transformer.glob(pattern):
                try:
                    if candidate.is_file() and candidate.stat().st_size > 0:
                        has_transformer_weights = True
                        break
                except OSError:
                    continue
            if has_transformer_weights:
                break
        if not has_transformer_weights:
            return "flux2/Components" if architecture == ARCH_FLUX2_KLEIN else "z-image/Components"
    return {
        ARCH_FLUX: "flux/Diffusers",
        ARCH_FLUX_KONTEXT: "flux/Components/FLUX.1-Kontext-dev",
        ARCH_FLUX2_KLEIN: "flux2/Diffusers",
        ARCH_Z_IMAGE: "z-image/Diffusers",
        ARCH_QWEN_IMAGE: "qwen-image/Diffusers",
        ARCH_SANA: "sana/Diffusers",
        ARCH_SANA_VIDEO: "sana-video/Diffusers",
        ARCH_KREA2: "krea2/Diffusers",
        ARCH_ANIMA: "anima/Diffusers",
        ARCH_LONGCAT_IMAGE: "longcat/Diffusers",
    }.get(architecture, "models to sort")


CHECKPOINT_ARCHITECTURES = {ARCH_SD15, ARCH_INPAINT, ARCH_SDXL, ARCH_SDXL_INPAINT, ARCH_SD35}


def _header_family_architecture(
    path: Path,
    *,
    model_layout_context: str = "",
) -> tuple[str, str, dict[str, str]] | None:
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

    local_context = f"{path.name} {model_layout_context}"
    if info.role == ROLE_VAE:
        known_vae_architecture = {
            ARCH_FLUX_VAE: ARCH_FLUX,
            ARCH_WAN_VAE: "wan",
            ARCH_LTX_VAE: "ltx",
            ARCH_LTX_AUDIO_VAE: "ltx",
        }.get(info.arch)
        architecture = known_vae_architecture or _architecture_from_text(
            f"{local_context} {info.display_name} {' '.join(info.raw_meta.values())}"
        )
        return "vae", architecture, identifiers
    details = f"{local_context} {info.display_name} {' '.join(info.raw_meta.values())}"
    if info.role == ROLE_LORA:
        # Adapter metadata and the adapter filename are more authoritative
        # than the folder it happens to be stored in (users often keep
        # adapters in the wrong family folder before sorting them).
        if info.arch == ARCH_FLUX2_KLEIN_LORA:
            architecture = ARCH_FLUX2_KLEIN
        elif info.arch == ARCH_FLUX_KONTEXT_LORA:
            architecture = ARCH_FLUX_KONTEXT
        elif info.arch in {ARCH_FLUX_LORA, ARCH_FLUX_TRANSFORMER}:
            architecture = ARCH_FLUX
        else:
            adapter_details = f"{info.display_name} {info.filename} {' '.join(info.raw_meta.values())}"
            architecture = _architecture_from_text(adapter_details)
        if architecture == "unknown":
            parts = path.parent.parts
            lora_root = next(
                (index for index in range(len(parts) - 1, -1, -1) if parts[index].casefold() in {"lora", "loras"}),
                None,
            )
            folder_context = Path(*parts[lora_root:]).as_posix() if lora_root is not None else path.parent.name
            architecture = _architecture_from_text(folder_context)
        if architecture == "unknown" and info.arch in {ARCH_FLUX_TRANSFORMER, ARCH_FLUX_LORA}:
            architecture = ARCH_FLUX
        return "lora", architecture, identifiers
    if info.arch == ARCH_LONGCAT_IMAGE:
        return "runtime_asset", ARCH_LONGCAT_IMAGE, identifiers
    if info.arch == ARCH_FLUX_KONTEXT_TRANSFORMER:
        return "runtime_asset", ARCH_FLUX_KONTEXT, identifiers
    if info.arch in {ARCH_FLUX_TRANSFORMER}:
        architecture = _architecture_from_text(details)
        if architecture in {ARCH_FLUX_KONTEXT, ARCH_FLUX2, ARCH_FLUX2_KLEIN, ARCH_Z_IMAGE, ARCH_LONGCAT_IMAGE}:
            return "runtime_asset", architecture, identifiers
        # Flux.1-Fill shares the transformer header signature with base Flux;
        # only the widened 384-channel image projection tells them apart, and
        # it matters because Fill is inpaint-only.
        if "fill" in path.name.lower() or detect_checkpoint_architecture(path) == ARCH_FLUX_FILL:
            return "runtime_asset", ARCH_FLUX_FILL, identifiers
        return "runtime_asset", ARCH_FLUX, identifiers
    if info.arch == ARCH_FLUX2_TRANSFORMER:
        return "runtime_asset", ARCH_FLUX2, identifiers
    if info.arch == ARCH_FLUX2_KLEIN_TRANSFORMER:
        return "runtime_asset", ARCH_FLUX2_KLEIN, identifiers
    if info.arch == ARCH_Z_IMAGE_TRANSFORMER:
        return "runtime_asset", ARCH_Z_IMAGE, identifiers
    if info.arch in {ARCH_FLUX_LORA}:
        architecture = _architecture_from_text(f"{local_context} {info.display_name} {' '.join(info.raw_meta.values())}")
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
    if info.arch == ARCH_CLIP:
        # Only use nearby model-layout folders for this family hint. Looking
        # at the full absolute path lets unrelated workspace names (for
        # example, a temp folder named "flux-tests") misclassify any CLIP
        # encoder as a Flux component and send it to the wrong destination.
        if _has_flux_component_layout(model_layout_context):
            return "text_encoder", ARCH_FLUX, identifiers
        # CLIP's tensor structure identifies the encoder role, but generic
        # family-name matching on its parent path is too weak to infer Flux.
        return "text_encoder", ARCH_UNKNOWN, identifiers

    if info.arch in {ARCH_WAN_TRANSFORMER, ARCH_WAN_TRANSFORMER_FP8}:
        return "wan", "wan", identifiers
    if info.arch == ARCH_WAN_LORA:
        return "lora", "wan", identifiers
    if info.arch == ARCH_WAN_VAE:
        return "vae", "wan", identifiers
    if info.arch == ARCH_UMT5_ENCODER:
        return "text_encoder", "wan", identifiers

    if info.role == ROLE_LORA:
        architecture = _architecture_from_text(f"{local_context} {info.display_name} {' '.join(info.raw_meta.values())}")
        return "lora", architecture, identifiers
    if info.role == ROLE_TEXT_ENCODER:
        architecture = _architecture_from_text(f"{local_context} {info.display_name} {' '.join(info.raw_meta.values())}")
        return "text_encoder", architecture, identifiers
    return None


def _matching_path_family(path: Path, roots: list[Path]) -> str | None:
    # Only model-root-relative folders describe an asset's role. Ancestors
    # above a configured root can be named after another family (for example
    # F:\\Shared\\controlnet\\models) and must not relabel every child.
    relative_parts = _relative_path_text(path, roots).replace("\\", "/").split("/")
    parent_parts = [part.lower() for part in relative_parts[:-1]]
    name = path.name.lower()
    if any(part in {"llm", "llms", "language models"} for part in parent_parts):
        return "llm"
    if any(part in {"audio", "audio_models", "musicgen", "mmaudio"} for part in parent_parts):
        return "audio"
    if (
        any(part in {"upscale_models", "upscalers", "upscaler"} for part in parent_parts)
        and not any("ltx" in part for part in (*parent_parts, name))
    ):
        return "upscaler"
    if "reactor" in parent_parts and "faces" in parent_parts:
        return "face_embedding"
    if any(part in {"ipadapter", "ip_adapter", "ip-adapter", "ip_adapters", "ip-adapters"} for part in parent_parts):
        return "ip_adapter"
    if "lama" in name:
        return "lama"
    preprocessor_markers = (
        "oneformer", "body_pose_model", "hand_pose_model", "dpt_hybrid", "mlsd_",
        "bsds500", "pidinet", "upernet", "scannet", "controlnethed",
    )
    if (
        any(part in {"annotator", "annotators", "preprocessor", "preprocessors"} for part in parent_parts)
        or any(marker in name for marker in preprocessor_markers)
    ):
        return "preprocessor"
    if (
        any(part in {"frame_interpolation", "frame-interpolation", "rife"} for part in parent_parts)
        or re.match(r"^rife(?:[._ -]?v?\d)", name)
    ):
        return "rife"
    if (
        any(
            part in {"controlnets", "control_net", "control-net", "sd_control_collection"}
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
    if any(re.fullmatch(r"wan(?:[._ -]?\d+(?:\.\d+)*)?", part) for part in parent_parts) or re.match(
        r"^wan(?:[._ -]?\d+(?:\.\d+)*)?(?:$|[._ -])", name
    ):
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
    explicit_pipeline_architecture = "unknown"
    if class_name.endswith("Pipeline"):
        explicit_pipeline_architecture = _architecture_from_text(class_name)
        pipeline_name = class_name.lower()
        if explicit_pipeline_architecture == "unknown":
            if "stablediffusionxl" in pipeline_name and "inpaint" in pipeline_name:
                # Keep the sorter and full-pipeline route's established SDXL
                # contract; the underlying pipeline class carries the inpaint
                # mode, while its catalog architecture remains SDXL.
                explicit_pipeline_architecture = ARCH_SDXL
            elif "stablediffusionxl" in pipeline_name:
                explicit_pipeline_architecture = ARCH_SDXL
            elif "stablediffusion3" in pipeline_name:
                explicit_pipeline_architecture = ARCH_SD35
            elif "stablediffusioninpaint" in pipeline_name:
                explicit_pipeline_architecture = ARCH_INPAINT
            elif "stablediffusion" in pipeline_name:
                explicit_pipeline_architecture = ARCH_SD15
    if explicit_pipeline_architecture != "unknown":
        # A recognized model_index pipeline class is stronger evidence than
        # incidental family words in the snapshot name or parent directories.
        architecture = explicit_pipeline_architecture
    if _has_flux2_klein_marker(path.name) and architecture in {"unknown", ARCH_FLUX}:
        architecture = ARCH_FLUX2_KLEIN
    lowered = text.lower()
    path_parts = {
        part.lower()
        for part in _relative_path_text(path, roots).replace("\\", "/").split("/")
    }
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

    if architecture == "wan" or (
        explicit_pipeline_architecture == "unknown" and "wan" in lowered
    ):
        family = "wan"
        architecture = "wan"
    elif architecture == "ltx" or (
        explicit_pipeline_architecture == "unknown" and "ltx" in lowered
    ):
        family = "runtime_asset"
        architecture = "ltx"
    elif architecture == ARCH_FLUX2:
        family = "runtime_asset"
    elif "components" in path_parts and architecture in {ARCH_FLUX2_KLEIN, ARCH_Z_IMAGE}:
        family = "text_encoder"
    elif architecture in {
        ARCH_FLUX,
        ARCH_FLUX2_KLEIN,
        ARCH_Z_IMAGE,
        ARCH_FLUX_KONTEXT,
        ARCH_KREA2,
        ARCH_ANIMA,
        ARCH_QWEN_IMAGE,
        ARCH_QWEN_IMAGE_EDIT,
        ARCH_QWEN_IMAGE_EDIT_PLUS,
        ARCH_QWEN_IMAGE_NUNCHAKU,
        ARCH_SANA,
        ARCH_SANA_VIDEO,
        ARCH_LONGCAT_IMAGE,
    }:
        family = "runtime_asset"
    elif explicit_pipeline_architecture == "unknown" and "flux" in lowered:
        family = "runtime_asset"
        architecture = ARCH_FLUX
    elif explicit_pipeline_architecture == "unknown" and (
        "stablediffusionxl" in class_name.lower() or "stable-diffusion-xl" in lowered
    ):
        architecture = ARCH_SDXL
    elif explicit_pipeline_architecture == "unknown" and (
        "stablediffusion3" in class_name.lower() or architecture == ARCH_SD35
    ):
        architecture = ARCH_SD35
    elif explicit_pipeline_architecture == "unknown" and (
        "stablediffusioninpaint" in class_name.lower() or "inpaint" in lowered
    ):
        architecture = ARCH_INPAINT
    elif explicit_pipeline_architecture == "unknown" and "stablediffusion" in class_name.lower():
        architecture = ARCH_SD15
    elif explicit_pipeline_architecture != "unknown":
        # A recognized class that is not one of the specialized runtime asset
        # families above is a complete Diffusers checkpoint. Do not let words
        # in an unrelated parent directory relabel it as Flux, Wan, or LTX.
        family = "checkpoint"
    else:
        family = "runtime_asset"

    current_subdir = _relative_subdir(path, roots)
    recommended = _recommended_subdir(family, architecture, path.name)
    if family == "runtime_asset" and class_name.endswith("Pipeline"):
        recommended = _recommended_diffusers_subdir(architecture, path, roots)
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

    if path.suffix.lower() == ".safetensors" and not safetensors_file_is_structurally_valid(path):
        architecture = _architecture_from_text(f"{path.name} {_relative_path_text(path, roots)}")
        current_subdir = _relative_subdir(path, roots)
        return ModelInventoryRecord(
            path=str(path.resolve()),
            filename=path.name,
            family="invalid_asset",
            architecture=architecture,
            current_subdir=current_subdir,
            recommended_subdir="models to sort",
            should_move=current_subdir.replace("\\", "/").lower() != "models to sort",
            header_identifiers={"integrity": "invalid_safetensors_tensor_ranges_or_framing"},
            metadata={},
        )

    metadata = read_safetensors_metadata(path)
    metadata_text = _metadata_text(metadata)
    torchscript_archive = path.suffix.lower() in {".ckpt", ".pt"} and is_torchscript_archive(path)
    path_family = _matching_path_family(path, roots)
    family = path_family or "unknown"
    model_layout_context = _relative_subdir(path, roots)
    architecture = _metadata_architecture(
        metadata,
        path,
        model_layout_context=model_layout_context,
    )
    # Large model folders carry authoritative modality. Avoid parsing tensor
    # tables from multi-gigabyte Chat/Audio weights to rediscover that context.
    header_match = None if path_family in {"llm", "audio"} else _header_family_architecture(
        path,
        model_layout_context=model_layout_context,
    )
    identifiers: dict[str, str] = {}
    shapes: dict[str, list[int]] = {}

    if path_family in {"llm", "audio"}:
        family = path_family
        architecture = path_family
        identifiers["path_marker"] = f"{path_family} model folder"
    elif path_family == "preprocessor":
        family = "preprocessor"
        architecture = ARCH_UNKNOWN
        identifiers["path_marker"] = "known ControlNet annotator/preprocessor asset"
    elif path_family == "rife":
        family = "runtime_asset"
        architecture = "rife"
        identifiers["path_marker"] = "RIFE frame interpolation model"
    elif path_family == "lama":
        family = "runtime_asset"
        architecture = "lama"
        identifiers["path_marker"] = "LaMa inpainting auxiliary model"
    elif path_family == "upscaler":
        family = "upscaler"
        architecture = "unknown"
        identifiers["path_marker"] = "upscaler asset directory"
    elif path.suffix.lower() == ".safetensors":
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
            controlnet_layout = any(
                part.casefold() in {"controlnet", "controlnets", "control_net", "control-net", "sd_control_collection"}
                for part in model_layout_context.replace("\\", "/").split("/")
            )
            if family == "controlnet" or controlnet_layout:
                family = "controlnet"
                identifiers["tensor_marker"] = "controlnet lora"
            else:
                family = "lora"
                identifiers["tensor_marker"] = "lora_up/lora_down"
        elif UNET_INPUT_KEY in shapes or "conditioner.embedders.1.model.ln_final.weight" in keys:
            family = "checkpoint"
            architecture = infer_architecture_from_shapes(shapes, filename=path.name)
            identifiers["tensor_marker"] = "diffusion checkpoint"

    if torchscript_archive:
        # These are auxiliary runtime assets, not Diffusers checkpoints. Keep
        # their current location and do not propose them for checkpoint sorting.
        family = "runtime_asset"
        architecture = ARCH_UNKNOWN
        identifiers["format_marker"] = "torchscript archive; weights not opened"

    if header_match and family not in {"controlnet", "face_embedding", "upscaler"}:
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
            elif header_family == "text_encoder":
                # An identified encoder with no family signature must not
                # inherit a family guess made from an incidental folder name.
                architecture = ARCH_UNKNOWN
            identifiers.update(header_identifiers)

    if header_match and header_match[0] == "lora" and any(
        part.casefold() in {"controlnet", "controlnets", "control_net", "control-net", "sd_control_collection"}
        for part in model_layout_context.replace("\\", "/").split("/")
    ):
        family = "controlnet"
        identifiers.update(header_match[2])
        identifiers["tensor_marker"] = "controlnet lora"

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
        elif architecture in {ARCH_KREA2, ARCH_ANIMA, ARCH_FLUX2} and path.suffix.lower() == ".safetensors":
            family = "runtime_asset"
            identifiers["filename_marker"] = f"{architecture} transformer"
        elif architecture == ARCH_QWEN_IMAGE_NUNCHAKU and path.suffix.lower() == ".safetensors":
            family = "runtime_asset"
            identifiers["filename_marker"] = "qwen nunchaku transformer"
        elif path.suffix.lower() == ".gguf" and "wan" in path.name.lower():
            family = "wan"
            architecture = "wan"
            identifiers["filename_marker"] = "wan gguf"
        elif path.suffix.lower() == ".gguf" and architecture in {ARCH_FLUX, ARCH_FLUX2, ARCH_FLUX2_KLEIN, ARCH_Z_IMAGE}:
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
    recommended = (
        current_subdir
        if torchscript_archive or path_family in {"llm", "audio", "rife", "lama"}
        else _recommended_subdir(family, architecture, path.name)
    )
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


def scan_model_inventory(
    flags: RuntimeFlags,
    *,
    walk_errors: dict[str, list[str]] | None = None,
) -> list[ModelInventoryRecord]:
    roots = model_inventory_roots(flags)
    seen: set[str] = set()
    records: list[ModelInventoryRecord] = []
    for root in roots:
        root_key = os.path.normcase(str(root.resolve()))

        def record_walk_error(error: OSError, *, _root_key: str = root_key) -> None:
            if walk_errors is None:
                return
            details = walk_errors.setdefault(_root_key, [])
            if len(details) < 20:
                details.append(str(error))

        for current, dir_names, file_names in os.walk(root, onerror=record_walk_error):
            # Snapshot replacement preserves the previous incomplete folder
            # here for recovery. It is deliberately outside the active model
            # inventory so backups never appear as selectable models.
            dir_names[:] = [name for name in dir_names if name.casefold() != ".aiwf-recovery"]
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
        "roots_signature": _fast_roots_signature(flags),
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
        root_key = os.path.normcase(str(root))
        try:
            for current, dir_names, file_names in os.walk(root):
                dir_names[:] = sorted(
                    (name for name in dir_names if name.casefold() != ".aiwf-recovery"),
                    key=str.lower,
                )
                current_path = Path(current)
                try:
                    stat = current_path.stat()
                    parts.append(f"{os.path.normcase(str(current_path))}:{stat.st_mtime_ns}")
                except OSError:
                    parts.append(os.path.normcase(str(current_path)))
                for filename in file_names:
                    if Path(filename).suffix.casefold() not in MODEL_EXTENSIONS:
                        continue
                    file_path = current_path / filename
                    try:
                        file_stat = file_path.stat()
                        parts.append(
                            f"{os.path.normcase(str(file_path))}:{file_stat.st_size}:{file_stat.st_mtime_ns}"
                        )
                    except OSError:
                        parts.append(os.path.normcase(str(file_path)))
        except OSError:
            parts.append(root_key)
    return "|".join(sorted(parts))


def _records_from_payload(payload: dict) -> list[ModelInventoryRecord]:
    records: list[ModelInventoryRecord] = []
    for item in payload.get("assets") or []:
        if not isinstance(item, dict):
            continue
        records.append(ModelInventoryRecord(**item))
    return records


def _inventory_paths_exist(records: list[ModelInventoryRecord]) -> bool:
    """Reject cached inventories that still reference removed model assets."""
    for record in records:
        try:
            if not Path(record.path).exists():
                return False
        except (OSError, ValueError):
            return False
    return True


def load_model_inventory(
    flags: RuntimeFlags, *, roots_signature: str | None = None
) -> list[ModelInventoryRecord] | None:
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
    if payload.get("roots_signature") != (roots_signature or _fast_roots_signature(flags)):
        return None
    records = _records_from_payload(payload)
    if not _inventory_paths_exist(records):
        return None
    return records


def invalidate_model_inventory_cache() -> None:
    _SESSION_INVENTORY.clear()


def get_model_inventory(flags: RuntimeFlags, *, force_rescan: bool = False) -> list[ModelInventoryRecord]:
    signature = _fast_roots_signature(flags)
    if not force_rescan:
        session_cached = _SESSION_INVENTORY.get(signature)
        if session_cached is not None:
            if _inventory_paths_exist(session_cached):
                return session_cached
            _SESSION_INVENTORY.pop(signature, None)
        disk_cached = load_model_inventory(flags, roots_signature=signature)
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


def scan_and_write_model_inventory(
    flags: RuntimeFlags,
    *,
    walk_errors: dict[str, list[str]] | None = None,
) -> list[ModelInventoryRecord]:
    if walk_errors is None:
        return get_model_inventory(flags, force_rescan=True)
    signature = _fast_roots_signature(flags)
    records = scan_model_inventory(flags, walk_errors=walk_errors)
    write_model_inventory(flags, records)
    _SESSION_INVENTORY[signature] = records
    _SESSION_INVENTORY[_fast_roots_signature(flags)] = records
    logger.info("Indexed %d local model asset(s) with %d root traversal error(s)", len(records), sum(map(len, walk_errors.values())))
    return records


def scan_model_inventory_report(
    flags: RuntimeFlags, *, proposal_limit: int | None = 250
) -> dict[str, object]:
    """Force an inventory scan, report traversal errors, and never move model files."""
    walk_errors: dict[str, list[str]] = {}
    records = scan_and_write_model_inventory(flags, walk_errors=walk_errors)
    configured: list[tuple[str, Path]] = [
        ("Models directory", flags.resolved_models_dir()),
        ("Checkpoint directory", flags.resolved_ckpt_dir()),
    ]
    configured.extend((f"Extra model directory {index + 1}", root) for index, root in enumerate(flags.resolved_extra_model_dirs()))
    configured.extend((f"Extra checkpoint directory {index + 1}", root) for index, root in enumerate(flags.resolved_extra_ckpt_dirs()))
    active_roots = {os.path.normcase(str(root.resolve())) for root in model_inventory_roots(flags)}
    summaries: list[dict[str, object]] = []
    for label, root in configured:
        try:
            resolved = root.resolve()
            exists = resolved.exists()
            is_directory = resolved.is_dir()
            readable = is_directory and os.access(resolved, os.R_OK)
        except OSError:
            resolved = root
            exists = False
            is_directory = False
            readable = False
        family_counts: dict[str, int] = {}
        if readable:
            for record in records:
                try:
                    Path(record.path).resolve().relative_to(resolved)
                except (OSError, ValueError):
                    continue
                family_counts[record.family] = family_counts.get(record.family, 0) + 1
        root_key = os.path.normcase(str(resolved))
        errors = walk_errors.get(root_key, [])
        scan_status = (
            "partial" if readable and root_key in active_roots and errors
            else "scanned" if readable and root_key in active_roots
            else "missing" if not exists
            else "not-directory" if not is_directory
            else "unreadable" if not readable
            else "nested"
        )
        summaries.append({
            "label": label,
            "path": str(resolved),
            "status": scan_status,
            "assetCount": sum(family_counts.values()),
            "familyCounts": family_counts,
            "errorCount": len(errors),
            "errors": errors[:5],
        })
    ordered_records = sorted(records, key=lambda record: record.path.lower())
    primary_models_root = flags.resolved_models_dir().resolve()
    extra_copy_roots = [
        root.resolve()
        for root in (*flags.resolved_extra_model_dirs(), *flags.resolved_extra_ckpt_dirs())
    ]
    overlapping_copy_roots = any(
        _is_relative_to(root, primary_models_root) or _is_relative_to(primary_models_root, root)
        for root in extra_copy_roots
    )
    # A "candidate" is specifically eligible for the shared-root copy flow.
    # Assets already under the main Models root belong to Reorganize, while
    # checkpoint/default roots do not have an equivalent copy action.
    def placement_for(record: ModelInventoryRecord) -> tuple[str, str]:
        if not record.should_move:
            return "in-place", ""
        confident, confidence_reason = model_asset_placement_confidence(record)
        if not confident:
            return "manual-review", confidence_reason
        try:
            source = Path(record.path).resolve(strict=True)
        except (OSError, RuntimeError):
            return "manual-review", "source path is no longer available"
        if _is_relative_to(source, primary_models_root):
            if source.is_file():
                return "reorganize-candidate", ""
            if source.is_dir() and (source / "model_index.json").is_file():
                # Reorganize has its own supported-pipeline and completeness
                # checks for Diffusers folders; the scan does not duplicate
                # those decisions or claim the folder will move.
                return "reorganize-check", ""
            return "manual-review", "folder is not a supported complete Diffusers model"
        if (
            not overlapping_copy_roots
            and str(source) == record.path
            and any(_is_relative_to(source, root) for root in extra_copy_roots)
            and source.is_file()
            and not source.is_symlink()
        ):
            return "candidate", ""
        if overlapping_copy_roots:
            return "manual-review", "configured model roots overlap, so an automatic destination is ambiguous"
        if source.is_symlink():
            return "manual-review", "source is a symbolic link and is not eligible for automatic copying"
        return "manual-review", "source is outside a supported automatic-copy root"

    all_proposals = []
    for record in ordered_records:
        placement, placement_reason = placement_for(record)
        all_proposals.append({
            "path": record.path,
            "filename": record.filename,
            "family": record.family,
            "architecture": record.architecture,
            "currentSubdir": record.current_subdir,
            "recommendedSubdir": record.recommended_subdir,
            "placement": placement,
            "placementReason": placement_reason,
            "signals": record.header_identifiers,
        })
    placement_priority = {
        "candidate": 0,
        "reorganize-candidate": 1,
        "manual-review": 2,
        "reorganize-check": 3,
        "in-place": 4,
    }
    # Keep actionable work visible on the first page. The inventory itself is
    # still path-stable within each placement group, while the full scan and
    # search endpoint retain every proposal for review.
    all_proposals.sort(key=lambda item: (placement_priority.get(item["placement"], 5), str(item["path"]).lower()))
    proposals = all_proposals if proposal_limit is None else all_proposals[:max(0, proposal_limit)]
    return {
        "inventoryCount": len(records),
        "roots": summaries,
        "assets": proposals,
        "assetsTruncated": max(0, len(ordered_records) - len(proposals)),
    }
