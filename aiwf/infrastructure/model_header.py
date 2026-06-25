"""Lightweight model-header reader for GGUF and safetensors files.

Returns a ModelInfo dataclass without loading any weights.
Used to populate dropdown labels so users see human-readable names
instead of raw filenames.

Detection priority:
  1. modelspec.architecture / general.architecture metadata
  2. Tensor key pattern matching (first ~40 keys sampled)
  3. Folder path convention (models/VAE, models/Textencoder, ...)
  4. Filename keyword heuristics

Precision detection uses element-weighted voting so giant quantised
weight matrices win over the many small F16 bias/norm tensors present
in every file.
"""
from __future__ import annotations

import atexit
import json
import logging
import re
import struct
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Architecture / role constants
# ---------------------------------------------------------------------------

ARCH_WAN_TRANSFORMER     = "wan-transformer"
ARCH_WAN_TRANSFORMER_FP8 = "wan-transformer-fp8"
ARCH_WAN_LORA            = "wan-lora"
ARCH_WAN_VAE             = "wan-vae"
ARCH_UMT5_ENCODER        = "umt5-encoder"
ARCH_T5XXL_ENCODER       = "t5xxl-encoder"
ARCH_CLIP                = "clip"
ARCH_FLUX_TRANSFORMER    = "flux-transformer"
ARCH_FLUX2_KLEIN_TRANSFORMER = "flux2-klein-transformer"
ARCH_Z_IMAGE_TRANSFORMER = "z-image-transformer"
ARCH_FLUX_LORA           = "flux-lora"
ARCH_FLUX_VAE            = "flux-vae"
ARCH_SDXL_CHECKPOINT     = "sdxl-checkpoint"
ARCH_SD35_CHECKPOINT     = "sd3.5-checkpoint"
ARCH_SD_CHECKPOINT       = "sd-checkpoint"
ARCH_SD_LORA             = "sd-lora"
ARCH_RIFE                = "rife"
ARCH_SAM                 = "sam"
ARCH_UNKNOWN             = "unknown"

ROLE_HIGH_NOISE    = "high-noise"
ROLE_LOW_NOISE     = "low-noise"
ROLE_TEXT_ENCODER  = "text-encoder"
ROLE_VAE           = "vae"
ROLE_LORA          = "lora"
ROLE_CHECKPOINT    = "checkpoint"
ROLE_UPSCALER      = "upscaler"
ROLE_UNKNOWN       = "unknown"

# ---------------------------------------------------------------------------
# Detection tables
# ---------------------------------------------------------------------------

# Safetensors tensor key prefix -> (arch, role_hint)
_ST_PATTERNS: list[tuple[str, str, str]] = [
    ("model.diffusion_model.blocks.0.cross_attn.k.weight_scale", ARCH_WAN_TRANSFORMER_FP8, ""),
    ("model.diffusion_model.blocks.0.cross_attn",                ARCH_WAN_TRANSFORMER_FP8, ""),
    ("diffusion_model.blocks.0.cross_attn.k.weight",             ARCH_WAN_TRANSFORMER,     ""),
    ("diffusion_model.blocks.0.cross_attn.k.lora_A.weight",      ARCH_WAN_LORA,            ROLE_LORA),
    ("encoder.block.0.layer.0.SelfAttention.q.weight",           ARCH_UMT5_ENCODER,        ROLE_TEXT_ENCODER),
    ("encoder.block.0.layer.0.SelfAttention",                    ARCH_UMT5_ENCODER,        ROLE_TEXT_ENCODER),
    ("text_model.encoder.layers.0",                              ARCH_CLIP,                ROLE_TEXT_ENCODER),
    ("cond_stage_model.transformer.resblocks.0",                 ARCH_CLIP,                ROLE_TEXT_ENCODER),
    ("double_blocks.0",                                          ARCH_FLUX_TRANSFORMER,    ""),
    ("single_blocks.0",                                          ARCH_FLUX_TRANSFORMER,    ""),
    ("transformer.double_blocks.0",                              ARCH_FLUX_TRANSFORMER,    ""),
    ("transformer.single_blocks.0",                              ARCH_FLUX_TRANSFORMER,    ""),
    ("model.diffusion_model.double_blocks.0",                    ARCH_FLUX_TRANSFORMER,    ""),
    ("model.diffusion_model.single_blocks.0",                    ARCH_FLUX_TRANSFORMER,    ""),
    ("lora_transformer_",                                        ARCH_FLUX_LORA,           ROLE_LORA),
    ("conditioner.embedders.0.transformer",                      ARCH_SDXL_CHECKPOINT,     ROLE_CHECKPOINT),
    ("model.diffusion_model.joint_blocks.0",                     ARCH_SD35_CHECKPOINT,     ROLE_CHECKPOINT),
    ("model.diffusion_model.input_blocks.0.0.weight",            ARCH_SD_CHECKPOINT,       ROLE_CHECKPOINT),
    ("cond_stage_model.model.transformer.resblocks.0",           ARCH_SD_CHECKPOINT,       ROLE_CHECKPOINT),
    ("lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight", ARCH_SD_LORA,        ROLE_LORA),
]

_GGUF_ARCH_MAP: dict[str, str] = {
    "wan":       ARCH_WAN_TRANSFORMER,
    "t5encoder": ARCH_T5XXL_ENCODER,
    "clip":      ARCH_CLIP,
    "flux":      ARCH_FLUX_TRANSFORMER,
    "lumina2":   ARCH_Z_IMAGE_TRANSFORMER,
}

_ST_DTYPE_DISPLAY: dict[str, str] = {
    "F8_E4M3": "FP8", "F8_E5M2": "FP8",
    "F16": "F16", "BF16": "BF16", "F32": "F32",
    "U8": "", "I8": "", "I32": "", "I64": "", "U16": "",
}

_FOLDER_CLUES: list[tuple[str, str, str]] = [
    ("flux/GGUF",       ARCH_FLUX_TRANSFORMER, ""),
    ("flux/UNet",       ARCH_FLUX_TRANSFORMER, ""),
    ("flux/Textencoder", ARCH_T5XXL_ENCODER,   ROLE_TEXT_ENCODER),
    ("flux/textencoder", ARCH_T5XXL_ENCODER,   ROLE_TEXT_ENCODER),
    ("flux/VAE",        ARCH_FLUX_VAE,         ROLE_VAE),
    ("flux/vae",        ARCH_FLUX_VAE,         ROLE_VAE),
    ("flux2/GGUF",      ARCH_FLUX2_KLEIN_TRANSFORMER, ""),
    ("flux2/UNet",      ARCH_FLUX2_KLEIN_TRANSFORMER, ""),
    ("flux2/Components", ARCH_FLUX2_KLEIN_TRANSFORMER, ""),
    ("z-image/GGUF",    ARCH_Z_IMAGE_TRANSFORMER, ""),
    ("z-image/UNet",    ARCH_Z_IMAGE_TRANSFORMER, ""),
    ("z-image/Components", ARCH_Z_IMAGE_TRANSFORMER, ""),
    ("Loras/Flux",      ARCH_FLUX_LORA,        ROLE_LORA),
    ("Lora/Flux",       ARCH_FLUX_LORA,        ROLE_LORA),
    ("wan/GGUF",       ARCH_WAN_TRANSFORMER,     ""),
    ("wan/Safetensor", ARCH_WAN_TRANSFORMER_FP8, ""),
    ("wan/lora",       ARCH_WAN_LORA,            ROLE_LORA),
    ("wan/Diffusers",  ARCH_WAN_TRANSFORMER,     ""),
    ("Textencoder",    ARCH_UMT5_ENCODER,        ROLE_TEXT_ENCODER),
    ("textencoder",    ARCH_UMT5_ENCODER,        ROLE_TEXT_ENCODER),
    ("TextEncoder",    ARCH_UMT5_ENCODER,        ROLE_TEXT_ENCODER),
    ("Clip",           ARCH_CLIP,                ROLE_TEXT_ENCODER),
    ("VAE",            ARCH_WAN_VAE,             ROLE_VAE),
    ("vae",            ARCH_WAN_VAE,             ROLE_VAE),
    ("sam",            ARCH_SAM,                 ROLE_UNKNOWN),
    ("rife",           ARCH_RIFE,                ROLE_UNKNOWN),
    ("Stable-diffusion", ARCH_SD_CHECKPOINT,     ROLE_CHECKPOINT),
    ("Lora",           ARCH_SD_LORA,             ROLE_LORA),
    ("ESRGAN",         "esrgan",                 ROLE_UPSCALER),
    ("RealESRGAN",     "realesrgan",             ROLE_UPSCALER),
]

_FILENAME_ROLE_HINTS: list[tuple[str, str]] = [
    ("high", ROLE_HIGH_NOISE),
    ("low",  ROLE_LOW_NOISE),
    ("lora", ROLE_LORA),
    ("vae",  ROLE_VAE),
    ("umt5", ROLE_TEXT_ENCODER),
    ("clip", ROLE_TEXT_ENCODER),
]

# GGUF quant base name -> filename keyword -> full name with sub-variant
_GGUF_SUBVARIANT: dict[str, list[tuple[str, str]]] = {
    "Q3_K": [("q3_k_s", "Q3_K_S"), ("q3_k_m", "Q3_K_M"), ("q3_k_l", "Q3_K_L")],
    "Q4_K": [("q4_k_s", "Q4_K_S"), ("q4_k_m", "Q4_K_M")],
    "Q5_K": [("q5_k_s", "Q5_K_S"), ("q5_k_m", "Q5_K_M")],
    "IQ3_K": [("iq3_k_s", "IQ3_K_S"), ("iq3_k_m", "IQ3_K_M")],
}

_ARCH_PREFIX: dict[str, str] = {
    ARCH_WAN_TRANSFORMER:     "Wan",
    ARCH_WAN_TRANSFORMER_FP8: "Wan",
    ARCH_WAN_LORA:            "Wan LoRA",
    ARCH_WAN_VAE:             "Wan VAE",
    ARCH_UMT5_ENCODER:        "UMT5-XXL",
    ARCH_T5XXL_ENCODER:       "T5-XXL",
    ARCH_CLIP:                "CLIP",
    ARCH_FLUX_TRANSFORMER:    "Flux",
    ARCH_FLUX2_KLEIN_TRANSFORMER: "Flux.2 Klein",
    ARCH_Z_IMAGE_TRANSFORMER: "Z-Image",
    ARCH_FLUX_LORA:           "Flux LoRA",
    ARCH_FLUX_VAE:            "Flux VAE",
    ARCH_SDXL_CHECKPOINT:     "SDXL",
    ARCH_SD35_CHECKPOINT:     "SD3.5",
    ARCH_SD_CHECKPOINT:       "SD",
    ARCH_SD_LORA:             "LoRA",
    ARCH_RIFE:                "RIFE",
    ARCH_SAM:                 "SAM",
    ARCH_UNKNOWN:             "",
}

_ROLE_SUFFIX: dict[str, str] = {
    ROLE_HIGH_NOISE:   "High Noise",
    ROLE_LOW_NOISE:    "Low Noise",
    ROLE_TEXT_ENCODER: "Encoder",
    ROLE_VAE:          "VAE",
    ROLE_LORA:         "LoRA",
    ROLE_CHECKPOINT:   "",
    ROLE_UPSCALER:     "Upscaler",
    ROLE_UNKNOWN:      "",
}

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class ModelInfo:
    """Rich header information about a local model file."""

    path: str
    filename: str
    arch: str
    role: str
    precision: str
    size_mb: float
    display_name: str
    raw_meta: dict = field(default_factory=dict)
    tensor_count: int = 0

    def is_wan_transformer(self) -> bool:
        return self.arch in (ARCH_WAN_TRANSFORMER, ARCH_WAN_TRANSFORMER_FP8)

    def is_high_noise(self) -> bool:
        return self.role == ROLE_HIGH_NOISE

    def is_low_noise(self) -> bool:
        return self.role == ROLE_LOW_NOISE

    def is_text_encoder(self) -> bool:
        return self.arch in (ARCH_UMT5_ENCODER, ARCH_T5XXL_ENCODER, ARCH_CLIP) or self.role == ROLE_TEXT_ENCODER

    def is_t5xxl(self) -> bool:
        """True if this looks like a T5-XXL (Flux/SD3) file — NOT for Wan."""
        if self.arch == ARCH_T5XXL_ENCODER:
            return True
        stem = Path(self.filename).stem.lower()
        return (stem.startswith("t5xxl") or stem in ("t5xxl_fp16", "t5xxl_fp8_e4m3fn")) \
            and "umt5" not in self.filename.lower()

    def size_label(self) -> str:
        if self.size_mb >= 1000:
            return f"{self.size_mb / 1000:.1f} GB"
        return f"{self.size_mb:.0f} MB"


# ---------------------------------------------------------------------------
# Persistent Header Cache
# ---------------------------------------------------------------------------



class ModelHeaderCache:
    def __init__(self):
        self.cache_file = Path(__file__).resolve().parents[2] / "cache" / "model_header_cache.json"
        self.data = {}
        self.loaded = False
        self.dirty = False

    def load(self):
        if self.loaded:
            return
        if self.cache_file.exists():
            try:
                self.data = json.loads(self.cache_file.read_text(encoding="utf-8"))
            except Exception:
                logger.debug("Failed to load model header cache, initializing empty", exc_info=True)
                self.data = {}
        self.loaded = True

    def save(self):
        if not self.dirty:
            return
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.cache_file.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
            self.dirty = False
        except Exception:
            logger.warning("Failed to save model header cache", exc_info=True)

    def get(self, path: Path) -> ModelInfo | None:
        self.load()
        key = str(path.resolve())
        if key not in self.data:
            return None
        entry = self.data[key]
        try:
            stat = path.stat()
            if entry.get("mtime") != stat.st_mtime or entry.get("size") != stat.st_size:
                return None
            
            # Reconstruct ModelInfo
            return ModelInfo(
                path=entry["path"],
                filename=entry["filename"],
                arch=entry["arch"],
                role=entry["role"],
                precision=entry["precision"],
                size_mb=entry["size_mb"],
                display_name=entry["display_name"],
                raw_meta=entry.get("raw_meta", {}),
                tensor_count=entry.get("tensor_count", 0),
            )
        except Exception:
            return None

    def set(self, path: Path, info: ModelInfo):
        self.load()
        try:
            stat = path.stat()
            key = str(path.resolve())
            
            # Check if it's already cached with correct mtime/size to avoid writing
            existing = self.data.get(key)
            if existing and existing.get("mtime") == stat.st_mtime and existing.get("size") == stat.st_size:
                return
                
            self.data[key] = {
                "path": info.path,
                "filename": info.filename,
                "arch": info.arch,
                "role": info.role,
                "precision": info.precision,
                "size_mb": info.size_mb,
                "display_name": info.display_name,
                "raw_meta": info.raw_meta,
                "tensor_count": info.tensor_count,
                "mtime": stat.st_mtime,
                "size": stat.st_size,
            }
            self.dirty = True
            self.save()
        except Exception:
            pass

_HEADER_CACHE = ModelHeaderCache()
atexit.register(_HEADER_CACHE.save)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def read_model_info(path) -> ModelInfo:
    """Read header metadata from a model file without loading weights."""
    p = Path(path).resolve()
    
    cached_info = _HEADER_CACHE.get(p)
    if cached_info is not None:
        return cached_info

    size_mb = _file_size_mb(p)
    suffix = p.suffix.lower()
    if suffix == ".gguf":
        info = _read_gguf(p, size_mb)
    elif suffix == ".safetensors":
        info = _read_safetensors(p, size_mb)
    else:
        arch, role = _arch_from_folder_and_filename(p)
        info = ModelInfo(
            path=str(p), filename=p.name, arch=arch, role=role,
            precision="", size_mb=size_mb,
            display_name=_make_display_name(_clean_stem(p.stem), arch, role, "", size_mb),
        )
        
    _HEADER_CACHE.set(p, info)
    return info


def read_model_info_batch(paths: Iterable) -> list:
    """Read headers for multiple files; errors are logged and skipped."""
    out: list = []
    for p in paths:
        try:
            out.append(read_model_info(p))
        except Exception:
            logger.debug("model_header: could not read %s", p, exc_info=True)
    return out


def model_choice_label(info: ModelInfo) -> tuple:
    """Return (display_label, path) for a Gradio dropdown."""
    return (info.display_name, info.path)


# ---------------------------------------------------------------------------
# GGUF reader
# ---------------------------------------------------------------------------


def _read_gguf(p: Path, size_mb: float) -> ModelInfo:
    try:
        import gguf as _gguf

        reader = _gguf.GGUFReader(str(p))
        meta = _gguf_meta(reader)
        arch_str = meta.get("general.architecture", "")
        arch = _GGUF_ARCH_MAP.get(arch_str.lower(), "")
        title = meta.get("general.name", "").strip()
        size_label_str = meta.get("general.size_label", "").strip()
        quant = _gguf_dominant_quant(reader, filename=p.name)
        family_arch = _transformer_family_arch_from_name(p, title)
        if family_arch:
            arch = family_arch

        # Refine arch / role
        if "umt5" in title.lower() or "umt5" in p.name.lower():
            arch = ARCH_UMT5_ENCODER
            role = ROLE_TEXT_ENCODER
        elif arch == ARCH_T5XXL_ENCODER:
            role = ROLE_TEXT_ENCODER
        elif arch == ARCH_FLUX_TRANSFORMER:
            role = _role_from_filename(p.name)
        elif arch == ARCH_WAN_TRANSFORMER:
            role = _role_from_filename(p.name)
        else:
            arch_f, role_f = _arch_from_folder_and_filename(p)
            arch = arch or arch_f
            role = role_f

        if not title:
            title = _clean_stem(p.stem)
        if size_label_str and size_label_str.upper() not in title.upper():
            title = f"{title} {size_label_str.upper()}"

        display = _make_display_name(title, arch, role, quant, size_mb)
        return ModelInfo(
            path=str(p), filename=p.name, arch=arch, role=role,
            precision=quant, size_mb=size_mb, display_name=display,
            raw_meta=meta, tensor_count=len(reader.tensors),
        )
    except ImportError:
        pass
    except Exception:
        logger.debug("model_header: GGUF read failed for %s", p, exc_info=True)

    # Fallback: filename heuristics only
    arch, role = _arch_from_folder_and_filename(p)
    quant = _quant_from_filename(p.name)
    return ModelInfo(
        path=str(p), filename=p.name, arch=arch, role=role,
        precision=quant, size_mb=size_mb,
        display_name=_make_display_name(_clean_stem(p.stem), arch, role, quant, size_mb),
    )


def _gguf_meta(reader) -> dict:
    out: dict = {}
    for key, fld in reader.fields.items():
        try:
            if not fld.types:
                continue
            tn = fld.types[0].name
            if tn == "STRING":
                val = bytes(fld.parts[fld.data[0]]).decode("utf-8", errors="replace")
            elif tn in ("UINT8","INT8","UINT16","INT16","UINT32","INT32","UINT64","INT64"):
                val = str(int(fld.data[0]))
            elif tn in ("FLOAT32", "FLOAT64"):
                val = str(float(fld.data[0]))
            elif tn == "BOOL":
                val = str(bool(fld.data[0]))
            else:
                continue
            out[str(key)] = val
        except Exception:
            pass
    return out


def _gguf_dominant_quant(reader, filename: str = "") -> str:
    """Element-weighted dominant quant; resolves sub-variant from filename."""
    weighted: Counter = Counter()
    for t in reader.tensors:
        total = 1
        for d in t.shape:
            total *= int(d)
        if total < 4096:
            continue
        try:
            weighted[t.tensor_type.name] += total
        except Exception:
            pass

    if not weighted:
        return _quant_from_filename(filename)

    dominant = weighted.most_common(1)[0][0]
    name_lower = filename.lower()
    candidates = _GGUF_SUBVARIANT.get(dominant)
    if candidates:
        for kw, full in candidates:
            if kw in name_lower:
                return full
    return dominant


# ---------------------------------------------------------------------------
# Safetensors reader
# ---------------------------------------------------------------------------


def _read_safetensors(p: Path, size_mb: float) -> ModelInfo:
    hdr = _st_read_header(p)
    if hdr is None:
        arch, role = _arch_from_folder_and_filename(p)
        return ModelInfo(
            path=str(p), filename=p.name, arch=arch, role=role,
            precision="", size_mb=size_mb,
            display_name=_make_display_name(_clean_stem(p.stem), arch, role, "", size_mb),
        )

    meta: dict = {str(k): str(v) for k, v in hdr.get("__metadata__", {}).items()}
    tensor_keys = [k for k in hdr if k != "__metadata__"]
    precision = _st_dominant_precision(hdr, tensor_keys)
    arch, role = _arch_from_st_meta_and_keys(meta, tensor_keys, p)

    title = (
        meta.get("modelspec.title", "")
        or meta.get("ss_output_name", "")
        or _clean_stem(p.stem)
    ).strip()
    if len(title) > 52:
        title = title[:49].rstrip() + "..."

    display = _make_display_name(title, arch, role, precision, size_mb)
    return ModelInfo(
        path=str(p), filename=p.name, arch=arch, role=role,
        precision=precision, size_mb=size_mb, display_name=display,
        raw_meta=meta, tensor_count=len(tensor_keys),
    )


def _st_read_header(p: Path) -> dict | None:
    try:
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            if n > 100 * 1024 * 1024:
                return None
            return json.loads(f.read(n))
    except Exception:
        logger.debug("model_header: safetensors header read failed for %s", p, exc_info=True)
        return None


def _st_dominant_precision(hdr: dict, tensor_keys: list) -> str:
    """Element-weighted dominant weight dtype."""
    weighted: Counter = Counter()
    for k in tensor_keys:
        v = hdr.get(k)
        if not isinstance(v, dict):
            continue
        display = _ST_DTYPE_DISPLAY.get(v.get("dtype", ""), "")
        if not display:
            continue
        total = 1
        for d in v.get("shape", []):
            total *= d
        if total < 4096:
            continue
        weighted[display] += total

    if weighted:
        return weighted.most_common(1)[0][0]

    for k in tensor_keys[:50]:
        v = hdr.get(k)
        if isinstance(v, dict):
            display = _ST_DTYPE_DISPLAY.get(v.get("dtype", ""), "")
            if display:
                return display
    return ""


def _looks_like_lora_keys(keys: Iterable[str]) -> bool:
    return any("lora_down" in key or "lora_up" in key or ".lora_A." in key or ".lora_B." in key for key in keys)


def _looks_like_flux_lora_keys(keys: Iterable[str]) -> bool:
    for key in keys:
        lowered = key.lower()
        if not ("lora_down" in lowered or "lora_up" in lowered or ".lora_a." in lowered or ".lora_b." in lowered):
            continue
        if (
            "lora_transformer" in lowered
            or "double_blocks" in lowered
            or "single_blocks" in lowered
            # diffusers-format Flux LoRAs name the blocks transformer.single_transformer_blocks.*
            # (single_transformer_blocks is Flux-specific; SD3 only has joint transformer_blocks).
            or "single_transformer_blocks" in lowered
            or "transformer_blocks" in lowered and "flux" in lowered
        ):
            return True
    return False


def _looks_like_t5xxl_file(p: Path, text: str) -> bool:
    combined = f"{p.name} {text}".lower().replace("-", "_")
    if "umt5" in combined:
        return False
    return any(token in combined for token in ("t5xxl", "t5_xxl", "t5_v1_1_xxl", "t5-v1_1-xxl"))


def _looks_like_vae_file(p: Path, tensor_keys: Iterable[str], text: str) -> bool:
    combined = f"{p.name} {p.as_posix()} {text}".lower()
    if "/vae/" in combined.replace("\\", "/") or " ae.safetensors" in f" {p.name.lower()}":
        return True
    return any(key.startswith(("encoder.", "decoder.", "quant_conv.", "post_quant_conv.")) for key in tensor_keys)


def _arch_from_st_meta_and_keys(meta: dict, tensor_keys: list, p: Path) -> tuple:
    # 1. modelspec.architecture
    spec_arch = meta.get("modelspec.architecture", "").lower()
    combined_meta = " ".join(str(v) for v in meta.values()).lower()
    if "wan" in spec_arch:
        return ARCH_WAN_TRANSFORMER_FP8, _role_from_meta_and_filename(meta, p.name)
    if "flux" in spec_arch or "flux" in combined_meta:
        if _looks_like_lora_keys(tensor_keys):
            return ARCH_FLUX_LORA, ROLE_LORA
        if _looks_like_vae_file(p, tensor_keys, combined_meta):
            return ARCH_FLUX_VAE, ROLE_VAE
        return ARCH_FLUX_TRANSFORMER, _role_from_meta_and_filename(meta, p.name)
    if _looks_like_t5xxl_file(p, combined_meta):
        return ARCH_T5XXL_ENCODER, ROLE_TEXT_ENCODER
    if p.name.lower() == "ae.safetensors":
        return ARCH_FLUX_VAE, ROLE_VAE

    # 2. Tensor key patterns (sample first 40)
    sample = set(tensor_keys[:40])
    if _looks_like_flux_lora_keys(sample):
        return ARCH_FLUX_LORA, ROLE_LORA
    for pattern, arch, role_hint in _ST_PATTERNS:
        for key in sample:
            if key.startswith(pattern) or key == pattern:
                role = role_hint or _role_from_meta_and_filename(meta, p.name)
                return arch, role

    # 3. Folder / filename
    return _arch_from_folder_and_filename(p)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _arch_from_folder_and_filename(p: Path) -> tuple:
    path_str = p.as_posix()
    transformer_arch = _transformer_family_arch_from_name(p)
    if transformer_arch:
        return transformer_arch, _role_from_filename(p.name)
    if p.name.lower() == "ae.safetensors":
        return ARCH_FLUX_VAE, ROLE_VAE
    if _looks_like_t5xxl_file(p, ""):
        return ARCH_T5XXL_ENCODER, ROLE_TEXT_ENCODER
    if "umt5" in p.name.lower():
        return ARCH_UMT5_ENCODER, ROLE_TEXT_ENCODER
    if "flux" in p.name.lower():
        if p.suffix.lower() == ".gguf":
            return ARCH_FLUX_TRANSFORMER, _role_from_filename(p.name)
        if "lora" in p.name.lower():
            return ARCH_FLUX_LORA, ROLE_LORA
    for fragment, arch, role in _FOLDER_CLUES:
        if f"/{fragment}/" in path_str or path_str.endswith(f"/{fragment}"):
            return arch, role or _role_from_filename(p.name)
    return ARCH_UNKNOWN, _role_from_filename(p.name)


def _transformer_family_arch_from_name(p: Path, title: str = "") -> str | None:
    text = f"{p.as_posix()} {title}".lower().replace("_", "-")
    compact = text.replace("-", "").replace(" ", "")
    if "z-image" in text or "zimage" in compact or "lumina2" in compact:
        return ARCH_Z_IMAGE_TRANSFORMER
    if "flux.2" in text or "flux2" in compact or "klein" in text:
        return ARCH_FLUX2_KLEIN_TRANSFORMER
    return None


def _role_from_filename(filename: str) -> str:
    name = filename.lower()
    if "high" in name:
        return ROLE_HIGH_NOISE
    if "low" in name:
        return ROLE_LOW_NOISE
    for kw, role in _FILENAME_ROLE_HINTS:
        if kw in name:
            return role
    return ROLE_UNKNOWN


def _role_from_meta_and_filename(meta: dict, filename: str) -> str:
    combined = (meta.get("modelspec.title", "") + " " + filename).lower()
    if "high" in combined:
        return ROLE_HIGH_NOISE
    if "low" in combined:
        return ROLE_LOW_NOISE
    return _role_from_filename(filename)


def _quant_from_filename(filename: str) -> str:
    name = filename.lower()
    for q in ("q4_k_m","q4_k_s","q5_k_m","q5_k_s","q3_k_s","q3_k_m",
              "q6_k","q8_0","q4_0","q2_k","iq2_xs","iq3_xxs"):
        if q in name:
            return q.upper()
    if "nf4" in name:
        return "NF4"
    if "fp4" in name:
        return "FP4"
    if "fp8" in name or "_f8" in name:
        return "FP8"
    if "fp16" in name or "_f16" in name:
        return "F16"
    if "bf16" in name:
        return "BF16"
    if "fp32" in name or "_f32" in name:
        return "F32"
    return ""


def _file_size_mb(p: Path) -> float:
    try:
        return p.stat().st_size / (1024 * 1024)
    except OSError:
        return 0.0


def _clean_stem(stem: str) -> str:
    """Convert a filename stem to a readable title."""
    cleaned = re.sub(
        r"[_\-]?(q\d+_k(?:_[sml])?|q\d+_\d|iq\d+_\w+|fp8|fp16|fp32|bf16|_f16|_f32"
        r"|rank_?\d+|lora_?rank_?\d+)[_\-]?",
        " ", stem, flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"[_\-]+", " ", cleaned).strip()
    return cleaned


# ---------------------------------------------------------------------------
# Display name construction
# ---------------------------------------------------------------------------


def _make_display_name(title: str, arch: str, role: str, precision: str, size_mb: float) -> str:
    """Build: '<Title> [<Prec> . <Size>]'"""
    use_title = title.strip()
    if arch == ARCH_SD35_CHECKPOINT and "sd3.5" not in use_title.lower():
        use_title = f"SD3.5 {use_title}".strip()

    # If no meaningful title, build from arch+role
    if not use_title or " " not in use_title:
        prefix = _ARCH_PREFIX.get(arch, "")
        role_sfx = _ROLE_SUFFIX.get(role, "")
        parts = [x for x in (prefix, role_sfx) if x]
        use_title = " ".join(parts) or use_title or "Model"

    size_str = f"{size_mb / 1000:.1f} GB" if size_mb >= 1000 else f"{size_mb:.0f} MB"
    if precision and size_mb:
        tag = f"[{precision} · {size_str}]"
    elif precision:
        tag = f"[{precision}]"
    elif size_mb:
        tag = f"[{size_str}]"
    else:
        tag = ""

    return f"{use_title} {tag}".strip()
