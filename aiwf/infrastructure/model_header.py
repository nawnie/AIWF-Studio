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
ARCH_CLIP                = "clip"
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
    ("conditioner.embedders.0.transformer",                      ARCH_SDXL_CHECKPOINT,     ROLE_CHECKPOINT),
    ("model.diffusion_model.joint_blocks.0",                     ARCH_SD35_CHECKPOINT,     ROLE_CHECKPOINT),
    ("model.diffusion_model.input_blocks.0.0.weight",            ARCH_SD_CHECKPOINT,       ROLE_CHECKPOINT),
    ("cond_stage_model.model.transformer.resblocks.0",           ARCH_SD_CHECKPOINT,       ROLE_CHECKPOINT),
    ("lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight", ARCH_SD_LORA,        ROLE_LORA),
]

_GGUF_ARCH_MAP: dict[str, str] = {
    "wan":       ARCH_WAN_TRANSFORMER,
    "t5encoder": ARCH_UMT5_ENCODER,
    "clip":      ARCH_CLIP,
}

_ST_DTYPE_DISPLAY: dict[str, str] = {
    "F8_E4M3": "FP8", "F8_E5M2": "FP8",
    "F16": "F16", "BF16": "BF16", "F32": "F32",
    "U8": "", "I8": "", "I32": "", "I64": "", "U16": "",
}

_FOLDER_CLUES: list[tuple[str, str, str]] = [
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
    ARCH_CLIP:                "CLIP",
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
        return self.arch in (ARCH_UMT5_ENCODER, ARCH_CLIP) or self.role == ROLE_TEXT_ENCODER

    def is_t5xxl(self) -> bool:
        """True if this looks like a T5-XXL (Flux/SD3) file — NOT for Wan."""
        stem = Path(self.filename).stem.lower()
        return (stem.startswith("t5xxl") or stem in ("t5xxl_fp16", "t5xxl_fp8_e4m3fn")) \
            and "umt5" not in self.filename.lower()

    def size_label(self) -> str:
        if self.size_mb >= 1000:
            return f"{self.size_mb / 1000:.1f} GB"
        return f"{self.size_mb:.0f} MB"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def read_model_info(path) -> ModelInfo:
    """Read header metadata from a model file without loading weights."""
    p = Path(path).resolve()
    size_mb = _file_size_mb(p)
    suffix = p.suffix.lower()
    if suffix == ".gguf":
        return _read_gguf(p, size_mb)
    elif suffix == ".safetensors":
        return _read_safetensors(p, size_mb)
    else:
        arch, role = _arch_from_folder_and_filename(p)
        return ModelInfo(
            path=str(p), filename=p.name, arch=arch, role=role,
            precision="", size_mb=size_mb,
            display_name=_make_display_name(_clean_stem(p.stem), arch, role, "", size_mb),
        )


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

        # Refine arch / role
        if arch == ARCH_UMT5_ENCODER or "umt5" in title.lower() or "umt5" in p.name.lower():
            arch = ARCH_UMT5_ENCODER
            role = ROLE_TEXT_ENCODER
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


def _arch_from_st_meta_and_keys(meta: dict, tensor_keys: list, p: Path) -> tuple:
    # 1. modelspec.architecture
    spec_arch = meta.get("modelspec.architecture", "").lower()
    if "wan" in spec_arch:
        return ARCH_WAN_TRANSFORMER_FP8, _role_from_meta_and_filename(meta, p.name)

    # 2. Tensor key patterns (sample first 40)
    sample = set(tensor_keys[:40])
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
    for fragment, arch, role in _FOLDER_CLUES:
        if f"/{fragment}/" in path_str or path_str.endswith(f"/{fragment}"):
            return arch, role or _role_from_filename(p.name)
    if "umt5" in p.name.lower():
        return ARCH_UMT5_ENCODER, ROLE_TEXT_ENCODER
    return ARCH_UNKNOWN, _role_from_filename(p.name)


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
