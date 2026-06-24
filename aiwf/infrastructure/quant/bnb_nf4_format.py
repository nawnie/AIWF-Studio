"""Detect and load Comfy/Forge bitsandbytes NF4/FP4 Flux safetensors.

These checkpoints store 4-bit linear weights plus ``quant_state`` sidecars
(``*.quant_state.bitsandbytes__nf4``). Diffusers loads them through
``BitsAndBytesConfig(load_in_4bit=True)`` with ``pre_quantized=True``.
"""
from __future__ import annotations

import json
import logging
import struct
from dataclasses import dataclass
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_BNB_NF4_MARKERS = ("bitsandbytes__nf4",)
_BNB_FP4_MARKERS = ("bitsandbytes__fp4",)


@dataclass(frozen=True)
class Bnb4BitReport:
    path: str
    format: str  # not_safetensors | diffusers_safetensors | bnb_nf4 | bnb_fp4 | unreadable
    quant_type: str = ""  # nf4 | fp4
    tensor_count: int = 0
    quantized_linear_layers: int = 0
    metadata_keys: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def is_bnb_4bit(self) -> bool:
        return self.format in {"bnb_nf4", "bnb_fp4"}

    @property
    def load_format_label(self) -> str:
        if self.format == "bnb_nf4":
            return "nf4"
        if self.format == "bnb_fp4":
            return "fp4"
        return "safetensors"


def _read_safetensors_header(path: Path) -> tuple[dict[str, object], list[str]]:
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        if header_size > 10 * 1024 * 1024:
            raise ValueError(f"safetensors header too large ({header_size} bytes)")
        header = json.loads(handle.read(header_size).decode("utf-8"))
    metadata = header.get("__metadata__", {}) or {}
    keys = [key for key in header if key != "__metadata__"]
    return metadata, keys


def _quant_type_from_filename(filename: str) -> str:
    name = filename.lower()
    if "nf4" in name:
        return "nf4"
    if "fp4" in name:
        return "fp4"
    return ""


def _quant_type_from_metadata(metadata: dict[str, object]) -> str:
    for key in ("fp", "format", "quantization", "quant_type"):
        raw = metadata.get(key)
        if raw is None:
            continue
        text = str(raw).lower()
        if "nf4" in text:
            return "nf4"
        if "fp4" in text:
            return "fp4"
    return ""


def inspect_bnb_4bit_safetensors(path: Path | str) -> Bnb4BitReport:
    """Header-only scan for bitsandbytes 4-bit safetensors (no weight load)."""
    resolved = Path(path)
    if resolved.suffix.lower() != ".safetensors" or not resolved.is_file():
        return Bnb4BitReport(path=str(resolved), format="not_safetensors")

    warnings: list[str] = []
    try:
        metadata, keys = _read_safetensors_header(resolved)
    except Exception as exc:
        return Bnb4BitReport(path=str(resolved), format="unreadable", warnings=(str(exc),))

    metadata_keys = tuple(sorted(str(key) for key in metadata.keys()))
    quant_type = ""
    quantized_layers = 0

    for key in keys:
        lower = key.lower()
        if any(marker in lower for marker in _BNB_NF4_MARKERS):
            quant_type = "nf4"
        elif any(marker in lower for marker in _BNB_FP4_MARKERS):
            quant_type = quant_type or "fp4"
        if lower.endswith(".weight") and ".quant_state." not in lower:
            base = key[: -len(".weight")]
            if any(f"{base}.quant_state.{marker}" in keys for marker in ("bitsandbytes__nf4", "bitsandbytes__fp4")):
                quantized_layers += 1

    if not quant_type:
        quant_type = _quant_type_from_metadata(metadata)
    if not quant_type:
        quant_type = _quant_type_from_filename(resolved.name)

    if quant_type == "nf4":
        fmt = "bnb_nf4"
    elif quant_type == "fp4":
        fmt = "bnb_fp4"
    else:
        fmt = "diffusers_safetensors"

    if fmt == "diffusers_safetensors" and quantized_layers:
        warnings.append("Found bnb quant_state sidecars but could not infer nf4/fp4 type.")

    return Bnb4BitReport(
        path=str(resolved),
        format=fmt,
        quant_type=quant_type,
        tensor_count=len(keys),
        quantized_linear_layers=quantized_layers,
        metadata_keys=metadata_keys,
        warnings=tuple(warnings),
    )


def is_bnb_4bit_safetensors(path: Path | str) -> bool:
    return inspect_bnb_4bit_safetensors(path).is_bnb_4bit


def build_bnb_4bit_quantization_config(
    report: Bnb4BitReport,
    *,
    compute_dtype: torch.dtype,
):
    """Return a Diffusers ``BitsAndBytesConfig`` for pre-quantized NF4/FP4 loads."""
    if not report.is_bnb_4bit:
        return None
    try:
        from diffusers import BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "bitsandbytes NF4/FP4 safetensors need diffusers BitsAndBytesConfig support."
        ) from exc

    quant_type = report.quant_type or ("nf4" if report.format == "bnb_nf4" else "fp4")
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def resolve_transformer_load_format(path: Path, *, suffix: str | None = None) -> str:
    """Human-readable load format label for logs/UI."""
    ext = (suffix or path.suffix).lower()
    if ext == ".gguf":
        return "gguf"
    if ext != ".safetensors":
        return ext.lstrip(".") or "unknown"
    report = inspect_bnb_4bit_safetensors(path)
    if report.is_bnb_4bit:
        return report.load_format_label
    return "safetensors"