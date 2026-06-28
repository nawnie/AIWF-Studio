"""Detect bitsandbytes NF4/FP4 Flux safetensors.

These checkpoints store 4-bit linear weights plus ``quant_state`` sidecars
(``*.quant_state.bitsandbytes__nf4``). Diffusers-format layouts can load
through ``BitsAndBytesConfig(load_in_4bit=True)``. Packed Comfy/Forge Flux
single-file layouts need a custom loader because their packed original keys are
not accepted by the standard Diffusers single-file converter.
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
_FLOAT8_COMPUTE_DTYPE_NAMES = {
    "float8_e4m3fn",
    "float8_e4m3fnuz",
    "float8_e5m2",
    "float8_e5m2fnuz",
}


@dataclass(frozen=True)
class Bnb4BitReport:
    path: str
    format: str  # not_safetensors | diffusers_safetensors | bnb_nf4 | bnb_fp4 | fp4_storage | nvfp4_storage | unreadable
    quant_type: str = ""  # nf4 | fp4 | nvfp4
    layout: str = ""  # diffusers_or_unknown | flux_original_bnb
    tensor_count: int = 0
    quantized_linear_layers: int = 0
    metadata_keys: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def is_bnb_4bit(self) -> bool:
        return self.format in {"bnb_nf4", "bnb_fp4"}

    @property
    def is_storage_only_4bit(self) -> bool:
        return self.format in {"fp4_storage", "nvfp4_storage"}

    @property
    def needs_custom_flux_bnb_loader(self) -> bool:
        return self.layout == "flux_original_bnb"

    @property
    def supports_diffusers_single_file(self) -> bool:
        return not self.needs_custom_flux_bnb_loader

    @property
    def load_format_label(self) -> str:
        if self.format == "bnb_nf4":
            return "nf4"
        if self.format == "bnb_fp4":
            return "fp4"
        return "safetensors"


@dataclass(frozen=True)
class Bnb4BitRuntime:
    available: bool
    backend: str
    compute_dtype: str
    reason: str = ""


def runtime_for_ada_4bit() -> Bnb4BitRuntime:
    """Best local 4-bit runtime for RTX 40-series image transformers.

    Ada cards do not expose a native NVFP4 image-inference path. The practical
    route is bitsandbytes pre-quantized NF4/FP4 safetensors with BF16 compute
    when CUDA/BF16 are available, otherwise FP16 compute.
    """
    try:
        import bitsandbytes  # noqa: F401
    except Exception as exc:
        return Bnb4BitRuntime(
            available=False,
            backend="bitsandbytes_4bit",
            compute_dtype="",
            reason=f"bitsandbytes unavailable: {exc}",
        )
    if not torch.cuda.is_available():
        return Bnb4BitRuntime(
            available=False,
            backend="bitsandbytes_4bit",
            compute_dtype="",
            reason="CUDA is required for bitsandbytes 4-bit image inference.",
        )
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return Bnb4BitRuntime(
        available=True,
        backend="bitsandbytes_4bit",
        compute_dtype=str(compute_dtype).replace("torch.", ""),
        reason="Ada path: bnb NF4/FP4 weights with BF16/FP16 compute; not NVFP4.",
    )


def normalize_bnb_4bit_compute_dtype(compute_dtype: torch.dtype) -> torch.dtype:
    """Return a bitsandbytes-safe compute dtype for 4-bit image inference."""
    dtype_name = str(compute_dtype).replace("torch.", "")
    if dtype_name not in _FLOAT8_COMPUTE_DTYPE_NAMES:
        return compute_dtype
    try:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
    except Exception:
        pass
    return torch.float16


def _read_safetensors_header(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    with path.open("rb") as handle:
        header_size = struct.unpack("<Q", handle.read(8))[0]
        if header_size > 10 * 1024 * 1024:
            raise ValueError(f"safetensors header too large ({header_size} bytes)")
        header = json.loads(handle.read(header_size).decode("utf-8"))
    metadata = header.get("__metadata__", {}) or {}
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    return metadata, tensors


def _quant_type_from_filename(filename: str) -> str:
    name = filename.lower()
    if "nvfp4" in name:
        return "nvfp4"
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
        if "nvfp4" in text:
            return "nvfp4"
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
        metadata, tensors = _read_safetensors_header(resolved)
    except Exception as exc:
        return Bnb4BitReport(path=str(resolved), format="unreadable", warnings=(str(exc),))

    keys = list(tensors)
    key_set = set(keys)
    metadata_keys = tuple(sorted(str(key) for key in metadata.keys()))
    quant_type = ""
    saw_bnb_sidecar = False
    quantized_layers = 0

    for key in keys:
        lower = key.lower()
        if any(marker in lower for marker in _BNB_NF4_MARKERS):
            quant_type = "nf4"
            saw_bnb_sidecar = True
        elif any(marker in lower for marker in _BNB_FP4_MARKERS):
            quant_type = quant_type or "fp4"
            saw_bnb_sidecar = True
        if lower.endswith(".weight") and ".quant_state." not in lower:
            base = key[: -len(".weight")]
            if any(
                sidecar in key_set
                for marker in ("bitsandbytes__nf4", "bitsandbytes__fp4")
                for sidecar in (f"{key}.quant_state.{marker}", f"{base}.quant_state.{marker}")
            ):
                quantized_layers += 1

    if not quant_type:
        quant_type = _quant_type_from_metadata(metadata)
    if not quant_type:
        quant_type = _quant_type_from_filename(resolved.name)

    if quant_type == "nf4":
        fmt = "bnb_nf4"
    elif quant_type == "fp4" and saw_bnb_sidecar:
        fmt = "bnb_fp4"
    elif quant_type == "nvfp4":
        fmt = "nvfp4_storage"
    elif quant_type == "fp4":
        fmt = "fp4_storage"
    else:
        fmt = "diffusers_safetensors"

    if fmt == "diffusers_safetensors" and quantized_layers:
        warnings.append("Found bnb quant_state sidecars but could not infer nf4/fp4 type.")
    layout = ""
    if fmt in {"bnb_nf4", "bnb_fp4"}:
        has_flux_original_keys = any(key.startswith(("double_blocks.", "single_blocks.")) for key in keys)
        has_diffusers_flux_keys = any(key.startswith(("transformer_blocks.", "single_transformer_blocks.")) for key in keys)
        if has_flux_original_keys and not has_diffusers_flux_keys:
            layout = "flux_original_bnb"
            warnings.append(
                "Packed Flux BNB single-file layout detected; the standard Diffusers single-file converter cannot load it."
            )
        else:
            layout = "diffusers_or_unknown"

    return Bnb4BitReport(
        path=str(resolved),
        format=fmt,
        quant_type=quant_type,
        layout=layout,
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
    compute_dtype = normalize_bnb_4bit_compute_dtype(compute_dtype)
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
