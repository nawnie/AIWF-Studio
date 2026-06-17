from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_SCALE_SUFFIXES = (
    ".weight_scale",
    ".weight_scale_2",
    ".scale_weight",
    ".pre_quant_scale",
    ".input_scale",
    ".scale_input",
)


@dataclass(frozen=True)
class WanQuantReport:
    path: str
    format: str
    tensor_count: int = 0
    quantized_weight_count: int = 0
    quantized_linear_layers: int = 0
    scale_tensor_count: int = 0
    weight_scale_count: int = 0
    input_scale_count: int = 0
    pre_quant_scale_count: int = 0
    missing_scale_keys: tuple[str, ...] = ()
    unsupported_quant_formats: tuple[str, ...] = ()
    metadata_keys: tuple[str, ...] = ()
    estimated_quantized_weight_mb: float = 0.0
    estimated_bf16_expanded_mb: float = 0.0
    warnings: tuple[str, ...] = ()

    @property
    def is_comfy_fp8(self) -> bool:
        return self.format == "comfy_fp8"

    @property
    def demo_ready(self) -> bool:
        return self.is_comfy_fp8 and not self.missing_scale_keys and not self.unsupported_quant_formats


def inspect_wan_quant_file(path: Path | str) -> WanQuantReport:
    p = Path(path)
    if p.suffix.lower() != ".safetensors":
        return WanQuantReport(path=str(p), format="not_safetensors")

    try:
        from safetensors import safe_open
    except Exception as exc:
        return WanQuantReport(
            path=str(p),
            format="unreadable",
            warnings=(f"safetensors is unavailable: {exc}",),
        )

    warnings: list[str] = []
    unsupported: set[str] = set()
    missing_scales: list[str] = []
    tensor_count = 0
    quantized_weight_count = 0
    quantized_linear_layers = 0
    scale_tensor_count = 0
    weight_scale_count = 0
    input_scale_count = 0
    pre_quant_scale_count = 0
    quantized_elements = 0
    metadata_keys: tuple[str, ...] = ()

    try:
        with safe_open(str(p), framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            metadata = handle.metadata() or {}
            metadata_keys = tuple(sorted(str(k) for k in metadata.keys()))
            unsupported.update(_unsupported_formats_from_metadata(metadata))

            for key in sorted(keys):
                tensor_count += 1
                key_l = key.lower()
                if key_l.endswith(_SCALE_SUFFIXES):
                    scale_tensor_count += 1
                    if key_l.endswith((".weight_scale", ".weight_scale_2", ".scale_weight")):
                        weight_scale_count += 1
                    if key_l.endswith((".input_scale", ".scale_input")):
                        input_scale_count += 1
                    if key_l.endswith(".pre_quant_scale"):
                        pre_quant_scale_count += 1
                    continue

                if key_l.endswith(".comfy_quant"):
                    continue

                try:
                    tensor_slice = handle.get_slice(key)
                    dtype_name = str(tensor_slice.get_dtype()).upper()
                except Exception:
                    continue

                if not dtype_name.startswith("F8"):
                    continue

                quantized_weight_count += 1
                if key_l.endswith(".weight"):
                    quantized_linear_layers += 1
                    base = key.removesuffix(".weight")
                    if not _has_any(keys, base, (".weight_scale", ".scale_weight", ".weight_scale_2")):
                        missing_scales.append(f"{base}.weight_scale")

                count = 1
                try:
                    for dim in tensor_slice.get_shape():
                        count *= int(dim)
                except Exception:
                    count = 0
                quantized_elements += count

            if quantized_weight_count and not weight_scale_count:
                warnings.append("FP8 tensors found without weight scale sidecars.")
            if any(k.lower().endswith(".comfy_quant") for k in keys) and not quantized_weight_count:
                warnings.append("Comfy quant metadata found, but no FP8 tensor weights were detected.")
    except Exception as exc:
        return WanQuantReport(path=str(p), format="unreadable", warnings=(str(exc),))

    fmt = "comfy_fp8" if quantized_weight_count or scale_tensor_count else "diffusers_safetensors"
    return WanQuantReport(
        path=str(p),
        format=fmt,
        tensor_count=tensor_count,
        quantized_weight_count=quantized_weight_count,
        quantized_linear_layers=quantized_linear_layers,
        scale_tensor_count=scale_tensor_count,
        weight_scale_count=weight_scale_count,
        input_scale_count=input_scale_count,
        pre_quant_scale_count=pre_quant_scale_count,
        missing_scale_keys=tuple(missing_scales),
        unsupported_quant_formats=tuple(sorted(unsupported)),
        metadata_keys=metadata_keys,
        estimated_quantized_weight_mb=round(quantized_elements / 1024**2, 3),
        estimated_bf16_expanded_mb=round((quantized_elements * 2) / 1024**2, 3),
        warnings=tuple(warnings),
    )


def _has_any(keys: set[str], base: str, suffixes: tuple[str, ...]) -> bool:
    return any(f"{base}{suffix}" in keys for suffix in suffixes)


def _unsupported_formats_from_metadata(metadata: dict[str, Any]) -> set[str]:
    unsupported: set[str] = set()
    for key in ("_quantization_metadata", "quantization_metadata"):
        raw = metadata.get(key)
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except Exception:
            unsupported.add("unparseable_metadata")
            continue
        for value in _walk_values(parsed):
            text = str(value).lower()
            if "float8" in text or "fp8" in text or "e4m3" in text or "e5m2" in text:
                continue
            if "int" in text or "nf4" in text or "fp4" in text:
                unsupported.add(str(value))
    return unsupported


def _walk_values(value: Any):
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_values(item)
    else:
        yield value
