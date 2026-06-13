"""Wan GGUF transformer loader — mmap quantized weights, on-the-fly dequant matmul."""
from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Any

import gguf
import torch
import torch.nn as nn
import torch.nn.functional as F

from aiwf.infrastructure.wan.gguf_dequant import (
    GGMLTensor,
    dequantize_tensor,
    is_quantized,
)
from aiwf.infrastructure.wan.gguf_policy import PrecisionTier, classify_gguf_tensor

logger = logging.getLogger(__name__)

_GGUF_DEQUANT_DEVICE = os.environ.get("AIWF_WAN_GGUF_DEQUANT_DEVICE", "auto").strip().lower()


class WanUnavailable(RuntimeError):
    """Raised when Wan GGUF weights cannot be loaded."""


def _module_parent_and_name(root: nn.Module, module_path: str):
    parent = root
    parts = module_path.split(".")
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]

_HANDLE_PREFIX = "model.diffusion_model."


def _get_orig_shape(reader: gguf.GGUFReader, tensor_name: str) -> torch.Size | None:
    field_key = f"comfy.gguf.orig_shape.{tensor_name}"
    field = reader.get_field(field_key)
    if field is None:
        return None
    if len(field.types) != 2 or field.types[0] != gguf.GGUFValueType.ARRAY:
        return None
    return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))


def mmap_gguf_state_dict(path: Path) -> dict[str, Any]:
    """Read a Wan GGUF file as mmap'd tensors (no full dequant expand)."""
    reader = gguf.GGUFReader(str(path))
    tensor_names = {tensor.name for tensor in reader.tensors}
    has_prefix = any(name.startswith(_HANDLE_PREFIX) for name in tensor_names)
    prefix_len = len(_HANDLE_PREFIX)

    state_dict: dict[str, Any] = {}
    qtype_counts: dict[str, int] = {}
    for tensor in reader.tensors:
        tensor_name = tensor.name
        if has_prefix:
            if not tensor_name.startswith(_HANDLE_PREFIX):
                continue
            sd_key = tensor_name[prefix_len:]
        else:
            sd_key = tensor_name

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            torch_tensor = torch.from_numpy(tensor.data)

        shape = _get_orig_shape(reader, tensor_name)
        if shape is None:
            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))

        if tensor.tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            torch_tensor = torch_tensor.view(*shape)
            state_dict[sd_key] = torch_tensor
        else:
            state_dict[sd_key] = GGMLTensor(
                torch_tensor,
                tensor_type=tensor.tensor_type,
                tensor_shape=shape,
            )

        if len(shape) <= 1 and tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
            state_dict[sd_key] = dequantize_tensor(state_dict[sd_key], dtype=torch.float32)

        qname = getattr(tensor.tensor_type, "name", repr(tensor.tensor_type))
        qtype_counts[qname] = qtype_counts.get(qname, 0) + 1

    logger.info("GGUF %s qtypes: %s", path.name, ", ".join(f"{k} ({v})" for k, v in sorted(qtype_counts.items())))
    return state_dict


class GGUFLinear(nn.Module):
    """Linear layer that dequantizes GGML weights on each forward (Comfy GGMLOps style)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight: torch.nn.Parameter | None = None
        self._cuda_dequant_warned = False
        if bias:
            self.bias: torch.nn.Parameter | None = None
        else:
            self.register_parameter("bias", None)

    def _apply(self, fn):
        """Move bias only — mmap'd GGUF weights stay on CPU until forward dequant."""
        if self.bias is not None:
            self.bias = nn.Parameter(fn(self.bias), requires_grad=False)
        return self

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.weight is None:
            raise RuntimeError("GGUFLinear weight was not assigned during load.")
        weight_src = self.weight
        weight = None
        if (
            input.is_cuda
            and is_quantized(weight_src)
            and _GGUF_DEQUANT_DEVICE not in {"cpu", "host"}
        ):
            try:
                # Move the smaller quantized payload to CUDA, then dequantize there.
                # The old path dequantized the full bf16 matrix on CPU and copied
                # that much larger tensor over PCIe for every Linear call.
                gpu_weight_src = weight_src.to(device=input.device, non_blocking=True)
                weight = dequantize_tensor(gpu_weight_src, dtype=input.dtype).contiguous()
            except Exception as exc:
                if not self._cuda_dequant_warned:
                    logger.warning(
                        "GGUF CUDA dequant failed (%s); falling back to CPU dequant for this layer.",
                        exc,
                    )
                    self._cuda_dequant_warned = True
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                weight = None

        if weight is None:
            if getattr(weight_src, "device", None) is not None and weight_src.device.type != "cpu":
                weight_src = weight_src.cpu()
            weight = dequantize_tensor(weight_src, dtype=input.dtype)
            weight = weight.to(device=input.device, dtype=input.dtype).contiguous()
        bias = None
        if self.bias is not None:
            bias = self.bias.to(device=input.device, dtype=input.dtype)
        return F.linear(input, weight, bias)


def _replace_linear_with_gguf(root: nn.Module, module_path: str) -> GGUFLinear:
    parent, name = _module_parent_and_name(root, module_path)
    current = parent[int(name)] if name.isdigit() else getattr(parent, name)
    if isinstance(current, GGUFLinear):
        return current
    if not isinstance(current, nn.Linear):
        raise WanUnavailable(
            f"Expected nn.Linear at '{module_path}' for GGUF load, found {type(current).__name__}."
        )
    replacement = GGUFLinear(current.in_features, current.out_features, bias=current.bias is not None)
    if name.isdigit():
        parent[int(name)] = replacement
    else:
        setattr(parent, name, replacement)
    return replacement


def load_gguf_transformer_weights(
    transformer: nn.Module,
    path: Path,
    *,
    torch_dtype: torch.dtype,
) -> tuple[list[str], list[str]]:
    """Load Wan GGUF transformer weights into a diffusers WanTransformer3DModel."""
    try:
        raw = mmap_gguf_state_dict(path)
    except Exception as exc:
        raise WanUnavailable(f"Could not mmap GGUF weights from {path.name}: {exc}") from exc

    from aiwf.infrastructure.wan.pipeline import _apply_wan_transformer_key_renames

    renamed = _apply_wan_transformer_key_renames(raw)
    loaded_gguf_keys: set[str] = set()
    gguf_modules: dict[str, GGUFLinear] = {}
    gguf_linear_count = 0
    dense_state: dict[str, Any] = {}
    deferred_dense: list[tuple[str, Any]] = []

    for key, tensor in renamed.items():
        tier = classify_gguf_tensor(key)
        if key.endswith(".weight") and tier == PrecisionTier.QUANTIZED and is_quantized(tensor):
            module_path = key.removesuffix(".weight")
            try:
                module = _replace_linear_with_gguf(transformer, module_path)
            except WanUnavailable:
                deferred_dense.append((key, dequantize_tensor(tensor, dtype=torch_dtype)))
                continue
            module.weight = tensor
            gguf_modules[module_path] = module
            loaded_gguf_keys.add(key)
            gguf_linear_count += 1

    for key, tensor in renamed.items():
        if key in loaded_gguf_keys:
            continue
        tier = classify_gguf_tensor(key)

        if key.endswith(".bias"):
            module_path = key.removesuffix(".bias")
            if module_path in gguf_modules:
                bias_tensor = tensor
                if is_quantized(bias_tensor):
                    bias_tensor = dequantize_tensor(bias_tensor, dtype=torch_dtype)
                elif hasattr(bias_tensor, "is_floating_point") and bias_tensor.is_floating_point():
                    bias_tensor = bias_tensor.to(dtype=torch_dtype)
                gguf_modules[module_path].bias = nn.Parameter(bias_tensor, requires_grad=False)
                loaded_gguf_keys.add(key)
                continue

        if is_quantized(tensor) or tier == PrecisionTier.HIGH:
            tensor = dequantize_tensor(tensor, dtype=torch_dtype)
        elif hasattr(tensor, "is_floating_point") and tensor.is_floating_point():
            tensor = tensor.to(dtype=torch_dtype)
        dense_state[key] = tensor

    dense_state.update(dict(deferred_dense))

    missing, unexpected = transformer.load_state_dict(dense_state, strict=False, assign=True)
    missing = [k for k in missing if k not in loaded_gguf_keys]

    logger.info(
        "Loaded %d GGUF quantized linear layers from %s (mmap + on-the-fly dequant).",
        gguf_linear_count,
        path.name,
    )
    return list(missing), list(unexpected)
