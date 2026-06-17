"""AIWF-native FP8 linear runtime.

This module is intentionally narrow: it handles Wan-style FP8 Linear weights
without trying to become a general tensor subclass or a Comfy runtime clone.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable

logger = logging.getLogger(__name__)


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def cuda_supports_tensorcore_fp8(torch_module: Any) -> bool:
    """Return whether the active CUDA device has native FP8 tensor-core support."""
    try:
        if not torch_module.cuda.is_available():
            return False
        major, minor = torch_module.cuda.get_device_capability()
        return (int(major), int(minor)) >= (8, 9)
    except Exception:
        return False


def torch_native_fp8_available() -> bool:
    try:
        import torch

        return bool(
            cuda_supports_tensorcore_fp8(torch)
            and hasattr(torch, "float8_e4m3fn")
            and hasattr(torch, "_scaled_mm")
        )
    except Exception:
        return False


def tensor_diag(tensor: Any) -> dict[str, Any]:
    """Return tensor metadata safe for diagnostics; never include tensor values."""
    device = getattr(tensor, "device", None)
    try:
        stride = [int(v) for v in tensor.stride()]
    except Exception:
        stride = None
    try:
        is_contiguous = bool(tensor.is_contiguous())
    except Exception:
        is_contiguous = None
    return {
        "shape": [int(v) for v in getattr(tensor, "shape", ())],
        "stride": stride,
        "dtype": str(getattr(tensor, "dtype", "")).replace("torch.", ""),
        "device_type": getattr(device, "type", None),
        "device_index": getattr(device, "index", None),
        "is_contiguous": is_contiguous,
    }


def fp8_scaled_mm_failure_payload(
    exc: BaseException,
    *,
    layer: Any,
    input_tensor: Any,
    x8: Any,
    weight_t: Any,
    scale_a: Any,
    scale_b: Any,
    rows: int,
    padded_rows: int,
    pad_m: int,
) -> dict[str, Any]:
    message = str(exc)
    if len(message) > 800:
        message = message[:797] + "..."
    return {
        "error": {
            "type": type(exc).__name__,
            "message": message,
        },
        "layer": {
            "class": layer.__class__.__name__,
            "in_features": int(getattr(layer, "in_features", 0)),
            "out_features": int(getattr(layer, "out_features", 0)),
            "has_bias": getattr(layer, "bias", None) is not None,
        },
        "input": tensor_diag(input_tensor),
        "matmul": {
            "lhs": tensor_diag(x8),
            "rhs": tensor_diag(weight_t),
            "scale_a": tensor_diag(scale_a),
            "scale_b": tensor_diag(scale_b),
            "rows": int(rows),
            "padded_rows": int(padded_rows),
            "pad_m": int(pad_m),
        },
    }


def trace_fp8_scaled_mm_fallback(payload: dict[str, Any]) -> None:
    try:
        from aiwf.dev.diagnostics import trace_safe

        trace_safe(
            "wan.fp8_scaled_mm_fallback",
            "Wan FP8 _scaled_mm fallback",
            **payload,
        )
    except Exception:
        logger.debug("Wan FP8 _scaled_mm diagnostic trace failed.", exc_info=True)


def fp8_generic_fallback_payload(
    reason: str,
    *,
    layer: Any,
    input_tensor: Any,
) -> dict[str, Any]:
    return {
        "error": {
            "type": "FP8Fallback",
            "message": reason,
        },
        "layer": {
            "class": layer.__class__.__name__,
            "in_features": int(getattr(layer, "in_features", 0)),
            "out_features": int(getattr(layer, "out_features", 0)),
            "has_bias": getattr(layer, "bias", None) is not None,
        },
        "input": tensor_diag(input_tensor),
        "matmul": {},
    }


class AIWFFP8Linear:
    """Quant-aware FP8 Linear using native ``torch._scaled_mm`` when legal."""

    def __new__(cls, *args: Any, **kwargs: Any):
        import torch

        if not issubclass(cls, torch.nn.Module):
            cls = type(cls.__name__, (cls, torch.nn.Module), {})
        return super().__new__(cls)

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        *,
        strict_exception_cls: type[Exception] = RuntimeError,
        fallback_tracer: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        import torch

        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self._strict_exception_cls = strict_exception_cls
        self._fallback_tracer = fallback_tracer or trace_fp8_scaled_mm_fallback
        self.weight = torch.nn.Parameter(
            torch.empty((out_features, in_features), device="meta", dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        if bias:
            self.bias = torch.nn.Parameter(torch.empty((out_features,), device="meta"), requires_grad=False)
        else:
            self.register_parameter("bias", None)
        self.register_buffer("weight_scale", torch.tensor(1.0, dtype=torch.float32), persistent=True)
        self.register_buffer("input_scale", None, persistent=True)
        self.orig_dtype = torch.bfloat16
        self.fast_mm_calls = 0
        self.fallback_calls = 0
        self.last_fallback_reason: str | None = None

    def load_quantized_weight(
        self,
        qweight: Any,
        weight_scale: Any,
        input_scale: Any | None = None,
        *,
        orig_dtype: Any | None = None,
        bias: Any | None = None,
    ) -> None:
        import torch

        self.weight = torch.nn.Parameter(qweight.detach(), requires_grad=False)
        self.weight_scale = weight_scale.detach().to(dtype=torch.float32)
        if input_scale is not None:
            self.input_scale = input_scale.detach().to(dtype=torch.float32)
        if bias is not None:
            self.bias = torch.nn.Parameter(bias.detach(), requires_grad=False)
        if orig_dtype is not None:
            self.orig_dtype = orig_dtype

    def _record_fallback(self, reason: str, *, payload: dict[str, Any] | None = None, exc: Exception | None = None) -> None:
        self.fallback_calls += 1
        self.last_fallback_reason = reason
        if not getattr(self, "_scaled_mm_warned", False):
            if payload is None:
                payload = {
                    "error": {"type": "FP8Fallback", "message": reason},
                    "layer": {
                        "class": self.__class__.__name__,
                        "in_features": self.in_features,
                        "out_features": self.out_features,
                        "has_bias": self.bias is not None,
                    },
                }
            logger.warning("%s fallback: %s", self.__class__.__name__, reason)
            self._fallback_tracer(payload)
            self._scaled_mm_warned = True
        if _env_flag("AIWF_WAN_STRICT_FP8", default=False):
            raise self._strict_exception_cls(
                "Wan strict FP8 mode refused slow bf16 fallback. "
                f"Reason: {reason}"
            ) from exc

    def forward(self, input: Any) -> Any:
        import torch
        import torch.nn.functional as F

        can_scaled_mm = (
            input.is_cuda
            and self.weight.is_cuda
            and hasattr(torch, "_scaled_mm")
            and self.in_features % 16 == 0
            and self.out_features % 16 == 0
        )
        if can_scaled_mm:
            original_shape = input.shape[:-1]
            x = input.reshape(-1, self.in_features).contiguous()
            m, _k = x.shape
            pad_m = (16 - m % 16) % 16
            if pad_m:
                x = F.pad(x, (0, 0, 0, pad_m))
            scale_a = self._input_scale_for(x)
            x8_source = x / scale_a if getattr(self, "input_scale", None) is not None else x
            x8 = x8_source.clamp(-448, 448).to(torch.float8_e4m3fn).contiguous()
            # cuBLASLt FP8 scaled matmul requires row-major lhs and
            # column-major rhs. ``self.weight.t()`` already has the required
            # column-major stride; making it contiguous changes it back to
            # row-major and forces the slow bf16 fallback.
            weight_t = self.weight.t()
            try:
                scale_b = self.weight_scale.to(device=x.device, dtype=torch.float32)
                y = torch._scaled_mm(
                    x8,
                    weight_t,
                    scale_a=scale_a,
                    scale_b=scale_b,
                    out_dtype=input.dtype
                    if input.dtype in (torch.float16, torch.bfloat16)
                    else torch.bfloat16,
                )
                if pad_m:
                    y = y[:m, :]
                if self.bias is not None:
                    y = y + self.bias.to(device=y.device, dtype=y.dtype)
                self.fast_mm_calls += 1
                return y.reshape(*original_shape, self.out_features)
            except Exception as exc:
                payload = fp8_scaled_mm_failure_payload(
                    exc,
                    layer=self,
                    input_tensor=input,
                    x8=x8,
                    weight_t=weight_t,
                    scale_a=scale_a,
                    scale_b=scale_b,
                    rows=m,
                    padded_rows=x8.shape[0],
                    pad_m=pad_m,
                )
                reason = f"_scaled_mm failed ({payload['error']['type']}: {payload['error']['message']})"
                self._record_fallback(reason, payload=payload, exc=exc)
        else:
            reasons = []
            if not getattr(input, "is_cuda", False):
                reasons.append("input is not CUDA")
            if not getattr(self.weight, "is_cuda", False):
                reasons.append("weight is not CUDA")
            if not hasattr(torch, "_scaled_mm"):
                reasons.append("torch._scaled_mm is unavailable")
            if self.in_features % 16 != 0:
                reasons.append("in_features is not divisible by 16")
            if self.out_features % 16 != 0:
                reasons.append("out_features is not divisible by 16")
            reason = "; ".join(reasons) or "scaled_mm preconditions were not met"
            self._record_fallback(
                reason,
                payload=fp8_generic_fallback_payload(reason, layer=self, input_tensor=input),
            )

        weight = (self.weight.float() * self.weight_scale.float()).contiguous()
        return F.linear(input, weight.to(device=input.device, dtype=input.dtype), self.bias)

    def _input_scale_for(self, x: Any) -> Any:
        import torch

        input_scale = getattr(self, "input_scale", None)
        if input_scale is None:
            return torch.ones((), device=x.device, dtype=torch.float32)
        return input_scale.to(device=x.device, dtype=torch.float32)


class FP8ScaledLinear(AIWFFP8Linear):
    """Compatibility name for existing Wan FP8 loader/tests."""


def collect_fp8_linear_metrics(*roots: Any) -> dict[str, Any]:
    layers = 0
    fast_mm_calls = 0
    fallback_calls = 0
    fallback_layers = 0
    fallback_reasons: list[str] = []
    seen: set[int] = set()
    for root in roots:
        if root is None or not hasattr(root, "modules"):
            continue
        try:
            modules = root.modules()
        except Exception:
            continue
        for module in modules:
            if not isinstance(module, AIWFFP8Linear):
                continue
            ident = id(module)
            if ident in seen:
                continue
            seen.add(ident)
            layers += 1
            fast_mm_calls += int(getattr(module, "fast_mm_calls", 0) or 0)
            module_fallbacks = int(getattr(module, "fallback_calls", 0) or 0)
            fallback_calls += module_fallbacks
            if module_fallbacks > 0:
                fallback_layers += 1
                reason = str(getattr(module, "last_fallback_reason", "") or "")
                if reason and reason not in fallback_reasons:
                    fallback_reasons.append(reason)
    return {
        "fp8_linear_layers": layers,
        "fp8_fast_mm_calls": fast_mm_calls,
        "fp8_fallback_calls": fallback_calls,
        "fp8_fallback_layers": fallback_layers,
        "fp8_fallback_reasons": fallback_reasons[:10],
        "fp8_strict_mode": _env_flag("AIWF_WAN_STRICT_FP8", default=False),
        "fp8_native_available": torch_native_fp8_available(),
    }
