"""Wan 2.2 image-to-video backend (diffusers WanImageToVideoPipeline).

Heavy imports (torch/diffusers) are lazy so the app loads fine without the Wan
stack installed or the model downloaded. Tuned for consumer GPUs via CPU
offloading + VAE tiling — slow on 8 GB, but it runs.

Supports:
- Full diffusers layouts (model_index.json + subfolders) via from_pretrained
- Standalone `.safetensors` transformer weights (Comfy diffusion_models style),
  including native Comfy scaled-FP8, loaded into diffusers WanTransformer3DModel.
- Shared sampler path: UMT5, scheduler, dual-stage denoise, VAE decode (like Comfy KSampler).
- Only the transformer loader differs by format; GGUF needs UnetLoaderGGUF-style load
  (`transformer_runtime.py`), not full dequant into diffusers weights.

For dual high/low (Wan 2.2 14B I2V), the caller (WanService) must pre-resolve
and pass `components_base` (the path to the Wan2.2-TI2V-5B-Diffusers layout
or equivalent providing text_encoder/tokenizer/scheduler). The internal
finder is only a best-effort fallback and is intentionally cwd-independent.
"""
from __future__ import annotations

import logging
import os
import inspect
import re
import threading
import time
from pathlib import Path
from typing import Any

from aiwf.infrastructure.torch.vram_budget import apply_cuda_vram_reserve

logger = logging.getLogger(__name__)

_COMFY_FP8_METADATA_SUFFIXES = (
    ".comfy_quant",
    ".weight_scale",
    ".weight_scale_2",
    ".pre_quant_scale",
    ".input_scale",
    ".scale_weight",
    ".scale_input",
)


def _torch_native_fp8_available() -> bool:
    try:
        import torch

        return bool(
            _cuda_supports_tensorcore_fp8(torch)
            and hasattr(torch, "float8_e4m3fn")
            and hasattr(torch, "_scaled_mm")
        )
    except Exception:
        return False


def _cuda_supports_tensorcore_fp8(torch_module: Any) -> bool:
    """Return whether the active CUDA device has native FP8 tensor-core support."""
    try:
        if not torch_module.cuda.is_available():
            return False
        major, minor = torch_module.cuda.get_device_capability()
        return (int(major), int(minor)) >= (8, 9)
    except Exception:
        return False


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _is_native_comfy_fp8_transformer(path: str | None) -> bool:
    if not path:
        return False
    pp = Path(path)
    return pp.suffix.lower() == ".safetensors" and _safetensors_uses_comfy_fp8_quant(pp)


def _is_wan_latent_output(value: Any) -> bool:
    try:
        import torch

        return bool(torch.is_tensor(value) and value.ndim == 5)
    except Exception:
        return False


def _call_accepts_kwarg(callable_obj: Any, name: str) -> bool:
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return True
    for param in sig.parameters.values():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return name in sig.parameters


def _flatten_wan_video_frames(value: Any) -> list:
    if isinstance(value, list):
        if value and isinstance(value[0], list):
            return list(value[0])
        return list(value)

    shape = getattr(value, "shape", None)
    if shape is None:
        return [value]

    if len(shape) == 5:
        value = value[0]
        shape = getattr(value, "shape", None)
    if shape is not None and len(shape) >= 1:
        return [value[i] for i in range(int(shape[0]))]
    return [value]


def _frames_from_wan_pipeline_output(output_frames: Any, *, pipe: Any, decode_latents: Any) -> list:
    if _is_wan_latent_output(output_frames):
        decoded = decode_latents(pipe, output_frames, output_type="pil")
        return _flatten_wan_video_frames(decoded)
    return _flatten_wan_video_frames(output_frames)


def _wan_output_type_for_pipe(pipe: Any) -> str:
    return "latent" if hasattr(pipe, "decode_latents") else "pil"


def _new_wan_euler_scheduler(
    base_scheduler: Any,
    *,
    flow_shift: float,
    sigma_type: str = "beta",
):
    """Build a FlowMatchEulerDiscreteScheduler with the requested sigma spacing.

    sigma_type choices:
      simple      — linear uniform spacing (fast but can look flat at low step counts)
      beta        — beta distribution spacing (smoother motion, best quality at 8-20 steps)
      exponential — more steps in high-noise range (fine detail bias)
      karras      — Karras et al. schedule (familiar from SD, good detail preservation)
    """
    from diffusers import FlowMatchEulerDiscreteScheduler

    config = getattr(base_scheduler, "config", base_scheduler)
    shift = float(flow_shift or getattr(config, "flow_shift", getattr(config, "shift", 5.0)) or 5.0)
    valid = ("simple", "beta", "exponential", "karras")
    sigma_type = sigma_type if sigma_type in valid else "beta"
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=int(getattr(config, "num_train_timesteps", 1000) or 1000),
        shift=shift,
        use_dynamic_shifting=bool(getattr(config, "use_dynamic_shifting", False)),
        time_shift_type=str(getattr(config, "time_shift_type", "exponential") or "exponential"),
        use_karras_sigmas=(sigma_type == "karras"),
        use_exponential_sigmas=(sigma_type == "exponential"),
        use_beta_sigmas=(sigma_type == "beta"),
    )
    logger.info("Wan scheduler: FlowMatch Euler | sigma=%s | shift=%s", sigma_type, shift)
    return scheduler


# Back-compat alias used by older callers / tests
def _new_wan_euler_simple_scheduler(base_scheduler: Any, *, flow_shift: float):
    return _new_wan_euler_scheduler(base_scheduler, flow_shift=flow_shift, sigma_type="simple")


class WanUnavailable(RuntimeError):
    """Raised when the Wan deps or model are missing/unloadable."""


WAN_I2V_A14B_TRANSFORMER_CONFIG = {
    "added_kv_proj_dim": None,
    "attention_head_dim": 128,
    "cross_attn_norm": True,
    "eps": 1e-06,
    "ffn_dim": 13824,
    "freq_dim": 256,
    "in_channels": 36,
    "num_attention_heads": 40,
    "num_layers": 40,
    "out_channels": 16,
    "patch_size": [1, 2, 2],
    "qk_norm": "rms_norm_across_heads",
    "text_dim": 4096,
    "rope_max_seq_len": 1024,
    "pos_embed_seq_len": None,
}

WAN_TI2V_5B_TRANSFORMER_CONFIG = {
    "added_kv_proj_dim": None,
    "attention_head_dim": 128,
    "cross_attn_norm": True,
    "eps": 1e-06,
    "ffn_dim": 14336,
    "freq_dim": 256,
    "in_channels": 48,
    "num_attention_heads": 24,
    "num_layers": 30,
    "out_channels": 48,
    "patch_size": [1, 2, 2],
    "qk_norm": "rms_norm_across_heads",
    "text_dim": 4096,
    "rope_max_seq_len": 1024,
    "pos_embed_seq_len": None,
}

WAN_TI2V_5B_VAE_CONFIG = {
    "attn_scales": [],
    "base_dim": 160,
    "decoder_base_dim": 256,
    "dim_mult": [1, 2, 4, 4],
    "dropout": 0.0,
    "in_channels": 12,
    "is_residual": True,
    "latents_mean": [
        -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
        -0.1382, 0.0542, 0.2813, 0.0891, 0.157, -0.0098, 0.0375, -0.1825,
        -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
        -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.123,
        -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.052, 0.3748,
        0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
    ],
    "latents_std": [
        0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.499, 0.4818, 0.5013,
        0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
        0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
        0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
        0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
        0.3971, 1.06, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
    ],
    "num_res_blocks": 2,
    "out_channels": 12,
    "patch_size": 2,
    "scale_factor_spatial": 16,
    "scale_factor_temporal": 4,
    "temperal_downsample": [False, True, True],
    "z_dim": 48,
}


def _video_status(message: str) -> None:
    print(f"[AIWF] Video: {message}", flush=True)
    try:
        from aiwf.dev.diagnostics import trace_safe

        trace_safe("wan.status", message, component="wan.pipeline")
    except Exception:
        logger.debug("Wan status trace failed.", exc_info=True)


def _cuda_total_vram_mb() -> int:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        return int(getattr(props, "total_memory", 0) // (1024 * 1024))
    except Exception:
        return 0


def _resident_min_vram_mb(default: int = 20 * 1024) -> int:
    raw = os.environ.get("AIWF_WAN_RESIDENT_MIN_VRAM_MB", "").strip()
    if not raw:
        return int(default)
    try:
        return max(0, int(raw))
    except ValueError:
        return int(default)


def _resolve_dual_stage_offload_for_hardware(offload: str, *, fast_fp8_pair: bool) -> str:
    if offload != "resident" or not fast_fp8_pair:
        return offload
    if _env_flag("AIWF_WAN_FORCE_RESIDENT", default=False):
        return offload

    total_mb = _cuda_total_vram_mb()
    min_mb = _resident_min_vram_mb()
    if total_mb and total_mb < min_mb:
        _video_status(
            "Resident mode was requested, but this GPU has "
            f"{total_mb} MB VRAM and the current safe threshold is {min_mb} MB. "
            "Falling back to Balanced so high/low FP8 stages swap instead of co-residing."
        )
        return "balanced"
    return offload


def _safetensors_uses_comfy_fp8_quant(path: Path) -> bool:
    if path.suffix.lower() != ".safetensors":
        return False
    try:
        from safetensors import safe_open
    except Exception:
        return False

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            key_l = key.lower()
            if key_l.endswith(_COMFY_FP8_METADATA_SUFFIXES):
                return True
            try:
                dtype = handle.get_slice(key).get_dtype()
            except Exception:
                continue
            if str(dtype).upper().startswith("F8"):
                return True
    return False


def _normalize_wan_transformer_key(key: str) -> str:
    """Normalize original/Comfy Wan keys before applying diffusers renames."""
    for prefix in ("model.diffusion_model.", "diffusion_model."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def _dequantize_comfy_fp8_state_dict(sd: dict, torch_dtype=None) -> dict:
    """Convert Comfy FP8 Wan tensors to regular torch tensors for diffusers.

    Comfy-style FP8 safetensors store quantized weights plus scalar
    ``*.weight_scale`` tensors. Diffusers does not consume those sidecar keys,
    so AIWF expands the weights before loading them into the Wan transformer.
    """
    try:
        import torch
    except Exception:  # pragma: no cover - torch is required to reach this path
        torch = None

    out: dict = {}
    for key, value in sd.items():
        key_l = key.lower()
        if key_l.endswith(_COMFY_FP8_METADATA_SUFFIXES):
            continue

        next_value = value
        dtype_name = str(getattr(value, "dtype", "")).lower()
        if "float8" in dtype_name:
            scale = None
            if key.endswith(".weight"):
                base_key = key.removesuffix(".weight")
                scale = sd.get(f"{base_key}.weight_scale")
                if scale is None:
                    scale = sd.get(f"{base_key}.scale_weight")
            next_value = value.float()
            if scale is not None:
                next_value = next_value * scale.float()
            if torch_dtype is not None:
                next_value = next_value.to(dtype=torch_dtype)
        elif torch_dtype is not None and torch is not None and getattr(value, "is_floating_point", lambda: False)():
            next_value = value.to(dtype=torch_dtype)

        out[key] = next_value
    return out


def _tensor_diag(tensor: Any) -> dict[str, Any]:
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


def _fp8_scaled_mm_failure_payload(
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
        "input": _tensor_diag(input_tensor),
        "matmul": {
            "lhs": _tensor_diag(x8),
            "rhs": _tensor_diag(weight_t),
            "scale_a": _tensor_diag(scale_a),
            "scale_b": _tensor_diag(scale_b),
            "rows": int(rows),
            "padded_rows": int(padded_rows),
            "pad_m": int(pad_m),
        },
    }


def _trace_fp8_scaled_mm_fallback(payload: dict[str, Any]) -> None:
    try:
        from aiwf.dev.diagnostics import trace_safe

        trace_safe(
            "wan.fp8_scaled_mm_fallback",
            "Wan FP8 _scaled_mm fallback",
            **payload,
        )
    except Exception:
        logger.debug("Wan FP8 _scaled_mm diagnostic trace failed.", exc_info=True)


def _fp8_generic_fallback_payload(
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
        "input": _tensor_diag(input_tensor),
        "matmul": {},
    }


def _new_fp8_scaled_linear(in_features: int, out_features: int, bias: bool):
    from aiwf.infrastructure.quant.fp8_linear import FP8ScaledLinear

    return FP8ScaledLinear(
        in_features,
        out_features,
        bias,
        strict_exception_cls=WanUnavailable,
        fallback_tracer=_trace_fp8_scaled_mm_fallback,
    )


def _module_parent_and_name(root, module_path: str):
    parent = root
    parts = module_path.split(".")
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def _replace_linear_with_fp8(root, module_path: str):
    import torch

    parent, name = _module_parent_and_name(root, module_path)
    current = parent[int(name)] if name.isdigit() else getattr(parent, name)
    if current.__class__.__name__ == "FP8ScaledLinear":
        return current
    if not isinstance(current, torch.nn.Linear):
        raise WanUnavailable(f"Expected Linear at '{module_path}', found {type(current).__name__}.")
    replacement = _new_fp8_scaled_linear(current.in_features, current.out_features, current.bias is not None)
    if name.isdigit():
        parent[int(name)] = replacement
    else:
        setattr(parent, name, replacement)
    return replacement


def _collect_fp8_linear_metrics(*roots: Any) -> dict[str, Any]:
    from aiwf.infrastructure.quant.fp8_linear import collect_fp8_linear_metrics

    return collect_fp8_linear_metrics(*roots)


def _collect_attention_metrics(*roots: Any) -> dict[str, Any]:
    backends: list[str] = []
    optimizations: list[str] = []
    seen: set[int] = set()
    for root in roots:
        if root is None:
            continue
        loaded = getattr(root, "_loaded_model", None)
        candidates = (root, loaded) if loaded is not None else (root,)
        for candidate in candidates:
            if candidate is None:
                continue
            ident = id(candidate)
            if ident in seen:
                continue
            seen.add(ident)
            backend = str(getattr(candidate, "_aiwf_attention_backend", "") or "").strip()
            if backend and backend not in backends:
                backends.append(backend)
            for item in tuple(getattr(candidate, "_aiwf_attention_optimizations", ()) or ()):
                text = str(item).strip()
                if text and text not in optimizations:
                    optimizations.append(text)
    return {
        "attention_backends": backends,
        "attention_optimizations": optimizations,
    }


def _module_has_accelerate_hook(module: Any) -> bool:
    """Return whether Accelerate device/offload hooks are already attached."""
    if module is None:
        return False
    if getattr(module, "_hf_hook", None) is not None:
        return True
    if getattr(module, "_diffusers_hook", None) is not None:
        return True
    if getattr(module, "_aiwf_group_offload", False):
        return True
    try:
        return any(
            getattr(child, "_hf_hook", None) is not None
            or getattr(child, "_diffusers_hook", None) is not None
            or getattr(child, "_aiwf_group_offload", False)
            for child in module.modules()
        )
    except Exception:
        return False


def _wan_group_offload_blocks() -> int:
    return _env_int("AIWF_WAN_GROUP_OFFLOAD_BLOCKS", 4)


def _wan_group_offload_stream_enabled() -> bool:
    return _env_flag("AIWF_WAN_GROUP_OFFLOAD_STREAM", default=False)


def _wan_group_offload_record_stream() -> bool:
    return _env_flag("AIWF_WAN_GROUP_OFFLOAD_RECORD_STREAM", default=True)


def _wan_group_offload_low_cpu_mem_usage() -> bool:
    return _env_flag("AIWF_WAN_GROUP_OFFLOAD_LOW_CPU_MEM", default=True)


def _install_group_offload_for_stage(
    model: Any,
    device: Any,
    *,
    blocks: int | None = None,
    use_stream: bool | None = None,
    record_stream: bool | None = None,
    low_cpu_mem_usage: bool | None = None,
) -> bool:
    """Install Diffusers block-level group offload on a Wan transformer stage."""
    if model is None:
        return False
    if getattr(model, "_aiwf_group_offload", False):
        return True
    enable = getattr(model, "enable_group_offload", None)
    if not callable(enable):
        return False
    try:
        import torch

        block_count = max(1, int(blocks or _wan_group_offload_blocks()))
        stream_enabled = _wan_group_offload_stream_enabled() if use_stream is None else bool(use_stream)
        record_stream_enabled = (
            _wan_group_offload_record_stream() if record_stream is None else bool(record_stream)
        )
        low_cpu_mem = (
            _wan_group_offload_low_cpu_mem_usage()
            if low_cpu_mem_usage is None
            else bool(low_cpu_mem_usage)
        )
        enable(
            onload_device=torch.device(device),
            offload_device=torch.device("cpu"),
            offload_type="block_level",
            num_blocks_per_group=block_count,
            use_stream=stream_enabled,
            record_stream=record_stream_enabled,
            low_cpu_mem_usage=low_cpu_mem,
            non_blocking=False,
        )
        model._aiwf_group_offload = True
        model._aiwf_group_offload_blocks = block_count
        model._aiwf_group_offload_stream = stream_enabled
        model._aiwf_group_offload_record_stream = record_stream_enabled
        model._aiwf_group_offload_low_cpu_mem_usage = low_cpu_mem
        return True
    except Exception as exc:
        logger.warning("Could not install Wan group offload: %s", exc)
        return False


def _install_sequential_cpu_offload_for_stage(model: Any, device: Any) -> bool:
    """Install Accelerate submodule CPU offload on a lazily loaded Wan stage.

    Diffusers can only attach sequential-offload hooks to components that exist
    when ``pipe.enable_sequential_cpu_offload()`` is called. AIWF's low-noise
    stage is intentionally lazy, so without this hook the first low step moves
    the entire 14B FP8 transformer to CUDA and consumes nearly all 16 GB VRAM.
    """
    if model is None:
        return False
    if _module_has_accelerate_hook(model):
        return True
    if not hasattr(model, "modules"):
        return False
    try:
        from accelerate import cpu_offload
    except Exception as exc:
        logger.warning("Accelerate cpu_offload unavailable for lazy Wan low stage: %s", exc)
        return False
    try:
        import torch

        target = torch.device(device)
        offload_buffers = bool(len(getattr(model, "_parameters", {}) or {}) > 0)
        cpu_offload(model, target, offload_buffers=offload_buffers)
        return True
    except Exception as exc:
        logger.warning("Could not install sequential CPU offload on lazy Wan low stage: %s", exc)
        return False


class _LazyWanTransformer:
    pass


def _new_lazy_wan_transformer(config, *, dtype, load_model, before_load=None):
    import contextlib
    import torch

    class LazyWanTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self._dtype = dtype
            self._load_model = load_model
            self._before_load = before_load
            self._loaded_model = None
            self._target_device = None
            # Accelerate's model-offload hook assumes every module in the
            # offload sequence has at least one parameter. The real low stage
            # is intentionally loaded later, so this sentinel keeps the hook
            # chain valid without materializing the 14B transformer early.
            self._aiwf_offload_sentinel = torch.nn.Parameter(
                torch.empty((), dtype=dtype), requires_grad=False
            )

        @property
        def dtype(self):
            if self._loaded_model is not None:
                return self._loaded_model.dtype
            return self._dtype

        def _ensure_loaded(self, device=None):
            target_device = device or self._target_device
            if self._loaded_model is None:
                if self._before_load is not None:
                    self._before_load()
                self._loaded_model = self._load_model()
                if target_device is not None and not _module_has_accelerate_hook(self._loaded_model):
                    self._loaded_model.to(target_device)
            elif target_device is not None:
                if not _module_has_accelerate_hook(self._loaded_model):
                    self._loaded_model.to(target_device)
            return self._loaded_model

        def to(self, *args, **kwargs):
            result = super().to(*args, **kwargs)
            self._target_device = self._aiwf_offload_sentinel.device
            if self._loaded_model is not None:
                self._loaded_model.to(*args, **kwargs)
            return result

        @contextlib.contextmanager
        def cache_context(self, name):
            model = self._ensure_loaded()
            with model.cache_context(name):
                yield

        def forward(self, *args, **kwargs):
            device = None
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and args:
                hidden_states = args[0]
            if hasattr(hidden_states, "device"):
                device = hidden_states.device
            model = self._ensure_loaded(device=device)
            return model(*args, **kwargs)

    return LazyWanTransformer()


def _is_gguf_transformer(path: str | None) -> bool:
    if not path:
        return False
    return Path(path).suffix.lower() == ".gguf"


def _load_transformer_state_dict(pth: str, *, label: str = "transformer") -> dict:
    """Load and rename standalone Wan transformer weights for diffusers load_state_dict."""
    from aiwf.infrastructure.wan.transformer_runtime import (
        WanTransformerFormat,
        require_diffusers_transformer_path,
    )

    pp = Path(pth)
    fmt = require_diffusers_transformer_path(pp, label=label)
    _video_status(f"Loading transformer weights ({fmt.value}): {pp.name}")
    if fmt == WanTransformerFormat.GGUF_QUANTIZED:
        try:
            raw = _load_gguf_state_dict(pp)
        except Exception as exc:
            raise WanUnavailable(f"GGUF dequant stub failed for {pp.name}. Install `gguf`.") from exc
    else:
        from safetensors.torch import load_file

        raw = load_file(str(pp))
        if _safetensors_uses_comfy_fp8_quant(pp):
            raise WanUnavailable(
                f"Internal loader error: {pp.name} is Comfy FP8 and should use the native FP8 path."
            )
    return _apply_wan_transformer_key_renames(raw)


def _load_gguf_transformer_weights(transformer, path: Path, *, torch_dtype) -> tuple[list[str], list[str]]:
    """Load Wan GGUF transformer weights (mmap + on-the-fly dequant, ComfyUI-GGUF style)."""
    from aiwf.infrastructure.wan.gguf_runtime import load_gguf_transformer_weights

    return load_gguf_transformer_weights(transformer, path, torch_dtype=torch_dtype)


def _load_comfy_fp8_transformer_weights(transformer, path: Path, *, torch_dtype) -> tuple[list[str], list[str]]:
    """Load Comfy scaled-FP8 transformer weights without expanding them to bf16."""
    import torch
    from safetensors import safe_open

    if not _torch_native_fp8_available():
        raise WanUnavailable(
            "This Wan transformer is ComfyUI scaled FP8, but native FP8 execution is unavailable. "
            "Use CUDA torch with float8 and `_scaled_mm`, or use a Diffusers bf16/fp16 transformer."
        )

    non_fp8_state: dict[str, Any] = {}
    loaded_fp8_state_keys: set[str] = set()
    fp8_loaded = 0
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        for raw_key in keys:
            raw_key_l = raw_key.lower()
            if raw_key_l.endswith(_COMFY_FP8_METADATA_SUFFIXES):
                continue
            tensor_slice = handle.get_slice(raw_key)
            dtype_name = str(tensor_slice.get_dtype()).upper()
            renamed_key = next(iter(_apply_wan_transformer_key_renames({raw_key: None}).keys()))

            if dtype_name.startswith("F8") and renamed_key.endswith(".weight"):
                module_path = renamed_key.removesuffix(".weight")
                scale_key = raw_key.removesuffix(".weight") + ".weight_scale"
                if scale_key not in keys:
                    scale_key = raw_key.removesuffix(".weight") + ".scale_weight"
                if scale_key not in keys:
                    raise WanUnavailable(f"FP8 tensor '{raw_key}' is missing a weight scale tensor.")

                module = _replace_linear_with_fp8(transformer, module_path)
                weight = handle.get_tensor(raw_key)
                scale = handle.get_tensor(scale_key).to(dtype=torch.float32)
                module.weight = torch.nn.Parameter(weight, requires_grad=False)
                if scale.numel() != 1:
                    raise WanUnavailable(
                        f"FP8 tensor '{raw_key}' has a per-channel weight scale "
                        f"({scale.numel()} values); this build only supports a single per-tensor "
                        "scale. Re-export with per-tensor FP8 scaling, or use the GGUF variant."
                    )
                module.weight_scale = scale.reshape(()).to(dtype=torch.float32)
                loaded_fp8_state_keys.add(renamed_key)
                loaded_fp8_state_keys.add(f"{module_path}.weight_scale")
                fp8_loaded += 1
                continue

            tensor = handle.get_tensor(raw_key)
            if hasattr(tensor, "is_floating_point") and tensor.is_floating_point():
                tensor = tensor.to(dtype=torch_dtype)
            non_fp8_state[renamed_key] = tensor

    missing, unexpected = transformer.load_state_dict(non_fp8_state, strict=False, assign=True)
    missing = [key for key in missing if key not in loaded_fp8_state_keys]
    logger.info("Loaded %d native FP8 scaled linear weights from %s.", fp8_loaded, path.name)
    return list(missing), list(unexpected)


def _apply_transformer_lora(transformer, lora_path: str | None, *, adapter_name: str, weight: float) -> None:
    if not lora_path:
        return
    try:
        try:
            transformer.load_lora_adapter(lora_path, adapter_name=adapter_name, prefix="transformer")
        except Exception:
            transformer.load_lora_adapter(lora_path, adapter_name=adapter_name, prefix=None)
        transformer.set_adapters(adapter_name, weights=float(weight))
        transformer.fuse_lora(adapter_names=[adapter_name], lora_scale=1.0, safe_fusing=True)
    except Exception as exc:
        raise WanUnavailable(
            f"Could not load Wan LoRA '{Path(lora_path).name}'. Make sure it is a Wan-compatible LoRA."
        ) from exc


def wan_supported() -> bool:
    try:
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline  # noqa: F401

        return True
    except Exception:
        return False


def _require_wan():
    try:
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline  # noqa: F401
    except Exception as exc:  # pragma: no cover - env check
        raise WanUnavailable(
            "Wan video needs a recent diffusers (>=0.35) with WanImageToVideoPipeline. "
            "Update diffusers and transformers, then restart."
        ) from exc

    from aiwf.infrastructure.torch.wan_perf import bootstrap_wan_cuda_settings, describe_missing_comfy_parity
    from aiwf.infrastructure.wan.transformer_runtime import describe_comfy_launcher_parity

    active = bootstrap_wan_cuda_settings()
    if active:
        logger.info("Wan CUDA bootstrap: %s", ", ".join(active))
    for hint in describe_missing_comfy_parity():
        logger.info("Wan perf hint: %s", hint)
    for hint in describe_comfy_launcher_parity():
        logger.info("Wan arch hint: %s", hint)


# Key renames to convert "original/ComfyUI" Wan transformer keys -> diffusers WanTransformer3DModel keys.
# (Subset focused on I2V/TI2V/T2V; harmless/no-op when weights are already in diffusers layout.)
TRANSFORMER_KEYS_RENAME_DICT = {
    "time_embedding.0": "condition_embedder.time_embedder.linear_1",
    "time_embedding.2": "condition_embedder.time_embedder.linear_2",
    "text_embedding.0": "condition_embedder.text_embedder.linear_1",
    "text_embedding.2": "condition_embedder.text_embedder.linear_2",
    "time_projection.1": "condition_embedder.time_proj",
    "head.modulation": "scale_shift_table",
    "head.head": "proj_out",
    "modulation": "scale_shift_table",
    "ffn.0": "ffn.net.0.proj",
    "ffn.2": "ffn.net.2",
    # norm swap (original uses norm1,norm3,norm2 ordering in places)
    "norm2": "norm__placeholder",
    "norm3": "norm2",
    "norm__placeholder": "norm3",
    # I2V image embed
    "img_emb.proj.0": "condition_embedder.image_embedder.norm1",
    "img_emb.proj.1": "condition_embedder.image_embedder.ff.net.0.proj",
    "img_emb.proj.3": "condition_embedder.image_embedder.ff.net.2",
    "img_emb.proj.4": "condition_embedder.image_embedder.norm2",
    # FLF2V
    "img_emb.emb_pos": "condition_embedder.image_embedder.pos_embed",
    # attention
    "self_attn.q": "attn1.to_q",
    "self_attn.k": "attn1.to_k",
    "self_attn.v": "attn1.to_v",
    "self_attn.o": "attn1.to_out.0",
    "self_attn.norm_q": "attn1.norm_q",
    "self_attn.norm_k": "attn1.norm_k",
    "cross_attn.q": "attn2.to_q",
    "cross_attn.k": "attn2.to_k",
    "cross_attn.v": "attn2.to_v",
    "cross_attn.o": "attn2.to_out.0",
    "cross_attn.norm_q": "attn2.norm_q",
    "cross_attn.norm_k": "attn2.norm_k",
    "attn2.to_k_img": "attn2.add_k_proj",
    "attn2.to_v_img": "attn2.add_v_proj",
    "attn2.norm_k_img": "attn2.norm_added_k",
}


def _apply_wan_transformer_key_renames(sd: dict) -> dict:
    """Return a new state_dict with keys renamed for diffusers WanTransformer3DModel."""
    renamed: dict = {}
    for key, value in sd.items():
        new_key = _normalize_wan_transformer_key(key)
        for old, new in TRANSFORMER_KEYS_RENAME_DICT.items():
            new_key = new_key.replace(old, new)
        renamed[new_key] = value
    return renamed


_WAN_VAE_FIXED_KEY_RENAMES = {
    "conv1.weight": "quant_conv.weight",
    "conv1.bias": "quant_conv.bias",
    "conv2.weight": "post_quant_conv.weight",
    "conv2.bias": "post_quant_conv.bias",
    "encoder.conv1.weight": "encoder.conv_in.weight",
    "encoder.conv1.bias": "encoder.conv_in.bias",
    "encoder.head.0.gamma": "encoder.norm_out.gamma",
    "encoder.head.2.weight": "encoder.conv_out.weight",
    "encoder.head.2.bias": "encoder.conv_out.bias",
    "decoder.conv1.weight": "decoder.conv_in.weight",
    "decoder.conv1.bias": "decoder.conv_in.bias",
    "decoder.head.0.gamma": "decoder.norm_out.gamma",
    "decoder.head.2.weight": "decoder.conv_out.weight",
    "decoder.head.2.bias": "decoder.conv_out.bias",
}


def _remap_wan_vae_residual_key(rest: str) -> str:
    rest = rest.replace("residual.0.gamma", "norm1.gamma")
    rest = rest.replace("residual.2.", "conv1.")
    rest = rest.replace("residual.3.gamma", "norm2.gamma")
    rest = rest.replace("residual.6.", "conv2.")
    rest = rest.replace("shortcut.", "conv_shortcut.")
    return rest


def _remap_wan_vae_key(key: str) -> str:
    """Map Comfy/original Wan VAE keys to diffusers AutoencoderKLWan keys."""
    if key in _WAN_VAE_FIXED_KEY_RENAMES:
        return _WAN_VAE_FIXED_KEY_RENAMES[key]

    match = re.match(r"^(encoder|decoder)\.middle\.(\d+)\.(.+)$", key)
    if match:
        side, block_index, rest = match.groups()
        if block_index == "1":
            return f"{side}.mid_block.attentions.0.{rest}"
        resnet_index = "0" if block_index == "0" else "1"
        return f"{side}.mid_block.resnets.{resnet_index}.{_remap_wan_vae_residual_key(rest)}"

    match = re.match(r"^encoder\.downsamples\.(\d+)\.downsamples\.(\d+)\.(.+)$", key)
    if match:
        block_index, sub_index, rest = match.groups()
        if sub_index in {"0", "1"}:
            return f"encoder.down_blocks.{block_index}.resnets.{sub_index}.{_remap_wan_vae_residual_key(rest)}"
        return f"encoder.down_blocks.{block_index}.downsampler.{rest}"

    match = re.match(r"^decoder\.upsamples\.(\d+)\.upsamples\.(\d+)\.(.+)$", key)
    if match:
        block_index, sub_index, rest = match.groups()
        if sub_index in {"0", "1", "2"}:
            return f"decoder.up_blocks.{block_index}.resnets.{sub_index}.{_remap_wan_vae_residual_key(rest)}"
        return f"decoder.up_blocks.{block_index}.upsampler.{rest}"

    return key


def _remap_wan_vae_state_dict(sd: dict) -> dict:
    return {_remap_wan_vae_key(key): value for key, value in sd.items()}


def estimate_gguf_expanded_gb(path: Path) -> float:
    """Rough host RAM needed to fully dequantize a GGUF for diffusers load."""
    from aiwf.infrastructure.wan.transformer_runtime import estimate_gguf_expanded_gb as _estimate

    return _estimate(path)


def _load_gguf_state_dict(path: Path, *, torch_dtype=None) -> dict:
    """DEV STUB: fully dequantize GGUF into diffusers weights. Not for production."""
    import gc

    import gguf
    import numpy as np
    import torch

    if torch_dtype is None:
        torch_dtype = torch.bfloat16

    expanded_gb = estimate_gguf_expanded_gb(path)
    if expanded_gb > 12.0 and os.environ.get("AIWF_WAN_ALLOW_EXPENSIVE_DEQUANT") != "1":
        raise WanUnavailable(
            f"GGUF {path.name} would expand to ~{expanded_gb:.0f} GB in RAM when dequantized for diffusers. "
            "AIWF does not run quantized GGUF inference yet (unlike ComfyUI-GGUF). "
            "Use ComfyUI scaled-FP8 `.safetensors` high/low pair instead, or set "
            "AIWF_WAN_ALLOW_EXPENSIVE_DEQUANT=1 if you accept the long load + high RAM use."
        )

    reader = gguf.GGUFReader(str(path))
    total = len(reader.tensors)
    _video_status(
        f"Dequantizing GGUF {path.name} ({total} tensors, est. ~{expanded_gb:.0f} GB RAM) "
        "-- this is slow; FP8 safetensors are much faster."
    )
    sd: dict[str, torch.Tensor] = {}
    for index, tensor in enumerate(reader.tensors, start=1):
        name = tensor.name
        arr = gguf.dequantize(tensor.data, tensor.tensor_type)
        t = torch.from_numpy(np.asarray(arr))
        if hasattr(t, "is_floating_point") and t.is_floating_point():
            t = t.to(dtype=torch_dtype)
        sd[name] = t
        if index % 100 == 0 or index == total:
            _video_status(f"GGUF dequant progress: {index}/{total}")
            gc.collect()
    return sd


def _has_wan_text_encoder(path: Path) -> bool:
    text_encoder = path / "text_encoder"
    return (
        (text_encoder / "config.json").is_file()
        and (
            (text_encoder / "model.safetensors").is_file()
            or (text_encoder / "model.safetensors.index.json").is_file()
        )
    )


def _has_wan_tokenizer(path: Path) -> bool:
    tokenizer = path / "tokenizer"
    return (tokenizer / "tokenizer.json").is_file()


def _has_wan_scheduler(path: Path) -> bool:
    return (path / "scheduler" / "scheduler_config.json").is_file()


def _is_wan_components_base(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "model_index.json").is_file()
        and _has_wan_text_encoder(path)
        and _has_wan_tokenizer(path)
        and _has_wan_scheduler(path)
    )


def _default_wan_search_roots() -> list[Path]:
    """Return reliable candidate roots for the Wan components base, independent of os.getcwd().

    Uses (in priority):
    - CWD-based (for normal launches from project root)
    - Source-relative: resolve from this file's location up to project root (aiwf/infrastructure/wan -> project)
    - Source-relative paths from the installed project tree
    """
    roots: list[Path] = []
    # 1. CWD relative (typical when launched via launch.py / webui.bat from project root)
    roots.append(Path("models/wan/Diffusers"))
    roots.append(Path("models/wan"))

    # 2. Module-relative (robust if python started from any cwd, as long as source tree is used)
    try:
        here = Path(__file__).resolve()
        # pipeline.py is at <project>/aiwf/infrastructure/wan/pipeline.py
        # parents[3] -> project root (same convention as RuntimeFlags default)
        project_root = here.parents[3]
        roots.append(project_root / "models" / "wan" / "Diffusers")
        roots.append(project_root / "models" / "wan")
    except Exception:
        pass

    # Dedup while preserving order
    seen: set[Path] = set()
    deduped: list[Path] = []
    for r in roots:
        rp = r.resolve() if not r.is_absolute() else r
        if rp not in seen:
            seen.add(rp)
            deduped.append(r)
    return deduped


def _find_wan_components_base(search_roots: list[Path] | None = None) -> str | None:
    """Find a full diffusers Wan layout we can use for text_encoder / tokenizer / scheduler when loading single-file weights.

    Upper layers (WanService) should pre-compute and pass an explicit components_base.
    This finder is a best-effort fallback and must be cwd-independent.
    """
    roots = search_roots or _default_wan_search_roots()

    for root in roots:
        preferred = root / "Wan2.2-TI2V-5B-Diffusers"
        if _is_wan_components_base(preferred):
            return str(preferred.resolve())

    # Also try the name directly under each wan root (older layout)
    for root in roots:
        direct = root / "Wan2.2-TI2V-5B-Diffusers"
        rp = direct.resolve() if not direct.is_absolute() else direct
        if _is_wan_components_base(rp):
            return str(rp)

    # Deep scan under wan-containing roots
    for wan_root in roots:
        try:
            base = wan_root if wan_root.is_absolute() else wan_root.resolve()
        except Exception:
            base = wan_root
        if base.exists():
            candidates = [base, *[child for child in base.rglob("*") if child.is_dir()]]
            for child in sorted(candidates):
                if _is_wan_components_base(child):
                    return str(child.resolve())

    return None


def _load_wan_vae(vae_or_base: str, torch_dtype) -> "AutoencoderKLWan":
    """Load Wan VAE from either:
    - a direct path to a .safetensors (Comfy style, recommended for Wan2.1 VAE)
    - or a diffusers folder (vae/ subdir or the root that has vae/)
    - or a base components dir that contains a vae/ subfolder.

    Strips unknown config keys (e.g. 'clip_output': False from some converted 2.2 VAEs)
    that the current diffusers AutoencoderKLWan does not declare.
    """
    from diffusers import AutoencoderKLWan
    import json
    from pathlib import Path as _P

    p = _P(vae_or_base)

    # Direct single-file VAE (most common for user's Comfy Wan 2.1 VAE)
    if p.is_file() and p.suffix.lower() in {".safetensors", ".pth", ".pt"}:
        try:
            _video_status(f"Loading local Wan VAE file: {p.name}")
            vae = AutoencoderKLWan.from_single_file(str(p), torch_dtype=torch_dtype)
            return vae
        except Exception as exc:
            try:
                from safetensors.torch import load_file as _load_st

                sd = _load_st(str(p), device="cpu")
                z_dim = int(getattr(sd.get("conv2.weight"), "shape", [0])[0] or 0)
                if z_dim == 48:
                    _video_status(f"Loading Wan 2.2 TI2V VAE file with local 48-channel config: {p.name}")
                    vae = AutoencoderKLWan.from_config(dict(WAN_TI2V_5B_VAE_CONFIG), torch_dtype=torch_dtype)
                    sd = _remap_wan_vae_state_dict(sd)
                    missing, unexpected = vae.load_state_dict(sd, strict=False, assign=True)
                    if missing:
                        logger.warning("Wan 2.2 TI2V VAE loaded with missing keys: %s", missing[:20])
                    if unexpected:
                        logger.debug("Wan 2.2 TI2V VAE ignored unexpected keys: %s", unexpected[:20])
                    vae.eval()
                    return vae
            except Exception:
                logger.debug("Could not load %s with Wan 2.2 TI2V VAE fallback.", p, exc_info=True)
            raise WanUnavailable(
                f"Selected VAE '{p.name}' could not be loaded as a Wan VAE. "
                "Choose a Wan VAE file such as 'wan2.1_vae.safetensors' instead of a generic SD VAE."
            ) from exc

    # If it's a dir that looks like a vae folder itself (has config.json + weights)
    if p.is_dir():
        cfg_path = p / "config.json"
        weights = p / "diffusion_pytorch_model.safetensors"
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                for bad in ("clip_output", "_diffusers_version"):
                    cfg.pop(bad, None)
                vae = AutoencoderKLWan.from_config(cfg, torch_dtype=torch_dtype)
                if weights.exists():
                    from safetensors.torch import load_file as _load_st
                    sd = _load_st(str(weights))
                    vae.load_state_dict(sd, strict=False)
                    return vae
            except Exception:
                pass
        # or it is a base dir containing vae/ subfolder
        vae_dir = p / "vae"
        cfg_path = vae_dir / "config.json"
        weights = vae_dir / "diffusion_pytorch_model.safetensors"
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                for bad in ("clip_output", "_diffusers_version"):
                    cfg.pop(bad, None)
                vae = AutoencoderKLWan.from_config(cfg, torch_dtype=torch_dtype)
                if weights.exists():
                    from safetensors.torch import load_file as _load_st
                    sd = _load_st(str(weights))
                    vae.load_state_dict(sd, strict=False)
                    return vae
            except Exception:
                pass

    # Last resort: try from_pretrained on the path (for HF ids or proper diffusers vae)
    try:
        if p.is_dir():
            _video_status(f"Loading local Wan VAE folder: {p}")
        return AutoencoderKLWan.from_pretrained(
            str(p),
            subfolder="vae" if p.is_dir() and (p / "vae").exists() else None,
            torch_dtype=torch_dtype,
            local_files_only=p.is_dir(),
        )
    except Exception:
        # final fallback without subfolder
        return AutoencoderKLWan.from_pretrained(
            str(p),
            torch_dtype=torch_dtype,
            local_files_only=p.is_dir(),
        )


def _load_umt5_text_encoder(text_encoder_dir: Path, torch_dtype):
    """Load Wan's UMT5 text encoder from either HF shards or AIWF's local single-file layout."""
    import json
    import torch
    from transformers import UMT5Config, UMT5EncoderModel

    text_encoder_dir = Path(text_encoder_dir)
    config_path = text_encoder_dir / "config.json"
    single_file = text_encoder_dir / "model.safetensors"

    if config_path.is_file() and single_file.is_file():
        from accelerate import init_empty_weights
        from safetensors.torch import load_file

        _video_status(f"Loading local UMT5 text encoder file: {single_file.name}")
        config = UMT5Config.from_dict(json.loads(config_path.read_text(encoding="utf-8")))
        with init_empty_weights():
            model = UMT5EncoderModel(config)

        expected_keys = set(model.state_dict().keys())
        raw = load_file(str(single_file), device="cpu")
        state_dict = {}
        skipped_keys: list[str] = []
        for key, value in raw.items():
            if key in {"spiece_model", "scaled_fp8"}:
                skipped_keys.append(key)
                continue
            if key not in expected_keys:
                skipped_keys.append(key)
                continue
            if hasattr(value, "is_floating_point") and value.is_floating_point():
                value = value.to(dtype=torch_dtype)
            state_dict[key] = value

        if (
            "shared.weight" in state_dict
            and "encoder.embed_tokens.weight" in expected_keys
            and "encoder.embed_tokens.weight" not in state_dict
        ):
            state_dict["encoder.embed_tokens.weight"] = state_dict["shared.weight"]

        missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
        if missing:
            logger.warning("Wan UMT5 text encoder loaded with missing keys: %s", missing[:20])
        if unexpected:
            logger.debug("Wan UMT5 text encoder ignored unexpected keys: %s", unexpected[:20])
        if skipped_keys:
            logger.debug("Wan UMT5 text encoder skipped non-model keys: %s", skipped_keys[:20])
        model.eval()
        return model

    return UMT5EncoderModel.from_pretrained(
        str(text_encoder_dir), torch_dtype=torch_dtype, local_files_only=text_encoder_dir.is_dir()
    )


def _remap_ggml_to_hf_umt5(key: str) -> str:
    """Translate a GGML-style GGUF tensor name to the HuggingFace diffusers key.

    GGUF files for T5/UMT5 use GGML (llama.cpp) naming conventions, not HF names.
    This remap follows the same table used by ComfyUI-GGUF for T5/UMT5 encoders.

    Examples:
      token_embd.weight          → shared.weight
      enc.blk.0.attn_q.weight   → encoder.block.0.layer.0.SelfAttention.q.weight
      enc.blk.0.ffn_gate.weight → encoder.block.0.layer.1.DenseReluDense.wi_0.weight
      output_norm.weight         → final_layer_norm.weight
    """
    # Order matters — longest/most-specific patterns first
    _REMAP = [
        ("enc.blk.", "encoder.block."),
        # UMT5 encoder final norm. The GGUF stores this as `enc.output_norm.weight`,
        # but HF UMT5EncoderModel nests the stack under `.encoder`, so the real key is
        # `encoder.final_layer_norm.weight`. The generic `output_norm`->`final_layer_norm`
        # rule below loses the `encoder.` prefix and the weight silently stays on `meta`
        # (then zero-filled), which zeroes the text-encoder output norm and corrupts ALL
        # prompt conditioning (black/garbled video). Map the prefixed form explicitly.
        ("enc.output_norm", "encoder.final_layer_norm"),
        ("token_embd", "shared"),
        ("output_norm", "final_layer_norm"),
        ("attn_q", "layer.0.SelfAttention.q"),
        ("attn_k", "layer.0.SelfAttention.k"),
        ("attn_v", "layer.0.SelfAttention.v"),
        ("attn_o", "layer.0.SelfAttention.o"),
        ("attn_rel_b", "layer.0.SelfAttention.relative_attention_bias"),
        ("attn_norm", "layer.0.layer_norm"),
        ("ffn_gate", "layer.1.DenseReluDense.wi_0"),
        ("ffn_up", "layer.1.DenseReluDense.wi_1"),
        ("ffn_down", "layer.1.DenseReluDense.wo"),
        ("ffn_norm", "layer.1.layer_norm"),
    ]
    for src, dst in _REMAP:
        key = key.replace(src, dst)
    return key


def _orient_umt5_gguf_tensor(key: str, tensor, expected_shape: tuple[int, ...]):
    """Return a GGUF tensor in the orientation expected by HF UMT5.

    GGUF metadata and the dequantized NumPy array can report embedding
    dimensions in opposite orders. Decide from the actual tensor shape that will
    be handed to load_state_dict, not from reader metadata.
    """
    if (
        key in ("shared.weight", "encoder.embed_tokens.weight")
        and len(expected_shape) == 2
        and getattr(tensor, "ndim", 0) == 2
        and tuple(tensor.shape) != expected_shape
        and tuple(tensor.t().shape) == expected_shape
    ):
        return tensor.t().contiguous()
    return tensor


def _materialize_meta_tensors(model, dtype) -> int:
    """Materialize leftover meta tensors after a strict=False load.

    After load_state_dict(strict=False, assign=True), anything NOT present in the
    state dict stays a meta tensor (no data). Two very different cases:

    * **Buffers** (position ids, derived masks, etc.) are recomputable/benign, so we
      zero-fill them — this is what keeps enable_model_cpu_offload() from crashing.
    * **Parameters** (weights/biases) must come from the file. Zero-filling a real
      weight silently corrupts output — e.g. a single zeroed ``encoder.final_layer_norm``
      destroys all prompt conditioning and yields black/garbled video. So if any
      parameter is still meta we FAIL LOUD with the exact key names instead of
      producing a broken-but-running text encoder.
    """
    import torch
    buf_count = 0
    for name, buf in list(model.named_buffers()):
        if buf.is_meta:
            real = torch.zeros(buf.shape, dtype=dtype, device="cpu")
            parts = name.split(".")
            mod = model
            for part in parts[:-1]:
                mod = getattr(mod, part)
            setattr(mod, parts[-1], real)
            buf_count += 1

    meta_params = [name for name, param in model.named_parameters() if param.is_meta]
    if meta_params:
        logger.error(
            "UMT5 text encoder: %d weight(s) did not load from the file and would be "
            "zeroed (corrupting all conditioning): %s",
            len(meta_params),
            meta_params[:20],
        )
        raise WanUnavailable(
            "Text encoder file is missing required weights "
            f"({len(meta_params)} tensor(s), e.g. {meta_params[:5]}). "
            "The file is likely an incompatible or truncated UMT5/T5 export — pick a "
            "different text encoder (a Wan UMT5-XXL .safetensors or .gguf)."
        )
    if buf_count:
        logger.debug("UMT5 text encoder: %d benign buffer(s) zero-filled.", buf_count)
    return buf_count


def _load_standalone_umt5_text_encoder(path: str, torch_dtype):
    """Load a standalone UMT5-XXL text encoder from a single .safetensors or .gguf file.

    Handles all key-naming conventions automatically:
    - GGUF with GGML names (enc.blk.N.attn_q.weight)  → remapped to HF names
    - GGUF with HF names already present               → loaded directly
    - FP8 .safetensors with weight_scale sidecars      → dequantized to torch_dtype
    - Standard bfloat16 .safetensors                   → loaded directly

    NOT for t5xxl files — those are T5-XXL (Flux/SD3) and NOT compatible with Wan.
    """
    import json
    import torch
    from pathlib import Path as _P
    from transformers import UMT5Config, UMT5EncoderModel

    pp = _P(path)
    if not pp.is_file():
        raise WanUnavailable(f"Text encoder file not found: {path}")

    suffix = pp.suffix.lower()
    _video_status(f"Loading standalone UMT5-XXL text encoder: {pp.name}")

    if suffix == ".gguf":
        # GGUF UMT5 — dequantize via gguf package.
        # NOTE: GGUF files use GGML naming (enc.blk.N.attn_q.weight), NOT diffusers
        # naming. We apply _remap_ggml_to_hf_umt5() to every tensor key before
        # matching against the model's expected_keys.
        try:
            import gguf
            import numpy as np
        except ImportError as exc:
            raise WanUnavailable(
                f"GGUF text encoder {pp.name} requires the `gguf` package. "
                "Install it: pip install gguf"
            ) from exc

        reader = gguf.GGUFReader(str(pp))
        _video_status(f"Dequantizing GGUF text encoder ({len(reader.tensors)} tensors)...")

        # Infer vocab_size from the embedding tensor.
        # GGUF calls it "token_embd.weight"; after remap it becomes "shared.weight".
        # Some GGUF files store the embedding TRANSPOSED as [d_model, vocab_size]
        # instead of [vocab_size, d_model]. Use max(shape) to pick the vocab dim
        # reliably — vocab_size (250k–256k) is always much larger than d_model (4096).
        _vocab_size = 250112
        for _t in reader.tensors:
            if _t.name in ("token_embd.weight", "shared.weight", "encoder.embed_tokens.weight"):
                _shape = list(_t.shape)
                _vocab_size = max(_shape)
                logger.debug(
                    "GGUF UMT5 inferred vocab_size=%d from %s shape=%s",
                    _vocab_size, _t.name, _shape,
                )
                break
        _UMT5_XXL_CONFIG = {
            "architectures": ["UMT5EncoderModel"],
            "d_ff": 10240,
            "d_kv": 64,
            "d_model": 4096,
            "dense_act_fn": "gelu_new",
            "feed_forward_proj": "gated-gelu",
            "is_gated_act": True,
            "model_type": "umt5",
            "num_heads": 64,
            "num_layers": 24,
            "vocab_size": _vocab_size,
        }
        config = UMT5Config(**_UMT5_XXL_CONFIG)

        from accelerate import init_empty_weights
        with init_empty_weights():
            model = UMT5EncoderModel(config)

        expected_state = model.state_dict()
        expected_keys = set(expected_state.keys())
        expected_shapes = {key: tuple(value.shape) for key, value in expected_state.items()}
        sd: dict[str, torch.Tensor] = {}
        unmatched: list[str] = []
        for tensor in reader.tensors:
            arr = gguf.dequantize(tensor.data, tensor.tensor_type)
            t = torch.from_numpy(np.array(arr, dtype=np.float32)).contiguous()
            if t.is_floating_point():
                t = t.to(dtype=torch_dtype)
            # Try the remapped key first (GGML→HF), then the raw name as fallback
            hf_key = _remap_ggml_to_hf_umt5(tensor.name)
            if hf_key not in expected_keys and tensor.name in expected_keys:
                hf_key = tensor.name
            if hf_key in expected_keys:
                t = _orient_umt5_gguf_tensor(hf_key, t, expected_shapes[hf_key])
                sd[hf_key] = t
            else:
                unmatched.append(tensor.name)

        if unmatched:
            logger.debug("GGUF UMT5 unmatched tensor names (first 10): %s", unmatched[:10])

        # shared.weight → encoder.embed_tokens.weight alias
        if "shared.weight" in sd and "encoder.embed_tokens.weight" in expected_keys and "encoder.embed_tokens.weight" not in sd:
            sd["encoder.embed_tokens.weight"] = sd["shared.weight"]

        missing, _ = model.load_state_dict(sd, strict=False, assign=True)
        if missing:
            logger.debug("GGUF UMT5 text encoder keys not in file: %d", len(missing))
        # Materialize any remaining meta tensors — prevents crash in enable_model_cpu_offload
        _materialize_meta_tensors(model, torch_dtype)
        model.eval()
        _video_status(f"Standalone GGUF UMT5 text encoder loaded: {pp.name}")
        return model

    # .safetensors path — handles both standard and FP8-scaled variants.
    # FP8 UMT5 text encoders (like nsfw_wan_umt5-xxl_fp8_scaled.safetensors) store
    # the encoder in fp8 with weight_scale sidecar tensors. We dequantize on load
    # exactly like the transformer FP8 path: expand to torch_dtype (bfloat16).
    from accelerate import init_empty_weights
    from safetensors import safe_open

    # Try to find a UMT5 config alongside the file, otherwise use hardcoded defaults
    config_path = pp.parent / "config.json"
    if config_path.is_file():
        try:
            config = UMT5Config.from_dict(json.loads(config_path.read_text(encoding="utf-8")))
        except Exception:
            config = None
    else:
        config = None

    if config is None:
        # Infer vocab_size from the embedding tensor's actual shape.
        # Different UMT5 variants use different vocabularies (e.g. 250112 vs 256384);
        # hardcoding 250112 causes a load_state_dict size mismatch at runtime.
        _vocab_size = 250112
        with safe_open(str(pp), framework="pt", device="cpu") as _h:
            for _candidate in ("shared.weight", "encoder.embed_tokens.weight"):
                if _candidate in _h.keys():
                    _vocab_size = int(_h.get_tensor(_candidate).shape[0])
                    logger.debug("Safetensors UMT5 inferred vocab_size=%d from %s", _vocab_size, _candidate)
                    break
        _UMT5_XXL_CONFIG = {
            "architectures": ["UMT5EncoderModel"],
            "d_ff": 10240,
            "d_kv": 64,
            "d_model": 4096,
            "dense_act_fn": "gelu_new",
            "feed_forward_proj": "gated-gelu",
            "is_gated_act": True,
            "model_type": "umt5",
            "num_heads": 64,
            "num_layers": 24,
            "vocab_size": _vocab_size,
        }
        config = UMT5Config(**_UMT5_XXL_CONFIG)

    with init_empty_weights():
        model = UMT5EncoderModel(config)

    expected_keys = set(model.state_dict().keys())
    sd: dict = {}
    skipped: list[str] = []
    fp8_scale_map: dict[str, torch.Tensor] = {}

    # First pass: collect scale tensors so we can dequantize fp8 weights
    with safe_open(str(pp), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            key_l = key.lower()
            if key_l.endswith((".weight_scale", ".scale_weight", ".pre_quant_scale")):
                fp8_scale_map[key] = handle.get_tensor(key).to(torch.float32)

    with safe_open(str(pp), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            key_l = key.lower()
            if key_l.endswith((".weight_scale", ".scale_weight", ".pre_quant_scale", "spiece_model", "scaled_fp8")):
                skipped.append(key)
                continue
            if key not in expected_keys and key not in ("shared.weight",):
                skipped.append(key)
                continue
            tensor = handle.get_tensor(key)
            dtype_name = str(tensor.dtype).upper()
            if "FLOAT8" in dtype_name:
                # Look for matching scale
                base = key.removesuffix(".weight")
                scale = fp8_scale_map.get(f"{base}.weight_scale") or fp8_scale_map.get(f"{base}.scale_weight")
                tensor = tensor.float()
                if scale is not None:
                    tensor = (tensor * scale.float())
                tensor = tensor.to(dtype=torch_dtype)
            elif tensor.is_floating_point():
                tensor = tensor.to(dtype=torch_dtype)
            sd[key] = tensor

    # shared.weight alias for embed_tokens
    if "shared.weight" in sd and "encoder.embed_tokens.weight" in expected_keys and "encoder.embed_tokens.weight" not in sd:
        sd["encoder.embed_tokens.weight"] = sd["shared.weight"]

    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    if missing:
        logger.warning("Standalone UMT5 text encoder missing keys: %s", missing[:10])
    if skipped:
        logger.debug("Standalone UMT5 text encoder skipped non-model keys: %d", len(skipped))
    # Materialize any remaining meta tensors — prevents crash in enable_model_cpu_offload
    _materialize_meta_tensors(model, torch_dtype)
    model.eval()
    _video_status(f"Standalone UMT5 text encoder loaded: {pp.name}")
    return model


def _materialize_wan_rope_buffers(transformer) -> int:
    """Recompute Wan rotary-embedding buffers on a real (non-meta) device.

    ``WanRotaryPosEmbed`` registers ``freqs_cos`` / ``freqs_sin`` as *non-persistent*
    buffers computed in ``__init__`` purely from config. They therefore:
      * appear in NO checkpoint (so ``load_state_dict`` never fills them), and
      * are created on the ``meta`` device when the module is built under
        ``init_empty_weights()``.

    Left on meta, the first ``.to()`` / ``enable_*_cpu_offload`` / pinned-cache move
    crashes with ``NotImplementedError: Cannot copy out of meta tensor; no data!``.
    Because the values depend only on config (not trained weights), we rebuild a
    fresh rope module on CPU and copy its real buffers in. Returns how many modules
    were fixed.
    """
    try:
        from diffusers.models.transformers.transformer_wan import WanRotaryPosEmbed
    except Exception:
        return 0

    fixed = 0
    for module in transformer.modules():
        if not isinstance(module, WanRotaryPosEmbed):
            continue
        needs_fix = any(
            getattr(module, name, None) is not None and module._buffers.get(name) is not None
            and module._buffers[name].device.type == "meta"
            for name in ("freqs_cos", "freqs_sin")
        )
        if not needs_fix:
            continue
        # Rebuilt outside any init_empty_weights() context -> real CPU tensors.
        fresh = WanRotaryPosEmbed(
            attention_head_dim=module.attention_head_dim,
            patch_size=module.patch_size,
            max_seq_len=module.max_seq_len,
        )
        module.register_buffer("freqs_cos", fresh.freqs_cos.detach().clone(), persistent=False)
        module.register_buffer("freqs_sin", fresh.freqs_sin.detach().clone(), persistent=False)
        fixed += 1
    if fixed:
        logger.debug("Materialized rope buffers on %d WanRotaryPosEmbed module(s).", fixed)
    return fixed


def _empty_wan_transformer(config):
    """Create a Wan transformer shell without random parameter allocation.

    The shell is built under ``init_empty_weights`` (all weights on meta, filled
    later from the checkpoint), but the rotary-embedding buffers are immediately
    materialized on CPU because they are config-derived and never appear in any
    checkpoint — otherwise they would remain meta and crash every later device move.
    """
    from accelerate import init_empty_weights
    from diffusers import WanTransformer3DModel

    with init_empty_weights():
        model = WanTransformer3DModel.from_config(config)
    _materialize_wan_rope_buffers(model)
    return model


def _boundary_ratio_for_step_split(scheduler, *, total_steps: int, high_steps: int) -> float:
    """Map a Comfy-style step split to Diffusers' boundary-timestep ratio."""
    import torch

    total = max(1, int(total_steps))
    high = max(1, min(int(high_steps), total - 1))
    scheduler.set_timesteps(total, device=torch.device("cpu"))
    timesteps = scheduler.timesteps.detach().cpu().float()
    if high >= len(timesteps):
        return 1.0
    previous_t = float(timesteps[high - 1])
    next_t = float(timesteps[high])
    boundary_timestep = (previous_t + next_t) / 2.0
    train_steps = float(getattr(scheduler.config, "num_train_timesteps", 1000) or 1000)
    return min(1.0, max(0.0, boundary_timestep / train_steps))


def _free_cuda_memory() -> None:
    """Best-effort VRAM cleanup before loading heavy Wan transformers."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


def _recover_cuda_after_pin_memory_failure() -> None:
    """Clear pending CUDA allocator/error state after page-locked host allocation fails."""
    import gc

    for _ in range(2):
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        except Exception:
            pass


def _wan_cache_mode(offload: str, *, fast_fp8_pair: bool, pinned_memory: bool = True) -> str:
    """Return how the dual-transformer CPU/GPU cache should behave.

    - ``none``: accelerate owns placement (sequential / generic model offload).
    - ``gpu_swap``: keep high/low on CPU between stages; stream active stage to GPU
      (model offload + native FP8, 12–16 GB cards).
    - ``full``: legacy pinned-CPU cache (only when the full pipeline stays on GPU).
    """
    from aiwf.infrastructure.wan.native.memory import select_initial_stage_cache_mode

    return select_initial_stage_cache_mode(
        offload,
        fast_quantized_pair=fast_fp8_pair,
        pinned_memory=pinned_memory,
    ).value


def _stage_cache_uses_cpu_standby(mode: str) -> bool:
    from aiwf.infrastructure.wan.native.memory import stage_cache_uses_cpu_standby

    return stage_cache_uses_cpu_standby(mode)


def _stage_cache_pins_tensors(mode: str) -> bool:
    from aiwf.infrastructure.wan.native.memory import stage_cache_pins_tensors

    return stage_cache_pins_tensors(mode)


def _stage_cache_is_gpu_active_cpu_standby(mode: str) -> bool:
    from aiwf.infrastructure.wan.native.memory import stage_cache_is_gpu_active_cpu_standby

    return stage_cache_is_gpu_active_cpu_standby(mode)


def _stage_cache_is_disk_sequential(mode: str) -> bool:
    from aiwf.infrastructure.wan.native.memory import stage_cache_is_disk_sequential

    return stage_cache_is_disk_sequential(mode)


def _stage_cache_is_dual_gpu_resident(mode: str) -> bool:
    from aiwf.infrastructure.wan.native.memory import WanStageCacheMode

    return str(mode) == WanStageCacheMode.DUAL_GPU_RESIDENT.value


def _resolve_stage_cache_after_pin_probe(
    mode: str,
    *,
    high_path: str | Path | None,
    low_path: str | Path | None,
    pin_available: bool,
):
    from aiwf.infrastructure.wan.native.memory import resolve_stage_cache_after_pin_probe

    return resolve_stage_cache_after_pin_probe(
        mode,
        high_path=high_path,
        low_path=low_path,
        pin_available=pin_available,
    )


def _ensure_wan_attention_processors(transformer, name: str = "transformer") -> None:
    """Guarantee WanAttention modules use WanAttnProcessor (not AttnProcessor2_0).

    LoRA fuse / accelerate offload hooks can leave a generic processor attached;
    that crashes with ``AttributeError: 'WanAttention' object has no attribute 'spatial_norm'``.
    """
    if transformer is None:
        return
    try:
        from diffusers.models.transformers.transformer_wan import WanAttention, WanAttnProcessor
    except Exception:
        return

    fixed = 0
    for module in transformer.modules():
        if not isinstance(module, WanAttention):
            continue
        proc = getattr(module, "processor", None)
        if proc is None or proc.__class__.__name__ not in ("WanAttnProcessor", "WanAttnProcessor2_0"):
            module.set_processor(WanAttnProcessor())
            fixed += 1
        if not hasattr(module, "spatial_norm"):
            module.spatial_norm = None
    if fixed:
        logger.debug("Reset %d WanAttention processor(s) on %s", fixed, name)


def _apply_wan_attention_optimizations(
    transformer,
    name: str = "transformer",
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    temporal_chunks: bool | None = None,
) -> None:
    """Apply Wan-specific attention/conv optimizations (SDP flash, sageattention, channels_last)."""
    from aiwf.infrastructure.torch.wan_perf import apply_wan_transformer_optimizations
    from aiwf.infrastructure.wan.sliced_sampler import install_temporal_chunk_forward

    active = apply_wan_transformer_optimizations(transformer, name=name)
    if active:
        _video_status(f"{name} optimizations: {', '.join(active)}")
    if install_temporal_chunk_forward(
        transformer,
        name=name,
        chunk_size=chunk_size,
        overlap=chunk_overlap,
        enabled=temporal_chunks,
    ):
        _video_status(
            f"{name}: temporal chunk denoise active "
            f"(chunk={getattr(transformer, '_aiwf_chunk_size', '?')}, "
            f"overlap={getattr(transformer, '_aiwf_chunk_overlap', '?')})."
        )

# Dual-Buffer Virtual VRAM Cache per user's detailed guidance.
# Keeps heavy models (14B Wan high/low, text encoder, vae) pinned in CPU RAM.
# Fast PCIe streaming for swaps instead of disk reloads on del/reload.
# Used to coordinate high <-> low without co-existing in VRAM, and keep objects alive.
class AIWFModelCacheManager:
    def __init__(self, device="cuda"):
        self.device = device
        # Active storage in CPU RAM (prevents disk reads on swap)
        self.cpu_cache = {}
        # Tracking what is currently occupying VRAM space
        self.active_in_vram = None
        self.transition_events: list[dict[str, Any]] = []
        # Set to False on the first pin_memory failure so subsequent models
        # never attempt pinning.  A failed pin can leave the CUDA driver context
        # in a dirty state; retrying (especially from a background thread while
        # CUDA runs on the main thread) reproduces the 0xC0000005 AV crash.
        self._global_pin_enabled = True

    def reset_transition_metrics(self) -> None:
        self.transition_events.clear()

    def _record_transition(self, operation: str, started: float, **details: Any) -> None:
        event = {
            "operation": str(operation),
            "duration_ms": round(max(0.0, time.perf_counter() - started) * 1000.0, 3),
            "active_in_vram": self.active_in_vram,
        }
        for key, value in details.items():
            if isinstance(value, float):
                event[key] = round(max(0.0, value), 3)
            elif isinstance(value, (str, int, bool)) or value is None:
                event[key] = value
        self.transition_events.append(event)
        if len(self.transition_events) > 64:
            del self.transition_events[: len(self.transition_events) - 64]

    def collect_transition_metrics(self) -> dict[str, Any]:
        events = list(self.transition_events)
        total_ms = sum(float(event.get("duration_ms") or 0.0) for event in events)
        h2d_ms = sum(float(event.get("h2d_ms") or 0.0) for event in events)
        d2h_ms = sum(float(event.get("d2h_ms") or 0.0) for event in events)
        cleanup_ms = sum(float(event.get("cleanup_ms") or 0.0) for event in events)
        return {
            "stage_transition_count": len(events),
            "stage_transition_total_ms": round(total_ms, 3),
            "stage_transition_h2d_ms": round(h2d_ms, 3),
            "stage_transition_d2h_ms": round(d2h_ms, 3),
            "stage_transition_cleanup_ms": round(cleanup_ms, 3),
            "stage_transition_events": events[-16:],
        }

    def register_model(self, model_key, model_object, *, pin: bool = True):
        """
        Store model weights in host CPU RAM for fast high/low swaps.

        Page-locked pinning is optional — on 16 GB cards with a 14B transformer,
        pinning can fail or trigger spurious CUDA OOM from prior GPU pressure, so
        we fall back to ordinary CPU tensors.
        """
        if model_key in self.cpu_cache:
            return
        started = time.perf_counter()
        logger.info("[AIWF] Registering %s to CPU cache (pin=%s)", model_key, pin)

        _free_cuda_memory()
        try:
            model_object.to("cpu")
        except Exception as exc:
            logger.warning("[AIWF] Could not move %s to CPU before cache register: %s", model_key, exc)

        # Pinning is page-locked host memory backed by the CUDA driver. On a 16 GB
        # card a 14B FP8 transformer can exhaust it; the FIRST failure leaves the CUDA
        # context in an error state, and continuing to call pin_memory on every
        # remaining tensor spams hundreds of warnings and can escalate to a hard
        # process crash (0xC0000005 access violation). So we disable pinning for the
        # WHOLE model on the first failure and fall back to ordinary CPU tensors —
        # swaps still work, just without the page-locked speedup.
        #
        # Critically, we also update _global_pin_enabled so that the NEXT model
        # (e.g. wan_low loaded in a background thread) never tries pin_memory either.
        # Calling pin_memory from a background thread while CUDA is active on the
        # main thread is what triggers the 0xC0000005 crash in practice.
        pin_state = {"enabled": bool(pin) and self._global_pin_enabled, "warned": False}

        def _store_tensor(tensor) -> None:
            if tensor is None:
                return
            if tensor.device.type != "cpu":
                tensor.data = tensor.data.to("cpu", non_blocking=False)
            if not pin_state["enabled"]:
                return
            try:
                tensor.data = tensor.data.pin_memory()
            except Exception as exc:  # RuntimeError / CUDA OOM (and anything else)
                pin_state["enabled"] = False
                self._global_pin_enabled = False  # propagate: no more pinning on this cache
                if not pin_state["warned"]:
                    pin_state["warned"] = True
                    logger.warning(
                        "[AIWF] pin_memory failed for %s (%s); keeping it on unpinned CPU RAM "
                        "for the rest of this model (this is normal for a 14B FP8 model on 16 GB).",
                        model_key, exc,
                    )
                    # pin_memory() failing with a CUDA error can leave the CUDA driver
                    # in a dirty state — subsequent CUDA allocs (e.g. text_encoder.to("cuda"))
                    # will report false OOMs until the async error is cleared.
                    # synchronize + empty_cache flushes the error queue.
                    _recover_cuda_after_pin_memory_failure()

        for param in model_object.parameters():
            _store_tensor(param)
            if param.grad is not None:
                param.grad = None

        for buffer in model_object.buffers():
            _store_tensor(buffer)

        self.cpu_cache[model_key] = model_object
        self._record_transition(
            "register_cpu_cache",
            started,
            model_key=str(model_key),
            pinned=bool(pin_state["enabled"]),
            pin_requested=bool(pin),
        )

    def _deferred_ipc_collect(self) -> None:
        """Run ipc_collect in a daemon thread — it takes 100–300 ms and need not block the swap."""
        try:
            import torch
            if torch.cuda.is_available() and hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        except Exception:
            pass

    def swap_models(self, old_key: str, new_key: str) -> None:
        """Evict old_key from VRAM and load new_key, overlapping cleanup with the new PCIe transfer.

        The key optimization: gc.collect() and empty_cache() run *while* the CPU→GPU transfer of
        the new model is already in flight over PCIe — saving 200–500 ms vs the old sequential
        pattern (evict → full cleanup → load).
        """
        import gc
        import torch

        if old_key not in self.cpu_cache or new_key not in self.cpu_cache:
            if old_key in self.cpu_cache:
                self.unload_from_vram(old_key)
            self.load_to_vram(new_key)
            return

        total_started = time.perf_counter()
        old_model = self.cpu_cache[old_key]
        new_model = self.cpu_cache[new_key]

        # 1. Issue non-blocking eviction of old model (GPU→CPU transfer starts).
        print(f"[AIWF] Swap {old_key} -> {new_key}: evicting from VRAM...")
        evict_started = time.perf_counter()
        old_model.to("cpu", non_blocking=True)
        self.active_in_vram = None

        # 2. Wait for old model's VRAM to be released (PCIe GPU→CPU done).
        torch.cuda.current_stream().synchronize()
        d2h_ms = max(0.0, time.perf_counter() - evict_started) * 1000.0

        # 3. If VRAM is still tight after eviction, return the freed pool to the allocator
        #    BEFORE streaming the new stage in, so two large stages are never committed at
        #    once (prevents OOM on 16 GB cards). Normally skipped — when the eviction freed
        #    plenty, we keep the eviction/stream overlap that makes swaps fast.
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            if total_bytes and free_bytes < 0.40 * total_bytes:
                torch.cuda.empty_cache()
        except Exception:
            pass

        # 4. Immediately start loading the new model (CPU→GPU PCIe transfer begins).
        print(f"[AIWF] Swap: streaming {new_key} to VRAM (cleanup overlapped with transfer)...")
        load_started = time.perf_counter()
        new_model.to(self.device, non_blocking=True)

        # 4. While PCIe transfer runs, do CPU-side cleanup (overlapped — free time!).
        cleanup_started = time.perf_counter()
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        cleanup_ms = max(0.0, time.perf_counter() - cleanup_started) * 1000.0

        # 5. Wait for new model to finish arriving in VRAM.
        torch.cuda.current_stream().synchronize()
        h2d_ms = max(0.0, time.perf_counter() - load_started) * 1000.0
        self.active_in_vram = new_key

        # 6. Defer ipc_collect (~100–300 ms, no urgency) so main thread returns immediately.
        t = threading.Thread(target=self._deferred_ipc_collect, daemon=True, name="aiwf-ipc-collect")
        t.start()
        self._record_transition(
            "swap_models",
            total_started,
            old_key=str(old_key),
            new_key=str(new_key),
            d2h_ms=d2h_ms,
            h2d_ms=h2d_ms,
            cleanup_ms=cleanup_ms,
        )

    def load_to_vram(self, target_key):
        """
        Instantly streams from pinned CPU RAM to VRAM via PCIe (non-blocking + sync).
        When eviction is needed, delegates to swap_models() which overlaps cleanup with the load.
        """
        import torch

        if self.active_in_vram == target_key:
            return

        if self.active_in_vram is not None:
            # swap_models overlaps cleanup with the new load — faster than sequential evict+load.
            self.swap_models(self.active_in_vram, target_key)
            return

        if target_key not in self.cpu_cache:
            print(f"[AIWF] Warning: {target_key} not registered in cache; skipping fast load.")
            return

        print(f"[AIWF] Streaming {target_key} to VRAM via PCIe (fast swap, no disk)...")
        started = time.perf_counter()
        target_model = self.cpu_cache[target_key]

        target_model.to(self.device, non_blocking=True)
        self.active_in_vram = target_key

        torch.cuda.current_stream().synchronize()
        self._record_transition(
            "load_to_vram",
            started,
            target_key=str(target_key),
            h2d_ms=max(0.0, time.perf_counter() - started) * 1000.0,
        )

    def unload_from_vram(self, target_key):
        """
        Evicts from VRAM back to CPU RAM (keeps object alive, no del/reload).
        Use swap_models() instead when immediately loading another model — it's faster.
        """
        import gc
        import torch

        if target_key not in self.cpu_cache:
            return

        print(f"[AIWF] Evicting {target_key} back to CPU RAM...")
        started = time.perf_counter()
        model = self.cpu_cache[target_key]

        model.to("cpu", non_blocking=True)
        self.active_in_vram = None

        if torch.cuda.is_available():
            try:
                # current_stream only — cheaper than full cuda.synchronize() (all streams).
                torch.cuda.current_stream().synchronize()
                torch.cuda.empty_cache()
            except Exception:
                pass
        gc.collect()
        self._record_transition(
            "unload_from_vram",
            started,
            target_key=str(target_key),
            d2h_ms=max(0.0, time.perf_counter() - started) * 1000.0,
        )


def _tune_wan_cpu_threads() -> None:
    """Set PyTorch intra/inter-op thread counts for hybrid-core CPUs (Intel 13th/14th gen).

    PyTorch defaults to ALL logical cores, which on a hybrid-core chip includes E-cores.
    E-cores are ~3× slower than P-cores for BF16/GGUF tensor math, so including them
    dilutes the thread pool and *increases* load time vs using P-cores only.

    Strategy: estimate P-core logical count and set that as intra-op thread count.
    Override with AIWF_WAN_CPU_THREADS env var (e.g. ``set AIWF_WAN_CPU_THREADS=12``).
    """
    import os

    try:
        import torch
    except ImportError:
        return

    def _set_threads(intra: int, inter: int, *, source: str) -> None:
        torch.set_num_threads(intra)
        try:
            torch.set_num_interop_threads(inter)
        except RuntimeError as exc:
            logger.debug(
                "[AIWF] CPU interop thread tuning skipped (%s): %s",
                source,
                exc,
            )
            return
        logger.info("[AIWF] CPU threads (%s): intra=%d, inter=%d", source, intra, inter)

    override = os.environ.get("AIWF_WAN_CPU_THREADS", "").strip()
    if override.isdigit() and int(override) > 0:
        n = int(override)
        _set_threads(n, max(1, n // 4), source="env override")
        return

    try:
        import psutil
        physical = psutil.cpu_count(logical=False) or 4
        logical_total = psutil.cpu_count(logical=True) or physical
        # Hybrid heuristic: P-cores have HT so they contribute extra logical threads.
        # E-cores are single-threaded. (logical - physical) ≈ HT threads = P-core count.
        # e.g. i5-13600K: 20 logical, 14 physical → 6 HT extras → 6 P-cores → 12 logical P-threads.
        p_cores = min(physical, max(1, logical_total - physical))
        n = max(4, p_cores * 2)
        _set_threads(n, 2, source=f"psutil: physical={physical} logical={logical_total} (P-cores only)")
    except ImportError:
        # psutil unavailable — use half of logical count as conservative P-core estimate.
        total = os.cpu_count() or 4
        n = max(4, total // 2)
        _set_threads(n, 2, source="os.cpu_count fallback")


class WanI2VBackend:
    """Loads a Wan image-to-video pipeline once and reuses it across renders.
    Uses AIWFModelCacheManager for pinned-CPU zero-disk swaps between high/low 14B
    and lighter components (text/VAE). This eliminates disk reloads on swaps and
    keeps models in RAM for instant PCIe .to() instead of del + safetensors load.
    """

    def __init__(self, *, async_offload: bool = True, pinned_memory: bool = True) -> None:
        self._pipe = None
        self._key = None
        self._cache_mode = "none"
        self._async_offload = bool(async_offload)
        self._pinned_memory = bool(pinned_memory)
        self.cache = AIWFModelCacheManager(device="cuda")
        self._reset_low_preload_state()
        _tune_wan_cpu_threads()

    def _reset_low_preload_state(self) -> None:
        thread = getattr(self, "_low_preload_thread", None)
        if thread is not None and thread.is_alive():
            try:
                thread.join(timeout=0.1)
            except Exception:
                pass
        self._preloaded_low = None
        self._low_preload_spec: dict[str, Any] | None = None
        self._low_preload_thread: threading.Thread | None = None
        self._low_preload_done = threading.Event()
        self._low_preload_done.set()
        self._low_preload_started = False
        self._low_preload_error: BaseException | None = None

    def _materialize_wan_transformer(
        self,
        target,
        weight_path: str,
        *,
        label: str,
        lora_path: str | None,
        lora_scale: float,
        lora_adapter: str,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        temporal_chunks: bool | None = None,
    ) -> tuple[list[str], list[str]]:
        import torch

        pp = Path(weight_path)
        if pp.suffix.lower() == ".gguf":
            _video_status(f"Loading GGUF {label} (mmap + on-the-fly dequant): {pp.name}")
            miss, unex = _load_gguf_transformer_weights(target, pp, torch_dtype=torch.bfloat16)
        elif pp.suffix.lower() == ".safetensors" and _safetensors_uses_comfy_fp8_quant(pp):
            _video_status(f"Loading native Comfy FP8 {label}: {pp.name}")
            miss, unex = _load_comfy_fp8_transformer_weights(target, pp, torch_dtype=torch.bfloat16)
        else:
            sd = _load_transformer_state_dict(weight_path, label=label)
            _video_status(f"Applying {label}: {pp.name}")
            miss, unex = target.load_state_dict(sd, strict=False, assign=True)
            del sd
        if lora_path:
            _video_status(f"Applying {label} LoRA: {Path(lora_path).name}")
        _apply_transformer_lora(target, lora_path, adapter_name=lora_adapter, weight=lora_scale)
        _ensure_wan_attention_processors(target, label)
        _apply_wan_attention_optimizations(
            target,
            label,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            temporal_chunks=temporal_chunks,
        )
        return list(miss), list(unex)

    def _run_low_preload_worker(self) -> None:
        spec = self._low_preload_spec
        if spec is None:
            self._low_preload_done.set()
            return
        try:
            low = _empty_wan_transformer(WAN_I2V_A14B_TRANSFORMER_CONFIG)
            miss, unex = self._materialize_wan_transformer(
                low,
                spec["low_path"],
                label="low-noise transformer",
                lora_path=spec.get("low_lora_path"),
                lora_scale=float(spec.get("low_lora_scale", 1.0)),
                lora_adapter="wan_low_lora",
                chunk_size=spec.get("chunk_size"),
                chunk_overlap=spec.get("chunk_overlap"),
                temporal_chunks=spec.get("temporal_chunks"),
            )
            if miss or unex:
                logger.warning(
                    "Low-noise background preload (%s): %d missing %d unexpected",
                    Path(spec["low_path"]).name,
                    len(miss),
                    len(unex),
                )
            self._preloaded_low = low
            if spec.get("use_cache"):
                self.cache.register_model("wan_low", low, pin=bool(spec.get("pin_tensors")))
            _video_status("Background low-noise transformer ready in CPU cache (fast VRAM swap at boundary).")
        except BaseException as exc:
            self._low_preload_error = exc
            logger.exception("Background low-noise transformer preload failed")
        finally:
            self._low_preload_done.set()

    def _maybe_start_background_low_preload(self) -> None:
        """Start loading low-noise weights to CPU while high stage runs on GPU."""
        if self._low_preload_spec is None or self._preloaded_low is not None:
            return
        if self._low_preload_started:
            return
        if self._low_preload_spec.get("disk_sequential"):
            return
        # Disk-sequential mode: don't start a background thread — low will load
        # synchronously at the boundary point AFTER wan_high is freed from the
        # CPU cache.  Starting a thread here (with CUDA active + no pinned memory)
        # is exactly what causes the 0xC0000005 AV crash on Windows.
        if not self._async_offload:
            return
        self._low_preload_started = True
        self._low_preload_done.clear()
        self._low_preload_error = None
        _video_status(
            "Background: loading low-noise transformer to CPU while high stage runs "
            "(boundary swap will stream to VRAM, not reload from disk)."
        )
        self._low_preload_thread = threading.Thread(
            target=self._run_low_preload_worker,
            name="aiwf-wan-low-preload",
            daemon=True,
        )
        self._low_preload_thread.start()

    def _ensure_low_preloaded(self) -> None:
        if self._preloaded_low is not None:
            return
        if self._low_preload_spec is None:
            raise WanUnavailable("Low-noise transformer was not configured for this pipeline.")
        if not self._low_preload_started:
            _video_status("Low-noise transformer not preloaded yet -- loading now before boundary swap.")
            self._run_low_preload_worker()
            return
        if not self._low_preload_done.is_set():
            _video_status("Waiting for background low-noise preload to finish...")
        self._low_preload_done.wait()
        if self._low_preload_error is not None:
            raise WanUnavailable(
                f"Low-noise transformer preload failed: {self._low_preload_error}"
            ) from self._low_preload_error
        if self._preloaded_low is None:
            raise WanUnavailable("Low-noise transformer preload finished without a model.")

    def _prepare_low_preload_for_generation(
        self,
        request,
        *,
        chunk_size: int,
        chunk_overlap: int,
        temporal_chunks: bool,
    ) -> None:
        """Start low-stage preload only when the pipeline did not already do it."""
        if self._preloaded_low is not None:
            _video_status("Low-noise transformer already preloaded; skipping duplicate preload.")
            return
        if self._low_preload_spec is None:
            self._low_preload_spec = {
                "low_path": str(request.low_noise_model_id),
                "low_lora_path": getattr(request, "low_noise_lora_id", None),
                "low_lora_scale": float(getattr(request, "low_noise_lora_scale", 1.0) or 1.0),
                "use_cache": _stage_cache_uses_cpu_standby(self._cache_mode),
                "pin_tensors": _stage_cache_pins_tensors(self._cache_mode) and self.cache._global_pin_enabled,
                "disk_sequential": _stage_cache_is_disk_sequential(self._cache_mode),
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "temporal_chunks": temporal_chunks,
            }
        self._maybe_start_background_low_preload()

    def available(self) -> bool:
        return wan_supported()

    def unload(self) -> None:
        self._reset_low_preload_state()
        # Drop any GPU-resident cached stage before releasing references so the
        # VRAM is actually reclaimed (not just unreferenced).
        try:
            if self.cache.active_in_vram is not None:
                self.cache.unload_from_vram(self.cache.active_in_vram)
        except Exception:
            pass
        try:
            if self._pipe is not None:
                for attr in ("transformer", "transformer_2"):
                    model = getattr(self._pipe, attr, None)
                    if _module_has_accelerate_hook(model):
                        setattr(self._pipe, attr, None)
        except Exception:
            pass
        self._pipe = None
        self._key = None
        self._cache_mode = "none"
        self._preloaded_low = None
        self.cache.cpu_cache.clear()
        self.cache.active_in_vram = None
        try:
            from aiwf.infrastructure.torch.wan_perf import restore_wan_attention_patch

            if restore_wan_attention_patch():
                _video_status("Restored default torch attention after Wan run.")
        except Exception:
            logger.debug("Wan attention cleanup failed.", exc_info=True)
        _free_cuda_memory()

    def _aspect_resize(self, pipe, image, max_area: int):
        """Resize the image to a Wan-valid size near ``max_area`` (model-aware)."""
        import numpy as np

        ar = image.height / image.width
        mod = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
        height = max(mod, round(np.sqrt(max_area * ar)) // mod * mod)
        width = max(mod, round(np.sqrt(max_area / ar)) // mod * mod)
        return image.resize((width, height)), int(height), int(width)

    def _ensure(
        self,
        *,
        high_noise_model_id: str | None = None,
        low_noise_model_id: str | None = None,
        boundary_ratio: float | None = None,
        vae_id: str | None = None,
        high_noise_lora_id: str | None = None,
        high_noise_lora_scale: float = 1.0,
        low_noise_lora_id: str | None = None,
        low_noise_lora_scale: float = 1.0,
        components_base: str | None = None,
        offload: str,
        flow_shift: float,
        sigma_type: str = "beta",
        sampler: str = "euler",
        text_encoder_path: str = "",
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        temporal_chunks: bool | None = None,
    ):
        import torch

        # Dual high/low is the only supported layout for Wan 2.2 I2V.
        if not (high_noise_model_id and low_noise_model_id):
            raise WanUnavailable(
                "Wan 2.2 image-to-video requires both a high-noise and a low-noise model "
                "(two-stage transformer pair). Select both before generating."
            )

        # NOTE: flow_shift, sigma_type, and sampler are NOT part of the cache key.
        # They affect only the scheduler (a cheap Python object rebuilt each run).
        # text_encoder_path IS in the cache key — a different encoder requires a reload.
        _cache_chunk_size = int(chunk_size or 24)
        _cache_chunk_overlap = int(chunk_overlap or 0)
        _cache_temporal_chunks = bool(temporal_chunks)
        fast_fp8_pair = (
            (
                _torch_native_fp8_available()
                and _is_native_comfy_fp8_transformer(high_noise_model_id)
                and _is_native_comfy_fp8_transformer(low_noise_model_id)
            )
            or (
                _is_gguf_transformer(high_noise_model_id)
                and _is_gguf_transformer(low_noise_model_id)
            )
        )
        offload = _resolve_dual_stage_offload_for_hardware(
            offload,
            fast_fp8_pair=fast_fp8_pair,
        )
        key = (
            "dual",
            high_noise_model_id,
            low_noise_model_id,
            boundary_ratio,
            vae_id or "default",
            high_noise_lora_id,
            round(float(high_noise_lora_scale), 3),
            low_noise_lora_id,
            round(float(low_noise_lora_scale), 3),
            components_base or "auto",
            offload,
            text_encoder_path or "",
            _cache_temporal_chunks,
            _cache_chunk_size,
            _cache_chunk_overlap,
        )

        if self._pipe is not None and self._key == key:
            return self._pipe

        # Switching to a different model set: fully evict the previous pipeline and
        # its pinned CPU cache (and flush VRAM) BEFORE building the new one. Otherwise
        # the prior generation's high/low transformers linger in RAM/VRAM and cause
        # OOM on the new load — even at low resolution, because it's weight memory,
        # not activation memory, that dominates for 14B models.
        if self._pipe is not None or self.cache.cpu_cache:
            _video_status("Releasing previous video pipeline before loading the new model set.")
            self.unload()

        _require_wan()
        logger.info(
            "Loading Wan I2V pipeline (dual high/low): high=%s low=%s boundary=%s vae=%s (offload=%s)",
            high_noise_model_id, low_noise_model_id, boundary_ratio, vae_id, offload,
        )
        _video_status(
            f"Loading video pipeline with local base components and dual transformers ({Path(high_noise_model_id).name} / {Path(low_noise_model_id).name})."
        )
        cache_mode = _wan_cache_mode(
            offload,
            fast_fp8_pair=fast_fp8_pair,
            pinned_memory=self._pinned_memory,
        )

        pipe = self._load_dual_pipeline(
            high_path=high_noise_model_id,
            low_path=low_noise_model_id,
            boundary_ratio=boundary_ratio or 0.875,
            vae_id=vae_id,
            high_lora_path=high_noise_lora_id,
            high_lora_scale=high_noise_lora_scale,
            low_lora_path=low_noise_lora_id,
            low_lora_scale=low_noise_lora_scale,
            components_base=components_base,
            offload=offload,
            cache_mode=cache_mode,
            flow_shift=flow_shift,
            text_encoder_path=text_encoder_path or "",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            temporal_chunks=_cache_temporal_chunks,
        )

        _sampler = str(sampler or "euler")
        _sigma = str(sigma_type or "beta")
        if _sampler == "heun":
            from diffusers import FlowMatchHeunDiscreteScheduler
            base_cfg = getattr(pipe.scheduler, "config", pipe.scheduler)
            shift = float(flow_shift or getattr(base_cfg, "flow_shift", getattr(base_cfg, "shift", 5.0)) or 5.0)
            pipe.scheduler = FlowMatchHeunDiscreteScheduler(
                num_train_timesteps=int(getattr(base_cfg, "num_train_timesteps", 1000) or 1000),
                shift=shift,
                use_dynamic_shifting=bool(getattr(base_cfg, "use_dynamic_shifting", False)),
            )
            _video_status(f"Using Wan sampler: FlowMatch Heun (2nd-order) | shift={shift:g}")
        else:
            pipe.scheduler = _new_wan_euler_scheduler(
                pipe.scheduler,
                flow_shift=float(flow_shift),
                sigma_type=_sigma,
            )
            _video_status(
                f"Using Wan sampler: FlowMatch Euler | scheduler={_sigma} | shift={float(flow_shift):g}"
            )

        if offload == "sequential":
            pipe.enable_sequential_cpu_offload()
        elif offload in {"group", "streamed"}:
            streamed = offload == "streamed"
            blocks = 1 if streamed else _wan_group_offload_blocks()
            stream_enabled = True if streamed else _wan_group_offload_stream_enabled()
            if not _install_group_offload_for_stage(
                pipe.transformer,
                "cuda:0",
                blocks=blocks,
                use_stream=stream_enabled,
                record_stream=True if streamed else None,
                low_cpu_mem_usage=_wan_group_offload_low_cpu_mem_usage() if streamed else None,
            ):
                raise WanUnavailable(
                    "Diffusers group offload could not be installed for the high-noise Wan transformer. "
                    "Use model or sequential offload for this run."
                )
            _video_status(
                f"Using {'streamed ' if streamed else ''}block-level group offload "
                f"for the high-noise transformer ({blocks} blocks/group, "
                f"stream={'on' if stream_enabled else 'off'})."
            )
            pipe._aiwf_execution_device = torch.device("cuda")
        elif offload in {"model", "balanced", "resident"}:
            if offload == "resident" and not fast_fp8_pair:
                raise WanUnavailable(
                    "Resident speed mode requires a quantized Wan high/low pair (native FP8 safetensors or GGUF). "
                    "Use Balanced or Low VRAM model offload for full-precision transformers."
                )
            if fast_fp8_pair:
                if offload == "balanced":
                    _video_status(
                        "Using balanced quantized placement: active Wan stage swaps, VAE stays on GPU, text encoder offloads."
                    )
                elif offload == "resident":
                    _video_status(
                        "Using resident quantized placement: high and low Wan transformers stay on GPU; text encoder/VAE offload."
                    )
                else:
                    _video_status(
                        "Using low-VRAM quantized placement: active Wan stage stays on GPU while text encoder/VAE offload."
                    )
                original_seq = getattr(pipe, "model_cpu_offload_seq", None)
                original_exclude = list(getattr(pipe, "_exclude_from_cpu_offload", []) or [])
                # Build safe offload seq - some manual Wan assemblies don't have image_encoder
                seq_parts = ["text_encoder"]
                if hasattr(pipe, "image_encoder") and getattr(pipe, "image_encoder", None) is not None:
                    seq_parts.append("image_encoder")
                if offload != "balanced":
                    seq_parts.append("vae")
                transformer_excludes = {"transformer", "transformer_2"}
                if offload == "balanced":
                    transformer_excludes.add("vae")
                pipe.model_cpu_offload_seq = "->".join(seq_parts)
                pipe._exclude_from_cpu_offload = sorted(
                    set(original_exclude).union(transformer_excludes)
                )
                try:
                    pipe.enable_model_cpu_offload()
                finally:
                    if original_seq:
                        pipe.model_cpu_offload_seq = original_seq
                if offload == "balanced":
                    try:
                        pipe.vae.to("cuda")
                    except Exception as exc:
                        raise WanUnavailable(
                            "Balanced mode could not keep the Wan VAE resident on GPU. "
                            "Use Low VRAM model offload for this run."
                        ) from exc
                elif offload == "resident":
                    try:
                        pipe.transformer.to("cuda")
                        if getattr(pipe, "transformer_2", None) is not None:
                            pipe.transformer_2.to("cuda")
                    except Exception as exc:
                        raise WanUnavailable(
                            "Resident mode could not keep both Wan FP8 stages on GPU. "
                            "Use Balanced or Low VRAM model offload for this run."
                        ) from exc
                    pipe._aiwf_execution_device = torch.device("cuda")
            else:
                pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")

        _ensure_wan_attention_processors(pipe.transformer, "high-noise transformer")
        # Apply SageAttention / flash SDPA / channels_last to the FP8 transformer.
        # This was previously only done on the GGUF path — FP8 was silently missing it.
        _apply_wan_attention_optimizations(
            pipe.transformer,
            "high-noise transformer",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            temporal_chunks=_cache_temporal_chunks,
        )
        if _stage_cache_is_gpu_active_cpu_standby(cache_mode):
            _free_cuda_memory()
            # In active-stage cache modes, do not put wan_high in VRAM before
            # prompt encoding. UMT5 temporarily needs several GB on CUDA, and a
            # resident 14B FP8 expert leaves too little headroom on 16 GB cards.
            # Native denoise reloads the stage before its first transformer call;
            # the pre-forward hook covers the non-native diffusers call path.
            _loaded = [False]

            def _deferred_vram_load(module, args, _cache=self.cache):
                if not _loaded[0]:
                    _loaded[0] = True
                    if getattr(_cache, "active_in_vram", None) != "wan_high":
                        _video_status(
                            "Deferred wan_high VRAM load after prompt encoding "
                            "(preserves text-encoder headroom)."
                        )
                        _cache.load_to_vram("wan_high")

            pipe.transformer.register_forward_pre_hook(_deferred_vram_load)
            if _stage_cache_is_disk_sequential(cache_mode):
                _video_status(
                    "Disk-sequential: wan_high VRAM load deferred to first denoising step "
                    "(encode_prompt needs that VRAM first)."
                )
            else:
                _video_status(
                    "Wan active-stage cache: wan_high VRAM load deferred until denoising "
                    "so prompt encoding has enough headroom."
                )

        for method in ("enable_tiling", "enable_slicing"):
            try:
                getattr(pipe.vae, method)()
            except Exception:
                pass

        self._pipe = pipe
        self._key = key
        self._cache_mode = str(getattr(pipe, "_aiwf_cache_mode", cache_mode) or cache_mode)
        return pipe

    def _ensure_single_5b(
        self,
        *,
        model_id: str,
        vae_id: str | None = None,
        components_base: str | None = None,
        text_encoder_path: str = "",
        offload: str,
        flow_shift: float,
        sigma_type: str = "beta",
        sampler: str = "euler",
    ):
        import torch
        from diffusers import WanImageToVideoPipeline
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        from transformers import AutoTokenizer

        key = (
            "single_5b",
            model_id,
            vae_id or "default",
            components_base or "auto",
            text_encoder_path or "base",
            offload,
            sampler,
            sigma_type,
            round(float(flow_shift), 3),
        )
        if self._pipe is not None and self._key == key:
            return self._pipe

        if self._pipe is not None or self.cache.cpu_cache:
            _video_status("Releasing previous video pipeline before loading the 5B demo model.")
            self.unload()

        _require_wan()
        model_path = Path(str(model_id))
        if model_path.is_file():
            base = components_base or _find_wan_components_base()
            if base is None:
                raise WanUnavailable(
                    "Standalone Wan 5B transformer needs a local components base with text_encoder, "
                    "tokenizer, and scheduler. See docs/WAN_LOCAL_COMPONENTS.md."
                )
            base_path = Path(base)
            _video_status(f"Using local video base for Wan 5B: {base_path}")
            if vae_id:
                vae = _load_wan_vae(vae_id, torch_dtype=torch.float32)
            else:
                vae = _load_wan_vae(str(base_path), torch_dtype=torch.float32)
            if text_encoder_path:
                text_encoder = _load_standalone_umt5_text_encoder(text_encoder_path, torch_dtype=torch.bfloat16)
            else:
                text_encoder = _load_umt5_text_encoder(base_path / "text_encoder", torch_dtype=torch.bfloat16)
            tokenizer = AutoTokenizer.from_pretrained(str(base_path / "tokenizer"), local_files_only=True)
            scheduler = UniPCMultistepScheduler.from_pretrained(
                str(base_path / "scheduler"), local_files_only=True
            )

            _video_status(f"Building Wan 5B pipeline from standalone transformer: {model_path.name}")
            transformer = _empty_wan_transformer(WAN_TI2V_5B_TRANSFORMER_CONFIG)
            if model_path.suffix.lower() == ".gguf":
                miss, unex = _load_gguf_transformer_weights(
                    transformer, model_path, torch_dtype=torch.bfloat16
                )
            elif model_path.suffix.lower() == ".safetensors" and _safetensors_uses_comfy_fp8_quant(model_path):
                miss, unex = _load_comfy_fp8_transformer_weights(
                    transformer, model_path, torch_dtype=torch.bfloat16
                )
            else:
                sd = _load_transformer_state_dict(str(model_path), label="Fast 5B transformer")
                miss, unex = transformer.load_state_dict(sd, strict=False, assign=True)
                del sd
            if miss or unex:
                logger.warning(
                    "Wan 5B weights (%s) vs base: %d missing %d unexpected",
                    model_path.name,
                    len(miss),
                    len(unex),
                )
            pipe = WanImageToVideoPipeline(
                transformer=transformer,
                transformer_2=None,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                vae=vae,
                scheduler=scheduler,
                expand_timesteps=True,
            )
        else:
            if model_path.exists() and not (model_path / "transformer" / "config.json").is_file():
                raise WanUnavailable(
                    "Fast 5B selected a shared component base, not a full pipeline. "
                    "Select the standalone Wan TI2V 5B transformer file instead."
                )
            _video_status(f"Loading standalone Wan 5B Diffusers pipeline: {model_id}")
            load_kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}
            if model_path.exists():
                load_kwargs["local_files_only"] = True
            pipe = WanImageToVideoPipeline.from_pretrained(str(model_id), **load_kwargs)

        if vae_id and not model_path.is_file():
            _video_status(f"Using explicit Wan VAE for 5B path: {Path(vae_id).name}")
            pipe.vae = _load_wan_vae(vae_id, torch_dtype=torch.float32)

        _sampler = str(sampler or "euler")
        _sigma = str(sigma_type or "beta")
        if _sampler == "heun":
            from diffusers import FlowMatchHeunDiscreteScheduler

            base_cfg = getattr(pipe.scheduler, "config", pipe.scheduler)
            shift = float(flow_shift or getattr(base_cfg, "flow_shift", getattr(base_cfg, "shift", 5.0)) or 5.0)
            pipe.scheduler = FlowMatchHeunDiscreteScheduler(
                num_train_timesteps=int(getattr(base_cfg, "num_train_timesteps", 1000) or 1000),
                shift=shift,
                use_dynamic_shifting=bool(getattr(base_cfg, "use_dynamic_shifting", False)),
            )
            _video_status(f"Using Wan sampler: FlowMatch Heun (2nd-order) | shift={shift:g}")
        else:
            pipe.scheduler = _new_wan_euler_scheduler(
                pipe.scheduler,
                flow_shift=float(flow_shift),
                sigma_type=_sigma,
            )
            _video_status(f"Using Wan sampler: FlowMatch Euler | scheduler={_sigma} | shift={float(flow_shift):g}")

        if offload == "sequential":
            pipe.enable_sequential_cpu_offload()
        elif offload in {"group", "streamed"}:
            streamed = offload == "streamed"
            blocks = 1 if streamed else _wan_group_offload_blocks()
            stream_enabled = True if streamed else _wan_group_offload_stream_enabled()
            if not _install_group_offload_for_stage(
                getattr(pipe, "transformer", None),
                "cuda:0",
                blocks=blocks,
                use_stream=stream_enabled,
                record_stream=True if streamed else None,
                low_cpu_mem_usage=_wan_group_offload_low_cpu_mem_usage() if streamed else None,
            ):
                raise WanUnavailable(
                    "Diffusers group offload could not be installed for the Wan 5B transformer. "
                    "Use model or sequential offload for this run."
                )
            _video_status(
                f"Using {'streamed ' if streamed else ''}block-level group offload "
                f"for Wan 5B ({blocks} blocks/group, stream={'on' if stream_enabled else 'off'})."
            )
            pipe._aiwf_execution_device = torch.device("cuda")
        elif offload == "model":
            pipe.enable_model_cpu_offload()
        elif offload == "balanced":
            pipe.enable_model_cpu_offload()
            try:
                pipe.vae.to("cuda")
            except Exception as exc:
                raise WanUnavailable(
                    "Balanced mode could not keep the Wan 5B VAE resident on GPU. "
                    "Use Low VRAM model offload for this run."
                ) from exc
        elif offload == "resident":
            pipe.to("cuda")
        else:
            pipe.to("cuda")

        transformer = getattr(pipe, "transformer", None)
        if transformer is not None:
            _ensure_wan_attention_processors(transformer, "5B transformer")
            _apply_wan_attention_optimizations(transformer, "5B transformer")

        for method in ("enable_tiling", "enable_slicing"):
            try:
                getattr(pipe.vae, method)()
            except Exception:
                pass

        self._pipe = pipe
        self._key = key
        self._cache_mode = "none"
        return pipe

    def _load_dual_pipeline(
        self,
        high_path: str,
        low_path: str,
        *,
        boundary_ratio: float,
        vae_id: str | None = None,
        high_lora_path: str | None = None,
        high_lora_scale: float = 1.0,
        low_lora_path: str | None = None,
        low_lora_scale: float = 1.0,
        components_base: str | None = None,
        offload: str,
        cache_mode: str = "none",
        flow_shift: float,
        text_encoder_path: str = "",
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        temporal_chunks: bool | None = None,
    ):
        """Load a WanImageToVideoPipeline configured with transformer (high-noise) + transformer_2 (low-noise)."""
        import torch
        from diffusers import WanImageToVideoPipeline, WanTransformer3DModel
        from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
        from transformers import AutoTokenizer

        base = components_base or _find_wan_components_base()
        if base is None:
            raise WanUnavailable(
                "Dual high/low Wan models require a components base (text_encoder + tokenizer + scheduler + model_index.json). "
                "WanService preflight should locate and pass it via components_base. "
                "Ensure models/wan/Diffusers/Wan2.2-TI2V-5B-Diffusers (or equivalent) has the required files. "
                "See docs/WAN_LOCAL_COMPONENTS.md. (No base found via explicit value or search.)"
            )
        _video_status(f"Using local video base: {base}")

        # Prefer explicit user VAE (your Comfy Wan 2.1 VAE .safetensors) because "oddly wan 2.2 uses 2.1 vae".
        # Falls back to cleaned VAE from the base components folder.
        if vae_id:
            vae = _load_wan_vae(vae_id, torch_dtype=torch.float32)
        else:
            vae = _load_wan_vae(base, torch_dtype=torch.float32)

        base_path = Path(base)
        _video_status("Loading local text encoder, tokenizer, and scheduler.")
        if text_encoder_path:
            text_encoder = _load_standalone_umt5_text_encoder(text_encoder_path, torch_dtype=torch.bfloat16)
        else:
            text_encoder = _load_umt5_text_encoder(base_path / "text_encoder", torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(str(base_path / "tokenizer"), local_files_only=True)
        scheduler = UniPCMultistepScheduler.from_pretrained(
            str(base_path / "scheduler"), local_files_only=True
        )

        use_cache = _stage_cache_uses_cpu_standby(cache_mode)
        dual_gpu_resident = _stage_cache_is_dual_gpu_resident(cache_mode)
        pin_tensors = _stage_cache_pins_tensors(cache_mode) and self._pinned_memory
        _video_status("Building local Wan A14B I2V pipeline components.")
        _video_status("Preparing empty high-noise transformer stage.")
        high_trans = _empty_wan_transformer(WAN_I2V_A14B_TRANSFORMER_CONFIG)
        pipe = WanImageToVideoPipeline(
            transformer=high_trans,
            transformer_2=None,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
            scheduler=scheduler,
            boundary_ratio=float(boundary_ratio),
        )
        _video_status("Base video pipeline loaded.")

        # Inject high into the primary transformer (overwrites the base one in-place)
        high_pp = Path(high_path)
        if high_pp.suffix.lower() == ".gguf":
            _video_status(f"Loading GGUF high-noise transformer (mmap + on-the-fly dequant): {high_pp.name}")
            miss_h, unex_h = _load_gguf_transformer_weights(
                pipe.transformer, high_pp, torch_dtype=torch.bfloat16
            )
        elif high_pp.suffix.lower() == ".safetensors" and _safetensors_uses_comfy_fp8_quant(high_pp):
            _video_status(f"Loading native Comfy FP8 high-noise transformer: {high_pp.name}")
            miss_h, unex_h = _load_comfy_fp8_transformer_weights(
                pipe.transformer, high_pp, torch_dtype=torch.bfloat16
            )
        else:
            high_sd = _load_transformer_state_dict(high_path, label="High-noise transformer")
            _video_status(f"Applying high-noise transformer: {Path(high_path).name}")
            miss_h, unex_h = pipe.transformer.load_state_dict(high_sd, strict=False, assign=True)
            del high_sd
        if miss_h or unex_h:
            logger.warning("High-noise weights (%s) vs base: %d missing %d unexpected", Path(high_path).name, len(miss_h), len(unex_h))
        if high_lora_path:
            _video_status(f"Applying high-noise LoRA: {Path(high_lora_path).name}")
        _apply_transformer_lora(
            pipe.transformer,
            high_lora_path,
            adapter_name="wan_high_lora",
            weight=high_lora_scale,
        )
        _ensure_wan_attention_processors(pipe.transformer, "high-noise transformer")

        # Apply memory-efficient attention (SDP + channels_last) to the loaded high stage.
        _apply_wan_attention_optimizations(
            pipe.transformer,
            "high-noise transformer",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            temporal_chunks=temporal_chunks,
        )

        if use_cache:
            self.cache.register_model("wan_high", pipe.transformer, pin=pin_tensors)
            decision = _resolve_stage_cache_after_pin_probe(
                cache_mode,
                high_path=high_path,
                low_path=low_path,
                pin_available=self.cache._global_pin_enabled,
            )
            if decision.mode.value != cache_mode:
                cache_mode = decision.mode.value
                use_cache = _stage_cache_uses_cpu_standby(cache_mode)
                pin_tensors = _stage_cache_pins_tensors(cache_mode) and self._pinned_memory
                if _stage_cache_is_disk_sequential(cache_mode):
                    _video_status(
                        "Survival mode: low-noise model will load from disk at stage boundary. "
                        "This is expected to be slow and is not demo-ready."
                    )
                    if decision.available_ram_gb is not None and decision.additional_required_gb is not None:
                        _video_status(
                            "Survival mode RAM preflight: "
                            f"available={decision.available_ram_gb:.1f} GB, "
                            f"required_for_low_standby={decision.additional_required_gb:.1f} GB."
                        )
                else:
                    _video_status(
                        "Wan stage cache: pinning unavailable; using unpinned CPU standby "
                        "so the low-noise stage can stay in RAM instead of loading from disk."
                    )
            pipe._aiwf_cache_mode = cache_mode
            pipe._aiwf_stage_cache = self.cache
            if cache_mode == "full":
                self.cache.load_to_vram("wan_high")

        low_pp = Path(low_path)
        # Disk-sequential mode: triggered when pin_memory failed for wan_high.
        #
        # Two problems occur when we try to load both 14B FP8 models at once:
        # 1. Background thread + safetensors mmap + active CUDA → 0xC0000005 AV crash.
        # 2. Synchronous load while wan_high occupies CPU cache → 2×14GB ≈ 28GB peaks
        #    RAM, which is too tight on a 32GB machine and causes another AV.
        #
        # Solution — disk-sequential: defer wan_low entirely. At the boundary point
        # _release_high_stage will DELETE wan_high from the CPU cache (freeing ~14GB),
        # then _load_low_stage loads wan_low from disk into the freed space.
        # No background thread, never two 14B models in RAM simultaneously.
        disk_sequential = _stage_cache_is_disk_sequential(cache_mode)
        if disk_sequential:
            _video_status(
                "Disk-sequential mode: wan_high will be freed from CPU cache at the "
                "boundary point and wan_low will load from disk then. "
                "(pin_memory unavailable — loading both 14B models simultaneously would OOM.)"
            )
        defer_low_preload = disk_sequential or (
            _stage_cache_is_gpu_active_cpu_standby(cache_mode) and self._async_offload
        )
        self._reset_low_preload_state()

        if defer_low_preload:
            self._low_preload_spec = {
                "low_path": low_path,
                "low_lora_path": low_lora_path,
                "low_lora_scale": low_lora_scale,
                "use_cache": use_cache,
                "pin_tensors": pin_tensors,
                "disk_sequential": disk_sequential,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "temporal_chunks": temporal_chunks,
            }
            _video_status(
                "Deferring low-noise load: will preload to CPU in background during high-stage denoising."
            )
            self._preloaded_low = None
        else:
            self._low_preload_spec = None
            _video_status("Preloading low-noise transformer weights to CPU.")
            preloaded_low = _empty_wan_transformer(WAN_I2V_A14B_TRANSFORMER_CONFIG)
            miss_l_pre, unex_l_pre = self._materialize_wan_transformer(
                preloaded_low,
                low_path,
                label="low-noise transformer",
                lora_path=low_lora_path,
                lora_scale=low_lora_scale,
                lora_adapter="wan_low_lora",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                temporal_chunks=temporal_chunks,
            )
            if miss_l_pre or unex_l_pre:
                logger.warning(
                    "Low-noise preload (%s): %d missing %d unexpected",
                    low_pp.name,
                    len(miss_l_pre),
                    len(unex_l_pre),
                )
            self._preloaded_low = preloaded_low
            if use_cache:
                self.cache.register_model("wan_low", preloaded_low, pin=pin_tensors)

        if dual_gpu_resident:
            pipe.transformer_2 = self._preloaded_low
            pipe.register_to_config(boundary_ratio=float(boundary_ratio))
            pipe._aiwf_cache_mode = cache_mode
            pipe._aiwf_stage_cache = self.cache
            _video_status(
                "Resident mode: high and low Wan transformers are materialized together; "
                "boundary swap is disabled for this run."
            )
            logger.info(
                "Configured resident dual-stage Wan pipeline: high=%s, low=%s, boundary_ratio=%s (from base %s)",
                Path(high_path).name, Path(low_path).name, boundary_ratio, Path(base).name
            )
            return pipe

        def _release_high_stage():
            _video_status(
                "High-noise stage complete; releasing high transformer for low-stage headroom."
            )
            if use_cache:
                self.cache.unload_from_vram("wan_high")
                if disk_sequential:
                    # Free wan_high from the CPU cache BEFORE loading wan_low from disk.
                    # Without this, both 14B FP8 models (~28GB) would live in RAM at
                    # the same time, which OOMs a 32GB machine and causes 0xC0000005.
                    _video_status(
                        "Disk-sequential: evicting wan_high from CPU cache to make room "
                        "for wan_low disk load (~14GB freed)."
                    )
                    self.cache.cpu_cache.pop("wan_high", None)
                    if _stage_cache_is_gpu_active_cpu_standby(cache_mode):
                        pipe.transformer = None
            elif pipe.transformer is not None:
                if not _module_has_accelerate_hook(pipe.transformer):
                    try:
                        pipe.transformer.to("cpu")
                    except Exception:
                        pass

            if pipe.transformer is not None and not _stage_cache_is_gpu_active_cpu_standby(cache_mode):
                pipe.transformer = None
            if not use_cache:
                # In sequential/no-cache placement the low-stage proxy moves the
                # real low transformer onto CUDA at first use. Make sure the high
                # stage's CUDA allocations are actually released before that
                # transfer starts; otherwise the first low step can run under
                # severe allocator pressure instead of failing cleanly.
                _free_cuda_memory()

        def _load_low_stage():
            _video_status("High stage done -- swapping to low-noise transformer.")

            self._ensure_low_preloaded()
            # swap_models() (called inside load_to_vram) handles gc + empty_cache overlapped
            # with the PCIe transfer — don't duplicate cleanup here.
            low_trans = self._preloaded_low
            if use_cache:
                self.cache.load_to_vram("wan_low")
            elif offload == "sequential":
                device = getattr(pipe, "_offload_device", None) or "cuda:0"
                if _install_sequential_cpu_offload_for_stage(low_trans, device):
                    _video_status(
                        "Sequential offload: lazy low-noise transformer will stream submodules to VRAM "
                        "instead of loading the full FP8 stage at once."
                    )
                else:
                    raise WanUnavailable(
                        "Sequential CPU offload could not be installed for the lazy low-noise transformer. "
                        "Refusing to load the full FP8 low stage into VRAM because that can exhaust a 16 GB GPU. "
                        "Try model offload/cache mode, or fix the Accelerate CPU offload install before retrying."
                    )
            elif offload in {"group", "streamed"}:
                streamed = offload == "streamed"
                blocks = 1 if streamed else _wan_group_offload_blocks()
                stream_enabled = True if streamed else _wan_group_offload_stream_enabled()
                if _install_group_offload_for_stage(
                    low_trans,
                    "cuda:0",
                    blocks=blocks,
                    use_stream=stream_enabled,
                    record_stream=True if streamed else None,
                    low_cpu_mem_usage=_wan_group_offload_low_cpu_mem_usage() if streamed else None,
                ):
                    _video_status(
                        f"{'Streamed group' if streamed else 'Group'} offload: "
                        "lazy low-noise transformer will stream block groups to VRAM "
                        f"({blocks} blocks/group, stream={'on' if stream_enabled else 'off'})."
                    )
                else:
                    raise WanUnavailable(
                        "Diffusers group offload could not be installed for the lazy low-noise transformer. "
                        "Use model or sequential offload for this run."
                    )
            elif offload != "sequential":
                try:
                    low_trans.to("cuda")
                except Exception as exc:
                    logger.warning("Could not move low-noise transformer to CUDA: %s", exc)
            return low_trans

        if defer_low_preload:
            _video_status("Low-noise transformer will load in background; boundary swap streams from CPU cache.")
        else:
            _video_status("Low-noise transformer preloaded to CPU; boundary swap should be fast.")
        pipe.transformer_2 = _new_lazy_wan_transformer(
            WAN_I2V_A14B_TRANSFORMER_CONFIG,
            dtype=torch.bfloat16,
            load_model=_load_low_stage,
            before_load=_release_high_stage,
        )

        # Tell the pipeline about the switch point (registered into self.config)
        # Typical values for Wan2.2 14B I2V high/low splits are around 0.875
        pipe.register_to_config(boundary_ratio=float(boundary_ratio))
        pipe._aiwf_cache_mode = cache_mode
        pipe._aiwf_stage_cache = self.cache

        logger.info(
            "Configured dual-stage Wan pipeline: high=%s, low=%s, boundary_ratio=%s (from base %s)",
            Path(high_path).name, Path(low_path).name, boundary_ratio, Path(base).name
        )
        return pipe

    def generate(self, request, image, *, on_progress=None, should_cancel=None):
        """Run image->video. Returns list of frames (PIL or numpy arrays from VAE decode)."""
        _require_wan()
        import torch

        self.cache.reset_transition_metrics()
        requires_dual = (
            request.requires_dual_transformers()
            if callable(getattr(request, "requires_dual_transformers", None))
            else True
        )
        if requires_dual and not (getattr(request, "uses_dual_transformers", None) and request.uses_dual_transformers()):
            raise WanUnavailable(
                "Wan 2.2 image-to-video needs BOTH a high-noise and a low-noise model. "
                "Select a High noise model and a Low noise model -- Wan 2.2 always uses a "
                "two-stage high/low transformer pair (required even when using LoRAs)."
            )

        _sigma_type = str(getattr(request, "sigma_type", "beta") or "beta")
        _sampler = str(getattr(request, "sampler", "euler") or "euler")
        _flow_shift = float(getattr(request, "flow_shift", 5.0) or 5.0)
        _te_path = str(getattr(request, "text_encoder_path", "") or "")

        _chunk_size = int(getattr(request, "chunk_size", 24) or 24)
        _chunk_overlap = int(getattr(request, "chunk_overlap", 0) or 0)
        _temporal_chunks = bool(getattr(request, "temporal_chunks", False))
        if os.environ.get("AIWF_WAN_TEMPORAL_CHUNKS", "").strip():
            _temporal_chunks = _env_flag("AIWF_WAN_TEMPORAL_CHUNKS", default=_temporal_chunks)
        _image_guidance_scale = float(getattr(request, "image_guidance_scale", 1.0) or 1.0)
        _manual_vae_decode = _env_flag("AIWF_WAN_MANUAL_VAE_DECODE", default=False)
        _vae_decode_chunk_frames = _env_int("AIWF_WAN_VAE_CHUNK_FRAMES", 4)
        # AIWF owns the denoise loop by default instead of treating
        # diffusers' pipe(**call_kwargs) as a black box.
        # Escape hatch retained in case real-hardware validation surfaces a regression.
        _use_native_denoise = _env_flag("AIWF_WAN_NATIVE_DENOISE", default=True)
        try:
            _reserve_device = torch.cuda.current_device() if torch.cuda.is_available() else 0
        except Exception:
            _reserve_device = 0
        vram_budget = apply_cuda_vram_reserve(
            enabled=bool(getattr(request, "vram_reserve_enabled", False)),
            reserve_mb=int(getattr(request, "vram_reserve_mb", 0) or 0),
            device=_reserve_device,
            torch_module=torch,
        )
        if vram_budget.enabled:
            if not vram_budget.applied:
                raise WanUnavailable(f"VRAM reserve could not be applied: {vram_budget.message}")
            _video_status(vram_budget.message)

        load_started = time.perf_counter()
        if requires_dual:
            pipe = self._ensure(
                high_noise_model_id=request.high_noise_model_id,
                low_noise_model_id=request.low_noise_model_id,
                boundary_ratio=getattr(request, "boundary_ratio", None),
                vae_id=getattr(request, "vae_id", None),
                high_noise_lora_id=getattr(request, "high_noise_lora_id", None),
                high_noise_lora_scale=float(getattr(request, "high_noise_lora_scale", 1.0) or 1.0),
                low_noise_lora_id=getattr(request, "low_noise_lora_id", None),
                low_noise_lora_scale=float(getattr(request, "low_noise_lora_scale", 1.0) or 1.0),
                components_base=getattr(request, "components_base", None),
                offload=str(getattr(request, "offload", "model") or "model"),
                flow_shift=_flow_shift,
                sigma_type=_sigma_type,
                sampler=_sampler,
                text_encoder_path=_te_path,
                chunk_size=_chunk_size,
                chunk_overlap=_chunk_overlap,
                temporal_chunks=_temporal_chunks,
            )
        else:
            pipe = self._ensure_single_5b(
                model_id=str(getattr(request, "model_id", "") or ""),
                vae_id=getattr(request, "vae_id", None),
                components_base=getattr(request, "components_base", None),
                text_encoder_path=_te_path,
                offload=str(getattr(request, "offload", "model") or "model"),
                flow_shift=_flow_shift,
                sigma_type=_sigma_type,
                sampler=_sampler,
            )
        load_seconds = max(0.0, time.perf_counter() - load_started)

        preprocess_started = time.perf_counter()
        if requires_dual:
            # Recompute boundary_ratio from the scheduler's actual timestep distribution
            # so high_noise_steps + low_noise_steps is honoured regardless of flow_shift.
            high_steps = max(1, int(getattr(request, "high_noise_steps", 4) or 4))
            low_steps = max(1, int(getattr(request, "low_noise_steps", 4) or 4))
            total_steps = high_steps + low_steps
            stage_boundary_ratio = _boundary_ratio_for_step_split(
                pipe.scheduler, total_steps=total_steps, high_steps=high_steps
            )
            req_boundary = float(getattr(request, "boundary_ratio", 0.875) or 0.875)
            if abs(stage_boundary_ratio - req_boundary) > 0.05:
                logger.info(
                    "Boundary ratio adjusted: scheduler-derived=%.3f request=%.3f",
                    stage_boundary_ratio, req_boundary,
                )
            pipe.register_to_config(boundary_ratio=stage_boundary_ratio)
        else:
            high_steps = max(1, int(getattr(request, "steps", 8) or 8))
            low_steps = 0
            total_steps = high_steps

        seed = int(getattr(request, "seed", -1))
        if seed < 0:
            import random
            seed = random.randint(0, 2 ** 32 - 1)

        max_area = int(getattr(request, "width", 480)) * int(getattr(request, "height", 480))
        image, h, w = self._aspect_resize(pipe, image, max_area)

        real_device = self._real_device(pipe)
        generator = torch.Generator(device=real_device).manual_seed(seed)

        if requires_dual:
            # Kick off the low-noise preload only if _load_dual_pipeline did not
            # already preload it. Resetting here discards a ready transformer and
            # causes a second safetensors/GGUF load before the same generation.
            self._prepare_low_preload_for_generation(
                request,
                chunk_size=_chunk_size,
                chunk_overlap=_chunk_overlap,
                temporal_chunks=_temporal_chunks,
            )

        num_frames = int(getattr(request, "num_frames", 49))
        num_frames = max(5, num_frames if (num_frames - 1) % 4 == 0 else num_frames - (num_frames - 1) % 4)

        _prompt = str(getattr(request, "prompt", "") or "")
        _negative_prompt = str(getattr(request, "negative_prompt", "") or "")
        _guidance_scale = float(getattr(request, "guidance_scale", 1.0) or 1.0)
        from aiwf.infrastructure.wan.sliced_sampler import (
            estimate_temporal_chunk_count,
            latent_frame_count_for_output_frames,
        )

        latent_frame_count = latent_frame_count_for_output_frames(num_frames)
        transformer_chunks_per_forward = estimate_temporal_chunk_count(
            latent_frame_count,
            chunk_size=_chunk_size,
            overlap=_chunk_overlap,
            enabled=_temporal_chunks,
        )
        cfg_passes_per_step = 2 if _guidance_scale > 1.0 else 1
        transformer_forwards_per_step = transformer_chunks_per_forward * cfg_passes_per_step
        step_summary = (
            f"{total_steps} steps (high={high_steps}/low={low_steps})"
            if requires_dual
            else f"{total_steps} steps (single 5B)"
        )

        _video_status(
            f"Generating {num_frames} frames at {w}x{h} -- "
            f"{step_summary}, "
            f"guidance={_guidance_scale:g}, shift={_flow_shift:g}, seed={seed}."
        )
        _video_status(
            "Wan temporal plan: "
            f"{num_frames} output frames -> {latent_frame_count} latent frames; "
            f"temporal chunks={'on' if _temporal_chunks else 'off'} "
            f"(chunk={_chunk_size}, overlap={_chunk_overlap}); "
            f"~{transformer_forwards_per_step} transformer forward stream(s)/step."
        )
        if (
            str(getattr(request, "offload", "") or "").lower() == "sequential"
            and transformer_chunks_per_forward > 1
        ):
            _video_status(
                "Performance warning: sequential offload plus temporal chunks streams "
                f"the transformer {transformer_chunks_per_forward} times per conditional pass. "
                "Disable chunks or raise latent chunk size above the latent frame count for speed."
            )
        preprocess_seconds = max(0.0, time.perf_counter() - preprocess_started)

        cancelled = [False]
        denoise_started = 0.0
        progress_steps = 0
        last_step_elapsed = 0.0
        high_stage_elapsed = 0.0
        hook_timings: dict[str, float] = {}

        def _install_call_timer(obj: Any, attr: str, metric: str):
            original = getattr(obj, attr, None)
            if original is None or not callable(original):
                return None

            def _timed(*args, **kwargs):
                started = time.perf_counter()
                try:
                    return original(*args, **kwargs)
                finally:
                    hook_timings[metric] = hook_timings.get(metric, 0.0) + max(
                        0.0,
                        time.perf_counter() - started,
                    )

            try:
                setattr(obj, attr, _timed)
            except Exception:
                logger.debug("Could not install Wan timing hook for %s.%s", obj.__class__.__name__, attr, exc_info=True)
                return None

            def _restore() -> None:
                try:
                    setattr(obj, attr, original)
                except Exception:
                    logger.debug(
                        "Could not restore Wan timing hook for %s.%s",
                        obj.__class__.__name__,
                        attr,
                        exc_info=True,
                    )

            return _restore

        def _emit_generate_progress(
            step: int,
            total: int,
            steps_per_second: float | None = None,
            message: str | None = None,
        ) -> None:
            if on_progress is None:
                return
            try:
                on_progress(step, total, steps_per_second, message)
            except TypeError:
                try:
                    on_progress(step, total, steps_per_second)
                except TypeError:
                    try:
                        on_progress(step, total)
                    except Exception:
                        pass
                except Exception:
                    pass
            except Exception:
                pass

        def _trace_step_rate(step: int, steps_per_second: float | None, seconds_per_step: float | None) -> None:
            try:
                from aiwf.dev.diagnostics import trace_safe

                trace_safe(
                    "wan.step",
                    "Wan denoise progress",
                    step=int(step),
                    total_steps=int(total_steps),
                    steps_per_second=round(float(steps_per_second), 6)
                    if steps_per_second is not None
                    else None,
                    iterations_per_second=round(float(steps_per_second), 6)
                    if steps_per_second is not None
                    else None,
                    seconds_per_step=round(float(seconds_per_step), 3)
                    if seconds_per_step is not None
                    else None,
                )
            except Exception:
                logger.debug("Wan step trace failed.", exc_info=True)

        def _step_callback(pipe, i, t, callback_kwargs):
            nonlocal progress_steps, last_step_elapsed, high_stage_elapsed
            if should_cancel is not None and should_cancel():
                cancelled[0] = True
                pipe._interrupt = True
            try:
                current_step = min(total_steps, max(1, int(i) + 1))
            except Exception:
                current_step = min(total_steps, progress_steps + 1)
            progress_steps = max(progress_steps, current_step)
            elapsed_steps = max(0.0, time.perf_counter() - denoise_started) if denoise_started else 0.0
            last_step_elapsed = max(last_step_elapsed, elapsed_steps)
            if current_step <= high_steps:
                high_stage_elapsed = max(high_stage_elapsed, elapsed_steps)
            steps_per_second = current_step / elapsed_steps if elapsed_steps > 0 else None
            seconds_per_step = elapsed_steps / current_step if current_step > 0 and elapsed_steps > 0 else None
            _trace_step_rate(current_step, steps_per_second, seconds_per_step)
            _emit_generate_progress(current_step, total_steps, steps_per_second)
            return callback_kwargs

        def _phase_progress(message: str) -> None:
            current_step = max(0, min(total_steps, progress_steps))
            elapsed_steps = max(0.0, time.perf_counter() - denoise_started) if denoise_started else 0.0
            steps_per_second = current_step / elapsed_steps if current_step > 0 and elapsed_steps > 0 else None
            _emit_generate_progress(current_step, total_steps, steps_per_second, message)

        try:
            output_type = "latent" if _manual_vae_decode else _wan_output_type_for_pipe(pipe)
            call_kwargs = dict(
                image=image,
                prompt=_prompt,
                negative_prompt=_negative_prompt,
                height=h,
                width=w,
                num_frames=num_frames,
                num_inference_steps=total_steps,
                guidance_scale=_guidance_scale,
                generator=generator,
                output_type=output_type,
                callback_on_step_end=_step_callback,
            )
            # Wan 2.2's dual high/low pair supports a SEPARATE low-noise CFG via
            # `guidance_scale_2` (honored by both diffusers 0.38's WanImageToVideoPipeline
            # and AIWF's native denoise loop). The UI's low-noise guidance slider maps
            # here. It is only meaningful on the dual path (a boundary_ratio is set); the
            # single 5B path uses one `guidance_scale`. diffusers 0.38 has NO
            # `image_guidance_scale` for Wan (that arg is InstructPix2Pix-only), so we
            # never send it — passing the value through the real knob instead of dropping it.
            if requires_dual and _image_guidance_scale and float(_image_guidance_scale) > 1.0:
                call_kwargs["guidance_scale_2"] = float(_image_guidance_scale)
            restore_hooks = [
                hook
                for hook in (
                    _install_call_timer(pipe, "encode_prompt", "prompt_encode_seconds"),
                    _install_call_timer(pipe, "encode_image", "image_encode_seconds"),
                    _install_call_timer(pipe, "prepare_latents", "latent_prepare_seconds"),
                    _install_call_timer(getattr(pipe, "vae", None), "decode", "vae_decode_seconds"),
                    _install_call_timer(
                        getattr(pipe, "video_processor", None),
                        "postprocess_video",
                        "video_postprocess_seconds",
                    ),
                    _install_call_timer(pipe, "maybe_free_model_hooks", "offload_cleanup_seconds"),
                )
                if hook is not None
            ]
            denoise_started = time.perf_counter()
            try:
                if _use_native_denoise:
                    from aiwf.infrastructure.wan.native.denoise import run_native_wan_denoise

                    call_kwargs["aiwf_on_phase_progress"] = _phase_progress
                    _emit_generate_progress(
                        0,
                        total_steps,
                        None,
                        "Starting denoise; GGUF first step can take several minutes",
                    )
                    output = run_native_wan_denoise(pipe, **call_kwargs)
                else:
                    _emit_generate_progress(0, total_steps, None, "Starting denoise")
                    output = pipe(**call_kwargs)
            finally:
                pipeline_seconds = max(0.0, time.perf_counter() - denoise_started)
                for restore in reversed(restore_hooks):
                    restore()
        except Exception as exc:
            try:
                _video_status("Wan generation failed; unloading cached video pipeline to release VRAM.")
                self.unload()
            except Exception:
                logger.debug("Wan backend cleanup after failed generation failed.", exc_info=True)
            if cancelled[0]:
                raise WanUnavailable("Generation cancelled by user.") from exc
            raise

        if cancelled[0]:
            raise WanUnavailable("Generation cancelled by user.")

        postprocess_started = time.perf_counter()
        output_frames = output.frames if hasattr(output, "frames") else output
        if _manual_vae_decode:
            restore_hooks = [
                hook
                for hook in (
                    _install_call_timer(getattr(pipe, "vae", None), "decode", "vae_decode_seconds"),
                    _install_call_timer(
                        getattr(pipe, "video_processor", None),
                        "postprocess_video",
                        "video_postprocess_seconds",
                    ),
                )
                if hook is not None
            ]
            try:
                from aiwf.infrastructure.torch.wan_vram import decode_wan_video_latents

                _phase_progress("Decoding video frames")
                decoded = decode_wan_video_latents(
                    pipe,
                    output_frames,
                    chunk_frames=_vae_decode_chunk_frames,
                    output_type="pil",
                )
                _phase_progress("Post-processing video frames")
                frames = _flatten_wan_video_frames(decoded)
            finally:
                for restore in reversed(restore_hooks):
                    restore()
        else:
            frames = _frames_from_wan_pipeline_output(
                output_frames,
                pipe=pipe,
                decode_latents=lambda p, v, **kw: p.decode_latents(v) if hasattr(p, "decode_latents") else [],
            )
        postprocess_seconds = max(0.0, time.perf_counter() - postprocess_started)

        completed_steps = max(progress_steps, total_steps)
        denoise_seconds = last_step_elapsed if last_step_elapsed > 0 else pipeline_seconds
        if high_stage_elapsed <= 0 and denoise_seconds > 0:
            high_stage_elapsed = denoise_seconds * min(1.0, max(0.0, high_steps / max(1, total_steps)))
        low_stage_elapsed = max(0.0, denoise_seconds - high_stage_elapsed)
        pipeline_overhead_seconds = max(0.0, pipeline_seconds - denoise_seconds)
        steps_per_second = completed_steps / denoise_seconds if denoise_seconds > 0 else None
        fp8_metrics = _collect_fp8_linear_metrics(
            getattr(pipe, "transformer", None),
            getattr(pipe, "transformer_2", None),
            getattr(self, "_preloaded_low", None),
        )
        attention_metrics = _collect_attention_metrics(
            getattr(pipe, "transformer", None),
            getattr(pipe, "transformer_2", None),
            getattr(self, "_preloaded_low", None),
        )
        transition_metrics = self.cache.collect_transition_metrics()
        try:
            from aiwf.infrastructure.torch.wan_perf import (
                describe_wan_hardware_fingerprint,
                measure_wan_transfer_bandwidth,
            )

            hardware_fingerprint = describe_wan_hardware_fingerprint()
            transfer_probe = (
                measure_wan_transfer_bandwidth()
                if _env_flag("AIWF_WAN_TRANSFER_PROBE", default=False)
                else {}
            )
        except Exception as exc:
            hardware_fingerprint = {"error": f"{type(exc).__name__}: {exc}"}
            transfer_probe = {}
        metrics = {
            "step_count": int(completed_steps),
            "load_seconds": round(float(load_seconds), 3),
            "preprocess_seconds": round(float(preprocess_seconds), 3),
            "prompt_encode_seconds": round(float(hook_timings.get("prompt_encode_seconds", 0.0)), 3),
            "image_encode_seconds": round(float(hook_timings.get("image_encode_seconds", 0.0)), 3),
            "latent_prepare_seconds": round(float(hook_timings.get("latent_prepare_seconds", 0.0)), 3),
            "denoise_seconds": round(float(denoise_seconds), 3),
            "high_denoise_seconds": round(float(high_stage_elapsed), 3),
            "low_denoise_seconds": round(float(low_stage_elapsed), 3),
            "pipeline_seconds": round(float(pipeline_seconds), 3),
            "pipeline_overhead_seconds": round(float(pipeline_overhead_seconds), 3),
            "vae_decode_seconds": round(float(hook_timings.get("vae_decode_seconds", 0.0)), 3),
            "manual_vae_decode": bool(_manual_vae_decode),
            "vae_decode_chunk_frames": int(_vae_decode_chunk_frames if _manual_vae_decode else 0),
            "latent_frame_count": int(latent_frame_count),
            "temporal_chunks": bool(_temporal_chunks),
            "temporal_chunk_size": int(_chunk_size if _temporal_chunks else 0),
            "temporal_chunk_overlap": int(_chunk_overlap if _temporal_chunks else 0),
            "transformer_chunks_per_forward": int(transformer_chunks_per_forward),
            "transformer_forwards_per_step": int(transformer_forwards_per_step),
            "video_postprocess_seconds": round(float(hook_timings.get("video_postprocess_seconds", 0.0)), 3),
            "offload_cleanup_seconds": round(float(hook_timings.get("offload_cleanup_seconds", 0.0)), 3),
            "postprocess_seconds": round(float(postprocess_seconds), 3),
            "steps_per_second": round(float(steps_per_second), 6) if steps_per_second is not None else None,
            "cache_mode": str(self._cache_mode or "none"),
            "vram_reserve_enabled": bool(vram_budget.enabled),
            "vram_reserve_mb": int(vram_budget.requested_reserve_mb),
            "vram_limit_mb": int(vram_budget.limit_mb),
            "vram_total_mb": int(vram_budget.total_mb),
            "vram_limit_fraction": round(float(vram_budget.fraction), 6),
            **fp8_metrics,
            **attention_metrics,
            **transition_metrics,
            "hardware_fingerprint": hardware_fingerprint,
            "transfer_probe": transfer_probe,
        }
        return frames, h, w, metrics

    def _real_device(self, pipe) -> str:
        explicit = getattr(pipe, "_aiwf_execution_device", None)
        if explicit is not None:
            return str(explicit)
        try:
            import torch
            dev = next(pipe.transformer.parameters()).device
            if dev.type != "meta":
                return str(dev)
        except Exception:
            pass
        return "cuda"
