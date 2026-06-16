"""
aiwf/infrastructure/quantization/torchao_quant.py

TorchAO quantization layer — int8 and fp8 weight compression.

Flag: AIWF_TORCHAO=1

What this does
--------------
torchao (https://github.com/pytorch/ao) provides post-training quantization
that compresses model weights from float16/bfloat16 down to int8 or fp8.
This saves VRAM and can improve throughput when memory bandwidth is the
bottleneck (true of most RTX consumer GPUs).

Expected gains (from torchao benchmarks, not yet verified locally):
* int8 weight-only: ~30-40% VRAM reduction, ~10-20% speed increase
* fp8 (RTX 40 series only): ~40-60% VRAM reduction, ~20-30% speed increase

API surface (torchao changed between 0.3 and 0.5)
--------------------------------------------------
This module detects the installed version at runtime and uses whichever
API is present.  Never hard-requires a specific version.

torch.compile interaction
--------------------------
AIWF_TORCH_COMPILE=1 enables torch.compile on top of quantization.
Both can be active simultaneously (compile sees the quantized graph).

Channels-last
-------------
AIWF_CHANNELS_LAST=1 converts Conv2D-heavy models (SD 1.x UNet) to
NHWC memory layout which is faster for cuDNN.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_TORCHAO_ENABLED      = os.environ.get("AIWF_TORCHAO", "0") == "1"
_TORCH_COMPILE_ENABLED = os.environ.get("AIWF_TORCH_COMPILE", "0") == "1"
_CHANNELS_LAST_ENABLED = os.environ.get("AIWF_CHANNELS_LAST", "0") == "1"
_FP8_ENABLED           = os.environ.get("AIWF_FP8", "0") == "1"


# ---------------------------------------------------------------------------
# TorchAO quantization
# ---------------------------------------------------------------------------

def _try_import_torchao() -> Any | None:
    try:
        import torchao
        return torchao
    except ImportError:
        return None


def _torchao_version() -> tuple[int, ...] | None:
    ao = _try_import_torchao()
    if ao is None:
        return None
    try:
        parts = getattr(ao, "__version__", "0.0.0").split(".")
        return tuple(int(p.split("+")[0]) for p in parts[:3])
    except Exception:
        return (0, 0, 0)


def apply_int8_weight_only(model: nn.Module) -> nn.Module:
    """Apply int8 weight-only quantization via torchao.

    Safe on all CUDA-capable GPUs.  Weights are stored as int8 and
    dequantized on-the-fly during matrix multiplications.  No calibration
    data required.

    Returns the model (in-place modification).
    """
    if not _TORCHAO_ENABLED:
        return model

    ao = _try_import_torchao()
    if ao is None:
        logger.warning("AIWF_TORCHAO=1 but torchao is not installed — skipping quantization")
        return model

    ver = _torchao_version()
    try:
        if ver and ver >= (0, 4, 0):
            # New API (0.4+): quantize_ with a quantizer object
            from torchao.quantization import quantize_, int8_weight_only
            quantize_(model, int8_weight_only())
        else:
            # Older API (0.3.x)
            from torchao.quantization import apply_dynamic_nf4_linear_quantization
            # 0.3 had different names — use what's available
            if hasattr(ao.quantization, "apply_weight_only_int8_quant"):
                ao.quantization.apply_weight_only_int8_quant(model)
            else:
                logger.warning("torchao %s: int8_weight_only API not found — skipping", ver)
                return model
        logger.info("TorchAO int8 weight-only quantization applied")
    except Exception as exc:
        logger.warning("TorchAO quantization failed (%s) — model unchanged", exc)

    return model


def apply_fp8_weight_only(model: nn.Module) -> nn.Module:
    """Apply fp8 weight-only quantization (RTX 40 series / Ada Lovelace only).

    Only RTX 40 series has native FP8 tensor cores.  On older cards this
    falls back to emulation and may be slower than fp16.  Detection happens
    at runtime via torch.cuda.get_device_capability().
    """
    if not (_TORCHAO_ENABLED and _FP8_ENABLED):
        return model

    if not torch.cuda.is_available():
        logger.warning("AIWF_FP8=1 but CUDA is not available — skipping fp8")
        return model

    cap = torch.cuda.get_device_capability()
    if cap < (8, 9):  # 8.9 = Ada Lovelace (RTX 4090 etc.)
        logger.warning(
            "AIWF_FP8=1 but GPU compute capability %d.%d < 8.9 (Ada Lovelace) — "
            "native fp8 not available, skipping", *cap
        )
        return model

    ao = _try_import_torchao()
    if ao is None:
        logger.warning("AIWF_FP8=1 but torchao not installed — skipping")
        return model

    try:
        from torchao.quantization import quantize_, fp8_weight_only
        quantize_(model, fp8_weight_only())
        logger.info("TorchAO fp8 weight-only quantization applied (Ada Lovelace path)")
    except (ImportError, AttributeError) as exc:
        logger.warning("TorchAO fp8 API not available in installed version (%s)", exc)

    return model


# ---------------------------------------------------------------------------
# torch.compile
# ---------------------------------------------------------------------------

def maybe_torch_compile(model: nn.Module, fullgraph: bool = False) -> nn.Module:
    """Apply torch.compile if AIWF_TORCH_COMPILE=1 and PyTorch ≥ 2.0.

    Mode "reduce-overhead" minimises Python overhead between kernels.
    ``fullgraph=False`` tolerates graph breaks (safer for diffusers models).
    """
    if not _TORCH_COMPILE_ENABLED:
        return model

    if not hasattr(torch, "compile"):
        logger.warning("AIWF_TORCH_COMPILE=1 but torch.compile not available (requires PyTorch ≥ 2.0)")
        return model

    try:
        compiled = torch.compile(model, mode="reduce-overhead", fullgraph=fullgraph)
        logger.info("torch.compile applied (mode=reduce-overhead, fullgraph=%s)", fullgraph)
        return compiled  # type: ignore[return-value]
    except Exception as exc:
        logger.warning("torch.compile failed (%s) — using eager mode", exc)
        return model


# ---------------------------------------------------------------------------
# Channels-last memory format
# ---------------------------------------------------------------------------

def maybe_channels_last(model: nn.Module) -> nn.Module:
    """Convert model to channels-last (NHWC) memory format if flagged.

    Only beneficial for Conv2D-heavy architectures (SD 1.x UNet).
    Skip for pure transformer models — they don't have 2D convolutions.
    """
    if not _CHANNELS_LAST_ENABLED:
        return model

    has_conv2d = any(isinstance(m, nn.Conv2d) for m in model.modules())
    if not has_conv2d:
        logger.debug("AIWF_CHANNELS_LAST: no Conv2d found, skipping channels_last")
        return model

    try:
        model.to(memory_format=torch.channels_last)
        logger.info("Model converted to channels_last memory format")
    except Exception as exc:
        logger.warning("channels_last conversion failed (%s)", exc)

    return model


# ---------------------------------------------------------------------------
# Convenience: apply all active optimizations in recommended order
# ---------------------------------------------------------------------------

def apply_all_optimizations(model: nn.Module, fullgraph: bool = False) -> nn.Module:
    """Apply all flag-gated optimizations in the correct order.

    Order matters:
    1. channels_last  — must be before quantization (layout change)
    2. int8 / fp8     — quantize weights
    3. torch.compile  — compile the (possibly quantized) graph
    """
    model = maybe_channels_last(model)
    if _FP8_ENABLED:
        model = apply_fp8_weight_only(model)
    else:
        model = apply_int8_weight_only(model)
    model = maybe_torch_compile(model, fullgraph=fullgraph)
    return model
