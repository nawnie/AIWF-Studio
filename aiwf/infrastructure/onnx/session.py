"""
aiwf/infrastructure/onnx/session.py

ONNX Runtime session management for AIWF inference engines.

Handles execution provider selection (CUDA EP > DirectML EP > CPU EP) and
session creation with optimal provider options.  No diffusers, no torch
required at module level.

ONNX Runtime execution providers
---------------------------------
* CUDAExecutionProvider   — NVIDIA GPU via CUDA; fastest on RTX cards
* DmlExecutionProvider    — DirectML; works on any DX12 GPU (AMD, Intel, NVIDIA)
* CPUExecutionProvider    — fallback; always available

Provider availability depends on which onnxruntime package is installed:
* onnxruntime-gpu         → CUDA EP available
* onnxruntime-directml    → DirectML EP available
* onnxruntime             → CPU EP only

Detection is done at runtime; the session is created with the best available
provider that is actually present in the installed package.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

ProviderPreference = Literal["cuda", "directml", "cpu", "auto"]


def _ort():
    """Lazy import onnxruntime so the module is importable without it installed."""
    try:
        import onnxruntime as ort
        return ort
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is not installed.  Install one of:\n"
            "  pip install onnxruntime-gpu        # CUDA EP\n"
            "  pip install onnxruntime-directml   # DirectML EP (Windows)\n"
            "  pip install onnxruntime            # CPU only"
        ) from exc


def get_available_providers() -> list[str]:
    """Return the list of execution providers available in the installed ORT."""
    return _ort().get_available_providers()


def _build_cuda_options(device_id: int = 0) -> tuple[str, dict]:
    return (
        "CUDAExecutionProvider",
        {
            "device_id": device_id,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "gpu_mem_limit": 8 * 1024 * 1024 * 1024,  # 8 GB arena cap
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": True,
        },
    )


def _build_dml_options(device_id: int = 0) -> tuple[str, dict]:
    return (
        "DmlExecutionProvider",
        {"device_id": device_id},
    )


def select_provider(
    preference: ProviderPreference = "auto",
    device_id: int = 0,
) -> list[tuple[str, dict] | str]:
    """Return the provider list to pass to ``onnxruntime.InferenceSession``.

    "auto" tries CUDA → DirectML → CPU in order.
    """
    available = set(get_available_providers())

    if preference == "auto":
        if "CUDAExecutionProvider" in available:
            preference = "cuda"
        elif "DmlExecutionProvider" in available:
            preference = "directml"
        else:
            preference = "cpu"

    if preference == "cuda":
        if "CUDAExecutionProvider" not in available:
            logger.warning("CUDA EP not available — falling back to CPU")
            return ["CPUExecutionProvider"]
        return [_build_cuda_options(device_id), "CPUExecutionProvider"]

    if preference == "directml":
        if "DmlExecutionProvider" not in available:
            logger.warning("DirectML EP not available — falling back to CPU")
            return ["CPUExecutionProvider"]
        return [_build_dml_options(device_id), "CPUExecutionProvider"]

    return ["CPUExecutionProvider"]


def load_session(
    model_path: Path,
    preference: ProviderPreference = "auto",
    device_id: int = 0,
    inter_op_threads: int = 1,
    intra_op_threads: int = 0,
) -> "onnxruntime.InferenceSession":  # noqa: F821
    """Load an ONNX model file into an optimised InferenceSession.

    Parameters
    ----------
    model_path:
        Path to a ``.onnx`` file.
    preference:
        Execution provider preference.  "auto" picks the best available.
    device_id:
        GPU device index (for CUDA/DML).
    inter_op_threads:
        Number of threads between operators.  1 is optimal for GPU paths.
    intra_op_threads:
        Number of threads within operators.  0 = use all logical cores.
    """
    ort = _ort()
    providers = select_provider(preference, device_id)
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = inter_op_threads
    opts.intra_op_num_threads = intra_op_threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_mem_pattern = True

    provider_names = [p if isinstance(p, str) else p[0] for p in providers]
    logger.info("Loading ONNX model %s with providers %s", model_path.name, provider_names)

    session = ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)
    return session
