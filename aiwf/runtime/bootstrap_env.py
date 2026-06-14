"""CUDA / host memory policy — must run before importing torch or diffusers."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class PerfLaunchFlags:
    """Comfy-parity throughput flags for Ada 16 GB cards."""

    async_offload: bool = True
    pinned_memory: bool = True
    cuda_malloc: bool = True


def parse_perf_argv(argv: list[str] | None = None) -> PerfLaunchFlags:
    args = argv if argv is not None else sys.argv[1:]
    return PerfLaunchFlags(
        async_offload="--no-async-offload" not in args,
        pinned_memory="--no-pinned-memory" not in args,
        cuda_malloc="--no-cuda-malloc" not in args,
    )


def apply_runtime_env(*, cuda_malloc: bool = True) -> list[str]:
    """Set process env before any CUDA-linked import."""
    applied: list[str] = []

    if cuda_malloc:
        os.environ.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF",
            "backend:cudaMallocAsync,max_split_size_mb:128",
        )
        applied.append("cudaMallocAsync")

    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    applied.append("CUDA_MODULE_LOADING=LAZY")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    applied.append("TOKENIZERS_PARALLELISM=false")

    return applied


def apply_from_argv(argv: list[str] | None = None) -> tuple[PerfLaunchFlags, list[str]]:
    perf = parse_perf_argv(argv)
    env = apply_runtime_env(cuda_malloc=perf.cuda_malloc)
    return perf, env