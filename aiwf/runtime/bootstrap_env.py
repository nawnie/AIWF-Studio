"""CUDA / host memory policy — must run before importing torch or diffusers."""
from __future__ import annotations

import os
import sys
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PerfLaunchFlags:
    """Comfy-parity throughput flags for Ada 16 GB cards."""

    async_offload: bool = True
    pinned_memory: bool = True
    cuda_malloc: bool = False


def _arg_value(args: list[str], flag: str) -> str | None:
    prefix = f"{flag}="
    for index, arg in enumerate(args):
        if arg.startswith(prefix):
            return arg[len(prefix):]
        if arg == flag and index + 1 < len(args):
            return args[index + 1]
    return None


def _saved_cuda_malloc(args: list[str]) -> bool | None:
    data_dir = Path(_arg_value(args, "--data-dir") or Path(__file__).resolve().parents[2])
    try:
        payload = json.loads((data_dir / "launch.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    value = payload.get("cuda_malloc")
    return value if isinstance(value, bool) else None


def parse_perf_argv(argv: list[str] | None = None) -> PerfLaunchFlags:
    args = argv if argv is not None else sys.argv[1:]
    if "--no-cuda-malloc" in args:
        cuda_malloc = False
    elif "--cuda-malloc" in args:
        cuda_malloc = True
    else:
        cuda_malloc = bool(_saved_cuda_malloc(args))
    return PerfLaunchFlags(
        async_offload="--no-async-offload" not in args,
        pinned_memory="--no-pinned-memory" not in args,
        cuda_malloc=cuda_malloc,
    )


def apply_runtime_env(*, cuda_malloc: bool = True) -> list[str]:
    """Set process env before any CUDA-linked import."""
    applied: list[str] = []

    if cuda_malloc:
        os.environ.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF",
            "backend:cudaMallocAsync",
        )
        applied.append("cudaMallocAsync")

    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    applied.append("CUDA_MODULE_LOADING=LAZY")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    applied.append("TOKENIZERS_PARALLELISM=false")

    # The optional `kernels` package pins huggingface-hub>=1.0 and breaks transformers.
    # Keep GGUF CUDA kernels opt-in only; never auto-enable from a detected install.
    os.environ.setdefault("DIFFUSERS_GGUF_CUDA_KERNELS", "false")

    return applied


def apply_from_argv(argv: list[str] | None = None) -> tuple[PerfLaunchFlags, list[str]]:
    perf = parse_perf_argv(argv)
    env = apply_runtime_env(cuda_malloc=perf.cuda_malloc)
    return perf, env
