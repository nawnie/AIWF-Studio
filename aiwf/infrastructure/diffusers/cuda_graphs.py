"""
aiwf/infrastructure/diffusers/cuda_graphs.py

CUDA Graph capture and replay for the denoising UNet/transformer forward pass.

Flag: AIWF_CUDA_GRAPHS=1

What CUDA Graphs do
-------------------
A CUDA Graph records a sequence of GPU kernel launches (the denoising step)
and replays them on subsequent calls without re-issuing the kernel launch
overhead from the CPU.  This saves the round-trip for every operator kernel
dispatch — useful for models with many small operators.

Expected gain: 5–15% on RTX 30/40 series at fixed resolution.

Limitations
-----------
* Static shapes only.  The graph captures with a specific (batch, channels,
  H, W) shape.  Changing resolution breaks the graph — it is discarded and
  a new graph is captured.
* First call captures (slow — one graph-capture forward pass).
* Subsequent calls replay (fast).
* Only the UNet/transformer forward is captured; VAE encode/decode is not.
* Incompatible with hooks that modify tensors between operator calls.

Usage
-----
Wrap the callable that performs a single denoising forward:

    graph = CUDAGraphDenoiser(unet_forward_fn)
    for step in sampler:
        denoised = graph(latent, timestep, encoder_hidden_states)

The denoiser is called with keyword arguments matching the UNet signature.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)

_ENABLED = os.environ.get("AIWF_CUDA_GRAPHS", "0") == "1"


def _cuda_graphs_available() -> bool:
    return (
        _ENABLED
        and torch.cuda.is_available()
        and hasattr(torch.cuda, "CUDAGraph")
    )


class CUDAGraphDenoiser:
    """Wraps a UNet/transformer forward callable with CUDA Graph capture/replay.

    The graph is captured on the first call and replayed on subsequent calls
    when the input shapes match.  If shapes change, the graph is discarded
    and a new capture is triggered.

    Parameters
    ----------
    forward_fn:
        Callable ``(**kwargs) → tensor``.  Must be the raw model forward pass
        that runs on GPU — no Python control flow that branches on tensor values.
    warmup_steps:
        Number of warmup calls before graph capture.  Required to initialise
        cuDNN / cuBLAS kernels and avoid capturing the first-run overhead.
    """

    def __init__(self, forward_fn: Callable[..., torch.Tensor], warmup_steps: int = 3) -> None:
        self._forward_fn = forward_fn
        self._warmup_steps = warmup_steps
        self._graph: torch.cuda.CUDAGraph | None = None
        self._static_inputs: dict[str, torch.Tensor] = {}
        self._static_output: torch.Tensor | None = None
        self._captured_shapes: dict[str, tuple] = {}
        self._call_count = 0

    def _shapes_match(self, kwargs: dict[str, Any]) -> bool:
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                if k not in self._captured_shapes:
                    return False
                if tuple(v.shape) != self._captured_shapes[k]:
                    return False
        return True

    def _capture(self, **kwargs: Any) -> None:
        """Warm up then capture the CUDA graph."""
        logger.info("CUDA Graphs: warming up (%d steps)…", self._warmup_steps)
        # Warm-up passes (not captured)
        with torch.cuda.stream(torch.cuda.Stream()):
            for _ in range(self._warmup_steps):
                _ = self._forward_fn(**kwargs)

        # Allocate static input copies (graph captures tensor *addresses*)
        self._static_inputs = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                self._static_inputs[k] = v.clone()
            else:
                self._static_inputs[k] = v

        logger.info("CUDA Graphs: capturing…")
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._static_output = self._forward_fn(**self._static_inputs)

        self._captured_shapes = {
            k: tuple(v.shape) for k, v in kwargs.items() if isinstance(v, torch.Tensor)
        }
        logger.info("CUDA Graphs: capture complete")

    def __call__(self, **kwargs: Any) -> torch.Tensor:
        if not _cuda_graphs_available():
            return self._forward_fn(**kwargs)

        # Invalidate graph if shapes changed
        if self._graph is not None and not self._shapes_match(kwargs):
            logger.debug("CUDA Graphs: shape change — discarding graph")
            self._graph = None
            self._static_inputs = {}
            self._static_output = None
            self._captured_shapes = {}
            self._call_count = 0

        # Capture on first call
        if self._graph is None:
            self._capture(**kwargs)

        # Replay: copy live tensors into the static buffers and replay
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor) and k in self._static_inputs:
                self._static_inputs[k].copy_(v)

        self._graph.replay()
        return self._static_output.clone()  # type: ignore[return-value]

    def reset(self) -> None:
        """Discard the captured graph (e.g. after a checkpoint change)."""
        self._graph = None
        self._static_inputs = {}
        self._static_output = None
        self._captured_shapes = {}
        self._call_count = 0
        logger.debug("CUDA Graphs: graph reset")


def maybe_wrap_with_cuda_graph(model: torch.nn.Module, warmup_steps: int = 3) -> "CUDAGraphDenoiser | torch.nn.Module":
    """Wrap *model* in a CUDAGraphDenoiser if CUDA Graphs are enabled.

    Returns the model unchanged if the flag is not set or CUDA is unavailable.

    Usage
    -----
        unet = maybe_wrap_with_cuda_graph(unet)
        # Then call unet(sample=latent, timestep=t, ...)
    """
    if not _cuda_graphs_available():
        if _ENABLED and not torch.cuda.is_available():
            logger.warning("AIWF_CUDA_GRAPHS=1 but CUDA is not available — disabled")
        return model

    logger.info("CUDA Graphs enabled — wrapping model forward pass")
    return CUDAGraphDenoiser(model, warmup_steps=warmup_steps)
