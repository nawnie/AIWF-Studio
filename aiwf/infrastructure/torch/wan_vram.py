"""VRAM tracing and memory-safe Wan video decode helpers (16 GB Ada cards)."""
from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Iterator

logger = logging.getLogger(__name__)

# RTX 4070 Ti SUPER — warn when PyTorch allocator crosses ~15.2 GB dedicated.
_DEFAULT_VRAM_WARN_GB = 15.2


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def vram_warn_threshold_gb() -> float:
    raw = os.environ.get("AIWF_WAN_VRAM_WARN_GB", "").strip()
    if not raw:
        return _DEFAULT_VRAM_WARN_GB
    try:
        return float(raw)
    except ValueError:
        return _DEFAULT_VRAM_WARN_GB


def reset_cuda_peak_stats() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
    except Exception:
        pass


def peak_vram_gb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024**3)
    except Exception:
        pass
    return None


def current_vram_gb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
    except Exception:
        pass
    return None


def log_step_vram_trace(
    *,
    step: int,
    total: int,
    stage: str,
    elapsed_s: float,
    status_print: Any | None = None,
) -> None:
    """Per-step VRAM trace — detects likely Windows shared-memory paging."""
    peak = peak_vram_gb()
    if peak is None:
        return
    threshold = vram_warn_threshold_gb()
    msg = (
        f"[AIWF Trace] step {step}/{total} ({stage}): {elapsed_s:.2f}s, "
        f"peak VRAM {peak:.2f} GB"
    )
    if peak > threshold:
        msg += (
            f" — CRITICAL: exceeds {threshold:.1f} GB; dedicated VRAM is likely full and "
            "Windows may be paging to shared system memory (~10x slower). "
            "Use Model offload (not Sequential), set NVIDIA 'Prefer No Sysmem Fallback', "
            "and keep resolution at 480p."
        )
        logger.warning(msg)
    else:
        logger.info(msg)
    if status_print:
        status_print(msg)


@contextmanager
def wan_inference_context() -> Iterator[None]:
    """Flash/mem-efficient SDPA around the denoise loop; math kernel off for VRAM."""
    try:
        import warnings

        import torch

        if not torch.cuda.is_available():
            yield
            return

        sdpa_kwargs = dict(enable_flash=True, enable_math=False, enable_mem_efficient=True)
        nn_attention = getattr(torch.nn, "attention", None)
        if nn_attention is not None and hasattr(nn_attention, "sdpa_kernel"):
            with nn_attention.sdpa_kernel(**sdpa_kwargs):
                yield
            return

        if hasattr(torch.backends.cuda, "sdp_kernel"):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*torch.backends.cuda.sdp_kernel.*deprecated.*",
                    category=FutureWarning,
                )
                with torch.backends.cuda.sdp_kernel(**sdpa_kwargs):
                    yield
            return
    except Exception:
        logger.debug("wan_inference_context sdpa kernel setup failed", exc_info=True)
    yield


def denormalize_wan_latents(vae, latents):
    """Apply Wan VAE latent mean/std (same as diffusers WanImageToVideoPipeline)."""
    import torch

    latents = latents.to(vae.dtype)
    z_dim = vae.config.z_dim
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = (
        1.0
        / torch.tensor(vae.config.latents_std)
        .view(1, z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    return latents / latents_std + latents_mean


def decode_wan_video_latents(
    pipe,
    latents,
    *,
    chunk_frames: int | None = None,
    output_type: str = "np",
):
    """Decode [B,C,F,H,W] latents in temporal chunks to avoid VAE VRAM spikes."""
    import gc

    import torch

    chunk_frames = chunk_frames or _env_int("AIWF_WAN_VAE_CHUNK_FRAMES", 4)
    vae = pipe.vae
    for method in ("enable_tiling", "enable_slicing"):
        try:
            getattr(vae, method)()
        except Exception:
            pass

    latents = denormalize_wan_latents(vae, latents)
    if latents.ndim != 5:
        raise ValueError(f"Expected 5D Wan latents [B,C,F,H,W], got shape {tuple(latents.shape)}")

    num_frames = int(latents.shape[2])
    if num_frames <= chunk_frames:
        with torch.no_grad():
            video = vae.decode(latents, return_dict=False)[0]
        return pipe.video_processor.postprocess_video(video, output_type=output_type)

    decoded_chunks: list[torch.Tensor] = []
    for start in range(0, num_frames, chunk_frames):
        end = min(start + chunk_frames, num_frames)
        chunk = latents[:, :, start:end, :, :]
        with torch.no_grad():
            decoded_chunks.append(vae.decode(chunk, return_dict=False)[0])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    video = torch.cat(decoded_chunks, dim=2)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pipe.video_processor.postprocess_video(video, output_type=output_type)


def profile_block(label: str, fn, *, status_print: Any | None = None):
    """Run a callable once and log wall time + peak VRAM."""
    reset_cuda_peak_stats()
    start = time.perf_counter()
    result = fn()
    if hasattr(__import__("torch").cuda, "is_available") and __import__("torch").cuda.is_available():
        try:
            __import__("torch").cuda.synchronize()
        except Exception:
            pass
    elapsed = time.perf_counter() - start
    peak = peak_vram_gb()
    msg = f"[AIWF Trace] {label}: {elapsed:.2f}s"
    if peak is not None:
        msg += f", peak VRAM {peak:.2f} GB"
        if peak > vram_warn_threshold_gb():
            msg += " (shared-memory paging risk)"
    logger.info(msg)
    if status_print:
        status_print(msg)
    return result