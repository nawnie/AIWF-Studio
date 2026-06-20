from __future__ import annotations

import logging
from contextlib import contextmanager

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _flag(flags, name: str) -> bool:
    return bool(getattr(flags, name, False))


def _attention_backend(flags) -> str:
    backend = getattr(flags, "attention_backend", None)
    if backend:
        normalized = str(backend).strip().lower().replace("-", "_")
    elif _flag(flags, "xformers"):
        normalized = "xformers"
    elif _flag(flags, "opt_sdp_attention") or _flag(flags, "opt_split_attention"):
        normalized = "sdpa"
    else:
        normalized = "sage_sdpa"
    if normalized in {"sage", "sageattention"}:
        normalized = "sage_sdpa"
    if normalized not in {"sage_sdpa", "sdpa", "xformers", "none"}:
        return "sage_sdpa"
    return normalized


def _sage_supported(q, k, v, attn_mask, dropout_p: float) -> bool:
    if attn_mask is not None or float(dropout_p or 0.0) != 0.0:
        return False
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return False
    if q.device.type != "cuda" or k.device.type != "cuda" or v.device.type != "cuda":
        return False
    return q.dtype in (torch.float16, torch.bfloat16) and k.dtype == q.dtype and v.dtype == q.dtype


@contextmanager
def attention_call_context(flags):
    """Apply per-call attention patches that must not leak outside generation."""
    if _attention_backend(flags) != "sage_sdpa":
        yield "none"
        return
    try:
        from sageattention import sageattn
    except Exception:
        yield "sdpa"
        return

    original = torch.nn.functional.scaled_dot_product_attention

    def _sage_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
        if not _sage_supported(query, key, value, attn_mask, dropout_p):
            return original(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                **kwargs,
            )
        try:
            return sageattn(query, key, value, is_causal=is_causal, tensor_layout="HND")
        except Exception:
            logger.debug("SageAttention image call failed; falling back to torch SDPA.", exc_info=True)
            return original(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                **kwargs,
            )

    torch.nn.functional.scaled_dot_product_attention = _sage_sdpa
    try:
        yield "sage_sdpa"
    finally:
        torch.nn.functional.scaled_dot_product_attention = original


def _has_conv2d(module) -> bool:
    try:
        return any(isinstance(child, nn.Conv2d) for child in module.modules())
    except Exception:
        return False


def _maybe_channels_last(module, *, label: str, flags) -> None:
    if module is None or not _flag(flags, "channels_last"):
        return
    if not _has_conv2d(module):
        logger.debug("%s channels_last skipped: no Conv2d modules", label)
        return
    try:
        module.to(memory_format=torch.channels_last)
        logger.info("%s memory format: channels_last", label)
    except Exception:
        logger.debug("%s channels_last tuning failed", label, exc_info=True)


def _maybe_compile_module(module, *, label: str, flags, compile_allowed: bool) -> object:
    if module is None or not _flag(flags, "torch_compile"):
        return module
    if not compile_allowed:
        logger.info("%s torch.compile skipped: CPU offload is active or expected.", label)
        return module
    if not hasattr(torch, "compile") or torch.compile is None:
        logger.warning("%s torch.compile skipped: torch.compile is unavailable.", label)
        return module
    try:
        compiled = torch.compile(module, mode="reduce-overhead", fullgraph=False)
        logger.info("%s torch.compile enabled (mode=reduce-overhead)", label)
        return compiled
    except Exception as exc:
        logger.warning("%s torch.compile failed (%s); using eager mode.", label, exc)
        return module


def apply_image_pipeline_optimizations(
    pipe,
    flags,
    *,
    compile_allowed: bool = True,
    include_unet: bool = True,
    include_vae: bool = True,
) -> None:
    """Apply flag-gated image-pipeline layout and compile optimizations."""
    if pipe is None:
        return
    try:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
    except Exception:
        logger.debug("cudnn tuning failed", exc_info=True)

    unet = getattr(pipe, "unet", None)
    vae = getattr(pipe, "vae", None)
    if include_unet:
        _maybe_channels_last(unet, label="UNet", flags=flags)
    if include_vae:
        _maybe_channels_last(vae, label="VAE", flags=flags)

    if include_unet:
        compiled_unet = _maybe_compile_module(
            unet,
            label="UNet",
            flags=flags,
            compile_allowed=compile_allowed,
        )
        if compiled_unet is not unet:
            pipe.unet = compiled_unet

    if include_vae and _flag(flags, "torch_compile") and vae is not None:
        if not compile_allowed:
            logger.info("VAE decode torch.compile skipped: CPU offload is active or expected.")
            return
        decode = getattr(vae, "decode", None)
        if callable(decode):
            compiled_decode = _maybe_compile_module(
                decode,
                label="VAE decode",
                flags=flags,
                compile_allowed=compile_allowed,
            )
            if compiled_decode is not decode:
                vae.decode = compiled_decode


def apply_attention_optimizations(pipe, flags, *, compile_allowed: bool = True) -> str:
    """Apply fastest available cross-attention optimization (Doggettx/xformers/SDP)."""
    if pipe is None:
        return "none"

    apply_image_pipeline_optimizations(pipe, flags, compile_allowed=compile_allowed)

    backend = _attention_backend(flags)
    if backend == "none":
        logger.info("Attention optimization: none (user selected)")
        return "none"

    if backend == "xformers":
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("Attention optimization: xformers")
            return "xformers"
        except Exception as exc:
            logger.warning("xformers unavailable (%s), trying fallback", exc)

    use_sdp = backend in {"sdpa", "sage_sdpa"} or _flag(flags, "opt_sdp_attention") or _flag(flags, "opt_split_attention")
    if use_sdp and hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        try:
            from diffusers.models.attention_processor import AttnProcessor2_0

            processor = AttnProcessor2_0()
            pipe.unet.set_attn_processor(processor)
            name = "sage_sdpa (SageAttention call patch + SDPA fallback)" if backend == "sage_sdpa" else "sdp-attention (split-attention equivalent)"
            logger.info("Attention optimization: %s", name)
            return "sage_sdpa" if backend == "sage_sdpa" else "sdp"
        except Exception as exc:
            logger.warning("SDP attention failed (%s)", exc)

    logger.info("Attention optimization: none (using default)")
    return "none"
