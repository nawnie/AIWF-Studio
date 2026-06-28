from __future__ import annotations

import logging
import warnings
from contextlib import contextmanager

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_CUDA_ATTN_BOOTSTRAPPED = False


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


def _align_sdpa_dtypes(q, k, v):
    if not (q.is_floating_point() and k.is_floating_point() and v.is_floating_point()):
        return q, k, v, False
    if q.dtype == k.dtype == v.dtype:
        return q, k, v, False
    target_dtype = v.dtype if v.dtype in (torch.float16, torch.bfloat16) else q.dtype
    if target_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return q, k, v, False
    return (
        q.to(target_dtype) if q.dtype != target_dtype else q,
        k.to(target_dtype) if k.dtype != target_dtype else k,
        v.to(target_dtype) if v.dtype != target_dtype else v,
        True,
    )


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
        query, key, value, aligned = _align_sdpa_dtypes(query, key, value)
        if aligned:
            return original(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                **kwargs,
            )
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
            return sageattn(query, key, value, is_causal=is_causal, tensor_layout="NHD")
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


_NATIVE_DIT_TRANSFORMER_NAMES = frozenset(
    {
        "FluxTransformer2DModel",
        "Flux2Transformer2DModel",
        "ZImageTransformer2DModel",
    }
)


def _ensure_cuda_attention_bootstrapped() -> list[str]:
    global _CUDA_ATTN_BOOTSTRAPPED
    if _CUDA_ATTN_BOOTSTRAPPED:
        return []
    from aiwf.infrastructure.torch.wan_perf import bootstrap_wan_cuda_settings

    active = bootstrap_wan_cuda_settings()
    _CUDA_ATTN_BOOTSTRAPPED = True
    if active:
        logger.info("CUDA attention bootstrap: %s", ", ".join(active))
    return active


def _sageattention_importable() -> bool:
    try:
        import sageattention  # noqa: F401

        return True
    except Exception:
        return False


def _sageattention_usable_in_diffusers() -> bool:
    try:
        from diffusers.utils import is_sageattention_available, is_sageattention_version

        return is_sageattention_available() and is_sageattention_version(">=", "2.1.1")
    except Exception:
        return False


def _flashattention_usable_in_diffusers() -> bool:
    try:
        from diffusers.models.attention_dispatch import _CAN_USE_FLASH_ATTN, _CAN_USE_FLASH_ATTN_3

        return bool(_CAN_USE_FLASH_ATTN or _CAN_USE_FLASH_ATTN_3)
    except Exception:
        return False


def _xformers_usable_in_diffusers() -> bool:
    try:
        from diffusers.models.attention_dispatch import _CAN_USE_XFORMERS_ATTN

        return bool(_CAN_USE_XFORMERS_ATTN)
    except Exception:
        return False


def resolve_best_diffusers_attention_backend(flags) -> str:
    """Pick the fastest Diffusers `set_attention_backend` value for this machine."""
    preference = _attention_backend(flags)
    if preference == "none":
        return "native"

    if preference == "xformers" and _xformers_usable_in_diffusers():
        return "xformers"

    if _flashattention_usable_in_diffusers():
        return "flash"

    if _sageattention_usable_in_diffusers() and preference in {"sage_sdpa", "sdpa"}:
        return "sage"

    # Ada 4070 Ti path: PyTorch flash SDPA via Diffusers native dispatch.
    return "_native_flash"


def describe_best_attention_stack(flags) -> str:
    preference = _attention_backend(flags)
    parts = [f"preference={preference}"]
    if _sageattention_usable_in_diffusers():
        parts.append("sageattention=2.x")
    elif _sageattention_importable():
        parts.append("sageattention=sdpa-patch")
    if _flashattention_usable_in_diffusers():
        parts.append("flash-attn=ok")
    if _xformers_usable_in_diffusers():
        parts.append("xformers=ok")
    parts.append(f"dit-backend={resolve_best_diffusers_attention_backend(flags)}")
    return ", ".join(parts)


def _uses_builtin_dit_attention(denoiser) -> bool:
    """True when the denoiser ships its own attention processors (not AttnProcessor2_0)."""
    if denoiser is None:
        return False
    if type(denoiser).__name__ in _NATIVE_DIT_TRANSFORMER_NAMES:
        return True
    for import_path, class_name in (
        ("diffusers.models.transformers.transformer_flux2", "Flux2Attention"),
        ("diffusers.models.transformers.transformer_flux", "FluxAttention"),
    ):
        try:
            module = __import__(import_path, fromlist=[class_name])
            attention_cls = getattr(module, class_name)
        except Exception:
            continue
        if any(isinstance(child, attention_cls) for child in denoiser.modules()):
            return True
    return False


def ensure_flux2_attention_processors(transformer, name: str = "transformer") -> None:
    """Restore Flux2-native processors if a generic AttnProcessor2_0 swap broke them."""
    if transformer is None:
        return
    try:
        from diffusers.models.transformers.transformer_flux2 import (
            Flux2Attention,
            Flux2AttnProcessor,
            Flux2ParallelSelfAttention,
            Flux2ParallelSelfAttnProcessor,
        )
    except Exception:
        return

    fixed = 0
    for module in transformer.modules():
        if isinstance(module, Flux2Attention):
            processor = getattr(module, "processor", None)
            if processor is None or processor.__class__.__name__ == "AttnProcessor2_0":
                module.set_processor(Flux2AttnProcessor())
                fixed += 1
        elif isinstance(module, Flux2ParallelSelfAttention):
            processor = getattr(module, "processor", None)
            if processor is None or processor.__class__.__name__ == "AttnProcessor2_0":
                module.set_processor(Flux2ParallelSelfAttnProcessor())
                fixed += 1
    if fixed:
        logger.info("Restored %d Flux2 attention processor(s) on %s", fixed, name)


def ensure_flux_attention_processors(transformer, name: str = "transformer") -> None:
    """Restore Flux-native processors if a generic AttnProcessor2_0 swap broke them."""
    if transformer is None:
        return
    try:
        from diffusers.models.transformers.transformer_flux import FluxAttention, FluxAttnProcessor
    except Exception:
        return

    fixed = 0
    for module in transformer.modules():
        if not isinstance(module, FluxAttention):
            continue
        processor = getattr(module, "processor", None)
        if processor is None or processor.__class__.__name__ == "AttnProcessor2_0":
            module.set_processor(FluxAttnProcessor())
            fixed += 1
    if fixed:
        logger.info("Restored %d Flux attention processor(s) on %s", fixed, name)


def apply_builtin_dit_attention_backend(denoiser, flags) -> str:
    """Enable the best safe attention path for Flux / Flux2 / Z-Image transformers."""
    ensure_flux2_attention_processors(denoiser)
    ensure_flux_attention_processors(denoiser)

    if not hasattr(denoiser, "set_attention_backend"):
        if _sageattention_importable() and _attention_backend(flags) == "sage_sdpa":
            return "sage_sdpa"
        return "sdp"

    backend = getattr(denoiser, "_aiwf_force_diffusers_attention_backend", None) or resolve_best_diffusers_attention_backend(flags)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Attention backends are an experimental feature.*")
            denoiser.set_attention_backend(backend)
        logger.info(
            "DiT attention backend: %s on %s",
            backend,
            type(denoiser).__name__,
        )
        return backend
    except Exception as exc:
        logger.warning(
            "Could not set DiT attention backend %s on %s (%s); using torch SDPA",
            backend,
            type(denoiser).__name__,
            exc,
        )
        return "sdp"


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
    transformer = getattr(pipe, "transformer", None)
    vae = getattr(pipe, "vae", None)
    if include_unet:
        _maybe_channels_last(unet, label="UNet", flags=flags)
        _maybe_channels_last(transformer, label="Transformer", flags=flags)
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
        compiled_transformer = _maybe_compile_module(
            transformer,
            label="Transformer",
            flags=flags,
            compile_allowed=compile_allowed,
        )
        if compiled_transformer is not transformer:
            pipe.transformer = compiled_transformer

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

    _ensure_cuda_attention_bootstrapped()
    apply_image_pipeline_optimizations(pipe, flags, compile_allowed=compile_allowed)

    backend = _attention_backend(flags)
    if backend == "none":
        logger.info("Attention optimization: none (user selected)")
        return "none"

    denoiser = getattr(pipe, "unet", None) or getattr(pipe, "transformer", None)
    if _uses_builtin_dit_attention(denoiser):
        dit_backend = apply_builtin_dit_attention_backend(denoiser, flags)
        if dit_backend == "sage_sdpa":
            logger.info(
                "Attention optimization: sage_sdpa + torch flash SDPA (%s)",
                type(denoiser).__name__,
            )
            return "sage_sdpa"
        logger.info(
            "Attention optimization: %s (%s uses built-in DiT processors)",
            dit_backend,
            type(denoiser).__name__,
        )
        if dit_backend == "_native_flash":
            return "sdp-flash"
        return dit_backend

    if backend == "xformers":
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("Attention optimization: xformers")
            return "xformers"
        except Exception as exc:
            logger.warning("xformers unavailable (%s), trying fallback", exc)

    use_sdp = backend in {"sdpa", "sage_sdpa"} or _flag(flags, "opt_sdp_attention") or _flag(flags, "opt_split_attention")
    if use_sdp and hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        if denoiser is not None and not hasattr(denoiser, "set_attn_processor"):
            # Some newer DiT transformers (e.g. Z-Image's ZImageTransformer2DModel,
            # Flux2 Klein) don't expose the AttnProcessor swap hook at all - they call
            # torch's scaled_dot_product_attention directly inside their own block
            # code. That means SDPA is already in effect; there is nothing to patch.
            # Previously this case fell through to the except-branch below, logged a
            # warning, and silently left the model on its default (often eager, much
            # more VRAM-hungry) attention path - a likely contributor to the OOM/crash
            # seen on Z-Image. Detect it up front and report success instead.
            logger.info(
                "Attention optimization: native sdpa (%s has no AttnProcessor hook; "
                "it already calls scaled_dot_product_attention internally)",
                type(denoiser).__name__,
            )
            return "sdp"
        try:
            from diffusers.models.attention_processor import AttnProcessor2_0

            processor = AttnProcessor2_0()
            denoiser.set_attn_processor(processor)
            name = "sage_sdpa (SageAttention call patch + SDPA fallback)" if backend == "sage_sdpa" else "sdp-attention (split-attention equivalent)"
            logger.info("Attention optimization: %s", name)
            return "sage_sdpa" if backend == "sage_sdpa" else "sdp"
        except Exception as exc:
            logger.warning(
                "SDP attention processor swap failed (%s); falling back to eager "
                "attention. This uses significantly more VRAM and may cause OOM on "
                "large models - consider a smaller quant if you see crashes.",
                exc,
            )

    logger.info("Attention optimization: none (using default)")
    return "none"
