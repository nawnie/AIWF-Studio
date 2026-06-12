from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def apply_attention_optimizations(pipe, flags) -> str:
    """Apply fastest available cross-attention optimization (Doggettx/xformers/SDP)."""
    if pipe is None:
        return "none"

    try:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        unet = getattr(pipe, "unet", None)
        if unet is not None:
            unet.to(memory_format=torch.channels_last)
            logger.info("UNet memory format: channels_last (faster convolutions)")
    except Exception:
        logger.debug("channels_last / cudnn tuning failed", exc_info=True)

    if flags.xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("Attention optimization: xformers")
            return "xformers"
        except Exception as exc:
            logger.warning("xformers unavailable (%s), trying fallback", exc)

    use_sdp = flags.opt_sdp_attention or flags.opt_split_attention
    if use_sdp and hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        try:
            from diffusers.models.attention_processor import AttnProcessor2_0

            processor = AttnProcessor2_0()
            pipe.unet.set_attn_processor(processor)
            name = "sdp-attention (split-attention equivalent)"
            logger.info("Attention optimization: %s", name)
            return "sdp"
        except Exception as exc:
            logger.warning("SDP attention failed (%s)", exc)

    logger.info("Attention optimization: none (using default)")
    return "none"