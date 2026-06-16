"""Runtime performance hooks for Wan video transformers (RTX / Comfy-parity path)."""
from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from importlib import import_module
from importlib.util import find_spec
from typing import Iterable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WanAccelerationCapability:
    name: str
    available: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _module_importable(name: str) -> bool:
    if find_spec(name) is None:
        return False
    try:
        import_module(name)
        return True
    except Exception:
        logger.debug("Optional Wan accelerator module is present but not importable: %s", name, exc_info=True)
        return False


def bootstrap_wan_cuda_settings() -> list[str]:
    """Global CUDA knobs safe to call once before Wan pipeline work."""
    active: list[str] = []
    try:
        import torch

        if not torch.cuda.is_available():
            return active
        torch.backends.cudnn.benchmark = True
        active.append("cudnn.benchmark")
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
            active.append("sdp.flash")
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            active.append("sdp.mem_efficient")
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        logger.debug("Wan CUDA bootstrap failed", exc_info=True)
    return active


def _try_sage_attention() -> str | None:
    """ComfyUI parity: --use-sage-attention patches SDPA with sageattention when installed."""
    if not _env_flag("AIWF_WAN_SAGE_ATTENTION") and not _env_flag("AIWF_USE_SAGE_ATTENTION"):
        # Auto-enable when the package is present (user already installed it for Comfy).
        try:
            import sageattention  # noqa: F401
        except ImportError:
            return None
    else:
        try:
            import sageattention  # noqa: F401
        except ImportError:
            logger.warning(
                "AIWF_WAN_SAGE_ATTENTION is set but `sageattention` is not installed. "
                "Install with: pip install sageattention"
            )
            return None
    try:
        from sageattention import sageattn

        import torch

        _orig = torch.nn.functional.scaled_dot_product_attention

        def _sage_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
            # Wan 3D tensors are (B, seq, heads, dim); sageattn expects (B, heads, seq, dim).
            if query.ndim == 4 and query.shape[2] <= 64 and query.shape[1] > query.shape[2]:
                q = query.transpose(1, 2)
                k = key.transpose(1, 2)
                v = value.transpose(1, 2)
                out = sageattn(q, k, v, is_causal=is_causal, tensor_layout="HND")
                return out.transpose(1, 2)
            return _orig(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                **kwargs,
            )

        torch.nn.functional.scaled_dot_product_attention = _sage_sdpa
        return "sageattention"
    except Exception as exc:
        logger.warning("sageattention hook failed (%s); using torch SDPA", exc)
        return None


def _flash_attn_dispatch_available() -> bool:
    """True only if diffusers' FLASH dispatch can actually call into flash-attn.

    diffusers sets ``flash_attn_func = None`` when ``_CAN_USE_FLASH_ATTN`` is False
    (flash-attn missing or too old). Selecting the FLASH backend in that state makes
    every attention call do ``None(...)`` -> ``TypeError: 'NoneType' object is not
    callable`` at the first denoising step. We must gate on the real symbol.
    """
    try:
        from diffusers.models import attention_dispatch as _ad
    except Exception:
        return False
    can_use = getattr(_ad, "_CAN_USE_FLASH_ATTN", None)
    func_ok = getattr(_ad, "flash_attn_func", None) is not None
    if can_use is None:
        # Symbol name differs across diffusers versions — trust the bound function.
        return func_ok
    return bool(can_use) and func_ok


def _set_wan_flash_backend(transformer) -> str | None:
    """Prefer diffusers FLASH dispatch ONLY when flash-attn is installed AND callable.

    When it is not, we leave the default SDPA dispatch in place (torch flash/mem-efficient
    SDPA, optionally patched by sageattention) — which needs no flash-attn package.
    """
    if not _flash_attn_dispatch_available():
        logger.debug("Wan flash backend skipped: flash-attn unavailable; using SDPA/sage instead.")
        return None
    try:
        from diffusers.models.attention_dispatch import AttentionBackendName
        from diffusers.models.transformers.transformer_wan import WanAttention

        backend = AttentionBackendName.FLASH
        count = 0
        for module in transformer.modules():
            if isinstance(module, WanAttention) and hasattr(module, "set_attention_backend"):
                try:
                    module.set_attention_backend(backend)
                    count += 1
                except Exception:
                    pass
        if count:
            return f"diffusers.{backend.value}({count})"
    except Exception:
        logger.debug("Wan flash backend setup skipped", exc_info=True)
    return None


def _sage_dispatch_available() -> bool:
    """True only if diffusers' SAGE dispatch can actually call into sageattention.

    diffusers gates this on ``_CAN_USE_SAGE_ATTN`` (sageattention installed AND
    version >= 2.1.1) and sets ``sageattn = None`` otherwise. Selecting the SAGE
    backend without that guard would call ``None(...)`` at the first step.
    """
    try:
        from diffusers.models import attention_dispatch as _ad
    except Exception:
        return False
    can_use = getattr(_ad, "_CAN_USE_SAGE_ATTN", None)
    func_ok = getattr(_ad, "sageattn", None) is not None
    if can_use is None:
        return func_ok
    return bool(can_use) and func_ok


def _set_wan_sage_backend(transformer) -> str | None:
    """Select diffusers' native SAGE attention backend per WanAttention module.

    This is the ComfyUI ``--use-sage-attention`` parity path: diffusers' own
    ``_sage_attention`` handles the Wan tensor layout correctly, so it is preferred
    over the global SDPA monkeypatch (which also leaks into non-Wan models). Gated
    on real availability so it can never crash with a None callable.
    """
    if not _sage_dispatch_available():
        return None
    try:
        from diffusers.models.attention_dispatch import AttentionBackendName
        from diffusers.models.transformers.transformer_wan import WanAttention

        backend = AttentionBackendName.SAGE
        count = 0
        for module in transformer.modules():
            if isinstance(module, WanAttention) and hasattr(module, "set_attention_backend"):
                try:
                    module.set_attention_backend(backend)
                    count += 1
                except Exception:
                    pass
        if count:
            return f"diffusers.{backend.value}({count})"
    except Exception:
        logger.debug("Wan sage backend setup skipped", exc_info=True)
    return None


def describe_wan_acceleration_capabilities() -> dict[str, dict[str, object]]:
    """Return JSON-friendly Wan accelerator availability for diagnostics/benchmarks."""
    capabilities = [
        WanAccelerationCapability(
            name="diffusers_sage",
            available=_sage_dispatch_available(),
            detail="Diffusers per-module SAGE attention backend.",
        ),
        WanAccelerationCapability(
            name="diffusers_flash",
            available=_flash_attn_dispatch_available(),
            detail="Diffusers per-module FLASH attention backend.",
        ),
        WanAccelerationCapability(
            name="sageattention_fallback",
            available=_module_importable("sageattention"),
            detail="AIWF fallback hook that patches torch SDPA for Wan tensors.",
        ),
        WanAccelerationCapability(
            name="gguf_runtime",
            available=_module_importable("gguf") and os.environ.get("AIWF_WAN_GGUF_RUNTIME", "").strip().lower() not in {"0", "false", "no", "off"},
            detail="AIWF mmap + on-the-fly dequant GGUF transformer runtime.",
        ),
        WanAccelerationCapability(
            name="gguf_cuda_kernels",
            available=_module_importable("kernels") and os.environ.get("DIFFUSERS_GGUF_CUDA_KERNELS", "").strip().lower() in {"1", "true", "yes", "on"},
            detail="Diffusers GGUF optimized CUDA kernels package and env flag.",
        ),
        WanAccelerationCapability(
            name="torchao",
            available=_module_importable("torchao"),
            detail="Optional TorchAO quantization package.",
        ),
    ]
    return {capability.name: capability.to_dict() for capability in capabilities}


def apply_wan_transformer_optimizations(transformer, *, name: str = "transformer") -> list[str]:
    """Apply the fastest attention/conv path available on this machine."""
    if transformer is None:
        return []

    active = bootstrap_wan_cuda_settings()

    # Attention backend priority (Comfy parity): diffusers SAGE -> diffusers FLASH ->
    # global SDPA-sage monkeypatch -> plain torch SDPA. The diffusers per-module
    # backends are preferred because they handle the Wan layout natively and do not
    # patch global SDPA (which would also affect SD image generation). All are gated
    # on the backend actually being callable, so none can raise 'NoneType' is not callable.
    backend = _set_wan_sage_backend(transformer) or _set_wan_flash_backend(transformer)
    if backend:
        active.append(backend)
    else:
        sage = _try_sage_attention()
        if sage:
            active.append(sage)

    try:
        import torch

        transformer.to(memory_format=torch.channels_last, non_blocking=True)
        active.append("channels_last")
    except Exception:
        pass

    if active:
        logger.info("Wan %s optimizations: %s", name, ", ".join(active))
    return active


def describe_missing_comfy_parity() -> Iterable[str]:
    """Hints for closing the speed gap vs ComfyUI on the same GPU."""
    notes: list[str] = []
    try:
        import sageattention  # noqa: F401
    except ImportError:
        notes.append(
            "Install `sageattention` (Comfy `--use-sage-attention`) for a large Wan step-speed boost on RTX 40-series."
        )
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        notes.append(
            "Optional `flash-attn` is not installed; torch SDPA flash is used instead."
        )
    notes.append(
        "Avoid Sequential offload on 16 GB when using FP8 safetensors — use Model offload instead "
        "(sequential moves every layer over PCIe each step and is ~3-10x slower)."
    )
    notes.append(
        "Comfy VSR / NVIDIA toolkit / aimdo hooks are Comfy-only; AIWF uses diffusers + native FP8 `_scaled_mm` instead."
    )
    notes.append(
        "NVIDIA Control Panel → CUDA Sysmem Fallback Policy → Prefer No Sysmem Fallback "
        "surfaces OOM instead of silent 10x paging on 16 GB cards."
    )
    return notes
