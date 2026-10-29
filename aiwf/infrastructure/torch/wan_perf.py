"""Runtime performance hooks for Wan video transformers (RTX / Comfy-parity path)."""
from __future__ import annotations

import logging
import os
import platform
import time
from dataclasses import asdict, dataclass
from importlib import import_module
from importlib import metadata
from importlib.util import find_spec
from typing import Any, Iterable

logger = logging.getLogger(__name__)

_ORIGINAL_SDPA = None
_SAGE_SDPA_INSTALLED = False


@dataclass(frozen=True)
class WanAccelerationCapability:
    name: str
    available: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _sage_preference() -> str:
    """User preference for SageAttention: auto (default), force, or off.

    Settings write AIWF_WAN_SAGE_ATTENTION as "auto"/"1"/"0"; the legacy
    AIWF_USE_SAGE_ATTENTION truthy flag still means force.
    """
    raw = os.environ.get("AIWF_WAN_SAGE_ATTENTION", "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return "off"
    if raw in {"1", "true", "yes", "on", "force"} or _env_flag("AIWF_USE_SAGE_ATTENTION"):
        return "force"
    return "auto"


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
    global _ORIGINAL_SDPA, _SAGE_SDPA_INSTALLED
    preference = _sage_preference()
    if preference == "off":
        return None
    if preference == "auto":
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

        if _SAGE_SDPA_INSTALLED:
            return "sageattention"
        _orig = torch.nn.functional.scaled_dot_product_attention
        if _ORIGINAL_SDPA is None:
            _ORIGINAL_SDPA = _orig

        def _sage_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
            if query.ndim == 4 and query.shape[2] <= 64 and query.shape[1] > query.shape[2]:
                return sageattn(query, key, value, is_causal=is_causal, tensor_layout="NHD")
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
        _SAGE_SDPA_INSTALLED = True
        return "sageattention"
    except Exception as exc:
        logger.warning("sageattention hook failed (%s); using torch SDPA", exc)
        return None


def restore_wan_attention_patch() -> bool:
    """Undo the fallback SageAttention SDPA monkeypatch after Wan work.

    The fallback hook is useful for Wan, but it is process-global. Leaving it
    installed can break unrelated models with different attention layouts.
    """
    global _SAGE_SDPA_INSTALLED
    if not _SAGE_SDPA_INSTALLED or _ORIGINAL_SDPA is None:
        return False
    try:
        import torch

        torch.nn.functional.scaled_dot_product_attention = _ORIGINAL_SDPA
        _SAGE_SDPA_INSTALLED = False
        return True
    except Exception:
        logger.debug("Failed to restore torch SDPA after Wan SageAttention patch.", exc_info=True)
        return False


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


def _clean_nvml_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def describe_wan_hardware_fingerprint() -> dict[str, Any]:
    """Return static board/runtime metadata for Wan performance receipts.

    This is intentionally non-invasive: no allocations, no transfer benchmark,
    and no shelling out to ``nvidia-smi``. NVML fields are advisory because the
    Python binding is optional.
    """
    fingerprint: dict[str, Any] = {
        "os_name": platform.platform(),
        "cpu_model": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "nvml_available": False,
    }
    try:
        import psutil

        fingerprint["physical_ram_bytes"] = int(psutil.virtual_memory().total)
        fingerprint["pagefile_limit_bytes"] = int(psutil.swap_memory().total)
    except Exception:
        fingerprint["physical_ram_bytes"] = None
        fingerprint["pagefile_limit_bytes"] = None

    try:
        import torch

        fingerprint["torch_version"] = str(getattr(torch, "__version__", ""))
        fingerprint["cuda_runtime_version"] = str(getattr(torch.version, "cuda", "") or "")
        cuda_available = bool(torch.cuda.is_available())
        fingerprint["cuda_available"] = cuda_available
        if cuda_available:
            index = int(torch.cuda.current_device())
            props = torch.cuda.get_device_properties(index)
            fingerprint.update(
                {
                    "gpu_index": index,
                    "gpu_name": str(getattr(props, "name", "")),
                    "compute_capability": [
                        int(getattr(props, "major", 0)),
                        int(getattr(props, "minor", 0)),
                    ],
                    "total_vram_bytes": int(getattr(props, "total_memory", 0) or 0),
                    "multiprocessor_count": int(getattr(props, "multi_processor_count", 0) or 0),
                    "max_threads_per_sm": getattr(props, "max_threads_per_multi_processor", None),
                    "l2_cache_bytes": getattr(props, "l2_cache_size", None),
                    "async_engine_count": getattr(props, "async_engine_count", None),
                }
            )
    except Exception as exc:
        fingerprint["cuda_error"] = f"{type(exc).__name__}: {exc}"

    try:
        fingerprint["triton_version"] = metadata.version("triton")
    except metadata.PackageNotFoundError:
        try:
            fingerprint["triton_version"] = metadata.version("triton-windows")
        except metadata.PackageNotFoundError:
            fingerprint["triton_version"] = None

    try:
        import pynvml

        pynvml.nvmlInit()
        index = int(fingerprint.get("gpu_index") or 0)
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        pci = pynvml.nvmlDeviceGetPciInfo(handle)
        fingerprint.update(
            {
                "nvml_available": True,
                "gpu_uuid": _clean_nvml_value(pynvml.nvmlDeviceGetUUID(handle)),
                "driver_version": _clean_nvml_value(pynvml.nvmlSystemGetDriverVersion()),
                "pci_bus_id": _clean_nvml_value(getattr(pci, "busId", "")),
                "pci_device_id": getattr(pci, "pciDeviceId", None),
                "pci_subsystem_id": getattr(pci, "pciSubSystemId", None),
            }
        )
        optional_calls = {
            "vbios_version": "nvmlDeviceGetVbiosVersion",
            "memory_bus_width_bits": "nvmlDeviceGetMemoryBusWidth",
            "pcie_current_generation": "nvmlDeviceGetCurrPcieLinkGeneration",
            "pcie_current_width": "nvmlDeviceGetCurrPcieLinkWidth",
            "pcie_max_generation": "nvmlDeviceGetMaxPcieLinkGeneration",
            "pcie_max_width": "nvmlDeviceGetMaxPcieLinkWidth",
        }
        for key, func_name in optional_calls.items():
            func = getattr(pynvml, func_name, None)
            if func is None:
                fingerprint[key] = None
                continue
            try:
                fingerprint[key] = _clean_nvml_value(func(handle))
            except Exception:
                fingerprint[key] = None
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    except Exception as exc:
        fingerprint["nvml_error"] = f"{type(exc).__name__}: {exc}"

    return fingerprint


def measure_wan_transfer_bandwidth(
    *,
    size_bytes: int = 64 * 1024 * 1024,
    include_pinned: bool = True,
) -> dict[str, Any]:
    """Run a small opt-in H2D/D2H copy probe for PCIe sanity checks."""
    try:
        import torch
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    if not torch.cuda.is_available():
        return {"available": False, "error": "CUDA unavailable"}

    results: list[dict[str, Any]] = []
    count = max(1, int(size_bytes) // 4)
    for pinned in ([False, True] if include_pinned else [False]):
        try:
            cpu = torch.empty(count, dtype=torch.float32, pin_memory=bool(pinned))
            gpu = torch.empty(count, dtype=torch.float32, device="cuda")
            back = torch.empty(count, dtype=torch.float32, pin_memory=bool(pinned))
            torch.cuda.synchronize()
            h2d_started = time.perf_counter()
            gpu.copy_(cpu, non_blocking=bool(pinned))
            torch.cuda.synchronize()
            h2d_seconds = max(1e-9, time.perf_counter() - h2d_started)
            d2h_started = time.perf_counter()
            back.copy_(gpu, non_blocking=bool(pinned))
            torch.cuda.synchronize()
            d2h_seconds = max(1e-9, time.perf_counter() - d2h_started)
            gb = float(size_bytes) / 1_000_000_000.0
            results.append(
                {
                    "pinned": bool(pinned),
                    "size_bytes": int(size_bytes),
                    "h2d_gbps": round(gb / h2d_seconds, 3),
                    "d2h_gbps": round(gb / d2h_seconds, 3),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "pinned": bool(pinned),
                    "size_bytes": int(size_bytes),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            try:
                del cpu, gpu, back
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    return {"available": True, "results": results}


def apply_wan_transformer_optimizations(transformer, *, name: str = "transformer") -> list[str]:
    """Apply the fastest attention/conv path available on this machine."""
    if transformer is None:
        return []

    existing = list(getattr(transformer, "_aiwf_attention_optimizations", ()) or ())
    active = bootstrap_wan_cuda_settings()

    # Attention backend priority (Comfy parity): diffusers SAGE -> diffusers FLASH ->
    # global SDPA-sage monkeypatch -> plain torch SDPA. The diffusers per-module
    # backends are preferred because they handle the Wan layout natively and do not
    # patch global SDPA (which would also affect SD image generation). All are gated
    # on the backend actually being callable, so none can raise 'NoneType' is not callable.
    sage_allowed = _sage_preference() != "off"
    backend = (_set_wan_sage_backend(transformer) if sage_allowed else None) or _set_wan_flash_backend(transformer)
    backend_label = "torch_sdpa"
    if backend:
        active.append(backend)
        backend_label = backend
    else:
        sage = _try_sage_attention()
        if sage:
            active.append(sage)
            backend_label = sage

    try:
        import torch

        if getattr(transformer, "_aiwf_group_offload", False):
            if "channels_last" in existing and "channels_last" not in active:
                active.append("channels_last")
        else:
            transformer.to(memory_format=torch.channels_last, non_blocking=True)
            active.append("channels_last")
    except Exception:
        pass

    try:
        transformer._aiwf_attention_backend = backend_label
        transformer._aiwf_attention_optimizations = tuple(active)
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
