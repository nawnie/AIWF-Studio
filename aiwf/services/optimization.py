from __future__ import annotations

import hashlib
import platform
from importlib import metadata
from typing import Iterable

from aiwf.core.domain.optimization import (
    AttentionBackend,
    AttentionPolicy,
    CapabilityFeature,
    CapabilityReport,
    CompilePolicy,
    CompileTarget,
    CpuOffloadPolicy,
    DTypePolicy,
    EnginePolicy,
    FastMethod,
    GpuCapability,
    MemoryFormat,
    MemoryPolicy,
    ModelFamily,
    OptimizationPlan,
    OptimizationProfile,
    OptimizationRequest,
    PipelineKind,
    PlannerDecision,
    ProfileReadiness,
    QualityModifiers,
    QuantPolicy,
    SchedulerPolicy,
    VaePolicy,
    VaeSwitch,
)


CORE_PACKAGES = (
    "torch",
    "diffusers",
    "transformers",
    "accelerate",
    "peft",
    "safetensors",
    "compel",
)

OPTIONAL_FEATURE_PACKAGES = {
    "attention.xformers": "xformers",
    "attention.flash": "flash-attn",
    "attention.sage": "sageattention",
    "quant.bitsandbytes": "bitsandbytes",
    "quant.torchao_fp8": "torchao",
    "quant.optimum_quanto": "optimum-quanto",
    "engine.tensorrt": "tensorrt",
    "engine.torch_tensorrt": "torch-tensorrt",
    "engine.onnx_runtime": "onnxruntime",
    "engine.onnx_runtime_gpu": "onnxruntime-gpu",
    "quant.gguf": "gguf",
    "quant.modelopt": "nvidia-modelopt",
}

_KNOWN_FAST_METHODS = {"lcm", "sdxl_turbo", "sdxl_lightning"}


def default_optimization_profiles() -> dict[str, OptimizationProfile]:
    """Return the conservative profile registry from the research package."""
    safe = OptimizationProfile(
        profile_id="safe_eager_cuda",
        display_name="Safe",
        readiness=ProfileReadiness.PRODUCTION,
        dtype_policy=DTypePolicy(unet="fp16", text_encoder="fp16", vae="fp16", controlnet="fp16"),
        attention_backend=AttentionPolicy(name=AttentionBackend.SDPA, fallback=AttentionBackend.SDPA),
        memory_policy=MemoryPolicy(memory_format=MemoryFormat.CONTIGUOUS),
        vae_policy=VaePolicy(slicing=VaeSwitch.OFF, tiling=VaeSwitch.OFF),
        scheduler_policy=SchedulerPolicy(scheduler="euler_or_dpmpp", steps=25, cfg_scale=7.0),
    )
    balanced = OptimizationProfile(
        profile_id="balanced_sdpa_fp16",
        display_name="Balanced",
        readiness=ProfileReadiness.PRODUCTION,
        dtype_policy=DTypePolicy(unet="fp16", text_encoder="fp16", vae="fp16", controlnet="fp16"),
        attention_backend=AttentionPolicy(name=AttentionBackend.SDPA, fallback=AttentionBackend.SDPA),
        memory_policy=MemoryPolicy(memory_format=MemoryFormat.AUTO),
        vae_policy=VaePolicy(slicing=VaeSwitch.AUTO, tiling=VaeSwitch.OFF),
        scheduler_policy=SchedulerPolicy(scheduler="dpmpp_or_euler", steps=25, cfg_scale=7.0),
    )
    quality = balanced.model_copy(
        deep=True,
        update={
            "profile_id": "quality_visible_modifiers",
            "display_name": "Quality",
            "scheduler_policy": SchedulerPolicy(scheduler="dpmpp_or_euler", steps=35, cfg_scale=6.0),
            "quality_modifiers": QualityModifiers(
                freeu=False,
                pag=False,
                refiner_sdxl=False,
                hires_fix=True,
                clip_skip_visible=True,
            ),
            "output_changes_visible": True,
        },
    )
    low_vram = safe.model_copy(
        deep=True,
        update={
            "profile_id": "low_vram_model_offload",
            "display_name": "Low VRAM",
            "memory_policy": MemoryPolicy(
                memory_format=MemoryFormat.CONTIGUOUS,
                cpu_offload=CpuOffloadPolicy.MODEL,
            ),
            "vae_policy": VaePolicy(slicing=VaeSwitch.AUTO, tiling=VaeSwitch.AUTO),
        },
    )
    fast = balanced.model_copy(
        deep=True,
        update={
            "profile_id": "fast_method_recipe",
            "display_name": "Fast Mode",
            "readiness": ProfileReadiness.BETA,
            "pipeline_kind": PipelineKind.FAST,
            "fast_method": FastMethod(name=None, requires_matching_checkpoint=True),
            "scheduler_policy": SchedulerPolicy(scheduler="method_recipe", steps=None, cfg_scale=None),
            "output_changes_visible": True,
        },
    )
    experimental = balanced.model_copy(
        deep=True,
        update={
            "profile_id": "experimental_feature_flags",
            "display_name": "Experimental Lab",
            "readiness": ProfileReadiness.EXPERIMENTAL,
        },
    )
    return {profile.profile_id: profile for profile in (safe, balanced, quality, low_vram, fast, experimental)}


class CapabilityDetector:
    """Lazy capability detection for optimization planning.

    Optional accelerator packages are checked with package metadata first. The
    detector only imports torch inside ``detect`` to inspect CUDA availability.
    """

    def __init__(
        self,
        *,
        core_packages: Iterable[str] = CORE_PACKAGES,
        optional_packages: dict[str, str] | None = None,
    ) -> None:
        self.core_packages = tuple(core_packages)
        self.optional_packages = dict(optional_packages or OPTIONAL_FEATURE_PACKAGES)

    def detect(self, *, include_gpu: bool = True) -> CapabilityReport:
        packages = {name: self._version_or_none(name) for name in self.core_packages}
        features = {
            key: self._feature_from_package(package)
            for key, package in sorted(self.optional_packages.items())
        }
        gpu = self._gpu_capability() if include_gpu else GpuCapability()
        payload = "|".join(
            [
                platform.platform(),
                platform.python_version(),
                gpu.name,
                ",".join(f"{k}={v}" for k, v in sorted(packages.items())),
                ",".join(f"{k}={v.available}:{v.version}" for k, v in sorted(features.items())),
            ]
        )
        report_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return CapabilityReport(
            report_id=report_id,
            os=platform.system(),
            python=platform.python_version(),
            platform=platform.platform(),
            packages=packages,
            gpu=gpu,
            features=features,
        )

    @staticmethod
    def _version_or_none(package: str) -> str | None:
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            return None

    def _feature_from_package(self, package: str) -> CapabilityFeature:
        version = self._version_or_none(package)
        if version is None:
            return CapabilityFeature(available=False, reason=f"{package} is not installed")
        return CapabilityFeature(available=True, version=version)

    @staticmethod
    def _gpu_capability() -> GpuCapability:
        try:
            import torch
        except Exception as exc:
            return GpuCapability(cuda_runtime="", driver_version="", name="", compute_capability="", vram_total_bytes=None)
        try:
            if not torch.cuda.is_available():
                return GpuCapability(cuda_runtime=getattr(torch.version, "cuda", "") or "")
            index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(index)
            major, minor = torch.cuda.get_device_capability(index)
            return GpuCapability(
                name=str(getattr(props, "name", "")),
                compute_capability=f"{major}.{minor}",
                vram_total_bytes=int(getattr(props, "total_memory", 0) or 0),
                cuda_runtime=getattr(torch.version, "cuda", "") or "",
                driver_version="",
            )
        except Exception:
            return GpuCapability(cuda_runtime=getattr(getattr(torch, "version", None), "cuda", "") or "")


class OptimizationPlanner:
    """Resolve profile requests into safe, explainable optimization plans."""

    def __init__(self, profiles: dict[str, OptimizationProfile] | None = None) -> None:
        self.profiles = dict(profiles or default_optimization_profiles())

    def resolve(
        self,
        request: OptimizationRequest,
        *,
        capabilities: CapabilityReport | None = None,
    ) -> OptimizationPlan:
        requested = self.profiles.get(request.profile_id)
        decisions: list[PlannerDecision] = []
        if requested is None:
            requested = self.profiles["safe_eager_cuda"]
            decisions.append(
                PlannerDecision(
                    key="profile",
                    decision="fallback",
                    reason=f"Unknown profile '{request.profile_id}', using Safe.",
                    severity="warning",
                )
            )

        effective = requested.model_copy(deep=True)
        fallback_profile_id: str | None = None
        blocked = False

        if request.experimental_flags.get("attention.xformers"):
            if self._feature_available(capabilities, "attention.xformers"):
                effective.attention_backend = AttentionPolicy(
                    name=AttentionBackend.XFORMERS,
                    fallback=AttentionBackend.SDPA,
                    requires_probe=True,
                )
                decisions.append(self._enabled("attention.xformers", "xFormers selected after capability check."))
            else:
                decisions.append(self._disabled("attention.xformers", "xFormers unavailable; using PyTorch SDPA."))

        if request.experimental_flags.get("attention.flash"):
            if self._feature_available(capabilities, "attention.flash"):
                effective.attention_backend = AttentionPolicy(
                    name=AttentionBackend.FLASH,
                    fallback=AttentionBackend.SDPA,
                    requires_probe=True,
                )
                decisions.append(self._enabled("attention.flash", "FlashAttention selected after capability check."))
            else:
                decisions.append(self._disabled("attention.flash", "FlashAttention unavailable; using PyTorch SDPA."))

        if request.experimental_flags.get("attention.sage"):
            if self._feature_available(capabilities, "attention.sage"):
                effective.attention_backend = AttentionPolicy(
                    name=AttentionBackend.SAGE,
                    fallback=AttentionBackend.SDPA,
                    requires_probe=True,
                )
                decisions.append(self._enabled("attention.sage", "SageAttention selected after capability check."))
            else:
                decisions.append(self._disabled("attention.sage", "SageAttention unavailable; using PyTorch SDPA."))

        if request.experimental_flags.get("compile.unet"):
            if request.lora_count > 0:
                fallback_profile_id = "safe_eager_cuda"
                decisions.append(
                    PlannerDecision(
                        key="compile.unet",
                        decision="blocked",
                        reason="torch.compile is blocked for arbitrary LoRA hotswap until a prepared compiled profile exists.",
                        severity="error",
                    )
                )
                blocked = True
            elif request.controlnet_count > 0 or request.pipeline_kind in (PipelineKind.INPAINT, PipelineKind.CONTROLNET):
                decisions.append(
                    PlannerDecision(
                        key="compile.unet",
                        decision="disabled",
                        reason="compile is kept off for dynamic inpaint/ControlNet paths until benchmarked.",
                        severity="warning",
                    )
                )
            else:
                effective.compile_policy = CompilePolicy(
                    enabled=True,
                    target=CompileTarget.UNET,
                    mode="reduce-overhead",
                    fullgraph=False,
                    dynamic=False,
                    fixed_shapes={
                        "width": request.width,
                        "height": request.height,
                        "batch_size": request.batch_size,
                    },
                )
                decisions.append(self._enabled("compile.unet", "Fixed-shape UNet compile selected."))

        if request.experimental_flags.get("quant.torchao_fp8"):
            gpu_cc = (capabilities.gpu.compute_capability if capabilities else "") or ""
            if self._feature_available(capabilities, "quant.torchao_fp8") and self._compute_capability_at_least(gpu_cc, 8, 9):
                effective.quant_policy = QuantPolicy(enabled=True, backend="torchao_fp8", dtype="fp8")
                decisions.append(self._enabled("quant.torchao_fp8", "torchao FP8 selected for experimental profile."))
            else:
                decisions.append(
                    self._disabled(
                        "quant.torchao_fp8",
                        "torchao FP8 requires torchao and compute capability >= 8.9; leaving quantization off.",
                    )
                )

        if request.experimental_flags.get("quant.fp4_nvfp4"):
            gpu_cc = (capabilities.gpu.compute_capability if capabilities else "") or ""
            if self._compute_capability_at_least(gpu_cc, 12, 0):
                decisions.append(
                    self._disabled(
                        "quant.fp4_nvfp4",
                        "FP4/NVFP4 remains disabled until a verified Blackwell runtime path and receipts exist.",
                    )
                )
            else:
                decisions.append(
                    self._disabled(
                        "quant.fp4_nvfp4",
                        "FP4/NVFP4 is Blackwell-facing and is not a runtime speed path for Ada RTX 40-series GPUs.",
                    )
                )

        if request.experimental_flags.get("engine.tensorrt"):
            fallback_profile_id = "safe_eager_cuda"
            effective.engine_policy = EnginePolicy(enabled=False, backend="pytorch")
            decisions.append(
                PlannerDecision(
                    key="engine.tensorrt",
                    decision="disabled",
                    reason="TensorRT remains Engine Lab only until engine cache, shape ranges, and LoRA refit are implemented.",
                    severity="warning",
                )
            )

        if request.experimental_flags.get("engine.onnx_runtime"):
            if self._feature_available(capabilities, "engine.onnx_runtime_gpu"):
                effective.engine_policy = EnginePolicy(enabled=True, backend="onnxruntime-gpu")
                decisions.append(self._enabled("engine.onnx_runtime", "ONNX Runtime GPU selected after capability check."))
            else:
                decisions.append(
                    self._disabled(
                        "engine.onnx_runtime",
                        "ONNX Runtime is not a CUDA speed path unless onnxruntime-gpu is installed and benchmarked.",
                    )
                )

        if request.vae_tiling_requested:
            effective.vae_policy.tiling = VaeSwitch.ON
            effective.output_changes_visible = True
            decisions.append(
                PlannerDecision(
                    key="vae.tiling",
                    decision="visible",
                    reason="VAE tiling can change tone/detail and must be visible in metadata and receipts.",
                    severity="info",
                )
            )
            if request.strict_quality_baseline:
                decisions.append(
                    PlannerDecision(
                        key="vae.tiling.baseline",
                        decision="warning",
                        reason="Strict baseline comparisons should disable VAE tiling or mark the comparison non-baseline.",
                        severity="warning",
                    )
                )

        if requested.profile_id == "fast_method_recipe":
            method = (request.fast_method or "").strip().lower()
            if not method:
                decisions.append(
                    PlannerDecision(
                        key="fast_method",
                        decision="blocked",
                        reason="Fast Mode requires an explicit method recipe such as lcm, sdxl_turbo, or sdxl_lightning.",
                        severity="error",
                    )
                )
                blocked = True
            elif method not in _KNOWN_FAST_METHODS:
                decisions.append(
                    PlannerDecision(
                        key="fast_method",
                        decision="blocked",
                        reason=f"Fast Mode method '{method}' has no explicit recipe yet.",
                        severity="error",
                    )
                )
                blocked = True
            else:
                effective.fast_method = self._fast_method(method)
                decisions.append(self._enabled("fast_method", f"Using explicit Fast Mode recipe '{method}'."))

        if request.lora_targets_text_encoder and effective.compile_policy.enabled:
            effective.compile_policy = CompilePolicy()
            decisions.append(
                PlannerDecision(
                    key="compile.text_encoder_lora",
                    decision="disabled",
                    reason="Compiled profiles do not hotswap text-encoder LoRA safely yet.",
                    severity="warning",
                )
            )

        return OptimizationPlan(
            requested_profile_id=request.profile_id,
            effective_profile=effective,
            capability_report_id=capabilities.report_id if capabilities else None,
            decisions=decisions,
            blocked=blocked,
            fallback_profile_id=fallback_profile_id,
        )

    @staticmethod
    def _enabled(key: str, reason: str) -> PlannerDecision:
        return PlannerDecision(key=key, decision="enabled", reason=reason)

    @staticmethod
    def _disabled(key: str, reason: str) -> PlannerDecision:
        return PlannerDecision(key=key, decision="disabled", reason=reason, severity="warning")

    @staticmethod
    def _feature_available(capabilities: CapabilityReport | None, key: str) -> bool:
        if capabilities is None:
            return False
        feature = capabilities.features.get(key)
        return bool(feature and feature.available)

    @staticmethod
    def _compute_capability_at_least(value: str, major: int, minor: int) -> bool:
        try:
            got_major, got_minor = (int(part) for part in value.split(".", 1))
        except Exception:
            return False
        return (got_major, got_minor) >= (major, minor)

    @staticmethod
    def _fast_method(method: str) -> FastMethod:
        if method == "lcm":
            return FastMethod(name="lcm", scheduler="lcm", steps=6, cfg_scale=1.5)
        if method == "sdxl_turbo":
            return FastMethod(name="sdxl_turbo", scheduler="euler", steps=2, cfg_scale=0.0)
        if method == "sdxl_lightning":
            return FastMethod(name="sdxl_lightning", scheduler="euler_trailing", steps=4, cfg_scale=0.0)
        return FastMethod(name=method, scheduler="method_recipe", steps=None, cfg_scale=None)
