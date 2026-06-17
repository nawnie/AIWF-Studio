"""AIWF-native Wan runner facade.

This is the pass-6 foundation: explicit readiness and preparation methods
without taking over the existing diffusers generation path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aiwf.infrastructure.quant.fp8_linear import torch_native_fp8_available
from aiwf.infrastructure.wan.comfy_quant_format import WanQuantReport, inspect_wan_quant_file


@dataclass(frozen=True)
class NativeWanReadiness:
    ok: bool
    mode: str
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    high_report: WanQuantReport | None = None
    low_report: WanQuantReport | None = None

    def message(self) -> str:
        if self.ok:
            return "Native Wan runtime preflight passed."
        return "; ".join(self.errors) if self.errors else "Native Wan runtime preflight failed."


@dataclass
class AIWFWanRunner:
    """Thin native runner shell for upcoming high/low execution work."""

    ops: Any
    cache_mode: str = "none"
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def inspect_high_low(self, high_path: str | Path, low_path: str | Path, *, require_fp8: bool = False) -> NativeWanReadiness:
        high = Path(high_path)
        low = Path(low_path)
        errors: list[str] = []
        warnings: list[str] = []

        if not high.exists():
            errors.append(f"High-noise transformer does not exist: {high}")
        if not low.exists():
            errors.append(f"Low-noise transformer does not exist: {low}")
        if errors:
            return NativeWanReadiness(ok=False, mode="native_high_low", errors=tuple(errors))

        high_report = inspect_wan_quant_file(high) if high.suffix.lower() == ".safetensors" else None
        low_report = inspect_wan_quant_file(low) if low.suffix.lower() == ".safetensors" else None

        for label, report in (("High-noise", high_report), ("Low-noise", low_report)):
            if report is None:
                continue
            if report.format == "unreadable":
                errors.append(f"{label} safetensors header could not be read: {report.path.name}")
            if report.missing_scale_keys:
                errors.append(f"{label} FP8 scales missing: {', '.join(report.missing_scale_keys[:3])}")
            if report.unsupported_quant_formats:
                errors.append(f"{label} unsupported quant metadata: {', '.join(report.unsupported_quant_formats[:3])}")
            warnings.extend(f"{label}: {warning}" for warning in report.warnings)

        if require_fp8:
            if not torch_native_fp8_available():
                errors.append("Native FP8 tensor-core runtime is unavailable on this CUDA/PyTorch setup.")
            if high_report is not None and not high_report.is_comfy_fp8:
                errors.append("High-noise transformer is not detected as Comfy FP8.")
            if low_report is not None and not low_report.is_comfy_fp8:
                errors.append("Low-noise transformer is not detected as Comfy FP8.")

        return NativeWanReadiness(
            ok=not errors,
            mode="native_high_low_fp8" if require_fp8 else "native_high_low",
            errors=tuple(errors),
            warnings=tuple(warnings),
            high_report=high_report,
            low_report=low_report,
        )

    def prepare_latents(self, image: Any, vae: Any, **kwargs: Any):
        from .conditioning import prepare_wan_i2v_latents

        bundle = prepare_wan_i2v_latents(image, vae, **kwargs)
        self.diagnostics["latent_shape"] = bundle.latent_shape
        self.diagnostics["has_concat_latent_image"] = bundle.concat_latent_image is not None
        self.diagnostics["has_concat_mask"] = bundle.concat_mask is not None
        return bundle
