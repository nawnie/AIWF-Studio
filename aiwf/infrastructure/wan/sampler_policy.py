"""Wan sampler / flow-shift calibration policy.

Wan2.2-TI2V-5B ships with UniPCMultistepScheduler at flow_shift=5.0. Euler/Heun
with elevated flow_shift (especially 8.0) is a known source of warped or garbage
motion in AIWF logs and benchmarks.
"""
from __future__ import annotations

from dataclasses import dataclass

from aiwf.core.domain.wan import (
    WAN_RUNTIME_FAST_5B,
    WAN_RUNTIME_HIGH_LOW,
    WAN_RUNTIME_HIGH_LOW_FP8,
    WanI2VRequest,
)

WAN_5B_NATIVE_FLOW_SHIFT = 5.0
WAN_5B_NATIVE_SAMPLER = "unipc"
WAN_5B_NATIVE_SIGMA = "simple"


@dataclass(frozen=True)
class WanSamplerAudit:
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    corrections: tuple[str, ...] = ()
    request: WanI2VRequest | None = None


def _clone_request(request: WanI2VRequest, **updates) -> WanI2VRequest:
    if hasattr(request, "model_copy"):
        return request.model_copy(update=updates)
    data = request.model_dump()
    data.update(updates)
    return WanI2VRequest(**data)


def audit_wan_sampler_settings(
    request: WanI2VRequest,
    *,
    enforce_5b_calibration: bool = True,
) -> WanSamplerAudit:
    errors: list[str] = []
    warnings: list[str] = []
    corrections: list[str] = []
    updated = request

    mode = str(getattr(request, "runtime_mode", "") or "")
    sampler = str(getattr(request, "sampler", WAN_5B_NATIVE_SAMPLER) or WAN_5B_NATIVE_SAMPLER).lower()
    flow_shift = float(getattr(request, "flow_shift", WAN_5B_NATIVE_FLOW_SHIFT) or WAN_5B_NATIVE_FLOW_SHIFT)
    sigma_type = str(getattr(request, "sigma_type", WAN_5B_NATIVE_SIGMA) or WAN_5B_NATIVE_SIGMA).lower()

    if mode == WAN_RUNTIME_FAST_5B:
        if enforce_5b_calibration:
            patch: dict[str, object] = {}
            if sampler != WAN_5B_NATIVE_SAMPLER:
                patch["sampler"] = WAN_5B_NATIVE_SAMPLER
                corrections.append(
                    f"5B sampler auto-corrected {sampler!r} -> {WAN_5B_NATIVE_SAMPLER!r} "
                    "(Wan2.2-TI2V-5B ships calibrated for UniPC)."
                )
            if abs(flow_shift - WAN_5B_NATIVE_FLOW_SHIFT) > 0.01:
                patch["flow_shift"] = WAN_5B_NATIVE_FLOW_SHIFT
                corrections.append(
                    f"5B flow_shift auto-corrected {flow_shift:g} -> {WAN_5B_NATIVE_FLOW_SHIFT:g} "
                    "(Wan2.2-TI2V-5B ships with flow_shift=5.0)."
                )
            if patch:
                updated = _clone_request(updated, **patch)
                sampler = str(getattr(updated, "sampler", WAN_5B_NATIVE_SAMPLER) or WAN_5B_NATIVE_SAMPLER).lower()
                flow_shift = float(
                    getattr(updated, "flow_shift", WAN_5B_NATIVE_FLOW_SHIFT) or WAN_5B_NATIVE_FLOW_SHIFT
                )

        if flow_shift >= 7.0 and not enforce_5b_calibration:
            errors.append(
                f"5B flow shift {flow_shift:g} is outside the calibrated range. "
                "Wan2.2-TI2V-5B ships with flow_shift=5.0; values around 8.0 commonly "
                "produce warped or garbage motion."
            )
        if sampler != WAN_5B_NATIVE_SAMPLER:
            warnings.append(
                f"5B sampler '{sampler}' is not the checkpoint-native UniPC solver. "
                "If motion looks wrong, switch Sampler to UniPC and Flow shift to 5.0."
            )
        if sampler in {"euler", "heun"} and flow_shift > WAN_5B_NATIVE_FLOW_SHIFT + 0.5:
            errors.append(
                f"5B {sampler} with flow_shift={flow_shift:g} is a known bad combo. "
                "Use UniPC + flow_shift 5.0, or lower flow_shift to 5.0 before using Euler/Heun."
            )

    elif mode == WAN_RUNTIME_HIGH_LOW_FP8:
        if str(getattr(request, "offload", "") or "") != "streamed":
            errors.append("14B FP8 route requires streamed group offload.")
        if sampler not in {"euler", "unipc", "heun"}:
            warnings.append(f"14B FP8 sampler '{sampler}' is unusual; Comfy 4-step routes often use Euler.")

    elif mode == WAN_RUNTIME_HIGH_LOW:
        if sampler not in {"euler", "unipc", "heun"}:
            warnings.append(f"GGUF route sampler '{sampler}' is unusual for Wan high/low pairs.")

    if sampler == "unipc" and sigma_type != WAN_5B_NATIVE_SIGMA and mode == WAN_RUNTIME_FAST_5B:
        warnings.append(
            "Scheduler (sigma type) only affects Euler/Heun on 5B. UniPC ignores it; "
            "leave Scheduler on Simple unless you switch away from UniPC."
        )

    return WanSamplerAudit(
        errors=tuple(errors),
        warnings=tuple(warnings),
        corrections=tuple(corrections),
        request=updated,
    )
