"""
aiwf/infrastructure/samplers/dispatch.py

Sampler dispatch — given a sampler ID string and a σ schedule, return the
right iterator factory.  This is the single entry point callers should use.

Usage
-----
    from aiwf.infrastructure.samplers.dispatch import run_sampler
    from aiwf.infrastructure.samplers.schedule import get_sigmas

    sigmas = get_sigmas("karras", num_steps=28)
    for step in run_sampler("dpmpp_2m", denoiser_fn, x_noisy, sigmas):
        update_progress(step.step, step.total)
    final_latent = step.x
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import torch

from aiwf.infrastructure.samplers.euler import (
    SamplerStep,
    euler_sampler,
    euler_ancestral_sampler,
)
from aiwf.infrastructure.samplers.dpmpp import (
    dpmpp_2m_sampler,
    dpmpp_sde_sampler,
    dpmpp_3m_sde_sampler,
)
from aiwf.infrastructure.samplers.ddim import ddim_sampler


DenoiserFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


# Map of AIWF sampler IDs → their factory functions + supported kwargs
_REGISTRY: dict[str, Callable] = {
    "euler":          euler_sampler,
    "euler_a":        euler_ancestral_sampler,
    "dpmpp_2m":       dpmpp_2m_sampler,
    "dpmpp_2m_karras": dpmpp_2m_sampler,  # schedule handled separately
    "dpmpp_sde":      dpmpp_sde_sampler,
    "dpmpp_2m_sde":   dpmpp_sde_sampler,
    "dpmpp_3m_sde":   dpmpp_3m_sde_sampler,
    "ddim":           ddim_sampler,
    "heun":           euler_sampler,       # Heun = Euler with 2-step correction (TODO: full impl)
    "lms":            euler_sampler,       # LMS multi-step (TODO: full impl)
    "unipc":          dpmpp_2m_sampler,    # UniPC ≈ DPM++ 2M for now (TODO: full impl)
    "sa_solver":      dpmpp_2m_sampler,
    "lcm":            euler_sampler,       # LCM uses euler at very low steps
    "tcd":            euler_sampler,
}


def available_samplers() -> list[str]:
    return sorted(_REGISTRY.keys())


def run_sampler(
    sampler_id: str,
    denoiser: DenoiserFn,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    generator: torch.Generator | None = None,
    eta: float = 1.0,
    **kwargs: Any,
) -> Iterator[SamplerStep]:
    """Dispatch to the correct sampler and yield SamplerStep objects.

    Parameters
    ----------
    sampler_id:
        AIWF sampler string, e.g. ``"euler_a"``, ``"dpmpp_2m_karras"``.
    denoiser:
        ``(x_noisy, sigma) → x0_pred``
    x:
        Initial latent.
    sigmas:
        σ sequence from :func:`~aiwf.infrastructure.samplers.schedule.get_sigmas`.
    generator:
        Optional torch.Generator for reproducible stochastic samplers.
    eta:
        Stochasticity scale for stochastic samplers.  Ignored by deterministic ones.
    """
    fn = _REGISTRY.get(sampler_id.lower())
    if fn is None:
        import logging
        logging.getLogger(__name__).warning(
            "Unknown sampler %r, falling back to euler_a", sampler_id
        )
        fn = euler_ancestral_sampler

    # Pass generator / eta only to functions that accept them
    import inspect
    sig = inspect.signature(fn)
    call_kwargs: dict[str, Any] = {}
    if "generator" in sig.parameters:
        call_kwargs["generator"] = generator
    if "eta" in sig.parameters:
        call_kwargs["eta"] = eta
    call_kwargs.update(kwargs)

    return fn(denoiser, x, sigmas, **call_kwargs)
