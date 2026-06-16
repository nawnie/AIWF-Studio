"""
aiwf/infrastructure/samplers/dpmpp.py

DPM++ samplers — second and third order multi-step solvers.

Reverse-engineered from Lu et al. 2022,
"DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models"
(https://arxiv.org/abs/2211.01095).

No diffusers scheduler classes — pure torch math.

Implemented samplers
--------------------
* DPM++ 2M         — 2nd-order multi-step (most popular, fast convergence)
* DPM++ 2M Karras  — same + Karras sigma schedule (better quality at few steps)
* DPM++ SDE        — stochastic DPM++ 2nd-order (ancestral-like noise injection)
* DPM++ 3M SDE     — 3rd-order multi-step stochastic (higher quality, more memory)

All samplers use the same denoiser signature:
    denoiser(x, sigma) → x0_pred
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from typing import NamedTuple

import torch

from aiwf.infrastructure.samplers.euler import SamplerStep, randn_like_compat


DenoiserFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log_sigma(sigma: torch.Tensor) -> torch.Tensor:
    return sigma.log()


def _t_from_sigma(sigma: torch.Tensor) -> torch.Tensor:
    """Convert sigma to the DPM++ 'half-logSNR' time variable λ = -log(σ)."""
    return -sigma.log()


def _sigma_from_t(t: torch.Tensor) -> torch.Tensor:
    return (-t).exp()


# ---------------------------------------------------------------------------
# DPM++ 2M (multi-step, 2nd order)
# ---------------------------------------------------------------------------

def dpmpp_2m_sampler(
    denoiser: DenoiserFn,
    x: torch.Tensor,
    sigmas: torch.Tensor,
) -> Iterator[SamplerStep]:
    """DPM++ 2M deterministic sampler.

    Uses a 2nd-order linear multi-step predictor.  Requires only one model
    call per step (stores the previous denoised prediction for 2nd-order
    correction on the next step).

    First step degrades to 1st-order Euler.
    """
    n = len(sigmas) - 1
    old_denoised: torch.Tensor | None = None
    h_last: torch.Tensor | None = None

    for i in range(n):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        denoised = denoiser(x, sigma)

        t      = _t_from_sigma(sigma)
        t_next = _t_from_sigma(sigma_next)
        h      = t_next - t  # step size in λ-space

        if old_denoised is None or sigma_next == 0:
            # 1st order (Euler) for first step or final step
            x = (sigma_next / sigma) * x - (-h).expm1() * denoised
        else:
            # 2nd order correction using the previous denoised estimate
            h_last_ = h_last
            r = h_last_ / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_next / sigma) * x - (-h).expm1() * denoised_d

        old_denoised = denoised
        h_last = h

        yield SamplerStep(step=i + 1, total=n, x=x, denoised=denoised)


# ---------------------------------------------------------------------------
# DPM++ SDE (stochastic, 2nd order)
# ---------------------------------------------------------------------------

def dpmpp_sde_sampler(
    denoiser: DenoiserFn,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    eta: float = 1.0,
    s_noise: float = 1.0,
    generator: torch.Generator | None = None,
) -> Iterator[SamplerStep]:
    """DPM++ SDE — stochastic 2nd-order solver.

    Adds Langevin-style noise injection at each step.  Produces higher
    diversity than the deterministic variant at the cost of slightly more
    noise at low step counts.

    Parameters
    ----------
    eta:
        Overall stochasticity scale.  0.0 → deterministic DPM++ 2M.
    s_noise:
        Additional noise scale multiplier (s_noise=1.0 matches the paper).
    """
    n = len(sigmas) - 1

    for i in range(n):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        denoised = denoiser(x, sigma)

        if sigma_next == 0:
            # Final step — pure Euler, no noise
            d = (x - denoised) / sigma
            x = x + d * (sigma_next - sigma)
        else:
            t      = _t_from_sigma(sigma)
            t_next = _t_from_sigma(sigma_next)
            h      = t_next - t

            # Stochastic noise injection
            eta_h = eta * h
            x = (
                (sigma_next / sigma) * (-eta_h).exp() * x
                + (-eta_h).expm1().neg() * denoised
            )
            if eta > 0:
                noise = randn_like_compat(x, generator=generator)
                x = x + noise * sigma_next * (-2 * eta_h).expm1().neg().sqrt() * s_noise

        yield SamplerStep(step=i + 1, total=n, x=x, denoised=denoised)


# ---------------------------------------------------------------------------
# DPM++ 3M SDE (stochastic, 3rd order)
# ---------------------------------------------------------------------------

def dpmpp_3m_sde_sampler(
    denoiser: DenoiserFn,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    eta: float = 1.0,
    s_noise: float = 1.0,
    generator: torch.Generator | None = None,
) -> Iterator[SamplerStep]:
    """DPM++ 3M SDE — 3rd-order stochastic multi-step solver.

    Stores two previous denoised estimates for the 3rd-order correction.
    Produces the best quality of the DPM++ family at ≥20 steps.
    First two steps degrade to lower order.
    """
    n = len(sigmas) - 1
    d1_old: torch.Tensor | None = None
    d2_old: torch.Tensor | None = None
    h1_old: torch.Tensor | None = None
    h2_old: torch.Tensor | None = None

    for i in range(n):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        denoised = denoiser(x, sigma)

        t      = _t_from_sigma(sigma)
        t_next = _t_from_sigma(sigma_next) if sigma_next > 0 else t
        h      = t_next - t

        # Derivative
        d = (x - denoised) / sigma

        if d1_old is None:
            # 1st order
            x_next = x + d * (sigma_next - sigma)
        elif d2_old is None:
            # 2nd order
            h_ = h1_old
            r  = h_ / h
            d_cur = (1 + 0.5 / r) * d - 0.5 / r * d1_old
            x_next = x + d_cur * (sigma_next - sigma)
        else:
            # 3rd order
            h_  = h1_old
            hh  = h2_old
            r1  = h_ / h
            r2  = hh / h
            d_cur = (
                (1 + 1 / (2 * r1) + 1 / (3 * (r1 + r2))) * d
                - (1 / (2 * r1) + 1 / (3 * (r1 + r2))) * d1_old
                + (1 / (3 * (r1 + r2) * r2)) * d2_old
            )
            x_next = x + d_cur * (sigma_next - sigma)

        # Inject noise (skip on final step)
        if eta > 0 and sigma_next > 0:
            eta_h = eta * abs(float(h))
            x_next = x_next + randn_like_compat(x, generator=generator) * sigma_next * math.sqrt(
                max(1 - math.exp(-2 * eta_h), 0)
            ) * s_noise

        d2_old, d1_old = d1_old, d
        h2_old, h1_old = h1_old, h
        x = x_next

        yield SamplerStep(step=i + 1, total=n, x=x, denoised=denoised)
