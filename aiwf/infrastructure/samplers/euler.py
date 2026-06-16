"""
aiwf/infrastructure/samplers/euler.py

Euler and Euler Ancestral samplers — first-order ODE/SDE solvers.

Reverse-engineered from first principles.  No diffusers scheduler classes.

Euler (deterministic)
---------------------
Algorithm 2 from Karras et al. 2022, "Elucidating the Design Space of
Diffusion-Based Generative Models":

    d   = (x - D(x, σ)) / σ        # derivative estimate (predicted direction)
    x   = x + (σ_{i+1} - σ_i) * d  # Euler step

where D(x, σ) is the model's denoised prediction.

Euler Ancestral (stochastic)
----------------------------
Adds Langevin noise at each step:

    σ_up    = √(σ_{i+1}² * (1 - (σ_{i+1}/σ_i)²))   # noise injected
    σ_down  = √(σ_{i+1}² - σ_up²)                    # deterministic progress
    d       = (x - D(x, σ_i)) / σ_i
    x       = x + (σ_down - σ_i) * d + σ_up * ε,  ε ~ N(0, I)

Both samplers share the same denoiser call signature:
    denoiser(x, sigma) -> x0_pred (predicted clean latent)
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import NamedTuple

import torch


DenoiserFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def randn_like_compat(x: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """torch.randn_like(generator=...) is unavailable in some Torch builds."""
    if generator is None:
        return torch.randn_like(x)
    return torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)


class SamplerStep(NamedTuple):
    step: int
    total: int
    x: torch.Tensor      # current latent
    denoised: torch.Tensor  # x0 prediction at this step


def euler_sampler(
    denoiser: DenoiserFn,
    x: torch.Tensor,
    sigmas: torch.Tensor,
) -> Iterator[SamplerStep]:
    """Deterministic Euler sampler.

    Yields a SamplerStep after each denoising step so callers can stream
    progress or grab previews.  The final yielded step contains the
    fully-denoised latent in ``x``.

    Parameters
    ----------
    denoiser:
        Callable ``(x_noisy, sigma) → x0_pred``.  The model's denoised
        prediction — not an epsilon prediction.  Wrap models that predict
        epsilon with :func:`epsilon_to_x0`.
    x:
        Starting latent.  Should be sampled from N(0, σ_max² I).
    sigmas:
        Sequence of σ values, length num_steps + 1, last element = 0
        (from :func:`~aiwf.infrastructure.samplers.schedule.get_sigmas`).
    """
    n = len(sigmas) - 1  # actual number of steps
    for i in range(n):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        denoised = denoiser(x, sigma)
        # Euler derivative: direction from noisy to clean
        d = (x - denoised) / sigma
        # Euler step
        dt = sigma_next - sigma
        x = x + d * dt

        yield SamplerStep(step=i + 1, total=n, x=x, denoised=denoised)


def euler_ancestral_sampler(
    denoiser: DenoiserFn,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    eta: float = 1.0,
    generator: torch.Generator | None = None,
) -> Iterator[SamplerStep]:
    """Stochastic Euler Ancestral sampler.

    Parameters
    ----------
    eta:
        Noise injection scale.  1.0 = full ancestral noise.  0.0 = deterministic
        (identical to Euler at η=0).
    generator:
        Optional torch.Generator for reproducible noise.
    """
    n = len(sigmas) - 1
    for i in range(n):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        denoised = denoiser(x, sigma)

        if eta > 0 and sigma_next > 0:
            # Stochastic noise budget split
            sigma_up = (sigma_next**2 * (1 - (sigma_next / sigma) ** 2)).sqrt() * eta
            sigma_down = (sigma_next**2 - sigma_up**2).sqrt()
        else:
            sigma_up = sigma.new_zeros(1)
            sigma_down = sigma_next

        d = (x - denoised) / sigma
        dt = sigma_down - sigma
        x = x + d * dt

        if eta > 0 and sigma_next > 0:
            noise = randn_like_compat(x, generator=generator)
            x = x + noise * sigma_up

        yield SamplerStep(step=i + 1, total=n, x=x, denoised=denoised)


# ---------------------------------------------------------------------------
# Utility: epsilon → x0 prediction conversion
# ---------------------------------------------------------------------------

def epsilon_to_x0(x_t: torch.Tensor, eps: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Convert an epsilon (noise) prediction to a denoised (x0) prediction.

    x0 = (x_t - σ * ε) / 1     [since x_t = x0 + σ·ε  ⟹  x0 = x_t - σ·ε]
    """
    return x_t - sigma * eps


def v_pred_to_x0(x_t: torch.Tensor, v: torch.Tensor, sigma: torch.Tensor, sigma_data: float = 1.0) -> torch.Tensor:
    """Convert a v-prediction to x0.  Used by some SDXL checkpoints.

    v = (x0 - x_t * σ²) / (σ * √(1 + σ²))    (simplified for σ_data=1)
    ⟹ x0 = v * σ / √(1 + σ²) + x_t / (1 + σ²)
    """
    c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
    c_out = -sigma * sigma_data / (sigma**2 + sigma_data**2).sqrt()
    return c_skip * x_t + c_out * v
