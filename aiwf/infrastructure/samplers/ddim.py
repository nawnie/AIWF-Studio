"""
aiwf/infrastructure/samplers/ddim.py

DDIM sampler — Denoising Diffusion Implicit Models (Song et al. 2020).
https://arxiv.org/abs/2010.02502

DDIM replaces the stochastic DDPM reverse process with a deterministic
implicit process parameterised by η.  At η=0 it's fully deterministic
(good for reproducibility / inversion).  At η=1 it matches DDPM.

Math
----
Given denoised prediction x0 at timestep t:

    ε_pred  = (x_t - √ᾱ_t · x0) / √(1 - ᾱ_t)     [predicted noise]
    x_{t-1} = √ᾱ_{t-1} · x0
              + √(1 - ᾱ_{t-1} - σ_t²) · ε_pred
              + σ_t · ε                              [σ_t = η · √(...)]

where ε ~ N(0,I) is fresh noise.  At η=0, σ_t=0 and the update is
deterministic.

No diffusers code.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator

import torch

from aiwf.infrastructure.samplers.euler import SamplerStep, randn_like_compat


DenoiserFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def ddim_sampler(
    denoiser: DenoiserFn,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    eta: float = 0.0,
    generator: torch.Generator | None = None,
) -> Iterator[SamplerStep]:
    """DDIM sampler operating in σ-space.

    Parameters
    ----------
    denoiser:
        ``(x_noisy, sigma) → x0_pred``
    x:
        Initial latent ~ N(0, σ_max² I)
    sigmas:
        Descending σ sequence, length n+1, last=0.
        (from :func:`~aiwf.infrastructure.samplers.schedule.get_sigmas`)
    eta:
        Stochasticity.  0.0 = DDIM deterministic.  1.0 = DDPM.
    """
    n = len(sigmas) - 1

    for i in range(n):
        sigma   = sigmas[i]
        sigma_next = sigmas[i + 1]

        x0_pred = denoiser(x, sigma)

        # α̅ values from σ: ᾱ = 1/(1+σ²)
        alpha_bar_t      = 1.0 / (1.0 + sigma**2)
        alpha_bar_t_next = 1.0 / (1.0 + sigma_next**2) if sigma_next > 0 else torch.ones_like(sigma_next)

        # Predicted noise from x0 prediction
        eps_pred = (x - alpha_bar_t.sqrt() * x0_pred) / (1.0 - alpha_bar_t).sqrt()

        # DDIM noise schedule parameter
        if eta > 0 and sigma_next > 0:
            sigma_t = eta * (
                ((1 - alpha_bar_t_next) / (1 - alpha_bar_t)) * (1 - alpha_bar_t / alpha_bar_t_next)
            ).sqrt()
        else:
            sigma_t = sigma.new_zeros(1)

        # "Direction pointing to x_t" coefficient
        direction_coeff = (1 - alpha_bar_t_next - sigma_t**2).clamp(min=0).sqrt()

        # DDIM update
        x = alpha_bar_t_next.sqrt() * x0_pred + direction_coeff * eps_pred

        if eta > 0 and sigma_next > 0:
            x = x + sigma_t * randn_like_compat(x, generator=generator)

        yield SamplerStep(step=i + 1, total=n, x=x, denoised=x0_pred)
