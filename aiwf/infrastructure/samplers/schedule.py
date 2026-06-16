"""
aiwf/infrastructure/samplers/schedule.py

Noise schedules — the sigma (σ) tables that drive the denoising process.

These are reverse-engineered from first principles, not from diffusers source.
Everything here is pure torch math.

A noise schedule defines how much noise is mixed into the latent at each
step.  The key quantity is σ (sigma), the standard deviation of the noise at
a given timestep:

    x_t = x_0 + σ_t * ε,   ε ~ N(0, I)

The denoiser D(x_t, σ_t) predicts x_0 (or ε) from the noisy observation.
Samplers iterate σ from σ_max → σ_min over T steps.

Supported schedules
-------------------
* linear        — equal steps in [β_start, β_end]
* scaled_linear — linear in √β (used by LDM / SD 1.x)
* cosine        — cosine annealing (used by Improved DDPM)
* karras        — Karras et al. 2022 "Elucidating the Design Space" table
* exponential   — geometric spacing in log(σ) space
"""
from __future__ import annotations

import math
from typing import Literal

import torch


ScheduleType = Literal["linear", "scaled_linear", "cosine", "karras", "exponential"]


# ---------------------------------------------------------------------------
# β schedules → α̅ (alpha_bar) → σ
# ---------------------------------------------------------------------------

def _linear_betas(num_steps: int, beta_start: float = 0.00085, beta_end: float = 0.012) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float64)


def _scaled_linear_betas(num_steps: int, beta_start: float = 0.00085, beta_end: float = 0.012) -> torch.Tensor:
    """Linear in √β — matches stable-diffusion-v1 checkpoint training."""
    return torch.linspace(beta_start**0.5, beta_end**0.5, num_steps, dtype=torch.float64) ** 2


def _cosine_betas(num_steps: int, s: float = 0.008) -> torch.Tensor:
    """Nichol & Dhariwal 2021 cosine schedule."""
    t = torch.linspace(0, num_steps, num_steps + 1, dtype=torch.float64)
    f = torch.cos((t / num_steps + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(max=0.9999)


def _betas_to_sigmas(betas: torch.Tensor) -> torch.Tensor:
    """Convert β schedule to σ table.

    α_t = 1 - β_t
    ᾱ_t = ∏ α_i   (cumulative product)
    σ_t  = √((1 - ᾱ_t) / ᾱ_t)
    """
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    sigmas = ((1 - alpha_bar) / alpha_bar).sqrt()
    return sigmas


# ---------------------------------------------------------------------------
# Karras σ table (Karras et al. 2022 — "Elucidating the Design Space")
# ---------------------------------------------------------------------------

def karras_sigmas(
    n: int,
    sigma_min: float = 0.0292,
    sigma_max: float = 14.6146,
    rho: float = 7.0,
) -> torch.Tensor:
    """Compute n Karras sigmas descending from sigma_max → sigma_min.

    σ_i = (σ_max^(1/ρ) + i/(n-1) * (σ_min^(1/ρ) - σ_max^(1/ρ)))^ρ

    The default σ_min/σ_max match the SD 1.x 1000-step schedule clipped to
    the range actually encountered during generation (steps 1..999).
    """
    rho_inv = 1.0 / rho
    min_inv_rho = sigma_min**rho_inv
    max_inv_rho = sigma_max**rho_inv
    t = torch.linspace(0, 1, n, dtype=torch.float64)
    sigmas = (max_inv_rho + t * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas.float()


def exponential_sigmas(n: int, sigma_min: float = 0.0292, sigma_max: float = 14.6146) -> torch.Tensor:
    """Geometric spacing in log-sigma space."""
    sigmas = torch.exp(torch.linspace(math.log(sigma_max), math.log(sigma_min), n))
    return sigmas.float()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_sigmas(
    schedule: ScheduleType,
    num_steps: int,
    num_train_timesteps: int = 1000,
    beta_start: float = 0.00085,
    beta_end: float = 0.012,
) -> torch.Tensor:
    """Return a σ sequence of length num_steps + 1 (last element is 0).

    The trailing 0 acts as the termination sentinel for samplers.

    Parameters
    ----------
    schedule:
        "linear" | "scaled_linear" | "cosine" | "karras" | "exponential"
    num_steps:
        Number of denoising steps requested.
    num_train_timesteps:
        Size of the discrete β schedule used during model training (1000 for SD).
    """
    if schedule in ("linear", "scaled_linear", "cosine"):
        if schedule == "linear":
            betas = _linear_betas(num_train_timesteps, beta_start, beta_end)
        elif schedule == "scaled_linear":
            betas = _scaled_linear_betas(num_train_timesteps, beta_start, beta_end)
        else:
            betas = _cosine_betas(num_train_timesteps)

        all_sigmas = _betas_to_sigmas(betas)
        # Subsample to num_steps steps, evenly spaced across the full table
        indices = torch.linspace(0, num_train_timesteps - 1, num_steps).long()
        sigmas = all_sigmas[indices].flip(0).float()  # descending

    elif schedule == "karras":
        # Derive sigma_min/max from the training schedule
        betas = _scaled_linear_betas(num_train_timesteps, beta_start, beta_end)
        all_sigmas = _betas_to_sigmas(betas).float()
        sigma_min = float(all_sigmas[all_sigmas > 0].min())
        sigma_max = float(all_sigmas.max())
        sigmas = karras_sigmas(num_steps, sigma_min=sigma_min, sigma_max=sigma_max)

    elif schedule == "exponential":
        betas = _scaled_linear_betas(num_train_timesteps, beta_start, beta_end)
        all_sigmas = _betas_to_sigmas(betas).float()
        sigma_min = float(all_sigmas[all_sigmas > 0].min())
        sigma_max = float(all_sigmas.max())
        sigmas = exponential_sigmas(num_steps, sigma_min=sigma_min, sigma_max=sigma_max)

    else:
        raise ValueError(f"Unknown schedule: {schedule!r}")

    # Append the sentinel zero
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def sigma_to_timestep(sigma: torch.Tensor, all_sigmas: torch.Tensor) -> torch.Tensor:
    """Map σ values to integer timestep indices via nearest-neighbour lookup.

    UNet forward passes expect a discrete timestep tensor, not a float sigma.
    This inverts the σ table to recover the closest training timestep.
    """
    log_sigma = sigma.log()
    log_all = all_sigmas.log()
    dists = (log_sigma.unsqueeze(-1) - log_all.unsqueeze(0)).abs()
    return dists.argmin(dim=-1).long()
