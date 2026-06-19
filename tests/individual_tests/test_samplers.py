"""tests/test_samplers.py — custom sampler math tests (no GPU, no diffusers)."""
from __future__ import annotations

import math
import pytest
import torch

from aiwf.infrastructure.samplers.schedule import (
    get_sigmas, karras_sigmas, exponential_sigmas, sigma_to_timestep,
    _betas_to_sigmas, _scaled_linear_betas,
)
from aiwf.infrastructure.samplers.euler import (
    euler_sampler, euler_ancestral_sampler, epsilon_to_x0, v_pred_to_x0, SamplerStep,
)
from aiwf.infrastructure.samplers.dpmpp import (
    dpmpp_2m_sampler, dpmpp_sde_sampler, dpmpp_3m_sde_sampler,
)
from aiwf.infrastructure.samplers.ddim import ddim_sampler
from aiwf.infrastructure.samplers.dispatch import run_sampler, available_samplers


# ---------------------------------------------------------------------------
# Dummy denoiser: identity (returns x unchanged — sigma-scaled)
# ---------------------------------------------------------------------------

def _identity_denoiser(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Denoiser that predicts x0 = 0 (predicts all noise)."""
    return torch.zeros_like(x)


def _echo_denoiser(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Denoiser that returns x/σ — drives toward 0."""
    return x / (sigma + 1e-8)


# ---------------------------------------------------------------------------
# Schedule tests
# ---------------------------------------------------------------------------

class TestGetSigmas:
    @pytest.mark.parametrize("schedule", ["linear", "scaled_linear", "cosine", "karras", "exponential"])
    def test_returns_correct_length(self, schedule: str) -> None:
        sigmas = get_sigmas(schedule, num_steps=20)
        assert len(sigmas) == 21  # num_steps + 1 sentinel

    @pytest.mark.parametrize("schedule", ["linear", "scaled_linear", "cosine", "karras", "exponential"])
    def test_last_element_is_zero(self, schedule: str) -> None:
        sigmas = get_sigmas(schedule, num_steps=10)
        assert sigmas[-1].item() == pytest.approx(0.0)

    @pytest.mark.parametrize("schedule", ["linear", "scaled_linear", "cosine", "karras", "exponential"])
    def test_all_positive_except_last(self, schedule: str) -> None:
        sigmas = get_sigmas(schedule, num_steps=10)
        assert (sigmas[:-1] > 0).all()

    def test_karras_descending(self) -> None:
        sigmas = get_sigmas("karras", num_steps=20)
        # All non-sentinel elements should be descending
        body = sigmas[:-1]
        assert (body[:-1] >= body[1:]).all()

    def test_unknown_schedule_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown schedule"):
            get_sigmas("unicorn", num_steps=10)  # type: ignore


class TestKarrasSigmas:
    def test_length(self) -> None:
        assert len(karras_sigmas(20)) == 20

    def test_descending(self) -> None:
        s = karras_sigmas(20)
        assert (s[:-1] >= s[1:]).all()

    def test_endpoints(self) -> None:
        s = karras_sigmas(20, sigma_min=0.1, sigma_max=10.0)
        assert s[0].item() == pytest.approx(10.0, rel=1e-3)
        assert s[-1].item() == pytest.approx(0.1, rel=1e-3)


class TestSigmaToTimestep:
    def test_roundtrip(self) -> None:
        betas = _scaled_linear_betas(1000)
        all_sigmas = _betas_to_sigmas(betas).float()
        sigma = all_sigmas[500]
        t = sigma_to_timestep(sigma.unsqueeze(0), all_sigmas)
        assert abs(int(t[0].item()) - 500) <= 2  # nearest neighbour


# ---------------------------------------------------------------------------
# Euler sampler tests
# ---------------------------------------------------------------------------

class TestEulerSampler:
    def test_yields_n_steps(self) -> None:
        sigmas = get_sigmas("karras", num_steps=10)
        x = torch.randn(1, 4, 8, 8)
        steps = list(euler_sampler(_identity_denoiser, x, sigmas))
        assert len(steps) == 10

    def test_step_fields(self) -> None:
        sigmas = get_sigmas("karras", num_steps=5)
        x = torch.randn(1, 4, 8, 8)
        for step in euler_sampler(_identity_denoiser, x, sigmas):
            assert isinstance(step, SamplerStep)
            assert step.x.shape == x.shape
            assert step.denoised.shape == x.shape

    def test_step_numbers_correct(self) -> None:
        n = 7
        sigmas = get_sigmas("karras", num_steps=n)
        x = torch.randn(1, 4, 8, 8)
        steps = list(euler_sampler(_identity_denoiser, x, sigmas))
        assert [s.step for s in steps] == list(range(1, n + 1))
        assert steps[-1].total == n

    def test_sampler_modifies_input(self) -> None:
        sigmas = get_sigmas("karras", num_steps=10)
        x = torch.ones(1, 4, 4, 4) * 5.0
        x_orig = x.clone()
        for step in euler_sampler(_identity_denoiser, x, sigmas):
            x = step.x
        # The sampler must have moved x
        assert not torch.allclose(x, x_orig)


class TestEulerAncestral:
    def test_yields_n_steps(self) -> None:
        sigmas = get_sigmas("karras", num_steps=10)
        x = torch.randn(1, 4, 8, 8)
        steps = list(euler_ancestral_sampler(_identity_denoiser, x, sigmas))
        assert len(steps) == 10

    def test_stochastic_differs_with_different_seeds(self) -> None:
        # Use a midpoint step (not the last) — noise shrinks to zero at sigma_next=0
        sigmas = get_sigmas("karras", num_steps=10)
        x = torch.zeros(1, 4, 8, 8)
        g1 = torch.Generator().manual_seed(42)
        g2 = torch.Generator().manual_seed(99)
        steps_1 = list(euler_ancestral_sampler(_identity_denoiser, x.clone(), sigmas, generator=g1))
        steps_2 = list(euler_ancestral_sampler(_identity_denoiser, x.clone(), sigmas, generator=g2))
        # Check an early step where noise injection is large
        early_1 = steps_1[3].x
        early_2 = steps_2[3].x
        assert not torch.allclose(early_1, early_2, atol=1e-4), (
            "Expected different trajectories with different seeds"
        )

    def test_eta_zero_matches_euler(self) -> None:
        sigmas = get_sigmas("karras", num_steps=10)
        x = torch.randn(1, 4, 4, 4)
        euler_steps = list(euler_sampler(_identity_denoiser, x.clone(), sigmas))
        anc_steps = list(euler_ancestral_sampler(_identity_denoiser, x.clone(), sigmas, eta=0.0))
        assert torch.allclose(euler_steps[-1].x, anc_steps[-1].x, atol=1e-5)


# ---------------------------------------------------------------------------
# Epsilon / v prediction conversion
# ---------------------------------------------------------------------------

class TestPredictionConversions:
    def test_epsilon_to_x0_round_trip(self) -> None:
        x0 = torch.randn(1, 4, 8, 8)
        sigma = torch.tensor(1.5)
        eps = torch.randn_like(x0)
        x_t = x0 + sigma * eps
        x0_recovered = epsilon_to_x0(x_t, eps, sigma)
        assert torch.allclose(x0_recovered, x0, atol=1e-5)

    def test_v_pred_output_shape(self) -> None:
        x_t = torch.randn(1, 4, 8, 8)
        v   = torch.randn_like(x_t)
        sigma = torch.tensor(0.8)
        x0 = v_pred_to_x0(x_t, v, sigma)
        assert x0.shape == x_t.shape


# ---------------------------------------------------------------------------
# DPM++ samplers
# ---------------------------------------------------------------------------

class TestDPMPP2M:
    def test_yields_n_steps(self) -> None:
        sigmas = get_sigmas("karras", num_steps=15)
        x = torch.randn(1, 4, 8, 8)
        steps = list(dpmpp_2m_sampler(_identity_denoiser, x, sigmas))
        assert len(steps) == 15

    def test_deterministic(self) -> None:
        sigmas = get_sigmas("karras", num_steps=10)
        x = torch.randn(1, 4, 4, 4)
        r1 = list(dpmpp_2m_sampler(_identity_denoiser, x.clone(), sigmas))[-1].x
        r2 = list(dpmpp_2m_sampler(_identity_denoiser, x.clone(), sigmas))[-1].x
        assert torch.allclose(r1, r2)


class TestDPMPPSDE:
    def test_yields_n_steps(self) -> None:
        sigmas = get_sigmas("karras", num_steps=10)
        x = torch.randn(1, 4, 8, 8)
        steps = list(dpmpp_sde_sampler(_identity_denoiser, x, sigmas))
        assert len(steps) == 10

    def test_eta_zero_is_deterministic(self) -> None:
        sigmas = get_sigmas("karras", num_steps=8)
        x = torch.randn(1, 4, 4, 4)
        r1 = list(dpmpp_sde_sampler(_identity_denoiser, x.clone(), sigmas, eta=0.0))[-1].x
        r2 = list(dpmpp_sde_sampler(_identity_denoiser, x.clone(), sigmas, eta=0.0))[-1].x
        assert torch.allclose(r1, r2)


class TestDPMPP3MSDE:
    def test_yields_n_steps(self) -> None:
        sigmas = get_sigmas("karras", num_steps=20)
        x = torch.randn(1, 4, 8, 8)
        steps = list(dpmpp_3m_sde_sampler(_identity_denoiser, x, sigmas))
        assert len(steps) == 20


# ---------------------------------------------------------------------------
# DDIM
# ---------------------------------------------------------------------------

class TestDDIM:
    def test_yields_n_steps(self) -> None:
        sigmas = get_sigmas("scaled_linear", num_steps=10)
        x = torch.randn(1, 4, 8, 8)
        steps = list(ddim_sampler(_identity_denoiser, x, sigmas))
        assert len(steps) == 10

    def test_eta_zero_deterministic(self) -> None:
        sigmas = get_sigmas("scaled_linear", num_steps=8)
        x = torch.randn(1, 4, 4, 4)
        r1 = list(ddim_sampler(_identity_denoiser, x.clone(), sigmas, eta=0.0))[-1].x
        r2 = list(ddim_sampler(_identity_denoiser, x.clone(), sigmas, eta=0.0))[-1].x
        assert torch.allclose(r1, r2)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

class TestDispatch:
    @pytest.mark.parametrize("sampler_id", [
        "euler", "euler_a", "dpmpp_2m", "dpmpp_2m_karras", "dpmpp_sde",
        "dpmpp_3m_sde", "ddim",
    ])
    def test_dispatch_runs(self, sampler_id: str) -> None:
        sigmas = get_sigmas("karras", num_steps=5)
        x = torch.randn(1, 4, 4, 4)
        steps = list(run_sampler(sampler_id, _identity_denoiser, x, sigmas))
        assert len(steps) == 5

    def test_unknown_sampler_falls_back(self) -> None:
        sigmas = get_sigmas("karras", num_steps=3)
        x = torch.randn(1, 4, 4, 4)
        steps = list(run_sampler("nonexistent_sampler", _identity_denoiser, x, sigmas))
        assert len(steps) == 3

    def test_available_samplers_nonempty(self) -> None:
        ids = available_samplers()
        assert len(ids) > 5
        assert "euler" in ids
        assert "dpmpp_2m" in ids
