from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def test_vram_warn_threshold_default():
    from aiwf.infrastructure.torch.wan_vram import vram_warn_threshold_gb

    assert vram_warn_threshold_gb() == pytest.approx(15.2)


def test_wan_inference_context_is_usable():
    from aiwf.infrastructure.torch.wan_vram import wan_inference_context

    with wan_inference_context():
        pass


def test_denormalize_wan_latents_shape():
    torch = pytest.importorskip("torch")
    from aiwf.infrastructure.torch.wan_vram import denormalize_wan_latents

    vae = MagicMock()
    vae.dtype = torch.float32
    vae.config.z_dim = 2
    vae.config.latents_mean = [0.0, 1.0]
    vae.config.latents_std = [1.0, 2.0]

    latents = torch.zeros(1, 2, 3, 4, 4)
    out = denormalize_wan_latents(vae, latents)
    assert out.shape == latents.shape
    assert out.dtype == torch.float32