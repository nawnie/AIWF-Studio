from __future__ import annotations

import pytest


def test_temporal_chunks_disabled_by_default(monkeypatch):
    from aiwf.infrastructure.wan.sliced_sampler import temporal_chunks_enabled

    monkeypatch.delenv("AIWF_WAN_TEMPORAL_CHUNKS", raising=False)

    assert temporal_chunks_enabled() is False


def test_temporal_chunks_can_be_enabled_by_env(monkeypatch):
    from aiwf.infrastructure.wan.sliced_sampler import temporal_chunks_enabled

    monkeypatch.setenv("AIWF_WAN_TEMPORAL_CHUNKS", "1")

    assert temporal_chunks_enabled() is True


def test_should_slice_only_when_frames_exceed_chunk():
    torch = pytest.importorskip("torch")
    from aiwf.infrastructure.wan.sliced_sampler import WanTemporalChunkCoordinator

    coord = WanTemporalChunkCoordinator(chunk_size=16, overlap=4)
    small = torch.zeros(1, 8, 12, 4, 4)
    large = torch.zeros(1, 8, 33, 4, 4)
    assert not coord.should_slice(small)
    assert coord.should_slice(large)


def test_sliced_forward_matches_full_when_single_chunk():
    torch = pytest.importorskip("torch")
    from aiwf.infrastructure.wan.sliced_sampler import WanTemporalChunkCoordinator

    def fake_forward(*, hidden_states, **_kwargs):
        return (hidden_states * 0.5,)

    coord = WanTemporalChunkCoordinator(chunk_size=16, overlap=4)
    x = torch.ones(1, 4, 10, 2, 2)
    out = coord.sliced_forward(
        fake_forward,
        hidden_states=x,
        timestep=torch.tensor([1]),
        encoder_hidden_states=torch.zeros(1, 3, 8),
        return_dict=False,
    )[0]
    assert torch.allclose(out, x * 0.5)


def test_install_temporal_chunk_forward_wraps_module():
    torch = pytest.importorskip("torch")

    class DummyTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = type("Cfg", (), {"patch_size": (1, 2, 2)})()

        def forward(self, hidden_states, **_kwargs):
            return (hidden_states + 1.0,)

    from aiwf.infrastructure.wan.sliced_sampler import install_temporal_chunk_forward

    model = DummyTransformer()
    assert install_temporal_chunk_forward(model, name="test", enabled=True)
    assert getattr(model, "_aiwf_temporal_chunks", False)
    x = torch.zeros(1, 2, 20, 4, 4)
    y = model.forward(
        hidden_states=x,
        timestep=torch.tensor([0]),
        encoder_hidden_states=torch.zeros(1, 1, 4),
        return_dict=False,
    )[0]
    assert y.shape == x.shape


def test_temporal_chunk_count_estimates_latent_splits():
    from aiwf.infrastructure.wan.sliced_sampler import (
        estimate_temporal_chunk_count,
        latent_frame_count_for_output_frames,
    )

    assert latent_frame_count_for_output_frames(81) == 21
    assert estimate_temporal_chunk_count(21, chunk_size=16, overlap=8, enabled=True) == 2
    assert estimate_temporal_chunk_count(21, chunk_size=24, overlap=0, enabled=True) == 1
    assert estimate_temporal_chunk_count(21, chunk_size=16, overlap=8, enabled=False) == 1
