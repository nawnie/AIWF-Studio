from __future__ import annotations

import pytest


def test_temporal_chunks_enabled_by_default():
    from aiwf.infrastructure.wan.sliced_sampler import temporal_chunks_enabled

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
    assert install_temporal_chunk_forward(model, name="test")
    assert getattr(model, "_aiwf_temporal_chunks", False)
    x = torch.zeros(1, 2, 20, 4, 4)
    y = model.forward(
        hidden_states=x,
        timestep=torch.tensor([0]),
        encoder_hidden_states=torch.zeros(1, 1, 4),
        return_dict=False,
    )[0]
    assert y.shape == x.shape