from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from aiwf.infrastructure.torch.attention import apply_image_pipeline_optimizations


class _TinyConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(4, 4, 3, padding=1)


class _TinyVae(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(4, 4, 3, padding=1)

    def decode(self, value):
        return value


class _Pipe:
    def __init__(self) -> None:
        self.unet = _TinyConv()
        self.vae = _TinyVae()


def _flags(**kwargs):
    values = {
        "channels_last": False,
        "torch_compile": False,
        "xformers": False,
        "opt_sdp_attention": False,
        "opt_split_attention": False,
    }
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_image_channels_last_is_flag_gated():
    pipe = _Pipe()

    apply_image_pipeline_optimizations(pipe, _flags(channels_last=False))

    assert not pipe.unet.conv.weight.is_contiguous(memory_format=torch.channels_last)
    assert not pipe.vae.conv.weight.is_contiguous(memory_format=torch.channels_last)

    apply_image_pipeline_optimizations(pipe, _flags(channels_last=True))

    assert pipe.unet.conv.weight.is_contiguous(memory_format=torch.channels_last)
    assert pipe.vae.conv.weight.is_contiguous(memory_format=torch.channels_last)


def test_image_compile_skips_when_offload_expected(monkeypatch):
    pipe = _Pipe()
    calls = []

    def fake_compile(module, **kwargs):
        calls.append((module, kwargs))
        return module

    monkeypatch.setattr(torch, "compile", fake_compile, raising=False)

    apply_image_pipeline_optimizations(
        pipe,
        _flags(torch_compile=True),
        compile_allowed=False,
    )

    assert calls == []


def test_image_compile_updates_unet_and_vae_decode(monkeypatch):
    pipe = _Pipe()
    compiled = []

    def fake_compile(module, **kwargs):
        compiled.append(module)

        def wrapped(*args, **inner_kwargs):
            return module(*args, **inner_kwargs)

        return wrapped

    monkeypatch.setattr(torch, "compile", fake_compile, raising=False)

    apply_image_pipeline_optimizations(
        pipe,
        _flags(torch_compile=True),
        compile_allowed=True,
    )

    assert len(compiled) == 2
    assert pipe.unet is not compiled[0]
    assert callable(pipe.vae.decode)
