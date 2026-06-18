"""Mock-based structural tests for AIWF's native Wan dual-stage denoise loop.

No GPU is available in this environment, so these tests cannot validate
generation quality. Instead they validate the contract the loop must honour:
correct boundary-ratio timestep switching between transformer/transformer_2,
correct classifier-free-guidance combination math, correct scheduler.step
call order, that the lazy stage-swap proxy is triggered by normal attribute
access (not reimplemented), and that AIWFFP8Linear-based metrics collection
sees zero fallback calls when fed the same transformer objects the loop uses.

Real-hardware validation (true Wan generation quality, fp8_fallback_calls==0
on actual FP8 weights, a 720x720/81-frame benchmark) is explicitly out of
scope here and remains a follow-up per docs/WAN_STANDALONE_RUNTIME_PLAN.md.
"""
from __future__ import annotations

from contextlib import contextmanager

import pytest


class _Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeTransformer:
    """Stands in for pipe.transformer / pipe.transformer_2.

    Records every cache_context entry and forward call so tests can assert
    exactly which stage handled which timestep.
    """

    def __init__(self, name, dtype, patch_size=(1, 2, 2), image_dim=None, output_value=0.0, uncond_output_value=None):
        self.name = name
        self.dtype = dtype
        self.config = _Config(patch_size=patch_size, image_dim=image_dim)
        self.output_value = output_value
        self.uncond_output_value = output_value if uncond_output_value is None else uncond_output_value
        self.cache_context_calls: list[str] = []
        self.forward_calls: list[str] = []
        self.loaded = True  # overridden by _LazyFakeTransformer

    @contextmanager
    def cache_context(self, mode: str):
        self._ensure_loaded()
        self.cache_context_calls.append(mode)
        yield

    def _ensure_loaded(self):
        pass

    def __call__(self, *, hidden_states, timestep, encoder_hidden_states, encoder_hidden_states_image, attention_kwargs, return_dict):
        import torch

        mode = self.cache_context_calls[-1]
        self.forward_calls.append(mode)
        value = self.output_value if mode == "cond" else self.uncond_output_value
        return (torch.full_like(hidden_states[:, :4], value),)


class _LazyFakeTransformer(_FakeTransformer):
    """Mimics _LazyWanTransformer: cache_context triggers load-on-first-use."""

    def __init__(self, *args, on_load=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loaded = False
        self._on_load = on_load

    def _ensure_loaded(self):
        if not self.loaded:
            self.loaded = True
            if self._on_load is not None:
                self._on_load()


class _DeviceAwareLazyTransformer(_FakeTransformer):
    """Lazy stage double that records device placement before cache_context."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loaded = False
        self.ensure_devices: list[str] = []
        self.cache_devices: list[str] = []

    def _ensure_loaded(self, device=None):
        device_text = "" if device is None else str(device)
        self.ensure_devices.append(device_text)
        self.loaded = True
        return self

    @contextmanager
    def cache_context(self, mode: str):
        self.cache_devices.append(self.ensure_devices[-1] if self.ensure_devices else "")
        self.cache_context_calls.append(mode)
        yield


class _FakeScheduler:
    def __init__(self, timesteps, num_train_timesteps=1000):
        import torch

        self._timesteps = torch.tensor([float(t) for t in timesteps])
        self.config = _Config(num_train_timesteps=num_train_timesteps)
        self.timesteps = self._timesteps
        self.step_calls: list[tuple[float, "object"]] = []

    def set_timesteps(self, num_inference_steps, device=None):
        self.timesteps = self._timesteps

    def step(self, noise_pred, t, latents, return_dict=False):
        self.step_calls.append((float(t), noise_pred.clone()))
        # Identity step keeps tensors simple/deterministic for assertions.
        return (latents,)


class _FakeVideoProcessor:
    def preprocess(self, image, height, width):
        import torch

        return torch.zeros((1, 3, 1, height, width))

    def postprocess_video(self, video, output_type):
        return video


class _FakeVAE:
    def __init__(self, z_dim=16):
        import torch

        self.config = _Config(z_dim=z_dim, latents_mean=[0.0] * z_dim, latents_std=[1.0] * z_dim)
        self.dtype = torch.float32

    def decode(self, latents, return_dict=False):
        return (latents,)


class _FakePipe:
    def __init__(self, transformer, transformer_2, scheduler, boundary_ratio, expand_timesteps=False):
        import torch

        self.transformer = transformer
        self.transformer_2 = transformer_2
        self.scheduler = scheduler
        self.vae = _FakeVAE()
        self.video_processor = _FakeVideoProcessor()
        self.config = _Config(boundary_ratio=boundary_ratio, expand_timesteps=expand_timesteps)
        self.vae_scale_factor_temporal = 4
        self.vae_scale_factor_spatial = 8
        self._execution_device = torch.device("cpu")
        self._interrupt = False
        self.do_classifier_free_guidance = True
        self.encode_prompt_calls = 0

    @property
    def interrupt(self):
        return self._interrupt

    def encode_prompt(self, *, prompt, negative_prompt, do_classifier_free_guidance, num_videos_per_prompt, prompt_embeds, negative_prompt_embeds, max_sequence_length, device):
        import torch

        self.encode_prompt_calls += 1
        return torch.ones(1, 4), torch.zeros(1, 4)

    def encode_image(self, image, device):
        raise AssertionError("encode_image should not be called when image_dim is None")

    def prepare_latents(self, image, batch_size, num_channels_latents, height, width, num_frames, dtype, device, generator, latents, last_image):
        import torch

        return torch.zeros(batch_size, 4), torch.zeros(batch_size, 4)

    def maybe_free_model_hooks(self):
        pass


def _make_pipe(boundary_ratio=0.5, timesteps=(1000.0, 800.0, 400.0, 100.0), low_on_load=None):
    torch = pytest.importorskip("torch")
    high = _FakeTransformer("high", torch.float32, output_value=1.0)
    low = _LazyFakeTransformer("low", torch.float32, output_value=3.0, on_load=low_on_load)
    scheduler = _FakeScheduler(timesteps)
    pipe = _FakePipe(high, low, scheduler, boundary_ratio=boundary_ratio)
    return pipe, high, low, scheduler


def test_boundary_ratio_switches_between_high_and_low_stage():
    """t=1000,800 should hit transformer (high); t=400,100 should hit transformer_2 (low)."""
    pytest.importorskip("torch")
    from aiwf.infrastructure.wan.native.denoise import run_native_wan_denoise

    pipe, high, low, scheduler = _make_pipe()

    output = run_native_wan_denoise(
        pipe,
        image=object(),
        prompt="a cat",
        negative_prompt="",
        height=64,
        width=64,
        num_frames=9,
        num_inference_steps=4,
        guidance_scale=2.0,
        output_type="latent",
    )

    assert output.frames is not None
    # 2 cond + 2 uncond calls per stage (4 timesteps total, half/half).
    assert high.forward_calls == ["cond", "uncond", "cond", "uncond"]
    assert low.forward_calls == ["cond", "uncond", "cond", "uncond"]
    assert [t for t, _ in scheduler.step_calls] == [1000.0, 800.0, 400.0, 100.0]


def test_lazy_low_stage_proxy_loads_only_at_boundary_crossing():
    """The denoise loop must not reimplement stage-swap logic -- it should
    fire automatically the first time transformer_2 is touched, and not
    before."""
    pytest.importorskip("torch")
    from aiwf.infrastructure.wan.native.denoise import run_native_wan_denoise

    load_events = []
    pipe, high, low, scheduler = _make_pipe(low_on_load=lambda: load_events.append("low_loaded"))

    assert low.loaded is False

    run_native_wan_denoise(
        pipe,
        image=object(),
        prompt="a cat",
        negative_prompt="",
        height=64,
        width=64,
        num_frames=9,
        num_inference_steps=4,
        guidance_scale=2.0,
        output_type="latent",
    )

    assert low.loaded is True
    assert load_events == ["low_loaded"]


def test_lazy_low_stage_is_ready_on_device_before_cache_context():
    """The real low-stage proxy must move/materialize on the denoise device
    before entering the wrapped transformer's cache_context. If placement
    happens inside forward instead, the first low step hides swap time and can
    run under stale high-stage CUDA pressure."""
    torch = pytest.importorskip("torch")
    from aiwf.infrastructure.wan.native.denoise import run_native_wan_denoise

    high = _FakeTransformer("high", torch.float32, output_value=1.0)
    low = _DeviceAwareLazyTransformer("low", torch.float32, output_value=3.0)
    scheduler = _FakeScheduler((1000.0, 800.0, 400.0, 100.0))
    pipe = _FakePipe(high, low, scheduler, boundary_ratio=0.5)

    run_native_wan_denoise(
        pipe,
        image=object(),
        prompt="a cat",
        negative_prompt="",
        height=64,
        width=64,
        num_frames=9,
        num_inference_steps=4,
        guidance_scale=2.0,
        output_type="latent",
    )

    assert low.ensure_devices
    assert low.ensure_devices[0] == "cpu"
    assert low.cache_devices
    assert low.cache_devices[0] == "cpu"


def test_classifier_free_guidance_math_matches_diffusers_formula():
    """noise_pred = uncond + scale * (cond - uncond)."""
    torch = pytest.importorskip("torch")
    from aiwf.infrastructure.wan.native.denoise import run_native_wan_denoise

    # Single-step, single-stage (boundary_ratio=None -> always "high").
    high = _FakeTransformer("high", torch.float32, output_value=5.0, uncond_output_value=1.0)
    low = _FakeTransformer("low", torch.float32, output_value=0.0)
    scheduler = _FakeScheduler((500.0,))
    pipe = _FakePipe(high, low, scheduler, boundary_ratio=None)

    guidance_scale = 3.0
    run_native_wan_denoise(
        pipe,
        image=object(),
        prompt="a cat",
        negative_prompt="",
        height=64,
        width=64,
        num_frames=9,
        num_inference_steps=1,
        guidance_scale=guidance_scale,
        output_type="latent",
    )

    expected = 1.0 + guidance_scale * (5.0 - 1.0)  # uncond + scale*(cond-uncond) = 13.0
    _, noise_pred = scheduler.step_calls[0]
    assert torch.allclose(noise_pred, torch.full_like(noise_pred, expected))


def test_image_dim_none_skips_image_embed_encoding():
    """Wan 2.2's dual-stage 14B transformers have image_dim=None; encode_image
    must not be called (the fake pipe raises if it is)."""
    pytest.importorskip("torch")
    from aiwf.infrastructure.wan.native.denoise import run_native_wan_denoise

    pipe, high, low, scheduler = _make_pipe()
    assert high.config.image_dim is None

    # Should not raise (encode_image would raise AssertionError if invoked).
    run_native_wan_denoise(
        pipe,
        image=object(),
        prompt="a cat",
        negative_prompt="",
        height=64,
        width=64,
        num_frames=9,
        num_inference_steps=4,
        guidance_scale=2.0,
        output_type="latent",
    )


def test_unknown_kwargs_are_accepted_and_ignored():
    """call_kwargs from pipeline.py may include forward-compatible keys like
    image_guidance_scale that this diffusers build doesn't support either;
    the native loop must not raise on them."""
    pytest.importorskip("torch")
    from aiwf.infrastructure.wan.native.denoise import run_native_wan_denoise

    pipe, high, low, scheduler = _make_pipe()

    run_native_wan_denoise(
        pipe,
        image=object(),
        prompt="a cat",
        negative_prompt="",
        height=64,
        width=64,
        num_frames=9,
        num_inference_steps=4,
        guidance_scale=2.0,
        output_type="latent",
        image_guidance_scale=1.5,
        return_dict=True,
    )


def test_fp8_fallback_metrics_are_zero_for_non_fp8_transformers():
    """collect_fp8_linear_metrics must report fp8_fallback_calls == 0 when
    fed the same transformer objects a native loop would use and none of
    them contain AIWFFP8Linear layers (the non-quantized BF16 path)."""
    torch = pytest.importorskip("torch")
    from aiwf.infrastructure.quant.fp8_linear import collect_fp8_linear_metrics

    class _PlainModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 8)

    transformer = _PlainModule()
    transformer_2 = _PlainModule()

    metrics = collect_fp8_linear_metrics(transformer, transformer_2, None)

    assert metrics["fp8_fallback_calls"] == 0
    assert metrics["fp8_linear_layers"] == 0


def test_fp8_fallback_metrics_are_zero_when_fp8_layers_have_no_fallbacks():
    """When real AIWFFP8Linear layers are present but none recorded a
    fallback, fp8_fallback_calls must still be 0 (the success case Pass 7
    needs to prove on real hardware)."""
    torch = pytest.importorskip("torch")
    from aiwf.infrastructure.quant.fp8_linear import AIWFFP8Linear, collect_fp8_linear_metrics

    class _FP8Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = AIWFFP8Linear(8, 8)

    transformer = _FP8Module()
    transformer.proj.fast_mm_calls = 10
    transformer.proj.fallback_calls = 0

    metrics = collect_fp8_linear_metrics(transformer, None, None)

    assert metrics["fp8_linear_layers"] == 1
    assert metrics["fp8_fallback_calls"] == 0
    assert metrics["fp8_fallback_layers"] == 0


class _GradRecordingTransformer(_FakeTransformer):
    """Records torch.is_grad_enabled() and the hidden_states channel count
    on every forward, so tests can assert the loop's inference guards."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_flags: list[bool] = []
        self.in_channels: list[int] = []

    def __call__(self, *, hidden_states, timestep, encoder_hidden_states, encoder_hidden_states_image, attention_kwargs, return_dict):
        import torch

        self.grad_flags.append(torch.is_grad_enabled())
        self.in_channels.append(int(hidden_states.shape[1]) if hidden_states.dim() > 1 else -1)
        return super().__call__(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_image=encoder_hidden_states_image,
            attention_kwargs=attention_kwargs,
            return_dict=return_dict,
        )


def test_native_denoise_runs_under_no_grad():
    """AIWF bypasses ``pipe.__call__`` to own the loop, which also bypasses
    diffusers' ``@torch.no_grad()`` decorator. The loop MUST re-establish that
    guard: without it ``prepare_latents``' VAE encode plus every denoise step
    build an autograd graph that OOMs real 14B 720x720/81-frame runs even
    though parameters are frozen (``model.eval()`` does not disable grad).
    """
    torch = pytest.importorskip("torch")
    from aiwf.infrastructure.wan.native.denoise import run_native_wan_denoise

    high = _GradRecordingTransformer("high", torch.float32, output_value=1.0)
    low = _GradRecordingTransformer("low", torch.float32, output_value=3.0)
    scheduler = _FakeScheduler((1000.0, 800.0, 400.0, 100.0))
    pipe = _FakePipe(high, low, scheduler, boundary_ratio=0.5)

    # Sanity: grad is enabled at the call site; the loop must turn it off.
    assert torch.is_grad_enabled() is True

    output = run_native_wan_denoise(
        pipe,
        image=object(),
        prompt="a cat",
        negative_prompt="",
        height=64,
        width=64,
        num_frames=9,
        num_inference_steps=4,
        guidance_scale=2.0,
        output_type="latent",
    )

    recorded = high.grad_flags + low.grad_flags
    assert recorded, "no transformer forwards were recorded"
    assert all(flag is False for flag in recorded), "denoise ran with autograd enabled (OOM risk)"
    assert getattr(output.frames, "requires_grad", False) is False
    # Caller's grad state must be restored after the loop returns.
    assert torch.is_grad_enabled() is True


def test_latent_model_input_concatenates_condition_channels():
    """Non-expand path must feed ``cat([latents, condition], dim=1)`` to the
    transformer. With 16 latent channels + 20 condition channels that is the
    36-channel Wan I2V input; this catches a regression in concat order or
    channel geometry that a CPU smoke test would otherwise miss."""
    torch = pytest.importorskip("torch")
    from aiwf.infrastructure.wan.native.denoise import run_native_wan_denoise

    high = _GradRecordingTransformer("high", torch.float32, output_value=1.0)
    low = _GradRecordingTransformer("low", torch.float32, output_value=3.0)
    scheduler = _FakeScheduler((1000.0, 800.0, 400.0, 100.0))
    pipe = _FakePipe(high, low, scheduler, boundary_ratio=0.5)

    def _prep(image, batch_size, num_channels_latents, height, width, num_frames, dtype, device, generator, latents, last_image):
        return torch.zeros(batch_size, 16, 3, 8, 8), torch.zeros(batch_size, 20, 3, 8, 8)

    pipe.prepare_latents = _prep

    run_native_wan_denoise(
        pipe,
        image=object(),
        prompt="a cat",
        negative_prompt="",
        height=64,
        width=64,
        num_frames=9,
        num_inference_steps=4,
        guidance_scale=2.0,
        output_type="latent",
    )

    channels = high.in_channels + low.in_channels
    assert channels, "no transformer forwards were recorded"
    assert all(c == 36 for c in channels), f"expected 36-channel input, saw {sorted(set(channels))}"
