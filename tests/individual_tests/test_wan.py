import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.wan import (
    WAN_TI2V_5B,
    WAN_RUNTIME_FAST_5B,
    WAN_RUNTIME_HIGH_LOW,
    WanI2VRequest,
    duration_seconds_for_frames,
    frames_for_duration_seconds,
    snap_num_frames,
)
from aiwf.infrastructure.wan.pipeline import (
    WanI2VBackend,
    WanUnavailable,
    _apply_wan_transformer_key_renames,
    _boundary_ratio_for_step_split,
    _dequantize_comfy_fp8_state_dict,
    _ensure_wan_attention_processors,
    _fp8_scaled_mm_failure_payload,
    _frames_from_wan_pipeline_output,
    _install_group_offload_for_stage,
    _install_sequential_cpu_offload_for_stage,
    _load_comfy_fp8_transformer_weights,
    _load_umt5_text_encoder,
    _load_wan_vae,
    _new_fp8_scaled_linear,
    _new_lazy_wan_transformer,
    _new_wan_euler_simple_scheduler,
    _orient_umt5_gguf_tensor,
    _call_accepts_kwarg,
    _collect_fp8_linear_metrics,
    _cuda_supports_tensorcore_fp8,
    _resolve_dual_stage_offload_for_hardware,
    _wan_output_type_for_pipe,
    _wan_cache_mode,
    estimate_gguf_expanded_gb,
)
from aiwf.infrastructure.video import VideoError
from aiwf.services.wan import WanService, wan_model_pair_compatibility


def _svc(tmp_path: Path) -> WanService:
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "out")
    return WanService(flags, UserSettings())


def _force_wan_available(service: WanService) -> None:
    service._backend.available = lambda: True


def _write_component_base(service: WanService) -> Path:
    base = service.models_dir() / "Diffusers" / "Wan2.2-TI2V-5B-Diffusers"
    (base / "text_encoder").mkdir(parents=True)
    (base / "tokenizer").mkdir()
    (base / "scheduler").mkdir()
    (base / "model_index.json").write_text("{}", encoding="utf-8")
    (base / "text_encoder" / "config.json").write_text("{}", encoding="utf-8")
    (base / "text_encoder" / "model.safetensors").write_bytes(b"fake")
    (base / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    (base / "scheduler" / "scheduler_config.json").write_text("{}", encoding="utf-8")
    return base


def _write_fake_safetensors(path: Path) -> None:
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    path.parent.mkdir(parents=True, exist_ok=True)
    safetensors.save_file({"blocks.0.weight": torch.ones(1)}, path)


def _write_fake_gguf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")


def _write_fake_video_frames(frames, output_path, *, fps: float) -> int:
    usable = [frame for frame in frames if frame is not None]
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"fake mp4")
    return len(usable)


def test_snap_num_frames():
    assert snap_num_frames(49) == 49
    assert snap_num_frames(50) == 49
    assert snap_num_frames(51) == 49  # nearest 4k+1 (banker's rounding)
    assert snap_num_frames(54) == 53
    assert snap_num_frames(1) == 5
    assert snap_num_frames(80) == 81


def test_duration_frame_helpers():
    assert frames_for_duration_seconds(16, 3) == 49
    assert frames_for_duration_seconds(16, 10) == 161
    assert duration_seconds_for_frames(49, 16) == 3.0


def test_request_defaults_and_helpers():
    r = WanI2VRequest()
    assert r.model_id == WAN_TI2V_5B
    assert r.runtime_mode == WAN_RUNTIME_FAST_5B
    assert r.fps == 16 and r.offload == "balanced"
    assert r.temporal_chunks is False
    assert r.chunk_size == 24
    assert r.chunk_overlap == 0
    assert r.vram_reserve_enabled is False
    assert r.vram_reserve_mb == 1536
    assert r.guidance_scale == 5.0
    assert r.sigma_type == "simple"
    assert r.flow_shift == 8.0
    assert r.normalized_frames() == 81
    assert r.effective_steps() == 20
    assert r.effective_boundary_ratio() == 1.0
    assert WanI2VRequest(width=512, height=320).max_area == 512 * 320


def test_request_accepts_streamed_offload():
    r = WanI2VRequest(offload="streamed")

    assert r.offload == "streamed"


def test_request_accepts_balanced_and_resident_offload():
    assert WanI2VRequest(offload="balanced").offload == "balanced"
    assert WanI2VRequest(offload="resident").offload == "resident"


def test_call_accepts_kwarg_handles_explicit_and_kwargs():
    def explicit(*, image_guidance_scale=1.0):
        pass

    def arbitrary(**kwargs):
        pass

    def missing(*, guidance_scale=1.0):
        pass

    assert _call_accepts_kwarg(explicit, "image_guidance_scale") is True
    assert _call_accepts_kwarg(arbitrary, "image_guidance_scale") is True
    assert _call_accepts_kwarg(missing, "image_guidance_scale") is False


def test_wan_output_type_uses_pil_when_decode_hook_is_missing():
    class WithDecode:
        def decode_latents(self):
            pass

    class WithoutDecode:
        pass

    assert _wan_output_type_for_pipe(WithDecode()) == "latent"
    assert _wan_output_type_for_pipe(WithoutDecode()) == "pil"


def test_boundary_ratio_for_step_split_maps_half_steps():
    scheduler_mod = pytest.importorskip("diffusers.schedulers.scheduling_unipc_multistep")
    scheduler = scheduler_mod.UniPCMultistepScheduler(
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        flow_shift=8.0,
    )

    ratio = _boundary_ratio_for_step_split(scheduler, total_steps=20, high_steps=10)

    assert 0.0 < ratio < 1.0


def test_estimate_gguf_expanded_gb_scales_file_size(tmp_path: Path):
    gguf_file = tmp_path / "tiny.gguf"
    gguf_file.write_bytes(b"x" * (2 * 1024 * 1024 * 1024))  # 2 GiB
    est = estimate_gguf_expanded_gb(gguf_file)
    assert 8.0 < est < 12.0


def test_detect_transformer_format_gguf(tmp_path: Path):
    from aiwf.infrastructure.wan.transformer_runtime import (
        WanTransformerFormat,
        detect_transformer_format,
    )

    gguf_file = tmp_path / "wan_high.gguf"
    gguf_file.write_bytes(b"fake")
    assert detect_transformer_format(gguf_file) == WanTransformerFormat.GGUF_QUANTIZED


def test_gguf_allowed_with_quantized_runtime(tmp_path: Path, monkeypatch):
    gguf = pytest.importorskip("gguf")
    from aiwf.infrastructure.wan.transformer_runtime import (
        WanTransformerFormat,
        require_diffusers_transformer_path,
    )

    gguf_file = tmp_path / "wan_high.gguf"
    gguf_file.write_bytes(b"fake")
    monkeypatch.delenv("AIWF_WAN_ALLOW_EXPENSIVE_DEQUANT", raising=False)
    monkeypatch.delenv("AIWF_WAN_GGUF_RUNTIME", raising=False)

    fmt = require_diffusers_transformer_path(gguf_file, label="High-noise transformer")
    assert fmt == WanTransformerFormat.GGUF_QUANTIZED


def test_wan_cache_mode_prefers_gpu_swap_for_model_offload_fp8():
    from aiwf.infrastructure.wan.native.memory import WanStageCacheMode

    assert _wan_cache_mode("sequential", fast_fp8_pair=True) == "none"
    assert _wan_cache_mode("group", fast_fp8_pair=True) == "none"
    assert _wan_cache_mode("balanced", fast_fp8_pair=False) == "none"
    assert (
        _wan_cache_mode("model", fast_fp8_pair=True)
        == WanStageCacheMode.GPU_ACTIVE_CPU_PINNED_STANDBY.value
    )
    assert (
        _wan_cache_mode("balanced", fast_fp8_pair=True)
        == WanStageCacheMode.GPU_ACTIVE_CPU_PINNED_STANDBY.value
    )
    assert (
        _wan_cache_mode("resident", fast_fp8_pair=True)
        == WanStageCacheMode.DUAL_GPU_RESIDENT.value
    )
    assert _wan_cache_mode("resident", fast_fp8_pair=False) == "none"
    assert (
        _wan_cache_mode("model", fast_fp8_pair=True, pinned_memory=False)
        == WanStageCacheMode.GPU_ACTIVE_CPU_UNPINNED_STANDBY.value
    )
    assert _wan_cache_mode("model", fast_fp8_pair=False) == "none"
    assert _wan_cache_mode("none", fast_fp8_pair=False) == "full"


def test_resident_offload_downgrades_on_16gb_gpu(monkeypatch):
    monkeypatch.delenv("AIWF_WAN_FORCE_RESIDENT", raising=False)
    monkeypatch.delenv("AIWF_WAN_RESIDENT_MIN_VRAM_MB", raising=False)

    with patch("aiwf.infrastructure.wan.pipeline._cuda_total_vram_mb", return_value=16376):
        assert _resolve_dual_stage_offload_for_hardware("resident", fast_fp8_pair=True) == "balanced"


def test_resident_offload_can_be_forced_for_research(monkeypatch):
    monkeypatch.setenv("AIWF_WAN_FORCE_RESIDENT", "1")

    with patch("aiwf.infrastructure.wan.pipeline._cuda_total_vram_mb", return_value=16376):
        assert _resolve_dual_stage_offload_for_hardware("resident", fast_fp8_pair=True) == "resident"


def test_wan_pair_check_allows_different_creator_names_with_same_type_and_quant():
    check = wan_model_pair_compatibility(
        "creator_a_wan_i2v_q4_high.gguf",
        "different_name_wan_i2v_low_q4.gguf",
    )

    assert check.ok is True
    assert not check.errors


def test_wan_pair_check_blocks_file_type_and_quant_mismatch():
    mixed_type = wan_model_pair_compatibility(
        "wan_i2v_high_q4.gguf",
        "wan_i2v_low_q4.safetensors",
    )
    mixed_quant = wan_model_pair_compatibility(
        "wan_i2v_high_q5.gguf",
        "wan_i2v_low_q4.gguf",
    )

    assert mixed_type.ok is False
    assert any("different storage formats" in error for error in mixed_type.errors)
    assert mixed_quant.ok is False
    assert any("different quantization tiers" in error for error in mixed_quant.errors)


def test_wan_pair_check_blocks_large_size_mismatch(tmp_path: Path):
    high = tmp_path / "wan_high_q4.gguf"
    low = tmp_path / "wan_low_q4.gguf"
    high.write_bytes(b"x" * 1024)
    low.write_bytes(b"x" * 2048)

    check = wan_model_pair_compatibility(str(high), str(low))

    assert check.ok is False
    assert any("file sizes differ too much" in error for error in check.errors)


def test_wan_latent_pipeline_output_is_decoded_to_frames():
    torch = pytest.importorskip("torch")
    from PIL import Image

    latents = torch.zeros(1, 16, 5, 4, 4)
    calls = []

    def decode_latents(pipe, value, **kwargs):
        calls.append((pipe, tuple(value.shape), kwargs))
        return [[Image.new("RGB", (8, 8), "black"), Image.new("RGB", (8, 8), "white")]]

    pipe = object()
    frames = _frames_from_wan_pipeline_output(latents, pipe=pipe, decode_latents=decode_latents)

    assert len(frames) == 2
    assert frames[0].size == (8, 8)
    assert calls == [(pipe, (1, 16, 5, 4, 4), {"output_type": "pil"})]


def test_wan_chunked_vae_decode_preserves_temporal_frame_count():
    torch = pytest.importorskip("torch")
    from aiwf.infrastructure.torch.wan_vram import decode_wan_video_latents

    class FakeVAE:
        def __init__(self):
            self.dtype = torch.float32
            self.config = SimpleNamespace(
                z_dim=16,
                latents_mean=[0.0] * 16,
                latents_std=[1.0] * 16,
            )
            self.decode_latent_sizes: list[int] = []

        def enable_tiling(self):
            pass

        def enable_slicing(self):
            pass

        def decode(self, latents, return_dict=False):
            latent_frames = int(latents.shape[2])
            self.decode_latent_sizes.append(latent_frames)
            video_frames = 1 + 4 * (latent_frames - 1)
            video = torch.zeros(
                latents.shape[0],
                3,
                video_frames,
                latents.shape[3],
                latents.shape[4],
                dtype=latents.dtype,
                device=latents.device,
            )
            return (video,)

    class FakeVideoProcessor:
        def postprocess_video(self, video, output_type):
            return video

    pipe = SimpleNamespace(vae=FakeVAE(), video_processor=FakeVideoProcessor())
    latents = torch.zeros(1, 16, 21, 2, 2)

    video = decode_wan_video_latents(pipe, latents, chunk_frames=4, output_type="tensor")

    assert int(video.shape[2]) == 81
    assert pipe.vae.decode_latent_sizes == [4, 5, 5, 5, 5, 2]


def test_wan_scheduler_defaults_to_euler_simple():
    scheduler_mod = pytest.importorskip("diffusers.schedulers.scheduling_unipc_multistep")
    from diffusers import FlowMatchEulerDiscreteScheduler

    base = scheduler_mod.UniPCMultistepScheduler(
        prediction_type="flow_prediction",
        use_flow_sigmas=True,
        flow_shift=8.0,
        num_train_timesteps=1000,
        time_shift_type="exponential",
    )

    scheduler = _new_wan_euler_simple_scheduler(base, flow_shift=5.0)

    assert isinstance(scheduler, FlowMatchEulerDiscreteScheduler)
    assert scheduler.config.shift == 5.0
    assert scheduler.config.use_karras_sigmas is False
    assert scheduler.config.use_exponential_sigmas is False
    assert scheduler.config.use_beta_sigmas is False


def test_ensure_wan_attention_processors_resets_generic_processor():
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    from diffusers.models.attention_processor import AttnProcessor2_0
    from diffusers.models.transformers.transformer_wan import WanAttention, WanAttnProcessor

    attn = WanAttention(dim=16, heads=2, dim_head=8, processor=AttnProcessor2_0())
    wrapper = torch.nn.Module()
    wrapper.block = torch.nn.Module()
    wrapper.block.attn1 = attn

    _ensure_wan_attention_processors(wrapper, "test")

    assert isinstance(wrapper.block.attn1.processor, WanAttnProcessor)
    assert wrapper.block.attn1.spatial_norm is None


def test_lazy_wan_transformer_has_offload_sentinel_parameter():
    torch = pytest.importorskip("torch")
    lazy = _new_lazy_wan_transformer(
        {"patch_size": [1, 2, 2]},
        dtype=torch.bfloat16,
        load_model=lambda: None,
    )

    first = next(lazy.parameters())

    assert first.numel() == 1
    assert first.dtype == torch.bfloat16


def test_sequential_offload_helper_treats_existing_hook_as_safe():
    class AlreadyHooked:
        _hf_hook = object()

    assert _install_sequential_cpu_offload_for_stage(AlreadyHooked(), "cuda:0") is True


def test_sequential_offload_helper_rejects_non_module():
    assert _install_sequential_cpu_offload_for_stage(object(), "cuda:0") is False


def test_group_offload_helper_can_force_streamed_one_block():
    captured = {}

    class GroupOffloadTarget:
        def enable_group_offload(self, **kwargs):
            captured.update(kwargs)

    target = GroupOffloadTarget()

    assert _install_group_offload_for_stage(
        target,
        "cuda:0",
        blocks=1,
        use_stream=True,
        record_stream=True,
        low_cpu_mem_usage=False,
    )
    assert captured["offload_type"] == "block_level"
    assert captured["num_blocks_per_group"] == 1
    assert captured["use_stream"] is True
    assert captured["record_stream"] is True
    assert captured["low_cpu_mem_usage"] is False
    assert getattr(target, "_aiwf_group_offload_stream") is True


def test_resolve_model_hf_default(tmp_path: Path):
    s = _svc(tmp_path)
    assert s.resolve_model(None) == WAN_TI2V_5B
    assert s.resolve_model("Wan-AI/Something-Else") == "Wan-AI/Something-Else"


def test_resolve_and_list_local_model(tmp_path: Path):
    s = _svc(tmp_path)
    d = s.models_dir() / "my-wan"
    d.mkdir(parents=True)
    (d / "model_index.json").write_text("{}", encoding="utf-8")
    assert s.resolve_model("my-wan") == str(d)
    assert "my-wan" in s.list_local_models()

    # Standalone safetensors / gguf (Comfy or GGUF quant style) are also discoverable and resolvable
    (s.models_dir() / "wan2.2_test_i2v.safetensors").write_bytes(b"fake")
    (s.models_dir() / "some_wan_ti2v_Q4.gguf").write_bytes(b"fake")
    models = s.list_local_models()
    assert "wan2.2_test_i2v.safetensors" in models
    assert "some_wan_ti2v_Q4.gguf" in models
    assert str(s.models_dir() / "wan2.2_test_i2v.safetensors") == s.resolve_model("wan2.2_test_i2v.safetensors")
    assert str(s.models_dir() / "some_wan_ti2v_Q4.gguf") == s.resolve_model("some_wan_ti2v_Q4.gguf")


def test_new_wan_folder_layout(tmp_path: Path):
    s = _svc(tmp_path)
    diffusers = s.models_dir() / "Diffusers" / "Wan-Full"
    gguf = s.models_dir() / "GGUF" / "wan2.2_i2v_high_noise_test.gguf"
    misplaced_safetensor = s.models_dir() / "GGUF" / "wan2.2_i2v_wrong_bucket.safetensors"
    safetensor = s.models_dir() / "Safetensor" / "wan2.2_i2v_low_noise_test.safetensors"
    misplaced_gguf = s.models_dir() / "Safetensor" / "wan2.2_i2v_wrong_bucket.gguf"
    diffusers_file = s.models_dir() / "Diffusers" / "wan2.2_i2v_not_a_weight.safetensors"
    lora = s.models_dir() / "lora" / "wan2.2_i2v_high_lora_test.safetensors"
    diffusers.mkdir(parents=True)
    gguf.parent.mkdir(parents=True)
    safetensor.parent.mkdir(parents=True)
    lora.parent.mkdir(parents=True)
    (diffusers / "model_index.json").write_text("{}", encoding="utf-8")
    gguf.write_bytes(b"fake")
    misplaced_safetensor.write_bytes(b"fake")
    safetensor.write_bytes(b"fake")
    misplaced_gguf.write_bytes(b"fake")
    diffusers_file.write_bytes(b"fake")
    lora.write_bytes(b"fake")

    models = s.list_local_models()
    assert "Diffusers/Wan-Full" in models
    assert "GGUF/wan2.2_i2v_high_noise_test.gguf" in models
    assert "Safetensor/wan2.2_i2v_low_noise_test.safetensors" in models
    assert "Diffusers/wan2.2_i2v_not_a_weight.safetensors" not in models
    assert "GGUF/wan2.2_i2v_wrong_bucket.safetensors" not in models
    assert "Safetensor/wan2.2_i2v_wrong_bucket.gguf" not in models
    assert "lora/wan2.2_i2v_high_lora_test.safetensors" not in models
    assert s.resolve_model("Diffusers/Wan-Full") == str(diffusers)
    assert s.resolve_model("GGUF/wan2.2_i2v_high_noise_test.gguf") == str(gguf)
    assert s.resolve_model("wan2.2_i2v_low_noise_test.safetensors") == str(safetensor.resolve())
    assert s.resolve_model("Diffusers/wan2.2_i2v_not_a_weight.safetensors") == "Diffusers/wan2.2_i2v_not_a_weight.safetensors"
    assert s.resolve_model("GGUF/wan2.2_i2v_wrong_bucket.safetensors") == "GGUF/wan2.2_i2v_wrong_bucket.safetensors"
    assert s.resolve_model("Safetensor/wan2.2_i2v_wrong_bucket.gguf") == "Safetensor/wan2.2_i2v_wrong_bucket.gguf"
    assert "wan2.2_i2v_high_lora_test.safetensors" in s.list_local_loras("high")
    assert s.resolve_lora("wan2.2_i2v_high_lora_test.safetensors") == str(lora.resolve())


def test_find_components_base_uses_diffusers_folder(tmp_path: Path):
    s = _svc(tmp_path)
    base = _write_component_base(s)

    assert s.find_components_base() == str(base.resolve())


def test_empty_components_base_is_not_valid(tmp_path: Path):
    s = _svc(tmp_path)
    base = s.models_dir() / "Diffusers" / "Wan2.2-TI2V-5B-Diffusers"
    (base / "text_encoder").mkdir(parents=True)
    (base / "tokenizer").mkdir()
    (base / "scheduler").mkdir()
    (base / "model_index.json").write_text("{}", encoding="utf-8")

    assert s.find_components_base() is None


def test_wan_preflight_passes_with_local_hybrid_components(tmp_path: Path):
    s = _svc(tmp_path)
    _force_wan_available(s)
    _write_component_base(s)
    high = s.models_dir() / "GGUF" / "wan-high-q4.gguf"
    low = s.models_dir() / "GGUF" / "wan-low-q4.gguf"
    vae = s.flags.resolved_models_dir() / "VAE" / "wan2.1_vae.safetensors"
    _write_fake_gguf(high)
    _write_fake_gguf(low)
    _write_fake_safetensors(vae)

    result = s.preflight(
        WanI2VRequest(
            runtime_mode=WAN_RUNTIME_HIGH_LOW,
            high_noise_model_id=f"GGUF/{high.name}",
            low_noise_model_id=f"GGUF/{low.name}",
        )
    )

    assert result.ok, result.message()
    assert result.components_base is not None
    assert result.high_noise_model == str(high.resolve())
    assert result.low_noise_model == str(low.resolve())
    assert result.vae == str(vae.resolve())


def test_wan_preflight_blocks_mismatched_quant_pair(tmp_path: Path):
    s = _svc(tmp_path)
    _force_wan_available(s)
    _write_component_base(s)
    high = s.models_dir() / "Safetensor" / "maker_a_high_fp8.safetensors"
    low = s.models_dir() / "Safetensor" / "maker_b_low_fp16.safetensors"
    vae = s.flags.resolved_models_dir() / "VAE" / "wan2.1_vae.safetensors"
    _write_fake_safetensors(high)
    _write_fake_safetensors(low)
    _write_fake_safetensors(vae)

    result = s.preflight(
        WanI2VRequest(runtime_mode=WAN_RUNTIME_HIGH_LOW, high_noise_model_id=high.name, low_noise_model_id=low.name)
    )

    assert not result.ok
    assert "different quantization tiers" in result.message()


def test_wan_preflight_blocks_missing_component_base(tmp_path: Path):
    s = _svc(tmp_path)
    _force_wan_available(s)
    high = s.models_dir() / "GGUF" / "wan-high-q4.gguf"
    low = s.models_dir() / "GGUF" / "wan-low-q4.gguf"
    vae = s.flags.resolved_models_dir() / "VAE" / "wan2.1_vae.safetensors"
    _write_fake_gguf(high)
    _write_fake_gguf(low)
    _write_fake_safetensors(vae)

    result = s.preflight(
        WanI2VRequest(runtime_mode=WAN_RUNTIME_HIGH_LOW, high_noise_model_id=high.name, low_noise_model_id=low.name)
    )

    assert not result.ok
    assert "Missing local Wan component base" in result.message()
    assert "scheduler_config.json" in result.message()


def test_wan_generation_unloads_image_models_before_video_load(tmp_path: Path):
    from PIL import Image

    calls: list[str] = []
    s = _svc(tmp_path)
    s._unload_image_models = lambda: calls.append("unload")
    _force_wan_available(s)
    _write_component_base(s)
    high = s.models_dir() / "GGUF" / "wan-high-q4.gguf"
    low = s.models_dir() / "GGUF" / "wan-low-q4.gguf"
    vae = s.flags.resolved_models_dir() / "VAE" / "wan2.1_vae.safetensors"
    _write_fake_gguf(high)
    _write_fake_gguf(low)
    _write_fake_safetensors(vae)

    def fake_generate(*args, **kwargs):
        calls.append("video")
        frame = Image.new("RGB", (8, 8), "black")
        return [frame], 8, 8

    s._backend.generate = fake_generate

    s.generate(
        WanI2VRequest(runtime_mode=WAN_RUNTIME_HIGH_LOW, high_noise_model_id=high.name, low_noise_model_id=low.name),
        Image.new("RGB", (8, 8)),
    )

    assert calls[:2] == ["unload", "video"]


def test_wan_generation_unloads_video_backend_after_failure(tmp_path: Path):
    from PIL import Image

    calls: list[str] = []
    s = _svc(tmp_path)
    _force_wan_available(s)
    _write_component_base(s)
    high = s.models_dir() / "GGUF" / "wan-high-q4.gguf"
    low = s.models_dir() / "GGUF" / "wan-low-q4.gguf"
    vae = s.flags.resolved_models_dir() / "VAE" / "wan2.1_vae.safetensors"
    _write_fake_gguf(high)
    _write_fake_gguf(low)
    _write_fake_safetensors(vae)

    def fail_generate(*args, **kwargs):
        calls.append("generate")
        raise RuntimeError("Allocation on device")

    s._backend.generate = fail_generate
    s._backend.unload = lambda: calls.append("unload")

    with pytest.raises(WanUnavailable, match="Allocation on device"):
        s.generate(
            WanI2VRequest(runtime_mode=WAN_RUNTIME_HIGH_LOW, high_noise_model_id=high.name, low_noise_model_id=low.name),
            Image.new("RGB", (8, 8)),
        )

    assert calls == ["generate", "unload"]


def test_wan_generation_records_video_throughput(tmp_path: Path, monkeypatch):
    from PIL import Image

    s = _svc(tmp_path)
    _force_wan_available(s)
    _write_component_base(s)
    high = s.models_dir() / "GGUF" / "wan-high-q4.gguf"
    low = s.models_dir() / "GGUF" / "wan-low-q4.gguf"
    vae = s.flags.resolved_models_dir() / "VAE" / "wan2.1_vae.safetensors"
    _write_fake_gguf(high)
    _write_fake_gguf(low)
    _write_fake_safetensors(vae)

    s._backend.generate = lambda *args, **kwargs: (
        [Image.new("RGB", (8, 8), "black")] * 5,
        8,
        8,
        {
            "step_count": 8,
            "load_seconds": 1.0,
            "preprocess_seconds": 0.25,
            "prompt_encode_seconds": 0.4,
            "image_encode_seconds": 0.3,
            "latent_prepare_seconds": 0.6,
            "denoise_seconds": 2.0,
            "high_denoise_seconds": 0.75,
            "low_denoise_seconds": 1.25,
            "pipeline_seconds": 2.5,
            "pipeline_overhead_seconds": 0.5,
            "vae_decode_seconds": 0.7,
            "manual_vae_decode": True,
            "vae_decode_chunk_frames": 4,
            "latent_frame_count": 13,
            "temporal_chunks": False,
            "temporal_chunk_size": 0,
            "temporal_chunk_overlap": 0,
            "transformer_chunks_per_forward": 1,
            "transformer_forwards_per_step": 1,
            "video_postprocess_seconds": 0.1,
            "offload_cleanup_seconds": 0.2,
            "postprocess_seconds": 0.2,
            "steps_per_second": 4.0,
            "fp8_linear_layers": 12,
            "fp8_fast_mm_calls": 96,
            "fp8_fallback_calls": 0,
            "fp8_fallback_layers": 0,
            "fp8_fallback_reasons": [],
            "fp8_strict_mode": True,
            "fp8_native_available": True,
            "cache_mode": "gpu_active_cpu_pinned_standby",
            "vram_reserve_enabled": True,
            "vram_reserve_mb": 1536,
            "vram_limit_mb": 14848,
            "vram_total_mb": 16384,
            "vram_limit_fraction": 0.90625,
        },
    )
    captured: dict[str, object] = {}
    progress_events: list[tuple[int, int, object, str | None]] = []
    monkeypatch.setattr("aiwf.services.wan.trace_model_throughput", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr("aiwf.services.wan.write_frames", _write_fake_video_frames)

    result = s.generate(
        WanI2VRequest(runtime_mode=WAN_RUNTIME_HIGH_LOW, high_noise_model_id=high.name, low_noise_model_id=low.name),
        Image.new("RGB", (8, 8)),
        on_progress=lambda step, total, rate=None, message=None: progress_events.append(
            (step, total, rate, message)
        ),
    )

    assert result.frame_count == 5
    assert [event[3] for event in progress_events] == [
        "Loading models and encoding inputs",
        "Writing video file",
        "Video file saved",
    ]
    import aiwf
    assert captured["kind"] == "wan.video"
    assert captured["units_label"] == "frames"
    assert captured["units"] == 5
    assert captured["step_count"] == 8
    assert captured["load_seconds"] == 1.0
    assert captured["preprocess_seconds"] == 0.25
    assert captured["prompt_encode_seconds"] == 0.4
    assert captured["image_encode_seconds"] == 0.3
    assert captured["latent_prepare_seconds"] == 0.6
    assert captured["denoise_seconds"] == 2.0
    assert captured["high_denoise_seconds"] == 0.75
    assert captured["low_denoise_seconds"] == 1.25
    assert captured["pipeline_seconds"] == 2.5
    assert captured["pipeline_overhead_seconds"] == 0.5
    assert captured["vae_decode_seconds"] == 0.7
    assert captured["manual_vae_decode"] is True
    assert captured["vae_decode_chunk_frames"] == 4
    assert captured["latent_frame_count"] == 13
    assert captured["temporal_chunks"] is False
    assert captured["temporal_chunk_size"] == 0
    assert captured["temporal_chunk_overlap"] == 0
    assert captured["transformer_chunks_per_forward"] == 1
    assert captured["transformer_forwards_per_step"] == 1
    assert captured["video_postprocess_seconds"] == 0.1
    assert captured["offload_cleanup_seconds"] == 0.2
    assert captured["postprocess_seconds"] == 0.2
    assert captured["video_write_seconds"] >= 0.0
    assert captured["steps_per_second"] == 4.0
    assert captured["iterations_per_second"] == 4.0
    assert captured["fp8_linear_layers"] == 12
    assert captured["fp8_fast_mm_calls"] == 96
    assert captured["fp8_fallback_calls"] == 0
    assert captured["fp8_fallback_layers"] == 0
    assert captured["fp8_fallback_reasons"] == []
    assert captured["fp8_strict_mode"] is True
    assert captured["fp8_native_available"] is True
    assert captured["cache_mode"] == "gpu_active_cpu_pinned_standby"
    assert captured["vram_reserve_enabled"] is True
    assert captured["vram_reserve_mb"] == 1536
    assert captured["vram_limit_mb"] == 14848
    assert captured["vram_total_mb"] == 16384
    assert captured["vram_limit_fraction"] == 0.90625
    assert captured.get("app_version") == aiwf.__version__
    assert Path(str(captured["high_noise_model_id"])).name == high.name
    assert result.step_count == 8
    assert result.load_seconds == 1.0
    assert result.preprocess_seconds == 0.25
    assert result.prompt_encode_seconds == 0.4
    assert result.image_encode_seconds == 0.3
    assert result.latent_prepare_seconds == 0.6
    assert result.denoise_seconds == 2.0
    assert result.high_denoise_seconds == 0.75
    assert result.low_denoise_seconds == 1.25
    assert result.pipeline_seconds == 2.5
    assert result.pipeline_overhead_seconds == 0.5
    assert result.vae_decode_seconds == 0.7
    assert result.manual_vae_decode is True
    assert result.vae_decode_chunk_frames == 4
    assert result.latent_frame_count == 13
    assert result.temporal_chunks is False
    assert result.temporal_chunk_size == 0
    assert result.temporal_chunk_overlap == 0
    assert result.transformer_chunks_per_forward == 1
    assert result.transformer_forwards_per_step == 1
    assert result.video_postprocess_seconds == 0.1
    assert result.offload_cleanup_seconds == 0.2
    assert result.postprocess_seconds == 0.2
    assert result.video_write_seconds >= 0.0
    assert result.steps_per_second == 4.0
    assert result.iterations_per_second == 4.0
    assert result.fp8_linear_layers == 12
    assert result.fp8_fast_mm_calls == 96
    assert result.fp8_fallback_calls == 0
    assert result.fp8_fallback_layers == 0
    assert result.fp8_fallback_reasons == []
    assert result.fp8_strict_mode is True
    assert result.fp8_native_available is True
    assert result.cache_mode == "gpu_active_cpu_pinned_standby"
    assert result.vram_reserve_enabled is True
    assert result.vram_reserve_mb == 1536
    assert result.vram_limit_mb == 14848
    assert result.vram_total_mb == 16384
    assert result.vram_limit_fraction == 0.90625
    assert "4.000 it/s" in result.message
    assert "FP8 fast path clean" in result.message
    assert "latent=13f" in result.message
    assert "cache=gpu_active_cpu_pinned_standby" in result.message
    assert "VRAM cap=14848/16384 MB" in result.message
    assert "keep_free=1536 MB" in result.message


def test_wan_generation_fast_5b_uses_local_model_without_high_low(tmp_path: Path, monkeypatch):
    from PIL import Image

    s = _svc(tmp_path)
    _force_wan_available(s)
    monkeypatch.setattr(s, "_wan_file_candidates", lambda: [])
    _write_component_base(s)
    transformer = s.models_dir() / "Safetensor" / "wan2.2_ti2v_5B_fp16.safetensors"
    vae = s.flags.resolved_models_dir() / "VAE" / "wan2.2_vae.safetensors"
    _write_fake_safetensors(transformer)
    _write_fake_safetensors(vae)
    captured: dict[str, object] = {}

    def fake_generate(request, *_args, **_kwargs):
        captured["runtime_mode"] = request.runtime_mode
        captured["model_id"] = request.model_id
        captured["high"] = request.high_noise_model_id
        captured["low"] = request.low_noise_model_id
        return [Image.new("RGB", (8, 8), "black")] * 5, 8, 8, {"step_count": 6, "denoise_seconds": 3.0}

    s._backend.generate = fake_generate
    monkeypatch.setattr("aiwf.services.wan.write_frames", _write_fake_video_frames)

    result = s.generate(WanI2VRequest(steps=6), Image.new("RGB", (8, 8)))

    assert result.frame_count == 5
    assert Path(result.output_path).is_absolute()
    assert Path(result.output_path).is_file()
    assert Path(result.output_path).parent == s.output_dir()
    assert captured["runtime_mode"] == WAN_RUNTIME_FAST_5B
    assert captured["model_id"] == str(transformer.resolve())
    assert captured["high"] is None
    assert captured["low"] is None


def test_wan_generation_rejects_missing_encoded_video(tmp_path: Path, monkeypatch):
    from PIL import Image

    s = _svc(tmp_path)
    _force_wan_available(s)
    monkeypatch.setattr(s, "_wan_file_candidates", lambda: [])
    _write_component_base(s)
    transformer = s.models_dir() / "Safetensor" / "wan2.2_ti2v_5B_fp16.safetensors"
    vae = s.flags.resolved_models_dir() / "VAE" / "wan2.2_vae.safetensors"
    _write_fake_safetensors(transformer)
    _write_fake_safetensors(vae)

    s._backend.generate = lambda *args, **kwargs: ([Image.new("RGB", (8, 8), "black")], 8, 8)
    monkeypatch.setattr("aiwf.services.wan.write_frames", lambda frames, output_path, fps: 1)

    with pytest.raises(VideoError, match="did not create output file"):
        s.generate(WanI2VRequest(), Image.new("RGB", (8, 8)))


def test_wan_generation_passes_resolved_paths_to_backend(tmp_path: Path, monkeypatch):
    from PIL import Image

    s = _svc(tmp_path)
    _force_wan_available(s)
    base = _write_component_base(s)
    high = s.models_dir() / "GGUF" / "wan-high-q4.gguf"
    low = s.models_dir() / "GGUF" / "wan-low-q4.gguf"
    vae = s.flags.resolved_models_dir() / "VAE" / "wan2.1_vae.safetensors"
    text_encoder = s.flags.resolved_models_dir() / "Textencoder" / "nsfw_wan_umt5-xxl_fp8_scaled.safetensors"
    _write_fake_gguf(high)
    _write_fake_gguf(low)
    _write_fake_safetensors(vae)
    _write_fake_safetensors(text_encoder)
    captured: dict[str, object] = {}

    def fake_generate(request, *_args, **_kwargs):
        captured["high"] = request.high_noise_model_id
        captured["low"] = request.low_noise_model_id
        captured["vae"] = request.vae_id
        captured["text_encoder"] = request.text_encoder_path
        captured["components_base"] = request.components_base
        return [Image.new("RGB", (8, 8), "black")], 8, 8

    s._backend.generate = fake_generate
    monkeypatch.setattr("aiwf.services.wan.write_frames", _write_fake_video_frames)

    s.generate(
        WanI2VRequest(runtime_mode=WAN_RUNTIME_HIGH_LOW, high_noise_model_id=high.name, low_noise_model_id=low.name),
        Image.new("RGB", (8, 8)),
    )

    assert captured["high"] == str(high.resolve())
    assert captured["low"] == str(low.resolve())
    assert captured["vae"] == str(vae.resolve())
    assert captured["text_encoder"] == str(text_encoder.resolve())
    assert captured["components_base"] == str(base.resolve())


def test_preflight_rejects_lora_from_wrong_runtime_size(tmp_path: Path):
    s = _svc(tmp_path)
    _force_wan_available(s)
    _write_component_base(s)
    high = s.models_dir() / "GGUF" / "wan-a14b-high-q4.gguf"
    low = s.models_dir() / "GGUF" / "wan-a14b-low-q4.gguf"
    vae = s.flags.resolved_models_dir() / "VAE" / "wan2.1_vae.safetensors"
    lora = s.flags.resolved_models_dir() / "Loras" / "Wan" / "motion_5b_ti2v_rank16.safetensors"
    _write_fake_gguf(high)
    _write_fake_gguf(low)
    _write_fake_safetensors(vae)
    _write_fake_safetensors(lora)

    result = s.preflight(
        WanI2VRequest(
            runtime_mode=WAN_RUNTIME_HIGH_LOW,
            high_noise_model_id=high.name,
            low_noise_model_id=low.name,
            high_noise_lora_id=lora.name,
        )
    )

    assert result.ok is False
    assert any("does not match the 14B high/low runtime" in error for error in result.errors)


def test_ensure_components_base_is_local_only(tmp_path: Path, monkeypatch):
    s = _svc(tmp_path)

    def fail_snapshot_download(*args, **kwargs):
        raise AssertionError("generation must not download Wan components")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fail_snapshot_download)
    with pytest.raises(WanUnavailable, match="will not auto-download"):
        s.ensure_components_base()


def test_output_dir_layout(tmp_path: Path):
    s = _svc(tmp_path)
    assert s.output_dir().name == "wan"
    assert s.output_dir().parent.name == "video"


def test_list_local_vaes_prefers_wan_named_entries(tmp_path: Path):
    s = _svc(tmp_path)
    vae_root = s.flags.resolved_models_dir() / "VAE"
    vae_root.mkdir(parents=True)
    (vae_root / "ae.safetensors").write_bytes(b"fake")
    (vae_root / "wan2.1_vae.safetensors").write_bytes(b"fake")
    (vae_root / "wan2.2_vae.safetensors").write_bytes(b"fake")

    vaes = s.list_local_vaes()
    assert vaes[0] == "wan2.1_vae.safetensors"
    assert s.preferred_vae() == "wan2.1_vae.safetensors"
    assert s.preferred_vae(WAN_RUNTIME_FAST_5B) == "wan2.2_vae.safetensors"
    assert s.preferred_vae(WAN_RUNTIME_HIGH_LOW) == "wan2.1_vae.safetensors"


def test_load_wan_vae_does_not_treat_single_file_as_json_config(tmp_path: Path):
    bogus = tmp_path / "ae.safetensors"
    bogus.write_bytes(b"not-a-wan-vae")

    with patch("diffusers.AutoencoderKLWan.from_single_file", side_effect=RuntimeError("bad vae")), patch(
        "diffusers.AutoencoderKLWan.from_pretrained", side_effect=AssertionError("should not call from_pretrained")
    ):
        with pytest.raises(WanUnavailable, match="could not be loaded as a Wan VAE"):
            _load_wan_vae(str(bogus), torch_dtype=None)


def test_load_umt5_text_encoder_supports_single_file_layout(tmp_path: Path):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    transformers = pytest.importorskip("transformers")

    cfg = transformers.UMT5Config(
        d_model=4,
        d_ff=8,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=1,
        vocab_size=8,
        relative_attention_num_buckets=4,
        relative_attention_max_distance=8,
        feed_forward_proj="gated-gelu",
        tie_word_embeddings=False,
    )
    text_encoder_dir = tmp_path / "text_encoder"
    text_encoder_dir.mkdir()
    (text_encoder_dir / "config.json").write_text(cfg.to_json_string(), encoding="utf-8")

    model = transformers.UMT5EncoderModel(cfg)
    state = model.state_dict()
    state.pop("encoder.embed_tokens.weight", None)
    state["spiece_model"] = torch.zeros(2, dtype=torch.uint8)
    state["scaled_fp8"] = torch.zeros(1, dtype=torch.uint8)
    safetensors.save_file(state, text_encoder_dir / "model.safetensors")

    loaded = _load_umt5_text_encoder(text_encoder_dir, torch_dtype=torch.float32)

    assert tuple(loaded.shared.weight.shape) == (8, 4)
    assert tuple(loaded.encoder.embed_tokens.weight.shape) == (8, 4)


def test_orient_umt5_gguf_tensor_transposes_swapped_embedding_shape():
    torch = pytest.importorskip("torch")

    swapped = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    oriented = _orient_umt5_gguf_tensor("shared.weight", swapped, (8, 4))

    assert tuple(oriented.shape) == (8, 4)
    assert torch.equal(oriented, swapped.t())


def test_materialize_wan_transformer_passes_temporal_chunk_settings(tmp_path: Path):
    torch = pytest.importorskip("torch")
    target = torch.nn.Module()
    seen: list[tuple[int | None, int | None, bool | None]] = []

    def load_state_dict(_state_dict, *, strict=False, assign=False):
        return [], []

    target.load_state_dict = load_state_dict

    def capture_optimizations(_target, _label, *, chunk_size=None, chunk_overlap=None, temporal_chunks=None):
        seen.append((chunk_size, chunk_overlap, temporal_chunks))

    backend = WanI2VBackend()
    with patch("aiwf.infrastructure.wan.pipeline._safetensors_uses_comfy_fp8_quant", return_value=False), patch(
        "aiwf.infrastructure.wan.pipeline._load_transformer_state_dict",
        return_value={},
    ), patch("aiwf.infrastructure.wan.pipeline._apply_transformer_lora"), patch(
        "aiwf.infrastructure.wan.pipeline._ensure_wan_attention_processors"
    ), patch(
        "aiwf.infrastructure.wan.pipeline._apply_wan_attention_optimizations",
        side_effect=capture_optimizations,
    ):
        missing, unexpected = backend._materialize_wan_transformer(
            target,
            str(tmp_path / "low.safetensors"),
            label="low-noise transformer",
            lora_path=None,
            lora_scale=1.0,
            lora_adapter="wan_low_lora",
            chunk_size=12,
            chunk_overlap=3,
            temporal_chunks=True,
        )

    assert missing == []
    assert unexpected == []
    assert seen == [(12, 3, True)]


def test_low_preload_worker_passes_temporal_chunk_settings():
    torch = pytest.importorskip("torch")
    backend = WanI2VBackend()
    backend._low_preload_spec = {
        "low_path": "low.safetensors",
        "low_lora_path": None,
        "low_lora_scale": 1.0,
        "use_cache": False,
        "pin_tensors": False,
        "chunk_size": 20,
        "chunk_overlap": 6,
        "temporal_chunks": True,
    }
    seen: list[tuple[int | None, int | None, bool | None]] = []

    def fake_materialize(_target, _path, **kwargs):
        seen.append((kwargs.get("chunk_size"), kwargs.get("chunk_overlap"), kwargs.get("temporal_chunks")))
        return [], []

    with patch("aiwf.infrastructure.wan.pipeline._empty_wan_transformer", return_value=torch.nn.Module()):
        with patch.object(backend, "_materialize_wan_transformer", side_effect=fake_materialize):
            backend._run_low_preload_worker()

    assert backend._low_preload_error is None
    assert seen == [(20, 6, True)]


def test_prepare_low_preload_skips_duplicate_when_low_is_ready():
    backend = WanI2VBackend()
    backend._preloaded_low = object()
    backend._low_preload_spec = None
    started: list[bool] = []

    with patch.object(backend, "_maybe_start_background_low_preload", side_effect=lambda: started.append(True)):
        backend._prepare_low_preload_for_generation(
            SimpleNamespace(
                low_noise_model_id="low.safetensors",
                low_noise_lora_id=None,
                low_noise_lora_scale=1.0,
            ),
            chunk_size=16,
            chunk_overlap=8,
            temporal_chunks=False,
        )

    assert started == []
    assert backend._low_preload_spec is None
    assert backend._preloaded_low is not None


def test_prepare_low_preload_sets_spec_when_low_is_not_ready():
    backend = WanI2VBackend()
    started: list[bool] = []

    with patch.object(backend, "_maybe_start_background_low_preload", side_effect=lambda: started.append(True)):
        backend._prepare_low_preload_for_generation(
            SimpleNamespace(
                low_noise_model_id="low.safetensors",
                low_noise_lora_id="low-lora.safetensors",
                low_noise_lora_scale=0.75,
            ),
            chunk_size=20,
            chunk_overlap=6,
            temporal_chunks=True,
        )

    assert started == [True]
    assert backend._low_preload_spec == {
        "low_path": "low.safetensors",
        "low_lora_path": "low-lora.safetensors",
        "low_lora_scale": 0.75,
        "use_cache": False,
        "pin_tensors": False,
        "disk_sequential": False,
        "chunk_size": 20,
        "chunk_overlap": 6,
        "temporal_chunks": True,
    }


def test_prepare_low_preload_marks_unpinned_cpu_standby_cache():
    from aiwf.infrastructure.wan.native.memory import WanStageCacheMode

    backend = WanI2VBackend()
    backend._cache_mode = WanStageCacheMode.GPU_ACTIVE_CPU_UNPINNED_STANDBY.value
    started: list[bool] = []

    with patch.object(backend, "_maybe_start_background_low_preload", side_effect=lambda: started.append(True)):
        backend._prepare_low_preload_for_generation(
            SimpleNamespace(
                low_noise_model_id="low.safetensors",
                low_noise_lora_id=None,
                low_noise_lora_scale=1.0,
            ),
            chunk_size=16,
            chunk_overlap=8,
            temporal_chunks=False,
        )

    assert started == [True]
    assert backend._low_preload_spec["use_cache"] is True
    assert backend._low_preload_spec["pin_tensors"] is False
    assert backend._low_preload_spec["disk_sequential"] is False


def test_disk_sequential_low_preload_does_not_start_background_thread():
    from aiwf.infrastructure.wan.native.memory import WanStageCacheMode

    backend = WanI2VBackend()
    backend._cache_mode = WanStageCacheMode.DISK_SEQUENTIAL.value
    backend._prepare_low_preload_for_generation(
        SimpleNamespace(
            low_noise_model_id="low.safetensors",
            low_noise_lora_id=None,
            low_noise_lora_scale=1.0,
        ),
        chunk_size=16,
        chunk_overlap=8,
        temporal_chunks=False,
    )

    assert backend._low_preload_spec["use_cache"] is True
    assert backend._low_preload_spec["pin_tensors"] is False
    assert backend._low_preload_spec["disk_sequential"] is True
    assert backend._low_preload_started is False


def test_wan_pipeline_cache_key_includes_temporal_chunk_settings():
    torch = pytest.importorskip("torch")

    class DummyVae:
        def enable_tiling(self):
            pass

        def enable_slicing(self):
            pass

    class DummyPipe:
        def __init__(self) -> None:
            self.scheduler = object()
            self.transformer = torch.nn.Module()
            self.vae = DummyVae()

        def enable_model_cpu_offload(self):
            pass

    backend = WanI2VBackend()
    loads: list[tuple[int | None, int | None, bool | None]] = []

    def fake_load_dual_pipeline(**kwargs):
        loads.append((kwargs.get("chunk_size"), kwargs.get("chunk_overlap"), kwargs.get("temporal_chunks")))
        return DummyPipe()

    ensure_kwargs = dict(
        high_noise_model_id="high.safetensors",
        low_noise_model_id="low.safetensors",
        boundary_ratio=0.5,
        vae_id="wan_vae.safetensors",
        high_noise_lora_id=None,
        high_noise_lora_scale=1.0,
        low_noise_lora_id=None,
        low_noise_lora_scale=1.0,
        components_base="components",
        offload="model",
        flow_shift=5.0,
        sigma_type="beta",
        sampler="euler",
        text_encoder_path="",
    )

    with patch("aiwf.infrastructure.wan.pipeline._require_wan"), patch(
        "aiwf.infrastructure.wan.pipeline._is_native_comfy_fp8_transformer",
        return_value=False,
    ), patch("aiwf.infrastructure.wan.pipeline._is_gguf_transformer", return_value=False), patch(
        "aiwf.infrastructure.wan.pipeline._new_wan_euler_scheduler",
        side_effect=lambda scheduler, **_kwargs: scheduler,
    ), patch(
        "aiwf.infrastructure.wan.pipeline._ensure_wan_attention_processors"
    ), patch(
        "aiwf.infrastructure.wan.pipeline._apply_wan_attention_optimizations"
    ), patch(
        "aiwf.infrastructure.wan.pipeline._free_cuda_memory"
    ), patch.object(
        backend,
        "_load_dual_pipeline",
        side_effect=fake_load_dual_pipeline,
    ):
        first = backend._ensure(**ensure_kwargs, chunk_size=16, chunk_overlap=8, temporal_chunks=False)
        second = backend._ensure(**ensure_kwargs, chunk_size=16, chunk_overlap=8, temporal_chunks=False)
        third = backend._ensure(**ensure_kwargs, chunk_size=16, chunk_overlap=8, temporal_chunks=True)
        fourth = backend._ensure(**ensure_kwargs, chunk_size=20, chunk_overlap=8, temporal_chunks=True)
        fifth = backend._ensure(**ensure_kwargs, chunk_size=20, chunk_overlap=4, temporal_chunks=True)

    assert first is second
    assert third is not first
    assert fourth is not third
    assert fifth is not fourth
    assert loads == [(16, 8, False), (16, 8, True), (20, 8, True), (20, 4, True)]


def test_wan_dual_cache_hit_reconfigures_scheduler_without_reload():
    torch = pytest.importorskip("torch")

    class DummyVae:
        def enable_tiling(self):
            pass

        def enable_slicing(self):
            pass

    class DummyPipe:
        def __init__(self) -> None:
            self.scheduler = SimpleNamespace(
                kind="base",
                config=SimpleNamespace(
                    num_train_timesteps=1000,
                    shift=5.0,
                    use_dynamic_shifting=False,
                    time_shift_type="exponential",
                ),
            )
            self.transformer = torch.nn.Module()
            self.vae = DummyVae()
            self.sequential_calls = 0

        def enable_sequential_cpu_offload(self):
            self.sequential_calls += 1

    backend = WanI2VBackend()
    loads = []
    scheduler_updates = []

    def fake_load_dual_pipeline(**_kwargs):
        loads.append("load")
        return DummyPipe()

    def fake_scheduler(current, *, flow_shift, sigma_type):
        scheduler_updates.append((getattr(current, "kind", ""), sigma_type, flow_shift))
        return SimpleNamespace(
            kind=sigma_type,
            config=SimpleNamespace(
                num_train_timesteps=1000,
                shift=flow_shift,
                use_dynamic_shifting=False,
                time_shift_type="exponential",
            ),
        )

    ensure_kwargs = dict(
        high_noise_model_id="high.safetensors",
        low_noise_model_id="low.safetensors",
        boundary_ratio=0.5,
        vae_id="wan_vae.safetensors",
        high_noise_lora_id=None,
        high_noise_lora_scale=1.0,
        low_noise_lora_id=None,
        low_noise_lora_scale=1.0,
        components_base="components",
        offload="sequential",
        sampler="euler",
        text_encoder_path="",
        chunk_size=16,
        chunk_overlap=8,
        temporal_chunks=False,
    )

    with patch("aiwf.infrastructure.wan.pipeline._require_wan"), patch(
        "aiwf.infrastructure.wan.pipeline._is_native_comfy_fp8_transformer",
        return_value=False,
    ), patch("aiwf.infrastructure.wan.pipeline._is_gguf_transformer", return_value=False), patch(
        "aiwf.infrastructure.wan.pipeline._new_wan_euler_scheduler",
        side_effect=fake_scheduler,
    ), patch("aiwf.infrastructure.wan.pipeline._ensure_wan_attention_processors"), patch(
        "aiwf.infrastructure.wan.pipeline._apply_wan_attention_optimizations"
    ), patch("aiwf.infrastructure.wan.pipeline._free_cuda_memory"), patch.object(
        backend,
        "_load_dual_pipeline",
        side_effect=fake_load_dual_pipeline,
    ):
        first = backend._ensure(**ensure_kwargs, flow_shift=5.0, sigma_type="simple")
        second = backend._ensure(**ensure_kwargs, flow_shift=9.0, sigma_type="karras")

    assert first is second
    assert loads == ["load"]
    assert second.scheduler.kind == "karras"
    assert scheduler_updates == [("base", "simple", 5.0), ("simple", "karras", 9.0)]


def test_wan_single_5b_passes_temporal_chunk_settings_to_transformer():
    torch = pytest.importorskip("torch")

    class DummyVae:
        def enable_tiling(self):
            pass

        def enable_slicing(self):
            pass

    class DummyPipe:
        def __init__(self) -> None:
            self.scheduler = object()
            self.transformer = torch.nn.Module()
            self.vae = DummyVae()

        def enable_model_cpu_offload(self):
            pass

        def to(self, _device):
            return self

    backend = WanI2VBackend()
    loaded: list[str] = []
    seen: list[tuple[int | None, int | None, bool | None]] = []

    def fake_from_pretrained(model_id, **_kwargs):
        loaded.append(str(model_id))
        return DummyPipe()

    def capture_optimizations(_target, _label, *, chunk_size=None, chunk_overlap=None, temporal_chunks=None):
        seen.append((chunk_size, chunk_overlap, temporal_chunks))

    ensure_kwargs = dict(
        model_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        vae_id=None,
        components_base=None,
        text_encoder_path="",
        offload="model",
        flow_shift=5.0,
        sigma_type="beta",
        sampler="euler",
        high_noise_lora_id=None,
        high_noise_lora_scale=1.0,
    )

    with patch("aiwf.infrastructure.wan.pipeline._require_wan"), patch(
        "diffusers.WanImageToVideoPipeline.from_pretrained",
        side_effect=fake_from_pretrained,
    ), patch(
        "aiwf.infrastructure.wan.pipeline._new_wan_euler_scheduler",
        side_effect=lambda scheduler, **_kwargs: scheduler,
    ), patch(
        "aiwf.infrastructure.wan.pipeline._ensure_wan_attention_processors"
    ), patch(
        "aiwf.infrastructure.wan.pipeline._apply_wan_attention_optimizations",
        side_effect=capture_optimizations,
    ), patch(
        "aiwf.infrastructure.wan.pipeline._free_cuda_memory"
    ):
        first = backend._ensure_single_5b(
            **ensure_kwargs,
            chunk_size=16,
            chunk_overlap=8,
            temporal_chunks=True,
        )
        second = backend._ensure_single_5b(
            **ensure_kwargs,
            chunk_size=16,
            chunk_overlap=8,
            temporal_chunks=True,
        )
        third = backend._ensure_single_5b(
            **ensure_kwargs,
            chunk_size=16,
            chunk_overlap=4,
            temporal_chunks=True,
        )

    assert first is second
    assert third is not first
    assert loaded == ["Wan-AI/Wan2.2-TI2V-5B-Diffusers", "Wan-AI/Wan2.2-TI2V-5B-Diffusers"]
    assert seen == [(16, 8, True), (16, 4, True)]


def test_wan_single_5b_applies_lora_and_caches_by_lora():
    torch = pytest.importorskip("torch")

    class DummyPipe:
        def __init__(self) -> None:
            self.scheduler = object()
            self.transformer = torch.nn.Module()
            self.vae = None

        def enable_model_cpu_offload(self):
            pass

        def to(self, _device):
            return self

    backend = WanI2VBackend()
    applied: list[tuple[str | None, str, float]] = []

    def fake_apply(_transformer, lora_path, *, adapter_name, weight):
        applied.append((lora_path, adapter_name, weight))

    with patch("aiwf.infrastructure.wan.pipeline._require_wan"), patch(
        "diffusers.WanImageToVideoPipeline.from_pretrained",
        side_effect=lambda *_args, **_kwargs: DummyPipe(),
    ), patch(
        "aiwf.infrastructure.wan.pipeline._new_wan_euler_scheduler",
        side_effect=lambda scheduler, **_kwargs: scheduler,
    ), patch(
        "aiwf.infrastructure.wan.pipeline._ensure_wan_attention_processors"
    ), patch(
        "aiwf.infrastructure.wan.pipeline._apply_wan_attention_optimizations"
    ), patch(
        "aiwf.infrastructure.wan.pipeline._apply_transformer_lora",
        side_effect=fake_apply,
    ), patch(
        "aiwf.infrastructure.wan.pipeline._free_cuda_memory"
    ):
        first = backend._ensure_single_5b(
            model_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            vae_id=None,
            components_base=None,
            text_encoder_path="",
            offload="model",
            flow_shift=8.0,
            sigma_type="simple",
            sampler="euler",
            high_noise_lora_id="turbo.safetensors",
            high_noise_lora_scale=0.75,
        )
        second = backend._ensure_single_5b(
            model_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            vae_id=None,
            components_base=None,
            text_encoder_path="",
            offload="model",
            flow_shift=8.0,
            sigma_type="simple",
            sampler="euler",
            high_noise_lora_id="turbo.safetensors",
            high_noise_lora_scale=0.75,
        )
        third = backend._ensure_single_5b(
            model_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
            vae_id=None,
            components_base=None,
            text_encoder_path="",
            offload="model",
            flow_shift=8.0,
            sigma_type="simple",
            sampler="euler",
            high_noise_lora_id="other.safetensors",
            high_noise_lora_scale=0.75,
        )

    assert first is second
    assert third is not first
    assert applied == [
        ("turbo.safetensors", "wan_5b_lora", 0.75),
        ("other.safetensors", "wan_5b_lora", 0.75),
    ]


def test_dequantize_comfy_fp8_state_dict_scales_weights():
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch build has no float8_e4m3fn")

    scale = torch.tensor(0.5, dtype=torch.float32)
    source = torch.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=torch.float32)
    quantized = (source / scale).to(torch.float8_e4m3fn)
    state = {
        "blocks.0.self_attn.q.weight": quantized,
        "blocks.0.self_attn.q.weight_scale": scale,
        "blocks.0.self_attn.q.comfy_quant": torch.tensor(list(b'{"format":"float8_e4m3fn"}'), dtype=torch.uint8),
        "blocks.0.self_attn.q.bias": torch.tensor([1.0, 2.0], dtype=torch.float32),
    }

    converted = _dequantize_comfy_fp8_state_dict(state, torch_dtype=torch.bfloat16)

    assert "blocks.0.self_attn.q.weight_scale" not in converted
    assert "blocks.0.self_attn.q.comfy_quant" not in converted
    assert converted["blocks.0.self_attn.q.weight"].dtype == torch.bfloat16
    assert converted["blocks.0.self_attn.q.bias"].dtype == torch.bfloat16
    assert torch.allclose(converted["blocks.0.self_attn.q.weight"].float(), source, atol=0.02)


def test_load_comfy_fp8_transformer_weights_uses_native_scaled_linear(tmp_path: Path):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    if not torch.cuda.is_available() or not hasattr(torch, "float8_e4m3fn") or not hasattr(torch, "_scaled_mm"):
        pytest.skip("native CUDA FP8 runtime is unavailable")

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(16, 32)

    model = Tiny()
    scale = torch.tensor(0.125, dtype=torch.float32)
    weight = (torch.randn(32, 16) / scale).to(torch.float8_e4m3fn)
    bias = torch.randn(32, dtype=torch.float16)
    path = tmp_path / "tiny_fp8.safetensors"
    safetensors.save_file(
        {
            "linear.weight": weight,
            "linear.weight_scale": scale,
            "linear.bias": bias,
        },
        path,
    )

    missing, unexpected = _load_comfy_fp8_transformer_weights(model, path, torch_dtype=torch.bfloat16)
    assert not unexpected
    assert model.linear.__class__.__name__ == "FP8ScaledLinear"
    assert model.linear.weight.dtype == torch.float8_e4m3fn

    model = model.cuda()
    x = torch.randn(2, 16, device="cuda", dtype=torch.bfloat16)
    y = model.linear(x)
    assert y.shape == (2, 32)
    assert y.dtype == torch.bfloat16
    assert "linear.weight" not in missing


def test_fp8_scaled_linear_uses_column_major_weight_for_scaled_mm():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not hasattr(torch, "float8_e4m3fn") or not hasattr(torch, "_scaled_mm"):
        pytest.skip("native CUDA FP8 runtime is unavailable")

    layer = _new_fp8_scaled_linear(32, 64, bias=False)
    weight = torch.randn(64, 32).clamp(-2, 2).to(device="cuda", dtype=torch.float8_e4m3fn)
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.weight_scale = torch.ones((), dtype=torch.float32)
    layer = layer.cuda()
    x = torch.randn(2, 32, device="cuda", dtype=torch.bfloat16)

    y = layer(x)

    assert y.shape == (2, 64)
    assert y.dtype == torch.bfloat16
    assert not getattr(layer, "_scaled_mm_warned", False)
    assert layer.fast_mm_calls == 1
    assert layer.fallback_calls == 0


def test_fp8_scaled_linear_counts_bf16_fallback_on_cpu(monkeypatch):
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch float8 unavailable")

    monkeypatch.setenv("AIWF_WAN_ALLOW_FP8_FALLBACK", "1")
    layer = _new_fp8_scaled_linear(16, 32, bias=False)
    weight = torch.randn(32, 16).clamp(-2, 2).to(dtype=torch.float8_e4m3fn)
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.weight_scale = torch.ones((), dtype=torch.float32)

    y = layer(torch.randn(2, 16, dtype=torch.bfloat16))

    assert y.shape == (2, 32)
    assert layer.fast_mm_calls == 0
    assert layer.fallback_calls == 1
    assert "input is not CUDA" in str(layer.last_fallback_reason)
    metrics = _collect_fp8_linear_metrics(layer)
    assert metrics["fp8_linear_layers"] == 1
    assert metrics["fp8_fallback_calls"] == 1
    assert metrics["fp8_fallback_layers"] == 1
    assert metrics["fp8_backend"] == "torch_scaled_mm_e4m3fn"
    assert metrics["fp8_backend_metadata"]["compute_entrypoint"] == "torch._scaled_mm"
    assert metrics["fp8_linear_shape_count"] == 1
    assert metrics["fp8_linear_shapes"][0]["input_shape"] == [2, 16]
    assert metrics["fp8_linear_shapes"][0]["path"] == "fallback"


def test_stage_transition_metrics_are_bounded_and_aggregated():
    import time

    from aiwf.infrastructure.wan.pipeline import AIWFModelCacheManager

    cache = AIWFModelCacheManager(device="cuda")
    cache._record_transition(
        "swap_models",
        time.perf_counter() - 0.001,
        old_key="wan_high",
        new_key="wan_low",
        h2d_ms=12.5,
        d2h_ms=7.25,
        cleanup_ms=1.0,
    )

    metrics = cache.collect_transition_metrics()

    assert metrics["stage_transition_count"] == 1
    assert metrics["stage_transition_h2d_ms"] == 12.5
    assert metrics["stage_transition_d2h_ms"] == 7.25
    assert metrics["stage_transition_cleanup_ms"] == 1.0
    assert metrics["stage_transition_events"][0]["operation"] == "swap_models"


def test_aiwf_fp8_linear_load_quantized_weight_api(monkeypatch):
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch float8 unavailable")

    monkeypatch.setenv("AIWF_WAN_ALLOW_FP8_FALLBACK", "1")
    from aiwf.infrastructure.quant.fp8_linear import AIWFFP8Linear

    layer = AIWFFP8Linear(16, 32, bias=False, strict_exception_cls=WanUnavailable)
    qweight = torch.randn(32, 16).clamp(-2, 2).to(dtype=torch.float8_e4m3fn)
    weight_scale = torch.tensor(0.25, dtype=torch.float32)
    input_scale = torch.tensor(0.5, dtype=torch.float32)

    layer.load_quantized_weight(qweight, weight_scale, input_scale, orig_dtype=torch.bfloat16)
    y = layer(torch.randn(2, 16, dtype=torch.bfloat16))

    assert y.shape == (2, 32)
    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight_scale.dtype == torch.float32
    assert layer.input_scale.dtype == torch.float32
    assert layer.input_scale_reciprocal.dtype == torch.float32
    assert layer.input_scale_reciprocal.item() == pytest.approx(2.0)
    assert layer.orig_dtype == torch.bfloat16
    assert layer.fallback_calls == 1


def test_fp8_scaled_linear_strict_mode_raises_on_fallback(monkeypatch):
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch float8 unavailable")

    monkeypatch.delenv("AIWF_WAN_ALLOW_FP8_FALLBACK", raising=False)
    monkeypatch.delenv("AIWF_WAN_STRICT_FP8", raising=False)
    layer = _new_fp8_scaled_linear(16, 32, bias=False)
    weight = torch.randn(32, 16).clamp(-2, 2).to(dtype=torch.float8_e4m3fn)
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.weight_scale = torch.ones((), dtype=torch.float32)

    with pytest.raises(WanUnavailable, match="strict FP8 mode refused"):
        layer(torch.randn(2, 16, dtype=torch.bfloat16))

    assert layer.fallback_calls == 1


def test_cuda_supports_tensorcore_fp8_requires_ada_or_newer():
    class _Cuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def get_device_capability():
            return (8, 6)

    assert _cuda_supports_tensorcore_fp8(SimpleNamespace(cuda=_Cuda)) is False

    class _AdaCuda(_Cuda):
        @staticmethod
        def get_device_capability():
            return (8, 9)

    assert _cuda_supports_tensorcore_fp8(SimpleNamespace(cuda=_AdaCuda)) is True


def test_fp8_scaled_mm_failure_payload_contains_only_tensor_metadata():
    torch = pytest.importorskip("torch")

    layer = _new_fp8_scaled_linear(4, 8, bias=False)
    input_tensor = torch.tensor([[12345.5, 23456.5, 34567.5, 45678.5]], dtype=torch.float32)
    x8 = input_tensor.contiguous()
    weight_t = torch.empty((4, 8), dtype=torch.float32).t()
    scale_a = torch.ones((), dtype=torch.float32)
    scale_b = torch.ones((), dtype=torch.float32)

    payload = _fp8_scaled_mm_failure_payload(
        RuntimeError("synthetic scaled-mm failure"),
        layer=layer,
        input_tensor=input_tensor,
        x8=x8,
        weight_t=weight_t,
        scale_a=scale_a,
        scale_b=scale_b,
        rows=1,
        padded_rows=16,
        pad_m=15,
    )
    encoded = json.dumps(payload, sort_keys=True)

    assert payload["error"]["type"] == "RuntimeError"
    assert payload["layer"]["in_features"] == 4
    assert payload["layer"]["out_features"] == 8
    assert payload["input"]["shape"] == [1, 4]
    assert payload["matmul"]["rhs"]["shape"] == [8, 4]
    assert payload["matmul"]["pad_m"] == 15
    assert "12345.5" not in encoded
    assert "23456.5" not in encoded
    assert "values" not in encoded.lower()


def test_wan_transformer_key_renames_strip_comfy_prefix():
    renamed = _apply_wan_transformer_key_renames(
        {
            "model.diffusion_model.blocks.0.self_attn.q.weight": object(),
            "diffusion_model.blocks.0.ffn.0.bias": object(),
        }
    )

    assert "blocks.0.attn1.to_q.weight" in renamed
    assert "blocks.0.ffn.net.0.proj.bias" in renamed


def test_resolve_local_lora(tmp_path: Path):
    s = _svc(tmp_path)
    lora_root = s.flags.resolved_models_dir() / "Loras" / "Wan"
    lora_root.mkdir(parents=True)
    high = lora_root / "wan-step-high.safetensors"
    low = lora_root / "wan-step-low.safetensors"
    neutral = lora_root / "wan-step-neutral.safetensors"
    high.write_bytes(b"fake")
    low.write_bytes(b"fake")
    neutral.write_bytes(b"fake")

    assert "Wan" in str(lora_root)
