from pathlib import Path
from unittest.mock import patch

import pytest

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.wan import (
    WAN_TI2V_5B,
    WanI2VRequest,
    duration_seconds_for_frames,
    frames_for_duration_seconds,
    snap_num_frames,
)
from aiwf.infrastructure.wan.pipeline import (
    WanUnavailable,
    _apply_wan_transformer_key_renames,
    _boundary_ratio_for_step_split,
    _dequantize_comfy_fp8_state_dict,
    _ensure_wan_attention_processors,
    _frames_from_wan_pipeline_output,
    _load_comfy_fp8_transformer_weights,
    _load_umt5_text_encoder,
    _load_wan_vae,
    _new_lazy_wan_transformer,
    _new_wan_euler_simple_scheduler,
    _wan_cache_mode,
    estimate_gguf_expanded_gb,
)
from aiwf.services.wan import WanService


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
    assert r.fps == 16 and r.offload == "model"
    assert r.guidance_scale == 1.0
    assert r.normalized_frames() == 49
    assert r.effective_steps() == 8
    assert r.effective_boundary_ratio() == 0.5
    assert WanI2VRequest(width=512, height=320).max_area == 512 * 320


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
    assert _wan_cache_mode("sequential", fast_fp8_pair=True) == "none"
    assert _wan_cache_mode("model", fast_fp8_pair=True) == "gpu_swap"
    assert _wan_cache_mode("model", fast_fp8_pair=False) == "none"
    assert _wan_cache_mode("none", fast_fp8_pair=False) == "full"


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
    high = s.models_dir() / "Safetensor" / "wan-high.safetensors"
    low = s.models_dir() / "Safetensor" / "wan-low.safetensors"
    vae = s.flags.resolved_models_dir() / "VAE" / "wan_2.1_vae.safetensors"
    _write_fake_safetensors(high)
    _write_fake_safetensors(low)
    _write_fake_safetensors(vae)

    result = s.preflight(WanI2VRequest(high_noise_model_id=high.name, low_noise_model_id=low.name))

    assert result.ok, result.message()
    assert result.components_base is not None
    assert result.high_noise_model == str(high.resolve())
    assert result.low_noise_model == str(low.resolve())
    assert result.vae == str(vae.resolve())


def test_wan_preflight_blocks_missing_component_base(tmp_path: Path):
    s = _svc(tmp_path)
    _force_wan_available(s)
    high = s.models_dir() / "Safetensor" / "wan-high.safetensors"
    low = s.models_dir() / "Safetensor" / "wan-low.safetensors"
    vae = s.flags.resolved_models_dir() / "VAE" / "wan_2.1_vae.safetensors"
    _write_fake_safetensors(high)
    _write_fake_safetensors(low)
    _write_fake_safetensors(vae)

    result = s.preflight(WanI2VRequest(high_noise_model_id=high.name, low_noise_model_id=low.name))

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
    high = s.models_dir() / "Safetensor" / "wan-high.safetensors"
    low = s.models_dir() / "Safetensor" / "wan-low.safetensors"
    vae = s.flags.resolved_models_dir() / "VAE" / "wan_2.1_vae.safetensors"
    _write_fake_safetensors(high)
    _write_fake_safetensors(low)
    _write_fake_safetensors(vae)

    def fake_generate(*args, **kwargs):
        calls.append("video")
        frame = Image.new("RGB", (8, 8), "black")
        return [frame], 8, 8

    s._backend.generate = fake_generate

    s.generate(WanI2VRequest(high_noise_model_id=high.name, low_noise_model_id=low.name), Image.new("RGB", (8, 8)))

    assert calls[:2] == ["unload", "video"]


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
    (vae_root / "wan_2.1_vae.safetensors").write_bytes(b"fake")

    vaes = s.list_local_vaes()
    assert vaes[0] == "wan_2.1_vae.safetensors"
    assert s.preferred_vae() == "wan_2.1_vae.safetensors"


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

    assert "Wan/wan-step-high.safetensors" in s.list_local_loras()
    assert "Wan/wan-step-high.safetensors" in s.list_local_loras("high")
    assert "Wan/wan-step-low.safetensors" not in s.list_local_loras("high")
    assert "Wan/wan-step-low.safetensors" in s.list_local_loras("low")
    assert "Wan/wan-step-neutral.safetensors" not in s.list_local_loras("high")
    assert s.resolve_lora("Wan/wan-step-high.safetensors") == str(high.resolve())
    assert s.resolve_lora("wan-step-high.safetensors") == str(high.resolve())


def test_available_is_boolean(tmp_path: Path):
    # No torch/diffusers Wan in the sandbox -> must degrade gracefully, not crash.
    assert isinstance(_svc(tmp_path).available(), bool)


# --- Dual high/low noise test for Wan 2.2 I2V (the models that contain "high" / "low" in the filename) ---

# User's smallest high/low pair from F:\ComfyUI\models\diffusion_models (fp8 scaled 14B I2V)
# These are the ones with "high" and "low" in the name suitable for the two-stage pipeline.
WAN_HIGH = "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
WAN_LOW = "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"


def _wan_high_low_present(svc) -> bool:
    names = set(svc.list_local_models())
    return WAN_HIGH in names and WAN_LOW in names


def test_wan_i2v_dual_high_low_16fps_1s_smoke(tmp_path: Path):
    """
    Local smoke test for the Wan 2.2 two-transformer (high-noise + low-noise) workflow.

    - 16 fps, ~1 second video (17 frames after snapping to 4k+1 requirement)
    - Uses the user's smallest high/low pair (the i2v 14B fp8 ones that literally say "high" and "low")
    - Very cheap settings (sequential offload, tiny steps/resolution, low guidance) so it can
      run on a single machine for workflow validation.
    - The actual heavy load+render is guarded by RUN_WAN_DUAL_SMOKE=1 so normal test runs
      and CI stay fast. When you set the env var it will resolve your high/low files,
      load both transformers, run the dual-stage denoising, and produce the short video.

    Example to run the real test:
        $env:RUN_WAN_DUAL_SMOKE=1; python -m pytest tests/test_wan.py::test_wan_i2v_dual_high_low_16fps_1s_smoke -s --tb=line
    """
    import os
    from PIL import Image

    # Use real project layout so the F:\ComfyUI... scan (and models/wan) discovery works
    flags = RuntimeFlags(
        data_dir=Path("."),
        models_dir=Path("models"),
        output_dir=tmp_path / "video_out",
    )
    svc = WanService(flags, UserSettings())

    if not _wan_high_low_present(svc):
        pytest.skip(
            f"Skipping dual high/low test — {WAN_HIGH} and/or {WAN_LOW} not found. "
            "Place (or symlink) the files in models/wan/ or ensure they are in your "
            "F:\\ComfyUI\\models\\diffusion_models so the WanService discovery picks them up."
        )

    if os.environ.get("RUN_WAN_DUAL_SMOKE") != "1":
        # Still validate that the pair is discoverable and resolvable (cheap)
        high_r = svc.resolve_model(WAN_HIGH)
        low_r = svc.resolve_model(WAN_LOW)
        assert "high_noise" in high_r and high_r.endswith(".safetensors")
        assert "low_noise" in low_r and low_r.endswith(".safetensors")
        print(f"\n[discovery only] high -> {high_r}\n[discovery only] low  -> {low_r}")
        pytest.skip("Heavy dual high/low generate skipped (set RUN_WAN_DUAL_SMOKE=1 to execute the real load+render with your high/low models)")

    # Tiny synthetic source image (the pipeline will resize to a valid multiple for the loaded model)
    img = Image.new("RGB", (320, 320), color=(70, 90, 110))

    # 16 fps 1-second target: 17 frames (snaps to valid 4*k+1). Duration will be (17-1)/16 == 1.0s
    req = WanI2VRequest(
        prompt="quick workflow smoke test, simple motion",
        negative_prompt="",
        num_frames=17,
        fps=16,
        steps=4,                    # ultra cheap for local validation
        guidance_scale=3.0,
        width=256,
        height=256,
        offload="sequential",       # most VRAM-friendly
        seed=42,
        high_noise_model_id=WAN_HIGH,
        low_noise_model_id=WAN_LOW,
        boundary_ratio=0.875,       # standard split point for these Wan2.2 high/low pairs
        vae_id=None,                # or point to your Wan 2.1 VAE .safetensors if you want explicit
        # model_id left at default (will be ignored for the dual path)
    )

    result = svc.generate(req, img)

    out_path = Path(result.output_path)
    assert out_path.exists(), "Output video was not written"
    assert out_path.stat().st_size > 50_000, "Output video suspiciously small"

    # Verify the written video has the requested fps and roughly the right frame count
    import cv2
    cap = cv2.VideoCapture(str(out_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        assert abs(fps - 16.0) < 0.6, f"Expected ~16 fps, got {fps}"
        # 17 frames is the target; allow a little tolerance in case of writer rounding
        assert 15 <= n_frames <= 19, f"Expected ~17 frames for 1s@16fps, got {n_frames}"
    finally:
        cap.release()

    print(f"\n[Wan dual high/low smoke] 16fps 1s video OK -> {out_path}")
