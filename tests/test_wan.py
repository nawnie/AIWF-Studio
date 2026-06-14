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
    WanI2VBackend,
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
    _orient_umt5_gguf_tensor,
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


def test_wan_generation_records_video_throughput(tmp_path: Path, monkeypatch):
    from PIL import Image

    s = _svc(tmp_path)
    _force_wan_available(s)
    _write_component_base(s)
    high = s.models_dir() / "Safetensor" / "wan-high.safetensors"
    low = s.models_dir() / "Safetensor" / "wan-low.safetensors"
    vae = s.flags.resolved_models_dir() / "VAE" / "wan_2.1_vae.safetensors"
    _write_fake_safetensors(high)
    _write_fake_safetensors(low)
    _write_fake_safetensors(vae)

    s._backend.generate = lambda *args, **kwargs: ([Image.new("RGB", (8, 8), "black")] * 5, 8, 8)
    captured: dict[str, object] = {}
    monkeypatch.setattr("aiwf.services.wan.trace_model_throughput", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr("aiwf.services.wan.write_frames", lambda frames, output_path, fps: len(frames))

    result = s.generate(WanI2VRequest(high_noise_model_id=high.name, low_noise_model_id=low.name), Image.new("RGB", (8, 8)))

    assert result.frame_count == 5
    assert captured["kind"] == "wan.video"
    assert captured["units_label"] == "frames"
    assert captured["units"] == 5
    assert captured["high_noise_model_id"] == high.name


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


def test_orient_umt5_gguf_tensor_transposes_swapped_embedding_shape():
    torch = pytest.importorskip("torch")

    swapped = torch.arange(32, dtype=torch.float32).reshape(4, 8)
    oriented = _orient_umt5_gguf_tensor("shared.weight", swapped, (8, 4))

    assert tuple(oriented.shape) == (8, 4)
    assert torch.equal(oriented, swapped.t())


def test_materialize_wan_transformer_passes_temporal_chunk_settings(tmp_path: Path):
    torch = pytest.importorskip("torch")
    target = torch.nn.Module()
    seen: list[tuple[int | None, int | None]] = []

    def load_state_dict(_state_dict, *, strict=False, assign=False):
        return [], []

    target.load_state_dict = load_state_dict

    def capture_optimizations(_target, _label, *, chunk_size=None, chunk_overlap=None):
        seen.append((chunk_size, chunk_overlap))

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
        )

    assert missing == []
    assert unexpected == []
    assert seen == [(12, 3)]


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
    }
    seen: list[tuple[int | None, int | None]] = []

    def fake_materialize(_target, _path, **kwargs):
        seen.append((kwargs.get("chunk_size"), kwargs.get("chunk_overlap")))
        return [], []

    with patch("aiwf.infrastructure.wan.pipeline._empty_wan_transformer", return_value=torch.nn.Module()):
        with patch.object(backend, "_materialize_wan_transformer", side_effect=fake_materialize):
            backend._run_low_preload_worker()

    assert backend._low_preload_error is None
    assert seen == [(20, 6)]


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
    loads: list[tuple[int | None, int | None]] = []

    def fake_load_dual_pipeline(**kwargs):
        loads.append((kwargs.get("chunk_size"), kwargs.get("chunk_overlap")))
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
        first = backend._ensure(**ensure_kwargs, chunk_size=16, chunk_overlap=8)
        second = backend._ensure(**ensure_kwargs, chunk_size=16, chunk_overlap=8)
        third = backend._ensure(**ensure_kwargs, chunk_size=20, chunk_overlap=8)
        fourth = backend._ensure(**ensure_kwargs, chunk_size=20, chunk_overlap=4)

    assert first is second
    assert third is not first
    assert fourth is not third
    assert loads == [(16, 8), (20, 8), (20, 4)]


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

    assert "Wan" in str(lora_root)
