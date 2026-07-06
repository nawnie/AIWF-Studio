from collections import OrderedDict
from types import SimpleNamespace

import torch

from aiwf.core.domain.generation import GenerationRequest
from aiwf.infrastructure.diffusers.backend import DiffusersBackend
from aiwf.infrastructure.diffusers.model_arch import (
    ARCH_ANIMA,
    ARCH_FLUX_FILL,
    ARCH_FLUX2_KLEIN,
    ARCH_FLUX_KONTEXT,
    ARCH_KREA2,
    ARCH_QWEN_IMAGE,
    ARCH_QWEN_IMAGE_NUNCHAKU,
    ARCH_SANA,
    ARCH_SANA_VIDEO,
    ARCH_SDXL,
    ARCH_SDXL_REFINER,
    ARCH_Z_IMAGE,
    detect_checkpoint_architecture,
)
from aiwf.infrastructure.diffusers.model_presets import resolve_model_preset
from aiwf.services.pipeline_preflight import preflight_image_runtime_pipelines


def test_qwen_sana_flux2_and_z_image_architecture_detection_from_names():
    assert detect_checkpoint_architecture("krea2_turbo_fp8_scaled.safetensors") == ARCH_KREA2
    assert detect_checkpoint_architecture("Krea-2-Raw") == ARCH_KREA2
    assert detect_checkpoint_architecture("split_files/diffusion_models/anima-base-v1.0.safetensors") == ARCH_ANIMA
    assert detect_checkpoint_architecture("WanAnimate_relight_lora_fp16.safetensors") != ARCH_ANIMA
    assert detect_checkpoint_architecture("Qwen-Image-2512") == ARCH_QWEN_IMAGE
    assert detect_checkpoint_architecture("qwen2.0-dev") == ARCH_QWEN_IMAGE
    assert detect_checkpoint_architecture("svdq-int4_r32-qwen-image-lightningv1.0-4steps.safetensors") == ARCH_QWEN_IMAGE_NUNCHAKU
    assert detect_checkpoint_architecture("SANA-Video_2B_480p_diffusers") == ARCH_SANA_VIDEO
    assert detect_checkpoint_architecture("Sana_Sprint_1.6B_1024px_diffusers") == ARCH_SANA
    assert detect_checkpoint_architecture("flux-kontext-4bit-fp4") == ARCH_FLUX_KONTEXT
    assert detect_checkpoint_architecture("flux1-fill-dev.safetensors") == ARCH_FLUX_FILL
    assert detect_checkpoint_architecture("sd_xl_refiner_1.0.safetensors") == ARCH_SDXL_REFINER
    assert detect_checkpoint_architecture("fluxtraitFLUX2KleinFLUXZ_klein4bQ4KM.gguf") == ARCH_FLUX2_KLEIN
    assert detect_checkpoint_architecture("fluxtraitFLUX2KleinFLUXZ_zImageV2GgufQ4.gguf") == ARCH_Z_IMAGE


def test_comfy_saved_flux2_klein_safetensor_metadata_detects_architecture(tmp_path):
    import json
    import struct

    header = {
        "model.diffusion_model.double_blocks.0.img_attn.qkv.weight": {
            "dtype": "F8_E4M3",
            "shape": [12288, 4096],
            "data_offsets": [0, 1],
        },
        "__metadata__": {
            "prompt": json.dumps(
                {
                    "1": {
                        "class_type": "UNETLoader",
                        "inputs": {"unet_name": "flux-2-klein-9b.safetensors"},
                    }
                }
            )
        },
    }
    payload = json.dumps(header).encode("utf-8")
    path = tmp_path / "snofsSexNudesAndOtherFunStuff_distilledV12Fp8.safetensors"
    path.write_bytes(struct.pack("<Q", len(payload)) + payload + b"\0")

    assert detect_checkpoint_architecture(path) == ARCH_FLUX2_KLEIN


def test_runtime_family_presets_are_sane_first_run_defaults():
    assert resolve_model_preset({}, "Qwen-Image-2512", ARCH_QWEN_IMAGE) == {
        "steps": 30,
        "cfg_scale": 4.0,
        "sampler": "euler",
        "scheduler": "automatic",
        "width": 1024,
        "height": 1024,
    }
    assert resolve_model_preset({}, "svdq-int4_r32-qwen-image-lightningv1.0-4steps", ARCH_QWEN_IMAGE_NUNCHAKU) == {
        "steps": 4,
        "cfg_scale": 1.0,
        "sampler": "euler",
        "scheduler": "automatic",
        "width": 1024,
        "height": 1024,
    }
    assert resolve_model_preset({}, "svdq-int4_r32-qwen-image-lightningv1.0-4steps", ARCH_QWEN_IMAGE)["steps"] == 4
    assert resolve_model_preset({}, "Sana_Sprint_1.6B_1024px_diffusers", ARCH_SANA)["steps"] == 2
    assert resolve_model_preset({}, "Sana_1600M_1024px_BF16_diffusers", ARCH_SANA)["steps"] == 20
    sana_video = resolve_model_preset({}, "SANA-Video_2B_480p_diffusers", ARCH_SANA_VIDEO)
    assert sana_video["steps"] == 50
    assert sana_video["cfg_scale"] == 6.0
    assert sana_video["width"] == 832
    assert sana_video["height"] == 480
    assert resolve_model_preset({}, "flux-kontext-4bit-fp4", ARCH_FLUX_KONTEXT)["cfg_scale"] == 3.5
    flux_fill = resolve_model_preset({}, "flux1-fill-dev", ARCH_FLUX_FILL)
    assert flux_fill["steps"] == 28
    assert flux_fill["cfg_scale"] == 3.5
    refiner = resolve_model_preset({}, "sd_xl_refiner_1.0", ARCH_SDXL_REFINER)
    assert refiner["steps"] == 10
    assert refiner["width"] == 1024
    flux2 = resolve_model_preset({}, "FLUX.2-klein-4B", ARCH_FLUX2_KLEIN)
    assert flux2["steps"] == 12
    assert flux2["sampler"] == "euler"
    z_image = resolve_model_preset({}, "Z-Image-Turbo", ARCH_Z_IMAGE)
    assert z_image["steps"] == 8
    assert z_image["sampler"] == "euler"
    krea_turbo = resolve_model_preset({}, "krea2_turbo_fp8_scaled", ARCH_KREA2)
    assert krea_turbo["steps"] == 8
    assert krea_turbo["cfg_scale"] == 0.0
    krea_raw = resolve_model_preset({}, "Krea-2-Raw", ARCH_KREA2)
    assert krea_raw["steps"] == 52
    assert krea_raw["cfg_scale"] == 3.5
    anima = resolve_model_preset({}, "anima-base-v1.0", ARCH_ANIMA)
    assert anima["steps"] == 36
    assert anima["cfg_scale"] == 4.5


def test_classic_sdxl_presets_ignore_unsafe_stale_smoke_settings():
    preset = resolve_model_preset(
        {
            "cyberrealisticPony_v125": {
                "steps": 2,
                "cfg_scale": 1.0,
                "sampler": "euler_a",
                "scheduler": "automatic",
                "width": 512,
                "height": 512,
            }
        },
        "cyberrealisticPony_v125",
        ARCH_SDXL,
    )

    assert preset["steps"] == 28
    assert preset["cfg_scale"] == 6.0
    assert preset["sampler"] == "dpmpp_2m"
    assert preset["width"] == 1024
    assert preset["height"] == 1024


def test_distilled_sdxl_presets_keep_low_step_settings():
    preset = resolve_model_preset(
        {
            "RealVisXL_V5.0_Lightning_fp16": {
                "steps": 2,
                "cfg_scale": 1.0,
                "sampler": "euler_a",
                "scheduler": "automatic",
                "width": 512,
                "height": 512,
            }
        },
        "RealVisXL_V5.0_Lightning_fp16",
        ARCH_SDXL,
    )

    assert preset["steps"] == 2
    assert preset["cfg_scale"] == 1.0
    assert preset["width"] == 512
    assert preset["height"] == 512


def test_sana_sprint_allows_manual_non_default_steps():
    calls = []

    def fake_call(self, **kwargs):
        calls.append(kwargs)
        return "ok"

    def fake_encode_prompt(self, **kwargs):
        return torch.ones(1, 2, 3), torch.ones(1, 2, dtype=torch.bool)

    pipe = type(
        "SanaSprintPipeline",
        (),
        {
            "__call__": fake_call,
            "encode_prompt": fake_encode_prompt,
            "_execution_device": torch.device("cpu"),
        },
    )()
    backend = DiffusersBackend.__new__(DiffusersBackend)
    backend.flags = SimpleNamespace(lowvram=False, medvram=False)
    backend.devices = SimpleNamespace(device=lambda: torch.device("cpu"), empty_cache=lambda: None)
    backend._active = None
    backend._sana_prompt_cache = OrderedDict()

    result = backend._run_sana_txt2img_pass(
        pipe,
        GenerationRequest(prompt="cat", cfg_scale=4.5),
        "cat",
        None,
        None,
        width=512,
        height=512,
        steps=4,
    )

    assert result == "ok"
    assert calls[0]["intermediate_timesteps"] is None
    assert calls[0]["prompt"] is None
    assert calls[0]["prompt_embeds"].shape == (1, 2, 3)


def test_qwen_image_pass_preencodes_prompt_embeddings():
    calls = []
    encoded_prompts = []

    def fake_call(self, **kwargs):
        calls.append(kwargs)
        return "ok"

    def fake_encode_prompt(self, **kwargs):
        encoded_prompts.append(kwargs["prompt"])
        return torch.ones(1, 2, 3), torch.ones(1, 2, dtype=torch.bool)

    pipe = type(
        "QwenImagePipeline",
        (),
        {
            "__call__": fake_call,
            "encode_prompt": fake_encode_prompt,
            "_execution_device": torch.device("cpu"),
        },
    )()
    backend = DiffusersBackend.__new__(DiffusersBackend)
    backend.flags = SimpleNamespace(lowvram=False, medvram=False)
    backend.devices = SimpleNamespace(device=lambda: torch.device("cpu"), empty_cache=lambda: None)
    backend._active = None
    backend._qwen_prompt_cache = OrderedDict()

    request = GenerationRequest(prompt="cat", negative_prompt="bad", cfg_scale=4.0)
    result = backend._run_qwen_image_txt2img_pass(
        pipe,
        request,
        "cat",
        None,
        None,
        width=512,
        height=512,
        steps=4,
    )

    assert result == "ok"
    assert encoded_prompts == ["cat", "bad"]
    assert calls[0]["prompt"] is None
    assert calls[0]["negative_prompt"] is None
    assert calls[0]["prompt_embeds"].shape == (1, 2, 3)
    assert calls[0]["negative_prompt_embeds"].shape == (1, 2, 3)


def test_krea2_turbo_pass_preencodes_prompt_without_negative_at_zero_guidance():
    calls = []
    encoded_prompts = []

    def fake_call(self, **kwargs):
        calls.append(kwargs)
        return "ok"

    def fake_encode_prompt(self, **kwargs):
        encoded_prompts.append(kwargs["prompt"])
        return torch.ones(1, 2, 3, 4), torch.ones(1, 2, dtype=torch.bool)

    pipe = type(
        "Krea2Pipeline",
        (),
        {
            "__call__": fake_call,
            "encode_prompt": fake_encode_prompt,
            "_execution_device": torch.device("cpu"),
        },
    )()
    backend = DiffusersBackend.__new__(DiffusersBackend)
    backend.flags = SimpleNamespace(lowvram=False, medvram=False)
    backend.devices = SimpleNamespace(device=lambda: torch.device("cpu"), empty_cache=lambda: None)
    backend._active = None
    backend._krea2_prompt_cache = OrderedDict()

    request = GenerationRequest(prompt="cat", negative_prompt="bad", cfg_scale=0.0)
    result = backend._run_krea2_txt2img_pass(
        pipe,
        request,
        "cat",
        None,
        None,
        width=512,
        height=512,
        steps=4,
    )

    assert result == "ok"
    assert encoded_prompts == ["cat"]
    assert calls[0]["prompt"] is None
    assert calls[0]["negative_prompt"] is None
    assert calls[0]["prompt_embeds"].shape == (1, 2, 3, 4)
    assert calls[0]["prompt_embeds_mask"].shape == (1, 2)
    assert calls[0]["negative_prompt_embeds"] is None
    assert calls[0]["guidance_scale"] == 0.0
    assert calls[0]["max_sequence_length"] == 512


def _policy_backend(total_vram_gb: float = 16.0) -> DiffusersBackend:
    backend = DiffusersBackend.__new__(DiffusersBackend)
    backend.flags = SimpleNamespace(lowvram=False, medvram=False, fluxfp8=False, fp8=False)
    backend.devices = SimpleNamespace(total_vram_gb=lambda: total_vram_gb)
    backend._offload_active = False
    return backend


def _checkpoint(name: str, size_gib: float, architecture: str):
    return SimpleNamespace(
        id=name,
        title=name,
        filename=name,
        path=name,
        architecture=architecture,
        size_bytes=int(size_gib * 1024**3),
    )


def test_krea2_mid_profile_uses_streamed_group_offload_on_16gb_card():
    calls = []
    backend = _policy_backend(total_vram_gb=16.0)
    backend.flags = SimpleNamespace(
        lowvram=False,
        medvram=True,
        highvram=False,
        effective_vram_profile=lambda: "mid",
    )
    backend.devices = SimpleNamespace(device=lambda: torch.device("cuda"), total_vram_gb=lambda: 16.0)

    transformer = SimpleNamespace(
        enable_group_offload=lambda **kwargs: calls.append(("group", kwargs)),
    )
    vae = SimpleNamespace(to=lambda device: calls.append(("vae", device)))
    pipe = SimpleNamespace(
        transformer=transformer,
        vae=vae,
        enable_sequential_cpu_offload=lambda: calls.append("sequential"),
        enable_model_cpu_offload=lambda: calls.append("model"),
    )

    assert backend._place_pipeline(
        pipe,
        architecture=ARCH_KREA2,
        checkpoint=_checkpoint("Krea-2-Turbo", 57.7, ARCH_KREA2),
    ) is pipe
    assert calls[0][0] == "group"
    assert calls[0][1]["offload_type"] == "block_level"
    assert calls[0][1]["num_blocks_per_group"] == 1
    assert calls[0][1]["use_stream"] is True
    assert calls[1] == ("vae", torch.device("cuda"))
    assert backend._offload_active is True


def test_krea2_mid_profile_keeps_group_offload_on_larger_cards():
    calls = []
    backend = _policy_backend(total_vram_gb=24.0)
    backend.flags = SimpleNamespace(
        lowvram=False,
        medvram=True,
        highvram=False,
        effective_vram_profile=lambda: "mid",
    )
    backend.devices = SimpleNamespace(device=lambda: torch.device("cuda"), total_vram_gb=lambda: 24.0)

    transformer = SimpleNamespace(
        enable_group_offload=lambda **kwargs: calls.append(("group", kwargs)),
    )
    pipe = SimpleNamespace(
        transformer=transformer,
        vae=SimpleNamespace(to=lambda device: calls.append(("vae", device))),
        enable_sequential_cpu_offload=lambda: calls.append("sequential"),
        enable_model_cpu_offload=lambda: calls.append("model"),
    )

    assert backend._place_pipeline(
        pipe,
        architecture=ARCH_KREA2,
        checkpoint=_checkpoint("Krea-2-Turbo", 57.7, ARCH_KREA2),
    ) is pipe
    assert calls[0][0] == "group"
    assert backend._offload_active is True


def test_krea2_normal_auto_offload_uses_streamed_group_offload_on_16gb_card():
    calls = []
    backend = _policy_backend(total_vram_gb=16.0)
    backend.flags = SimpleNamespace(
        lowvram=False,
        medvram=False,
        highvram=False,
        effective_vram_profile=lambda: "normal",
    )
    backend.devices = SimpleNamespace(device=lambda: torch.device("cuda"), total_vram_gb=lambda: 16.0)

    transformer = SimpleNamespace(
        enable_group_offload=lambda **kwargs: calls.append(("group", kwargs)),
    )
    pipe = SimpleNamespace(
        transformer=transformer,
        vae=SimpleNamespace(to=lambda device: calls.append(("vae", device))),
        enable_sequential_cpu_offload=lambda: calls.append("sequential"),
        enable_model_cpu_offload=lambda: calls.append("model"),
    )

    assert backend._place_pipeline(
        pipe,
        architecture=ARCH_KREA2,
        checkpoint=_checkpoint("Krea-2-Turbo", 57.7, ARCH_KREA2),
        prefer_offload=True,
    ) is pipe
    assert calls[0][0] == "group"
    assert backend._offload_active is True


def test_krea2_high_profile_uses_fp8_resident_transformer_with_text_encoder_cpu():
    calls = []
    backend = _policy_backend(total_vram_gb=16.0)
    backend.flags = SimpleNamespace(
        lowvram=False,
        medvram=False,
        highvram=True,
        effective_vram_profile=lambda: "high",
    )
    backend.devices = SimpleNamespace(device=lambda: torch.device("cuda"), total_vram_gb=lambda: 16.0)
    backend._dtype_for_architecture = lambda architecture: torch.bfloat16

    transformer = SimpleNamespace(
        enable_layerwise_casting=lambda **kwargs: calls.append(("fp8", kwargs)),
        to=lambda device: calls.append(("transformer", device)),
    )
    vae = SimpleNamespace(to=lambda device: calls.append(("vae", device)))
    text_encoder = SimpleNamespace(to=lambda device: calls.append(("text_encoder", device)))
    pipe = SimpleNamespace(
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        to=lambda device: calls.append(("pipe", device)),
    )

    assert backend._place_transformer_pipeline_keep_text_cpu(
        pipe,
        architecture=ARCH_KREA2,
        checkpoint=_checkpoint("Krea-2-Turbo", 57.7, ARCH_KREA2),
    ) is pipe
    assert calls[0][0] == "fp8"
    assert calls[0][1]["storage_dtype"] == torch.float8_e4m3fn
    assert ("transformer", torch.device("cuda")) in calls
    assert ("vae", torch.device("cuda")) in calls
    assert ("pipe", torch.device("cuda")) in calls
    assert ("text_encoder", "cpu") in calls
    assert pipe.text_encoder is text_encoder
    assert backend._offload_active is False


def test_size_aware_transformer_policy_keeps_small_sana_sprint_resident_on_16gb():
    backend = _policy_backend(total_vram_gb=16.0)

    assert backend._wants_offload(
        ARCH_SANA,
        checkpoint=_checkpoint("Sana_Sprint_0.6B_1024px_diffusers", 7.17, ARCH_SANA),
    ) is False
    assert backend._wants_offload(
        ARCH_QWEN_IMAGE,
        checkpoint=_checkpoint("Qwen-Image", 15.7, ARCH_QWEN_IMAGE),
    ) is True
    assert backend._wants_offload(
        ARCH_FLUX2_KLEIN,
        checkpoint=_checkpoint("fluxtraitFLUX2KleinFLUXZ_klein9bV2Q4KM", 4.99, ARCH_FLUX2_KLEIN),
    ) is False
    assert backend._wants_offload(
        ARCH_KREA2,
        checkpoint=_checkpoint("krea2_turbo_fp8_scaled", 12.25, ARCH_KREA2),
    ) is True
    assert backend._wants_offload(
        ARCH_ANIMA,
        checkpoint=_checkpoint("anima-base-v1.0", 3.9, ARCH_ANIMA),
    ) is False


def test_high_vram_profile_disables_auto_offload_for_large_transformers():
    backend = _policy_backend(total_vram_gb=16.0)
    backend.flags.highvram = True
    backend.flags.effective_vram_profile = lambda: "high"

    assert backend._wants_offload(
        ARCH_KREA2,
        checkpoint=_checkpoint("krea2_turbo_fp8_scaled", 12.25, ARCH_KREA2),
    ) is False


def test_krea2_qwen3vl_rope_parameters_backfill_rope_scaling():
    config = SimpleNamespace(
        text_config=SimpleNamespace(
            rope_scaling=None,
            rope_parameters={
                "mrope_interleaved": True,
                "mrope_section": [24, 20, 20],
                "rope_theta": 5000000,
                "rope_type": "default",
            },
        )
    )

    changed = DiffusersBackend._normalize_qwen3vl_rope_config(config)

    assert changed is True
    assert config.text_config.rope_scaling == {
        "mrope_interleaved": True,
        "mrope_section": [24, 20, 20],
        "rope_theta": 5000000,
        "rope_type": "default",
    }


def test_sana_sprint_vae_policy_slices_without_tiling():
    backend = _policy_backend(total_vram_gb=16.0)
    calls: list[str] = []

    class Vae:
        def enable_slicing(self):
            calls.append("slicing")

        def enable_tiling(self):
            calls.append("tiling")

        def disable_tiling(self):
            calls.append("disable_tiling")

    pipe = type("SanaSprintPipeline", (), {"vae": Vae()})()
    backend._tune_vae_memory(
        pipe,
        ARCH_SANA,
        _checkpoint("Sana_Sprint_0.6B_1024px_diffusers", 7.17, ARCH_SANA),
    )

    assert calls == ["slicing", "disable_tiling"]
    assert pipe._aiwf_vae_slicing_enabled is True
    assert pipe._aiwf_vae_tiling_enabled is False


def test_prompt_embedding_cache_is_bounded_and_keeps_recent_entries():
    backend = DiffusersBackend.__new__(DiffusersBackend)
    backend._PROMPT_EMBED_CACHE_LIMIT = 2
    cache = OrderedDict()

    backend._prompt_cache_put(cache, "first", "a", "Test")
    backend._prompt_cache_put(cache, "second", "b", "Test")
    assert backend._prompt_cache_get(cache, "first", "Test") == "a"

    backend._prompt_cache_put(cache, "third", "c", "Test")

    assert list(cache) == ["first", "third"]
    assert "second" not in cache


def test_flux2_large_text_encoder_uses_bnb_nf4_prompt_encoder(monkeypatch, tmp_path):
    backend = DiffusersBackend.__new__(DiffusersBackend)
    backend.devices = SimpleNamespace(device=lambda: torch.device("cuda"))
    monkeypatch.setattr(
        "aiwf.infrastructure.diffusers.backend.asset_size_bytes",
        lambda _path: int(15.3 * 1024**3),
    )

    kwargs, precision = backend._flux2_text_encoder_load_kwargs(tmp_path, torch.bfloat16)

    assert precision == "bnb_nf4"
    assert kwargs["device_map"] == {"": 0}
    quantization_config = kwargs["quantization_config"]
    assert quantization_config.load_in_4bit is True
    assert quantization_config.bnb_4bit_quant_type == "nf4"


def test_flux2_small_text_encoder_keeps_full_precision(monkeypatch, tmp_path):
    backend = DiffusersBackend.__new__(DiffusersBackend)
    backend.devices = SimpleNamespace(device=lambda: torch.device("cuda"))
    monkeypatch.setattr(
        "aiwf.infrastructure.diffusers.backend.asset_size_bytes",
        lambda _path: int(7.5 * 1024**3),
    )

    kwargs, precision = backend._flux2_text_encoder_load_kwargs(tmp_path, torch.bfloat16)

    assert precision == "bf16/fp16"
    assert "quantization_config" not in kwargs
    assert "device_map" not in kwargs


def test_text_encoder_gpu_swap_moves_denoisers_aside_for_prompt_encode():
    backend = _policy_backend(total_vram_gb=16.0)
    calls = {"empty_cache": 0}
    backend.devices = SimpleNamespace(
        total_vram_gb=lambda: 16.0,
        empty_cache=lambda: calls.__setitem__("empty_cache", calls["empty_cache"] + 1),
    )

    class Module:
        def __init__(self):
            self.device = torch.device("cpu")
            self.moves: list[str] = []

        def to(self, device):
            self.moves.append(str(device))
            self.device = torch.device(device)
            return self

    text_encoder = Module()
    transformer = Module()
    unet = Module()
    pipe = SimpleNamespace(
        text_encoder=text_encoder,
        transformer=transformer,
        unet=unet,
        _execution_device=torch.device("cpu"),
    )
    seen_devices: list[str] = []

    result = backend._encode_with_text_encoder_gpu_swap(
        pipe,
        architecture="Flux.2 Klein",
        device=torch.device("cuda"),
        encode=lambda device: seen_devices.append(str(device)) or "encoded",
    )

    assert result == "encoded"
    assert seen_devices == ["cuda"]
    assert text_encoder.moves == ["cuda", "cpu"]
    assert transformer.moves == ["cpu", "cuda"]
    assert unet.moves == ["cpu", "cuda"]
    assert pipe._execution_device == torch.device("cuda")
    assert calls["empty_cache"] >= 2


def test_image_runtime_preflight_has_required_diffusers_classes():
    result = preflight_image_runtime_pipelines()

    assert result.ok is True
    assert {item.name for item in result.items} >= {
        "Flux2KleinPipeline",
        "ZImagePipeline",
        "QwenImagePipeline",
        "SanaPipeline",
        "SanaSprintPipeline",
    }
