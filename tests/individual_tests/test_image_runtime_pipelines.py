from collections import OrderedDict
from types import SimpleNamespace

import torch

from aiwf.core.domain.generation import GenerationRequest
from aiwf.infrastructure.diffusers.backend import DiffusersBackend
from aiwf.infrastructure.diffusers.model_arch import (
    ARCH_FLUX2_KLEIN,
    ARCH_FLUX_KONTEXT,
    ARCH_QWEN_IMAGE,
    ARCH_QWEN_IMAGE_NUNCHAKU,
    ARCH_SANA,
    ARCH_SANA_VIDEO,
    ARCH_SDXL,
    ARCH_Z_IMAGE,
    detect_checkpoint_architecture,
)
from aiwf.infrastructure.diffusers.model_presets import resolve_model_preset
from aiwf.services.pipeline_preflight import preflight_image_runtime_pipelines


def test_qwen_sana_flux2_and_z_image_architecture_detection_from_names():
    assert detect_checkpoint_architecture("Qwen-Image-2512") == ARCH_QWEN_IMAGE
    assert detect_checkpoint_architecture("qwen2.0-dev") == ARCH_QWEN_IMAGE
    assert detect_checkpoint_architecture("svdq-int4_r32-qwen-image-lightningv1.0-4steps.safetensors") == ARCH_QWEN_IMAGE_NUNCHAKU
    assert detect_checkpoint_architecture("SANA-Video_2B_480p_diffusers") == ARCH_SANA_VIDEO
    assert detect_checkpoint_architecture("Sana_Sprint_1.6B_1024px_diffusers") == ARCH_SANA
    assert detect_checkpoint_architecture("flux-kontext-4bit-fp4") == ARCH_FLUX_KONTEXT
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
    flux2 = resolve_model_preset({}, "FLUX.2-klein-4B", ARCH_FLUX2_KLEIN)
    assert flux2["steps"] == 12
    assert flux2["sampler"] == "euler"
    z_image = resolve_model_preset({}, "Z-Image-Turbo", ARCH_Z_IMAGE)
    assert z_image["steps"] == 8
    assert z_image["sampler"] == "euler"


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
