from aiwf.core.domain.generation import GenerationRequest
from aiwf.infrastructure.diffusers.backend import DiffusersBackend
from aiwf.infrastructure.diffusers.model_arch import (
    ARCH_FLUX2_KLEIN,
    ARCH_FLUX_KONTEXT,
    ARCH_QWEN_IMAGE,
    ARCH_QWEN_IMAGE_NUNCHAKU,
    ARCH_SANA,
    ARCH_SANA_VIDEO,
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


def test_sana_sprint_allows_manual_non_default_steps():
    calls = []

    def fake_call(self, **kwargs):
        calls.append(kwargs)
        return "ok"

    pipe = type("SanaSprintPipeline", (), {"__call__": fake_call})()
    backend = DiffusersBackend.__new__(DiffusersBackend)
    backend.flags = None

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
