from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from aiwf.core.config.settings import RuntimeFlags
from aiwf.core.domain.controlnet import ControlNetUnit
from aiwf.core.domain.generation import GenerationMode, GenerationRequest
from aiwf.infrastructure.diffusers.controlnet_pipe import (
    ControlNetModelCache,
    assert_controlnet_checkpoint_compatible,
    infer_controlnet_architecture,
)
from aiwf.infrastructure.diffusers import controlnet_pipe
from aiwf.infrastructure.diffusers.backend import DiffusersBackend
from aiwf.services.controlnet import ControlNetService
from aiwf.web.studio.controlnet_stack import StudioControlNetSlot, build_controlnet_stack


def test_infer_controlnet_architecture_detects_sd15_lora():
    name = "control_lora_rank128_v11p_sd15_canny_fp16.safetensors"
    assert infer_controlnet_architecture(name) == "sd15"


def test_infer_controlnet_architecture_detects_sdxl_filename():
    assert infer_controlnet_architecture("diffusion_pytorch_model_sdxl_canny.safetensors") == "sdxl"


def test_infer_controlnet_architecture_detects_sdxl_folder():
    assert infer_controlnet_architecture("models/ControlNet/controlnet-union-sdxl-1.0") == "sdxl"


def test_assert_rejects_sd15_controlnet_with_sdxl_checkpoint():
    with pytest.raises(ValueError, match="SD1.5 ControlNet"):
        assert_controlnet_checkpoint_compatible(
            "control_lora_rank128_v11p_sd15_canny_fp16.safetensors",
            "sdxl",
        )


def test_assert_allows_sd15_pairing():
    assert_controlnet_checkpoint_compatible(
        "control_lora_rank128_v11p_sd15_canny_fp16.safetensors",
        "sd15",
    )


def test_assert_rejects_sd35_controlnet_pairing():
    with pytest.raises(ValueError, match="SD3.5 ControlNet"):
        assert_controlnet_checkpoint_compatible(
            "control_lora_rank128_v11p_sd15_canny_fp16.safetensors",
            "sd35",
        )


def test_validate_enabled_requires_model_image_and_supported_mode(tmp_path: Path):
    service = ControlNetService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    service.ensure_dir()
    (service.models_dir() / "control_canny.safetensors").write_bytes(b"x")

    with pytest.raises(ValueError, match="control image"):
        service.validate_enabled(enabled=True, mode="inpaint", model_id="control_canny", control_image=None)

    with pytest.raises(ValueError, match="control image"):
        service.validate_enabled(enabled=True, mode="txt2img", model_id="control_canny", control_image=None)

    service.validate_enabled(
        enabled=True,
        mode="txt2img",
        model_id="control_canny",
        control_image=Image.new("RGB", (8, 8), "red"),
    )
    service.validate_enabled(
        enabled=True,
        mode="inpaint",
        model_id="control_canny",
        control_image=Image.new("RGB", (8, 8), "red"),
    )


def test_controlnet_cache_loads_diffusers_folder(monkeypatch, tmp_path: Path):
    folder = tmp_path / "controlnet-union-sdxl-1.0"
    folder.mkdir()
    (folder / "config.json").write_text("{}", encoding="utf-8")
    (folder / "diffusion_pytorch_model.safetensors").write_text("x", encoding="utf-8")
    sentinel = object()
    calls = {}

    def fake_from_pretrained(path, **kwargs):
        calls["path"] = path
        calls["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(controlnet_pipe.ControlNetModel, "from_pretrained", fake_from_pretrained)
    assert ControlNetModelCache().load(str(folder), dtype="float16") is sentinel
    assert calls["path"] == str(folder.resolve())
    assert calls["kwargs"]["torch_dtype"] == "float16"


def test_controlnet_cache_passes_adjacent_yaml_to_single_file(monkeypatch, tmp_path: Path):
    model = tmp_path / "control_v11p_sd15_canny.pth"
    config = tmp_path / "control_v11p_sd15_canny.yaml"
    model.write_text("x", encoding="utf-8")
    config.write_text("model: controlnet", encoding="utf-8")
    sentinel = object()
    calls = {}

    def fake_from_single_file(path, **kwargs):
        calls["path"] = path
        calls["kwargs"] = kwargs
        return sentinel

    monkeypatch.setattr(controlnet_pipe.ControlNetModel, "from_single_file", fake_from_single_file)
    assert ControlNetModelCache().load(str(model), dtype="float16") is sentinel
    assert calls["path"] == str(model.resolve())
    assert calls["kwargs"]["torch_dtype"] == "float16"
    assert calls["kwargs"]["original_config"] == str(config)


def test_build_controlnet_pipeline_uses_inpaint_class(monkeypatch):
    class FakeBasePipe:
        vae = object()
        text_encoder = object()
        tokenizer = object()
        unet = object()
        scheduler = object()
        text_encoder_2 = None

    class FakeInpaintPipe:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    controlnet = object()
    monkeypatch.setattr(controlnet_pipe, "StableDiffusionControlNetInpaintPipeline", FakeInpaintPipe)

    pipe = controlnet_pipe.build_controlnet_pipeline(FakeBasePipe(), controlnet, mode="inpaint")

    assert isinstance(pipe, FakeInpaintPipe)
    assert pipe.kwargs["controlnet"] is controlnet
    assert pipe.kwargs["requires_safety_checker"] is False


def test_studio_controlnet_stack_builds_multiple_units(tmp_path: Path):
    service = ControlNetService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    service.ensure_dir()
    for name in ("control_canny", "control_depth", "control_openpose"):
        (service.models_dir() / f"{name}.safetensors").write_bytes(b"x")

    image = Image.new("RGB", (8, 8), "red")
    units, images = build_controlnet_stack(
        mode="txt2img",
        controlnet=service,
        slots=[
            StudioControlNetSlot("Unit 1", True, "control_canny", "canny", image, 1.0, 0.0, 1.0, 100, 200),
            StudioControlNetSlot("Unit 2", True, "control_depth", "depth", image, 0.75, 0.1, 0.8, 64, 64),
            StudioControlNetSlot("Unit 3", True, "control_openpose", "openpose", image, 0.5, 0.2, 0.9, 64, 64),
        ],
    )

    assert [unit.model for unit in units] == ["control_canny", "control_depth", "control_openpose"]
    assert [unit.weight for unit in units] == [1.0, 0.75, 0.5]
    assert images == [image, image, image]


def test_backend_prepares_multiple_controlnet_units(tmp_path: Path):
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models")
    service = ControlNetService(flags)
    service.ensure_dir()
    (service.models_dir() / "control_a.safetensors").write_bytes(b"x")
    (service.models_dir() / "control_b.safetensors").write_bytes(b"x")
    backend = DiffusersBackend.__new__(DiffusersBackend)
    backend.flags = flags

    request = GenerationRequest(
        mode=GenerationMode.TXT2IMG,
        controlnet_units=[
            ControlNetUnit(model="control_a", module="none", weight=0.4),
            ControlNetUnit(model="control_b", module="none", weight=0.8),
        ],
    )
    image_a = Image.new("RGB", (16, 16), "red")
    image_b = Image.new("RGB", (16, 16), "blue")

    prepared = backend._prepare_controlnets(request, [image_a, image_b])

    assert [unit.model for unit, _image, _path in prepared] == ["control_a", "control_b"]
    assert [image.getpixel((0, 0)) for _unit, image, _path in prepared] == [(255, 0, 0), (0, 0, 255)]
    assert [path.name for _unit, _image, path in prepared] == ["control_a.safetensors", "control_b.safetensors"]


def test_controlnet_pass_sends_multi_unit_images_and_scales():
    backend = DiffusersBackend.__new__(DiffusersBackend)
    request = GenerationRequest(
        mode=GenerationMode.TXT2IMG,
        width=64,
        height=64,
        steps=4,
        controlnet_units=[
            ControlNetUnit(model="control_a", module="none", weight=0.4, guidance_start=0.0, guidance_end=0.6),
            ControlNetUnit(model="control_b", module="none", weight=0.8, guidance_start=0.2, guidance_end=1.0),
        ],
    )

    class CapturePipe:
        text_encoder_2 = None

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return type("Output", (), {"images": [Image.new("RGB", (64, 64), "green")]})()

    pipe = CapturePipe()
    output, width, height = backend._run_controlnet_pass(
        pipe,
        request,
        "cat",
        generator=None,
        callback=None,
        units=request.controlnet_units,
        control_images=[
            Image.new("RGB", (32, 48), "red"),
            Image.new("RGB", (48, 32), "blue"),
        ],
        init_images=None,
    )

    assert width == 64
    assert height == 64
    assert len(output.images) == 1
    assert pipe.kwargs["controlnet_conditioning_scale"] == [0.4, 0.8]
    assert pipe.kwargs["control_guidance_start"] == [0.0, 0.2]
    assert pipe.kwargs["control_guidance_end"] == [0.6, 1.0]
    assert [image.size for image in pipe.kwargs["image"]] == [(64, 64), (64, 64)]


def test_controlnet_inpaint_pass_sends_image_mask_and_control_image():
    backend = DiffusersBackend.__new__(DiffusersBackend)
    request = GenerationRequest(
        mode=GenerationMode.INPAINT,
        width=64,
        height=64,
        steps=4,
        controlnet_units=[
            ControlNetUnit(model="control_a", module="none", weight=0.4),
        ],
    )

    class CapturePipe:
        text_encoder_2 = None

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return type("Output", (), {"images": [Image.new("RGB", (64, 64), "green")]})()

    pipe = CapturePipe()
    output, width, height = backend._run_controlnet_pass(
        pipe,
        request,
        "cat",
        generator=None,
        callback=None,
        units=request.controlnet_units,
        control_images=[Image.new("RGB", (32, 48), "red")],
        init_images=[Image.new("RGB", (64, 64), "blue")],
        mask_images=[Image.new("L", (64, 64), 255)],
    )

    assert width == 64
    assert height == 64
    assert len(output.images) == 1
    assert pipe.kwargs["image"].size == (64, 64)
    assert pipe.kwargs["mask_image"].size == (64, 64)
    assert pipe.kwargs["control_image"].size == (64, 64)
