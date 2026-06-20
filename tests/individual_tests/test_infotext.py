from aiwf.core.domain.generation import GenerationMode, GenerationRequest
from aiwf.core.domain.models import Checkpoint
from aiwf.core.infotext import format_infotext, infotext_to_request_updates, parse_infotext


def test_parse_a1111_style_infotext():
    text = (
        "a cat in a hat\n"
        "Negative prompt: ugly, blurry\n"
        "Steps: 24, Sampler: Euler a, CFG scale: 8, Seed: 42, Size: 768x512, Model: test"
    )
    params = parse_infotext(text)
    assert params["Prompt"] == "a cat in a hat"
    assert params["Negative prompt"] == "ugly, blurry"
    assert params["Steps"] == "24"
    assert params["Seed"] == "42"
    assert params["Size-1"] == 768
    assert params["Size-2"] == 512


def test_round_trip_txt2img_request():
    request = GenerationRequest(
        prompt="sunset",
        negative_prompt="fog",
        steps=30,
        sampler="euler_a",
        cfg_scale=6.5,
        width=640,
        height=384,
        clip_skip=2,
        enable_hr=True,
        hr_scale=2.0,
        hr_steps=15,
        hr_denoising_strength=0.4,
    )
    checkpoint = Checkpoint(id="test", title="test [abc]", filename="test.safetensors", path="/tmp/test.safetensors")
    text = format_infotext(request, 99, checkpoint, output_width=1280, output_height=768)
    params = parse_infotext(text)
    updates = infotext_to_request_updates(params, GenerationMode.TXT2IMG)
    assert updates["prompt"] == "sunset"
    assert updates["negative_prompt"] == "fog"
    assert updates["steps"] == 30
    assert updates["sampler"] == "euler_a"
    assert updates["seed"] == 99
    assert updates["width"] == 640
    assert updates["height"] == 384
    assert updates["enable_hr"] is True
    assert updates["hr_steps"] == 15
    assert params["Hires resize-1"] == 1280
    assert params["Hires resize-2"] == 768


def test_format_infotext_includes_tags():
    request = GenerationRequest(prompt="test", tags=["archive", "v2"])
    checkpoint = Checkpoint(id="t", title="model", filename="m.safetensors", path="/m.safetensors")
    text = format_infotext(request, 1, checkpoint)
    assert "Tags: #archive #v2" in text
