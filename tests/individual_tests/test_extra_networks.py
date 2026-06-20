import pytest

from aiwf.core.domain.extra_networks import LoraRef, parse_extra_networks
from aiwf.core.domain.models import LoraInfo
from aiwf.infrastructure.diffusers.extra_networks import apply_loras


def test_parse_lora_tags():
    parsed = parse_extra_networks("photo of cat <lora:DetailTweaker:0.65> in garden")
    assert parsed.prompt == "photo of cat in garden"
    assert len(parsed.loras) == 1
    assert parsed.loras[0].name == "DetailTweaker"
    assert parsed.loras[0].weight == 0.65


def test_parse_multiple_loras():
    parsed = parse_extra_networks("<lora:StyleA:1> portrait <lora:StyleB:0.5>")
    assert parsed.prompt == "portrait"
    assert len(parsed.loras) == 2


def test_apply_loras_rejects_wrong_base_architecture():
    lora = LoraInfo(
        id="xl_style",
        title="XL Style",
        filename="xl_style.safetensors",
        path="xl_style.safetensors",
        architecture="sdxl",
    )

    with pytest.raises(ValueError, match="targets sdxl"):
        apply_loras(object(), [LoraRef("xl_style", 1.0)], [lora], base_architecture="sd15")
