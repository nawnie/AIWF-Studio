import pytest

from aiwf.core.domain.extra_networks import LoraRef, parse_extra_networks
from aiwf.core.domain.models import LoraInfo
from aiwf.infrastructure.diffusers.extra_networks import apply_loras


class FakeLoraPipe:
    def __init__(self):
        self.loaded = []
        self.adapters = None
        self.unloaded = 0

    def unload_lora_weights(self):
        self.unloaded += 1

    def load_lora_weights(self, path, **kwargs):
        self.loaded.append((path, kwargs))

    def set_adapters(self, adapter_names, adapter_weights):
        self.adapters = (list(adapter_names), list(adapter_weights))


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


def test_apply_loras_rejects_sd15_lora_on_sd35():
    lora = LoraInfo(
        id="sd15_style",
        title="SD15 Style",
        filename="sd15_style.safetensors",
        path="sd15_style.safetensors",
        architecture="sd15",
    )

    with pytest.raises(ValueError, match="targets sd15"):
        apply_loras(object(), [LoraRef("sd15_style", 1.0)], [lora], base_architecture="sd35")


def test_apply_loras_unloads_when_prompt_has_no_loras():
    pipe = FakeLoraPipe()

    assert apply_loras(pipe, [], []) == []

    assert pipe.unloaded == 1
    assert pipe.loaded == []
    assert pipe.adapters is None


def test_apply_loras_reloads_changed_adapter_stack(tmp_path):
    style_a = tmp_path / "style_a.safetensors"
    style_b = tmp_path / "style_b.safetensors"
    style_a.write_bytes(b"a")
    style_b.write_bytes(b"b")
    catalog = [
        LoraInfo(id="style_a", title="Style A", filename=style_a.name, path=str(style_a), architecture="sd15"),
        LoraInfo(id="style_b", title="Style B", filename=style_b.name, path=str(style_b), architecture="sd15"),
    ]
    pipe = FakeLoraPipe()

    first = apply_loras(pipe, [LoraRef("style_a", 0.7)], catalog, base_architecture="sd15")
    second = apply_loras(pipe, [LoraRef("style_b", 0.4)], catalog, base_architecture="sd15")

    assert first == ["aiwf_lora_0"]
    assert second == ["aiwf_lora_0"]
    assert pipe.unloaded == 2
    assert pipe.loaded[0][1]["weight_name"] == "style_a.safetensors"
    assert pipe.loaded[1][1]["weight_name"] == "style_b.safetensors"
    assert pipe.adapters == (["aiwf_lora_0"], [0.4])
