from aiwf.core.domain.extra_networks import parse_extra_networks


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