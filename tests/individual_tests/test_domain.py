import pytest

from aiwf.core.domain.generation import GenerationRequest


def test_dimensions_must_be_multiple_of_8():
    with pytest.raises(ValueError):
        GenerationRequest(width=513, height=512)


def test_defaults_are_valid():
    request = GenerationRequest()
    assert request.steps == 20
    assert request.width == 512
    assert request.hr_upscaler == "lanczos"


def test_legacy_latent_hr_upscaler_maps_to_lanczos():
    request = GenerationRequest(hr_upscaler="latent")
    assert request.hr_upscaler == "lanczos"
