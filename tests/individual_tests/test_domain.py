import pytest

from aiwf.core.domain.generation import GenerationRequest


def test_dimensions_must_be_multiple_of_8():
    with pytest.raises(ValueError):
        GenerationRequest(width=513, height=512)


def test_defaults_are_valid():
    request = GenerationRequest()
    assert request.steps == 20
    assert request.width == 512