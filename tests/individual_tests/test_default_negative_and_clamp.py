from types import SimpleNamespace

from aiwf.core.config.settings import UserSettings
from aiwf.core.domain.generation import GenerationRequest
from aiwf.services.generation import DEFAULT_NEGATIVE_PROMPT, GenerationService


def _svc(settings):
    return GenerationService(
        backend=None, store=None, metadata=None, queue=None, events=None, settings=settings
    )


class _Lightning:
    title = "RealVisXL_V5.0_Lightning_fp16"
    filename = "RealVisXL_V5.0_Lightning_fp16.safetensors"
    id = "lightning"


def test_default_negative_applied_when_blank():
    out = _svc(UserSettings())._apply_default_negative(GenerationRequest(prompt="cat", negative_prompt=""))
    assert out.negative_prompt == DEFAULT_NEGATIVE_PROMPT


def test_default_negative_preserves_user_value():
    out = _svc(UserSettings())._apply_default_negative(GenerationRequest(prompt="cat", negative_prompt="ugly"))
    assert out.negative_prompt == "ugly"


def test_default_negative_can_be_disabled():
    out = _svc(SimpleNamespace(use_default_negative=False))._apply_default_negative(
        GenerationRequest(prompt="cat", negative_prompt="")
    )
    assert out.negative_prompt == ""


def test_cfg_clamp_default_on():
    out = _svc(UserSettings())._guard_distilled_cfg(GenerationRequest(prompt="x", cfg_scale=7.0), _Lightning())
    assert out.cfg_scale < 7.0  # clamped


def test_cfg_clamp_can_be_disabled():
    out = _svc(SimpleNamespace(auto_cfg_for_distilled=False))._guard_distilled_cfg(
        GenerationRequest(prompt="x", cfg_scale=7.0), _Lightning()
    )
    assert out.cfg_scale == 7.0  # left alone
