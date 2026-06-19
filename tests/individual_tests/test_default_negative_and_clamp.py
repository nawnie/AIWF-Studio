from types import SimpleNamespace

from aiwf.core.config.settings import UserSettings
from aiwf.core.domain.generation import GenerationRequest
from aiwf.services.generation import DEFAULT_NEGATIVE_PROMPT, GenerationService
from aiwf.services.optimization import OptimizationPlanner


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


def test_generation_resolves_balanced_optimization_plan():
    svc = GenerationService(
        backend=None,
        store=None,
        metadata=None,
        queue=None,
        events=None,
        settings=UserSettings(),
        optimization_planner=OptimizationPlanner(),
    )

    plan = svc._resolve_optimization_plan(GenerationRequest(prompt="cat"), _Lightning())

    assert plan is not None
    assert plan.profile_id == "balanced_sdpa_fp16"


def test_generation_optimization_plan_counts_prompt_loras():
    settings = UserSettings(optimization_profile_id="experimental_feature_flags")
    svc = GenerationService(
        backend=None,
        store=None,
        metadata=None,
        queue=None,
        events=None,
        settings=settings,
        optimization_planner=OptimizationPlanner(),
    )

    request = GenerationRequest(prompt="portrait <lora:detail:0.8>")
    plan = svc._resolve_optimization_plan(request, _Lightning())

    assert plan is not None
    assert plan.profile_id == "experimental_feature_flags"
