from __future__ import annotations

from types import SimpleNamespace

from diffusers import DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, SASolverScheduler

from aiwf.core.config.settings import RuntimeFlags
from aiwf.core.domain.models import allowed_schedule_ids_for_sampler, normalize_schedule_id_for_sampler
from aiwf.infrastructure.diffusers.backend import DiffusersBackend


def _backend(tmp_path):
    return DiffusersBackend(RuntimeFlags(data_dir=tmp_path), SimpleNamespace())


def _pipe():
    return SimpleNamespace(scheduler=EulerAncestralDiscreteScheduler())


def test_apply_sampler_does_not_leak_sde_algorithm_type_between_samplers(tmp_path):
    backend = _backend(tmp_path)
    pipe = _pipe()

    backend._remember_base_scheduler_config(pipe)
    backend._apply_sampler(pipe, "dpmpp_2m_sde", "automatic")
    assert isinstance(pipe.scheduler, DPMSolverMultistepScheduler)

    backend._apply_sampler(pipe, "sa_solver", "automatic")
    assert isinstance(pipe.scheduler, SASolverScheduler)


def test_apply_sampler_does_not_keep_beta_sigmas_when_switching_to_karras_sampler(tmp_path):
    backend = _backend(tmp_path)
    pipe = _pipe()

    backend._remember_base_scheduler_config(pipe)
    backend._apply_sampler(pipe, "dpmpp_2m", "beta")
    assert isinstance(pipe.scheduler, DPMSolverMultistepScheduler)

    backend._apply_sampler(pipe, "dpmpp_2m_karras", "automatic")
    assert isinstance(pipe.scheduler, DPMSolverMultistepScheduler)
    assert getattr(pipe.scheduler.config, "use_karras_sigmas", False) is True
    assert getattr(pipe.scheduler.config, "use_beta_sigmas", False) is False


def test_apply_sampler_leaves_krea2_flowmatch_scheduler_unchanged(tmp_path):
    backend = _backend(tmp_path)
    scheduler = object()
    pipe = type("Krea2Pipeline", (), {"scheduler": scheduler})()

    backend._apply_sampler(pipe, "dpmpp_2m", "beta")

    assert pipe.scheduler is scheduler
    assert pipe._aiwf_scheduler_signature == "dpmpp_2m|beta"


def test_schedule_normalization_restricts_builtin_karras_sampler():
    assert allowed_schedule_ids_for_sampler("dpmpp_2m_karras") == ["automatic"]
    assert normalize_schedule_id_for_sampler("dpmpp_2m_karras", "beta") == "automatic"
    assert normalize_schedule_id_for_sampler("dpmpp_2m", "beta") == "beta"
