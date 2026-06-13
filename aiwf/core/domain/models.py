from __future__ import annotations

from pydantic import BaseModel, Field


class Checkpoint(BaseModel):
    id: str
    title: str
    filename: str
    path: str
    hash: str | None = None
    kind: str = "checkpoint"
    architecture: str = "sd15"


class LoraInfo(BaseModel):
    id: str
    title: str
    filename: str
    path: str


class VaeInfo(BaseModel):
    id: str
    title: str
    filename: str
    path: str


class EmbeddingInfo(BaseModel):
    id: str
    title: str
    filename: str
    path: str


class SamplerInfo(BaseModel):
    id: str
    label: str
    family: str = "diffusers"
    supports_karras: bool = False


class SchedulerInfo(BaseModel):
    id: str
    label: str


SAMPLERS: list[SamplerInfo] = [
    SamplerInfo(id="euler", label="Euler", family="diffusers"),
    SamplerInfo(id="euler_a", label="Euler a", family="diffusers"),
    SamplerInfo(id="heun", label="Heun", family="diffusers"),
    SamplerInfo(id="lms", label="LMS", family="diffusers"),
    SamplerInfo(id="ddim", label="DDIM", family="diffusers"),
    SamplerInfo(id="unipc", label="UniPC", family="diffusers"),
    SamplerInfo(id="dpm2", label="DPM2", family="diffusers", supports_karras=True),
    SamplerInfo(id="dpm2_a", label="DPM2 a", family="diffusers", supports_karras=True),
    SamplerInfo(id="deis", label="DEIS", family="diffusers"),
    SamplerInfo(id="dpmpp_2m", label="DPM++ 2M", family="diffusers", supports_karras=True),
    SamplerInfo(id="dpmpp_2m_sde", label="DPM++ 2M SDE", family="diffusers", supports_karras=True),
    SamplerInfo(id="dpmpp_3m_sde", label="DPM++ 3M SDE", family="diffusers", supports_karras=True),
    SamplerInfo(id="dpmpp_sde", label="DPM++ SDE", family="diffusers", supports_karras=True),
    SamplerInfo(id="dpmpp_2m_karras", label="DPM++ 2M Karras", family="diffusers", supports_karras=True),
    SamplerInfo(id="sa_solver", label="SA-Solver", family="diffusers"),
    SamplerInfo(id="lcm", label="LCM", family="diffusers"),
    SamplerInfo(id="tcd", label="TCD", family="diffusers"),
]


SCHEDULE_TYPES: list[SchedulerInfo] = [
    SchedulerInfo(id="automatic", label="Automatic"),
    SchedulerInfo(id="uniform", label="Uniform"),
    SchedulerInfo(id="karras", label="Karras"),
    SchedulerInfo(id="exponential", label="Exponential"),
    SchedulerInfo(id="sgm_uniform", label="SGM Uniform"),
    SchedulerInfo(id="beta", label="Beta"),
]

_SCHEDULE_COMPATIBILITY: dict[str, set[str]] = {
    # This sampler already bakes Karras into the sampler choice itself.
    "dpmpp_2m_karras": {"automatic"},
}


def allowed_schedule_ids_for_sampler(sampler_id: str | None) -> list[str]:
    allowed = _SCHEDULE_COMPATIBILITY.get((sampler_id or "").lower())
    if allowed is None:
        return [item.id for item in SCHEDULE_TYPES]
    return [item.id for item in SCHEDULE_TYPES if item.id in allowed]


def normalize_schedule_id_for_sampler(sampler_id: str | None, schedule_id: str | None) -> str:
    normalized = (schedule_id or "automatic").lower()
    allowed = set(allowed_schedule_ids_for_sampler(sampler_id))
    if normalized in allowed:
        return normalized
    return "automatic"
