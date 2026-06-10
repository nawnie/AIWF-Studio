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


class SamplerInfo(BaseModel):
    id: str
    label: str
    family: str = "diffusers"
    supports_karras: bool = False


SAMPLERS: list[SamplerInfo] = [
    SamplerInfo(id="euler", label="Euler", family="diffusers"),
    SamplerInfo(id="euler_a", label="Euler a", family="diffusers"),
    SamplerInfo(id="lms", label="LMS", family="diffusers"),
    SamplerInfo(id="dpmpp_2m", label="DPM++ 2M", family="diffusers"),
    SamplerInfo(id="dpmpp_2m_karras", label="DPM++ 2M Karras", family="diffusers", supports_karras=True),
]