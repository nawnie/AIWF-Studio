from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from aiwf.core.domain.generation import GenerationRequest, GenerationResult


@dataclass(frozen=True)
class AppStarted:
    pass


@dataclass(frozen=True)
class JobQueued:
    job_id: UUID
    request: GenerationRequest


@dataclass(frozen=True)
class JobStarted:
    job_id: UUID
    request: GenerationRequest


@dataclass(frozen=True)
class JobProgressed:
    job_id: UUID
    step: int
    total_steps: int
    message: str


@dataclass(frozen=True)
class JobFinished:
    job_id: UUID
    result: GenerationResult


@dataclass(frozen=True)
class JobCancelled:
    job_id: UUID


@dataclass(frozen=True)
class JobFailed:
    job_id: UUID
    error: str


@dataclass(frozen=True)
class BeforeGenerate:
    job_id: UUID
    request: GenerationRequest


@dataclass(frozen=True)
class AfterGenerate:
    job_id: UUID
    result: GenerationResult