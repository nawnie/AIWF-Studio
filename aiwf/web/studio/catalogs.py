from __future__ import annotations

from dataclasses import dataclass

from aiwf.bootstrap import AppContext
from aiwf.core.domain.models import SCHEDULE_TYPES


@dataclass(frozen=True)
class StudioCatalogs:
    """Sampler/schedule lookup tables built once per Studio tab mount."""

    sampler_map: dict[str, str]
    sampler_id_to_label: dict[str, str]
    default_sampler_label: str
    schedule_map: dict[str, str]
    schedule_id_to_label: dict[str, str]
    default_schedule_label: str

    @classmethod
    def from_context(cls, ctx: AppContext) -> StudioCatalogs:
        samplers = ctx.generation.list_samplers()
        sampler_map = {s.label: s.id for s in samplers}
        sampler_id_to_label = {s.id: s.label for s in samplers}
        fallback = samplers[1].label if len(samplers) > 1 else (samplers[0].label if samplers else None)
        default_sampler_label = sampler_id_to_label.get(ctx.settings.default_sampler, fallback)
        schedule_map = {s.label: s.id for s in SCHEDULE_TYPES}
        schedule_id_to_label = {s.id: s.label for s in SCHEDULE_TYPES}
        default_schedule_label = schedule_id_to_label.get(
            getattr(ctx.settings, "default_scheduler", "automatic"),
            "Automatic",
        )
        return cls(
            sampler_map=sampler_map,
            sampler_id_to_label=sampler_id_to_label,
            default_sampler_label=default_sampler_label,
            schedule_map=schedule_map,
            schedule_id_to_label=schedule_id_to_label,
            default_schedule_label=default_schedule_label,
        )