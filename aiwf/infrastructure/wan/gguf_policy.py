"""GGUF precision tier policy (master plan §1.3)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class PrecisionTier(str, Enum):
    HIGH = "high_precision"
    QUANTIZED = "quantized"


@dataclass(frozen=True)
class GGUFPrecisionRule:
    contains: str
    tier: PrecisionTier


DEFAULT_GGUF_RULES = (
    GGUFPrecisionRule("embed", PrecisionTier.HIGH),
    GGUFPrecisionRule("time_embedding", PrecisionTier.HIGH),
    GGUFPrecisionRule("timestep", PrecisionTier.HIGH),
    GGUFPrecisionRule("norm", PrecisionTier.HIGH),
    GGUFPrecisionRule("proj_out", PrecisionTier.HIGH),
    GGUFPrecisionRule("final", PrecisionTier.HIGH),
    GGUFPrecisionRule("head.head", PrecisionTier.HIGH),
    GGUFPrecisionRule("scale_shift", PrecisionTier.HIGH),
    GGUFPrecisionRule("modulation", PrecisionTier.HIGH),
)


def classify_gguf_tensor(
    tensor_name: str,
    rules: Iterable[GGUFPrecisionRule] = DEFAULT_GGUF_RULES,
) -> PrecisionTier:
    lowered = tensor_name.lower()
    for rule in rules:
        if rule.contains.lower() in lowered:
            return rule.tier
    return PrecisionTier.QUANTIZED