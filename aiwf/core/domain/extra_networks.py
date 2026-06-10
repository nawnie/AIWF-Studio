from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

RE_EXTRA_NET = re.compile(r"<(\w+):([^>]+)>")


@dataclass(frozen=True)
class LoraRef:
    name: str
    weight: float = 1.0


@dataclass
class ParsedPrompt:
    prompt: str
    loras: list[LoraRef] = field(default_factory=list)


def parse_extra_networks(prompt: str) -> ParsedPrompt:
    """Extract <lora:name:weight> tags and return a clean prompt."""
    loras: list[LoraRef] = []

    def replace(match: re.Match[str]) -> str:
        kind = match.group(1).lower()
        if kind != "lora":
            return match.group(0)
        parts = match.group(2).split(":")
        name = parts[0].strip()
        weight = float(parts[1]) if len(parts) > 1 else 1.0
        loras.append(LoraRef(name=name, weight=weight))
        return ""

    cleaned = re.sub(RE_EXTRA_NET, replace, prompt)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return ParsedPrompt(prompt=cleaned, loras=loras)