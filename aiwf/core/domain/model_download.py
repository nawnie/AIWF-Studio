from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ModelCategory = Literal[
    "checkpoint",
    "lora",
    "vae",
    "controlnet",
    "upscaler",
    "faceswap",
    "embedding",
    "hypernetwork",
    "other",
]

ModelSource = Literal["huggingface", "civitai", "direct"]


@dataclass(frozen=True)
class CatalogEntry:
    key: str
    title: str
    category: ModelCategory
    source: ModelSource
    size_mb: int | None = None
    repo_id: str = ""
    filename: str = ""
    civitai_model_id: int | None = None
    civitai_version_id: int | None = None
    url: str = ""
    notes: str = ""

    def choice_label(self, *, installed: bool = False) -> str:
        size = f" · {self.size_mb}MB" if self.size_mb else ""
        mark = "  ✓ installed" if installed else ""
        return f"{self.title} [{self.category}]{size}{mark}"