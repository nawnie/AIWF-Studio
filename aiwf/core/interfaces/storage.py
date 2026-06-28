from __future__ import annotations

from typing import Protocol

from PIL import Image

from aiwf.core.domain.generation import SavedArtifact


class ImageStore(Protocol):
    def save(
        self,
        image: Image.Image,
        infotext: str,
        subdir: str,
        *,
        seed: int | None = None,
        index: int | None = None,
        model_name: str | None = None,
        prefix: str = "",
        filename_stem: str | None = None,
        format_override: str | None = None,
    ) -> SavedArtifact: ...
