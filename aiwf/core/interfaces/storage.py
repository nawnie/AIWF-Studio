from __future__ import annotations

from typing import Protocol

from PIL import Image

from aiwf.core.domain.generation import SavedArtifact


class ImageStore(Protocol):
    def save(self, image: Image.Image, infotext: str, subdir: str) -> SavedArtifact: ...