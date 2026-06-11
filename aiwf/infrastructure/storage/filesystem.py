from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PIL import Image, PngImagePlugin

from aiwf.core.domain.generation import SavedArtifact


def _pnginfo_for_save(image: Image.Image, infotext: str) -> PngImagePlugin.PngInfo | None:
    """Preserve PNG text chunks — a plain image.save(path) drops metadata."""
    meta = PngImagePlugin.PngInfo()
    added = False

    if hasattr(image, "text") and image.text:
        for key, value in image.text.items():
            if value is not None:
                meta.add_text(key, value)
                added = True

    if infotext and (not hasattr(image, "text") or image.text.get("parameters") != infotext):
        meta.add_text("parameters", infotext)
        added = True

    return meta if added else None


class FilesystemImageStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, image: Image.Image, infotext: str, subdir: str) -> SavedArtifact:
        target_dir = self.root / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = target_dir / f"{stamp}.png"
        pnginfo = _pnginfo_for_save(image, infotext)
        if pnginfo is not None:
            image.save(path, pnginfo=pnginfo)
        else:
            image.save(path)
        return SavedArtifact(path=str(path), infotext=infotext)