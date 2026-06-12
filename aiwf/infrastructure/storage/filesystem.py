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
    def __init__(self, root: Path, *, settings=None) -> None:
        self.root = root
        self._settings = settings
        self.root.mkdir(parents=True, exist_ok=True)

    def _format(self) -> tuple[str, int]:
        if self._settings is None:
            return "png", 95
        fmt = (getattr(self._settings, "image_format", "png") or "png").lower()
        quality = int(getattr(self._settings, "image_quality", 95) or 95)
        return fmt, max(10, min(100, quality))

    def save(self, image: Image.Image, infotext: str, subdir: str) -> SavedArtifact:
        target_dir = self.root / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        fmt, quality = self._format()

        if fmt in ("jpg", "jpeg"):
            path = target_dir / f"{stamp}.jpg"
            image.convert("RGB").save(path, format="JPEG", quality=quality)
        elif fmt == "webp":
            path = target_dir / f"{stamp}.webp"
            image.save(path, format="WEBP", quality=quality)
        else:
            path = target_dir / f"{stamp}.png"
            pnginfo = _pnginfo_for_save(image, infotext)
            if pnginfo is not None:
                image.save(path, pnginfo=pnginfo)
            else:
                image.save(path)
        return SavedArtifact(path=str(path), infotext=infotext)