from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from PIL import Image, PngImagePlugin

from aiwf.core.domain.generation import SavedArtifact

_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')


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


def _sanitize(token: str) -> str:
    return _UNSAFE.sub("_", str(token)).strip(". ") or "_"


def render_filename(
    pattern: str,
    *,
    when: datetime | None = None,
    seed: int | None = None,
    index: int | None = None,
    model_name: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> str:
    """Render a filename stem from a template.

    Tokens: [datetime] [date] [time] [seed] [model_name] [width] [height] [seq].
    The default pattern "[datetime]" reproduces the legacy %Y%m%d-%H%M%S name.
    """
    when = when or datetime.now()
    pattern = (pattern or "[datetime]").strip() or "[datetime]"
    replacements = {
        "[datetime]": when.strftime("%Y%m%d-%H%M%S"),
        "[date]": when.strftime("%Y%m%d"),
        "[time]": when.strftime("%H%M%S"),
        "[seed]": "" if seed is None else str(seed),
        "[model_name]": _sanitize(model_name) if model_name else "",
        "[width]": "" if width is None else str(width),
        "[height]": "" if height is None else str(height),
        "[seq]": "" if index is None else str(index),
    }
    out = pattern
    for token, value in replacements.items():
        out = out.replace(token, value)
    out = _UNSAFE.sub("_", out)
    out = re.sub(r"[-_]{2,}", "-", out).strip("-_ ")
    return out or when.strftime("%Y%m%d-%H%M%S")


def make_grid(images: list[Image.Image]) -> Image.Image | None:
    """Tile images into a single contact-sheet grid (None if fewer than 2)."""
    usable = [im for im in images if im is not None]
    if len(usable) < 2:
        return None
    cols = int(len(usable) ** 0.5 + 0.999)
    rows = (len(usable) + cols - 1) // cols
    cell_w = max(im.width for im in usable)
    cell_h = max(im.height for im in usable)
    grid = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    for i, im in enumerate(usable):
        x = (i % cols) * cell_w
        y = (i // cols) * cell_h
        grid.paste(im.convert("RGB"), (x, y))
    return grid


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

    def _pattern(self) -> str:
        return getattr(self._settings, "filename_pattern", "[datetime]") or "[datetime]"

    def _sidecar_enabled(self) -> bool:
        return bool(getattr(self._settings, "save_sidecar_txt", False))

    def _unique_path(self, target_dir: Path, stem: str, ext: str) -> Path:
        """Never overwrite: append -N when a file already exists."""
        candidate = target_dir / f"{stem}.{ext}"
        counter = 1
        while candidate.exists():
            candidate = target_dir / f"{stem}-{counter}.{ext}"
            counter += 1
        return candidate

    def _write_image(self, image: Image.Image, path: Path, infotext: str, fmt: str, quality: int) -> None:
        if fmt in ("jpg", "jpeg"):
            image.convert("RGB").save(path, format="JPEG", quality=quality)
        elif fmt == "webp":
            image.save(path, format="WEBP", quality=quality)
        else:
            pnginfo = _pnginfo_for_save(image, infotext)
            if pnginfo is not None:
                image.save(path, pnginfo=pnginfo)
            else:
                image.save(path)

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
    ) -> SavedArtifact:
        target_dir = self.root / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        fmt, quality = self._format()
        if format_override:
            fmt = str(format_override).lower()
        ext = "jpg" if fmt in ("jpg", "jpeg") else ("webp" if fmt == "webp" else "png")

        if filename_stem:
            stem = _sanitize(filename_stem)
        else:
            stem = render_filename(
                self._pattern(),
                seed=seed,
                index=index,
                model_name=model_name,
                width=getattr(image, "width", None),
                height=getattr(image, "height", None),
            )
        if prefix:
            stem = f"{prefix}{stem}"
        path = self._unique_path(target_dir, stem, ext)
        self._write_image(image, path, infotext, fmt, quality)

        if self._sidecar_enabled() and infotext:
            try:
                path.with_suffix(".txt").write_text(infotext, encoding="utf-8")
            except OSError:
                pass  # sidecar is best-effort; never fail a save over it

        return SavedArtifact(path=str(path), infotext=infotext)

    def save_grid(self, images: list[Image.Image], subdir: str, *, infotext: str = "") -> SavedArtifact | None:
        """Save a contact-sheet grid of a batch. Returns None for <2 images."""
        grid = make_grid(images)
        if grid is None:
            return None
        return self.save(grid, infotext, subdir, prefix="grid-")
