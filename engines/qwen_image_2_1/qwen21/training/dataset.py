"""Dataset inspection for Qwen-Image 2.1 LoRA training (pure stdlib + optional Pillow)."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class DatasetReport:
    folder: Path
    images: list[Path] = field(default_factory=list)
    captions_missing: list[Path] = field(default_factory=list)
    rgba_images: list[Path] = field(default_factory=list)
    control_dirs: list[Path] = field(default_factory=list)
    control_missing: dict[str, list[str]] = field(default_factory=dict)  # control dir -> missing basenames
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        lines = [f"{len(self.images)} images in {self.folder}"]
        if self.captions_missing:
            lines.append(f"{len(self.captions_missing)} images have no .txt caption")
        if self.rgba_images:
            lines.append(f"{len(self.rgba_images)} images carry an alpha channel (RGBA training)")
        for cdir in self.control_dirs:
            missing = self.control_missing.get(str(cdir), [])
            lines.append(f"control folder {cdir.name}: {len(missing)} targets without a matching control image")
        lines.extend(f"WARNING: {w}" for w in self.warnings)
        lines.extend(f"ERROR: {e}" for e in self.errors)
        return "\n".join(lines)


def _has_alpha(path: Path) -> bool:
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as im:
            return im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info)
    except Exception:
        return False


def scan_dataset(folder: str | Path, control_dirs: list[str | Path] | None = None,
                 check_alpha: bool = True, min_images: int = 5) -> DatasetReport:
    folder = Path(folder)
    report = DatasetReport(folder=folder)
    if not folder.is_dir():
        report.errors.append(f"dataset folder does not exist: {folder}")
        return report
    report.images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file())
    if not report.images:
        report.errors.append("no .png/.jpg/.jpeg/.webp images found")
        return report
    if len(report.images) < min_images:
        report.warnings.append(f"only {len(report.images)} images; 15-40 varied images is the usual sweet spot")
    for img in report.images:
        if not img.with_suffix(".txt").is_file():
            report.captions_missing.append(img)
        if check_alpha and img.suffix.lower() == ".png" and _has_alpha(img):
            report.rgba_images.append(img)
    if report.captions_missing and len(report.captions_missing) == len(report.images):
        report.warnings.append("no captions at all; use a trigger word / default caption or write captions first")
    names = {p.stem for p in report.images}
    for cdir in control_dirs or []:
        cdir = Path(cdir)
        report.control_dirs.append(cdir)
        if not cdir.is_dir():
            report.errors.append(f"control folder does not exist: {cdir}")
            continue
        have = {p.stem for p in cdir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
        missing = sorted(names - have)
        report.control_missing[str(cdir)] = missing
        if missing:
            report.errors.append(f"{cdir.name}: {len(missing)} target images have no control image with the same "
                                 f"basename (e.g. {missing[0]})")
    return report


def ensure_captions(folder: str | Path, default_caption: str, trigger: str = "", overwrite: bool = False) -> int:
    """Write ``<image>.txt`` for images lacking one. Returns how many files were written."""
    folder = Path(folder)
    text = f"{trigger} {default_caption}".strip() if trigger else default_caption
    written = 0
    for img in sorted(folder.iterdir()):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        cap = img.with_suffix(".txt")
        if cap.exists() and not overwrite:
            continue
        cap.write_text(text, encoding="utf-8")
        written += 1
    return written


def prepend_trigger(folder: str | Path, trigger: str) -> int:
    """Prefix every caption with the trigger word once. Returns the number of captions changed."""
    folder = Path(folder)
    changed = 0
    for cap in sorted(folder.glob("*.txt")):
        body = cap.read_text(encoding="utf-8", errors="replace").strip()
        if body.startswith(trigger):
            continue
        cap.write_text(f"{trigger}, {body}" if body else trigger, encoding="utf-8")
        changed += 1
    return changed


def write_diffsynth_metadata(folder: str | Path, control_dirs: list[str | Path] | None = None,
                             default_caption: str = "", out_name: str | None = None) -> Path:
    """Create DiffSynth's metadata file.

    Text-to-image: ``metadata.csv`` with columns ``image,prompt``.
    Edit (control folders given): ``metadata.json`` rows ``{"image", "edit_image": [...], "prompt"}``
    where paths are relative to the dataset base path.
    """
    folder = Path(folder)
    report = scan_dataset(folder, control_dirs, check_alpha=False, min_images=1)
    if report.errors:
        raise ValueError("; ".join(report.errors))
    rows = []
    for img in report.images:
        cap = img.with_suffix(".txt")
        prompt = cap.read_text(encoding="utf-8", errors="replace").strip() if cap.is_file() else default_caption
        rows.append((img, prompt))
    if control_dirs:
        out = folder / (out_name or "metadata.json")
        data = []
        for img, prompt in rows:
            edits = []
            for cdir in control_dirs:
                cdir = Path(cdir)
                match = next((p for p in cdir.iterdir() if p.stem == img.stem and p.suffix.lower() in IMAGE_EXTS), None)
                if match is None:
                    raise ValueError(f"{cdir}: no control image for {img.name}")
                edits.append(_relative(match, folder))
            data.append({"image": img.name, "edit_image": edits if len(edits) > 1 else edits[0], "prompt": prompt})
        out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return out
    out = folder / (out_name or "metadata.csv")
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image", "prompt"])
        for img, prompt in rows:
            writer.writerow([img.name, prompt])
    return out


def _relative(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()
