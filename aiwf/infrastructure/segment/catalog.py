from __future__ import annotations

import logging
from pathlib import Path
from urllib.request import urlretrieve

from aiwf.core.config.settings import RuntimeFlags
from aiwf.core.domain.segment import SamModelInfo

logger = logging.getLogger(__name__)

SAM_EXTENSIONS = {".pth", ".pt"}
KNOWN_SAM_FILES = {
    "sam_vit_h_4b8939.pth": ("vit_h", "SAM ViT-H"),
    "sam_vit_l_0b3195.pth": ("vit_l", "SAM ViT-L"),
    "sam_vit_b_01ec64.pth": ("vit_b", "SAM ViT-B"),
}
DEFAULT_SAM_FILENAME = "sam_vit_b_01ec64.pth"
DEFAULT_SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
def sam_model_type(filename: str) -> str | None:
    """Map checkpoint filename to the official segment_anything registry key."""
    if filename in KNOWN_SAM_FILES:
        return KNOWN_SAM_FILES[filename][0]
    stem = Path(filename).stem
    if stem.startswith("sam_hq_vit_"):
        return stem.removeprefix("sam_hq_")
    if stem.startswith("sam_vit_"):
        parts = stem.split("_")
        if len(parts) >= 3:
            return "_".join(parts[1:3])
    return None


def scan_sam_models(flags: RuntimeFlags) -> list[SamModelInfo]:
    models_dir = flags.resolved_models_dir()
    roots = [models_dir / "sam"]
    seen: set[str] = set()
    results: list[SamModelInfo] = []

    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.iterdir()):
            if not path.is_file() or path.suffix.lower() not in SAM_EXTENSIONS:
                continue
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            arch = sam_model_type(path.name)
            if arch is None:
                logger.debug("Skipping unrecognized SAM file: %s", path.name)
                continue
            seen.add(resolved)
            title = KNOWN_SAM_FILES.get(path.name, (arch, path.stem))[1]
            results.append(
                SamModelInfo(
                    id=path.stem,
                    title=title,
                    filename=path.name,
                    path=resolved,
                    architecture=arch,
                )
            )

    return results


def ensure_default_sam_model(flags: RuntimeFlags, downloader=urlretrieve) -> Path | None:
    """Download the smallest official SAM checkpoint when no SAM model is present."""
    if scan_sam_models(flags):
        return None

    sam_dir = flags.resolved_models_dir() / "sam"
    sam_dir.mkdir(parents=True, exist_ok=True)
    destination = sam_dir / DEFAULT_SAM_FILENAME
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.exists():
        partial.unlink()

    logger.info("Downloading default SAM model: %s", DEFAULT_SAM_FILENAME)
    downloader(DEFAULT_SAM_URL, partial)
    if not partial.exists() or partial.stat().st_size == 0:
        raise RuntimeError(f"Default SAM download failed: {DEFAULT_SAM_FILENAME}")
    partial.replace(destination)
    logger.info("Default SAM model ready: %s", destination)
    return destination
