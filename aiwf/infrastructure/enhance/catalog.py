from __future__ import annotations

import logging
from pathlib import Path
from urllib.request import urlretrieve

from aiwf.core.config.settings import RuntimeFlags
from aiwf.core.domain.enhance import EnhanceModel, EnhanceModelKind

logger = logging.getLogger(__name__)

BUILTIN_UPSCALERS: list[dict] = [
    {
        "id": "realesrgan-x4plus",
        "title": "RealESRGAN 4x+",
        "filename": "RealESRGAN_x4plus.pth",
        "architecture": "ESRGAN",
        "scale": 4,
        "download_url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    },
    {
        "id": "realesrgan-x4plus-anime",
        "title": "RealESRGAN 4x+ Anime6B",
        "filename": "RealESRGAN_x4plus_anime_6B.pth",
        "architecture": "ESRGAN",
        "scale": 4,
        "download_url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
    },
    {
        "id": "realesrgan-x2plus",
        "title": "RealESRGAN 2x+",
        "filename": "RealESRGAN_x2plus.pth",
        "architecture": "ESRGAN",
        "scale": 2,
        "download_url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    },
    {
        "id": "realesrgan-general-x4v3",
        "title": "R-ESRGAN General 4xV3",
        "filename": "realesr-general-x4v3.pth",
        "architecture": "ESRGAN",
        "scale": 4,
        "download_url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    },
]

BUILTIN_RESTORERS: list[dict] = [
    {
        "id": "gfpgan-v1.4",
        "title": "GFPGAN v1.4",
        "filename": "GFPGANv1.4.pth",
        "architecture": "GFPGAN",
        "scale": 1,
        "download_url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
    },
    {
        "id": "codeformer-v0.1.0",
        "title": "CodeFormer v0.1.0",
        "filename": "codeformer-v0.1.0.pth",
        "architecture": "CodeFormer",
        "scale": 1,
        "download_url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
    },
]


def _search_roots(flags: RuntimeFlags) -> dict[str, list[Path]]:
    models_dir = flags.resolved_models_dir()
    roots = {
        "upscaler": [
            models_dir / "RealESRGAN",
            models_dir / "ESRGAN",
            models_dir,
        ],
        "restorer": [
            models_dir / "GFPGAN",
            models_dir / "Codeformer",
            models_dir,
        ],
    }
    return roots


def _find_local_file(roots: list[Path], filename: str) -> Path | None:
    for root in roots:
        if not root.exists():
            continue
        direct = root / filename
        if direct.is_file():
            return direct.resolve()
        try:
            for path in root.rglob(filename):
                if path.is_file():
                    return path.resolve()
        except OSError:
            continue
    return None


class EnhanceModelCatalog:
    def __init__(self, flags: RuntimeFlags) -> None:
        self.flags = flags
        self._cache: list[EnhanceModel] | None = None

    def invalidate(self) -> None:
        self._cache = None

    def model_dir(self, kind: EnhanceModelKind) -> Path:
        models_dir = self.flags.resolved_models_dir()
        if kind == EnhanceModelKind.UPSCALER:
            target = models_dir / "RealESRGAN"
        else:
            target = models_dir / "GFPGAN"
        target.mkdir(parents=True, exist_ok=True)
        return target

    def list_models(self, *, kind: EnhanceModelKind | None = None) -> list[EnhanceModel]:
        if self._cache is None:
            self._cache = self._build_catalog()
        if kind is None:
            return list(self._cache)
        return [model for model in self._cache if model.kind == kind]

    def get_model(self, model_id: str) -> EnhanceModel | None:
        for model in self.list_models():
            if model.id == model_id:
                return model
        return None

    def ensure_model_path(self, model: EnhanceModel) -> Path:
        existing = Path(model.path)
        if existing.is_file():
            return existing

        if model.download_url:
            dest_dir = self.model_dir(model.kind)
            dest = dest_dir / model.filename
            if not dest.exists():
                logger.info("Downloading enhance model %s", model.title)
                dest.parent.mkdir(parents=True, exist_ok=True)
                urlretrieve(model.download_url, dest)
            return dest.resolve()

        raise FileNotFoundError(f"Model file not found for {model.title}: {model.path}")

    def _build_catalog(self) -> list[EnhanceModel]:
        roots = _search_roots(self.flags)
        models: list[EnhanceModel] = []
        seen: set[str] = set()

        for entry in BUILTIN_UPSCALERS:
            local = _find_local_file(roots["upscaler"], entry["filename"])
            path = str(local) if local else str(self.model_dir(EnhanceModelKind.UPSCALER) / entry["filename"])
            models.append(
                EnhanceModel(
                    id=entry["id"],
                    title=entry["title"],
                    filename=entry["filename"],
                    path=path,
                    kind=EnhanceModelKind.UPSCALER,
                    architecture=entry["architecture"],
                    scale=entry["scale"],
                    download_url=entry["download_url"],
                )
            )
            seen.add(entry["filename"].lower())

        for entry in BUILTIN_RESTORERS:
            local = _find_local_file(roots["restorer"], entry["filename"])
            if entry["architecture"] == "CodeFormer":
                local = local or _find_local_file(roots["restorer"], "codeformer.pth")
            path = str(local) if local else str(self.model_dir(EnhanceModelKind.RESTORER) / entry["filename"])
            models.append(
                EnhanceModel(
                    id=entry["id"],
                    title=entry["title"],
                    filename=entry["filename"],
                    path=path,
                    kind=EnhanceModelKind.RESTORER,
                    architecture=entry["architecture"],
                    scale=entry["scale"],
                    download_url=entry["download_url"],
                )
            )
            seen.add(entry["filename"].lower())

        for root in roots["upscaler"]:
            models.extend(self._scan_extra(root, EnhanceModelKind.UPSCALER, "ESRGAN", seen))
        for root in roots["restorer"]:
            models.extend(self._scan_extra_restorer(root, seen))

        models.sort(key=lambda item: (item.kind.value, item.title.lower()))
        return models

    def _scan_extra_restorer(self, root: Path, seen: set[str]) -> list[EnhanceModel]:
        if not root.exists():
            return []
        results: list[EnhanceModel] = []
        try:
            paths = sorted(root.rglob("*.pth"))
        except OSError:
            return []
        for path in paths:
            key = path.name.lower()
            if key in seen:
                continue
            seen.add(key)
            architecture = "CodeFormer" if "codeformer" in key else "GFPGAN"
            results.append(
                EnhanceModel(
                    id=path.stem,
                    title=path.stem,
                    filename=path.name,
                    path=str(path.resolve()),
                    kind=EnhanceModelKind.RESTORER,
                    architecture=architecture,
                    scale=1,
                )
            )
        return results

    def _scan_extra(self, root: Path, kind: EnhanceModelKind, architecture: str, seen: set[str]) -> list[EnhanceModel]:
        if not root.exists():
            return []
        results: list[EnhanceModel] = []
        try:
            paths = sorted(root.rglob("*.pth"))
        except OSError:
            return []
        for path in paths:
            key = path.name.lower()
            if key in seen:
                continue
            seen.add(key)
            model_id = path.stem
            results.append(
                EnhanceModel(
                    id=model_id,
                    title=path.stem,
                    filename=path.name,
                    path=str(path.resolve()),
                    kind=kind,
                    architecture=architecture,
                    scale=4 if kind == EnhanceModelKind.UPSCALER else 1,
                )
            )
        return results

    def folder_help(self) -> str:
        models_dir = self.flags.resolved_models_dir()
        return (
            f"**Upscaler models** → `{models_dir / 'RealESRGAN'}`  \n"
            f"**Restoration models** → `{models_dir / 'GFPGAN'}` and `{models_dir / 'Codeformer'}`  \n\n"
            "Built-in models download automatically on first use. "
            "You can also copy `.pth` files into these dedicated folders."
        )
