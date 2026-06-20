from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

from aiwf.core.config.settings import RuntimeFlags
from aiwf.core.domain.controlnet import ControlNetModelInfo, ControlNetUnit
from aiwf.infrastructure.controlnet.catalog import iter_controlnet_model_paths, resolve_controlnet_roots
from aiwf.infrastructure.controlnet.images import decode_control_image
from aiwf.infrastructure.controlnet.preprocess import (
    PREPROCESS_MODULES,
    PreprocessParams,
    preprocess_control_image,
)
from aiwf.services.model_download_catalog import MODEL_DOWNLOAD_CATALOG

# Preprocessor vocabulary is owned by the preprocess module (single source of truth).
CONTROLNET_MODULES = list(PREPROCESS_MODULES)

logger = logging.getLogger(__name__)

# SD1.5 ControlNet-v1.1 from comfyanonymous/ControlNet-v1-1_fp16_safetensors.
# Light "rank128" checkpoints (~129 MB) are Control LoRA weights — much smaller than
# full fp16 files (~723 MB) and load via PEFT on a scaffold ControlNet config.
_CN_BASE = "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main"
_CN_LIGHT_SIZE_MB = 129


@dataclass(frozen=True)
class DownloadableControlNet:
    key: str
    title: str
    filename: str
    url: str
    preprocessor: str
    size_mb: int
    base: str = "SD1.5"
    repo_id: str = ""
    snapshot: bool = False


def _cn(key: str, title: str, filename: str, preproc: str, size_mb: int) -> "DownloadableControlNet":
    return DownloadableControlNet(
        key=key,
        title=title,
        filename=filename,
        url=f"{_CN_BASE}/{filename}",
        preprocessor=preproc,
        size_mb=size_mb,
    )


DOWNLOADABLE_CONTROLNETS: list[DownloadableControlNet] = [
    _cn(
        "canny",
        "Canny (edges) — Light",
        "control_lora_rank128_v11p_sd15_canny_fp16.safetensors",
        "canny",
        _CN_LIGHT_SIZE_MB,
    ),
    _cn(
        "depth",
        "Depth — Light",
        "control_lora_rank128_v11f1p_sd15_depth_fp16.safetensors",
        "depth",
        _CN_LIGHT_SIZE_MB,
    ),
    _cn(
        "openpose",
        "OpenPose (poses) — Light",
        "control_lora_rank128_v11p_sd15_openpose_fp16.safetensors",
        "openpose",
        _CN_LIGHT_SIZE_MB,
    ),
    _cn(
        "lineart",
        "Lineart — Light",
        "control_lora_rank128_v11p_sd15_lineart_fp16.safetensors",
        "lineart",
        _CN_LIGHT_SIZE_MB,
    ),
    _cn(
        "scribble",
        "Scribble — Light",
        "control_lora_rank128_v11p_sd15_scribble_fp16.safetensors",
        "scribble",
        _CN_LIGHT_SIZE_MB,
    ),
    _cn(
        "softedge",
        "SoftEdge — Light",
        "control_lora_rank128_v11p_sd15_softedge_fp16.safetensors",
        "softedge",
        _CN_LIGHT_SIZE_MB,
    ),
    _cn(
        "normalbae",
        "Normal (BAE) — Light",
        "control_lora_rank128_v11p_sd15_normalbae_fp16.safetensors",
        "normal",
        _CN_LIGHT_SIZE_MB,
    ),
    _cn(
        "seg",
        "Segmentation — Light",
        "control_lora_rank128_v11p_sd15_seg_fp16.safetensors",
        "segmentation",
        _CN_LIGHT_SIZE_MB,
    ),
    _cn(
        "tile",
        "Tile (detail/upscale) — Light",
        "control_lora_rank128_v11f1e_sd15_tile_fp16.safetensors",
        "tile",
        _CN_LIGHT_SIZE_MB,
    ),
    _cn(
        "inpaint",
        "Inpaint — Light",
        "control_lora_rank128_v11p_sd15_inpaint_fp16.safetensors",
        "none",
        _CN_LIGHT_SIZE_MB,
    ),
]


def _catalog_preprocessor(entry_key: str, title: str) -> str:
    text = f"{entry_key} {title}".lower()
    for key, module in (
        ("canny", "canny"),
        ("depth", "depth"),
        ("openpose", "openpose"),
        ("pose", "openpose"),
        ("softedge", "softedge"),
        ("lineart", "lineart"),
        ("tile", "tile"),
        ("normal", "normal"),
        ("scribble", "scribble"),
        ("seg", "segmentation"),
        ("inpaint", "none"),
    ):
        if key in text:
            return module
    return "none"


def _catalog_controlnets() -> list[DownloadableControlNet]:
    items: list[DownloadableControlNet] = []
    seen: set[str] = set()
    for entry in MODEL_DOWNLOAD_CATALOG:
        if entry.category != "controlnet" or entry.key in seen:
            continue
        seen.add(entry.key)
        text = f"{entry.key} {entry.title} {entry.repo_id}".lower()
        base = "SDXL" if "sdxl" in text else "SD1.5"
        filename = entry.filename
        if not filename and entry.url:
            filename = Path(urlparse(entry.url).path).name
        if not filename:
            filename = entry.repo_id.split("/")[-1] if entry.repo_id else entry.key
        items.append(
            DownloadableControlNet(
                key=entry.key,
                title=entry.title,
                filename=filename,
                url=entry.url or (f"https://huggingface.co/{entry.repo_id}" if entry.repo_id else ""),
                preprocessor=_catalog_preprocessor(entry.key, entry.title),
                size_mb=int(entry.size_mb or 0),
                base=base,
                repo_id=entry.repo_id,
                snapshot=entry.snapshot,
            )
        )
    return items or list(DOWNLOADABLE_CONTROLNETS)

class ControlNetService:
    """Catalog and request surface for ControlNet without coupling UI/API to diffusers."""

    def __init__(self, flags: RuntimeFlags) -> None:
        self.flags = flags

    def models_dir(self) -> Path:
        return self.flags.resolved_models_dir() / "ControlNet"

    def annotators_dir(self) -> Path:
        return self.models_dir() / "Annotators"

    def ensure_dir(self) -> None:
        self.models_dir().mkdir(parents=True, exist_ok=True)
        self.annotators_dir().mkdir(parents=True, exist_ok=True)

    def list_models(self) -> list[ControlNetModelInfo]:
        return [ControlNetModelInfo.from_path(path) for path in iter_controlnet_model_paths(self.flags)]

    def model_ids(self) -> list[str]:
        return [model.id for model in self.list_models()]

    def list_modules(self) -> list[str]:
        return list(CONTROLNET_MODULES)

    def resolve_model(self, model_id: str | None) -> ControlNetModelInfo | None:
        if not model_id:
            return None
        for model in self.list_models():
            if model.id == model_id or model.title == model_id:
                return model
        return None

    def validate_enabled(
        self,
        *,
        enabled: bool,
        mode: str,
        model_id: str | None,
        control_image: Image.Image | None,
    ) -> None:
        """Raise ValueError when ControlNet is enabled but the request cannot run."""
        if not enabled:
            return
        if mode not in ("txt2img", "img2img", "inpaint"):
            raise ValueError(
                "ControlNet is only available in Text, Image2Image, and Inpaint modes. "
                "Disable ControlNet or switch mode."
            )
        if not model_id:
            raise ValueError("Select a ControlNet model or disable ControlNet.")
        if self.resolve_model(model_id) is None:
            raise ValueError(
                f"ControlNet model '{model_id}' was not found. "
                "Refresh models or download one in Models → ControlNet."
            )
        if control_image is None:
            raise ValueError("Upload a control image or disable ControlNet.")

    def preprocess(
        self,
        image: Image.Image,
        module: str,
        *,
        processor_res: int = 512,
        threshold_a: float = 100.0,
        threshold_b: float = 200.0,
    ) -> Image.Image:
        """Annotate a source image into a ControlNet control map (for preview/use)."""
        params = PreprocessParams(
            processor_res=int(processor_res),
            threshold_a=float(threshold_a),
            threshold_b=float(threshold_b),
            annotator_dir=str(self.annotators_dir()),
        )
        return preprocess_control_image(image, module or "none", params)

    @staticmethod
    def decode_control_image(value: str | None) -> Image.Image | None:
        """Decode a ControlNetUnit.image (base64 data URL, raw base64, or path)."""
        return decode_control_image(value)

    def active_units(self, units: list[ControlNetUnit] | None) -> list[ControlNetUnit]:
        """Enabled units that name a model we can actually resolve."""
        resolved = []
        for unit in units or []:
            if not unit.enabled:
                continue
            if self.resolve_model(unit.model) is None:
                continue
            resolved.append(unit)
        return resolved

    def list_downloadable(self) -> list[DownloadableControlNet]:
        return _catalog_controlnets()

    def is_installed(self, item: DownloadableControlNet) -> bool:
        if item.snapshot and item.repo_id:
            target = self.models_dir() / item.repo_id.split("/")[-1]
            return target.is_dir() and any(target.iterdir())
        return (self.models_dir() / item.filename).is_file()

    def find_downloadable(self, key: str) -> DownloadableControlNet | None:
        for item in self.list_downloadable():
            if item.key == key:
                return item
        return None

    def download_model(self, key: str, *, on_progress=None) -> Path:
        """Download a catalog ControlNet into models/ControlNet (idempotent).

        Streams to a temp file and renames on success. Sends an optional HF token
        from the HUGGINGFACE_TOKEN / HF_TOKEN env var. on_progress(done, total).
        """
        item = self.find_downloadable(key)
        if item is None:
            raise ValueError(f"Unknown ControlNet '{key}'")
        self.ensure_dir()
        dest = self.models_dir() / item.filename
        if item.snapshot:
            if not item.repo_id:
                raise ValueError(f"ControlNet '{key}' is a folder download but has no Hugging Face repo id.")
            from huggingface_hub import snapshot_download

            target = self.models_dir() / item.repo_id.split("/")[-1]
            if target.is_dir() and any(target.iterdir()):
                return target
            target.mkdir(parents=True, exist_ok=True)
            snapshot_download(repo_id=item.repo_id, local_dir=str(target))
            return target
        if dest.is_file():
            return dest

        token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
        request = urllib.request.Request(item.url)
        if token:
            request.add_header("Authorization", f"Bearer {token}")

        tmp = dest.with_suffix(dest.suffix + ".part")
        logger.info("Downloading ControlNet %s -> %s", item.title, dest)
        try:
            with urllib.request.urlopen(request) as response:
                total = int(response.headers.get("Content-Length") or 0)
                done = 0
                chunk = 1024 * 256
                with open(tmp, "wb") as handle:
                    while True:
                        block = response.read(chunk)
                        if not block:
                            break
                        handle.write(block)
                        done += len(block)
                        if on_progress:
                            on_progress(done, total)
            tmp.replace(dest)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        return dest
