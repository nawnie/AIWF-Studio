from __future__ import annotations

import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from aiwf.core.config.settings import RuntimeFlags
from aiwf.core.domain.controlnet import ControlNetModelInfo, ControlNetUnit
from aiwf.infrastructure.controlnet.images import decode_control_image
from aiwf.infrastructure.controlnet.preprocess import (
    PREPROCESS_MODULES,
    PreprocessParams,
    preprocess_control_image,
)

CONTROLNET_EXTENSIONS = {".safetensors", ".ckpt", ".pt", ".pth", ".bin"}
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


class ControlNetService:
    """Catalog and request surface for ControlNet without coupling UI/API to diffusers."""

    def __init__(self, flags: RuntimeFlags) -> None:
        self.flags = flags

    def models_dir(self) -> Path:
        return self.flags.resolved_models_dir() / "ControlNet"

    def ensure_dir(self) -> None:
        self.models_dir().mkdir(parents=True, exist_ok=True)

    def list_models(self) -> list[ControlNetModelInfo]:
        root = self.models_dir()
        if not root.exists():
            return []
        files = [
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in CONTROLNET_EXTENSIONS
        ]
        return [ControlNetModelInfo.from_path(path) for path in sorted(files, key=lambda item: item.name.lower())]

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
        return list(DOWNLOADABLE_CONTROLNETS)

    def is_installed(self, item: DownloadableControlNet) -> bool:
        return (self.models_dir() / item.filename).is_file()

    def find_downloadable(self, key: str) -> DownloadableControlNet | None:
        for item in DOWNLOADABLE_CONTROLNETS:
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
