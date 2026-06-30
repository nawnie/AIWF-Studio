from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from aiwf import __version__
from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.enhance import (
    EnhanceModel,
    EnhanceModelKind,
    EnhanceResult,
    RestoreOptions,
    UpscaleOptions,
)
from aiwf.core.domain.engine import EngineTenant
from aiwf.core.domain.photo_restore import PhotoRestoreOptions
from aiwf.core.domain.video import VideoProcessResult
from aiwf.infrastructure.enhance.catalog import EnhanceModelCatalog
from aiwf.infrastructure.enhance.loader import load_spandrel_model
from aiwf.infrastructure.enhance.photo_restore import crop_to_box, run_photo_restore_stages
from aiwf.infrastructure.enhance.restore import restore_image
from aiwf.infrastructure.enhance.upscale import upscale_image
from aiwf.infrastructure.storage.filesystem import FilesystemImageStore
from aiwf.infrastructure.torch.devices import DeviceManager
from aiwf.infrastructure.video import process_video_file

logger = logging.getLogger(__name__)


class EnhanceService:
    def __init__(
        self,
        flags: RuntimeFlags,
        settings: UserSettings,
        devices: DeviceManager,
        store: FilesystemImageStore,
        supervisor=None,
    ) -> None:
        self.flags = flags
        self.settings = settings
        self.devices = devices
        self.store = store
        self.catalog = EnhanceModelCatalog(flags)
        self._loaded: dict[str, Any] = {}
        self.supervisor = supervisor

    @contextmanager
    def _gpu_tenant(self, reason: str):
        if self.supervisor is None:
            yield
            return
        with self.supervisor.tenant_session(EngineTenant.ENHANCE, reason=reason):
            yield

    def list_upscalers(self) -> list[EnhanceModel]:
        return self.catalog.list_models(kind=EnhanceModelKind.UPSCALER)

    def list_restorers(self) -> list[EnhanceModel]:
        return self.catalog.list_models(kind=EnhanceModelKind.RESTORER)

    def refresh_catalog(self) -> list[EnhanceModel]:
        self.catalog.invalidate()
        self._loaded.clear()
        return self.catalog.list_models()

    def _load_descriptor(self, model: EnhanceModel):
        if model.id in self._loaded:
            return self._loaded[model.id]

        try:
            import spandrel  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "spandrel is required for upscaling and restoration. "
                "Run: pip install spandrel spandrel-extra-arches opencv-python-headless facexlib"
            ) from exc

        path = self.catalog.ensure_model_path(model)
        device = self.devices.device()
        prefer_half = not self.flags.no_half and device.type == "cuda"
        descriptor = load_spandrel_model(str(path), device=device, prefer_half=prefer_half)
        self._loaded[model.id] = descriptor
        logger.info("Loaded enhance model %s from %s", model.title, path)
        return descriptor

    def upscale(self, image: Image.Image, options: UpscaleOptions) -> Image.Image:
        with self._gpu_tenant("Enhance upscale"):
            model_info = self.catalog.get_model(options.model_id)
            if model_info is None:
                raise ValueError(f"Unknown upscaler: {options.model_id}")
            descriptor = self._load_descriptor(model_info)
            return upscale_image(
                image.convert("RGB"),
                descriptor,
                model_info=model_info,
                options=options,
            )

    def restore(self, image: Image.Image, options: RestoreOptions) -> Image.Image:
        with self._gpu_tenant("Enhance restore"):
            model_info = self.catalog.get_model(options.model_id)
            if model_info is None:
                raise ValueError(f"Unknown restorer: {options.model_id}")
            descriptor = self._load_descriptor(model_info)
            return restore_image(
                image.convert("RGB"),
                descriptor.model,
                model_info=model_info,
                options=options,
                device=self.devices.device(),
            )

    def run_pipeline(
        self,
        image: Image.Image,
        *,
        restore: RestoreOptions | None = None,
        upscale: UpscaleOptions | None = None,
        restore_first: bool = True,
    ) -> tuple[Image.Image, str]:
        if image is None:
            raise ValueError("Upload an image first.")

        with self._gpu_tenant("Enhance pipeline"):
            working = image.convert("RGB")
            steps: list[str] = []

            def apply_restore() -> None:
                nonlocal working
                if restore is None:
                    return
                working = self.restore(working, restore)
                model = self.catalog.get_model(restore.model_id)
                steps.append(f"Restore: {model.title if model else restore.model_id}")

            def apply_upscale() -> None:
                nonlocal working
                if upscale is None:
                    return
                working = self.upscale(working, upscale)
                model = self.catalog.get_model(upscale.model_id)
                steps.append(f"Upscale: {model.title if model else upscale.model_id} ({upscale.scale}x)")

            if restore_first:
                apply_restore()
                apply_upscale()
            else:
                apply_upscale()
                apply_restore()

            infotext = " | ".join(steps) if steps else "Enhance"
            return working, infotext

    def _video_output_path(self, input_path: str | Path, output_path: str | Path | None = None) -> Path:
        if output_path is not None:
            return Path(output_path)
        root = self.store.root / self.settings.enhance_output_subdir / "videos"
        root.mkdir(parents=True, exist_ok=True)
        stem = Path(input_path).stem or "video"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        candidate = root / f"{stem}-enhance-{stamp}.mp4"
        counter = 1
        while candidate.exists():
            candidate = root / f"{stem}-enhance-{stamp}-{counter}.mp4"
            counter += 1
        return candidate

    def run_video_pipeline(
        self,
        input_video: str | Path,
        *,
        output_path: str | Path | None = None,
        restore: RestoreOptions | None = None,
        upscale: UpscaleOptions | None = None,
        restore_first: bool = True,
        on_progress: Callable[[int, int], None] | None = None,
        max_frames: int | None = None,
    ) -> VideoProcessResult:
        """Apply the image Enhance pipeline to each frame of a video.

        Audio is not copied by the shared video layer yet.
        """
        if restore is None and upscale is None:
            raise ValueError("Enable restore and/or upscale for video processing.")
        with self._gpu_tenant("Enhance video"):
            dest = self._video_output_path(input_video, output_path)
            last_infotext = "Enhance video"

            def process_frame(frame: Image.Image, _index: int) -> Image.Image:
                nonlocal last_infotext
                processed, last_infotext = self.run_pipeline(
                    frame,
                    restore=restore,
                    upscale=upscale,
                    restore_first=restore_first,
                )
                return processed

            result = process_video_file(
                input_video,
                dest,
                process_frame,
                on_progress=on_progress,
                max_frames=max_frames,
            )
            infotext = f"Video: {last_infotext}"
            return result.model_copy(
                update={
                    "infotext": infotext,
                    "message": f"Enhance video complete. {result.message}",
                }
            )

    def upscale_video(
        self,
        input_video: str | Path,
        options: UpscaleOptions,
        *,
        output_path: str | Path | None = None,
        on_progress: Callable[[int, int], None] | None = None,
        max_frames: int | None = None,
    ) -> VideoProcessResult:
        return self.run_video_pipeline(
            input_video,
            output_path=output_path,
            upscale=options,
            on_progress=on_progress,
            max_frames=max_frames,
        )

    def run_photo_restore(self, image: Image.Image, options: PhotoRestoreOptions) -> tuple[Image.Image, str]:
        """BOPBTL-inspired staged restoration: scratches -> global -> faces -> optional upscale."""

        with self._gpu_tenant("Photo restore"):
            def face_restore_fn(img: Image.Image) -> Image.Image:
                if options.restore is None:
                    return img
                return self.restore(img, options.restore)

            working, steps, crop_box = run_photo_restore_stages(
                image,
                options,
                face_restore_fn=face_restore_fn,
            )

            if options.upscale is not None:
                working = self.upscale(working, options.upscale)
                model = self.catalog.get_model(options.upscale.model_id)
                steps.append(f"Upscale: {model.title if model else options.upscale.model_id} ({options.upscale.scale}x)")
            elif crop_box != (0, 0, working.width, working.height):
                working = crop_to_box(working, crop_box)

            infotext = " -> ".join(steps) if steps else "Photo restore"
            return working, infotext

    def save_result(
        self,
        image: Image.Image,
        infotext: str,
        *,
        source_image: Image.Image | None = None,
        route: str = "enhance",
        upscale: UpscaleOptions | None = None,
        restore: RestoreOptions | None = None,
        extra: dict[str, Any] | None = None,
    ) -> EnhanceResult:
        if not self.settings.save_images:
            return EnhanceResult(infotext=infotext, message="Done (save disabled in Settings)")
        artifact = self.store.save(image, infotext, self.settings.enhance_output_subdir)
        receipt_path = self._write_receipt(
            Path(artifact.path),
            image,
            infotext,
            source_image=source_image,
            route=route,
            upscale=upscale,
            restore=restore,
            extra=extra,
        )
        return EnhanceResult(
            image_path=artifact.path,
            receipt_path=str(receipt_path),
            infotext=infotext,
            message=f"Saved to {artifact.path}",
        )

    def _write_receipt(
        self,
        output_path: Path,
        image: Image.Image,
        infotext: str,
        *,
        source_image: Image.Image | None,
        route: str,
        upscale: UpscaleOptions | None,
        restore: RestoreOptions | None,
        extra: dict[str, Any] | None,
    ) -> Path:
        payload: dict[str, Any] = {
            "receipt_type": "enhance",
            "receipt_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "app_version": __version__,
            "route": route,
            "infotext": infotext,
            "output": {
                "path": str(output_path),
                "width": int(image.width),
                "height": int(image.height),
                "mode": image.mode,
            },
            "input": (
                {
                    "width": int(source_image.width),
                    "height": int(source_image.height),
                    "mode": source_image.mode,
                }
                if source_image is not None
                else {}
            ),
            "upscale": _option_payload(upscale),
            "restore": _option_payload(restore),
            "settings": {
                "enhance_output_subdir": self.settings.enhance_output_subdir,
                "save_images": bool(self.settings.save_images),
                "upscale_tile_size": int(self.settings.upscale_tile_size),
                "upscale_tile_overlap": int(self.settings.upscale_tile_overlap),
            },
            "extra": extra or {},
        }
        payload["receipt_id"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:20]
        receipt_path = output_path.with_name(f"{output_path.name}.receipt.json")
        receipt_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return receipt_path

    def unload_models(self) -> None:
        self._loaded.clear()
        self.devices.empty_cache()


def _option_payload(option: UpscaleOptions | RestoreOptions | None) -> dict[str, Any]:
    if option is None:
        return {}
    return option.model_dump(mode="json")
