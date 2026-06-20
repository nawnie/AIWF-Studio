from __future__ import annotations

import logging
import os
import sys
import urllib.request
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from aiwf.core.config.settings import RuntimeFlags
from aiwf.core.domain.engine import EngineTenant
from aiwf.core.domain.faceswap import FaceSwapModelInfo, FaceSwapOptions
from aiwf.core.domain.video import VideoProcessResult
from aiwf.infrastructure.faceswap import FaceSwapper, FaceSwapUnavailable
from aiwf.infrastructure.video import process_video_file

logger = logging.getLogger(__name__)

FACESWAP_EXTENSIONS = {".onnx"}
REACTOR_FACE_MODEL_EXTENSIONS = {".safetensors"}

# ReActor assets repo (single-file ONNX). ~554 MB.
_REACTOR_BASE = "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models"


@dataclass(frozen=True)
class DownloadableFaceSwap:
    key: str
    title: str
    filename: str
    url: str
    size_mb: int


DOWNLOADABLE_FACESWAP: list[DownloadableFaceSwap] = [
    DownloadableFaceSwap(
        key="inswapper_128_fp16",
        title="inswapper_128 fp16 (ReActor) — half size",
        filename="inswapper_128_fp16.onnx",
        url=f"{_REACTOR_BASE}/inswapper_128_fp16.onnx",
        size_mb=264,
    ),
    DownloadableFaceSwap(
        key="inswapper_128",
        title="inswapper_128 (ReActor)",
        filename="inswapper_128.onnx",
        url=f"{_REACTOR_BASE}/inswapper_128.onnx",
        size_mb=529,
    ),
]


class FaceSwapService:
    """ReActor-style face swap: model catalog/download + swap orchestration.

    Workflow inspired by https://github.com/Gourieff/sd-webui-reactor
    (see docs/ATTRIBUTION.md). Heavy onnxruntime/insightface work is isolated in
    the infrastructure layer and loaded lazily.
    Direct swap failures stay explicit; video pipelines can catch
    FaceSwapUnavailable and keep the input/previous output as a soft-fail stage.
    """

    def __init__(self, flags: RuntimeFlags, supervisor=None) -> None:
        self.flags = flags
        self.supervisor = supervisor
        self._swapper: FaceSwapper | None = None
        self._swapper_path: str | None = None
        self._face_model_cache: dict[str, Any] = {}

    @contextmanager
    def _gpu_tenant(self, reason: str):
        """Borrow the enhance tenant while insightface/onnxruntime owns VRAM."""
        supervisor = getattr(self, "supervisor", None)
        if supervisor is None:
            yield
            return
        manager = supervisor.tenant_session(EngineTenant.ENHANCE, reason=reason)
        try:
            manager.__enter__()
        except RuntimeError as exc:
            raise FaceSwapUnavailable(f"GPU busy: {exc}") from exc
        try:
            yield
        except BaseException:
            if not manager.__exit__(*sys.exc_info()):
                raise
        else:
            manager.__exit__(None, None, None)

    def models_dir(self) -> Path:
        return self.flags.resolved_models_dir() / "insightface"

    def ensure_dir(self) -> None:
        self.models_dir().mkdir(parents=True, exist_ok=True)

    def folder_help(self) -> str:
        return (
            f"**Face-swap model** → `{self.models_dir()}`  \n"
            "Download `inswapper_128.onnx` below, or drop it in manually. "
            "Needs the optional `insightface` + `onnxruntime` packages."
        )

    def list_models(self) -> list[FaceSwapModelInfo]:
        root = self.models_dir()
        if not root.exists():
            return []
        return [
            FaceSwapModelInfo.from_path(path)
            for path in sorted(root.glob("*.onnx"), key=lambda p: p.name.lower())
        ]

    def resolve_model_path(self, model_id: str | None) -> Path | None:
        models = self.list_models()
        if not models:
            return None
        for model in models:
            if model_id is None or model.id == model_id or model.title == model_id:
                return Path(model.path)
        # Requested id not installed but a model exists — fall back to the
        # first available so a stale/blank selection still swaps.
        return Path(models[0].path)

    def face_models_dir(self) -> Path:
        return self.flags.resolved_models_dir() / "reactor" / "faces"

    def list_face_models(self) -> list[FaceSwapModelInfo]:
        root = self.face_models_dir()
        if not root.exists():
            return []
        return [
            FaceSwapModelInfo.from_path(path)
            for path in sorted(root.iterdir(), key=lambda p: p.name.lower())
            if path.is_file() and path.suffix.lower() in REACTOR_FACE_MODEL_EXTENSIONS
        ]

    def resolve_face_model_path(self, model_id: str | None) -> Path | None:
        models = self.list_face_models()
        if not models:
            return None
        for model in models:
            path = Path(model.path)
            if model_id is None or model.id == model_id or model.title == model_id or path.name == model_id:
                return path
        return None

    def load_face_model(self, model_id: str | None):
        """Load a saved ReActor face embedding from ``models/reactor/faces``."""
        path = self.resolve_face_model_path(model_id)
        if path is None or not path.is_file():
            raise FaceSwapUnavailable(f"No ReActor face model found in {self.face_models_dir()}.")
        cache_key = str(path)
        cached = self._face_model_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            import numpy as np
            from insightface.app.common import Face
            from safetensors import safe_open
        except Exception as exc:  # pragma: no cover - optional dependency check
            raise FaceSwapUnavailable(
                "Saved ReActor face models need `insightface` and `safetensors` installed."
            ) from exc

        data: dict[str, Any] = {}
        try:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    value = handle.get_tensor(key)
                    if hasattr(value, "detach"):
                        value = value.detach()
                    if hasattr(value, "cpu"):
                        value = value.cpu()
                    if hasattr(value, "numpy"):
                        value = value.numpy()
                    array = np.asarray(value)
                    if key in {"age", "gender"} and array.size == 1:
                        data[key] = int(array.reshape(-1)[0])
                    elif key == "det_score" and array.size == 1:
                        data[key] = float(array.reshape(-1)[0])
                    else:
                        data[key] = array
        except Exception as exc:
            raise FaceSwapUnavailable(f"Could not load ReActor face model: {path.name}") from exc

        face = Face(data)
        self._face_model_cache[cache_key] = face
        return face

    # -- downloads -------------------------------------------------------
    def list_downloadable(self) -> list[DownloadableFaceSwap]:
        return list(DOWNLOADABLE_FACESWAP)

    def is_installed(self, item: DownloadableFaceSwap) -> bool:
        return (self.models_dir() / item.filename).is_file()

    def find_downloadable(self, key: str) -> DownloadableFaceSwap | None:
        for item in DOWNLOADABLE_FACESWAP:
            if item.key == key:
                return item
        return None

    def download_model(self, key: str, *, on_progress: Callable[[int, int], None] | None = None) -> Path:
        item = self.find_downloadable(key)
        if item is None:
            raise ValueError(f"Unknown face-swap model '{key}'")
        self.ensure_dir()
        dest = self.models_dir() / item.filename
        if dest.is_file():
            return dest

        token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
        request = urllib.request.Request(item.url)
        if token:
            request.add_header("Authorization", f"Bearer {token}")

        tmp = dest.with_suffix(dest.suffix + ".part")
        logger.info("Downloading face-swap model %s -> %s", item.title, dest)
        try:
            with urllib.request.urlopen(request) as response:
                total = int(response.headers.get("Content-Length") or 0)
                done = 0
                with open(tmp, "wb") as handle:
                    while True:
                        block = response.read(1024 * 256)
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

    # -- swap ------------------------------------------------------------
    def available(self) -> bool:
        return bool(self.list_models())

    def _video_output_path(self, input_path: str | Path, output_path: str | Path | None = None) -> Path:
        if output_path is not None:
            return Path(output_path)
        root = self.flags.resolved_output_dir() / "faceswap-videos"
        root.mkdir(parents=True, exist_ok=True)
        stem = Path(input_path).stem or "video"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        candidate = root / f"{stem}-faceswap-{stamp}.mp4"
        counter = 1
        while candidate.exists():
            candidate = root / f"{stem}-faceswap-{stamp}-{counter}.mp4"
            counter += 1
        return candidate

    def _provider_list(self) -> list[str]:
        try:
            import onnxruntime

            available = set(onnxruntime.get_available_providers())
        except Exception:
            return ["CPUExecutionProvider"]
        preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return [p for p in preferred if p in available] or ["CPUExecutionProvider"]

    def _get_swapper(self, options: FaceSwapOptions) -> FaceSwapper:
        path = self.resolve_model_path(options.model_id)
        if path is None or not path.is_file():
            raise FaceSwapUnavailable(
                f"No face-swap model in {self.models_dir()}. Download inswapper_128 first."
            )
        if self._swapper is None or self._swapper_path != str(path):
            self._swapper = FaceSwapper(str(path), providers=self._provider_list())
            self._swapper_path = str(path)
        return self._swapper

    @staticmethod
    def _source_indices(options: FaceSwapOptions) -> list[int]:
        indices = list(options.source_faces_index or [])
        single_index = int(options.source_face_index)
        if indices == [0] and single_index != 0:
            return [single_index]
        return indices

    def swap(
        self,
        target: Image.Image,
        source: Image.Image,
        options: FaceSwapOptions | None = None,
        *,
        restore_fn: Callable[[Image.Image], Image.Image] | None = None,
    ) -> Image.Image:
        """Swap ``source``'s face onto ``target``; optionally restore via ``restore_fn``."""
        if target is None or source is None:
            raise FaceSwapUnavailable("Provide both a source face and a target image.")
        options = options or FaceSwapOptions()
        with self._gpu_tenant("Face swap"):
            swapper = self._get_swapper(options)
            result = swapper.swap(
                target,
                source,
                source_index=options.source_face_index,
                target_index=options.target_face_index,
                source_faces_index=self._source_indices(options),
                target_faces_index=list(options.target_faces_index or []),
                gender_source=int(options.gender_source),
                gender_target=int(options.gender_target),
                mask_face=bool(options.mask_face),
            )
            if options.restore_face and restore_fn is not None:
                result = restore_fn(result)
            return result

    def swap_with_face_model(
        self,
        target: Image.Image,
        face_model_id: str,
        options: FaceSwapOptions | None = None,
        *,
        restore_fn: Callable[[Image.Image], Image.Image] | None = None,
    ) -> Image.Image:
        """Swap a saved ReActor face model onto ``target``."""
        if target is None:
            raise FaceSwapUnavailable("Provide a target image.")
        options = options or FaceSwapOptions()
        with self._gpu_tenant("Face swap"):
            source_face = self.load_face_model(face_model_id)
            swapper = self._get_swapper(options)
            result = swapper.swap_with_source_face(
                target,
                source_face,
                target_index=options.target_face_index,
                target_faces_index=list(options.target_faces_index or []),
                gender_target=int(options.gender_target),
                mask_face=bool(options.mask_face),
            )
            if options.restore_face and restore_fn is not None:
                result = restore_fn(result)
            return result

    def _swap_frame_with_source_face(
        self,
        swapper: FaceSwapper,
        frame: Image.Image,
        source_face,
        options: FaceSwapOptions,
        restore_fn: Callable[[Image.Image], Image.Image] | None,
    ) -> Image.Image:
        result = swapper.swap_with_source_face(
            frame,
            source_face,
            target_index=options.target_face_index,
            target_faces_index=list(options.target_faces_index or []),
            gender_target=int(options.gender_target),
            mask_face=bool(options.mask_face),
        )
        if options.restore_face and restore_fn is not None:
            result = restore_fn(result)
        return result

    def swap_video(
        self,
        target_video: str | Path,
        source: Image.Image,
        options: FaceSwapOptions | None = None,
        *,
        output_path: str | Path | None = None,
        restore_fn: Callable[[Image.Image], Image.Image] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
        max_frames: int | None = None,
    ) -> VideoProcessResult:
        """Apply face swap to each frame of ``target_video``.

        The shared video layer is frame-only: the output video does not include
        the source audio track yet.
        """
        if source is None:
            raise FaceSwapUnavailable("Provide a source face image.")
        options = options or FaceSwapOptions()
        dest = self._video_output_path(target_video, output_path)

        with self._gpu_tenant("Face swap video"):
            swapper = self._get_swapper(options)
            source_face = swapper.source_face_from_image(
                source,
                source_index=options.source_face_index,
                source_faces_index=self._source_indices(options),
                gender_source=int(options.gender_source),
            )

            def process_frame(frame: Image.Image, _index: int) -> Image.Image:
                return self._swap_frame_with_source_face(swapper, frame, source_face, options, restore_fn)

            result = process_video_file(
                target_video,
                dest,
                process_frame,
                on_progress=on_progress,
                max_frames=max_frames,
            )
        infotext = f"Face swap video: {options.model_id}"
        return result.model_copy(
            update={
                "infotext": infotext,
                "message": f"Face swap video complete. {result.message}",
            }
        )

    def swap_video_with_face_model(
        self,
        target_video: str | Path,
        face_model_id: str,
        options: FaceSwapOptions | None = None,
        *,
        output_path: str | Path | None = None,
        restore_fn: Callable[[Image.Image], Image.Image] | None = None,
        on_progress: Callable[[int, int], None] | None = None,
        max_frames: int | None = None,
    ) -> VideoProcessResult:
        """Apply a saved ReActor face model to each frame of ``target_video``."""
        options = options or FaceSwapOptions()
        dest = self._video_output_path(target_video, output_path)

        with self._gpu_tenant("Face swap video"):
            source_face = self.load_face_model(face_model_id)
            swapper = self._get_swapper(options)

            def process_frame(frame: Image.Image, _index: int) -> Image.Image:
                return self._swap_frame_with_source_face(swapper, frame, source_face, options, restore_fn)

            result = process_video_file(
                target_video,
                dest,
                process_frame,
                on_progress=on_progress,
                max_frames=max_frames,
            )

        infotext = f"Face swap video: {options.model_id} | face model: {face_model_id}"
        return result.model_copy(
            update={
                "infotext": infotext,
                "message": f"Face swap video complete. {result.message}",
            }
        )

    def unload(self) -> None:
        if self._swapper is not None:
            self._swapper.unload()
        self._swapper = None
        self._swapper_path = None
        self._face_model_cache.clear()
