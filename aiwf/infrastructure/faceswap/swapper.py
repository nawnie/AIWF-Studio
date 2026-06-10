"""ReActor-style face swapping via insightface + the inswapper ONNX model.

Reimplemented from first principles; behavior is inspired by sd-webui-reactor
(see docs/ATTRIBUTION.md). insightface / onnxruntime are imported lazily so the
rest of the app loads even when the optional face-swap stack is not installed.
"""
from __future__ import annotations

import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class FaceSwapUnavailable(RuntimeError):
    """Raised when the optional face-swap dependencies or model are missing."""


def _require_deps():
    try:
        import insightface  # noqa: F401
        import onnxruntime  # noqa: F401
    except Exception as exc:  # pragma: no cover - import-time environment check
        raise FaceSwapUnavailable(
            "Face swap needs the optional packages `insightface` and `onnxruntime` "
            "(or `onnxruntime-gpu`). Install them, then restart."
        ) from exc


def _to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert("RGB"))
    return rgb[:, :, ::-1].copy()


def _to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(bgr[:, :, ::-1].copy(), "RGB")


class FaceSwapper:
    """Loads the analyzer + inswapper model once and reuses them."""

    def __init__(self, model_path: str, *, providers: list[str] | None = None) -> None:
        self.model_path = model_path
        self._providers = providers
        self._analyzer = None
        self._swapper = None

    def _ensure_loaded(self) -> None:
        if self._swapper is not None:
            return
        _require_deps()
        import insightface
        from insightface.app import FaceAnalysis

        providers = self._providers or ["CPUExecutionProvider"]
        logger.info("Loading insightface analyzer (buffalo_l)")
        analyzer = FaceAnalysis(name="buffalo_l", providers=providers)
        analyzer.prepare(ctx_id=0, det_size=(640, 640))
        logger.info("Loading inswapper model %s", self.model_path)
        swapper = insightface.model_zoo.get_model(self.model_path, providers=providers)
        self._analyzer = analyzer
        self._swapper = swapper

    def _detect(self, bgr: np.ndarray) -> list:
        faces = self._analyzer.get(bgr)
        # Stable left-to-right ordering so indices are predictable.
        return sorted(faces, key=lambda f: f.bbox[0])

    def swap(
        self,
        target: Image.Image,
        source: Image.Image,
        *,
        source_index: int = 0,
        target_index: int = -1,
    ) -> Image.Image:
        """Swap the source face onto one (or all) target faces. Returns a new image."""
        self._ensure_loaded()
        target_bgr = _to_bgr(target)
        source_bgr = _to_bgr(source)

        source_faces = self._detect(source_bgr)
        if not source_faces:
            raise FaceSwapUnavailable("No face found in the source image.")
        if source_index >= len(source_faces):
            source_index = 0
        source_face = source_faces[source_index]

        target_faces = self._detect(target_bgr)
        if not target_faces:
            raise FaceSwapUnavailable("No face found in the target image.")

        if target_index is None or target_index < 0:
            targets = target_faces
        else:
            if target_index >= len(target_faces):
                raise FaceSwapUnavailable(
                    f"Target face index {target_index} out of range "
                    f"({len(target_faces)} face(s) detected)."
                )
            targets = [target_faces[target_index]]

        result = target_bgr
        for face in targets:
            result = self._swapper.get(result, face, source_face, paste_back=True)
        return _to_pil(result)

    def unload(self) -> None:
        self._analyzer = None
        self._swapper = None
