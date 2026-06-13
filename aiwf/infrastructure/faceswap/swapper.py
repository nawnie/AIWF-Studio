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


def _face_sex(face) -> str | None:
    """Return 'M'/'F' for a detected face, tolerating insightface variants."""
    sex = getattr(face, "sex", None)
    if isinstance(sex, str) and sex:
        return sex.upper()[0]
    gender = getattr(face, "gender", None)
    if gender is None:
        return None
    # insightface buffalo_l: gender 1 = male, 0 = female.
    return "M" if int(gender) == 1 else "F"


def _gender_filter(faces: list, gender: int) -> list:
    """gender: 0 = any, 1 = female only, 2 = male only."""
    if gender == 1:
        return [f for f in faces if _face_sex(f) == "F"]
    if gender == 2:
        return [f for f in faces if _face_sex(f) == "M"]
    return faces


def _feather_face(original_bgr: np.ndarray, swapped_bgr: np.ndarray, face) -> np.ndarray:
    """Face Mask Correction: soft-blend the swapped face region over the
    original to reduce hard edges / pixelation around the contour.

    Uses a Gaussian-feathered ellipse around the detected bbox — a lightweight
    stand-in for ReActor's parser-based mask that needs no extra model.
    """
    try:
        import cv2
    except Exception:
        return swapped_bgr
    h, w = original_bgr.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in face.bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return swapped_bgr
    mask = np.zeros((h, w), dtype=np.float32)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    ax, ay = max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2)
    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 1.0, -1)
    k = max(3, (min(ax, ay) // 2) | 1)  # odd kernel proportional to face size
    mask = cv2.GaussianBlur(mask, (k, k), 0)[:, :, None]
    blended = swapped_bgr.astype(np.float32) * mask + original_bgr.astype(np.float32) * (1.0 - mask)
    return blended.clip(0, 255).astype(np.uint8)


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
        source_faces_index: list[int] | None = None,
        target_faces_index: list[int] | None = None,
        gender_source: int = 0,
        gender_target: int = 0,
        mask_face: bool = False,
    ) -> Image.Image:
        """Swap the source face onto one (or all) target faces. Returns a new image."""
        self._ensure_loaded()
        target_bgr = _to_bgr(target)
        source_bgr = _to_bgr(source)

        # ----- pick the source face -----
        source_faces = _gender_filter(self._detect(source_bgr), gender_source)
        if not source_faces:
            raise FaceSwapUnavailable("No matching face found in the source image.")
        src_indices = source_faces_index if source_faces_index else [source_index]
        s_idx = src_indices[0]
        if s_idx < 0 or s_idx >= len(source_faces):
            s_idx = 0
        source_face = source_faces[s_idx]

        # ----- pick the target face(s) -----
        target_faces = _gender_filter(self._detect(target_bgr), gender_target)
        if not target_faces:
            raise FaceSwapUnavailable("No matching face found in the target image.")

        wanted = target_faces_index if target_faces_index else (
            [] if (target_index is None or target_index < 0) else [target_index]
        )
        if not wanted:
            targets = target_faces  # all faces
        else:
            targets = []
            for i in wanted:
                if 0 <= i < len(target_faces):
                    targets.append(target_faces[i])
            if not targets:
                raise FaceSwapUnavailable(
                    f"Target face index {wanted} out of range "
                    f"({len(target_faces)} face(s) detected)."
                )

        result = target_bgr
        for face in targets:
            swapped = self._swapper.get(result, face, source_face, paste_back=True)
            result = _feather_face(result, swapped, face) if mask_face else swapped
        return _to_pil(result)

    def unload(self) -> None:
        self._analyzer = None
        self._swapper = None
