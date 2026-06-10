"""ControlNet control-image preprocessors (annotators).

cv2/numpy only — no torch — so the preview path is fast and testable without a
GPU. Modules that require heavy annotator models (depth, openpose, normal, ...)
are intentionally implemented as pass-throughs here: the user supplies an
already-computed control map and selects the matching ControlNet model, exactly
like choosing the ``none`` preprocessor in A1111's sd-webui-controlnet.

Behavior and module names are reimplemented from first principles; only the
public-facing module vocabulary is shared with sd-webui-controlnet for
familiarity (see docs/ATTRIBUTION.md).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PreprocessParams:
    processor_res: int = 512
    threshold_a: float = 100.0
    threshold_b: float = 200.0


def _to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))


def _resize_to_res(image: Image.Image, res: int) -> Image.Image:
    """Scale so the longest side equals ``res`` (annotator working resolution)."""
    res = max(64, int(res))
    w, h = image.size
    if max(w, h) == res:
        return image
    scale = res / float(max(w, h))
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _module_none(image: Image.Image, params: PreprocessParams) -> Image.Image:
    """Pass-through: user supplies a precomputed control map."""
    return _resize_to_res(image, params.processor_res).convert("RGB")


def _module_canny(image: Image.Image, params: PreprocessParams) -> Image.Image:
    resized = _resize_to_res(image, params.processor_res)
    arr = _to_rgb_array(resized)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    low = int(min(params.threshold_a, params.threshold_b))
    high = int(max(params.threshold_a, params.threshold_b))
    edges = cv2.Canny(gray, low, high)
    rgb = np.stack([edges, edges, edges], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def _module_invert(image: Image.Image, params: PreprocessParams) -> Image.Image:
    resized = _resize_to_res(image, params.processor_res)
    arr = _to_rgb_array(resized)
    return Image.fromarray(255 - arr, mode="RGB")


def _module_grayscale(image: Image.Image, params: PreprocessParams) -> Image.Image:
    resized = _resize_to_res(image, params.processor_res)
    gray = cv2.cvtColor(_to_rgb_array(resized), cv2.COLOR_RGB2GRAY)
    return Image.fromarray(np.stack([gray, gray, gray], axis=-1), mode="RGB")


def _module_lineart(image: Image.Image, params: PreprocessParams) -> Image.Image:
    """Coarse lineart: Canny edges inverted to black lines on white."""
    canny = _module_canny(image, params)
    arr = np.asarray(canny)
    return Image.fromarray(255 - arr, mode="RGB")


def _module_scribble(image: Image.Image, params: PreprocessParams) -> Image.Image:
    """Thickened Canny edges — a usable stand-in for scribble control maps."""
    canny = _module_canny(image, params)
    arr = np.asarray(canny.convert("L"))
    kernel = np.ones((3, 3), np.uint8)
    thick = cv2.dilate(arr, kernel, iterations=2)
    return Image.fromarray(np.stack([thick, thick, thick], axis=-1), mode="RGB")


# Modules with a real cv2 annotator implementation.
CV2_MODULES: dict[str, callable] = {
    "none": _module_none,
    "canny": _module_canny,
    "invert": _module_invert,
    "grayscale": _module_grayscale,
    "lineart": _module_lineart,
    "scribble": _module_scribble,
}

# Modules that need an external annotator model we don't ship. They behave as
# pass-throughs so a user-supplied precomputed map still works end-to-end.
PASSTHROUGH_MODULES: tuple[str, ...] = (
    "depth",
    "normal",
    "openpose",
    "softedge",
    "segmentation",
    "tile",
    "reference",
)

# Full ordered vocabulary surfaced to the UI/API.
PREPROCESS_MODULES: list[str] = list(CV2_MODULES.keys()) + list(PASSTHROUGH_MODULES)


def preprocess_control_image(
    image: Image.Image,
    module: str,
    params: PreprocessParams | None = None,
) -> Image.Image:
    """Run the named annotator and return an RGB control image.

    Unknown or annotator-only modules fall back to a resize-only pass-through so
    the pipeline can still consume a user-supplied control map.
    """
    if image is None:
        raise ValueError("A control image is required.")
    params = params or PreprocessParams()
    fn = CV2_MODULES.get(module, _module_none)
    return fn(image, params)
