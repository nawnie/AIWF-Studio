"""ControlNet control-image preprocessors (annotators).

The built-in OpenCV modules are lightweight and always available. Heavier
annotators are optional: when ``controlnet_aux`` is installed, AIWF loads them
from ``models/ControlNet/Annotators`` first and falls back to resized source
images when the optional package or weights are unavailable.

Behavior and module names are reimplemented from first principles; only the
public-facing module vocabulary is shared with sd-webui-controlnet for
familiarity (see docs/ATTRIBUTION.md).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PreprocessParams:
    processor_res: int = 512
    threshold_a: float = 100.0
    threshold_b: float = 200.0
    annotator_dir: str | None = None


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
    """Thickened Canny edges; a usable stand-in for scribble control maps."""
    canny = _module_canny(image, params)
    arr = np.asarray(canny.convert("L"))
    kernel = np.ones((3, 3), np.uint8)
    thick = cv2.dilate(arr, kernel, iterations=2)
    return Image.fromarray(np.stack([thick, thick, thick], axis=-1), mode="RGB")


def _annotator_source(params: PreprocessParams) -> str:
    if params.annotator_dir:
        path = Path(params.annotator_dir)
        if path.exists():
            return str(path)
    return "lllyasviel/Annotators"


@lru_cache(maxsize=16)
def _load_aux_detector(module: str, source: str) -> Any:
    import controlnet_aux as aux  # type: ignore

    attr_by_module = {
        "depth": "MidasDetector",
        "depth_midas": "MidasDetector",
        "depth_zoe": "ZoeDetector",
        "hed": "HEDdetector",
        "lineart": "LineartDetector",
        "lineart_anime": "LineartAnimeDetector",
        "mlsd": "MLSDdetector",
        "normal": "NormalBaeDetector",
        "normalbae": "NormalBaeDetector",
        "openpose": "OpenposeDetector",
        "pidinet": "PidiNetDetector",
        "segmentation": "OneformerDetector",
        "shuffle": "ContentShuffleDetector",
        "softedge": "HEDdetector",
    }
    attr = attr_by_module[module]
    detector_cls = getattr(aux, attr, None)
    if detector_cls is None:
        raise ImportError(f"controlnet_aux does not expose {attr}")
    if hasattr(detector_cls, "from_pretrained"):
        return detector_cls.from_pretrained(source)
    return detector_cls()


def _run_aux_detector(
    image: Image.Image,
    module: str,
    params: PreprocessParams,
    **kwargs,
) -> Image.Image | None:
    try:
        detector = _load_aux_detector(module, _annotator_source(params))
    except Exception:
        return None

    call_kwargs = dict(kwargs)
    call_kwargs.setdefault("detect_resolution", int(params.processor_res))
    call_kwargs.setdefault("image_resolution", int(params.processor_res))
    try:
        out = detector(image.convert("RGB"), **call_kwargs)
    except TypeError:
        try:
            out = detector(image.convert("RGB"))
        except Exception:
            return None
    except Exception:
        return None

    if isinstance(out, Image.Image):
        result = out.convert("RGB")
    else:
        try:
            result = Image.fromarray(np.asarray(out)).convert("RGB")
        except Exception:
            return None
    return _resize_to_res(result, params.processor_res).convert("RGB")


def _aux_or_passthrough(image: Image.Image, params: PreprocessParams, module: str, **kwargs) -> Image.Image:
    annotated = _run_aux_detector(image, module, params, **kwargs)
    if annotated is not None:
        return annotated
    return _module_none(image, params)


def _module_depth(image: Image.Image, params: PreprocessParams) -> Image.Image:
    return _aux_or_passthrough(image, params, "depth")


def _module_depth_midas(image: Image.Image, params: PreprocessParams) -> Image.Image:
    return _aux_or_passthrough(image, params, "depth_midas")


def _module_depth_zoe(image: Image.Image, params: PreprocessParams) -> Image.Image:
    return _aux_or_passthrough(image, params, "depth_zoe")


def _module_hed(image: Image.Image, params: PreprocessParams) -> Image.Image:
    return _aux_or_passthrough(image, params, "hed")


def _module_lineart_anime(image: Image.Image, params: PreprocessParams) -> Image.Image:
    return _aux_or_passthrough(image, params, "lineart_anime")


def _module_mlsd(image: Image.Image, params: PreprocessParams) -> Image.Image:
    return _aux_or_passthrough(
        image,
        params,
        "mlsd",
        thr_v=float(params.threshold_a) / 100.0,
        thr_d=float(params.threshold_b) / 100.0,
    )


def _module_normal(image: Image.Image, params: PreprocessParams) -> Image.Image:
    return _aux_or_passthrough(image, params, "normal")


def _module_openpose(image: Image.Image, params: PreprocessParams) -> Image.Image:
    return _aux_or_passthrough(image, params, "openpose", hand_and_face=True)


def _module_pidinet(image: Image.Image, params: PreprocessParams) -> Image.Image:
    return _aux_or_passthrough(image, params, "pidinet")


def _module_segmentation(image: Image.Image, params: PreprocessParams) -> Image.Image:
    return _aux_or_passthrough(image, params, "segmentation")


def _module_shuffle(image: Image.Image, params: PreprocessParams) -> Image.Image:
    return _aux_or_passthrough(image, params, "shuffle")


def _module_softedge(image: Image.Image, params: PreprocessParams) -> Image.Image:
    return _aux_or_passthrough(image, params, "softedge")


CV2_MODULES: dict[str, Callable[[Image.Image, PreprocessParams], Image.Image]] = {
    "none": _module_none,
    "canny": _module_canny,
    "invert": _module_invert,
    "grayscale": _module_grayscale,
    "lineart": _module_lineart,
    "scribble": _module_scribble,
}

OPTIONAL_AUX_MODULES: dict[str, Callable[[Image.Image, PreprocessParams], Image.Image]] = {
    "depth": _module_depth,
    "depth_midas": _module_depth_midas,
    "depth_zoe": _module_depth_zoe,
    "hed": _module_hed,
    "lineart_anime": _module_lineart_anime,
    "mlsd": _module_mlsd,
    "normal": _module_normal,
    "normalbae": _module_normal,
    "openpose": _module_openpose,
    "pidinet": _module_pidinet,
    "segmentation": _module_segmentation,
    "shuffle": _module_shuffle,
    "softedge": _module_softedge,
}

PREPROCESS_MODULES: list[str] = [
    "none",
    "canny",
    "invert",
    "grayscale",
    "lineart",
    "lineart_anime",
    "scribble",
    "softedge",
    "hed",
    "pidinet",
    "mlsd",
    "depth",
    "depth_midas",
    "depth_zoe",
    "normal",
    "normalbae",
    "openpose",
    "segmentation",
    "shuffle",
    "tile",
    "inpaint",
    "reference",
]


def preprocess_control_image(
    image: Image.Image,
    module: str,
    params: PreprocessParams | None = None,
) -> Image.Image:
    """Run the named annotator and return an RGB control image.

    Unknown or unavailable annotator modules fall back to a resize-only
    pass-through so the pipeline can still consume a user-supplied control map.
    """
    if image is None:
        raise ValueError("A control image is required.")
    params = params or PreprocessParams()
    fn = CV2_MODULES.get(module) or OPTIONAL_AUX_MODULES.get(module) or _module_none
    return fn(image, params)
