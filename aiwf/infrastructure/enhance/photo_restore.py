from __future__ import annotations

import logging
import math

import cv2
import numpy as np
from PIL import Image

from aiwf.core.domain.photo_restore import PhotoRestoreOptions

logger = logging.getLogger(__name__)


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))[:, :, ::-1]


def _bgr_to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(image[:, :, ::-1].astype(np.uint8), "RGB")


def pad_to_multiple(image: Image.Image, multiple: int, *, mode: str = "edge") -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Pad image to the nearest multiple; returns padded image and original crop box."""
    rgb = image.convert("RGB")
    if multiple <= 1:
        return rgb, (0, 0, rgb.width, rgb.height)

    width, height = rgb.size
    target_w = int(math.ceil(width / multiple) * multiple)
    target_h = int(math.ceil(height / multiple) * multiple)
    if target_w == width and target_h == height:
        return rgb, (0, 0, width, height)

    pad_left = (target_w - width) // 2
    pad_top = (target_h - height) // 2
    pad_right = target_w - width - pad_left
    pad_bottom = target_h - height - pad_top

    border_type = {
        "reflect": cv2.BORDER_REFLECT_101,
        "symmetric": cv2.BORDER_REFLECT,
        "constant": cv2.BORDER_CONSTANT,
    }.get(mode, cv2.BORDER_REPLICATE)
    arr = cv2.copyMakeBorder(
        np.array(rgb),
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=border_type,
    )
    padded = Image.fromarray(arr, "RGB")
    crop_box = (pad_left, pad_top, pad_left + width, pad_top + height)
    return padded, crop_box


def crop_to_box(image: Image.Image, crop_box: tuple[int, int, int, int]) -> Image.Image:
    left, top, right, bottom = crop_box
    if (left, top) == (0, 0) and (right, bottom) == image.size:
        return image
    return image.crop(crop_box)


def detect_scratch_mask(image: Image.Image, *, sensitivity: float, dilation: int) -> Image.Image:
    """Heuristic scratch / crease mask for faded prints (clean-room, OpenCV-based)."""
    bgr = _pil_to_bgr(image)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray, 3)
    diff = cv2.absdiff(gray, median)

    threshold = max(6, int(28 * (1.05 - sensitivity)))
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Emphasize thin line-like defects common on aged prints.
    edges = cv2.Canny(gray, max(20, threshold), max(40, threshold * 2))
    mask = cv2.bitwise_or(mask, edges)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    if dilation > 0:
        mask = cv2.dilate(mask, kernel, iterations=dilation)

    coverage = float(np.count_nonzero(mask)) / mask.size
    if coverage > 0.35:
        logger.warning("Scratch mask coverage %.1f%% — suppressing aggressive mask", coverage * 100)
        mask = cv2.erode(mask, kernel, iterations=max(1, dilation))

    return Image.fromarray(mask, mode="L")


def inpaint_scratches(image: Image.Image, mask: Image.Image) -> Image.Image:
    bgr = _pil_to_bgr(image)
    mask_arr = np.array(mask.convert("L"))
    if mask_arr.max() == 0:
        return image
    inpainted = cv2.inpaint(bgr, mask_arr, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return _bgr_to_pil(inpainted)


def global_restore_image(
    image: Image.Image,
    *,
    denoise_strength: float,
    color_boost: float,
) -> Image.Image:
    """Non-generative global restoration: denoise, local contrast, mild sharpening."""
    if denoise_strength <= 0 and color_boost <= 0:
        return image

    bgr = _pil_to_bgr(image)
    working = bgr

    if denoise_strength > 0:
        h = max(3, int(3 + denoise_strength * 7))
        template = max(7, int(7 + denoise_strength * 14))
        search = max(15, int(15 + denoise_strength * 6))
        working = cv2.fastNlMeansDenoisingColored(working, None, h, h, template, search)

    if color_boost > 0:
        lab = cv2.cvtColor(working, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clip = 1.5 + color_boost * 2.5
        tile = max(4, int(8 - color_boost * 4))
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
        l_channel = clahe.apply(l_channel)
        lab = cv2.merge([l_channel, a_channel, b_channel])
        working = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        if color_boost > 0.25:
            blur = cv2.GaussianBlur(working, (0, 0), 1.2)
            working = cv2.addWeighted(working, 1.0 + color_boost * 0.15, blur, -color_boost * 0.15, 0)

    return _bgr_to_pil(working)


def run_photo_restore_stages(
    image: Image.Image,
    options: PhotoRestoreOptions,
    *,
    face_restore_fn,
) -> tuple[Image.Image, list[str]]:
    """Run enabled restoration stages; ``face_restore_fn`` receives a PIL image."""
    if image is None:
        raise ValueError("Upload an image first.")

    working = image.convert("RGB")
    steps: list[str] = []
    crop_box = (0, 0, working.width, working.height)

    if options.pad_multiple > 1:
        working, crop_box = pad_to_multiple(working, options.pad_multiple)
        if crop_box != (0, 0, image.width, image.height):
            steps.append(f"Pad ×{options.pad_multiple}")

    scratch_mask = None
    if options.scratch_detection:
        scratch_mask = detect_scratch_mask(
            working,
            sensitivity=options.scratch_sensitivity,
            dilation=options.scratch_dilation,
        )
        if np.array(scratch_mask).max() > 0:
            steps.append("Scratch detect")
        else:
            scratch_mask = None

    if options.scratch_inpaint and scratch_mask is not None:
        working = inpaint_scratches(working, scratch_mask)
        steps.append("Scratch inpaint")

    if options.global_restore:
        working = global_restore_image(
            working,
            denoise_strength=options.denoise_strength,
            color_boost=options.color_boost,
        )
        steps.append("Global restore")

    if options.face_restore and options.restore is not None:
        working = face_restore_fn(working)
        steps.append(f"Face enhance ({options.restore.model_id})")

    return working, steps, crop_box