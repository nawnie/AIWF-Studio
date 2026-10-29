from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation


def dilate_mask(mask: Image.Image, amount: int) -> Image.Image:
    """Expand a binary mask — same idea as sd-webui-segment-anything mask dilation."""
    if amount <= 0:
        return mask.convert("L")
    binary = np.array(mask.convert("1"))
    radius = max(1, amount // 2)
    y, x = np.ogrid[: radius * 2 + 1, : radius * 2 + 1]
    center = radius
    kernel = ((x - center) ** 2 + (y - center) ** 2 <= center**2).astype(np.uint8)
    dilated = binary_dilation(binary, structure=kernel)
    return Image.fromarray((dilated.astype(np.uint8) * 255), mode="L")


def overlay_masks(image: Image.Image, mask: Image.Image, *, alpha: float = 0.45) -> Image.Image:
    rgb = np.array(image.convert("RGB"))
    mask_np = np.array(mask.convert("L")) > 127
    color = np.array([91, 141, 239], dtype=np.float32)
    blended = rgb.astype(np.float32)
    blended[mask_np] = blended[mask_np] * (1 - alpha) + color * alpha
    return Image.fromarray(blended.astype(np.uint8))


def mask_from_bool_array(array: np.ndarray) -> Image.Image:
    return Image.fromarray((array.astype(np.uint8) * 255), mode="L")


def select_mask(masks: np.ndarray, index: int) -> Image.Image:
    """Pick one mask from SAM output `(n, h, w)` or `(n, 1, h, w)`."""
    if masks.ndim == 4:
        masks = masks[:, 0]
    if masks.shape[0] == 0:
        raise ValueError("SAM returned no masks")
    index = min(index, masks.shape[0] - 1)
    return mask_from_bool_array(masks[index])