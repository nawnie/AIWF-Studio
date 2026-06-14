from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter

EditorValue = dict[str, Any]


def _to_pil(image: Image.Image | np.ndarray | None) -> Image.Image | None:
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, np.ndarray):
        return Image.fromarray(image)
    if isinstance(image, dict):
        for key in ("image", "path", "name"):
            if image.get(key) is not None:
                return _to_pil(image.get(key))
        return None
    if isinstance(image, (str, Path)):
        try:
            with Image.open(image) as loaded:
                return loaded.copy()
        except OSError:
            return None
    return None


def mask_from_editor(editor: EditorValue | Image.Image | np.ndarray | None) -> Image.Image | None:
    """Extract an inpaint mask from a Gradio ImageEditor value or a plain mask image."""
    if editor is None:
        return None

    if isinstance(editor, (Image.Image, np.ndarray)):
        return prepare_inpaint_mask(_to_pil(editor))

    if not isinstance(editor, dict):
        return None

    layers = editor.get("layers") or []
    pil_layers = [_to_pil(layer) for layer in layers if layer is not None]
    pil_layers = [layer for layer in pil_layers if layer is not None]

    if pil_layers:
        width, height = pil_layers[0].size
        combined = Image.new("L", (width, height), 0)
        for layer in pil_layers:
            if layer.size != (width, height):
                layer = layer.resize((width, height), Image.Resampling.NEAREST)
            alpha = layer.convert("RGBA").split()[-1]
            combined = Image.fromarray(
                np.maximum(np.array(combined), np.array(alpha)),
            )
        return prepare_inpaint_mask(combined)

    composite = _to_pil(editor.get("composite"))
    background = _to_pil(editor.get("background"))
    if composite is not None and background is not None:
        if composite.size != background.size:
            composite = composite.resize(background.size, Image.Resampling.LANCZOS)
        diff = Image.fromarray(
            np.any(np.abs(np.array(composite.convert("RGB")) - np.array(background.convert("RGB"))) > 8, axis=-1).astype(np.uint8)
            * 255,
        )
        return prepare_inpaint_mask(diff)

    return None


def prepare_inpaint_mask(mask: Image.Image | None, size: tuple[int, int] | None = None) -> Image.Image | None:
    """Normalize mask to L mode where white pixels are inpainted."""
    if mask is None:
        return None

    if mask.mode == "RGBA":
        gray = mask.split()[-1]
    elif mask.mode == "LA":
        gray = mask.split()[-1]
    else:
        gray = mask.convert("L")

    binary = gray.point(lambda value: 255 if value > 127 else 0)

    if size is not None and binary.size != size:
        binary = binary.resize(size, Image.Resampling.NEAREST)

    return binary


def blur_mask(mask: Image.Image, radius: int) -> Image.Image:
    if radius <= 0:
        return mask
    return mask.filter(ImageFilter.GaussianBlur(radius=radius))


def erode_mask(mask: Image.Image, pixels: int) -> Image.Image:
    """Shrink the white (inpaint) region — helps eat hard edges before composite."""
    if pixels <= 0:
        return mask
    gray = mask.convert("L")
    for _ in range(int(pixels)):
        gray = gray.filter(ImageFilter.MinFilter(3))
    return gray


def composite_inpaint_result(
    generated: Image.Image,
    original: Image.Image,
    mask: Image.Image,
    *,
    mask_blur: int = 0,
    seam_erode: int = 0,
) -> Image.Image:
    """Crop-and-stitch / whole-picture paste with optional seam cleanup.

    Blur softens the composite boundary; erode pulls the paste region slightly
    inward so diffused edge pixels do not leave a visible halo (A1111-style seam control).
    """
    orig = original.convert("RGB")
    gen = generated.convert("RGB")
    if gen.size != orig.size:
        gen = gen.resize(orig.size, Image.Resampling.LANCZOS)

    paste_mask = prepare_inpaint_mask(mask, size=orig.size)
    if paste_mask is None:
        return gen
    if seam_erode > 0:
        paste_mask = erode_mask(paste_mask, seam_erode)
    if mask_blur > 0:
        paste_mask = blur_mask(paste_mask, mask_blur)
    return Image.composite(gen, orig, paste_mask)


def align_to_multiple_of_8(width: int, height: int) -> tuple[int, int]:
    return max(8, (width // 8) * 8), max(8, (height // 8) * 8)


def merge_inpaint_masks(
    painted: Image.Image | None,
    sam_mask: Image.Image | None,
    size: tuple[int, int],
) -> Image.Image | None:
    """Union hand-painted and SAM masks into one inpaint mask."""
    normalized = []
    for candidate in (painted, sam_mask):
        if candidate is None:
            continue
        norm = prepare_inpaint_mask(candidate, size=size)
        if norm is not None and norm.getbbox() is not None:
            normalized.append(norm)
    if not normalized:
        return None
    combined = normalized[0]
    for extra in normalized[1:]:
        combined = Image.fromarray(
            np.maximum(np.array(combined), np.array(extra)),
        )
    return combined


def inpaint_session_background(
    source_choice: str,
    workspace_image: Image.Image | None,
    editor_value: EditorValue | None,
    session: dict,
) -> Image.Image | None:
    """Pick original upload vs latest workspace result for inpaint."""
    original = session.get("original")
    if source_choice == "original" and original is not None:
        return original.copy()
    if workspace_image is not None:
        return workspace_image.copy()
    if isinstance(editor_value, dict) and editor_value.get("background") is not None:
        return editor_value["background"].copy()
    if original is not None:
        return original.copy()
    return None


def _editor_has_paint(editor_value: EditorValue | Image.Image | np.ndarray | None) -> bool:
    if editor_value is None:
        return False
    if not isinstance(editor_value, dict):
        return True
    layers = editor_value.get("layers") or []
    if layers:
        return True
    composite = _to_pil(editor_value.get("composite"))
    background = _to_pil(editor_value.get("background"))
    return composite is not None and background is not None


def resolve_inpaint_mask(
    editor_value: EditorValue | Image.Image | np.ndarray | None,
    session: dict,
    sam_mask: Image.Image | None,
    size: tuple[int, int],
    *,
    editing_mask: bool = False,
) -> Image.Image | None:
    """Combine painted layers, SAM mask, or the last saved session mask."""
    painted = None
    if editing_mask or _editor_has_paint(editor_value):
        painted = mask_from_editor(editor_value)
    mask = merge_inpaint_masks(painted, sam_mask, size)
    if mask is not None and mask.getbbox() is not None:
        return mask
    stored = session.get("mask")
    if stored is not None:
        restored = prepare_inpaint_mask(stored.copy(), size=size)
        if restored is not None and restored.getbbox() is not None:
            return restored
    return mask


def editor_from_mask(background: Image.Image, mask: Image.Image) -> EditorValue:
    """Build a Gradio ImageEditor value from a SAM/binary mask."""
    bg = background.convert("RGB")
    rgba = Image.new("RGBA", bg.size, (0, 0, 0, 0))
    alpha = mask.convert("L").resize(background.size, Image.Resampling.NEAREST)
    rgba.paste((91, 141, 239, 170), mask=alpha)

    composite = bg.convert("RGBA")
    composite.alpha_composite(rgba)
    return {"background": bg, "layers": [rgba], "composite": composite.convert("RGB")}


def resize_for_inpaint(
    image: Image.Image,
    mask: Image.Image,
    width: int | None = None,
    height: int | None = None,
) -> tuple[Image.Image, Image.Image, int, int]:
    """Resize image and mask together, keeping dimensions aligned to multiples of 8."""
    target_w = width or image.width
    target_h = height or image.height
    target_w, target_h = align_to_multiple_of_8(target_w, target_h)

    rgb = image.convert("RGB").resize((target_w, target_h), Image.Resampling.LANCZOS)
    mask_l = prepare_inpaint_mask(mask, size=(target_w, target_h))
    assert mask_l is not None
    return rgb, mask_l, target_w, target_h


def prepare_outpaint(
    image: Image.Image,
    *,
    left: int = 0,
    right: int = 0,
    up: int = 0,
    down: int = 0,
    fill: str = "edge",
    mask_overlap: int = 8,
) -> tuple[Image.Image, Image.Image]:
    """Extend a canvas for outpainting.

    Returns ``(padded_rgb, mask_L)`` where the mask is white over the newly
    added border regions (to be generated) and black over the original image.
    ``fill`` seeds the new pixels: ``edge`` replicates border pixels, ``reflect``
    mirrors them, ``noise`` uses random noise. ``mask_overlap`` extends the mask a
    few pixels into the original so the seam blends.
    """
    left, right, up, down = (max(0, int(v)) for v in (left, right, up, down))
    if left == right == up == down == 0:
        raise ValueError("Choose at least one direction and a pixel amount to outpaint.")

    src = np.asarray(image.convert("RGB"))
    h, w = src.shape[:2]
    new_h, new_w = h + up + down, w + left + right

    pad_width = ((up, down), (left, right), (0, 0))
    if fill == "noise":
        rng = np.random.default_rng()
        canvas = rng.integers(0, 256, size=(new_h, new_w, 3), dtype=np.uint8)
        canvas[up : up + h, left : left + w] = src
    elif fill == "reflect" and h > 1 and w > 1:
        canvas = np.pad(src, pad_width, mode="reflect")
    else:  # "edge" (default) and safe fallback
        canvas = np.pad(src, pad_width, mode="edge")

    mask = np.full((new_h, new_w), 255, dtype=np.uint8)
    overlap = max(0, int(mask_overlap))
    # Keep the untouched core of the original image black (do not regenerate it).
    mask[up:up + h, left:left + w] = 0
    # Re-open a thin seam ring inside the original near extended sides.
    if overlap:
        if up:
            mask[up:up + overlap, left:left + w] = 255
        if down:
            mask[up + h - overlap:up + h, left:left + w] = 255
        if left:
            mask[up:up + h, left:left + overlap] = 255
        if right:
            mask[up:up + h, left + w - overlap:left + w] = 255

    return Image.fromarray(canvas, "RGB"), Image.fromarray(mask, "L")


def apply_masked_content(
    image: Image.Image, mask: Image.Image, content: str = "original"
) -> Image.Image:
    """Apply 'Masked content' prefill to the masked region of the init image.

    Supported content values (A1111-style):
      - "original": leave masked pixels as they are in the source (pipeline will noise them)
      - "fill": replace masked pixels with the average color of the unmasked area
      - "latent noise" / "latent_noise": replace with random noise (so latent starts noisy)
      - "latent nothing" / "latent_nothing" / "nothing": replace with neutral gray (128)
    """
    if not content or content == "original":
        return image.convert("RGB")

    rgb = image.convert("RGB")
    arr = np.asarray(rgb).copy()
    m = np.asarray(prepare_inpaint_mask(mask, size=rgb.size)) > 127
    if not np.any(m):
        return rgb

    c = content.lower().replace(" ", "_")
    if c == "fill":
        unmasked = ~m
        if np.any(unmasked):
            avg = arr[unmasked].mean(axis=0).astype(np.uint8)
        else:
            avg = np.array([128, 128, 128], dtype=np.uint8)
        arr[m] = avg
    elif c in ("latent_noise", "latentnoise"):
        rng = np.random.default_rng()
        noise = rng.integers(0, 256, size=arr.shape, dtype=np.uint8)
        arr[m] = noise[m]
    else:
        # latent_nothing / nothing
        arr[m] = 128
    return Image.fromarray(arr, "RGB")


def _expand_bbox_to_multiple_of_8(
    box: tuple[int, int, int, int], max_w: int, max_h: int, padding: int
) -> tuple[int, int, int, int]:
    left, top, right, bot = [int(v) for v in box]
    left = max(0, left - int(padding))
    top = max(0, top - int(padding))
    right = min(max_w, right + int(padding))
    bot = min(max_h, bot + int(padding))
    w = right - left
    h = bot - top
    target_w = max(8, ((w + 7) // 8) * 8)
    target_h = max(8, ((h + 7) // 8) * 8)
    # Expand, preferring right/bottom, then shift if needed.
    dw = target_w - w
    dh = target_h - h
    right = min(max_w, right + dw)
    bot = min(max_h, bot + dh)
    if (right - left) < target_w:
        left = max(0, right - target_w)
    if (bot - top) < target_h:
        top = max(0, bot - target_h)

    w = right - left
    h = bot - top
    # Image bounds may not be multiples of 8 — shrink the crop to the largest aligned box.
    aligned_w = max(8, (w // 8) * 8)
    aligned_h = max(8, (h // 8) * 8)
    if aligned_w < w or aligned_h < h:
        right = left + aligned_w
        bot = top + aligned_h
        if right > max_w:
            left = max(0, max_w - aligned_w)
            right = left + aligned_w
        if bot > max_h:
            top = max(0, max_h - aligned_h)
            bot = top + aligned_h

    return left, top, right, bot


def crop_to_masked(
    image: Image.Image, mask: Image.Image, padding: int = 32
) -> tuple[Image.Image, Image.Image, tuple[int, int, int, int]]:
    """Crop image + mask to the mask's bounding box (+padding, 8-aligned).

    Returns (cropped_rgb, cropped_mask_L_white_for_inpaint, crop_box).
    If no mask region, returns full image (degenerate case).
    """
    img = image.convert("RGB")
    m_full = prepare_inpaint_mask(mask, size=img.size)
    if m_full is None:
        m_full = Image.new("L", img.size, 255)
    bbox = m_full.getbbox()
    if bbox is None:
        # Nothing to mask — treat as whole image, aligned for diffusers.
        aligned_w, aligned_h = align_to_multiple_of_8(img.width, img.height)
        if (aligned_w, aligned_h) != img.size:
            img = img.resize((aligned_w, aligned_h), Image.Resampling.LANCZOS)
            m_full = prepare_inpaint_mask(m_full, size=(aligned_w, aligned_h))
        return img, m_full, (0, 0, aligned_w, aligned_h)
    left, top, right, bot = _expand_bbox_to_multiple_of_8(bbox, img.width, img.height, padding)
    crop_box = (left, top, right, bot)
    cropped_img = img.crop(crop_box)
    cropped_m = m_full.crop(crop_box)
    cropped_m = prepare_inpaint_mask(cropped_m, size=cropped_img.size)
    return cropped_img, cropped_m, crop_box
