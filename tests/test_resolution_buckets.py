from __future__ import annotations

from PIL import Image

from aiwf.infrastructure.diffusers.mask import composite_inpaint_result, erode_mask
from aiwf.web.studio.resolution import DEFAULT_UPLOAD_BUCKET, MAX_BUCKET_8GB, bucket_label, resize_to_bucket


def test_resize_to_bucket_scales_long_side():
    image = Image.new("RGB", (2000, 1000), color=(10, 20, 30))
    resized, note = resize_to_bucket(image, 768)
    assert resized.width == 768
    assert resized.height == 384
    assert "768" in note


def test_resize_clamps_above_8gb_cap():
    image = Image.new("RGB", (3000, 2000), color=(1, 1, 1))
    resized, _note = resize_to_bucket(image, 1536)
    assert max(resized.size) <= MAX_BUCKET_8GB


def test_bucket_labels_show_aspect_examples():
    label = bucket_label(DEFAULT_UPLOAD_BUCKET, recommended=True)
    assert "768" in label
    assert "768×432" in label
    assert "8GB" in label


def test_resize_to_bucket_original_aligns_to_eight():
    image = Image.new("RGB", (905, 905), color=(1, 2, 3))
    resized, _note = resize_to_bucket(image, 0)
    assert resized.width % 8 == 0
    assert resized.height % 8 == 0


def test_composite_inpaint_seam_erode_changes_edge():
    original = Image.new("RGB", (64, 64), color=(0, 0, 0))
    generated = Image.new("RGB", (64, 64), color=(255, 0, 0))
    mask = Image.new("L", (64, 64), 0)
    for x in range(20, 44):
        for y in range(20, 44):
            mask.putpixel((x, y), 255)

    without = composite_inpaint_result(generated, original, mask, mask_blur=0, seam_erode=0)
    with_erode = composite_inpaint_result(generated, original, mask, mask_blur=4, seam_erode=2)
    assert without.getpixel((10, 32)) == (0, 0, 0)
    assert with_erode.getpixel((10, 32)) == (0, 0, 0)
    assert without.getpixel((32, 32)) == (255, 0, 0)
    assert erode_mask(mask, 2).getpixel((20, 32)) == 0