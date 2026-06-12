from PIL import Image

from aiwf.infrastructure.diffusers.mask import (
    align_to_multiple_of_8,
    crop_to_masked,
    editor_from_mask,
    inpaint_session_background,
    mask_from_editor,
    prepare_inpaint_mask,
    resize_for_inpaint,
    resolve_inpaint_mask,
)


def test_prepare_inpaint_mask_from_rgba_alpha():
    mask = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    mask.paste((255, 255, 255, 255), (8, 8, 24, 24))
    prepared = prepare_inpaint_mask(mask)
    assert prepared is not None
    assert prepared.mode == "L"
    assert prepared.getpixel((16, 16)) == 255
    assert prepared.getpixel((0, 0)) == 0


def test_mask_from_editor_layers():
    background = Image.new("RGBA", (64, 64), (50, 50, 50, 255))
    layer = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    layer.paste((255, 255, 255, 200), (10, 10, 30, 30))
    mask = mask_from_editor({"background": background, "layers": [layer], "composite": None})
    assert mask is not None
    assert mask.getpixel((20, 20)) == 255
    assert mask.getpixel((0, 0)) == 0


def test_editor_from_mask_includes_visible_composite_and_extractable_layer():
    background = Image.new("RGB", (32, 32), (20, 20, 20))
    source_mask = Image.new("L", (32, 32), 0)
    source_mask.paste(255, (8, 8, 16, 16))

    editor = editor_from_mask(background, source_mask)
    extracted = mask_from_editor(editor)

    assert editor["composite"] is not None
    assert editor["composite"].getpixel((10, 10)) != background.getpixel((10, 10))
    assert extracted is not None
    assert extracted.getpixel((10, 10)) == 255
    assert extracted.getpixel((0, 0)) == 0


def test_mask_from_editor_background_only_returns_none():
    background = Image.new("RGB", (32, 32), (40, 80, 120))
    assert mask_from_editor({"background": background, "layers": [], "composite": None}) is None


def test_mask_from_editor_supports_path_dict_layers(tmp_path):
    background = Image.new("RGB", (32, 32), (40, 80, 120))
    layer = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    layer.paste((255, 255, 255, 255), (8, 8, 20, 20))
    layer_path = tmp_path / "mask-layer.png"
    layer.save(layer_path)

    mask = mask_from_editor(
        {
            "background": background,
            "layers": [{"path": str(layer_path)}],
            "composite": None,
        }
    )

    assert mask is not None
    assert mask.getpixel((10, 10)) == 255
    assert mask.getpixel((0, 0)) == 0


def test_inpaint_session_background_prefers_original_or_result():
    original = Image.new("RGB", (16, 16), (10, 10, 10))
    result = Image.new("RGB", (16, 16), (200, 200, 200))
    session = {"original": original, "mask": None}

    assert inpaint_session_background("original", result, None, session) is not original
    assert inpaint_session_background("original", result, None, session).getpixel((0, 0)) == (10, 10, 10)
    assert inpaint_session_background("result", result, None, session).getpixel((0, 0)) == (200, 200, 200)


def test_resolve_inpaint_mask_reuses_stored_mask_when_editor_hidden():
    background = Image.new("RGB", (32, 32), (30, 30, 30))
    stored = Image.new("L", (32, 32), 0)
    stored.paste(255, (4, 4, 12, 12))
    session = {"original": background, "mask": stored}
    editor = {"background": background, "layers": [], "composite": None}

    resolved = resolve_inpaint_mask(editor, session, None, background.size, editing_mask=False)

    assert resolved is not None
    assert resolved.getpixel((8, 8)) == 255
    assert resolved.getpixel((0, 0)) == 0


def test_resize_for_inpaint_aligns_dimensions():
    image = Image.new("RGB", (515, 515), (128, 128, 128))
    mask = Image.new("L", (515, 515), 255)
    resized_image, resized_mask, width, height = resize_for_inpaint(image, mask)
    assert (width, height) == align_to_multiple_of_8(515, 515)
    assert resized_image.size == (width, height)
    assert resized_mask.size == (width, height)


def test_crop_to_masked_aligns_non_multiple_of_8_image():
    image = Image.new("RGB", (986, 904), (128, 128, 128))
    mask = Image.new("L", (986, 904), 0)
    mask.paste(255, (200, 150, 800, 700))

    cropped_image, cropped_mask, crop_box = crop_to_masked(image, mask, padding=32)
    crop_w = crop_box[2] - crop_box[0]
    crop_h = crop_box[3] - crop_box[1]

    assert crop_w % 8 == 0
    assert crop_h % 8 == 0
    assert cropped_image.size == (crop_w, crop_h)
    assert cropped_mask.size == (crop_w, crop_h)


def test_crop_to_masked_aligns_full_image_degenerate_case():
    image = Image.new("RGB", (986, 904), (128, 128, 128))
    mask = Image.new("L", (986, 904), 0)

    cropped_image, cropped_mask, crop_box = crop_to_masked(image, mask)
    expected_w, expected_h = align_to_multiple_of_8(986, 904)

    assert crop_box == (0, 0, expected_w, expected_h)
    assert cropped_image.size == (expected_w, expected_h)
    assert cropped_mask.size == (expected_w, expected_h)
