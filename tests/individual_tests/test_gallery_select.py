"""Tests for gallery select → workspace / seed / size send-back."""
from __future__ import annotations

import types
import pytest


class _FakeSettings:
    send_seed_on_click: bool = True
    send_size_on_click: bool = True


class _FakeCtx:
    settings = _FakeSettings()


class _FakeSelectData:
    """Minimal stand-in for gr.SelectData."""
    def __init__(self, index: int, value):
        self.index = index
        self.value = value


# ---------------------------------------------------------------------------
# Extract the handler logic as a plain function for unit-testing
# (mirrors what tab.py registers in gallery.select)
# ---------------------------------------------------------------------------

def _on_gallery_select(ctx, evt, seeds, img_w, img_h):
    """Extracted from tab.py _on_gallery_select."""
    import gradio as gr

    selected_image = evt.value
    if isinstance(selected_image, dict):
        selected_image = selected_image.get("image") or selected_image.get("value")

    seed_update = gr.update()
    width_update = gr.update()
    height_update = gr.update()

    if getattr(ctx.settings, "send_seed_on_click", True) and seeds:
        idx = evt.index if isinstance(evt.index, int) else (evt.index[0] if evt.index else 0)
        if 0 <= idx < len(seeds):
            seed_update = gr.update(value=seeds[idx])

    if getattr(ctx.settings, "send_size_on_click", True) and selected_image is not None:
        try:
            from PIL import Image as _PILImage
            if isinstance(selected_image, _PILImage.Image):
                width_update = gr.update(value=selected_image.width)
                height_update = gr.update(value=selected_image.height)
        except Exception:
            pass

    return selected_image, seed_update, width_update, height_update


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_gallery_select_sends_seed():
    """Clicking the second gallery image sends seed[1]."""
    import gradio as gr
    from PIL import Image

    ctx = _FakeCtx()
    img = Image.new("RGB", (512, 512))
    seeds = [1001, 1002, 1003]
    evt = _FakeSelectData(index=1, value=img)

    selected, seed_upd, w_upd, h_upd = _on_gallery_select(ctx, evt, seeds, 512, 512)

    assert selected is img
    assert seed_upd["value"] == 1002


def test_gallery_select_sends_size():
    """Clicking sends the image dimensions back to width/height."""
    import gradio as gr
    from PIL import Image

    ctx = _FakeCtx()
    img = Image.new("RGB", (768, 432))
    evt = _FakeSelectData(index=0, value=img)

    _, _, w_upd, h_upd = _on_gallery_select(ctx, evt, [999], 512, 512)

    assert w_upd["value"] == 768
    assert h_upd["value"] == 432


def test_gallery_select_no_seed_when_disabled():
    """send_seed_on_click=False → seed update is a no-op."""
    import gradio as gr
    from PIL import Image

    ctx = _FakeCtx()
    ctx.settings = types.SimpleNamespace(send_seed_on_click=False, send_size_on_click=False)
    img = Image.new("RGB", (512, 512))
    evt = _FakeSelectData(index=0, value=img)

    _, seed_upd, w_upd, h_upd = _on_gallery_select(ctx, evt, [1234], 512, 512)

    # Both updates should be empty dicts (gr.update() with no args)
    assert "value" not in seed_upd
    assert "value" not in w_upd


def test_gallery_select_empty_seeds():
    """Empty seed list → seed update is a no-op, no crash."""
    import gradio as gr
    from PIL import Image

    ctx = _FakeCtx()
    img = Image.new("RGB", (256, 256))
    evt = _FakeSelectData(index=0, value=img)

    selected, seed_upd, _, _ = _on_gallery_select(ctx, evt, [], 256, 256)

    assert selected is img
    assert "value" not in seed_upd


def test_gallery_select_index_out_of_range():
    """Index beyond seeds list → no seed update, no crash."""
    import gradio as gr
    from PIL import Image

    ctx = _FakeCtx()
    img = Image.new("RGB", (512, 512))
    evt = _FakeSelectData(index=5, value=img)

    _, seed_upd, _, _ = _on_gallery_select(ctx, evt, [100, 200], 512, 512)

    assert "value" not in seed_upd


def test_gallery_select_dict_value():
    """Gradio may pass the image as a dict with 'image' key."""
    from PIL import Image

    ctx = _FakeCtx()
    img = Image.new("RGB", (512, 512))
    evt = _FakeSelectData(index=0, value={"image": img, "caption": None})

    selected, _, w_upd, h_upd = _on_gallery_select(ctx, evt, [42], 512, 512)

    assert selected is img
    assert w_upd["value"] == 512


def test_gallery_select_none_image():
    """None value → no size update, no crash."""
    ctx = _FakeCtx()
    evt = _FakeSelectData(index=0, value=None)

    selected, _, w_upd, h_upd = _on_gallery_select(ctx, evt, [1], 512, 512)

    assert selected is None
    assert "value" not in w_upd
