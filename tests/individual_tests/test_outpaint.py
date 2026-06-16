from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from aiwf.infrastructure.diffusers.mask import prepare_outpaint


def _solid(w=40, h=30, color=(10, 20, 30)) -> Image.Image:
    return Image.new("RGB", (w, h), color)


def test_outpaint_expands_canvas_and_masks_border():
    padded, mask = prepare_outpaint(_solid(), left=10, right=5, up=8, down=0, fill="edge", mask_overlap=0)
    assert padded.size == (55, 38)
    assert mask.size == (55, 38)
    m = np.asarray(mask)
    assert m[8:38, 10:50].max() == 0  # untouched original core is black
    assert m[0:8, :].min() == 255     # top band is white
    assert m[:, 0:10].min() == 255    # left band is white
    assert set(np.unique(m)).issubset({0, 255})


def test_outpaint_edge_fill_preserves_core_pixels():
    padded, _ = prepare_outpaint(_solid(), left=10, up=8, fill="edge", mask_overlap=0)
    arr = np.asarray(padded)
    assert (arr[8:38, 10:50] == [10, 20, 30]).all()


def test_outpaint_overlap_reopens_seam():
    _, mask = prepare_outpaint(_solid(), right=10, mask_overlap=4)
    m = np.asarray(mask)
    assert m[:, 36:40].min() == 255  # seam ring inside original near right edge
    assert m[:, 0:30].max() == 0     # far side of original stays black


def test_outpaint_noise_keeps_original():
    padded, _ = prepare_outpaint(_solid(), down=12, fill="noise")
    arr = np.asarray(padded)
    assert (arr[0:30, :] == [10, 20, 30]).all()


def test_outpaint_requires_a_direction():
    with pytest.raises(ValueError):
        prepare_outpaint(_solid())
