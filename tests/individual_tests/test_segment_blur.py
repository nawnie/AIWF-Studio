from __future__ import annotations

from unittest.mock import MagicMock

from PIL import Image

from aiwf.core.domain.segment import SegmentRequest
from aiwf.services.segment import SegmentService


def test_segment_applies_mask_blur_after_dilation(monkeypatch):
    image = Image.new("RGB", (32, 32), color=(128, 128, 128))
    raw_mask = Image.new("L", (32, 32), 0)
    raw_mask.putpixel((16, 16), 255)

    backend = MagicMock()
    backend.segment.return_value = (raw_mask.copy(), [], "ok")

    service = SegmentService.__new__(SegmentService)
    service._backend = backend
    service.resolve_model = MagicMock(return_value=MagicMock(id="sam"))

    request = SegmentRequest(text_prompt="person", dilation=2, mask_blur=6)
    mask, _preview, _candidates, status = service.segment(image, request)

    assert "blur=6" in status
    assert mask.getpixel((16, 16)) > 0
    assert mask.getpixel((16, 16)) < 255