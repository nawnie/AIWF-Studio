from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.segment import SegmentRequest
from aiwf.infrastructure.segment.catalog import DEFAULT_SAM_FILENAME, ensure_default_sam_model, sam_model_type, scan_sam_models
from aiwf.infrastructure.segment.mask_ops import dilate_mask, mask_from_bool_array
from aiwf.infrastructure.segment.sam_backend import _union_box_masks
from aiwf.infrastructure.segment.text_boxes import _post_process_grounded_detection
from aiwf.services.segment import SegmentService


def test_sam_model_type_recognizes_standard_filenames():
    assert sam_model_type("sam_vit_b_01ec64.pth") == "vit_b"
    assert sam_model_type("sam_vit_h_4b8939.pth") == "vit_h"


def test_scan_sam_models_finds_file(tmp_path: Path):
    sam_dir = tmp_path / "models" / "sam"
    sam_dir.mkdir(parents=True)
    (sam_dir / "sam_vit_b_01ec64.pth").write_bytes(b"placeholder")

    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models")
    models = scan_sam_models(flags)
    assert len(models) == 1
    assert models[0].architecture == "vit_b"


def test_ensure_default_sam_model_downloads_when_missing(tmp_path: Path):
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models")

    def fake_download(_url, destination):
        Path(destination).write_bytes(b"sam")

    path = ensure_default_sam_model(flags, downloader=fake_download)

    assert path == tmp_path / "models" / "sam" / DEFAULT_SAM_FILENAME
    assert path.read_bytes() == b"sam"


def test_ensure_default_sam_model_skips_when_present(tmp_path: Path):
    sam_dir = tmp_path / "models" / "sam"
    sam_dir.mkdir(parents=True)
    existing = sam_dir / DEFAULT_SAM_FILENAME
    existing.write_bytes(b"already here")
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models")

    downloader = MagicMock()

    assert ensure_default_sam_model(flags, downloader=downloader) is None
    downloader.assert_not_called()


def test_dilate_mask_expands_pixels():
    mask = Image.new("L", (8, 8), 0)
    mask.putpixel((4, 4), 255)
    dilated = dilate_mask(mask, 4)
    assert int(np.array(dilated).sum()) > 255


def test_segment_service_requires_models(tmp_path: Path):
    flags = RuntimeFlags(data_dir=tmp_path)
    devices = MagicMock()
    devices.device.return_value = "cpu"
    service = SegmentService(flags, UserSettings(), devices)
    with pytest.raises(RuntimeError, match="No SAM models"):
        service.resolve_model(None)


@patch("aiwf.services.segment.ensure_grounding_dino_model")
@patch("aiwf.services.segment.ensure_default_sam_model")
def test_segment_service_ensures_default_models(mock_sam, mock_grounding, tmp_path: Path):
    flags = RuntimeFlags(data_dir=tmp_path)
    devices = MagicMock()
    devices.device.return_value = "cpu"
    service = SegmentService(flags, UserSettings(), devices)

    service.ensure_default_models()

    mock_sam.assert_called_once_with(flags)
    mock_grounding.assert_called_once()


def test_grounding_dino_postprocess_supports_current_threshold_signature():
    class Processor:
        def __init__(self):
            self.kwargs = None

        def post_process_grounded_object_detection(
            self,
            outputs,
            input_ids=None,
            threshold=0.25,
            text_threshold=0.25,
            target_sizes=None,
            text_labels=None,
        ):
            self.kwargs = {
                "threshold": threshold,
                "text_threshold": text_threshold,
                "target_sizes": target_sizes,
                "text_labels": text_labels,
            }
            return [{"boxes": []}]

    processor = Processor()
    _post_process_grounded_detection(processor, "outputs", "ids", threshold=0.4, target_sizes="sizes", labels=["cat"])

    assert processor.kwargs == {
        "threshold": 0.4,
        "text_threshold": 0.4,
        "target_sizes": "sizes",
        "text_labels": [["cat"]],
    }


def test_grounding_dino_postprocess_supports_old_box_threshold_signature():
    class Processor:
        def __init__(self):
            self.kwargs = None

        def post_process_grounded_object_detection(
            self,
            outputs,
            input_ids=None,
            box_threshold=0.25,
            text_threshold=0.25,
            target_sizes=None,
        ):
            self.kwargs = {
                "box_threshold": box_threshold,
                "text_threshold": text_threshold,
                "target_sizes": target_sizes,
            }
            return [{"boxes": []}]

    processor = Processor()
    _post_process_grounded_detection(processor, "outputs", "ids", threshold=0.4, target_sizes="sizes", labels=["cat"])

    assert processor.kwargs == {
        "box_threshold": 0.4,
        "text_threshold": 0.4,
        "target_sizes": "sizes",
    }


def test_union_box_masks_combines_detected_text_boxes_by_candidate():
    import torch

    mask_tensor = torch.zeros((2, 3, 6, 6), dtype=torch.bool)
    mask_tensor[0, 0, 1, 1] = True
    mask_tensor[1, 0, 4, 4] = True
    mask_tensor[0, 1, 2, 2] = True

    combined = _union_box_masks(mask_tensor)

    assert combined.shape == (3, 6, 6)
    assert combined[0, 1, 1]
    assert combined[0, 4, 4]
    assert combined[1, 2, 2]


@patch("aiwf.infrastructure.segment.sam_backend.SamSegmenter.segment")
def test_segment_service_delegates(mock_segment, tmp_path: Path):
    sam_dir = tmp_path / "models" / "sam"
    sam_dir.mkdir(parents=True)
    (sam_dir / "sam_vit_b_01ec64.pth").write_bytes(b"x")

    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models")
    devices = MagicMock()
    devices.device.return_value = "cpu"
    service = SegmentService(flags, UserSettings(), devices)

    image = Image.new("RGB", (32, 32), color=(100, 120, 140))
    mask = mask_from_bool_array(np.ones((32, 32), dtype=bool))
    mock_segment.return_value = (mask, [mask], "ok")

    result_mask, preview, candidates, status = service.segment(
        image,
        SegmentRequest(text_prompt="cat"),
    )
    assert result_mask.size == (32, 32)
    assert preview.size == (32, 32)
    assert status.startswith("ok")
