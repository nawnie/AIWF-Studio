from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from PIL import Image

from aiwf.core.domain.segment import SamModelInfo, SegmentBox, SegmentPoint
from aiwf.infrastructure.segment.mask_ops import select_mask
from aiwf.infrastructure.segment.text_boxes import detect_boxes_from_text

logger = logging.getLogger(__name__)


def _union_box_masks(mask_tensor: torch.Tensor) -> np.ndarray:
    """Convert SAM box output `(boxes, candidates, h, w)` into unioned candidates."""
    if mask_tensor.ndim != 4:
        raise ValueError(f"Unexpected SAM mask tensor shape: {tuple(mask_tensor.shape)}")
    return mask_tensor.any(dim=0).cpu().numpy()


class SamSegmenter:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self._models: dict[str, Any] = {}
        self._predictors: dict[str, Any] = {}

    def unload(self) -> None:
        self._predictors.clear()
        self._models.clear()

    def _load_model(self, model_info: SamModelInfo):
        if model_info.id in self._models:
            return self._models[model_info.id]

        try:
            from segment_anything import sam_model_registry
        except ImportError as exc:
            raise RuntimeError(
                "segment_anything is required. Run: pip install segment-anything"
            ) from exc

        arch = model_info.architecture
        if arch not in sam_model_registry:
            raise ValueError(f"Unsupported SAM architecture: {arch}")

        logger.info("Loading SAM %s from %s", model_info.title, model_info.path)
        sam = sam_model_registry[arch](checkpoint=model_info.path)
        sam.to(device=self.device)
        sam.eval()
        self._models[model_info.id] = sam
        return sam

    def _predictor(self, model_info: SamModelInfo):
        if model_info.id in self._predictors:
            return self._predictors[model_info.id]

        from segment_anything import SamPredictor

        sam = self._load_model(model_info)
        predictor = SamPredictor(sam)
        self._predictors[model_info.id] = predictor
        return predictor

    def segment(
        self,
        image: Image.Image,
        model_info: SamModelInfo,
        *,
        text_prompt: str = "",
        box_threshold: float = 0.3,
        points: list[SegmentPoint] | None = None,
        box: SegmentBox | None = None,
        mask_index: int = 0,
        multimask_output: bool = True,
    ) -> tuple[Image.Image, list[Image.Image], str]:
        rgb = np.array(image.convert("RGB"))
        predictor = self._predictor(model_info)
        predictor.set_image(rgb)

        boxes = detect_boxes_from_text(image, text_prompt, threshold=box_threshold) if text_prompt else np.zeros((0, 4))
        point_list = points or []

        masks: np.ndarray
        status: str

        if boxes.shape[0] > 1:
            transformed = predictor.transform.apply_boxes_torch(
                torch.from_numpy(boxes),
                rgb.shape[:2],
            )
            with torch.no_grad():
                mask_tensor, _, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed.to(self.device),
                    multimask_output=True,
                )
            masks = _union_box_masks(mask_tensor)
            status = f"SAM: {boxes.shape[0]} boxes from text prompt"
        else:
            point_coords = None
            point_labels = None
            if point_list:
                point_coords = np.array([[point.x, point.y] for point in point_list])
                point_labels = np.array([point.label for point in point_list])

            sam_box = None
            if box is not None:
                sam_box = np.array([box.x1, box.y1, box.x2, box.y2], dtype=np.float32)
            elif boxes.shape[0] == 1:
                sam_box = boxes[0]

            if point_coords is None and sam_box is None:
                raise ValueError(
                    "Provide a text prompt, click points on the image, or draw a bounding box."
                )

            mask_output, _, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=sam_box,
                multimask_output=multimask_output,
            )
            masks = mask_output[:, None, ...]
            status = f"SAM: {len(point_list)} points, box={'yes' if sam_box is not None else 'no'}"

        candidates = [select_mask(masks, index) for index in range(masks.shape[0])]
        chosen = candidates[min(mask_index, len(candidates) - 1)]
        return chosen, candidates, status
