from __future__ import annotations

import logging
from inspect import signature
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)
GROUNDING_DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"


def ensure_grounding_dino_model() -> None:
    """Warm the Hugging Face cache for text-prompt segmentation."""
    try:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("transformers is required for text-prompt segmentation") from exc

    logger.info("Ensuring GroundingDINO model is cached: %s", GROUNDING_DINO_MODEL_ID)
    AutoProcessor.from_pretrained(GROUNDING_DINO_MODEL_ID)
    AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_DINO_MODEL_ID)


def _post_process_grounded_detection(
    processor: Any,
    outputs: Any,
    input_ids: Any,
    *,
    threshold: float,
    target_sizes: Any,
    labels: list[str],
):
    method = processor.post_process_grounded_object_detection
    params = signature(method).parameters
    kwargs: dict[str, Any] = {
        "text_threshold": threshold,
        "target_sizes": target_sizes,
    }
    if "box_threshold" in params:
        kwargs["box_threshold"] = threshold
    else:
        kwargs["threshold"] = threshold
    if "text_labels" in params:
        kwargs["text_labels"] = [labels]
    return method(outputs, input_ids, **kwargs)


def detect_boxes_from_text(
    image: Image.Image,
    text_prompt: str,
    *,
    threshold: float = 0.25,
) -> np.ndarray:
    """Text → bounding boxes via transformers zero-shot detection (no GroundingDINO compile)."""
    prompt = (text_prompt or "").strip()
    if not prompt:
        return np.zeros((0, 4), dtype=np.float32)

    labels = [part.strip() for part in prompt.replace(",", ".").split(".") if part.strip()]
    if not labels:
        return np.zeros((0, 4), dtype=np.float32)

    try:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("transformers is required for text-prompt segmentation") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(GROUNDING_DINO_MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(GROUNDING_DINO_MODEL_ID).to(device)
    model.eval()

    rgb = image.convert("RGB")
    inputs = processor(images=rgb, text=[labels], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([[rgb.height, rgb.width]], device=device)
    results = _post_process_grounded_detection(
        processor,
        outputs,
        inputs.input_ids,
        labels=labels,
        threshold=threshold,
        target_sizes=target_sizes,
    )
    boxes = results[0]["boxes"].detach().cpu().numpy()
    logger.info("Text prompt %r detected %d boxes", prompt, len(boxes))
    return boxes.astype(np.float32)
