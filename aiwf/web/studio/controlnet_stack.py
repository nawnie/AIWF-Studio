from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image

from aiwf.core.domain.controlnet import ControlNetUnit
from aiwf.infrastructure.diffusers.controlnet_pipe import assert_controlnet_checkpoint_compatible
from aiwf.services.controlnet import ControlNetService


@dataclass(frozen=True)
class StudioControlNetSlot:
    label: str
    enabled: bool
    model_id: str | None
    module: str | None
    image: Any
    weight: float
    guidance_start: float
    guidance_end: float
    threshold_a: float
    threshold_b: float


def build_controlnet_stack(
    *,
    slots: list[StudioControlNetSlot],
    mode: str,
    controlnet: ControlNetService | None = None,
    checkpoint_architecture: str | None = None,
) -> tuple[list[ControlNetUnit], list[Image.Image]]:
    """Validate Studio ControlNet slots and return active units plus images."""
    units: list[ControlNetUnit] = []
    control_images: list[Image.Image] = []
    for slot in slots:
        if not slot.enabled:
            continue
        try:
            if controlnet is not None:
                controlnet.validate_enabled(
                    enabled=True,
                    mode=mode,
                    model_id=slot.model_id,
                    control_image=slot.image,
                )
            elif mode not in ("txt2img", "img2img", "inpaint"):
                raise ValueError("ControlNet is only available in Text, Image2Image, and Inpaint modes.")
            elif not slot.model_id:
                raise ValueError("Select a ControlNet model or disable ControlNet.")
            elif slot.image is None:
                raise ValueError("Upload a control image or disable ControlNet.")
            if controlnet is not None and checkpoint_architecture and slot.model_id:
                resolved = controlnet.resolve_model(slot.model_id)
                if resolved is not None:
                    assert_controlnet_checkpoint_compatible(resolved.path, checkpoint_architecture)
        except ValueError as exc:
            raise ValueError(f"{slot.label}: {exc}") from exc

        if slot.model_id and slot.image is not None and mode in ("txt2img", "img2img", "inpaint"):
            units.append(
                ControlNetUnit(
                    enabled=True,
                    model=slot.model_id,
                    module=slot.module or "none",
                    weight=float(slot.weight),
                    guidance_start=float(slot.guidance_start),
                    guidance_end=float(slot.guidance_end),
                    threshold_a=float(slot.threshold_a),
                    threshold_b=float(slot.threshold_b),
                )
            )
            control_images.append(slot.image)
    return units, control_images
