from __future__ import annotations

import logging
from contextlib import contextmanager

from PIL import Image, ImageChops, ImageFilter

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.engine import EngineTenant
from aiwf.core.domain.segment import SamModelInfo, SegmentBox, SegmentPoint, SegmentRequest
from aiwf.infrastructure.segment.catalog import ensure_default_sam_model, scan_sam_models
from aiwf.infrastructure.diffusers.mask import blur_mask
from aiwf.infrastructure.segment.mask_ops import dilate_mask, overlay_masks
from aiwf.infrastructure.segment.sam_backend import SamSegmenter
from aiwf.infrastructure.segment.text_boxes import ensure_grounding_dino_model
from aiwf.infrastructure.torch.devices import DeviceManager

logger = logging.getLogger(__name__)


def _feather_mask(mask: Image.Image, radius: int) -> Image.Image:
    """Add a soft outer transition while preserving an opaque mask core.

    A second whole-mask Gaussian blur would merely duplicate ``mask_blur`` and
    could make narrow selections translucent. Feathering instead derives a
    hard core, softens its boundary, and keeps the stronger value from the
    incoming mask at every pixel. Dilation still controls coverage; feather
    controls only the blend ramp around that coverage.
    """
    base = mask.convert("L")
    if radius <= 0:
        return base
    core = base.point(lambda value: 255 if value >= 128 else 0)
    edge = core.filter(ImageFilter.GaussianBlur(radius=max(0.5, radius / 2.0)))
    return ImageChops.lighter(base, edge)


class SegmentService:
    """SAM-based segmentation for inpaint/img2img/workflow masks.

    Workflow and UI patterns inspired by
    https://github.com/continue-revolution/sd-webui-segment-anything
    """

    def __init__(
        self,
        flags: RuntimeFlags,
        settings: UserSettings,
        devices: DeviceManager,
        supervisor=None,
    ) -> None:
        self.flags = flags
        self.settings = settings
        self.devices = devices
        self.supervisor = supervisor
        self._backend = SamSegmenter(devices.device())
        self._catalog: list[SamModelInfo] | None = None

    @contextmanager
    def _gpu_tenant(self, reason: str):
        supervisor = getattr(self, "supervisor", None)
        if supervisor is None:
            yield
            return
        with supervisor.tenant_session(EngineTenant.ENHANCE, reason=reason):
            yield

    def sam_dir(self) -> str:
        return str((self.flags.resolved_models_dir() / "sam").resolve())

    def folder_help(self) -> str:
        return (
            f"**SAM models** → `{self.sam_dir()}`  \n"
            "Place `sam_vit_b_01ec64.pth`, `sam_vit_l_0b3195.pth`, or `sam_vit_h_4b8939.pth` "
            "(same filenames as [sd-webui-segment-anything](https://github.com/continue-revolution/sd-webui-segment-anything))."
        )

    def ensure_default_models(self) -> None:
        ensure_default_sam_model(self.flags)
        ensure_grounding_dino_model()
        self._catalog = None

    def list_models(self) -> list[SamModelInfo]:
        if self._catalog is None:
            self._catalog = scan_sam_models(self.flags)
        return self._catalog

    def refresh_models(self) -> list[SamModelInfo]:
        self._catalog = None
        self._backend.unload()
        return self.list_models()

    def default_model(self) -> SamModelInfo | None:
        models = self.list_models()
        if not models:
            return None
        for preferred in ("sam_vit_b_01ec64", "sam_vit_l_0b3195", "sam_vit_h_4b8939"):
            for model in models:
                if model.id == preferred:
                    return model
        return models[0]

    def resolve_model(self, model_id: str | None) -> SamModelInfo:
        models = self.list_models()
        if not models:
            raise RuntimeError(
                f"No SAM models found in {self.sam_dir()}. "
                "Download a checkpoint from the segment-anything README."
            )
        if model_id:
            for model in models:
                if model.id == model_id or model.filename == model_id:
                    return model
        default = self.default_model()
        assert default is not None
        return default

    def segment(
        self,
        image: Image.Image,
        request: SegmentRequest,
        *,
        model_id: str | None = None,
    ) -> tuple[Image.Image, Image.Image, list[Image.Image], str]:
        if image is None:
            raise ValueError("Upload an image to segment.")

        with self._gpu_tenant("Segment mask"):
            model = self.resolve_model(model_id)
            mask, candidates, status = self._backend.segment(
                image,
                model,
                text_prompt=request.text_prompt,
                box_threshold=request.box_threshold,
                points=request.points,
                box=request.box,
                mask_index=request.mask_index,
                multimask_output=request.multimask_output,
            )
        if request.dilation > 0:
            mask = dilate_mask(mask, request.dilation)
            status += f", dilation={request.dilation}"
        if request.mask_blur > 0:
            mask = blur_mask(mask, request.mask_blur)
            status += f", blur={request.mask_blur}"
        if request.feather > 0:
            mask = _feather_mask(mask, request.feather)
            status += f", feather={request.feather}"

        preview = overlay_masks(image, mask)
        return mask, preview, candidates, status

    def segment_from_workflow_params(
        self,
        image: Image.Image,
        params: dict,
    ) -> tuple[Image.Image, str]:
        points = [
            SegmentPoint(x=int(item["x"]), y=int(item["y"]), label=int(item.get("label", 1)))
            for item in params.get("points", [])
            if "x" in item and "y" in item
        ]
        box = None
        raw_box = params.get("box")
        if isinstance(raw_box, dict):
            box = SegmentBox(
                x1=int(raw_box["x1"]),
                y1=int(raw_box["y1"]),
                x2=int(raw_box["x2"]),
                y2=int(raw_box["y2"]),
            )

        request = SegmentRequest(
            text_prompt=str(params.get("text_prompt", "")),
            box_threshold=float(params.get("box_threshold", 0.3)),
            points=points,
            box=box,
            mask_index=int(params.get("mask_index", 0)),
            dilation=int(params.get("dilation", 0)),
            mask_blur=int(params.get("mask_blur", 4)),
            feather=int(params.get("feather", 6)),
        )
        mask, _preview, _candidates, status = self.segment(
            image,
            request,
            model_id=params.get("model_id"),
        )
        return mask, status

    def unload(self) -> None:
        self._backend.unload()
        self.devices.empty_cache()
