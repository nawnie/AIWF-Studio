from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from PIL import Image, ImageDraw

from aiwf.core.domain.generation import GenerationMode, GenerationResult, JobRecord, JobState
from aiwf.core.domain.image_workflow import ImageWorkflowSettings
from aiwf.services.image_workflow import ImageWorkflowService, preset_image_settings


class _Flags:
    def __init__(self, root):
        self.root = root

    def resolved_output_dir(self):
        return self.root


def test_deterministic_image_workflow_saves_output_and_manifest(tmp_path) -> None:
    ctx = SimpleNamespace(flags=_Flags(tmp_path))
    service = ImageWorkflowService(ctx)
    source = Image.new("RGB", (120, 80), (90, 120, 160))
    settings = ImageWorkflowSettings(
        stages=["denoise", "tone", "resize", "export"],
        denoise_radius=1,
        denoise_strength=0.2,
        contrast=1.05,
        saturation=0.9,
        resize_width=60,
        resize_height=0,
        export_format="png",
    )
    result = service.process(source, settings)
    assert result.image.size == (60, 40)
    assert result.output_path.endswith(".png")
    assert result.manifest_path.endswith("job.json")
    assert result.receipt_path == result.manifest_path
    assert len(result.stage_log) == 3


class _FakeSegment:
    def __init__(self):
        self.calls = []

    def segment(self, image, request, *, model_id=None):
        self.calls.append((request, model_id))
        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask).rectangle((16, 16, image.width - 16, image.height - 16), fill=255)
        preview = image.copy()
        return mask, preview, [], "person mask ready"


class _FakeGeneration:
    def __init__(self):
        self.calls = []

    def submit(self, request, init_images=None, mask_images=None):
        self.calls.append((request, init_images, mask_images))
        image = Image.new("RGB", (request.width, request.height), (128, 80, 160))
        return JobRecord(
            request=request,
            state=JobState.COMPLETED,
            result=GenerationResult(
                job_id=uuid4(),
                images=[image],
                seeds=[request.seed],
                infotexts=["inpaint"],
                mode=request.mode,
            ),
        )


def test_segment_to_inpaint_preset_writes_receipt_manifest(tmp_path) -> None:
    segment = _FakeSegment()
    generation = _FakeGeneration()
    ctx = SimpleNamespace(flags=_Flags(tmp_path), segment=segment, generation=generation)
    service = ImageWorkflowService(ctx)
    source = Image.new("RGB", (128, 96), (90, 120, 160))
    settings = preset_image_settings("object_replace").model_copy(
        update={"checkpoint_id": "inpaint-checkpoint", "seed": 44}
    )

    result = service.process(source, settings)

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    request, init_images, mask_images = generation.calls[0]
    assert manifest["receipt_type"] == "image_workflow"
    assert manifest["route"] == "segment-to-inpaint"
    assert manifest["status"] == "completed"
    assert manifest["mask"]["source"] == "auto_mask"
    assert manifest["mask"]["preset"] == "person"
    assert manifest["mask_path"].endswith("mask.png")
    assert manifest["inpaint"]["checkpoint_id"] == "inpaint-checkpoint"
    assert manifest["inpaint"]["seed"] == 44
    assert manifest["receipt_id"]
    assert result.receipt_path == result.manifest_path
    assert Path(manifest["mask_path"]).is_file()
    assert request.mode == GenerationMode.INPAINT
    assert request.save_images is False
    assert init_images[0].size == source.size
    assert mask_images[0].mode == "L"
    assert segment.calls
