from __future__ import annotations

from types import SimpleNamespace

from PIL import Image

from aiwf.core.domain.image_workflow import ImageWorkflowSettings
from aiwf.services.image_workflow import ImageWorkflowService


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
    assert len(result.stage_log) == 3
