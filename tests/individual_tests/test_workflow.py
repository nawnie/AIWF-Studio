from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.enhance import EnhanceModel, EnhanceModelKind
from aiwf.core.domain.generation import GenerationMode, GenerationRequest, GenerationResult, JobRecord, JobState
from aiwf.core.domain.workflow import WorkflowDefinition, WorkflowStep, WorkflowStepType
from aiwf.services.workflow import WorkflowService
from aiwf.services.workflow_executor import validate_workflow
from aiwf.services.workflow_store import BUILTIN_WORKFLOWS, WorkflowStore


@pytest.fixture
def workflow_service(tmp_path: Path):
    flags = RuntimeFlags(data_dir=tmp_path)
    settings = UserSettings(save_images=False)

    generation = MagicMock()
    enhance = MagicMock()
    enhance.store.save.return_value = MagicMock(path=str(tmp_path / "out.png"))
    segment = MagicMock()

    return WorkflowService(flags, settings, generation, enhance, segment), generation, enhance, segment


def test_builtin_workflow_has_steps():
    assert len(BUILTIN_WORKFLOWS) >= 2
    assert BUILTIN_WORKFLOWS[0].steps[0].type == WorkflowStepType.TXT2IMG


def test_builtin_generation_steps_validate_request_defaults():
    mode_by_step = {
        WorkflowStepType.TXT2IMG: GenerationMode.TXT2IMG,
        WorkflowStepType.IMG2IMG: GenerationMode.IMG2IMG,
        WorkflowStepType.INPAINT: GenerationMode.INPAINT,
    }
    for workflow in BUILTIN_WORKFLOWS:
        for step in workflow.steps:
            mode = mode_by_step.get(step.type)
            if mode is not None:
                request = GenerationRequest(mode=mode, **step.params)
                assert request.width % 8 == 0
                assert request.height % 8 == 0
                assert request.seed == step.params.get("seed", -1)


def test_workflow_store_save_and_load(tmp_path: Path):
    store = WorkflowStore(RuntimeFlags(data_dir=tmp_path), UserSettings())
    workflow = WorkflowDefinition(
        name="Test Chain",
        steps=[WorkflowStep(type=WorkflowStepType.UPSCALE, label="Up", params={"scale": 2})],
    )
    path = store.save(workflow)
    assert path.is_file()
    loaded = store.load_by_key(path.stem)
    assert loaded is not None
    assert loaded.name == "Test Chain"


def test_from_json_roundtrip(workflow_service):
    service, _, _, _ = workflow_service
    raw = service.to_json(BUILTIN_WORKFLOWS[0])
    parsed = service.from_json(raw)
    assert parsed.name == BUILTIN_WORKFLOWS[0].name
    assert len(parsed.steps) == len(BUILTIN_WORKFLOWS[0].steps)


def test_validate_requires_seed_for_img2img_first():
    workflow = WorkflowDefinition(
        name="x",
        steps=[WorkflowStep(type=WorkflowStepType.IMG2IMG, params={"prompt": "test"})],
    )
    with pytest.raises(ValueError, match="seed image"):
        validate_workflow(workflow, has_seed_image=False)


def test_executor_runs_segment_then_inpaint(workflow_service):
    service, generation, _enhance, segment = workflow_service
    source = Image.new("RGB", (64, 64), color=(1, 2, 3))
    inpainted = Image.new("RGB", (64, 64), color=(4, 5, 6))
    mask = Image.new("L", (64, 64), 255)

    gen_job = JobRecord(
        request=GenerationRequest(mode=GenerationMode.TXT2IMG, prompt="bag"),
        state=JobState.COMPLETED,
        result=GenerationResult(
            job_id=__import__("uuid").uuid4(),
            images=[source],
            seeds=[1],
            infotexts=[""],
            mode=GenerationMode.TXT2IMG,
        ),
    )
    inpaint_job = JobRecord(
        request=GenerationRequest(mode=GenerationMode.INPAINT, prompt="blue bag"),
        state=JobState.COMPLETED,
        result=GenerationResult(
            job_id=__import__("uuid").uuid4(),
            images=[inpainted],
            seeds=[2],
            infotexts=[""],
            mode=GenerationMode.INPAINT,
        ),
    )
    generation.submit.side_effect = [gen_job, inpaint_job]
    segment.segment_from_workflow_params.return_value = (mask, "segmented")

    workflow = WorkflowDefinition(
        name="chain",
        save_intermediate=False,
        steps=[
            WorkflowStep(type=WorkflowStepType.TXT2IMG, params={"prompt": "bag"}),
            WorkflowStep(type=WorkflowStepType.SEGMENT, params={"text_prompt": "bag"}),
            WorkflowStep(type=WorkflowStepType.INPAINT, params={"prompt": "blue bag"}),
        ],
    )
    run, images = service.run(workflow)
    assert images[-1] is inpainted
    assert len(run.steps) == 3
    segment.segment_from_workflow_params.assert_called_once()


def test_executor_does_not_save_none_for_segment_intermediate(tmp_path: Path):
    settings = UserSettings(save_images=True)
    generation = MagicMock()
    enhance = MagicMock()
    enhance.store.save.side_effect = AssertionError("segment steps do not produce an image to save")
    segment = MagicMock()
    service = WorkflowService(RuntimeFlags(data_dir=tmp_path), settings, generation, enhance, segment)

    source = Image.new("RGB", (64, 64), color=(1, 2, 3))
    mask = Image.new("L", (64, 64), 255)
    segment.segment_from_workflow_params.return_value = (mask, "segmented")

    workflow = WorkflowDefinition(
        name="mask-only",
        save_intermediate=True,
        steps=[
            WorkflowStep(type=WorkflowStepType.SEGMENT, params={"text_prompt": "bag"}),
        ],
    )

    run, images = service.run(workflow, seed_image=source)

    assert images == [source]
    assert run.steps[0].image_path is None
    enhance.store.save.assert_not_called()


def test_executor_runs_txt2img_then_upscale(workflow_service):
    service, generation, enhance, _segment = workflow_service
    fake_image = Image.new("RGB", (64, 64), color=(10, 20, 30))
    job = JobRecord(
        request=GenerationRequest(mode=GenerationMode.TXT2IMG, prompt="cat"),
        state=JobState.COMPLETED,
        result=GenerationResult(
            job_id=job_id if (job_id := __import__("uuid").uuid4()) else job_id,
            images=[fake_image],
            seeds=[42],
            infotexts=["prompt: cat"],
            mode=GenerationMode.TXT2IMG,
        ),
    )
    generation.submit.return_value = job
    upscaled = Image.new("RGB", (128, 128), color=(40, 50, 60))
    enhance.upscale.return_value = upscaled

    workflow = WorkflowDefinition(
        name="mini",
        save_intermediate=False,
        steps=[
            WorkflowStep(type=WorkflowStepType.TXT2IMG, label="Gen", params={"prompt": "cat"}),
            WorkflowStep(
                type=WorkflowStepType.UPSCALE,
                label="Up",
                params={"model_id": "realesrgan-x2plus", "scale": 2},
            ),
        ],
    )

    run, images = service.run(workflow)
    assert len(images) == 2
    assert images[-1] is upscaled
    assert len(run.steps) == 2
    generation.submit.assert_called_once()
    enhance.upscale.assert_called_once()
