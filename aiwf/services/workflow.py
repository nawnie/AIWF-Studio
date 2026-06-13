from __future__ import annotations

from PIL import Image

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.workflow import WorkflowDefinition, WorkflowRunResult
from aiwf.services.enhance import EnhanceService
from aiwf.services.generation import GenerationService
from aiwf.services.segment import SegmentService
from aiwf.services.workflow_executor import ProgressCallback, WorkflowExecutor
from aiwf.services.workflow_store import WorkflowStore


class WorkflowService:
    """Load/save workflow JSON and execute chained pipelines."""

    def __init__(
        self,
        flags: RuntimeFlags,
        settings: UserSettings,
        generation: GenerationService,
        enhance: EnhanceService,
        segment: SegmentService,
    ) -> None:
        self.store = WorkflowStore(flags, settings)
        self.executor = WorkflowExecutor(generation, enhance, segment, settings)
        self.settings = settings

    def ensure_dir(self) -> None:
        self.store.ensure_dir()

    def list_choices(self) -> list[tuple[str, str]]:
        return self.store.all_choices()

    def load(self, key: str | None) -> WorkflowDefinition | None:
        return self.store.load_by_key(key)

    def save(self, workflow: WorkflowDefinition):
        return self.store.save(workflow)

    def delete(self, key: str) -> bool:
        return self.store.delete(key)

    def to_json(self, workflow: WorkflowDefinition) -> str:
        return self.store.to_json(workflow)

    def from_json(self, raw: str) -> WorkflowDefinition:
        return self.store.from_json(raw)

    def workflows_dir(self) -> str:
        return str(self.store.workflows_dir())

    def run(
        self,
        workflow: WorkflowDefinition,
        *,
        seed_image: Image.Image | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> tuple[WorkflowRunResult, list[Image.Image]]:
        return self.executor.run(workflow, seed_image=seed_image, on_progress=on_progress)