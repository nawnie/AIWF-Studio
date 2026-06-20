from __future__ import annotations

from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator


class WorkflowStepType(str, Enum):
    TXT2IMG = "txt2img"
    IMG2IMG = "img2img"
    INPAINT = "inpaint"
    SEGMENT = "segment"
    UPSCALE = "upscale"
    RESTORE = "restore"
    ENHANCE = "enhance"
    PHOTO_RESTORE = "photo_restore"


class WorkflowStep(BaseModel):
    """Serializable workflow node.

    `params` intentionally stays generic because each step type owns its own
    request schema at execution time.
    """

    id: str = ""
    type: WorkflowStepType
    label: str = ""
    params: dict[str, Any] = Field(default_factory=dict)


class WorkflowDefinition(BaseModel):
    """User-authored workflow saved to disk and replayed by services."""

    name: str
    description: str = ""
    version: int = 1
    save_intermediate: bool = True
    steps: list[WorkflowStep] = Field(default_factory=list)

    @model_validator(mode="after")
    def assign_step_ids(self) -> WorkflowDefinition:
        """Backfill stable IDs for older/simple workflow files."""

        for index, step in enumerate(self.steps):
            if not step.id:
                step.id = f"step_{index + 1}"
        return self


class WorkflowStepResult(BaseModel):
    step_id: str
    step_type: WorkflowStepType
    label: str
    infotext: str = ""
    message: str = ""
    image_path: str | None = None
    seed: int | None = None

    model_config = {"arbitrary_types_allowed": True}


class WorkflowRunResult(BaseModel):
    run_id: UUID = Field(default_factory=uuid4)
    workflow_name: str
    steps: list[WorkflowStepResult] = Field(default_factory=list)
    final_image_path: str | None = None
    summary: str = ""

    model_config = {"arbitrary_types_allowed": True}
