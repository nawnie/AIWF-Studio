from __future__ import annotations

import json
import re
from pathlib import Path

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.workflow import WorkflowDefinition, WorkflowStep, WorkflowStepType

_SAFE_NAME = re.compile(r"[^a-zA-Z0-9._-]+")


def _slugify(name: str) -> str:
    cleaned = _SAFE_NAME.sub("_", name.strip()).strip("._")
    return cleaned or "workflow"


BUILTIN_WORKFLOWS: list[WorkflowDefinition] = [
    WorkflowDefinition(
        name="Generate → Upscale → Restyle",
        description="Txt2img base, RealESRGAN upscale, then img2img style pass.",
        steps=[
            WorkflowStep(
                id="generate",
                type=WorkflowStepType.TXT2IMG,
                label="Generate base image",
                params={
                    "prompt": "a detailed landscape, golden hour",
                    "negative_prompt": "blurry, low quality",
                    "width": 768,
                    "height": 512,
                    "steps": 24,
                    "cfg_scale": 7,
                    "sampler": "euler_a",
                    "seed": -1,
                },
            ),
            WorkflowStep(
                id="upscale",
                type=WorkflowStepType.UPSCALE,
                label="Upscale 2x",
                params={
                    "model_id": "realesrgan-x2plus",
                    "scale": 2,
                    "tile_size": 256,
                    "tile_overlap": 32,
                },
            ),
            WorkflowStep(
                id="restyle",
                type=WorkflowStepType.IMG2IMG,
                label="Oil painting style",
                params={
                    "prompt": "oil painting, thick brushstrokes, masterpiece",
                    "negative_prompt": "photo, realistic",
                    "denoising_strength": 0.45,
                    "steps": 20,
                    "cfg_scale": 7,
                    "sampler": "euler_a",
                },
            ),
        ],
    ),
    WorkflowDefinition(
        name="Segment → Inpaint",
        description="Auto-mask with SAM text prompt, then inpaint the selection.",
        steps=[
            WorkflowStep(
                id="source",
                type=WorkflowStepType.TXT2IMG,
                label="Generate source",
                params={
                    "prompt": "photo of a red backpack on a bench",
                    "width": 768,
                    "height": 512,
                    "steps": 20,
                    "sampler": "euler_a",
                },
            ),
            WorkflowStep(
                id="mask",
                type=WorkflowStepType.SEGMENT,
                label="Mask backpack",
                params={"text_prompt": "backpack", "dilation": 6, "mask_index": 0},
            ),
            WorkflowStep(
                id="inpaint",
                type=WorkflowStepType.INPAINT,
                label="Replace with blue backpack",
                params={
                    "prompt": "blue leather backpack, product photo",
                    "denoising_strength": 0.75,
                    "steps": 24,
                    "mask_blur": 4,
                },
            ),
        ],
    ),
    WorkflowDefinition(
        name="Old photo restore",
        description="Scratch repair, global restore, face enhancement, optional upscale — BOPBTL-inspired stages.",
        steps=[
            WorkflowStep(
                id="restore",
                type=WorkflowStepType.PHOTO_RESTORE,
                label="Restore old photo",
                params={
                    "scratch_detection": True,
                    "scratch_inpaint": True,
                    "scratch_sensitivity": 0.45,
                    "global_restore": True,
                    "denoise_strength": 0.65,
                    "color_boost": 0.55,
                    "face_restore": True,
                    "restore_model_id": "gfpgan-v1.4",
                    "visibility": 0.85,
                    "upscale": True,
                    "upscale_model_id": "realesrgan-x2plus",
                    "scale": 2,
                    "tile_size": 256,
                    "tile_overlap": 32,
                },
            ),
        ],
    ),
    WorkflowDefinition(
        name="Portrait polish",
        description="Generate portrait, restore face, upscale.",
        steps=[
            WorkflowStep(
                id="generate",
                type=WorkflowStepType.TXT2IMG,
                label="Portrait",
                params={
                    "prompt": "portrait photo of a woman, soft light, 85mm",
                    "negative_prompt": "blurry, deformed",
                    "width": 512,
                    "height": 768,
                    "steps": 28,
                    "cfg_scale": 6.5,
                    "sampler": "euler_a",
                },
            ),
            WorkflowStep(
                id="enhance",
                type=WorkflowStepType.ENHANCE,
                label="Restore + upscale",
                params={
                    "restore": True,
                    "upscale": True,
                    "restore_first": True,
                    "restore_model_id": "gfpgan-v1.4",
                    "upscale_model_id": "realesrgan-x4plus",
                    "scale": 2,
                    "visibility": 1.0,
                    "tile_size": 256,
                    "tile_overlap": 32,
                },
            ),
        ],
    ),
]


class WorkflowStore:
    def __init__(self, flags: RuntimeFlags, settings: UserSettings) -> None:
        self.flags = flags
        self.settings = settings

    def workflows_dir(self) -> Path:
        return (self.flags.data_dir / self.settings.workflows_dir).resolve()

    def ensure_dir(self) -> None:
        self.workflows_dir().mkdir(parents=True, exist_ok=True)

    def list_saved(self) -> list[tuple[str, str]]:
        root = self.workflows_dir()
        if not root.exists():
            return []
        files: list[tuple[str, str]] = []
        for path in sorted(root.glob("*.json")):
            try:
                workflow = self.load_file(path)
                files.append((workflow.name, path.stem))
            except (json.JSONDecodeError, ValueError):
                files.append((path.stem, path.stem))
        return files

    def list_builtin(self) -> list[tuple[str, str]]:
        return [(workflow.name, f"builtin:{workflow.name}") for workflow in BUILTIN_WORKFLOWS]

    def all_choices(self) -> list[tuple[str, str]]:
        return self.list_builtin() + self.list_saved()

    def load_by_key(self, key: str | None) -> WorkflowDefinition | None:
        if not key:
            return None
        if key.startswith("builtin:"):
            name = key.removeprefix("builtin:")
            for workflow in BUILTIN_WORKFLOWS:
                if workflow.name == name:
                    return workflow.model_copy(deep=True)
            return None
        return self.load_file(self.workflows_dir() / f"{key}.json")

    def load_file(self, path: Path) -> WorkflowDefinition:
        data = json.loads(path.read_text(encoding="utf-8"))
        return WorkflowDefinition.model_validate(data)

    def save(self, workflow: WorkflowDefinition) -> Path:
        self.ensure_dir()
        path = self.workflows_dir() / f"{_slugify(workflow.name)}.json"
        path.write_text(workflow.model_dump_json(indent=2), encoding="utf-8")
        return path

    def delete(self, key: str) -> bool:
        if key.startswith("builtin:"):
            return False
        path = self.workflows_dir() / f"{key}.json"
        if path.is_file():
            path.unlink()
            return True
        return False

    def to_json(self, workflow: WorkflowDefinition) -> str:
        return workflow.model_dump_json(indent=2)

    def from_json(self, raw: str) -> WorkflowDefinition:
        return WorkflowDefinition.model_validate(json.loads(raw))