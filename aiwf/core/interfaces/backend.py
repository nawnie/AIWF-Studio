from __future__ import annotations

from typing import Callable, Protocol

from PIL import Image

from aiwf.core.domain.generation import GenerationRequest, GenerationResult
from aiwf.core.domain.models import Checkpoint, LoraInfo, SamplerInfo, VaeInfo


ProgressCallback = Callable[[int, int, str, Image.Image | None], None]


class InferenceBackend(Protocol):
    """Swappable inference engine. Diffusers today; Comfy/custom tomorrow."""

    def list_checkpoints(self) -> list[Checkpoint]: ...

    def list_samplers(self) -> list[SamplerInfo]: ...

    def list_loras(self) -> list[LoraInfo]: ...

    def list_vaes(self) -> list[VaeInfo]: ...

    def resolve_checkpoint(self, checkpoint_id: str | None = None) -> Checkpoint: ...

    def load_checkpoint(self, checkpoint_id: str | None = None) -> Checkpoint: ...

    def unload(self) -> None: ...

    def generate(
        self,
        request: GenerationRequest,
        init_images: list[Image.Image] | None = None,
        mask_images: list[Image.Image] | None = None,
        control_images: list[Image.Image] | None = None,
        on_progress: ProgressCallback | None = None,
        should_cancel: Callable[[], bool] | None = None,
        preview_every_n_steps: int = 0,
    ) -> GenerationResult: ...