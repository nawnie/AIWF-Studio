"""Dual inference backend: Diffusers + stable-diffusion.cpp in one process.

Routes each GenerationRequest to an engine based on ``request.pipeline_backend``
(``"sdcpp"`` -> stable-diffusion.cpp subprocess, anything else -> the primary
Diffusers backend). Catalog queries (checkpoints, loras, vaes, embeddings)
are served by the primary backend — both engines scan the same model folders,
so the catalogs are identical.

VRAM note: before an sdcpp job runs, the primary backend's pipeline is
unloaded so the subprocess gets the GPU to itself (a warm SDXL Diffusers
pipeline plus an sdcpp SDXL load will not fit in 16 GB together). Disable
with AIWF_DUAL_UNLOAD_TORCH=0 if you have headroom.
"""
from __future__ import annotations

import logging
import os
from typing import Callable

from PIL import Image

from aiwf.core.domain.generation import GenerationRequest, GenerationResult
from aiwf.core.domain.models import Checkpoint, EmbeddingInfo, LoraInfo, SamplerInfo, VaeInfo
from aiwf.core.interfaces.backend import ProgressCallback

logger = logging.getLogger(__name__)


def _unload_torch_enabled() -> bool:
    return str(os.environ.get("AIWF_DUAL_UNLOAD_TORCH", "1")).strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


class DualInferenceBackend:
    """InferenceBackend that fans out to Diffusers or stable-diffusion.cpp per request."""

    def __init__(self, primary, sdcpp) -> None:
        self.primary = primary
        self.sdcpp = sdcpp

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
    @staticmethod
    def _wants_sdcpp(request: GenerationRequest) -> bool:
        value = str(getattr(request, "pipeline_backend", None) or "").strip().lower().replace("-", "_")
        return value in {"sdcpp", "stable_diffusion.cpp", "stable_diffusion_cpp"}

    def backend_for(self, request: GenerationRequest):
        return self.sdcpp if self._wants_sdcpp(request) else self.primary

    # ------------------------------------------------------------------
    # Catalog / lifecycle — served by the primary backend
    # ------------------------------------------------------------------
    def list_checkpoints(self) -> list[Checkpoint]:
        return self.primary.list_checkpoints()

    def list_samplers(self) -> list[SamplerInfo]:
        return self.primary.list_samplers()

    def list_loras(self) -> list[LoraInfo]:
        return self.primary.list_loras()

    def list_vaes(self) -> list[VaeInfo]:
        return self.primary.list_vaes()

    def list_embeddings(self) -> list[EmbeddingInfo]:
        return self.primary.list_embeddings()

    def resolve_checkpoint(self, checkpoint_id: str | None = None) -> Checkpoint:
        return self.primary.resolve_checkpoint(checkpoint_id)

    def load_checkpoint(self, checkpoint_id: str | None = None) -> Checkpoint:
        return self.primary.load_checkpoint(checkpoint_id)

    def unload(self) -> None:
        self.primary.unload()
        self.sdcpp.unload()

    def invalidate_checkpoints(self) -> None:
        for backend in (self.primary, self.sdcpp):
            invalidate = getattr(backend, "invalidate_checkpoints", None)
            if callable(invalidate):
                invalidate()

    def invalidate_loras(self) -> None:
        for backend in (self.primary, self.sdcpp):
            invalidate = getattr(backend, "invalidate_loras", None)
            if callable(invalidate):
                invalidate()

    def invalidate_vaes(self) -> None:
        for backend in (self.primary, self.sdcpp):
            invalidate = getattr(backend, "invalidate_vaes", None)
            if callable(invalidate):
                invalidate()

    def invalidate_embeddings(self) -> None:
        for backend in (self.primary, self.sdcpp):
            invalidate = getattr(backend, "invalidate_embeddings", None)
            if callable(invalidate):
                invalidate()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(
        self,
        request: GenerationRequest,
        init_images: list[Image.Image] | None = None,
        mask_images: list[Image.Image] | None = None,
        control_images: list[Image.Image] | None = None,
        on_progress: ProgressCallback | None = None,
        should_cancel: Callable[[], bool] | None = None,
        preview_every_n_steps: int = 0,
    ) -> GenerationResult:
        if self._wants_sdcpp(request):
            if _unload_torch_enabled():
                try:
                    self.primary.unload()
                except Exception:  # noqa: BLE001 - freeing VRAM is best-effort
                    logger.exception("dual backend: failed to unload primary backend before sdcpp job")
            logger.info("dual backend: routing job to stable-diffusion.cpp")
            engine = self.sdcpp
        else:
            engine = self.primary
        return engine.generate(
            request,
            init_images=init_images,
            mask_images=mask_images,
            control_images=control_images,
            on_progress=on_progress,
            should_cancel=should_cancel,
            preview_every_n_steps=preview_every_n_steps,
        )

    # Services occasionally reach for backend-specific extras (device info,
    # optimization hooks, ...). Those belong to the primary engine.
    def __getattr__(self, name: str):
        return getattr(self.primary, name)
