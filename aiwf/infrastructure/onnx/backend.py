"""
aiwf/infrastructure/onnx/backend.py

ONNXBackend — implements the InferenceBackend protocol using ONNX Runtime.

This backend runs models exported to ONNX format with zero dependency on
diffusers.  Inference is handled entirely by:
  * onnxruntime (CUDA EP or DirectML EP)
  * AIWF's own sampler math (aiwf.infrastructure.samplers)
  * AIWF's own schedule math

Compatible model directories
-----------------------------
Any directory produced by `optimum-cli export onnx --model <hf_id>` or by
manual export from safetensors/ckpt checkpoints.  The backend discovers
available models by scanning a configured root directory for subdirectories
that contain the required ONNX files.

Backend selection
-----------------
Set in config.json (added to UserSettings separately):
    "inference_backend": "onnx"
    "onnx_model_dir": "/path/to/sd_onnx_models/my_model"
    "onnx_provider": "auto"   # or "cuda", "directml", "cpu"
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from PIL import Image

from aiwf.core.domain.generation import GenerationRequest, GenerationResult
from aiwf.core.domain.models import Checkpoint, EmbeddingInfo, LoraInfo, SamplerInfo, VaeInfo
from aiwf.infrastructure.onnx.pipeline import ONNXPipeline
from aiwf.infrastructure.onnx.session import ProviderPreference
from aiwf.services.pipeline_preflight import PipelinePreflightResult, preflight_onnx_pipeline

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str, "Image.Image | None"], None]

_REQUIRED_SUBDIRS = ("text_encoder", "unet", "vae_decoder")


def _is_valid_onnx_model_dir(path: Path) -> bool:
    return all((path / sub / "model.onnx").is_file() for sub in _REQUIRED_SUBDIRS)


class ONNXBackend:
    """InferenceBackend that uses ONNX Runtime — no diffusers.

    Parameters
    ----------
    models_root:
        Directory containing one or more ONNX model subdirectories.
        Each child directory that has text_encoder/, unet/, vae_decoder/
        sub-subdirectories is treated as a selectable checkpoint.
    provider:
        ORT execution provider preference.
    """

    def __init__(
        self,
        models_root: Path,
        provider: ProviderPreference = "auto",
        device_id: int = 0,
    ) -> None:
        self._models_root = models_root
        self._provider = provider
        self._device_id = device_id
        self._pipeline: ONNXPipeline | None = None
        self._active_checkpoint_id: str | None = None

    # ------------------------------------------------------------------
    # Catalog
    # ------------------------------------------------------------------

    def _discover_models(self) -> list[Path]:
        if not self._models_root.is_dir():
            return []
        return sorted(
            p for p in self._models_root.iterdir()
            if p.is_dir() and _is_valid_onnx_model_dir(p)
        )

    def list_checkpoints(self) -> list[Checkpoint]:
        return [
            Checkpoint(
                id=p.name,
                title=p.name,
                filename=p.name,
                path=str(p),
                kind="onnx",
            )
            for p in self._discover_models()
        ]

    def list_samplers(self) -> list[SamplerInfo]:
        from aiwf.infrastructure.samplers.dispatch import available_samplers
        from aiwf.core.domain.models import SAMPLERS
        # Return only the samplers we have native implementations for
        native_ids = set(available_samplers())
        return [s for s in SAMPLERS if s.id in native_ids]

    def list_loras(self) -> list[LoraInfo]:
        # LoRA support for ONNX models requires ONNX LoRA merging — not yet implemented
        return []

    def list_vaes(self) -> list[VaeInfo]:
        # VAE is baked into the model dir
        return []

    def list_embeddings(self) -> list[EmbeddingInfo]:
        # Textual inversion is not implemented for ONNX models yet.
        return []

    def preflight_checkpoint(self, checkpoint_id: str | None = None) -> PipelinePreflightResult:
        """Return a non-loading readiness report for an ONNX checkpoint."""
        ckpt = self.resolve_checkpoint(checkpoint_id)
        if not ckpt.path:
            return preflight_onnx_pipeline(self._models_root / (checkpoint_id or ""), provider_preference=self._provider)
        return preflight_onnx_pipeline(ckpt.path, provider_preference=self._provider)

    def resolve_checkpoint(self, checkpoint_id: str | None = None) -> Checkpoint:
        available = self.list_checkpoints()
        if not available:
            return Checkpoint(id="none", title="No ONNX models found", filename="", path="")
        if checkpoint_id:
            for ckpt in available:
                if ckpt.id == checkpoint_id:
                    return ckpt
        return available[0]

    def load_checkpoint(self, checkpoint_id: str | None = None) -> Checkpoint:
        ckpt = self.resolve_checkpoint(checkpoint_id)
        if ckpt.path and ckpt.id != "none":
            if self._active_checkpoint_id != ckpt.id:
                # Unload previous pipeline
                if self._pipeline:
                    self._pipeline.unload()
                self._pipeline = ONNXPipeline(
                    Path(ckpt.path),
                    provider=self._provider,
                    device_id=self._device_id,
                )
                self._active_checkpoint_id = ckpt.id
                logger.info("Loaded ONNX checkpoint: %s", ckpt.id)
        return ckpt

    def unload(self) -> None:
        if self._pipeline:
            self._pipeline.unload()
            self._pipeline = None
        self._active_checkpoint_id = None

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        request: GenerationRequest,
        init_images=None,
        mask_images=None,
        control_images=None,
        on_progress: ProgressCallback | None = None,
        should_cancel: Callable[[], bool] | None = None,
        preview_every_n_steps: int = 0,
    ) -> GenerationResult:
        if self._pipeline is None:
            self.load_checkpoint(request.checkpoint_id)

        if self._pipeline is None:
            raise RuntimeError("No ONNX models found in configured models_root.")

        return self._pipeline.generate(
            request,
            on_progress=on_progress,
            should_cancel=should_cancel,
        )
