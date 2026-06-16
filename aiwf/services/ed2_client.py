"""EveryDream2 full fine-tuning service client.

Mirrors the pattern of ``aiwf.services.kohya_client`` — validates the request,
acquires the GPU tenant lock (ED2 has the highest priority: 100), and submits
a subprocess job through the EngineSupervisor.

Usage::

    from aiwf.services.ed2_client import ED2Client
    from aiwf.core.domain.training import ED2TrainingRequest

    client = ED2Client()
    request = ED2TrainingRequest(
        job_name="my_finetune",
        base_model_path="models/Stable-diffusion/base.safetensors",
        dataset_dir="datasets/prepared/my_concept",
        max_epochs=20,
    )
    job_id = client.submit(request, outputs_root="outputs")
    # Job runs in a subprocess; listen for events via supervisor.add_progress_listener()
"""
from __future__ import annotations

import logging
from pathlib import Path

from aiwf.core.domain.training import ED2TrainingRequest
from aiwf.services.engine_supervisor import EngineSupervisor, get_supervisor
from aiwf.services.gpu_tenant_lock import get_gpu_lock

logger = logging.getLogger(__name__)
_MODEL_EXTENSIONS = {".safetensors", ".ckpt", ".pt", ".bin", ".pth"}


class ED2Unavailable(RuntimeError):
    """Raised when the ED2 engine is not configured or the GPU is blocked."""


class ED2Client:
    """Thin service client: validates → GPU lock → supervisor.submit_subprocess_job().

    ED2 holds the highest GPU priority (100), meaning it will block all other
    engines.  Wan generation, Kohya, and Ollama will be blocked while ED2 trains.
    This is intentional: full model training should never be interrupted.
    """

    def __init__(self, supervisor: EngineSupervisor | None = None) -> None:
        self._supervisor = supervisor or get_supervisor()

    def submit(
        self,
        request: ED2TrainingRequest,
        *,
        outputs_root: "Path | str" = "outputs",
    ) -> str:
        """Validate the request, acquire the GPU tenant lock, and spawn an ED2 worker.

        Returns the job_id.  The job runs asynchronously — register a listener on
        the supervisor to receive progress events::

            supervisor.add_progress_listener(my_callback)

        Raises
        ------
        ED2Unavailable
            If the GPU is held by another tenant, or the ED2 engine is not configured.
        pydantic.ValidationError
            If the request fields are invalid.
        """
        from launch import _build_engine_registry, _load_engines_config, _engine_enabled

        # Find the ED2 spec from the registry
        cfg = _load_engines_config()
        if not _engine_enabled("ed2", cfg, default=False):
            raise ED2Unavailable(
                "ED2 engine is not enabled. "
                "Set 'enabled': true in engines.json and restart AIWF Studio."
            )

        specs = {s.name: s for s in _build_engine_registry()}
        spec = specs.get("ed2")
        if spec is None or not spec.is_ready():
            raise ED2Unavailable(
                f"ED2 engine venv not ready at {spec.venv_dir if spec else '?'}. "
                "Ensure the EveryDream2trainer repo is cloned and launch.py has run setup."
            )

        # Validate dataset exists
        if not Path(request.dataset_dir).exists():
            raise ValueError(
                f"Dataset directory not found: {request.dataset_dir}. "
                "Prepare your training images there before submitting a job."
            )

        # Validate base model exists (or is a clear HF ID).
        base = Path(request.base_model_path)
        if not base.exists() and not _looks_like_hf_repo_id(request.base_model_path):
            raise ValueError(
                f"Base model not found: {request.base_model_path}. "
                "Provide an existing path or a Hugging Face model ID (org/repo)."
            )

        # GPU tenant lock — ed2 priority=100 (highest)
        gpu_lock = get_gpu_lock()
        import uuid as _uuid
        job_id_tentative = f"ed2_{_uuid.uuid4().hex[:8]}"
        with gpu_lock.acquire("ed2", job_id_tentative) as granted:
            if not granted:
                raise ED2Unavailable(
                    f"Cannot start ED2 training: {gpu_lock.blocked_message('ed2')}"
                )
            return self._supervisor.submit_subprocess_job(
                spec,
                request.model_dump(),
                outputs_root=outputs_root,
                job_id=job_id_tentative,
            )

    def cancel(self, job_id: str) -> None:
        """Cancel a running ED2 job (sends SIGTERM to the worker process)."""
        self._supervisor.cancel_job(job_id)

    def status(self, job_id: str) -> str:
        """Return a human-readable status string for the job."""
        record = self._supervisor.get_job(job_id)
        if record is None:
            return f"Job {job_id} not found."
        return f"[{record.phase.value}] {record.message or 'running'}"


def _looks_like_hf_repo_id(value: str) -> bool:
    text = str(value).strip()
    if not text or "\\" in text or text.startswith(("/", "./", "../")):
        return False
    if ":" in text:
        return False
    if Path(text).suffix.lower() in _MODEL_EXTENSIONS:
        return False
    return text.count("/") == 1
