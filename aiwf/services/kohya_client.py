"""Kohya LoRA training service client.

Mirrors the pattern of ``aiwf.services.wan`` — validates the request,
acquires the GPU tenant lock, and submits a subprocess job through the
EngineSupervisor.

Usage::

    from aiwf.services.kohya_client import KohyaClient
    from aiwf.core.domain.training import KohyaLoraRequest

    client = KohyaClient()
    request = KohyaLoraRequest(
        job_name="my_lora",
        base_model_path="models/Stable-diffusion/base.safetensors",
        base_arch="sdxl",
        dataset_dir="datasets/prepared/my_concept",
    )
    job_id = client.submit(request, outputs_root="outputs")
    # Job runs in a subprocess; listen for events via supervisor.add_progress_listener()
"""
from __future__ import annotations

import logging
from pathlib import Path

from aiwf.core.domain.training import KohyaLoraRequest
from aiwf.services.engine_supervisor import EngineSupervisor, get_supervisor
from aiwf.services.gpu_tenant_lock import get_gpu_lock

logger = logging.getLogger(__name__)


class KohyaUnavailable(RuntimeError):
    """Raised when the Kohya engine is not configured or the GPU is blocked."""


class KohyaClient:
    """Thin service client: validates → GPU lock → supervisor.submit_subprocess_job()."""

    def __init__(self, supervisor: EngineSupervisor | None = None) -> None:
        self._supervisor = supervisor or get_supervisor()

    def submit(
        self,
        request: KohyaLoraRequest,
        *,
        outputs_root: "Path | str" = "outputs",
    ) -> str:
        """Validate the request, acquire the GPU tenant lock, and spawn a Kohya worker.

        Returns the job_id.  The job runs asynchronously — register a listener on
        the supervisor to receive progress events::

            supervisor.add_progress_listener(my_callback)

        Raises
        ------
        KohyaUnavailable
            If the GPU is held by a higher-priority tenant, or the Kohya engine
            is not configured (venv not created).
        pydantic.ValidationError
            If the request fields are invalid.
        """
        from launch import _build_engine_registry, _load_engines_config, _engine_enabled

        # Find the Kohya spec from the registry
        cfg = _load_engines_config()
        if not _engine_enabled("kohya", cfg, default=False):
            raise KohyaUnavailable(
                "Kohya engine is not enabled. "
                "Set 'enabled': true in engines.json and restart AIWF Studio."
            )

        specs = {s.name: s for s in _build_engine_registry()}
        spec = specs.get("kohya")
        if spec is None or not spec.is_ready():
            raise KohyaUnavailable(
                f"Kohya engine venv not ready at {spec.venv_dir if spec else '?'}. "
                "Ensure the Kohya repo is cloned and launch.py has run setup."
            )

        # Validate dataset exists
        if not Path(request.dataset_dir).exists():
            raise ValueError(
                f"Dataset directory not found: {request.dataset_dir}. "
                "Prepare your training images there before submitting a job."
            )

        # GPU tenant lock
        gpu_lock = get_gpu_lock()
        import uuid as _uuid
        job_id_tentative = f"kohya_{_uuid.uuid4().hex[:8]}"
        with gpu_lock.acquire("kohya", job_id_tentative) as granted:
            if not granted:
                raise KohyaUnavailable(
                    f"Cannot start Kohya training: {gpu_lock.blocked_message('kohya')}"
                )
            return self._supervisor.submit_subprocess_job(
                spec,
                request.model_dump(),
                outputs_root=outputs_root,
                job_id=job_id_tentative,
            )
        # GPU lock is held for the duration of submit_subprocess_job() setup only.
        # The actual training subprocess manages its own GPU ownership.

    def cancel(self, job_id: str) -> None:
        """Cancel a running Kohya job (sends SIGTERM to the worker)."""
        self._supervisor.cancel_job(job_id)

    def status(self, job_id: str) -> str:
        """Return a human-readable status string for the job."""
        record = self._supervisor.get_job(job_id)
        if record is None:
            return f"Job {job_id} not found."
        return f"[{record.phase.value}] {record.message or 'running'}"
