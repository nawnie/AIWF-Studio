from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from aiwf.core.domain.engine import EngineTenant
from aiwf.core.domain.worker import WorkerCommand
from aiwf.services.model_ops import PreflightResult
from aiwf.web.tabs.model_manager import _model_op_tenant, _run_model_setup_operation


def _preflight(command_name: str | None) -> PreflightResult:
    command = None
    if command_name is not None:
        command = WorkerCommand(
            args=["python", "-m", "aiwf.workers.model_ops", "noop"],
            cwd=Path("."),
            env={},
            name=command_name,
        )
    return PreflightResult(True, "ready", command=command)


def test_diffusers_model_ops_use_image_tenant():
    assert _model_op_tenant(_preflight("model-ops-lora-fuse")) == EngineTenant.IMAGE
    assert _model_op_tenant(_preflight("model-ops-convert")) == EngineTenant.IMAGE


def test_cpu_or_receipt_model_ops_do_not_take_gpu_tenant():
    assert _model_op_tenant(_preflight("model-ops-checkpoint-blend")) is None
    assert _model_op_tenant(_preflight("model-ops-quantize")) is None
    assert _model_op_tenant(_preflight(None)) is None


def test_model_manager_setup_operations_reject_active_video_generation(tmp_path):
    from aiwf.web.pro_api import _pro_video_job_finish, _pro_video_job_start

    ctx = SimpleNamespace(flags=SimpleNamespace(data_dir=tmp_path))
    calls = []
    job_id = _pro_video_job_start(ctx, SimpleNamespace(steps=1), message="test video job")

    with pytest.raises(HTTPException) as exc_info:
        _run_model_setup_operation(ctx, lambda: calls.append("mutated"))

    assert exc_info.value.status_code == 409
    assert "GPU generation job is active" in str(exc_info.value.detail)
    assert calls == []
    _pro_video_job_finish(ctx, job_id, "failed", error="test complete")
