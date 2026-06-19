from pathlib import Path

from aiwf.core.domain.engine import EngineTenant
from aiwf.core.domain.worker import WorkerCommand
from aiwf.services.model_ops import PreflightResult
from aiwf.web.tabs.model_manager import _model_op_tenant


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
