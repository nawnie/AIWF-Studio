from __future__ import annotations

from types import SimpleNamespace

from aiwf.core.domain.engine import EngineTenant
from aiwf.services.training.dataset_validator import ValidationResult
from aiwf.web.tabs import training


def test_format_validation_result_reports_errors_and_warnings():
    result = ValidationResult.failed(["dataset missing"], warnings=["caption sparse"])

    text = training._format_validation_result(result)

    assert "ERROR: dataset missing" in text
    assert "WARNING: caption sparse" in text


def test_format_validation_result_reports_ok():
    text = training._format_validation_result(ValidationResult.passed())

    assert text == "OK: Dataset looks good."


def test_release_tenant_uses_supervisor_idle_switch():
    calls = []

    class Supervisor:
        def request_switch(self, request):
            calls.append(request)

    ctx = SimpleNamespace(supervisor=Supervisor())

    training._release_tenant(ctx, "done")

    assert calls
    assert calls[0].target == EngineTenant.IDLE
    assert calls[0].reason == "done"


def test_release_tenant_without_supervisor_is_noop():
    training._release_tenant(SimpleNamespace(), "done")
