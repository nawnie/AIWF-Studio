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


def test_lora_training_uses_kohya_runner(monkeypatch):
    created = object()

    class FakeKohyaRunner:
        def __new__(cls):
            return created

    monkeypatch.setattr("aiwf.services.training.kohya_runner.KohyaRunner", FakeKohyaRunner)

    assert training._runner_for_engine("Kohya LoRA") is created


def test_ed2_training_uses_ed2_runner(monkeypatch):
    created = object()

    class FakeED2Runner:
        def __new__(cls):
            return created

    monkeypatch.setattr("aiwf.services.training.ed2_runner.ED2Runner", FakeED2Runner)

    assert training._runner_for_engine("ED2 Full Fine-tune") is created


def test_validate_training_request_routes_lora_to_kohya_validator(monkeypatch):
    calls = []

    class FakeValidator:
        def validate_kohya(self, request):
            calls.append(("kohya", request))
            return ValidationResult.passed()

        def validate_ed2(self, request):
            calls.append(("ed2", request))
            return ValidationResult.passed()

    monkeypatch.setattr(training, "_validator", FakeValidator())

    training._validate_training_request("Kohya LoRA", {"job_name": "x"})

    assert calls == [("kohya", {"job_name": "x"})]


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
