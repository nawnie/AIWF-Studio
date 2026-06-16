from __future__ import annotations

from pathlib import Path

from aiwf.core.domain.worker import WorkerCommand
from aiwf.services.worker_probe import WorkerProbeService, _parse_event


class FakeRegistry:
    def __init__(self, root: Path) -> None:
        self.repo_root = root

    def build_command(self, engine: str, request_path: Path) -> WorkerCommand:
        return WorkerCommand(
            args=["python", "worker.py", str(request_path)],
            cwd=self.repo_root,
            env={},
            name=engine,
        )


class FakeSupervisor:
    def __init__(self, lines):
        self.lines = list(lines)
        self.started: list[tuple[str, WorkerCommand]] = []

    def start(self, worker_name: str, command: WorkerCommand):
        self.started.append((worker_name, command))
        yield from self.lines


def test_parse_event_ignores_non_json_lines():
    assert _parse_event("hello") is None
    assert _parse_event("{}") is None
    assert _parse_event('{"kind":"status","message":"ok"}') == {"kind": "status", "message": "ok"}


def test_worker_probe_success_writes_request_and_collects_events(tmp_path: Path):
    supervisor = FakeSupervisor(
        [
            '{"kind":"status","job_id":"x","message":"starting"}',
            '{"kind":"complete","job_id":"x","message":"done"}',
        ]
    )
    service = WorkerProbeService(
        tmp_path,
        registry=FakeRegistry(tmp_path),
        supervisor=supervisor,
    )

    result = service.probe("wan")

    assert result.ok
    assert result.message == "done"
    assert result.request_path.exists()
    assert result.events[-1]["kind"] == "complete"
    assert supervisor.started[0][0] == "wan"


def test_worker_probe_error_event_marks_failure(tmp_path: Path):
    supervisor = FakeSupervisor(['{"kind":"error","job_id":"x","message":"bad"}'])
    service = WorkerProbeService(
        tmp_path,
        registry=FakeRegistry(tmp_path),
        supervisor=supervisor,
    )

    result = service.probe("wan")

    assert not result.ok
    assert result.message == "bad"
