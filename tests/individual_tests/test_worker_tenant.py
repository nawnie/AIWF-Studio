from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from aiwf.services.worker_tenant import WorkerTenantRegistry, WorkerTenantSpec, python_exe_for_venv


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "launch.py").is_file():
            return parent
    raise RuntimeError("Could not find AIWF repo root")


def _write_engine_files(root: Path, name: str = "wan", *, venv: str = "engines/wan/.venv") -> tuple[Path, Path]:
    worker = root / "engines" / name / "worker.py"
    python = python_exe_for_venv(root / venv)
    worker.parent.mkdir(parents=True, exist_ok=True)
    python.parent.mkdir(parents=True, exist_ok=True)
    worker.write_text("print('worker')", encoding="utf-8")
    python.write_text("", encoding="utf-8")
    return worker, python


def test_worker_tenant_reports_disabled_missing_engine(tmp_path: Path):
    registry = WorkerTenantRegistry(tmp_path)

    status = registry.status("wan")

    assert not status.ready
    assert not status.enabled
    assert any("enabled=true" in message for message in status.messages)
    assert any("AIWF worker missing" in message for message in status.messages)


def test_worker_tenant_ready_for_isolated_venv(tmp_path: Path):
    worker, python = _write_engine_files(tmp_path)
    (tmp_path / "engines.json").write_text(
        json.dumps({"wan": {"enabled": True}}),
        encoding="utf-8",
    )

    status = WorkerTenantRegistry(tmp_path).status("wan")

    assert status.ready
    assert status.worker_script == worker.resolve()
    assert status.python_exe == python
    assert not status.uses_studio_venv


def test_worker_tenant_supports_shared_studio_venv_alias(tmp_path: Path):
    worker = tmp_path / "engines" / "wan" / "worker.py"
    python = python_exe_for_venv(tmp_path / "venv")
    worker.parent.mkdir(parents=True)
    python.parent.mkdir(parents=True)
    worker.write_text("print('worker')", encoding="utf-8")
    python.write_text("", encoding="utf-8")
    (tmp_path / "engines.json").write_text(
        json.dumps({"wan": {"enabled": True, "venv_dir": "studio"}}),
        encoding="utf-8",
    )

    status = WorkerTenantRegistry(tmp_path).status("wan")

    assert status.ready
    assert status.uses_studio_venv
    assert status.python_exe == python
    assert "Studio runtime" in status.markdown_line()


def test_worker_tenant_ed2_default_repo_lives_under_engines(tmp_path: Path):
    registry = WorkerTenantRegistry(tmp_path)

    status = registry.status("ed2")

    assert status.repo_dir == (tmp_path / "engines" / "ed2" / "EveryDream2trainer").resolve()
    assert any("EveryDream2trainer" in message for message in status.messages)


def test_worker_tenant_build_command_injects_root_and_pythonpath(tmp_path: Path):
    worker, python = _write_engine_files(tmp_path)
    request = tmp_path / "request.json"
    request.write_text("{}", encoding="utf-8")
    (tmp_path / "engines.json").write_text(
        json.dumps({"wan": {"enabled": True}}),
        encoding="utf-8",
    )

    cmd = WorkerTenantRegistry(tmp_path).build_command(
        "wan",
        request,
        env={"PYTHONPATH": "existing"},
    )

    assert cmd.args[:3] == [str(python), str(worker.resolve()), str(request.resolve())]
    assert cmd.cwd == tmp_path.resolve()
    assert cmd.name == "wan"
    assert cmd.env["AIWF_ROOT"] == str(tmp_path.resolve())
    assert cmd.env["PYTHONPATH"].split(os.pathsep)[0] == str(tmp_path.resolve())


def test_worker_tenant_build_command_blocks_unready_engine(tmp_path: Path):
    registry = WorkerTenantRegistry(tmp_path)

    with pytest.raises(RuntimeError, match="not ready"):
        registry.build_command("wan", tmp_path / "request.json")


def test_worker_tenant_can_use_custom_specs(tmp_path: Path):
    specs = {
        "audio": WorkerTenantSpec(
            name="audio",
            label="Audio Engine",
            worker_script="engines/audio/worker.py",
            venv_dir="engines/audio/.venv",
        )
    }
    worker = tmp_path / "engines" / "audio" / "worker.py"
    python = python_exe_for_venv(tmp_path / "engines" / "audio" / ".venv")
    worker.parent.mkdir(parents=True)
    python.parent.mkdir(parents=True)
    worker.write_text("print('worker')", encoding="utf-8")
    python.write_text("", encoding="utf-8")
    (tmp_path / "engines.json").write_text(
        json.dumps({"audio": {"enabled": True}}),
        encoding="utf-8",
    )

    status = WorkerTenantRegistry(tmp_path, specs=specs).status("audio")

    assert status.ready
    assert status.label == "Audio Engine"


def test_wan_worker_probe_emits_jsonl_events(tmp_path: Path):
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps({"_job_id": "wan-probe-test", "_engine": "wan", "mode": "probe"}),
        encoding="utf-8",
    )

    root = _repo_root()
    result = subprocess.run(
        [sys.executable, "engines/wan/worker.py", str(request)],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    events = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert events[0]["kind"] == "status"
    assert any(event["kind"] == "complete" for event in events)
