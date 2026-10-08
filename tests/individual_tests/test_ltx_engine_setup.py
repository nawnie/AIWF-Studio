from __future__ import annotations

import subprocess
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from aiwf.services import ltx_engine_setup


def test_ltx_enable_preserves_malformed_engine_config_and_fails_closed(tmp_path: Path) -> None:
    if sys.platform != "win32":
        pytest.skip("LTX bootstrap is Windows-only")
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell is None:
        pytest.skip("PowerShell is unavailable")
    bootstrap = Path(__file__).resolve().parents[2] / "scripts" / "bootstrap_ltx.ps1"
    source = bootstrap.read_text(encoding="utf-8")
    match = re.search(r"(?ms)^function Enable-LtxEngine\s*\{.*?^\}", source)
    assert match is not None
    engines_json = tmp_path / "engines.json"
    original = '{"wan":'
    engines_json.write_text(original, encoding="utf-8")
    script_path = tmp_path / "verify_ltx_config_guard.ps1"
    script_path.write_text(
        "$ErrorActionPreference = 'Stop'\n"
        + "$EnginesJson = " + "'" + str(engines_json).replace("'", "''") + "'\n"
        + match.group(0)
        + "\nEnable-LtxEngine\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(script_path)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "it was left unchanged" in (result.stdout + result.stderr).lower()
    assert engines_json.read_text(encoding="utf-8") == original


def test_ltx_engine_installer_runs_only_the_fixed_repo_bootstrap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = tmp_path / "scripts" / "bootstrap_ltx.ps1"
    script.parent.mkdir(parents=True)
    script.write_text("# fixture", encoding="utf-8")
    captured = {}
    process = SimpleNamespace(pid=4321, poll=lambda: None)

    def popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return process

    monkeypatch.setattr(ltx_engine_setup.subprocess, "Popen", popen)
    monkeypatch.setattr(ltx_engine_setup, "_ACTIVE_INSTALLS", {})

    result = ltx_engine_setup.start_ltx_engine_install(tmp_path)

    assert result["status"] == "started"
    assert result["pid"] == 4321
    assert Path(result["logPath"]).is_file()
    assert captured["command"] == [
        "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", str(script.resolve()), "-Enable",
    ]
    assert captured["kwargs"]["cwd"] == str(tmp_path.resolve())
    assert captured["kwargs"]["stdin"] == subprocess.DEVNULL
    assert "shell" not in captured["kwargs"]


def test_ltx_engine_installer_prevents_duplicate_start_and_reports_completion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = tmp_path / "scripts" / "bootstrap_ltx.ps1"
    script.parent.mkdir(parents=True)
    script.write_text("# fixture", encoding="utf-8")
    state = {"exit_code": None}
    process = SimpleNamespace(pid=4322, poll=lambda: state["exit_code"])
    calls = []
    monkeypatch.setattr(ltx_engine_setup.subprocess, "Popen", lambda *_args, **_kwargs: calls.append(True) or process)
    monkeypatch.setattr(ltx_engine_setup, "_ACTIVE_INSTALLS", {})

    first = ltx_engine_setup.start_ltx_engine_install(tmp_path)
    second = ltx_engine_setup.start_ltx_engine_install(tmp_path)

    assert first["status"] == "started"
    assert second["status"] == "already_running"
    assert second["logPath"] == first["logPath"]
    assert len(calls) == 1
    assert ltx_engine_setup.ltx_engine_install_status(tmp_path)["running"] is True

    state["exit_code"] = 0
    status = ltx_engine_setup.ltx_engine_install_status(tmp_path)
    assert status["status"] == "finished"
    assert status["exitCode"] == 0


def test_ltx_engine_installer_fails_closed_when_fixed_script_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ltx_engine_setup, "_ACTIVE_INSTALLS", {})
    with pytest.raises(FileNotFoundError, match="bootstrap script is unavailable"):
        ltx_engine_setup.start_ltx_engine_install(tmp_path)
