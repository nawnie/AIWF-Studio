"""Contract tests for the MMAudio bootstrap script without running its installer."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
INSTALLER = ROOT / "scripts" / "bootstrap_mmaudio.ps1"


def _function_source(source: str, name: str) -> str:
    match = re.search(rf"(?ms)^function {re.escape(name)}\s*\{{.*?^\}}", source)
    assert match is not None, f"installer function {name!r} was not found"
    return match.group(0)


def _powershell() -> str:
    executable = shutil.which("pwsh") or shutil.which("powershell")
    if executable is None:
        pytest.skip("PowerShell is unavailable")
    return executable


def test_mmaudio_installer_checks_every_native_install_command():
    source = INSTALLER.read_text(encoding="utf-8")

    assert "Invoke-CheckedNative -Label \"MMAudio repository clone\"" in source
    assert "Invoke-CheckedNative -Label \"Audio engine pip upgrade\"" in source
    assert "Invoke-CheckedNative -Label \"CUDA PyTorch installation for audio engine\"" in source
    assert "Invoke-CheckedNative -Label \"MMAudio editable package installation\"" in source
    assert re.search(r"(?ms)function Invoke-CheckedNative\s*\{.*?\$exitCode = \$LASTEXITCODE.*?if \(\$exitCode -ne 0\).*?throw", source)
    assert "Write-Host \"[AIWF] MMAudio bootstrap complete" in source


def test_mmaudio_venv_prefers_py310_then_validates_fallback_python():
    source = INSTALLER.read_text(encoding="utf-8")
    venv = _function_source(source, "New-AudioEngineVenv")
    path_guard = _function_source(source, "Assert-AudioEnginePath")
    base_probe = _function_source(source, "Test-SupportedAudioPythonCommand")
    version_probe = _function_source(source, "Test-SupportedAudioPython")

    assert '"-3.10"' in venv
    assert '"-m", "venv", $resolvedEnvironmentPath' in venv
    assert "Test-SupportedAudioPythonCommand -Command $commandName" in venv
    assert "Test-SupportedAudioPython -PythonPath $PythonPath" in venv
    assert "'^3\\.(10|11|12)$'" in version_probe
    assert "'^3\\.(10|11|12)$'" in base_probe
    assert "Assert-AudioEnginePath -Path $EnvironmentPath" in venv
    assert "Resolve-AudioEnginePath -Path $Path" in path_guard
    assert "Remove-AudioEngineAttemptDirectory -Path $resolvedEnvironmentPath" in venv
    assert "Move-Item -LiteralPath $stagingPath -Destination $EnvironmentPath" not in venv
    assert "Move or repair it before retrying" in venv
    assert "Test-SupportedAudioPython -PythonPath $Python" in source
    assert "MMAudio repository folder is missing or empty" in source
    assert "Assert-AudioEnginePath -Path $stagingRepo" in source
    assert "Assert-AudioEnginePath -Path $RepoDir" in source
    assert "Remove-AudioEngineAttemptDirectory -Path $resolvedStagingRepo" in source


def test_native_failure_helper_throws_with_label_and_exit_code(tmp_path: Path):
    powershell = _powershell()
    source = INSTALLER.read_text(encoding="utf-8")
    helper = _function_source(source, "Invoke-CheckedNative")
    script_path = tmp_path / "checked_native.ps1"
    python_path = "'" + sys.executable.replace("'", "''") + "'"
    script_path.write_text(
        "$ErrorActionPreference = 'Stop'\n"
        + helper
        + "\ntry {\n"
        + "  Invoke-CheckedNative -Label 'mock pip install' -Command { & "
        + python_path
        + " -c 'import sys; sys.exit(17)' }\n"
        + "  exit 2\n"
        + "} catch {\n"
        + "  if ($_.Exception.Message -ne 'mock pip install failed with exit code 17.') { exit 3 }\n"
        + "  exit 0\n"
        + "}\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-File", str(script_path)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_recursive_cleanup_is_confined_to_resolved_engine_root(tmp_path: Path):
    powershell = _powershell()
    source = INSTALLER.read_text(encoding="utf-8")
    helpers = "\n".join(
        _function_source(source, name)
        for name in ("Resolve-AudioEnginePath", "Assert-AudioEnginePath", "Remove-AudioEngineAttemptDirectory")
    )
    engine_root = tmp_path / "engines" / "audio"
    outside = tmp_path / "outside-attempt"
    script_path = tmp_path / "path_guard.ps1"
    script_path.write_text(
        "$ErrorActionPreference = 'Stop'\n"
        + "$EngineDir = '"
        + str(engine_root).replace("'", "''")
        + "'\n"
        + "New-Item -ItemType Directory -Force -Path $EngineDir | Out-Null\n"
        + helpers
        + "\n$inside = Join-Path $EngineDir 'owned-attempt'\n"
        + "New-Item -ItemType Directory -Path $inside | Out-Null\n"
        + "Remove-AudioEngineAttemptDirectory -Path $inside\n"
        + "if (Test-Path $inside) { exit 1 }\n"
        + "$outside = '"
        + str(outside).replace("'", "''")
        + "'\nNew-Item -ItemType Directory -Path $outside | Out-Null\n"
        + "try { Remove-AudioEngineAttemptDirectory -Path $outside; exit 2 } catch {}\n"
        + "if (!(Test-Path $outside)) { exit 3 }\n"
        + "exit 0\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-File", str(script_path)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_installer_script_parses_without_execution(tmp_path: Path):
    powershell = _powershell()
    script_path = tmp_path / "parse_installer.ps1"
    script_path.write_text(
        "$tokens = $null; $errors = $null; "
        + "[System.Management.Automation.Language.Parser]::ParseFile('"
        + str(INSTALLER).replace("'", "''")
        + "', [ref]$tokens, [ref]$errors) | Out-Null; "
        + "if ($errors.Count) { $errors | ForEach-Object { Write-Error $_ }; exit 1 }; exit 0",
        encoding="utf-8",
    )

    result = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-File", str(script_path)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
