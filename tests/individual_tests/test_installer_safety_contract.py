"""Regression tests for safe recovery of the Windows installer environments."""

import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
INSTALLER = ROOT / "scripts" / "install_aiwf_studio.ps1"


def _function_source(source: str, name: str) -> str:
    match = re.search(rf"(?ms)^function {re.escape(name)}\s*\{{.*?^\}}", source)
    assert match is not None, f"installer function {name!r} was not found"
    return match.group(0)


def _function_body(source: str, name: str) -> str:
    match = re.search(rf"(?ms)^function {re.escape(name)}\s*\{{(?P<body>.*?)^\}}", source)
    assert match is not None, f"installer function {name!r} was not found"
    return match.group("body")


def _powershell_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _powershell(tmp_path: Path) -> str:
    executable = shutil.which("powershell") or shutil.which("pwsh")
    if executable is None:
        pytest.skip("PowerShell is unavailable")
    return executable


def test_python_environment_setup_is_serialized_with_an_exclusive_file_lock():
    source = INSTALLER.read_text(encoding="utf-8")
    lock_body = _function_body(source, "Invoke-WithInstallerEnvironmentLock")
    setup_body = _function_body(source, "Ensure-PythonVenv")

    assert 'Join-Path $localRoot "installer-setup.lock"' in lock_body
    assert "[System.IO.FileShare]::None" in lock_body
    assert "Dispose()" in lock_body
    assert "Invoke-WithInstallerEnvironmentLock -Action {" not in setup_body
    assert re.search(
        r"(?ms)Invoke-WithInstallerEnvironmentLock -Action \{\s+Ensure-PythonVenv\s+Prepare-AiwfRuntime\s+Install-DefaultBaseModel\s+Build-ProFrontend\s+\}",
        source,
    )


def test_conda_python_rebuild_uses_unique_recoverable_backups():
    source = INSTALLER.read_text(encoding="utf-8")
    move_source = _function_source(source, "Move-StaleCondaEnv")
    conda_body = _function_body(source, "Ensure-PythonVenv-Conda")

    assert "[guid]::NewGuid().ToString('N')" in move_source
    assert 'Join-Path (Join-Path $Root "_trash") $backupName' in move_source
    assert "Move-Item -LiteralPath $PythonEnvironmentDir -Destination $trash" in move_source
    assert "Move-StaleCondaEnv -PythonEnvironmentDir $pyenv" in conda_body
    assert conda_body.index("Move-StaleCondaEnv -PythonEnvironmentDir $pyenv") < conda_body.index(
        'Invoke-External "Create conda Python'
    )
    assert "Remove-Item" not in conda_body


@pytest.mark.skipif(sys.platform != "win32", reason="AIWF's installer is Windows-only")
def test_stale_conda_environment_move_preserves_contents_and_unique_retries(tmp_path: Path):
    powershell = _powershell(tmp_path)
    source = INSTALLER.read_text(encoding="utf-8")
    move_source = _function_source(source, "Move-StaleCondaEnv")
    guard_source = _function_source(source, "Assert-TaskLocalWritePath")
    env_path = tmp_path / "_pyenv312"
    env_path.mkdir()
    (env_path / "sentinel.txt").write_text("original environment", encoding="utf-8")

    script_path = tmp_path / "exercise_move.ps1"
    script = (
        "$ErrorActionPreference = 'Stop'\n$TaskLocalMode = $false\n$Root = "
        + _powershell_quote(str(tmp_path))
        + "\n"
        + guard_source
        + "\n"
        + move_source
        + "\nMove-StaleCondaEnv -PythonEnvironmentDir (Join-Path $Root '_pyenv312')\n"
        + "New-Item -ItemType Directory -Path (Join-Path $Root '_pyenv312') | Out-Null\n"
        + "Set-Content -LiteralPath (Join-Path (Join-Path $Root '_pyenv312') 'sentinel.txt') -Value 'second environment'\n"
        + "Move-StaleCondaEnv -PythonEnvironmentDir (Join-Path $Root '_pyenv312')\n"
    )
    script_path.write_text(script, encoding="utf-8")
    result = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(script_path)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    backups = sorted((tmp_path / "_trash").glob("installer-conda-python-*"))
    assert len(backups) == 2
    assert backups[0].name != backups[1].name
    assert sorted((backup / "sentinel.txt").read_text(encoding="utf-8").rstrip("\r\n") for backup in backups) == [
        "original environment",
        "second environment",
    ]
    assert not env_path.exists()


@pytest.mark.skipif(sys.platform != "win32", reason="AIWF's installer is Windows-only")
def test_environment_lock_rejects_a_second_installer_process(tmp_path: Path):
    powershell = _powershell(tmp_path)
    source = INSTALLER.read_text(encoding="utf-8")
    lock_source = _function_source(source, "Invoke-WithInstallerEnvironmentLock")
    guard_source = _function_source(source, "Assert-TaskLocalWritePath")
    root_literal = _powershell_quote(str(tmp_path))
    first_path = tmp_path / "hold_lock.ps1"
    second_path = tmp_path / "try_lock.ps1"
    first_path.write_text(
        "$Root = "
        + root_literal
        + "\n"
        + lock_source
        + "\nInvoke-WithInstallerEnvironmentLock -Action { [System.IO.File]::WriteAllText((Join-Path $Root 'lock-ready.txt'), 'ready'); Start-Sleep -Seconds 3 }\n",
        encoding="utf-8",
    )
    second_path.write_text(
        "$Root = "
        + root_literal
        + "\n"
        + lock_source
        + "\nInvoke-WithInstallerEnvironmentLock -Action { Write-Output 'second acquired' }\n",
        encoding="utf-8",
    )

    first = subprocess.Popen(
        [powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(first_path)],
        cwd=tmp_path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        deadline = time.monotonic() + 10
        ready = tmp_path / "lock-ready.txt"
        while not ready.exists() and first.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert ready.exists(), "first process did not acquire the exclusive installer lock"
        second = subprocess.run(
            [powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(second_path)],
            cwd=tmp_path,
            text=True,
            capture_output=True,
            check=False,
        )
        assert second.returncode != 0
        assert "another aiwf studio installer is performing setup" in (second.stdout + second.stderr).lower()
    finally:
        if first.poll() is None:
            first.terminate()
        first.communicate(timeout=10)
@pytest.mark.skipif(sys.platform != "win32", reason="AIWF's installer is Windows-only")
def test_task_local_mode_targets_marked_copy_without_shortcuts_or_models(tmp_path: Path):
    powershell = _powershell(tmp_path)
    (tmp_path / ".aiwf-task-local-install").write_text("disposable test target", encoding="utf-8")
    (tmp_path / "launch.py").write_text("# isolated marker", encoding="utf-8")
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "package-lock.json").write_text("{}", encoding="utf-8")
    env_path = tmp_path / "_pyenv312"
    env_path.mkdir()
    (env_path / "sentinel.txt").write_text("preserve-me", encoding="utf-8")
    script_path = tmp_path / "run_task_local_dryrun.ps1"
    environment_names = [
        "UV_CACHE_DIR",
        "UV_PYTHON_INSTALL_DIR",
        "UV_PYTHON_BIN_DIR",
        "PIP_CACHE_DIR",
        "NPM_CONFIG_CACHE",
        "CONDA_PKGS_DIRS",
        "CONDARC",
    ]
    script_lines = [
        "$env:UV_CACHE_DIR = 'prior-uv-cache'",
        "$env:UV_PYTHON_INSTALL_DIR = 'prior-uv-python'",
        "$env:UV_PYTHON_BIN_DIR = 'prior-uv-python-bin'",
        "$env:PIP_CACHE_DIR = 'prior-pip-cache'",
        "$env:NPM_CONFIG_CACHE = 'prior-npm-cache'",
        "$env:CONDA_PKGS_DIRS = 'prior-conda-pkgs'",
        "$env:CONDARC = 'prior-condarc'",
        "& " + _powershell_quote(str(INSTALLER)) + " -Mode express -TaskLocalMode -TaskLocalRoot "
        + _powershell_quote(str(tmp_path))
        + " -DryRun -SkipRuntimeSetup -SkipFrontendBuild",
        "[System.IO.File]::WriteAllLines("
        + _powershell_quote(str(tmp_path / "environment-after.txt"))
        + ", @(" + ", ".join("$env:" + name for name in environment_names) + "))",
    ]
    script_path.write_text("\n".join(script_lines), encoding="utf-8")

    result = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(script_path)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert str(tmp_path / "venv") in output
    assert "Skipping Desktop and Start Menu shortcuts in task-local mode." in output
    assert "Would install Stable Diffusion" not in output
    assert (tmp_path / "_local" / "uv-cache").is_dir()
    assert (tmp_path / "_local" / "uv-python").is_dir()
    assert (tmp_path / "_local" / "installer-setup.lock").is_file()
    assert (tmp_path / "environment-after.txt").read_text(encoding="utf-8").splitlines() == [
        "prior-uv-cache",
        "prior-uv-python",
        "prior-uv-python-bin",
        "prior-pip-cache",
        "prior-npm-cache",
        "prior-conda-pkgs",
        "prior-condarc",
    ]
    assert (env_path / "sentinel.txt").read_text(encoding="utf-8") == "preserve-me"
    assert not (tmp_path / "venv").exists()


@pytest.mark.skipif(sys.platform != "win32", reason="AIWF's installer is Windows-only")
def test_environment_lock_releases_and_restores_task_local_uv_paths_after_failure(tmp_path: Path):
    powershell = _powershell(tmp_path)
    source = INSTALLER.read_text(encoding="utf-8")
    guard_source = _function_source(source, "Assert-TaskLocalWritePath")
    lock_source = _function_source(source, "Invoke-WithInstallerEnvironmentLock")
    script_path = tmp_path / "failure_retry.ps1"
    script = (
        "$Root = "
        + _powershell_quote(str(tmp_path))
        + "\n$TaskLocalMode = $true\n"
        + "$env:UV_CACHE_DIR = 'prior-cache'\n"
        + "$env:UV_PYTHON_INSTALL_DIR = 'prior-python'\n"
        + guard_source
        + "\n"
        + lock_source
        + "\n$failed = $false\n"
        + "try { Invoke-WithInstallerEnvironmentLock -Action { "
        + "[System.IO.File]::WriteAllText((Join-Path $Root 'paths.txt'), ($env:UV_CACHE_DIR + [Environment]::NewLine + $env:UV_PYTHON_INSTALL_DIR)); "
        + "throw 'intentional setup failure' } } catch { $failed = $true }\n"
        + "if (-not $failed) { exit 10 }\n"
        + "if ($env:UV_CACHE_DIR -ne 'prior-cache' -or $env:UV_PYTHON_INSTALL_DIR -ne 'prior-python') { exit 11 }\n"
        + "Invoke-WithInstallerEnvironmentLock -Action { [System.IO.File]::WriteAllText((Join-Path $Root 'retry.txt'), 'acquired') }\n"
    )
    script_path.write_text(script, encoding="utf-8")

    result = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(script_path)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    paths = (tmp_path / "paths.txt").read_text(encoding="utf-8").splitlines()
    assert paths == [
        str(tmp_path / "_local" / "uv-cache"),
        str(tmp_path / "_local" / "uv-python"),
    ]
    assert (tmp_path / "retry.txt").read_text(encoding="utf-8") == "acquired"

@pytest.mark.skipif(sys.platform != "win32", reason="AIWF's installer is Windows-only")
def test_task_local_root_rejects_checkout_and_nested_paths(tmp_path: Path):
    powershell = _powershell(tmp_path)
    source = INSTALLER.read_text(encoding="utf-8")
    root_source = _function_source(source, "Resolve-TaskLocalRoot")
    nested = tmp_path / "nested"
    nested.mkdir()
    script_path = tmp_path / "reject_checkout_paths.ps1"
    script_path.write_text(
        root_source
        + "\n$failedExact = $false\ntry { Resolve-TaskLocalRoot -Candidate "
        + _powershell_quote(str(tmp_path))
        + " -SourceCheckout "
        + _powershell_quote(str(tmp_path))
        + " } catch { $failedExact = $_.Exception.Message -like '*source checkout*' }\n"
        + "$failedNested = $false\ntry { Resolve-TaskLocalRoot -Candidate "
        + _powershell_quote(str(nested))
        + " -SourceCheckout "
        + _powershell_quote(str(tmp_path))
        + " } catch { $failedNested = $_.Exception.Message -like '*source checkout*' }\n"
        + "if (-not $failedExact -or -not $failedNested) { exit 10 }\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(script_path)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(sys.platform != "win32", reason="AIWF's installer is Windows-only")
def test_task_local_conda_fallback_is_disabled_to_avoid_user_profile_registration(tmp_path: Path):
    powershell = _powershell(tmp_path)
    source = INSTALLER.read_text(encoding="utf-8")
    conda_source = _function_source(source, "Ensure-PythonVenv-Conda")
    script_path = tmp_path / "reject_global_miniconda.ps1"
    script_path.write_text(
        "$TaskLocalMode = $true\n$PythonVersion = '3.12'\n"
        + "function Get-CondaCommand { throw 'conda discovery must not run' }\n"
        + "function Read-Host { throw 'unexpected prompt' }\n"
        + "function Install-Miniconda { throw 'unexpected global install' }\n"
        + conda_source
        + "\n$rejected = $false\ntry { Ensure-PythonVenv-Conda } catch { $rejected = $_.Exception.Message -like '*will not use conda*' }\n"
        + "if (-not $rejected) { exit 10 }\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(script_path)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_task_local_mode_rejects_default_model_and_requires_a_marked_copy():
    source = INSTALLER.read_text(encoding="utf-8")
    root_validation = _function_body(source, "Resolve-TaskLocalRoot")
    task_local_block = source.split("if ($TaskLocalMode) {", 1)[1].split("} elseif", 1)[0]

    assert "ReparsePoint" in root_validation
    assert "StartsWith($sourcePrefix" in root_validation
    assert '".aiwf-task-local-install", "launch.py", "frontend\\package-lock.json"' in task_local_block
    assert "-or $WithDefaultModel -or $WithNvidiaVideoFx -or $FullImageStack" in task_local_block


@pytest.mark.skipif(sys.platform != "win32", reason="AIWF's installer is Windows-only")
def test_task_local_write_path_guard_rejects_nested_junction(tmp_path: Path):
    powershell = _powershell(tmp_path)
    source = INSTALLER.read_text(encoding="utf-8")
    guard_source = _function_source(source, "Assert-TaskLocalWritePath")
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "copy"
    root.mkdir()
    junction = root / "_local"
    link = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
        text=True,
        capture_output=True,
        check=False,
    )
    if link.returncode != 0:
        pytest.skip("Windows junction creation is unavailable")
    script_path = tmp_path / "reject_nested_junction.ps1"
    script_path.write_text(
        "$TaskLocalMode = $true\n$Root = "
        + _powershell_quote(str(root))
        + "\n"
        + guard_source
        + "\n$rejected = $false\ntry { Assert-TaskLocalWritePath -Path (Join-Path $Root '_local\\uv-cache') } catch { $rejected = $_.Exception.Message -like '*junction or symbolic-link component*' }\n"
        + "if (-not $rejected) { exit 10 }\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(script_path)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_task_local_installer_isolates_package_caches_and_serializes_frontend_build():
    source = INSTALLER.read_text(encoding="utf-8")
    assert 'SetEnvironmentVariable("PIP_CACHE_DIR"' in source
    assert 'SetEnvironmentVariable("NPM_CONFIG_CACHE"' in source
    assert 'SetEnvironmentVariable("CONDA_PKGS_DIRS"' in source
    assert 'SetEnvironmentVariable("CONDARC"' in source
    assert 'SetEnvironmentVariable($name, $PreviousTaskLocalEnvironment[$name], "Process")' in source
    assert re.search(
        r"(?ms)Invoke-WithInstallerEnvironmentLock -Action \{\s+Ensure-PythonVenv\s+Prepare-AiwfRuntime\s+Install-DefaultBaseModel\s+Build-ProFrontend\s+\}",
        source,
    )


@pytest.mark.skipif(sys.platform != "win32", reason="AIWF's installer is Windows-only")
def test_task_local_mode_rejects_incomplete_and_model_install_requests(tmp_path: Path):
    powershell = _powershell(tmp_path)
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    result = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(INSTALLER), "-TaskLocalMode", "-TaskLocalRoot", str(incomplete), "-DryRun"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "not a marked aiwf studio copy" in (result.stdout + result.stderr).lower()

    (tmp_path / ".aiwf-task-local-install").write_text("disposable test target", encoding="utf-8")
    (tmp_path / "launch.py").write_text("# isolated marker", encoding="utf-8")
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "package-lock.json").write_text("{}", encoding="utf-8")
    result = subprocess.run(
        [powershell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(INSTALLER), "-TaskLocalMode", "-TaskLocalRoot", str(tmp_path), "-WithDefaultModel", "-DryRun"],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "supports express only" in (result.stdout + result.stderr).lower()
