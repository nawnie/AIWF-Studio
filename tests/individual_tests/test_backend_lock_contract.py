from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
LOCK_DIR = ROOT / "dependencies" / "windows-py312"


def test_windows_backend_lock_declares_its_supported_runtime_and_sources() -> None:
    project = (LOCK_DIR / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.12.13,<3.13"' in project
    assert '"sys_platform == \'win32\' and platform_machine == \'AMD64\'"' in project
    assert 'name = "pytorch-cu130"' in project
    assert 'explicit = true' in project
    assert 'torch = { index = "pytorch-cu130" }' in project
    assert 'torchvision = { index = "pytorch-cu130" }' in project
    assert 'build-constraint-dependencies = ["setuptools==78.1.0", "wheel==0.48.0"]' in project


def test_lock_hashes_filterpy_source_and_records_its_pinned_builders() -> None:
    lock = (LOCK_DIR / "uv.lock").read_text(encoding="utf-8")
    artifacts = re.findall(r'hash = "(sha256:[0-9a-f]{64})"', lock)
    assert len(artifacts) == 375
    filterpy = re.search(r'\[\[package\]\]\nname = "filterpy".*?(?=\n\[\[package\]\]|\Z)', lock, re.S)
    assert filterpy is not None
    assert 'sdist = {' in filterpy.group(0)
    assert 'name = "setuptools"\nversion = "78.1.0"' in lock
    assert 'name = "wheel"\nversion = "0.48.0"' in lock


def test_installer_lock_is_explicit_and_gated_without_replacing_default() -> None:
    installer = (ROOT / "scripts" / "install_aiwf_studio.ps1").read_text(encoding="utf-8-sig")
    assert '[switch]$UseBackendLock' in installer
    assert "Assert-BackendLockHost" in installer
    assert "Assert-BackendLockPython" in installer
    assert "if (Test-BackendLockPythonVersion -Version $verifiedPython) { $uvOk = $true }" in installer
    assert 'elseif ((Get-VenvPythonMinor) -eq $PythonVersion)' in installer
    assert "CPython >=3.12.13,<3.13 on Windows AMD64" in installer
    assert '"sync", "--project", $lockProject, "--locked", "--inexact"' in installer
    assert '"import launch; launch.prepare(False, False, [])"' in installer
    assert '"UV_PROJECT_ENVIRONMENT", $previousProjectEnvironment, "Process"' in installer


def test_locked_install_docs_name_the_scope_and_explicit_command() -> None:
    docs = (ROOT / "docs" / "DEPENDENCY_POLICY.md").read_text(encoding="utf-8")
    scripts_docs = (ROOT / "scripts" / "README.md").read_text(encoding="utf-8")
    assert "-UseBackendLock" in docs
    assert "Windows AMD64" in " ".join(docs.split())
    assert "-UseBackendLock" in scripts_docs


def test_installer_uv_provisioning_branches_with_mocked_external_calls() -> None:
    pwsh = shutil.which("pwsh") or shutil.which("powershell")
    if pwsh is None:
        pytest.skip("PowerShell is required for the installer behavior harness")
    harness = Path(__file__).with_name("installer_uv_behavior.ps1")
    installer = ROOT / "scripts" / "install_aiwf_studio.ps1"
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as output:
        result = subprocess.run(
            [pwsh, "-NoProfile", "-File", str(harness), str(installer)],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=output,
            timeout=60,
        )
        output.seek(0)
        transcript = output.read()
    assert result.returncode == 0, transcript
    assert "PASS:" in transcript
