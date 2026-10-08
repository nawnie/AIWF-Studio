from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from aiwf.services import qwen_nunchaku_engine_setup as setup
from aiwf.core.config.settings import RuntimeFlags
from aiwf.services.qwen_nunchaku import QwenNunchakuService


def test_qwen_nunchaku_engine_install_starts_fixed_bootstrap_and_tracks_status(tmp_path: Path, monkeypatch):
    root = tmp_path / "project"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    bootstrap = scripts / "bootstrap_qwen_nunchaku.ps1"
    bootstrap.write_text("Write-Host setup", encoding="utf-8")
    created = {}

    class FakeProcess:
        pid = 4242

        def __init__(self):
            self.exit_code = None

        def poll(self):
            return self.exit_code

    process = FakeProcess()

    def fake_popen(command, **kwargs):
        created["command"] = command
        created["kwargs"] = kwargs
        return process

    monkeypatch.setattr(setup.subprocess, "Popen", fake_popen)
    data_root = tmp_path / "custom-data"
    started = setup.start_qwen_nunchaku_engine_install(root, data_root)

    assert started["status"] == "started"
    assert started["pid"] == 4242
    assert created["command"][-4:] == ["-File", str(bootstrap), "-DataRoot", str(data_root.resolve())]
    assert QwenNunchakuService(RuntimeFlags(data_dir=data_root)).engine_root() == data_root.resolve() / "engines" / "qwen_nunchaku"
    assert created["kwargs"]["stdin"] is setup.subprocess.DEVNULL
    assert "generation stays blocked" in started["message"]
    assert setup.qwen_nunchaku_engine_install_status(data_root)["running"] is True

    process.exit_code = 0
    finished = setup.qwen_nunchaku_engine_install_status(data_root)
    assert finished["status"] == "finished"
    assert finished["exitCode"] == 0


def test_qwen_nunchaku_engine_install_rejects_missing_fixed_bootstrap(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="bootstrap is unavailable"):
        setup.start_qwen_nunchaku_engine_install(tmp_path)


def test_qwen_nunchaku_bootstrap_pins_and_hash_checks_isolated_runtime():
    root = Path(__file__).resolve().parents[2]
    script = (root / "scripts" / "bootstrap_qwen_nunchaku.ps1").read_text(encoding="utf-8")
    requirements = (root / "engines" / "qwen_nunchaku" / "requirements.txt").read_text(encoding="utf-8")

    assert "-3.12 -m venv" in script
    assert "torch==2.11.0" in script and "cu130" in script
    assert "WheelSha256" in script and "Get-FileHash" in script
    assert "pip install --no-deps $WheelPath" in script
    assert 'Join-Path $DataRoot "engines\\qwen_nunchaku"' in script
    assert 'NewGuid().ToString("N")' in script
    assert 'Join-Path $Root "engines\\qwen_nunchaku\\run_qwen_lightning.py"' in script
    assert 'Join-Path $EngineDir "run_qwen_lightning.py"' in script
    assert 'Copy-Item -LiteralPath $RunnerSource -Destination $RunnerPath -Force' in script
    assert "Generation" not in script
    assert "diffusers==0.36.0" in requirements
    assert "transformers==4.55.2" in requirements
