from __future__ import annotations

import json
from pathlib import Path

from aiwf.services.training.llm_installer import install_llm_trainer_addon


def test_install_llm_trainer_addon_only_configures_engine(tmp_path: Path):
    engine = tmp_path / "engines" / "llm"
    engine.mkdir(parents=True)
    (engine / "worker.py").write_text("print('worker')", encoding="utf-8")
    (engine / "requirements.txt").write_text("trl\n", encoding="utf-8")

    logs = install_llm_trainer_addon(tmp_path)

    config = json.loads((tmp_path / "engines.json").read_text(encoding="utf-8"))
    assert config["llm"]["enabled"] is True
    assert config["llm"]["venv_dir"] == "engines/llm/.venv"
    assert any("Restart AIWF Studio" in line for line in logs)
    assert not (tmp_path / "engines" / "llm" / ".venv").exists()
