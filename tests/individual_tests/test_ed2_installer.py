from __future__ import annotations

import json
from pathlib import Path

from aiwf.services.training.ed2_installer import install_ed2_addon


def test_install_ed2_addon_configures_existing_repo_without_network(tmp_path: Path):
    repo = tmp_path / "training" / "EveryDream2trainer"
    repo.mkdir(parents=True)
    (repo / "train.py").write_text("print('train')", encoding="utf-8")

    logs = install_ed2_addon(tmp_path, install_requirements=False)

    config = json.loads((tmp_path / "engines.json").read_text(encoding="utf-8"))
    assert config["ed2"]["enabled"] is True
    assert config["ed2"]["repo_dir"] == "training/EveryDream2trainer"
    assert config["ed2"]["venv_dir"] == "studio"
    assert any("already present" in line for line in logs)
