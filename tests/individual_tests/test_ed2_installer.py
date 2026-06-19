from __future__ import annotations

import json
import subprocess
from pathlib import Path

from aiwf.services.training.ed2_installer import install_ed2_addon


def test_install_ed2_addon_configures_existing_repo_without_network(tmp_path: Path):
    repo = tmp_path / "engines" / "ed2" / "EveryDream2trainer"
    repo.mkdir(parents=True)
    (repo / "train.py").write_text("print('train')", encoding="utf-8")

    logs = install_ed2_addon(tmp_path, install_requirements=False)

    config = json.loads((tmp_path / "engines.json").read_text(encoding="utf-8"))
    assert config["ed2"]["enabled"] is True
    assert config["ed2"]["repo_dir"] == "engines/ed2/EveryDream2trainer"
    assert config["ed2"]["venv_dir"] == "studio"
    assert any("already present" in line for line in logs)


def test_install_ed2_addon_clones_configured_fork_into_engines(tmp_path: Path):
    commands: list[list[str]] = []

    def fake_run(command, **kwargs):
        commands.append(list(command))
        target = Path(command[-1])
        target.mkdir(parents=True)
        (target / "train.py").write_text("print('train')", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="")

    logs = install_ed2_addon(
        tmp_path,
        repo_url="https://github.com/shawn/EveryDream2trainer.git",
        install_requirements=False,
        run_command=fake_run,
    )

    assert commands == [
        [
            "git",
            "clone",
            "https://github.com/shawn/EveryDream2trainer.git",
            str(tmp_path / "engines" / "ed2" / "EveryDream2trainer"),
        ]
    ]
    assert (tmp_path / "engines" / "ed2" / "EveryDream2trainer" / "train.py").exists()
    assert any("shawn/EveryDream2trainer" in line for line in logs)
