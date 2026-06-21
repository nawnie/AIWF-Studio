from __future__ import annotations

import json
from pathlib import Path

from aiwf.services.training.engine_status import (
    FULL_TUNE_ENGINE,
    full_tune_ready,
    full_tune_status_markdown,
    ready_engine_choices,
    training_engine_statuses,
    training_status_markdown,
)


def test_training_engine_status_reports_disabled_missing_venv(tmp_path: Path):
    (tmp_path / "engines" / "kohya" / "kohya_ss").mkdir(parents=True)
    (tmp_path / "engines" / "kohya" / "worker.py").write_text("print('worker')", encoding="utf-8")
    (tmp_path / "engines.json").write_text(
        json.dumps({"kohya": {"enabled": False}}),
        encoding="utf-8",
    )

    statuses = training_engine_statuses(tmp_path)

    assert not statuses["kohya"].ready
    assert not statuses["kohya"].enabled
    assert any("enabled=true" in message for message in statuses["kohya"].messages)


def test_training_engine_status_ready_when_enabled_repo_venv_and_worker_exist(tmp_path: Path):
    repo = tmp_path / "engines" / "kohya" / "kohya_ss"
    python = tmp_path / "engines" / "kohya" / ".venv" / "Scripts" / "python.exe"
    worker = tmp_path / "engines" / "kohya" / "worker.py"
    repo.mkdir(parents=True)
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    worker.write_text("print('worker')", encoding="utf-8")
    (tmp_path / "engines.json").write_text(
        json.dumps({"kohya": {"enabled": True}}),
        encoding="utf-8",
    )

    statuses = training_engine_statuses(tmp_path)

    assert statuses["kohya"].ready
    assert ready_engine_choices(statuses) == ["Kohya LoRA"]


def test_ed2_requires_train_py_for_full_tune_readiness(tmp_path: Path):
    repo = tmp_path / "engines" / "ed2" / "EveryDream2trainer"
    python = tmp_path / "engines" / "ed2" / ".venv" / "Scripts" / "python.exe"
    worker = tmp_path / "engines" / "ed2" / "worker.py"
    repo.mkdir(parents=True)
    python.parent.mkdir(parents=True)
    worker.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("", encoding="utf-8")
    worker.write_text("print('worker')", encoding="utf-8")
    (tmp_path / "engines.json").write_text(
        json.dumps({"ed2": {"enabled": True}}),
        encoding="utf-8",
    )

    statuses = training_engine_statuses(tmp_path)

    assert FULL_TUNE_ENGINE == "ed2"
    assert not statuses["ed2"].ready
    assert not full_tune_ready(statuses)
    assert "train.py" in full_tune_status_markdown(statuses)


def test_ed2_ready_satisfies_full_tune_policy(tmp_path: Path):
    repo = tmp_path / "engines" / "ed2" / "EveryDream2trainer"
    python = tmp_path / "engines" / "ed2" / ".venv" / "Scripts" / "python.exe"
    worker = tmp_path / "engines" / "ed2" / "worker.py"
    repo.mkdir(parents=True)
    python.parent.mkdir(parents=True)
    worker.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("", encoding="utf-8")
    worker.write_text("print('worker')", encoding="utf-8")
    (repo / "train.py").write_text("print('train')", encoding="utf-8")
    (tmp_path / "engines.json").write_text(
        json.dumps({"ed2": {"enabled": True}}),
        encoding="utf-8",
    )

    statuses = training_engine_statuses(tmp_path)

    assert statuses["ed2"].ready
    assert full_tune_ready(statuses)
    assert "ED2 Full Fine-tune" in ready_engine_choices(statuses)


def test_ed2_can_use_shared_studio_venv_for_full_tune_readiness(tmp_path: Path):
    repo = tmp_path / "engines" / "ed2" / "EveryDream2trainer"
    python = tmp_path / "venv" / "Scripts" / "python.exe"
    worker = tmp_path / "engines" / "ed2" / "worker.py"
    repo.mkdir(parents=True)
    python.parent.mkdir(parents=True)
    worker.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("", encoding="utf-8")
    worker.write_text("print('worker')", encoding="utf-8")
    (repo / "train.py").write_text("print('train')", encoding="utf-8")
    (tmp_path / "engines.json").write_text(
        json.dumps({"ed2": {"enabled": True, "venv_dir": "studio"}}),
        encoding="utf-8",
    )

    statuses = training_engine_statuses(tmp_path)

    assert statuses["ed2"].ready
    assert statuses["ed2"].uses_studio_venv
    assert statuses["ed2"].python_exe == python.resolve()
    assert "Studio runtime" in statuses["ed2"].markdown_line()


def test_llm_training_engine_ready_when_enabled_venv_and_worker_exist(tmp_path: Path):
    repo = tmp_path / "engines" / "llm"
    python = tmp_path / "engines" / "llm" / ".venv" / "Scripts" / "python.exe"
    worker = tmp_path / "engines" / "llm" / "worker.py"
    repo.mkdir(parents=True)
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    worker.write_text("print('worker')", encoding="utf-8")
    (tmp_path / "engines.json").write_text(
        json.dumps({"llm": {"enabled": True}}),
        encoding="utf-8",
    )

    statuses = training_engine_statuses(tmp_path)

    assert statuses["llm"].ready
    assert "AI Bot Trainer" in ready_engine_choices(statuses)


def test_training_status_markdown_mentions_optional_setup(tmp_path: Path):
    statuses = training_engine_statuses(tmp_path)

    text = training_status_markdown(statuses)

    assert "Training engines are optional" in text
    assert "engines.json" in text
