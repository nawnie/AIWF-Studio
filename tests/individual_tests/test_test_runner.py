from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_runner():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "run_tests.py"
    spec = importlib.util.spec_from_file_location("aiwf_test_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_runner_lists_core_suites():
    runner = _load_runner()

    text = runner.list_suites()

    assert "core" in text
    assert "training" in text
    assert "wan" in text


def test_runner_resolves_named_suite_to_individual_folder():
    runner = _load_runner()

    paths = runner.paths_for_suite("core")

    assert paths
    assert all("individual_tests" in path for path in paths)
    assert any(path.endswith("test_launch.py") for path in paths)


def test_runner_resolves_individual_test_by_short_name():
    runner = _load_runner()

    path = runner.resolve_test_path("launch")

    assert path.endswith(str(Path("individual_tests") / "test_launch.py"))


def test_runner_resolves_multi_suite_and_dedupes():
    runner = _load_runner()

    targets = runner.resolve_targets(
        full=False,
        suites=["core", "ui"],
        tests=[],
        selections=["core"],
    )

    assert len(targets) == len(set(targets))
    assert any(target.endswith("test_settings.py") for target in targets)


def test_runner_builds_pytest_command():
    runner = _load_runner()

    command = runner.build_pytest_command(["tests/individual_tests/test_launch.py"], pytest_args=["-x"])

    assert command[:3] == [sys.executable, "-m", "pytest"]
    assert "-q" in command
    assert command[-1] == "-x"


def test_runner_rejects_unknown_suite():
    runner = _load_runner()

    with pytest.raises(ValueError, match="Unknown suite"):
        runner.paths_for_suite("missing")
