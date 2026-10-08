from __future__ import annotations

from pathlib import Path

from aiwf.infrastructure.model_header import ModelHeaderCache


def test_model_header_cache_writes_to_explicit_override(tmp_path, monkeypatch):
    cache_path = tmp_path / "isolated" / "model_header_cache.json"
    monkeypatch.setenv("AIWF_MODEL_HEADER_CACHE_FILE", str(cache_path))

    cache = ModelHeaderCache()
    cache.dirty = True
    cache.save()

    assert cache.cache_file == cache_path
    assert cache_path.read_text(encoding="utf-8") == "{}"


def test_model_header_cache_keeps_production_default_without_override(monkeypatch):
    monkeypatch.delenv("AIWF_MODEL_HEADER_CACHE_FILE", raising=False)

    cache = ModelHeaderCache()

    assert cache.cache_file == Path(__file__).resolve().parents[2] / "cache" / "model_header_cache.json"

import pytest

from conftest import _test_cache_file


def test_pytest_cache_rejects_in_checkout_basetemp(tmp_path):
    project_root = tmp_path / "checkout"
    project_root.mkdir()

    with pytest.raises(RuntimeError, match="outside the AIWF source checkout"):
        _test_cache_file(project_root, str(project_root / "tests-temp"))


def test_pytest_cache_resolves_external_basetemp(tmp_path):
    project_root = tmp_path / "checkout"
    project_root.mkdir()
    basetemp = tmp_path / "pytest-temp"

    assert _test_cache_file(project_root, str(basetemp)) == basetemp / "model_header_cache.json"
