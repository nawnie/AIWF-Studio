from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path


def _test_cache_file(project_root: Path, basetemp: str | None) -> Path:
    """Resolve a test cache path and reject locations inside the source checkout."""
    if basetemp:
        cache_root = Path(basetemp)
    else:
        cache_root = Path(tempfile.gettempdir()) / f"aiwf-pytest-{uuid.uuid4().hex}"
    resolved_root = cache_root.resolve()
    resolved_project = project_root.resolve()
    try:
        resolved_root.relative_to(resolved_project)
    except ValueError:
        return resolved_root / "model_header_cache.json"
    raise RuntimeError(f"pytest basetemp must be outside the AIWF source checkout: {resolved_root}")


def pytest_configure(config) -> None:
    """Keep the persistent model-header cache outside the source checkout in tests."""
    basetemp = getattr(config.option, "basetemp", None)
    project_root = Path(__file__).resolve().parents[1]
    os.environ["AIWF_MODEL_HEADER_CACHE_FILE"] = str(_test_cache_file(project_root, basetemp))
