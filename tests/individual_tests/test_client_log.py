"""Dev-only client logging tests — run with: pytest -m dev"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiwf.api.v1.client_log import build_client_log_router
from aiwf.core.config.settings import RuntimeFlags

pytestmark = pytest.mark.dev


def test_client_error_endpoint_logs(tmp_path):
    flags = RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models")
    ctx = SimpleNamespace(flags=flags)
    app = FastAPI()
    app.include_router(build_client_log_router(ctx), prefix="/api/v1")
    client = TestClient(app)

    response = client.post(
        "/api/v1/client-errors",
        json={
            "kind": "unhandledrejection",
            "message": "Cannot read properties of null (reading 'get_data')",
            "stack": "TypeError: Cannot read properties of null",
            "source": "Blocks-IWJfIfJH.js:3:7443",
            "url": "http://127.0.0.1:7860/",
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "logged"
    log_path = tmp_path / "outputs" / "client-errors.log"
    assert log_path.is_file()
    text = log_path.read_text(encoding="utf-8")
    assert "get_data" in text
    assert "unhandledrejection" in text