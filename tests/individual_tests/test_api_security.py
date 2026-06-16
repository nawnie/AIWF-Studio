from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiwf.api.security import (
    ApiRateLimitMiddleware,
    api_security_warnings,
    is_private_url,
    parse_cors_origins,
)


def test_parse_cors_origins_accepts_commas_and_newlines():
    assert parse_cors_origins("http://127.0.0.1:3000, https://studio.example\nhttp://localhost:5173") == [
        "http://127.0.0.1:3000",
        "https://studio.example",
        "http://localhost:5173",
    ]


def test_api_security_warnings_for_network_without_auth():
    warnings = api_security_warnings(listen=True, gradio_auth="", api=True, nowebui=False)
    assert any("without UI auth" in warning for warning in warnings)
    assert any("REST API" in warning for warning in warnings)


def test_is_private_url_blocks_local_network_targets():
    assert is_private_url("http://127.0.0.1/model.safetensors") is True
    assert is_private_url("http://10.0.0.5/model.safetensors") is True
    assert is_private_url("file:///C:/models/model.safetensors") is True
    assert is_private_url("https://8.8.8.8/model.safetensors") is False


def test_api_rate_limit_middleware_limits_api_routes_only():
    app = FastAPI()

    @app.get("/api/v1/ping")
    def api_ping():
        return {"ok": True}

    @app.get("/health")
    def health():
        return {"ok": True}

    app.add_middleware(ApiRateLimitMiddleware, requests_per_minute=1)
    client = TestClient(app)

    assert client.get("/api/v1/ping").status_code == 200
    limited = client.get("/api/v1/ping")
    assert limited.status_code == 429
    assert limited.json()["detail"] == "API rate limit exceeded"
    assert client.get("/health").status_code == 200
