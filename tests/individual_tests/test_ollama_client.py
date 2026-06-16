"""Tests for aiwf/services/ollama_client.py.

All tests mock httpx — no live Ollama server required.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, call

import pytest

from aiwf.services.ollama_client import OllamaClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(status_code: int = 200, json_body: dict | None = None, text: str = "OK"):
    """Build a mock httpx.Response-like object."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body or {}
    resp.text = text
    resp.raise_for_status = MagicMock(
        side_effect=None if status_code < 400 else Exception(f"HTTP {status_code}")
    )
    return resp


# ---------------------------------------------------------------------------
# OllamaClient basics
# ---------------------------------------------------------------------------

class TestOllamaClientInit:
    def test_default_base_url(self):
        c = OllamaClient()
        assert c._base_url == "http://127.0.0.1:11434"

    def test_custom_base_url_strips_trailing_slash(self):
        c = OllamaClient(base_url="http://myhost:11434/")
        assert c._base_url == "http://myhost:11434"


# ---------------------------------------------------------------------------
# healthcheck
# ---------------------------------------------------------------------------

class TestHealthcheck:
    @patch("httpx.get")
    def test_returns_true_on_200(self, mock_get):
        mock_get.return_value = _make_response(200)
        c = OllamaClient()
        assert c.healthcheck() is True

    @patch("httpx.get")
    def test_returns_false_on_connection_error(self, mock_get):
        mock_get.side_effect = Exception("connection refused")
        c = OllamaClient()
        assert c.healthcheck() is False

    @patch("httpx.get")
    def test_returns_false_on_non_200(self, mock_get):
        mock_get.return_value = _make_response(503)
        c = OllamaClient()
        # healthcheck checks status_code == 200
        assert c.healthcheck() is False


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------

class TestListModels:
    @patch("httpx.get")
    def test_returns_model_names(self, mock_get):
        mock_get.return_value = _make_response(
            json_body={"models": [{"name": "llama3:8b"}, {"name": "mistral:latest"}]}
        )
        c = OllamaClient()
        models = c.list_models()
        assert models == ["llama3:8b", "mistral:latest"]

    @patch("httpx.get")
    def test_returns_empty_list_on_error(self, mock_get):
        mock_get.side_effect = Exception("no connection")
        c = OllamaClient()
        assert c.list_models() == []

    @patch("httpx.get")
    def test_empty_models_key(self, mock_get):
        mock_get.return_value = _make_response(json_body={"models": []})
        c = OllamaClient()
        assert c.list_models() == []


# ---------------------------------------------------------------------------
# unload
# ---------------------------------------------------------------------------

class TestUnload:
    @patch("httpx.post")
    def test_returns_true_on_success(self, mock_post):
        mock_post.return_value = _make_response(200)
        c = OllamaClient()
        assert c.unload("llama3:8b") is True
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert kwargs["json"]["keep_alive"] == 0
        assert kwargs["json"]["model"] == "llama3:8b"

    @patch("httpx.post")
    def test_returns_false_on_error(self, mock_post):
        mock_post.side_effect = Exception("timeout")
        c = OllamaClient()
        assert c.unload("llama3:8b") is False

    def test_empty_model_returns_false_without_http(self):
        c = OllamaClient()
        assert c.unload("") is False


# ---------------------------------------------------------------------------
# stream_chat
# ---------------------------------------------------------------------------

class TestStreamChat:
    def _make_stream_response(self, tokens: list[str], done_after: int = -1):
        """Build a mock streaming context manager that yields JSONL lines."""
        lines = []
        for i, token in enumerate(tokens):
            is_last = i == len(tokens) - 1
            lines.append(json.dumps({
                "message": {"role": "assistant", "content": token},
                "done": is_last,
            }))

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_lines = MagicMock(return_value=iter(lines))

        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=mock_response)
        ctx.__exit__ = MagicMock(return_value=False)
        return ctx

    @patch("httpx.stream")
    def test_yields_tokens(self, mock_stream):
        mock_stream.return_value = self._make_stream_response(["Hello", ", ", "world", "!"])
        c = OllamaClient()
        tokens = list(c.stream_chat("llama3:8b", [{"role": "user", "content": "Hi"}]))
        assert tokens == ["Hello", ", ", "world", "!"]

    @patch("httpx.stream")
    def test_skips_empty_content(self, mock_stream):
        lines = [
            json.dumps({"message": {"role": "assistant", "content": "A"}, "done": False}),
            json.dumps({"message": {"role": "assistant", "content": ""}, "done": False}),
            json.dumps({"message": {"role": "assistant", "content": "B"}, "done": True}),
        ]
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_lines = MagicMock(return_value=iter(lines))
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=mock_response)
        ctx.__exit__ = MagicMock(return_value=False)
        mock_stream.return_value = ctx

        c = OllamaClient()
        tokens = list(c.stream_chat("llama3:8b", []))
        assert tokens == ["A", "B"]

    @patch("httpx.stream")
    def test_raises_on_error_chunk(self, mock_stream):
        lines = [json.dumps({"error": "model not found"})]
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_lines = MagicMock(return_value=iter(lines))
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=mock_response)
        ctx.__exit__ = MagicMock(return_value=False)
        mock_stream.return_value = ctx

        c = OllamaClient()
        with pytest.raises(RuntimeError, match="model not found"):
            list(c.stream_chat("bad-model", []))


# ---------------------------------------------------------------------------
# model_info
# ---------------------------------------------------------------------------

class TestModelInfo:
    @patch("httpx.post")
    def test_returns_dict(self, mock_post):
        payload = {"details": {"family": "llama"}, "parameters": "temperature 0.8"}
        mock_post.return_value = _make_response(json_body=payload)
        c = OllamaClient()
        info = c.model_info("llama3:8b")
        assert info["details"]["family"] == "llama"

    @patch("httpx.post")
    def test_returns_empty_dict_on_error(self, mock_post):
        mock_post.side_effect = Exception("not found")
        c = OllamaClient()
        assert c.model_info("bad-model") == {}
