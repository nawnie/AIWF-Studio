"""
aiwf/services/ollama_client.py

Thin HTTP wrapper around Ollama's REST API.

No mandatory install — if ``httpx`` is not available the client raises
``ImportError`` with a helpful install hint on first use.

Ollama must be running locally (default http://127.0.0.1:11434).
The client does NOT start or install Ollama — it just talks to it.

Public API
----------
    client = OllamaClient()

    if client.healthcheck():
        models = client.list_models()
        for token in client.stream_chat("llama3:8b", messages, {}):
            print(token, end="", flush=True)
        client.unload("llama3:8b")
"""
from __future__ import annotations

import json
import logging
from typing import Iterator

logger = logging.getLogger(__name__)

# Default timeout for non-streaming requests (healthcheck, list, unload)
_DEFAULT_TIMEOUT = 8.0
# Timeout for the initial response header on a streaming chat request
_STREAM_CONNECT_TIMEOUT = 15.0
# Read timeout per chunk during streaming (0 = no per-chunk timeout)
_STREAM_READ_TIMEOUT = 0


def _httpx():
    """Import httpx or raise a friendly error."""
    try:
        import httpx
        return httpx
    except ImportError as exc:
        raise ImportError(
            "httpx is required for OllamaClient. "
            "Install it with:  pip install httpx"
        ) from exc


class OllamaClient:
    """Minimal Ollama REST client.

    Args:
        base_url: Root URL of the Ollama server.  Defaults to the local
                  default port.  Do NOT include a trailing slash.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:11434") -> None:
        self._base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def healthcheck(self) -> bool:
        """Return True if Ollama is reachable and responding."""
        httpx = _httpx()
        try:
            r = httpx.get(f"{self._base_url}/", timeout=_DEFAULT_TIMEOUT)
            return r.status_code == 200
        except Exception as exc:
            logger.debug("[OllamaClient] healthcheck failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Model listing
    # ------------------------------------------------------------------

    def list_models(self) -> list[str]:
        """Return a list of model names available in Ollama.

        Returns an empty list (instead of raising) if Ollama is unreachable.
        Model names are in the form ``llama3:8b``, ``mistral:latest``, etc.
        """
        httpx = _httpx()
        try:
            r = httpx.get(f"{self._base_url}/api/tags", timeout=_DEFAULT_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception as exc:
            logger.warning("[OllamaClient] list_models failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Unload
    # ------------------------------------------------------------------

    def unload(self, model: str) -> bool:
        """Ask Ollama to evict *model* from VRAM (keep_alive: 0).

        Returns True on success, False if Ollama rejected the request or is
        unreachable.
        """
        if not model:
            return False
        httpx = _httpx()
        try:
            r = httpx.post(
                f"{self._base_url}/api/generate",
                json={"model": model, "keep_alive": 0},
                timeout=_DEFAULT_TIMEOUT,
            )
            r.raise_for_status()
            logger.info("[OllamaClient] Unloaded model %r", model)
            return True
        except Exception as exc:
            logger.warning("[OllamaClient] unload(%r) failed: %s", model, exc)
            return False

    # ------------------------------------------------------------------
    # Chat (streaming)
    # ------------------------------------------------------------------

    def stream_chat(
        self,
        model: str,
        messages: list[dict],
        options: dict | None = None,
    ) -> Iterator[str]:
        """Stream chat completion tokens from Ollama.

        Args:
            model:    Model tag, e.g. ``"llama3:8b"``.
            messages: OpenAI-style message list, e.g.:
                      ``[{"role": "user", "content": "Hello!"}]``
            options:  Optional Ollama model parameters, e.g.
                      ``{"temperature": 0.7, "num_ctx": 4096}``.

        Yields
        ------
        str
            Each token/chunk of the assistant response, as it arrives.

        Raises
        ------
        httpx.HTTPStatusError
            If Ollama returns a non-2xx response.
        RuntimeError
            If the response stream contains an error object.
        """
        httpx = _httpx()
        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if options:
            payload["options"] = options

        timeout = httpx.Timeout(
            connect=_STREAM_CONNECT_TIMEOUT,
            read=None,   # no per-chunk timeout — generation can be slow
            write=_DEFAULT_TIMEOUT,
            pool=_DEFAULT_TIMEOUT,
        )

        with httpx.stream(
            "POST",
            f"{self._base_url}/api/chat",
            json=payload,
            timeout=timeout,
        ) as response:
            response.raise_for_status()
            for raw in response.iter_lines():
                if not raw:
                    continue
                try:
                    chunk = json.loads(raw)
                except json.JSONDecodeError:
                    logger.debug("[OllamaClient] Non-JSON line: %r", raw)
                    continue

                if chunk.get("error"):
                    raise RuntimeError(f"Ollama error: {chunk['error']}")

                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token

                if chunk.get("done"):
                    break

    # ------------------------------------------------------------------
    # Model info
    # ------------------------------------------------------------------

    def model_info(self, model: str) -> dict:
        """Return Ollama's ``/api/show`` payload for *model*.

        Returns an empty dict if the model is not found or Ollama is down.
        Useful keys: ``"details"``, ``"parameters"``, ``"template"``.
        """
        httpx = _httpx()
        try:
            r = httpx.post(
                f"{self._base_url}/api/show",
                json={"name": model},
                timeout=_DEFAULT_TIMEOUT,
            )
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            logger.debug("[OllamaClient] model_info(%r) failed: %s", model, exc)
            return {}
