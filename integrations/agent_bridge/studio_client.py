"""Shared HTTP client for agents talking to AIWF Studio Pro's unified workspace API.

Used by both aiwf_studio_cli.py and aiwf_studio_mcp.py. Standard library only,
so the CLI runs on any Python 3.10+ without a virtual environment.

The operation list comes from aiwf/services/unified_contract.py, loaded by file
path so that importing it never pulls in the rest of AIWF Studio.

Configuration (environment variables):
    AIWF_STUDIO_URL     base URL of AIWF Studio Pro (default http://127.0.0.1:7860).
                        Must be a loopback address; anything else is refused.
    AIWF_STUDIO_TIMEOUT seconds per request (default 300; Qwen answers and
                        ReTrain preflights can take minutes on first model load).
"""

from __future__ import annotations

import importlib.util
import ipaddress
import json
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from types import ModuleType
from typing import Any


# --- locations ---------------------------------------------------------------------
STUDIO_ROOT = Path(__file__).resolve().parents[2]                     # F:\AIWF_Studio
CONTRACT_FILE = STUDIO_ROOT / "aiwf" / "services" / "unified_contract.py"
LOCAL_ONLY_LAUNCHER = STUDIO_ROOT / "AIWF Studio Pro (Local Only).bat"


# --- contract loading ----------------------------------------------------------------
def load_contract() -> ModuleType:
    """Load the operation contract by file path (no AIWF package import)."""
    spec = importlib.util.spec_from_file_location("aiwf_unified_contract", CONTRACT_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load the unified contract from {CONTRACT_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StudioError(Exception):
    """An operation failed. `code` is stable for agents; `message` is for people."""

    def __init__(self, status: int, code: str, message: str, detail: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.detail = detail

    def as_dict(self) -> dict[str, Any]:
        return {"ok": False, "status": self.status, "code": self.code, "message": self.message}


def _is_loopback(url: str) -> bool:
    host = urllib.parse.urlparse(url).hostname or ""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


# --- the client ------------------------------------------------------------------------
class StudioClient:
    def __init__(self, base_url: str | None = None, timeout: float | None = None) -> None:
        self.contract = load_contract()
        self.base_url = (base_url or os.environ.get("AIWF_STUDIO_URL") or self.contract.DEFAULT_BASE_URL).rstrip("/")
        self.timeout = float(timeout or os.environ.get("AIWF_STUDIO_TIMEOUT") or 300)
        if not _is_loopback(self.base_url):
            raise StudioError(0, "non_loopback_url", f"Refusing a non-loopback AIWF Studio URL: {self.base_url}")

    def operations(self) -> list[dict[str, Any]]:
        return self.contract.OPERATIONS

    # this method turns one operation + arguments into exactly one HTTP request
    def call(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        operation = self.contract.find_operation(name)
        arguments = dict(arguments or {})
        known = {param["name"] for param in operation["params"]}
        unknown = sorted(set(arguments) - known)
        if unknown:
            raise StudioError(0, "invalid_arguments", f"{operation['name']} does not take: {', '.join(unknown)}")
        missing = [param["name"] for param in operation["params"] if param["required"] and arguments.get(param["name"]) in (None, "")]
        if missing:
            raise StudioError(0, "invalid_arguments", f"{operation['name']} needs: {', '.join(missing)}")

        # path parameters are URL-quoted; query and body parameters are split by location
        path = operation["path"]
        query: dict[str, Any] = {}
        body: dict[str, Any] = {}
        for param in operation["params"]:
            if param["name"] not in arguments or arguments[param["name"]] is None:
                continue
            value = arguments[param["name"]]
            if param["in"] == "path":
                path = path.replace("{" + param["name"] + "}", urllib.parse.quote(str(value), safe=""))
            elif param["in"] == "query":
                query[param["name"]] = value
            else:
                body[param["name"]] = value
        url = f"{self.base_url}{self.contract.BASE_PATH}{path}"
        if query:
            url += "?" + urllib.parse.urlencode(query)
        data = json.dumps(body).encode("utf-8") if operation["method"] == "POST" else None
        return self._send(operation["method"], url, data)

    def _send(self, method: str, url: str, data: bytes | None) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, method=method, headers=headers)
        # proxies are bypassed: these are loopback calls only
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise _error_from_response(exc) from exc
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, TimeoutError) or "timed out" in str(reason):
                raise StudioError(0, "timeout", f"AIWF Studio Pro did not answer within {self.timeout:.0f} s.") from exc
            raise StudioError(
                0, "pro_not_running",
                f"AIWF Studio Pro is not running at {self.base_url}. Start 'AIWF Studio Pro (Local Only)' or run: aiwf-studio start-pro",
            ) from exc
        except json.JSONDecodeError as exc:
            raise StudioError(0, "bad_response", "AIWF Studio Pro returned a response that is not JSON.") from exc

    # --- readiness and launch ------------------------------------------------------------
    def is_running(self) -> bool:
        try:
            self.call("describe")
            return True
        except StudioError:
            return False

    def start_pro(self, wait_seconds: float = 240.0) -> dict[str, Any]:
        """Start AIWF Studio Pro with the loopback-only launcher unless it already answers.

        Uses 'AIWF Studio Pro (Local Only).bat' with no app window and no loading
        window, so it binds 127.0.0.1 only. Returns the unified status once ready.
        """
        if self.is_running():
            return {"ok": True, "started": False, "message": "AIWF Studio Pro is already running.", "status": self.call("status")}
        if not _is_default_local_url(self.base_url):
            raise StudioError(0, "cannot_start", f"start-pro only launches the local default ({self.contract.DEFAULT_BASE_URL}); {self.base_url} is configured.")
        if not LOCAL_ONLY_LAUNCHER.is_file():
            raise StudioError(0, "cannot_start", f"Launcher not found: {LOCAL_ONLY_LAUNCHER}")
        env = dict(os.environ, AIWF_PRO_LOADING_WINDOW="0")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.Popen(
            ["cmd.exe", "/c", str(LOCAL_ONLY_LAUNCHER), "--no-autolaunch"],
            cwd=str(STUDIO_ROOT), env=env, creationflags=flags,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        # this loop waits for the real backend: the launcher's placeholder answers 503 first
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            time.sleep(2)
            try:
                return {"ok": True, "started": True, "message": "AIWF Studio Pro started (loopback only).", "status": self.call("status")}
            except StudioError:
                continue
        raise StudioError(0, "start_timeout", f"AIWF Studio Pro did not become ready within {wait_seconds:.0f} s. See F:\\AIWF_Studio\\logs\\pro-hidden-launch.err.log")


def _is_default_local_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    return parsed.hostname in {"127.0.0.1", "localhost"} and (parsed.port or 80) == 7860


def _error_from_response(exc: urllib.error.HTTPError) -> StudioError:
    """Map AIWF's {detail: {code, message}} (or FastAPI validation errors) to StudioError."""
    try:
        payload = json.loads(exc.read().decode("utf-8"))
    except (ValueError, OSError):
        payload = None
    detail = payload.get("detail") if isinstance(payload, dict) else None
    if isinstance(detail, dict) and "message" in detail:
        return StudioError(exc.code, str(detail.get("code", "error")), str(detail["message"]), detail)
    if isinstance(detail, list):
        problems = "; ".join(f"{'.'.join(str(p) for p in item.get('loc', [])[1:])}: {item.get('msg')}" for item in detail if isinstance(item, dict))
        return StudioError(exc.code, "validation_error", f"Request rejected: {problems}", detail)
    if exc.code == 404:
        return StudioError(404, "not_found", "Route not found. This AIWF Pro build may predate the unified workspace bridge.", detail)
    if exc.code == 503:
        return StudioError(503, "pro_starting", "AIWF Studio Pro is still starting; retry in a few seconds.", detail)
    return StudioError(exc.code, "http_error", str(detail or f"HTTP {exc.code}"), detail)
