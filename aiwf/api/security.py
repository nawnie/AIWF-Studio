from __future__ import annotations

import ipaddress
import secrets
import socket
import time
import urllib.parse
from collections import defaultdict, deque
from collections.abc import Iterable
from pathlib import Path

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


API_PREFIXES = ("/api/", "/sdapi/")
LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


def parse_cors_origins(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = value.replace("\n", ",").split(",")
    else:
        raw = list(value)
    return [origin.strip() for origin in raw if str(origin).strip()]


def api_security_warnings(*, listen: bool, gradio_auth: str | None, api: bool, nowebui: bool) -> list[str]:
    warnings: list[str] = []
    if listen and not (gradio_auth or "").strip():
        warnings.append("Listening on the network without UI auth exposes AIWF to reachable devices.")
    if (api or nowebui) and listen and not (gradio_auth or "").strip():
        warnings.append("REST API is network-reachable without Gradio auth; use a trusted LAN/VPN only.")
    return warnings


def is_private_url(url: str) -> bool:
    parsed = urllib.parse.urlparse((url or "").strip())
    if parsed.scheme not in {"http", "https"}:
        return True
    host = parsed.hostname
    if not host:
        return True
    try:
        addresses = [ipaddress.ip_address(host)]
    except ValueError:
        try:
            infos = socket.getaddrinfo(host, None)
        except OSError:
            return False
        addresses = []
        for info in infos:
            sockaddr = info[4]
            if sockaddr:
                try:
                    addresses.append(ipaddress.ip_address(sockaddr[0]))
                except ValueError:
                    continue
    return any(
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
        for addr in addresses
    )


class ApiRateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, requests_per_minute: int = 0) -> None:
        super().__init__(app)
        self.requests_per_minute = max(0, int(requests_per_minute or 0))
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        if self.requests_per_minute <= 0 or not request.url.path.startswith(API_PREFIXES):
            return await call_next(request)

        client = request.client.host if request.client else "unknown"
        now = time.monotonic()
        hits = self._hits[client]
        cutoff = now - 60
        while hits and hits[0] < cutoff:
            hits.popleft()
        if len(hits) >= self.requests_per_minute:
            return JSONResponse(
                {"detail": "API rate limit exceeded"},
                status_code=429,
                headers={"Retry-After": "60"},
            )
        hits.append(now)
        return await call_next(request)


class MobileTokenAuthMiddleware(BaseHTTPMiddleware):
    """Requires ``X-AIWF-Token`` on ``/api/pro`` requests from non-loopback clients.

    Desktop traffic (the app's own frontend, always loopback) is never
    challenged. ``/api/pro/ping`` stays reachable without a token so an
    unpaired phone can discover and identify the server before it has one.
    Enforcement is read fresh from disk on every request (not cached at
    process start) so toggling mobile access on/off in Settings takes effect
    immediately, without restarting the backend.
    """

    _OPEN_PATHS = {"/api/pro/ping"}

    def __init__(self, app, *, data_dir: Path) -> None:
        super().__init__(app)
        self.data_dir = Path(data_dir)

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if not path.startswith("/api/pro") or path in self._OPEN_PATHS:
            return await call_next(request)
        client_host = request.client.host if request.client else ""
        if client_host in LOOPBACK_HOSTS:
            return await call_next(request)

        from aiwf.core.config.mobile_auth import load_mobile_auth

        state = load_mobile_auth(self.data_dir)
        if not state["enabled"] or not state["token"]:
            return JSONResponse(
                {"detail": "Mobile API access is disabled. Enable pairing from the local AIWF Studio app."},
                status_code=403,
            )
        supplied = request.headers.get("x-aiwf-token", "")
        if not supplied or not secrets.compare_digest(supplied, state["token"]):
            return JSONResponse({"detail": "Missing or invalid mobile pairing token."}, status_code=401)
        return await call_next(request)
