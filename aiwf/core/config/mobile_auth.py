from __future__ import annotations

import json
import secrets
from pathlib import Path
from typing import Any

_MOBILE_AUTH_FILENAME = "mobile_auth.json"
_TOKEN_BYTES = 24


def mobile_auth_path(data_dir: Path) -> Path:
    return Path(data_dir) / "_local" / _MOBILE_AUTH_FILENAME


def _generate_token() -> str:
    return secrets.token_hex(_TOKEN_BYTES)


def load_mobile_auth(data_dir: Path) -> dict[str, Any]:
    path = mobile_auth_path(data_dir)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"enabled": False, "token": ""}
    if not isinstance(data, dict):
        return {"enabled": False, "token": ""}
    return {"enabled": bool(data.get("enabled", False)), "token": str(data.get("token") or "")}


def save_mobile_auth(data_dir: Path, *, enabled: bool, token: str) -> dict[str, Any]:
    path = mobile_auth_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"enabled": bool(enabled), "token": str(token or "")}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def ensure_mobile_token(data_dir: Path) -> str:
    """Return the persisted pairing token, generating one on first use.

    Generating a token does not enable remote access. The user must enable
    mobile access before a non-loopback client can use it.
    """
    state = load_mobile_auth(data_dir)
    if state["token"]:
        return state["token"]
    token = _generate_token()
    save_mobile_auth(data_dir, enabled=state["enabled"], token=token)
    return token


def set_mobile_access_enabled(data_dir: Path, *, enabled: bool, rotate: bool = False) -> dict[str, Any]:
    state = load_mobile_auth(data_dir)
    token = state["token"] or _generate_token()
    if rotate:
        token = _generate_token()
    return save_mobile_auth(data_dir, enabled=enabled, token=token)
