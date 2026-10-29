"""AIWF engine API: the unified workspace API on its own, for the native Windows app.

AIWF Studio for Windows (native/, C++/WinRT + WinUI 3) needs the engine API
(/api/pro/unified/*) but not the whole Pro web server with its React UI and
diffusion runtime. This host serves exactly the same router, built from the
same code and contract (aiwf/web/unified_api.py, aiwf/services/unified_contract.py),
in a small process that starts in a couple of seconds:

    venv\\Scripts\\python.exe -m aiwf.engine_api [--port 7870]

It always binds 127.0.0.1; there is deliberately no option to listen on the
network. Studio's saved launch settings (launch.json) are read only to find the
same data and output folders Pro uses, so projects, ledgers and generated images
are shared between the native app, Pro, the CLI and the MCP server.

The native app's engine supervisor starts this process inside a Job Object, so it
stops when the app closes. Nothing here starts training or touches sibling apps
except through the bridge's loopback-only clients.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import secrets
import socket
import tempfile
from types import SimpleNamespace
from pathlib import Path
from typing import Any

HOST = "127.0.0.1"          # loopback only, by design
DEFAULT_PORT = 18870         # Pro keeps 7860; the native app's engines.json points here


def token_file_for_port(port: int) -> Path:
    """Return the per-user token location shared with the native app and agents."""
    override = os.environ.get("AIWF_STUDIO_DATA_DIR", "").strip()
    if override:
        root = Path(override).expanduser()
        if not root.is_absolute() or root == Path(root.anchor):
            raise RuntimeError("AIWF_STUDIO_DATA_DIR must be an absolute, non-root folder path.")
        local_root = root
    else:
        local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
        if local_app_data:
            local_root = Path(local_app_data) / "Programs" / "AES Studio Data"
        elif os.name == "nt":
            local_root = Path.home() / "AppData" / "Local" / "Programs" / "AES Studio Data"
        else:
            local_root = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "AES Studio"
    return local_root / "_local" / f"engine-api-{int(port)}.token"


def _restrict_token_file_acl(path: Path) -> None:
    """Protect a newly written token file from other Windows accounts."""
    if os.name != "nt":
        return
    from ctypes import wintypes

    security_descriptor = ctypes.c_void_p()
    convert = ctypes.windll.advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW
    convert.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(wintypes.DWORD)]
    convert.restype = wintypes.BOOL
    if not convert("D:P(A;;FA;;;OW)", 1, ctypes.byref(security_descriptor), None):
        raise ctypes.WinError()
    try:
        set_security = ctypes.windll.advapi32.SetFileSecurityW
        set_security.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_void_p]
        set_security.restype = wintypes.BOOL
        dacl_information = 0x00000004 | 0x80000000  # DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION
        if not set_security(str(path), dacl_information, security_descriptor):
            raise ctypes.WinError()
    finally:
        ctypes.windll.kernel32.LocalFree(security_descriptor)


def _rotate_token_file(path: Path) -> str:
    """Create a fresh process token using a protected atomic replacement."""
    token = secrets.token_urlsafe(32)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".engine-api-token-", dir=path.parent)
    try:
        temp_path = Path(temp_name)
        if os.name != "nt":
            os.chmod(temp_path, 0o600)
        else:
            _restrict_token_file_acl(temp_path)
        with os.fdopen(fd, "w", encoding="ascii", newline="\n") as stream:
            stream.write(token + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        Path(temp_name).unlink(missing_ok=True)
        raise
    return token


def build_app(
    *, port: int = DEFAULT_PORT, token_file: Path | None = None, flags: Any | None = None,
    source_root: Path | None = None,
) -> Any:
    """FastAPI app with only the unified router mounted."""
    from fastapi import FastAPI

    from aiwf.api.security import NativeEngineApiSecurityMiddleware
    from aiwf.native_compat import native_api_health_metadata
    from aiwf.web.unified_api import build_unified_router

    if flags is None:
        from aiwf.core.config.launch import launch_settings_path, load_launch_settings, merge_launch_settings
        from aiwf.core.config.settings import RuntimeFlags

        # this block resolves the same data/output folders Pro would use, without Pro's CLI parsing
        defaults = RuntimeFlags()
        saved = load_launch_settings(launch_settings_path(defaults.data_dir))
        flags = merge_launch_settings(defaults, saved, explicit=set())

    compatibility_root = Path(source_root) if source_root is not None else Path(__file__).absolute().parents[1]
    compatibility = native_api_health_metadata(compatibility_root)
    source_fingerprint_provider = lambda: native_api_health_metadata(compatibility_root)["api_source_fingerprint"]
    token_path = Path(token_file) if token_file is not None else token_file_for_port(port)
    token = _rotate_token_file(token_path)

    app = FastAPI(title="AIWF engine API", docs_url=None, redoc_url=None, openapi_url=None)
    app.include_router(build_unified_router(SimpleNamespace(flags=flags)))
    app.add_middleware(
        NativeEngineApiSecurityMiddleware,
        port=port,
        token_provider=lambda: token,
        source_fingerprint_provider=source_fingerprint_provider,
        startup_fingerprint=compatibility["api_source_fingerprint"],
    )
    app.state.native_api_compatibility = compatibility

    # A nonsecret landing document; the engine API middleware still requires
    # the local bearer file here, just like other paths outside the health probes.
    @app.get("/")
    def root() -> dict[str, Any]:
        return {
            "service": "aiwf-engine-api",
            "base_path": "/api/pro/unified",
            "describe": "/api/pro/unified/describe",
        }

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"service": "aiwf-engine-api", "state": "ready", **compatibility}

    return app


def _server_loop() -> str:
    """Avoid the Windows Proactor listener closing after a reset during accept.

    This loopback API does not use asyncio subprocesses. Windows' selector is
    limited to 512 sockets, which is appropriate for this single-user service.
    See https://github.com/python/cpython/issues/93821.
    """
    return "asyncio:SelectorEventLoop" if os.name == "nt" else "auto"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="AIWF engine API (loopback only) for AIWF Studio for Windows.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"loopback port (default {DEFAULT_PORT})")
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")

    import uvicorn

    # Acquire the port before rotating its shared token file. A second process
    # that cannot bind must never invalidate the incumbent process's token.
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        server_socket.bind((HOST, args.port))
        server_socket.listen(2048)
        server_socket.setblocking(False)
        app = build_app(port=args.port)
        config = uvicorn.Config(app, host=HOST, port=args.port, log_level="info", access_log=False, loop=_server_loop())
        uvicorn.Server(config).run(sockets=[server_socket])
    finally:
        server_socket.close()


if __name__ == "__main__":
    main()
