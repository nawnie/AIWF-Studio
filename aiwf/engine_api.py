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
from types import SimpleNamespace
from typing import Any

HOST = "127.0.0.1"          # loopback only, by design
DEFAULT_PORT = 7870         # Pro keeps 7860; the native app's engines.json points here


def build_app() -> Any:
    """FastAPI app with only the unified router mounted."""
    from fastapi import FastAPI

    from aiwf.core.config.launch import launch_settings_path, load_launch_settings, merge_launch_settings
    from aiwf.core.config.settings import RuntimeFlags
    from aiwf.web.unified_api import build_unified_router

    # this block resolves the same data/output folders Pro would use, without Pro's CLI parsing
    defaults = RuntimeFlags()
    saved = load_launch_settings(launch_settings_path(defaults.data_dir))
    flags = merge_launch_settings(defaults, saved, explicit=set())

    app = FastAPI(title="AIWF engine API", docs_url=None, redoc_url=None, openapi_url=None)
    app.include_router(build_unified_router(SimpleNamespace(flags=flags)))

    # a tiny landing document so a person opening the port sees what it is
    @app.get("/")
    def root() -> dict[str, Any]:
        return {
            "service": "aiwf-engine-api",
            "base_path": "/api/pro/unified",
            "describe": "/api/pro/unified/describe",
            "output_dir": str(flags.resolved_output_dir()),
        }

    return app


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="AIWF engine API (loopback only) for AIWF Studio for Windows.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"loopback port (default {DEFAULT_PORT})")
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")

    import uvicorn

    uvicorn.run(build_app(), host=HOST, port=args.port, log_level="info", access_log=False)


if __name__ == "__main__":
    main()
