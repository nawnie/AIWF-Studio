from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from starlette.middleware import Middleware

from aiwf.app import (
    _api_security_middleware,
    _configure_logging,
    _friendly_device_name,
    _friendly_library_message,
    _resolve_flags,
    _startup_message,
)
from aiwf.bootstrap import AppContext, build_context
from aiwf.core.util.network import find_free_port
from aiwf.web.pro_api import build_router

logger = logging.getLogger("aiwf")


def _frontend_dist() -> Path:
    return Path(__file__).resolve().parent.parent / "frontend" / "dist"


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _mount_frontend(app: FastAPI, dist: Path) -> bool:
    index = dist / "index.html"
    if not dist.is_dir() or not index.is_file():
        return False

    @app.get("/{requested_path:path}", include_in_schema=False)
    def frontend(requested_path: str = ""):
        target = (dist / requested_path).resolve() if requested_path else index
        if requested_path and _is_inside(target, dist) and target.is_file():
            return FileResponse(target)
        return FileResponse(index)

    return True


def create_app(
    ctx: AppContext,
    *,
    middleware: list[Middleware] | None = None,
    frontend_dist: Path | None = None,
) -> FastAPI:
    app = FastAPI(title="AIWF Studio Pro", middleware=middleware or [])
    app.include_router(build_router(ctx))
    _mount_frontend(app, frontend_dist or _frontend_dist())
    return app


def _log_security_warnings(flags) -> None:
    from aiwf.api.security import api_security_warnings

    for warning in api_security_warnings(
        listen=flags.listen,
        gradio_auth=flags.gradio_auth,
        api=True,
        nowebui=True,
    ):
        logger.warning("Security: %s", warning)


def run() -> None:
    flags = _resolve_flags()
    _configure_logging(flags.data_dir)
    _startup_message("Starting AIWF Studio Pro...")
    _startup_message("Checking your hardware and loading tools...")
    ctx = build_context(flags)

    host = "0.0.0.0" if flags.listen else "127.0.0.1"
    port = flags.port
    try:
        find_free_port(port, attempts=1)
    except OSError:
        port = find_free_port(flags.port + 1, attempts=32)
        logger.warning("Port %d is already in use. AIWF Studio Pro will use %d instead.", flags.port, port)
    ctx.runtime_port = port

    _startup_message(f"Using {_friendly_device_name(ctx.generation.backend.devices.describe())}.")
    checkpoint_count = len(ctx.generation.list_checkpoints())
    lora_count = len(ctx.generation.list_loras())
    _startup_message(_friendly_library_message(checkpoint_count, lora_count))

    middleware = _api_security_middleware(flags)
    app = create_app(ctx, middleware=middleware)
    frontend_ready = (_frontend_dist() / "index.html").is_file()
    _log_security_warnings(flags)

    local_url = f"http://127.0.0.1:{port}"
    _startup_message("AIWF Studio Pro is ready.")
    if frontend_ready:
        _startup_message(f"Open in your browser: {local_url}")
    else:
        _startup_message(f"API ready at {local_url}/api/pro")
        _startup_message("React frontend build not found at frontend/dist; run the frontend build to serve it here.")

    import uvicorn

    uvicorn.run(app, host=host, port=port)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
