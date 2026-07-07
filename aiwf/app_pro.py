from __future__ import annotations

import logging
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware import Middleware

from aiwf.api.v1.client_log import build_client_log_router
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
from aiwf.web.ext_api import build_extension_router

logger = logging.getLogger("aiwf")


def _generation_backend_device_label(ctx: AppContext) -> str:
    backend = getattr(getattr(ctx, "generation", None), "backend", None)
    devices = getattr(backend, "devices", None)
    if devices is not None and callable(getattr(devices, "describe", None)):
        try:
            return _friendly_device_name(devices.describe())
        except Exception:
            logger.debug("Could not describe generation backend device.", exc_info=True)
    backend_name = backend.__class__.__name__ if backend is not None else "generation backend"
    return backend_name.replace("StableDiffusionCppBackend", "stable-diffusion.cpp backend")


def _pro_icon_path(name: str) -> Path:
    return Path(__file__).resolve().parent.parent / "static" / "icons" / name


def _pro_autolaunch_enabled(argv: list[str]) -> bool:
    return "--no-autolaunch" not in argv


def _browser_app_command(url: str, *, profile_dir: Path | None = None) -> list[str] | None:
    candidates = [
        "chrome",
        "chrome.exe",
        "msedge",
        "msedge.exe",
    ]
    for name in candidates:
        executable = shutil.which(name)
        if executable:
            command = [executable]
            if profile_dir is not None:
                command.append(f"--user-data-dir={profile_dir}")
            command.extend(["--new-window", "--start-maximized", f"--app={url}"])
            return command

    known_paths = [
        Path(os.environ.get("ProgramFiles", "")) / "Google" / "Chrome" / "Application" / "chrome.exe",
        Path(os.environ.get("ProgramFiles(x86)", "")) / "Google" / "Chrome" / "Application" / "chrome.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "Chrome" / "Application" / "chrome.exe",
        Path(os.environ.get("ProgramFiles", "")) / "Microsoft" / "Edge" / "Application" / "msedge.exe",
        Path(os.environ.get("ProgramFiles(x86)", "")) / "Microsoft" / "Edge" / "Application" / "msedge.exe",
    ]
    for path in known_paths:
        if path.is_file():
            command = [str(path)]
            if profile_dir is not None:
                command.append(f"--user-data-dir={profile_dir}")
            command.extend(["--new-window", "--start-maximized", f"--app={url}"])
            return command
    return None


def _monitor_app_window(proc: subprocess.Popen) -> None:
    proc.wait()
    logger.info("AIWF Studio Pro app window closed; stopping the backend process.")
    os._exit(0)


def _open_app_window(
    url: str,
    *,
    profile_dir: Path | None = None,
    shutdown_on_close: bool = False,
) -> bool:
    if profile_dir is not None:
        profile_dir.mkdir(parents=True, exist_ok=True)
    command = _browser_app_command(url, profile_dir=profile_dir)
    if command is not None:
        try:
            proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if shutdown_on_close:
                threading.Thread(
                    target=_monitor_app_window,
                    args=(proc,),
                    name="aiwf-pro-window-monitor",
                    daemon=True,
                ).start()
            return True
        except OSError as exc:
            logger.warning("Could not open Pro app window: %s", exc)
    return bool(webbrowser.open(url))


def _wait_for_http(url: str, *, timeout_seconds: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0):
                return True
        except OSError:
            time.sleep(0.25)
    return False


def _existing_pro_runtime(url: str) -> bool:
    runtime_url = f"{url.rstrip('/')}/api/pro/runtime"
    try:
        with urllib.request.urlopen(runtime_url, timeout=1.5) as response:
            if getattr(response, "status", 200) >= 400:
                return False
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return False
    return isinstance(payload, dict) and "status" in payload and "backend" in payload


def _open_app_window_when_ready(
    url: str,
    *,
    ready_url: str | None = None,
    profile_dir: Path | None = None,
    shutdown_on_close: bool = False,
) -> None:
    if not _wait_for_http(ready_url or url):
        logger.warning("AIWF Studio Pro did not answer before the app window timeout.")
    _open_app_window(url, profile_dir=profile_dir, shutdown_on_close=shutdown_on_close)


def _schedule_app_window_open(
    url: str,
    *,
    ready_url: str | None = None,
    profile_dir: Path | None = None,
    shutdown_on_close: bool = False,
) -> None:
    threading.Thread(
        target=_open_app_window_when_ready,
        kwargs={"url": url, "ready_url": ready_url, "profile_dir": profile_dir, "shutdown_on_close": shutdown_on_close},
        name="aiwf-pro-autolaunch",
        daemon=True,
    ).start()


def _frontend_dist() -> Path:
    return Path(__file__).resolve().parent.parent / "frontend" / "dist"


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _mount_frontend(app: FastAPI, dist: Path) -> bool:
    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return FileResponse(_pro_icon_path("aiwf-studio-pro.ico"), media_type="image/x-icon")

    @app.get("/app-icon.png", include_in_schema=False)
    def app_icon():
        return FileResponse(_pro_icon_path("aiwf-studio-pro.png"), media_type="image/png")

    @app.get("/manifest.webmanifest", include_in_schema=False)
    def web_manifest():
        return JSONResponse(
            {
                "name": "AIWF Studio Pro",
                "short_name": "AIWF Pro",
                "start_url": "/",
                "scope": "/",
                "display": "fullscreen",
                "background_color": "#101820",
                "theme_color": "#101820",
                "icons": [
                    {"src": "/app-icon.png", "sizes": "512x512", "type": "image/png"},
                    {"src": "/favicon.ico", "sizes": "16x16 32x32 48x48", "type": "image/x-icon"},
                ],
            }
        )

    index = dist / "index.html"
    if not dist.is_dir() or not index.is_file():
        return False

    @app.get("/{requested_path:path}", include_in_schema=False)
    def frontend(requested_path: str = ""):
        if requested_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API route not found")
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

    @app.middleware("http")
    async def add_pro_timing_header(request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        if request.url.path.startswith("/api/pro"):
            elapsed_ms = (time.perf_counter() - started) * 1000
            response.headers["X-AIWF-Elapsed-Ms"] = f"{elapsed_ms:.1f}"
        return response

    app.include_router(build_client_log_router(ctx), prefix="/api/v1")
    app.include_router(build_router(ctx))
    # User extensions: routers registered via ctx.plugins.register_api(id, router)
    # are served under /api/ext/<plugin-id>/. A broken extension must never
    # take the whole app down, so mounting failures are logged and skipped.
    for plugin_id, plugin_router in getattr(getattr(ctx, "plugins", None), "api_routers", []) or []:
        try:
            app.include_router(plugin_router, prefix=f"/api/ext/{plugin_id}")
            logger.info("Mounted extension API: /api/ext/%s", plugin_id)
        except Exception:
            logger.exception("Could not mount extension API for %s", plugin_id)
    # Built-in extension API (workflow blocks, Ollama chat bridge, media browse).
    app.include_router(build_extension_router(ctx))
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
    if _pro_autolaunch_enabled(sys.argv[1:]):
        flags = flags.model_copy(update={"autolaunch": True})
    _configure_logging(flags.data_dir)
    _startup_message("Starting AIWF Studio Pro...")
    _startup_message("Checking your hardware and loading tools...")
    ctx = build_context(flags)

    host = "0.0.0.0" if flags.listen else "127.0.0.1"
    port = flags.port
    local_url = f"http://127.0.0.1:{port}"
    try:
        find_free_port(port, attempts=1)
    except OSError as exc:
        if _existing_pro_runtime(local_url):
            logger.warning("AIWF Studio Pro is already running at %s. Reusing the existing session.", local_url)
            _startup_message(f"AIWF Studio Pro is already running at {local_url}")
            if flags.autolaunch:
                _schedule_app_window_open(local_url, ready_url=f"{local_url}/api/pro/startup")
            return
        raise RuntimeError(
            f"Port {port} is already in use by another process. Close the stale listener or free the port, then relaunch Pro."
        ) from exc
    ctx.runtime_port = port

    _startup_message(f"Using {_generation_backend_device_label(ctx)}.")
    checkpoint_count = len(ctx.generation.list_checkpoints())
    lora_count = len(ctx.generation.list_loras())
    _startup_message(_friendly_library_message(checkpoint_count, lora_count))

    middleware = _api_security_middleware(flags)
    app = create_app(ctx, middleware=middleware)
    frontend_ready = (_frontend_dist() / "index.html").is_file()
    browser_profile_dir = Path(flags.data_dir) / "_local" / "pro-browser-profile"
    _log_security_warnings(flags)

    _startup_message("AIWF Studio Pro is ready.")
    if frontend_ready:
        _startup_message(f"AIWF Studio Pro is available at {local_url}")
        if flags.autolaunch:
            @app.on_event("startup")
            def open_pro_app_window() -> None:
                _schedule_app_window_open(
                    local_url,
                    ready_url=f"{local_url}/api/pro/startup",
                    profile_dir=browser_profile_dir,
                    shutdown_on_close=True,
                )
    else:
        _startup_message(f"API ready at {local_url}/api/pro")
        _startup_message("React frontend build not found at frontend/dist; run the frontend build to serve it here.")

    import uvicorn

    uvicorn.run(app, host=host, port=port)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
