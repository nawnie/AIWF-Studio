from __future__ import annotations

import logging
import os

from aiwf.app import (
    _api_security_middleware,
    _auth_pairs,
    _configure_logging,
    _friendly_device_name,
    _friendly_library_message,
    _log_api_security_warnings,
    _mount_gradio_extensions,
    _resolve_flags,
    _startup_message,
)
from aiwf.bootstrap import build_context
from aiwf.core.util.access import build_network_access_info
from aiwf.core.util.network import find_free_port

logger = logging.getLogger("aiwf")


def run() -> None:
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    flags = _resolve_flags()
    _configure_logging(flags.data_dir)
    _startup_message("Starting AIWF Studio Modern...")
    _startup_message("Checking your hardware and loading tools...")
    ctx = build_context(flags)

    server_name = "0.0.0.0" if flags.listen else "127.0.0.1"
    port = flags.port
    try:
        find_free_port(port, attempts=1)
    except OSError:
        port = find_free_port(flags.port + 1, attempts=32)
        logger.warning("Port %d is already in use. AIWF Studio Modern will use %d instead.", flags.port, port)
    ctx.runtime_port = port

    from aiwf.web.modern import create_modern_web_ui

    _startup_message(f"Using {_friendly_device_name(ctx.generation.backend.devices.describe())}.")
    checkpoint_count = len(ctx.generation.list_checkpoints())
    lora_count = len(ctx.generation.list_loras())
    _startup_message(_friendly_library_message(checkpoint_count, lora_count))
    _startup_message("Building the modern workspace...")
    demo, theme, css, js = create_modern_web_ui(ctx)

    launch_kwargs = dict(
        server_name=server_name,
        server_port=port,
        share=flags.share,
        inbrowser=flags.autolaunch,
        auth=_auth_pairs(flags.gradio_auth),
        prevent_thread_lock=True,
        theme=theme,
        css=css,
        js=js,
        quiet=True,
    )
    security_middleware = _api_security_middleware(flags)
    if security_middleware:
        launch_kwargs["app_kwargs"] = {"middleware": security_middleware}
    _log_api_security_warnings(flags)
    access = build_network_access_info(listen=flags.listen, port=port)
    app, local_url, share_url = demo.launch(**launch_kwargs)
    _mount_gradio_extensions(app, ctx)
    if flags.api:
        from aiwf.api.v1.routes import build_router

        app.include_router(build_router(ctx))
        logger.info("API mounted at /api/v1")

    _startup_message("AIWF Studio Modern is ready.")
    _startup_message(f"Open in your browser: {local_url}")
    if access.recommended_phone_url:
        _startup_message(f"Phone and tablet access: {access.recommended_phone_url}")
    elif not flags.listen:
        _startup_message("Phone and tablet access is off. Turn it on later in Settings -> Remote access.")
    if share_url:
        _startup_message(f"Public share link: {share_url}")

    try:
        import time

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutdown requested")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
