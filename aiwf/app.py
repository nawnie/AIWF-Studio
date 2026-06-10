from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

os.environ.setdefault("XFORMERS_FORCE_DISABLE_TRITON", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from aiwf.bootstrap import build_context
from aiwf.core.config.launch import (
    explicit_cli_flags,
    launch_settings_path,
    load_launch_settings,
    merge_launch_settings,
)
from aiwf.core.config.settings import RuntimeFlags
from aiwf.core.util.access import build_network_access_info
from aiwf.core.util.network import find_free_port

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("xformers").setLevel(logging.ERROR)
logger = logging.getLogger("aiwf")


def _parse_cli() -> RuntimeFlags:
    parser = argparse.ArgumentParser(description="AIWF Studio")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--ckpt-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--listen", action="store_true")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--autolaunch", action="store_true")
    parser.add_argument("--api", action="store_true")
    parser.add_argument("--nowebui", action="store_true")
    parser.add_argument("--theme", choices=["dark", "light"], default="dark")
    parser.add_argument("--gradio-auth", type=str, default=None)
    parser.add_argument("--no-half", action="store_true")
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference even when a GPU is available (useful for testing)",
    )
    parser.add_argument("--medvram", action="store_true")
    parser.add_argument("--lowvram", action="store_true")
    parser.add_argument("--xformers", action="store_true", help="Use xformers memory-efficient attention")
    parser.add_argument(
        "--opt-sdp-attention",
        action="store_true",
        help="PyTorch scaled dot product attention (fast on RTX 30/40 series)",
    )
    parser.add_argument(
        "--opt-split-attention",
        action="store_true",
        help="Doggettx-style split attention (maps to SDP in diffusers backend)",
    )
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--skip-prepare-environment", action="store_true")
    parser.add_argument("--ckpt", type=Path, default=None, dest="default_checkpoint")
    args = parser.parse_args()

    return RuntimeFlags(
        data_dir=args.data_dir.resolve(),
        models_dir=args.models_dir.resolve() if args.models_dir else None,
        ckpt_dir=args.ckpt_dir.resolve() if args.ckpt_dir else None,
        output_dir=args.output_dir.resolve() if args.output_dir else None,
        port=args.port,
        listen=args.listen,
        share=args.share,
        autolaunch=args.autolaunch,
        api=args.api,
        nowebui=args.nowebui,
        theme=args.theme,
        gradio_auth=args.gradio_auth,
        no_half=args.no_half,
        cpu=args.cpu,
        medvram=args.medvram,
        lowvram=args.lowvram,
        xformers=args.xformers,
        opt_sdp_attention=args.opt_sdp_attention,
        opt_split_attention=args.opt_split_attention,
        skip_install=args.skip_install,
        skip_prepare_environment=args.skip_prepare_environment,
        default_checkpoint=args.default_checkpoint,
    )


def _auth_pairs(auth: str | None):
    if not auth:
        return None
    return [tuple(chunk.strip().split(":", 1)) for chunk in auth.split(",") if ":" in chunk]


def _resolve_flags() -> RuntimeFlags:
    cli_flags = _parse_cli()
    saved = load_launch_settings(launch_settings_path(cli_flags.data_dir))
    return merge_launch_settings(cli_flags, saved, explicit=explicit_cli_flags())


def run() -> None:
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    flags = _resolve_flags()
    ctx = build_context(flags)

    if flags.nowebui:
        import uvicorn
        from fastapi import FastAPI

        app = FastAPI(title="AIWF Studio API")
        from aiwf.api.v1.routes import build_router

        app.include_router(build_router(ctx))
        host = "0.0.0.0" if flags.listen else "127.0.0.1"
        uvicorn.run(app, host=host, port=flags.port)
        return

    server_name = "0.0.0.0" if flags.listen else "127.0.0.1"

    port = flags.port
    try:
        find_free_port(port, attempts=1)
    except OSError:
        port = find_free_port(flags.port + 1, attempts=32)
        logger.warning("Port %d is busy (likely old WebUI still running). Using %d instead.", flags.port, port)

    ctx.runtime_port = port

    from aiwf.web.app import create_web_ui

    demo, theme, css, js = create_web_ui(ctx)

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
    )
    logger.info("Starting AIWF Studio at http://%s:%d", "localhost" if server_name == "127.0.0.1" else server_name, port)
    access = build_network_access_info(listen=flags.listen, port=port)
    if access.recommended_phone_url:
        logger.info("Phone/tablet URL: %s", access.recommended_phone_url)
    elif not flags.listen:
        logger.info("Remote access: restart with --listen, then open Settings → Phone & tablet access")

    if flags.api:
        from aiwf.api.v1.routes import build_router

        app, _, _ = demo.launch(**launch_kwargs)
        app.include_router(build_router(ctx))
        logger.info("API mounted at /api/v1")
    else:
        demo.launch(**launch_kwargs)

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
