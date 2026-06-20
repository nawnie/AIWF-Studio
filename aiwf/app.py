from __future__ import annotations

import argparse
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from aiwf.runtime.bootstrap_env import apply_from_argv

_PERF_FLAGS, _RUNTIME_ENV = apply_from_argv(sys.argv[1:])

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

logger = logging.getLogger("aiwf")


def _configure_logging(data_dir: Path) -> None:
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(formatter)
    root.addHandler(console)

    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            data_dir / "aiwf.log",
            maxBytes=2 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError:
        root.warning("Could not create aiwf.log; continuing with console-only logging.")

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("xformers").setLevel(logging.ERROR)


def _startup_message(message: str) -> None:
    print(f"[AIWF] {message}", flush=True)


def _friendly_device_name(description: str) -> str:
    if description.startswith("CUDA ("):
        return description.removeprefix("CUDA (").rstrip(")").split(",", 1)[0].strip()
    if description.startswith("DirectML ("):
        return description.removeprefix("DirectML (").rstrip(")").strip()
    if description.startswith("Apple MPS"):
        return "Apple Silicon GPU"
    if description.startswith("CPU"):
        return "CPU mode"
    return description


def _friendly_library_message(checkpoint_count: int, lora_count: int) -> str:
    if checkpoint_count <= 0:
        return "No base models were found yet. Add one in Models or import another library in Settings."
    if lora_count <= 0:
        return f"Library ready with {checkpoint_count} base model{'s' if checkpoint_count != 1 else ''}."
    return (
        f"Library ready with {checkpoint_count} base model{'s' if checkpoint_count != 1 else ''} "
        f"and {lora_count} LoRA{'s' if lora_count != 1 else ''}."
    )


def _parse_cli() -> RuntimeFlags:
    parser = argparse.ArgumentParser(description="AIWF Studio")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--ckpt-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--extra-model-dir", type=Path, action="append", default=None)
    parser.add_argument("--extra-ckpt-dir", type=Path, action="append", default=None)
    parser.add_argument("--nvidia-vfx-sdk-root", type=Path, default=None)
    parser.add_argument("--vsr-video-effects-app", type=Path, default=None)
    parser.add_argument("--vsr-upscale-app", type=Path, default=None)
    parser.add_argument("--videofx-denoise-app", type=Path, default=None)
    parser.add_argument("--videofx-aigs-app", type=Path, default=None)
    parser.add_argument("--videofx-relight-app", type=Path, default=None)
    parser.add_argument("--vsr-model-dir", type=Path, default=None)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--listen", action="store_true")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--autolaunch", action="store_true")
    parser.add_argument("--api", action="store_true")
    parser.add_argument("--nowebui", action="store_true")
    parser.add_argument("--theme", choices=["dark", "light"], default="dark")
    parser.add_argument("--gradio-auth", type=str, default=None)
    parser.add_argument("--api-cors-origins", type=str, default="")
    parser.add_argument("--api-rate-limit-per-minute", type=int, default=0)
    parser.add_argument("--allow-private-download-urls", action="store_true")
    parser.add_argument(
        "--genlog",
        action="store_true",
        help="Append local generation speed/settings entries to outputs/genlog/generation-log.jsonl",
    )
    parser.add_argument("--no-half", action="store_true")
    parser.add_argument(
        "--directml",
        action="store_true",
        help="Use DirectML for AMD/Intel GPUs on Windows (requires the torch-directml package)",
    )
    parser.add_argument(
        "--fp8",
        action="store_true",
        help="Store UNet weights in FP8 (halves UNet VRAM; tiny quality cost; great for SDXL on 8GB)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference even when a GPU is available (useful for testing)",
    )
    parser.add_argument(
        "--inference-backend",
        choices=["diffusers", "onnx"],
        default="diffusers",
        help="Studio image pipeline family: diffusers or onnx",
    )
    parser.add_argument(
        "--onnx-provider",
        choices=["auto", "cuda", "directml", "cpu"],
        default="auto",
        help="ONNX Runtime execution provider when --inference-backend=onnx",
    )
    parser.add_argument("--medvram", action="store_true")
    parser.add_argument("--lowvram", action="store_true")
    parser.add_argument(
        "--attention-backend",
        choices=["sage_sdpa", "sdpa", "xformers", "none"],
        default=None,
        help="Image attention backend: SageAttention with SDPA fallback, SDPA, xFormers, or none",
    )
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
    parser.add_argument(
        "--no-async-offload",
        action="store_true",
        help="Disable background Wan low-transformer preload during high-stage denoise",
    )
    parser.add_argument(
        "--no-pinned-memory",
        action="store_true",
        help="Disable page-locked CPU cache for Wan high/low PCIe swaps",
    )
    parser.add_argument(
        "--no-cuda-malloc",
        action="store_true",
        help="Disable cudaMallocAsync allocator (PYTORCH_CUDA_ALLOC_CONF)",
    )
    parser.add_argument("--cuda-graphs", action="store_true", help="Enable experimental CUDA Graph replay")
    parser.add_argument("--torchao", action="store_true", help="Enable experimental TorchAO int8 quantization")
    parser.add_argument("--fp8-quant", action="store_true", help="Enable experimental TorchAO FP8 quantization")
    parser.add_argument("--torch-compile", action="store_true", help="Enable experimental torch.compile")
    parser.add_argument("--channels-last", action="store_true", help="Use channels-last memory layout where supported")
    parser.add_argument("--nvenc", action="store_true", help="Prefer NVIDIA NVENC video encoding")
    parser.add_argument("--hevc", action="store_true", help="Prefer HEVC/H.265 video encoding")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--skip-prepare-environment", action="store_true")
    parser.add_argument("--ckpt", type=Path, default=None, dest="default_checkpoint")
    args = parser.parse_args()

    return RuntimeFlags(
        data_dir=args.data_dir.resolve(),
        models_dir=args.models_dir.resolve() if args.models_dir else None,
        ckpt_dir=args.ckpt_dir.resolve() if args.ckpt_dir else None,
        output_dir=args.output_dir.resolve() if args.output_dir else None,
        extra_model_dirs=[path.resolve() for path in (args.extra_model_dir or [])],
        extra_ckpt_dirs=[path.resolve() for path in (args.extra_ckpt_dir or [])],
        nvidia_vfx_sdk_root=args.nvidia_vfx_sdk_root.resolve() if args.nvidia_vfx_sdk_root else None,
        vsr_video_effects_app=args.vsr_video_effects_app.resolve() if args.vsr_video_effects_app else None,
        vsr_upscale_app=args.vsr_upscale_app.resolve() if args.vsr_upscale_app else None,
        videofx_denoise_app=args.videofx_denoise_app.resolve() if args.videofx_denoise_app else None,
        videofx_aigs_app=args.videofx_aigs_app.resolve() if args.videofx_aigs_app else None,
        videofx_relight_app=args.videofx_relight_app.resolve() if args.videofx_relight_app else None,
        vsr_model_dir=args.vsr_model_dir.resolve() if args.vsr_model_dir else None,
        port=args.port,
        listen=args.listen,
        share=args.share,
        autolaunch=args.autolaunch,
        api=args.api,
        nowebui=args.nowebui,
        theme=args.theme,
        gradio_auth=args.gradio_auth,
        api_cors_origins=args.api_cors_origins,
        api_rate_limit_per_minute=args.api_rate_limit_per_minute,
        block_private_download_urls=not args.allow_private_download_urls,
        genlog=args.genlog,
        no_half=args.no_half,
        fp8=args.fp8,
        directml=args.directml,
        cpu=args.cpu,
        inference_backend=args.inference_backend,
        onnx_provider=args.onnx_provider,
        medvram=args.medvram,
        lowvram=args.lowvram,
        attention_backend=(
            args.attention_backend
            or ("xformers" if args.xformers else "sdpa" if args.opt_sdp_attention or args.opt_split_attention else "sage_sdpa")
        ),
        xformers=args.xformers,
        opt_sdp_attention=args.opt_sdp_attention,
        opt_split_attention=args.opt_split_attention,
        async_offload=not args.no_async_offload,
        pinned_memory=not args.no_pinned_memory,
        cuda_malloc=not args.no_cuda_malloc,
        cuda_graphs=args.cuda_graphs,
        torchao=args.torchao,
        fp8_quant=args.fp8_quant,
        torch_compile=args.torch_compile,
        channels_last=args.channels_last,
        nvenc=args.nvenc,
        hevc=args.hevc,
        skip_install=args.skip_install,
        skip_prepare_environment=args.skip_prepare_environment,
        default_checkpoint=args.default_checkpoint,
    )


def _mount_gradio_extensions(app, ctx) -> None:
    """Routes that should exist whenever the Gradio UI is served."""
    from aiwf.api.v1.client_log import build_client_log_router

    app.include_router(build_client_log_router(ctx), prefix="/api/v1")


def _configure_api_security(app, flags: RuntimeFlags) -> None:
    from fastapi.middleware.cors import CORSMiddleware

    from aiwf.api.security import ApiRateLimitMiddleware, api_security_warnings, parse_cors_origins

    origins = parse_cors_origins(flags.api_cors_origins)
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    if flags.api_rate_limit_per_minute:
        app.add_middleware(
            ApiRateLimitMiddleware,
            requests_per_minute=flags.api_rate_limit_per_minute,
        )
    for warning in api_security_warnings(
        listen=flags.listen,
        gradio_auth=flags.gradio_auth,
        api=flags.api,
        nowebui=flags.nowebui,
    ):
        logger.warning("Security: %s", warning)


def _api_security_middleware(flags: RuntimeFlags):
    from fastapi.middleware.cors import CORSMiddleware
    from starlette.middleware import Middleware

    from aiwf.api.security import ApiRateLimitMiddleware, parse_cors_origins

    middleware = []
    origins = parse_cors_origins(flags.api_cors_origins)
    if origins:
        middleware.append(
            Middleware(
                CORSMiddleware,
                allow_origins=origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        )
    if flags.api_rate_limit_per_minute:
        middleware.append(
            Middleware(
                ApiRateLimitMiddleware,
                requests_per_minute=flags.api_rate_limit_per_minute,
            )
        )
    return middleware


def _log_api_security_warnings(flags: RuntimeFlags) -> None:
    from aiwf.api.security import api_security_warnings

    for warning in api_security_warnings(
        listen=flags.listen,
        gradio_auth=flags.gradio_auth,
        api=flags.api,
        nowebui=flags.nowebui,
    ):
        logger.warning("Security: %s", warning)


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
    _configure_logging(flags.data_dir)
    _startup_message("Starting AIWF Studio...")
    _startup_message("Checking your hardware and loading tools...")
    if flags.cuda_malloc:
        _startup_message("CUDA allocator: cudaMallocAsync (Comfy --cuda-malloc parity).")
    if flags.genlog:
        _startup_message(f"Generation log enabled: {flags.resolved_output_dir() / 'genlog' / 'generation-log.jsonl'}")
    wan_perf = []
    if flags.async_offload:
        wan_perf.append("async-offload")
    if flags.pinned_memory:
        wan_perf.append("pinned-memory")
    if wan_perf:
        _startup_message(f"Wan throughput flags: {', '.join(wan_perf)}.")
    ctx = build_context(flags)

    if flags.nowebui:
        import uvicorn
        from fastapi import FastAPI

        app = FastAPI(title="AIWF Studio API")
        from aiwf.api.v1.routes import build_router

        _configure_api_security(app, flags)
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
        logger.warning("Port %d is already in use. AIWF Studio will use %d instead.", flags.port, port)

    ctx.runtime_port = port

    from aiwf.web.app import create_web_ui

    _startup_message(f"Using {_friendly_device_name(ctx.generation.backend.devices.describe())}.")
    _startup_message("Scanning your model library...")
    checkpoint_count = len(ctx.generation.list_checkpoints())
    lora_count = len(ctx.generation.list_loras())
    _startup_message(_friendly_library_message(checkpoint_count, lora_count))
    _startup_message("Building the workspace...")
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

    _startup_message("AIWF Studio is ready.")
    _startup_message(f"Open in your browser: {local_url}")
    if access.recommended_phone_url:
        _startup_message(f"Phone and tablet access: {access.recommended_phone_url}")
    elif not flags.listen:
        _startup_message("Phone and tablet access is off for now. Turn it on later in Settings -> Remote access.")
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
