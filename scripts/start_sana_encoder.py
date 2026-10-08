from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the loopback-only Sana Sprint prompt encoder.")
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--port", type=int, default=8794)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--min-free-ram-gib", type=float, default=10.0)
    parser.add_argument("--token-env", default="AIWF_SANA_ENCODER_TOKEN")
    parser.add_argument("--max-encodes", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = (os.environ.get(args.token_env) or "").strip()
    if len(token) < 24:
        raise SystemExit(f"Set {args.token_env} to a random token containing at least 24 characters.")

    from aiwf.infrastructure.distributed.sana_encoder_server import (
        SanaSprintEncoderRuntime,
        create_sana_encoder_app,
    )

    runtime = SanaSprintEncoderRuntime.load(
        args.model_root,
        device=args.device,
        dtype=args.dtype,
        min_free_ram_gib=args.min_free_ram_gib,
    )
    import uvicorn

    shutdown_event = threading.Event()
    encode_count = 0
    encode_lock = threading.Lock()

    def on_encode() -> None:
        nonlocal encode_count
        if args.max_encodes <= 0:
            return
        with encode_lock:
            encode_count += 1
            if encode_count >= args.max_encodes:
                shutdown_event.set()

    app = create_sana_encoder_app(runtime, token=token, on_encode=on_encode)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=args.port, access_log=False)
    )

    def stop_after_limit() -> None:
        shutdown_event.wait()
        server.should_exit = True

    monitor = None
    if args.max_encodes > 0:
        monitor = threading.Thread(target=stop_after_limit, name="sana-encode-limit", daemon=True)
        monitor.start()
    server.run()
    if monitor is not None:
        shutdown_event.set()
        monitor.join(timeout=5.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
