from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aiwf.infrastructure.distributed.sana_split import (
    SanaEncoderClient,
    SanaSplitError,
    sana_model_fingerprint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Sana Sprint with a remote prompt encoder.")
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--encoder-url", default="http://127.0.0.1:18794")
    parser.add_argument("--token-env", default="AIWF_SANA_ENCODER_TOKEN")
    parser.add_argument("--prompt", default="A small copper robot reading beside a warm workshop window.")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--local-encoder", action="store_true")
    parser.add_argument("--output")
    return parser.parse_args()


def load_denoiser(model_root: Path, *, device: torch.device, dtype: torch.dtype):
    from diffusers import AutoencoderDC, SCMScheduler, SanaSprintPipeline, SanaTransformer2DModel

    common = {"local_files_only": True, "torch_dtype": dtype}
    try:
        transformer = SanaTransformer2DModel.from_pretrained(
            str(model_root / "transformer"),
            **common,
        )
        vae = AutoencoderDC.from_pretrained(str(model_root / "vae"), **common)
    except TypeError:
        common["dtype"] = common.pop("torch_dtype")
        transformer = SanaTransformer2DModel.from_pretrained(str(model_root / "transformer"), **common)
        vae = AutoencoderDC.from_pretrained(str(model_root / "vae"), **common)
    scheduler = SCMScheduler.from_pretrained(str(model_root / "scheduler"), local_files_only=True)
    transformer.to(device)
    vae.to(device)
    pipe = SanaSprintPipeline(
        tokenizer=None,
        text_encoder=None,
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
    )
    pipe.set_progress_bar_config(disable=True)
    return pipe


def start_local_encoder(
    model_root: Path,
    *,
    token: str,
    device: str,
    dtype: str,
):
    import uvicorn

    from aiwf.infrastructure.distributed.sana_encoder_server import (
        SanaSprintEncoderRuntime,
        create_sana_encoder_app,
    )

    runtime = SanaSprintEncoderRuntime.load(
        model_root,
        device=device,
        dtype=dtype,
    )
    app = create_sana_encoder_app(runtime, token=token)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, access_log=False, log_level="warning")
    )
    thread = threading.Thread(target=server.run, name="sana-loopback-encoder", daemon=True)
    thread.start()
    client = SanaEncoderClient(f"http://127.0.0.1:{port}", token)
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        try:
            client.health()
            return client, server, thread
        except SanaSplitError:
            time.sleep(0.1)
    server.should_exit = True
    thread.join(timeout=5.0)
    raise RuntimeError("The local Sana encoder did not become ready.")


def main() -> int:
    args = parse_args()
    token = (os.environ.get(args.token_env) or "").strip()
    model_root = Path(args.model_root).expanduser().resolve()
    local_server = None
    local_thread = None
    if args.local_encoder:
        client, local_server, local_thread = start_local_encoder(
            model_root,
            token=token,
            device=args.device,
            dtype=args.dtype,
        )
        topology = "single-host-loopback-validation"
    else:
        client = SanaEncoderClient(args.encoder_url, token)
        topology = "two-host-ssh-tunnel"
    local_fingerprint = sana_model_fingerprint(model_root)
    try:
        health = client.health()
        if health.get("model_fingerprint") != local_fingerprint:
            raise SystemExit("Laptop encoder and desktop denoiser model fingerprints do not match.")

        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise SystemExit("The Sana split smoke requires a CUDA desktop GPU.")
        dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

        encode_started = time.perf_counter()
        encoding = client.encode(args.prompt)
        encode_seconds = time.perf_counter() - encode_started

        load_started = time.perf_counter()
        pipe = load_denoiser(model_root, device=device, dtype=dtype)
        load_seconds = time.perf_counter() - load_started
        prompt_embeds = encoding.prompt_embeds.to(device=device, dtype=pipe.transformer.dtype)
        prompt_attention_mask = encoding.prompt_attention_mask.to(device=device)

        generation_started = time.perf_counter()
        with torch.inference_mode():
            image = pipe(
                prompt=None,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                num_inference_steps=args.steps,
                guidance_scale=4.5,
                num_images_per_prompt=1,
                generator=torch.Generator(device=device).manual_seed(args.seed),
                width=args.width,
                height=args.height,
                output_type="pil",
                clean_caption=False,
                use_resolution_binning=True,
            ).images[0]
        generation_seconds = time.perf_counter() - generation_started

        output_bytes = io.BytesIO()
        image.save(output_bytes, format="PNG")
        png = output_bytes.getvalue()
        output_path = None
        if args.output:
            target = Path(args.output).expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(png)
            output_path = str(target)

        index = device.index or 0
        print(
            json.dumps(
                {
                    "status": "passed",
                    "topology": topology,
                    "protocol_version": health.get("protocol_version"),
                    "encoder_device": health.get("device"),
                    "encoder_disk_offload": health.get("disk_offload"),
                    "model_fingerprint": local_fingerprint,
                    "encode_seconds": round(encode_seconds, 3),
                    "desktop_load_seconds": round(load_seconds, 3),
                    "generation_seconds": round(generation_seconds, 3),
                    "desktop_peak_vram_mib": round(torch.cuda.max_memory_allocated(index) / (1024.0**2), 1),
                    "width": image.width,
                    "height": image.height,
                    "png_bytes": len(png),
                    "png_sha256": hashlib.sha256(png).hexdigest(),
                    "output_path": output_path,
                },
                indent=2,
            )
        )
        return 0
    finally:
        if local_server is not None:
            local_server.should_exit = True
        if local_thread is not None:
            local_thread.join(timeout=10.0)


if __name__ == "__main__":
    raise SystemExit(main())
