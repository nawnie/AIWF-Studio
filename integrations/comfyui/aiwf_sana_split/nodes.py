from __future__ import annotations

import base64
import gc
import os
import secrets
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


AIWF_ROOT = Path(os.environ.get("AIWF_STUDIO_ROOT", r"F:\AIWF_Studio")).resolve()
MODEL_ROOT = Path(
    os.environ.get(
        "AIWF_SANA_MODEL_ROOT",
        str(AIWF_ROOT / "models" / "sana" / "Diffusers" / "Sana_Sprint_0.6B_1024px_diffusers"),
    )
).resolve()
LAPTOP_HOST = os.environ.get("AIWF_SANA_LAPTOP_HOST", "4070-laptop-lan")
LAPTOP_ROOT = os.environ.get(
    "AIWF_SANA_LAPTOP_ROOT",
    r"C:\Users\shawn\AppData\Local\AIWF\SanaEncoder",
)

if str(AIWF_ROOT) not in sys.path:
    sys.path.insert(0, str(AIWF_ROOT))

from aiwf.infrastructure.distributed.sana_split import (  # noqa: E402
    SanaEncoderClient,
    SanaSplitError,
    sana_model_fingerprint,
)


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _remote_encoder_command(token: str, remote_receipt: str) -> str:
    return f'''$ErrorActionPreference = "Stop"
$env:AIWF_SANA_ENCODER_TOKEN = "{token}"
$root = "{LAPTOP_ROOT}"
$python = Join-Path $root ".venv310\\Scripts\\python.exe"
$guard = Join-Path $root "tools\\run_with_vram_guard.py"
& $python $guard --gpu-index 0 --receipt "{remote_receipt}" -- $python (Join-Path $root "scripts\\start_sana_encoder.py") --model-root (Join-Path $root "models\\sana-sprint") --port 8794 --device cuda:0 --dtype bfloat16 --max-encodes 1
exit $LASTEXITCODE
'''


def _start_encoder(token: str, tunnel_port: int) -> subprocess.Popen[bytes]:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    receipt = f"{LAPTOP_ROOT}\\receipts\\comfy-sana-encoder-{stamp}.json"
    encoded = base64.b64encode(_remote_encoder_command(token, receipt).encode("utf-16-le")).decode("ascii")
    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    return subprocess.Popen(
        [
            "ssh",
            "-o",
            "ExitOnForwardFailure=yes",
            "-L",
            f"{tunnel_port}:127.0.0.1:8794",
            LAPTOP_HOST,
            "powershell.exe",
            "-NoProfile",
            "-EncodedCommand",
            encoded,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        creationflags=creation_flags,
    )


def _wait_for_encoder(client: SanaEncoderClient, process: subprocess.Popen[bytes]) -> dict[str, object]:
    deadline = time.monotonic() + 240.0
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stderr = (process.stderr.read() if process.stderr else b"").decode("utf-8", errors="replace")
            raise RuntimeError(f"Laptop encoder exited before becoming ready: {stderr[-4000:]}")
        try:
            health = client.health()
            if health.get("ready"):
                return health
        except SanaSplitError as exc:
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"Laptop encoder did not become ready within 240 seconds: {last_error}")


def _stop_owned_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)


def _load_desktop_pipeline(device: torch.device, dtype: torch.dtype):
    from diffusers import AutoencoderDC, SCMScheduler, SanaSprintPipeline, SanaTransformer2DModel

    common = {"local_files_only": True, "torch_dtype": dtype}
    try:
        transformer = SanaTransformer2DModel.from_pretrained(str(MODEL_ROOT / "transformer"), **common)
        vae = AutoencoderDC.from_pretrained(str(MODEL_ROOT / "vae"), **common)
    except TypeError:
        common["dtype"] = common.pop("torch_dtype")
        transformer = SanaTransformer2DModel.from_pretrained(str(MODEL_ROOT / "transformer"), **common)
        vae = AutoencoderDC.from_pretrained(str(MODEL_ROOT / "vae"), **common)
    scheduler = SCMScheduler.from_pretrained(str(MODEL_ROOT / "scheduler"), local_files_only=True)
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


class AIWFSanaSplitGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "default": "A small copper robot reading beside a warm workshop window.",
                        "multiline": True,
                    },
                ),
                "steps": ("INT", {"default": 2, "min": 1, "max": 20}),
                "width": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 32}),
                "height": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 32}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "AIWF/Distributed"
    OUTPUT_NODE = True

    def generate(self, prompt: str, steps: int, width: int, height: int, seed: int):
        if not prompt.strip():
            raise ValueError("Prompt cannot be blank.")
        if not MODEL_ROOT.is_dir():
            raise RuntimeError(f"The Sana model root is missing: {MODEL_ROOT}")
        if not torch.cuda.is_available():
            raise RuntimeError("The AIWF Sana split node requires the desktop CUDA GPU.")

        try:
            import comfy.model_management as model_management

            model_management.unload_all_models()
            model_management.soft_empty_cache()
        except ImportError:
            pass

        device = torch.device("cuda:0")
        free_bytes, _ = torch.cuda.mem_get_info(device)
        if free_bytes < 6 * 1024**3:
            raise RuntimeError(
                f"Refusing the Sana run with only {free_bytes / 1024**3:.2f} GiB free desktop VRAM."
            )

        token = secrets.token_hex(32)
        tunnel_port = _free_loopback_port()
        process = _start_encoder(token, tunnel_port)
        pipe = None
        try:
            client = SanaEncoderClient(f"http://127.0.0.1:{tunnel_port}", token)
            health = _wait_for_encoder(client, process)
            local_fingerprint = sana_model_fingerprint(MODEL_ROOT)
            if health.get("model_fingerprint") != local_fingerprint:
                raise RuntimeError("Laptop encoder and desktop denoiser model fingerprints do not match.")
            encoding = client.encode(prompt.strip())
            process.wait(timeout=60.0)
            if process.returncode != 0:
                stderr = (process.stderr.read() if process.stderr else b"").decode("utf-8", errors="replace")
                raise RuntimeError(f"Laptop encoder guard exited {process.returncode}: {stderr[-4000:]}")

            dtype = torch.bfloat16
            pipe = _load_desktop_pipeline(device, dtype)
            prompt_embeds = encoding.prompt_embeds.to(device=device, dtype=pipe.transformer.dtype)
            prompt_attention_mask = encoding.prompt_attention_mask.to(device=device)
            with torch.inference_mode():
                image = pipe(
                    prompt=None,
                    prompt_embeds=prompt_embeds,
                    prompt_attention_mask=prompt_attention_mask,
                    num_inference_steps=int(steps),
                    guidance_scale=4.5,
                    num_images_per_prompt=1,
                    generator=torch.Generator(device=device).manual_seed(int(seed)),
                    width=int(width),
                    height=int(height),
                    output_type="pil",
                    clean_caption=False,
                    use_resolution_binning=True,
                ).images[0]
            pixels = np.asarray(image, dtype=np.float32) / 255.0
            return (torch.from_numpy(pixels).unsqueeze(0),)
        finally:
            _stop_owned_process(process)
            if pipe is not None:
                del pipe
            gc.collect()
            torch.cuda.empty_cache()


NODE_CLASS_MAPPINGS = {"AIWFSanaSplitGenerate": AIWFSanaSplitGenerate}
NODE_DISPLAY_NAME_MAPPINGS = {"AIWFSanaSplitGenerate": "AIWF Sana Split Generate"}
