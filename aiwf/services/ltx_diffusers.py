from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

def _ltx_dtype():
    """Resolve the LTX compute dtype from settings env (AIWF_LTX_DTYPE).

    LTX-Video is distributed and calibrated in bfloat16; fp16 can overflow in
    the transformer and produce artifacts. Default to bf16 whenever the GPU
    supports it, fall back to fp16 otherwise.
    """
    import torch

    choice = os.environ.get("AIWF_LTX_DTYPE", "").strip().lower()
    if choice in {"fp16", "float16", "half"}:
        return torch.float16
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if choice in {"bf16", "bfloat16"}:
        return torch.bfloat16 if bf16_ok else torch.float16
    return torch.bfloat16 if bf16_ok else torch.float16


def _ltx_offload_mode() -> str:
    mode = os.environ.get("AIWF_LTX_CPU_OFFLOAD", "").strip().lower()
    return mode if mode in {"auto", "model", "none"} else "auto"


def _ltx_should_offload(checkpoint: Path) -> bool:
    mode = _ltx_offload_mode()
    if mode == "model":
        return True
    if mode == "none":
        return False
    # auto: keep small checkpoints (e.g. LTX 2B distilled) fully resident when
    # VRAM allows — skipping per-component transfers removes real overhead.
    try:
        import torch

        if not torch.cuda.is_available():
            return True
        total_gb = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1024**3
        ckpt_gb = checkpoint.stat().st_size / 1024**3
        # ~10 GB margin covers the T5XXL encoder, VAE, and decode activations.
        return (ckpt_gb + 10.0) > total_gb
    except Exception:
        return True


@dataclass(frozen=True)
class Ltx2BDiffusersResult:
    output_path: Path
    frame_count: int
    fps: int
    width: int
    height: int
    bytes: int
    cache_hit: bool = False


@dataclass(frozen=True)
class _Ltx2BCacheKey:
    checkpoint: str
    t5_weights: str
    tokenizer_id: str
    dtype: str = ""
    offload: str = ""


@dataclass
class _Ltx2BCacheEntry:
    key: _Ltx2BCacheKey
    pipe: Any


_CACHE_LOCK = RLock()
_PIPE_CACHE: _Ltx2BCacheEntry | None = None


def run_ltx2b_diffusers(
    *,
    checkpoint: Path,
    t5_weights: Path,
    tokenizer_id: str,
    output: Path,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    frames: int,
    fps: int,
    steps: int,
    seed: int,
    max_sequence_length: int = 128,
    guidance_scale: float = 1.0,
    use_cache: bool = True,
) -> Ltx2BDiffusersResult:
    import torch
    from diffusers.utils import export_to_video

    checkpoint = checkpoint.resolve()
    t5_weights = t5_weights.resolve()
    output = output.resolve()
    _require_file(checkpoint, "LTX 2B checkpoint")
    _require_file(t5_weights, "T5XXL text encoder weights")
    if frames % 8 != 1:
        raise ValueError("LTX frame count must satisfy 8*k+1; use 9 frames for an 8-frame smoke.")

    pipe, cache_hit = load_ltx2b_pipeline(
        checkpoint=checkpoint,
        t5_weights=t5_weights,
        tokenizer_id=tokenizer_id,
        use_cache=use_cache,
    )

    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.inference_mode():
        video_frames = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=frames,
            frame_rate=fps,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            max_sequence_length=max_sequence_length,
            generator=generator,
        ).frames[0]

    output.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(video_frames, str(output), fps=fps)
    if not output.is_file() or output.stat().st_size <= 0:
        raise RuntimeError(f"Diffusers export did not create a non-empty mp4: {output}")

    return Ltx2BDiffusersResult(
        output_path=output,
        frame_count=frames,
        fps=fps,
        width=width,
        height=height,
        bytes=output.stat().st_size,
        cache_hit=cache_hit,
    )


def load_ltx2b_pipeline(*, checkpoint: Path, t5_weights: Path, tokenizer_id: str, use_cache: bool = True):
    global _PIPE_CACHE

    import torch
    from diffusers import LTXPipeline
    from transformers import AutoTokenizer

    dtype = _ltx_dtype()
    offload = _ltx_should_offload(checkpoint)
    key = _Ltx2BCacheKey(
        checkpoint=str(checkpoint.resolve()),
        t5_weights=str(t5_weights.resolve()),
        tokenizer_id=str(tokenizer_id),
        dtype=str(dtype).replace("torch.", ""),
        offload="model" if offload else "none",
    )
    with _CACHE_LOCK:
        if use_cache and _PIPE_CACHE is not None and _PIPE_CACHE.key == key:
            return _PIPE_CACHE.pipe, True
        if use_cache and _PIPE_CACHE is not None and _PIPE_CACHE.key != key:
            unload_ltx2b_diffusers_cache()

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, local_files_only=True)
        text_encoder = load_t5_encoder(t5_weights, dtype=dtype)
        pipe = LTXPipeline.from_single_file(
            str(checkpoint),
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            torch_dtype=dtype,
            local_files_only=True,
        )
        if offload:
            pipe.enable_model_cpu_offload()
        elif torch.cuda.is_available():
            pipe.to("cuda")
        vae = getattr(pipe, "vae", None)
        if hasattr(vae, "enable_tiling"):
            vae.enable_tiling()
        if use_cache:
            _PIPE_CACHE = _Ltx2BCacheEntry(key=key, pipe=pipe)
        return pipe, False


def unload_ltx2b_diffusers_cache() -> bool:
    global _PIPE_CACHE
    with _CACHE_LOCK:
        had_cache = _PIPE_CACHE is not None
        _PIPE_CACHE = None
    if had_cache:
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        except Exception:
            pass
    return had_cache


def load_t5_encoder(path: Path, *, dtype=None):
    import torch
    from safetensors.torch import load_file
    from transformers import T5Config, T5EncoderModel

    if dtype is None:
        dtype = _ltx_dtype()

    config = T5Config(
        vocab_size=32128,
        d_model=4096,
        d_kv=64,
        d_ff=10240,
        num_layers=24,
        num_decoder_layers=24,
        num_heads=64,
        relative_attention_num_buckets=32,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        feed_forward_proj="gated-gelu",
        is_encoder_decoder=True,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
        tie_word_embeddings=False,
    )
    model = T5EncoderModel(config).to(dtype=dtype)
    state = load_file(str(path), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"T5XXL weights did not match T5EncoderModel: missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    model.eval()
    return model


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
