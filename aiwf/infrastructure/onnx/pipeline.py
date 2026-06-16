"""
aiwf/infrastructure/onnx/pipeline.py

ONNX txt2img pipeline — no diffusers, no transformers.

Runs a full Stable Diffusion txt2img pass using three ONNX models:
  text_encoder/model.onnx  — CLIP text encoder
  unet/model.onnx           — UNet noise predictor
  vae_decoder/model.onnx    — VAE latent → pixel decoder

And our own sampler implementations from aiwf.infrastructure.samplers.

Model layout
------------
The pipeline expects a directory like:

    sd_onnx/
        text_encoder/model.onnx
        unet/model.onnx
        vae_decoder/model.onnx
        tokenizer/          (optional — see note below)

Such a directory can be produced by running `optimum-cli export onnx` from
the Hugging Face Optimum library, or by converting manually.

Tokenizer note
--------------
This pipeline does not bundle its own CLIP tokenizer because the tokenizer
is pure Python and is NOT part of diffusers' heavy dependency tree — it's
in the `transformers` package.  If transformers is installed, it's used
automatically.  If not, the user must pass pre-tokenized inputs.

No diffusers code anywhere in this file.
"""
from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from aiwf.core.domain.generation import (
    GenerationMode,
    GenerationRequest,
    GenerationResult,
)
from aiwf.infrastructure.onnx.session import load_session, ProviderPreference
from aiwf.infrastructure.samplers.schedule import get_sigmas
from aiwf.infrastructure.samplers.dispatch import run_sampler

logger = logging.getLogger(__name__)

# VAE scale factor — latent space is 8× downsampled from pixel space in SD1/SDXL
_VAE_SCALE = 0.18215

ProgressCallback = Callable[[int, int, str, "Image.Image | None"], None]


# ---------------------------------------------------------------------------
# Tokenizer (lazy — uses transformers if available, else raises clearly)
# ---------------------------------------------------------------------------

def _tokenize(text: str, tokenizer_dir: Path, max_length: int = 77) -> np.ndarray:
    """Tokenize a string using the model folder's local CLIP tokenizer.

    Returns token id array of shape (1, max_length).
    """
    if not tokenizer_dir.is_dir():
        raise FileNotFoundError(
            "ONNX model folder is missing tokenizer assets. Expected a local "
            f"tokenizer directory at: {tokenizer_dir}"
        )

    try:
        from transformers import CLIPTokenizer
    except ImportError as exc:
        raise ImportError(
            "The ONNX pipeline requires `transformers` for tokenization:\n"
            "  pip install transformers\n"
            "(transformers is lightweight — no torch required for tokenizing)"
        ) from exc

    tokenizer = CLIPTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
    tokens = tokenizer(
        text,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="np",
    )
    return tokens.input_ids.astype(np.int64)  # (1, 77)


# ---------------------------------------------------------------------------
# Latent → PIL conversion
# ---------------------------------------------------------------------------

def _latent_to_pil(latent: np.ndarray, vae_decoder_session) -> Image.Image:
    """Decode a single latent tensor to a PIL image via the VAE ONNX model.

    latent: (1, 4, H/8, W/8) float32 numpy array
    """
    latent = (latent / _VAE_SCALE).astype(np.float32)
    # ONNX VAE decoder expects (batch, channels, height, width)
    input_name = vae_decoder_session.get_inputs()[0].name
    outputs = vae_decoder_session.run(None, {input_name: latent})
    # Output: (1, 3, H, W) in [-1, 1] range
    img_array = outputs[0][0]  # (3, H, W)
    img_array = (img_array.transpose(1, 2, 0) + 1.0) / 2.0  # HWC, [0,1]
    img_array = (img_array.clip(0, 1) * 255).astype(np.uint8)
    return Image.fromarray(img_array, mode="RGB")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class ONNXPipeline:
    """Full Stable Diffusion txt2img pipeline using ONNX Runtime.

    Zero diffusers dependencies.  Uses AIWF's own sampler implementations.

    Parameters
    ----------
    model_dir:
        Path to an ONNX model directory with subdirectories:
        ``text_encoder/``, ``unet/``, ``vae_decoder/``.
    provider:
        ORT execution provider preference: "cuda" | "directml" | "cpu" | "auto".
    device_id:
        GPU device index.
    """

    def __init__(
        self,
        model_dir: Path,
        provider: ProviderPreference = "auto",
        device_id: int = 0,
    ) -> None:
        self.model_dir = model_dir
        self._provider = provider
        self._device_id = device_id

        # Sessions loaded lazily on first generate call
        self._text_encoder_session = None
        self._unet_session = None
        self._vae_session = None

    def _load_sessions(self) -> None:
        if self._unet_session is not None:
            return
        logger.info("Loading ONNX sessions from %s", self.model_dir)
        self._text_encoder_session = load_session(
            self.model_dir / "text_encoder" / "model.onnx",
            preference=self._provider,
            device_id=self._device_id,
        )
        self._unet_session = load_session(
            self.model_dir / "unet" / "model.onnx",
            preference=self._provider,
            device_id=self._device_id,
        )
        self._vae_session = load_session(
            self.model_dir / "vae_decoder" / "model.onnx",
            preference=self._provider,
            device_id=self._device_id,
        )
        logger.info("ONNX sessions loaded")

    def _encode_text(self, text: str) -> np.ndarray:
        """Run CLIP text encoder.  Returns (1, seq_len, hidden_size) float32."""
        tokens = _tokenize(text, self.model_dir / "tokenizer")
        input_name = self._text_encoder_session.get_inputs()[0].name
        outputs = self._text_encoder_session.run(None, {input_name: tokens})
        return outputs[0]  # last_hidden_state: (1, 77, 768)

    def _unet_forward(
        self,
        latent: np.ndarray,
        timestep: int,
        encoder_hidden_states: np.ndarray,
    ) -> np.ndarray:
        """Run the UNet noise predictor.

        Returns noise prediction of the same shape as latent.
        """
        inputs = {
            "sample": latent.astype(np.float32),
            "timestep": np.array([timestep], dtype=np.int64),
            "encoder_hidden_states": encoder_hidden_states.astype(np.float32),
        }
        outputs = self._unet_session.run(None, inputs)
        return outputs[0]  # (1, 4, H/8, W/8)

    def generate(
        self,
        request: GenerationRequest,
        on_progress: ProgressCallback | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> GenerationResult:
        self._load_sessions()

        job_id = uuid.uuid4()
        t0 = time.monotonic()

        seed = request.seed if request.seed >= 0 else int(torch.randint(0, 2**31 - 1, (1,)).item())
        rng = torch.Generator().manual_seed(seed)

        # --- Text conditioning ---
        logger.info("Encoding prompt…")
        pos_embeds = self._encode_text(request.prompt)
        neg_embeds = self._encode_text(request.negative_prompt or "")
        # Classifier-free guidance requires concatenating negative + positive
        # along the batch dimension for a single batched UNet forward
        text_embeds = np.concatenate([neg_embeds, pos_embeds], axis=0)  # (2, 77, 768)

        # --- Initial latent noise ---
        h_lat = request.height // 8
        w_lat = request.width // 8
        latent_shape = (1, 4, h_lat, w_lat)
        latent_torch = torch.randn(latent_shape, generator=rng, dtype=torch.float32)

        # --- Sigma schedule ---
        sigmas = get_sigmas(
            schedule="karras" if "karras" in request.scheduler.lower() else "scaled_linear",
            num_steps=request.steps,
        )
        # Scale initial latent by σ_max
        latent_torch = latent_torch * sigmas[0]

        # --- Build denoiser callable for the sampler ---
        # The sigma → timestep lookup table (1000-step training schedule)
        from aiwf.infrastructure.samplers.schedule import get_sigmas as _gs, _scaled_linear_betas, _betas_to_sigmas
        all_sigmas = _betas_to_sigmas(_scaled_linear_betas(1000)).float()

        def denoiser(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            # Map sigma → discrete timestep index for the UNet
            from aiwf.infrastructure.samplers.schedule import sigma_to_timestep
            t = int(sigma_to_timestep(sigma.unsqueeze(0), all_sigmas)[0].item())

            # c_in scaling: input normalization (EDM formulation)
            c_in = 1.0 / (sigma**2 + 1.0).sqrt()
            x_in = (c_in * x).numpy()

            # Duplicate for CFG: (neg, pos)
            x_in_cfg = np.concatenate([x_in, x_in], axis=0)  # (2, 4, H, W)

            noise_pred = self._unet_forward(x_in_cfg, t, text_embeds)  # (2, 4, H, W)

            # CFG: noise_uncond + scale * (noise_cond - noise_uncond)
            noise_uncond, noise_cond = noise_pred[0:1], noise_pred[1:2]
            guided = noise_uncond + request.cfg_scale * (noise_cond - noise_uncond)
            guided_t = torch.from_numpy(guided)

            # Convert epsilon prediction → x0
            from aiwf.infrastructure.samplers.euler import epsilon_to_x0
            return epsilon_to_x0(x, guided_t, sigma)

        # --- Sampling loop ---
        logger.info("Sampling with %s, %d steps…", request.sampler, request.steps)
        x = latent_torch

        for step in run_sampler(request.sampler, denoiser, x, sigmas, generator=rng):
            if should_cancel and should_cancel():
                raise RuntimeError("Generation cancelled by user.")
            if on_progress:
                on_progress(step.step, step.total, "sampling…", None)
            x = step.x

        # --- VAE decode ---
        logger.info("Decoding latent…")
        latent_np = x.numpy()
        image = _latent_to_pil(latent_np, self._vae_session)

        elapsed = time.monotonic() - t0
        if on_progress:
            on_progress(request.steps, request.steps, "done", image)

        return GenerationResult(
            job_id=job_id,
            images=[image],
            seeds=[seed],
            infotexts=[
                f"ONNX • {request.prompt[:120]} • seed={seed} "
                f"• steps={request.steps} • cfg={request.cfg_scale}"
            ],
            mode=request.mode,
            elapsed_seconds=elapsed,
        )

    def unload(self) -> None:
        """Release all ONNX sessions."""
        self._text_encoder_session = None
        self._unet_session = None
        self._vae_session = None
        logger.info("ONNX sessions released")
