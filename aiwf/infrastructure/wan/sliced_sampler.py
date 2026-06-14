"""Temporal chunk coordinator for Wan 3D transformer denoise (16 GB VRAM path).

Slices ``hidden_states`` along the frame axis before each transformer forward so
attention memory scales with chunk size instead of full clip length.

Seam handling:
- Overlap blending: the `overlap` frames at each boundary are feather-blended
  between adjacent chunks using a linear ramp weight.
- Luminance normalization: before accumulating each chunk (after the first),
  the chunk's output is scaled so its magnitude in the overlap zone matches the
  previous chunk's prediction for those same frames. This eliminates the
  inter-chunk brightness discontinuities that cause the "darkening every X frames"
  artifact without touching content.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Callable

logger = logging.getLogger(__name__)


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def temporal_chunks_enabled() -> bool:
    """Default on — set AIWF_WAN_TEMPORAL_CHUNKS=0 to disable."""
    return _env_flag("AIWF_WAN_TEMPORAL_CHUNKS", default=True)


def default_chunk_size() -> int:
    return _env_int("AIWF_WAN_CHUNK_SIZE", 16)


def default_chunk_overlap() -> int:
    # Increased from 4 → 8: wider blend zone reduces visible seam area.
    return _env_int("AIWF_WAN_CHUNK_OVERLAP", 8)


def _tokens_per_frame(hidden_states, patch_size: tuple[int, int, int]) -> int:
    _p_t, p_h, p_w = patch_size
    _b, _c, _f, height, width = hidden_states.shape
    return max(1, (height // p_h) * (width // p_w))


def _slice_timestep_for_frames(
    timestep,
    *,
    frame_start: int,
    frame_end: int,
    num_frames: int,
    tokens_per_frame: int,
):
    if timestep.ndim != 2:
        return timestep
    seq_start = frame_start * tokens_per_frame
    seq_end = frame_end * tokens_per_frame
    expected = num_frames * tokens_per_frame
    if timestep.shape[1] == expected:
        return timestep[:, seq_start:seq_end]
    # Fallback: proportional slice when seq_len does not match geometry.
    ratio_start = frame_start / max(num_frames, 1)
    ratio_end = frame_end / max(num_frames, 1)
    seq_len = timestep.shape[1]
    seq_start = int(ratio_start * seq_len)
    seq_end = max(seq_start + 1, int(ratio_end * seq_len))
    return timestep[:, seq_start:seq_end]


def _seam_scale(prev_output: "torch.Tensor", curr_output: "torch.Tensor") -> float:  # type: ignore[name-defined]
    """Compute a scale factor to match curr_output's RMS to prev_output's.

    Both tensors are the chunk prediction for the same overlap frames.
    prev_output is the already-accumulated (normalized) prediction from the
    previous chunk; curr_output is the raw new chunk's prediction.

    We match RMS (root-mean-square) rather than mean because the noise
    prediction can be zero-centred — matching the mean would be meaningless.

    Returns a scalar clamped to [0.5, 2.0] to avoid extreme corrections.
    """
    import torch

    prev_rms = prev_output.float().pow(2).mean().sqrt()
    curr_rms = curr_output.float().pow(2).mean().sqrt()
    if curr_rms < 1e-6 or prev_rms < 1e-6:
        return 1.0
    scale = float((prev_rms / curr_rms).clamp(0.5, 2.0))
    return scale


class WanTemporalChunkCoordinator:
    def __init__(self, *, chunk_size: int = 16, overlap: int = 8) -> None:
        self.chunk_size = max(1, int(chunk_size))
        self.overlap = max(0, min(int(overlap), self.chunk_size - 1))

    def should_slice(self, hidden_states) -> bool:
        if hidden_states.ndim != 5:
            return False
        num_frames = int(hidden_states.shape[2])
        return num_frames > self.chunk_size

    def sliced_forward(
        self,
        orig_forward: Callable[..., Any],
        *,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_hidden_states_image=None,
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        patch_size: tuple[int, int, int] = (1, 2, 2),
    ):
        import torch

        batch, _channels, num_frames, height, width = hidden_states.shape
        # output_buffer is initialised lazily from the first chunk's actual output
        # shape because Wan I2V transformers take in_channels=36 (noisy latent +
        # image condition concatenated) but emit out_channels=16 (predicted noise).
        # Using zeros_like(hidden_states) would create a 36-channel buffer that
        # mismatches the 16-channel chunk_out and raises a shape error.
        output_buffer: "torch.Tensor | None" = None
        weight_buffer: "torch.Tensor | None" = None
        tokens_per_frame = _tokens_per_frame(hidden_states, patch_size)
        overlap = self.overlap
        chunk_size = self.chunk_size
        chunk_idx = 0

        step_start = 0
        while step_start < num_frames:
            step_end = min(step_start + chunk_size, num_frames)
            slice_len = step_end - step_start
            latent_slice = hidden_states[:, :, step_start:step_end, :, :]

            blend_mask = torch.ones(
                (batch, 1, slice_len, 1, 1),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            if step_start > 0 and overlap > 0:
                fade = min(overlap, slice_len)
                for i in range(fade):
                    blend_mask[:, :, i, :, :] = (i + 1) / fade
            if step_end < num_frames and overlap > 0:
                fade = min(overlap, slice_len)
                for i in range(fade):
                    blend_mask[:, :, slice_len - 1 - i, :, :] = (i + 1) / fade

            ts = _slice_timestep_for_frames(
                timestep,
                frame_start=step_start,
                frame_end=step_end,
                num_frames=num_frames,
                tokens_per_frame=tokens_per_frame,
            )

            chunk_out = orig_forward(
                hidden_states=latent_slice,
                timestep=ts,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_image=encoder_hidden_states_image,
                return_dict=False,
                attention_kwargs=attention_kwargs,
            )[0]

            # Lazy-init on first chunk using chunk_out's actual output shape.
            # Wan I2V: in_channels=36 (noisy + image cond) but out_channels=16
            # (predicted noise/velocity), so we can't infer output_channels from
            # hidden_states.shape[1].
            if output_buffer is None:
                out_channels = chunk_out.shape[1]
                output_buffer = torch.zeros(
                    (batch, out_channels, num_frames, height, width),
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
                weight_buffer = torch.zeros(
                    (batch, 1, num_frames, 1, 1),
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )

            # Seam normalization: scale each non-first chunk so its magnitude in
            # the overlap zone matches the previous chunk's already-accumulated
            # prediction for those frames. This eliminates inter-chunk brightness
            # discontinuities ("darkening every X frames") caused by each chunk's
            # attention operating on a different context window.
            if chunk_idx > 0 and overlap > 0:
                actual_overlap = min(overlap, slice_len)
                # Previous chunk's normalized prediction for the overlap zone
                prev_pred = output_buffer[:, :, step_start:step_start + actual_overlap, :, :] / \
                    weight_buffer[:, :, step_start:step_start + actual_overlap, :, :].clamp(min=1e-6)
                # Current chunk's prediction for its leading overlap frames
                curr_pred = chunk_out[:, :, :actual_overlap, :, :]
                scale = _seam_scale(prev_pred, curr_pred)
                if abs(scale - 1.0) > 0.01:  # only apply if there's a meaningful difference
                    chunk_out = chunk_out * scale
                    logger.debug(
                        "Chunk %d seam normalization: scale=%.4f (overlap=%d frames)",
                        chunk_idx, scale, actual_overlap,
                    )

            output_buffer[:, :, step_start:step_end, :, :] += chunk_out * blend_mask
            weight_buffer[:, :, step_start:step_end, :, :] += blend_mask

            chunk_idx += 1
            if step_end >= num_frames:
                break
            step_start += chunk_size - overlap

        merged = output_buffer / weight_buffer.clamp(min=1e-6)
        if not return_dict:
            return (merged,)
        try:
            from diffusers.models.modeling_outputs import Transformer2DModelOutput

            return Transformer2DModelOutput(sample=merged)
        except Exception:
            return (merged,)


def install_temporal_chunk_forward(
    transformer,
    *,
    name: str = "transformer",
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> bool:
    """Wrap ``WanTransformer3DModel.forward`` with temporal chunk blending."""
    if transformer is None or getattr(transformer, "_aiwf_temporal_chunks", False):
        return False
    if not temporal_chunks_enabled():
        return False

    chunk_size = chunk_size if chunk_size is not None else default_chunk_size()
    overlap = overlap if overlap is not None else default_chunk_overlap()
    coordinator = WanTemporalChunkCoordinator(chunk_size=chunk_size, overlap=overlap)
    orig_forward = transformer.forward
    patch_size = tuple(getattr(getattr(transformer, "config", None), "patch_size", (1, 2, 2)))

    def forward(
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_hidden_states_image=None,
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        if coordinator.should_slice(hidden_states):
            return coordinator.sliced_forward(
                orig_forward,
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_image=encoder_hidden_states_image,
                return_dict=return_dict,
                attention_kwargs=attention_kwargs,
                patch_size=patch_size,
            )
        return orig_forward(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_image=encoder_hidden_states_image,
            return_dict=return_dict,
            attention_kwargs=attention_kwargs,
            **kwargs,
        )

    transformer.forward = forward  # type: ignore[method-assign]
    transformer._aiwf_temporal_chunks = True
    transformer._aiwf_chunk_size = chunk_size
    transformer._aiwf_chunk_overlap = overlap
    logger.info(
        "Wan %s temporal chunk forward: chunk_size=%d overlap=%d",
        name,
        chunk_size,
        overlap,
    )
    return True


def describe_temporal_chunk_settings(num_frames: int) -> str:
    if not temporal_chunks_enabled():
        return "Temporal chunk denoise: disabled (AIWF_WAN_TEMPORAL_CHUNKS=0)."
    chunk = default_chunk_size()
    overlap = default_chunk_overlap()
    if num_frames <= chunk:
        return f"Temporal chunk denoise: enabled but skipped ({num_frames} frames <= chunk {chunk})."
    return (
        f"Temporal chunk denoise: {chunk} frames/chunk, overlap {overlap} "
        f"({num_frames} frames total)."
    )
