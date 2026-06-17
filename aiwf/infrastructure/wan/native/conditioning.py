"""Clean-room Wan image-conditioning helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WanConditioningBundle:
    latent: Any
    concat_latent_image: Any | None
    concat_mask: Any | None
    clip_vision_output: Any | None = None

    @property
    def latent_shape(self) -> tuple[int, ...]:
        return tuple(int(v) for v in getattr(self.latent, "shape", ()))


def wan_latent_frame_count(frames: int) -> int:
    return ((max(1, int(frames)) - 1) // 4) + 1


def wan_latent_shape(*, width: int, height: int, frames: int, batch_size: int = 1) -> tuple[int, int, int, int, int]:
    return (
        max(1, int(batch_size)),
        16,
        wan_latent_frame_count(frames),
        max(1, int(height) // 8),
        max(1, int(width) // 8),
    )


def _resize_image(image: Any, *, width: int, height: int) -> Any:
    from PIL import Image

    if image.mode != "RGB":
        image = image.convert("RGB")
    return image.resize((int(width), int(height)), Image.Resampling.LANCZOS)


def _image_to_tensor(image: Any, *, dtype: Any, device: Any) -> Any:
    import numpy as np
    import torch

    arr = np.asarray(image).astype("float32") / 127.5 - 1.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=dtype)


def _extract_vae_latents(encoded: Any) -> Any:
    latent_dist = getattr(encoded, "latent_dist", None)
    if latent_dist is not None:
        if hasattr(latent_dist, "mode"):
            return latent_dist.mode()
        if hasattr(latent_dist, "sample"):
            return latent_dist.sample()
    sample = getattr(encoded, "sample", None)
    if sample is not None:
        return sample
    if isinstance(encoded, (tuple, list)) and encoded:
        return encoded[0]
    return encoded


def _encode_image_latent(vae: Any, image: Any, *, width: int, height: int, latent_frames: int, dtype: Any, device: Any) -> Any:
    import torch
    import torch.nn.functional as F

    if vae is None:
        return torch.zeros((1, 16, latent_frames, height // 8, width // 8), device=device, dtype=dtype)

    image_tensor = _image_to_tensor(image, dtype=dtype, device=device)
    with torch.no_grad():
        encoded = vae.encode(image_tensor)
    latents = _extract_vae_latents(encoded)
    if latents.ndim == 4:
        latents = latents.unsqueeze(2)
    if latents.shape[1] != 16:
        latents = latents[:, :16, ...] if latents.shape[1] > 16 else F.pad(latents, (0, 0, 0, 0, 0, 0, 0, 16 - latents.shape[1]))
    if latents.shape[2] == 1 and latent_frames > 1:
        pad = torch.zeros(
            (latents.shape[0], latents.shape[1], latent_frames - 1, latents.shape[3], latents.shape[4]),
            device=latents.device,
            dtype=latents.dtype,
        )
        latents = torch.cat([latents, pad], dim=2)
    return latents[:, :, :latent_frames, :, :].to(device=device, dtype=dtype)


def prepare_wan_i2v_latents(
    image: Any,
    vae: Any,
    *,
    width: int,
    height: int,
    frames: int,
    batch_size: int = 1,
    dtype: Any | None = None,
    device: Any | None = None,
) -> WanConditioningBundle:
    """Prepare Wan I2V latent geometry and explicit image conditioning tensors."""
    import torch

    dtype = dtype or torch.bfloat16
    device = device or torch.device("cpu")
    latent_shape = wan_latent_shape(width=width, height=height, frames=frames, batch_size=batch_size)
    latent = torch.zeros(latent_shape, device=device, dtype=dtype)

    if image is None:
        return WanConditioningBundle(latent=latent, concat_latent_image=None, concat_mask=None)

    resized = _resize_image(image, width=width, height=height)
    concat_latent_image = _encode_image_latent(
        vae,
        resized,
        width=width,
        height=height,
        latent_frames=latent_shape[2],
        dtype=dtype,
        device=device,
    )
    if concat_latent_image.shape[0] != latent_shape[0]:
        concat_latent_image = concat_latent_image.repeat(latent_shape[0], 1, 1, 1, 1)
    concat_mask = torch.ones(
        (latent_shape[0], 1, latent_shape[2], latent_shape[3], latent_shape[4]),
        device=device,
        dtype=dtype,
    )
    concat_mask[:, :, 0, :, :] = 0

    return WanConditioningBundle(
        latent=latent,
        concat_latent_image=concat_latent_image,
        concat_mask=concat_mask,
    )
