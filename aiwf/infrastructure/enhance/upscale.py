from __future__ import annotations

import logging

import numpy as np
import torch
from PIL import Image

from aiwf.core.domain.enhance import EnhanceModel, UpscaleOptions
from aiwf.infrastructure.enhance.tiles import Grid, combine_grid, split_grid

logger = logging.getLogger(__name__)


def _pil_to_bgr_tensor(image: Image.Image, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    arr = np.array(image.convert("RGB"))
    arr = arr[:, :, ::-1]
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.ascontiguousarray(arr) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).to(device=device, dtype=dtype)


def _bgr_tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    arr = tensor.float().detach().cpu().clamp_(0, 1).numpy()
    arr = 255.0 * np.moveaxis(arr, 0, 2)
    arr = arr.round().astype(np.uint8)
    arr = arr[:, :, ::-1]
    return Image.fromarray(arr, "RGB")


def _resolve_model_parts(model) -> tuple[object, torch.device, torch.dtype]:
    net = getattr(model, "model", model)
    device = getattr(model, "device", None)
    dtype = getattr(model, "dtype", None)

    if device is not None and dtype is not None:
        return net, device, dtype

    if hasattr(net, "parameters"):
        params = iter(net.parameters())
        first_param = next(params, None)
        if first_param is not None:
            return net, first_param.device, first_param.dtype

    if hasattr(net, "buffers"):
        buffers = iter(net.buffers())
        first_buffer = next(buffers, None)
        if first_buffer is not None:
            return net, first_buffer.device, first_buffer.dtype

    raise TypeError("Upscale model does not expose an inference device/dtype.")


def _upscale_tile(model, image: Image.Image) -> Image.Image:
    net, device, dtype = _resolve_model_parts(model)
    with torch.inference_mode():
        tensor = _pil_to_bgr_tensor(image, device, dtype)
        output = net(tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        return _bgr_tensor_to_pil(output)


def upscale_image(
    image: Image.Image,
    model,
    *,
    model_info: EnhanceModel,
    options: UpscaleOptions,
) -> Image.Image:
    tile_size = int(options.tile_size or 0)
    tile_overlap = int(options.tile_overlap or 0)

    if tile_size <= 0:
        upscaled = _upscale_tile(model, image)
    else:
        grid = split_grid(image, tile_size, tile_size, tile_overlap)
        scale_factor = 1
        newtiles = []
        for y, h, row in grid.tiles:
            newrow = []
            for x, w, tile in row:
                output = _upscale_tile(model, tile)
                scale_factor = max(1, output.width // max(tile.width, 1))
                newrow.append([x * scale_factor, w * scale_factor, output])
            newtiles.append([y * scale_factor, h * scale_factor, newrow])
        newgrid = Grid(
            newtiles,
            grid.tile_w * scale_factor,
            grid.tile_h * scale_factor,
            grid.image_w * scale_factor,
            grid.image_h * scale_factor,
            grid.overlap * scale_factor,
        )
        upscaled = combine_grid(newgrid)

    target_scale = float(options.scale)
    native_scale = float(model_info.scale or 4)
    if abs(target_scale - native_scale) > 0.01:
        new_w = max(1, int(image.width * target_scale))
        new_h = max(1, int(image.height * target_scale))
        upscaled = upscaled.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return upscaled
