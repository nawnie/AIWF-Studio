from __future__ import annotations

import logging
from collections.abc import Callable

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import normalize

from aiwf.core.domain.enhance import EnhanceModel, RestoreOptions

logger = logging.getLogger(__name__)


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return rgb[:, :, ::-1]


def _bgr_to_pil(image: np.ndarray) -> Image.Image:
    rgb = image[:, :, ::-1]
    return Image.fromarray(rgb.astype(np.uint8), "RGB")


def _bgr_to_tensor(image: np.ndarray) -> torch.Tensor:
    if image.dtype in (np.float32, np.float64):
        rgb = image[:, :, ::-1].astype(np.float32, copy=False)
        return torch.from_numpy(rgb.transpose(2, 0, 1)).float()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.transpose(2, 0, 1)).float()


def _tensor_to_bgr(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.squeeze(0).float().detach().cpu().clamp_(-1, 1).numpy()
    arr = (arr.transpose(1, 2, 0) + 1.0) / 2.0
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _create_face_helper(device: torch.device):
    from facexlib.utils.face_restoration_helper import FaceRestoreHelper

    return FaceRestoreHelper(
        upscale_factor=1,
        face_size=512,
        crop_ratio=(1, 1),
        det_model="retinaface_resnet50",
        save_ext="png",
        use_parse=True,
        device=device,
    )


def _restore_faces(
    np_image: np.ndarray,
    face_helper,
    restore_face: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    original_resolution = np_image.shape[0:2]
    try:
        face_helper.clean_all()
        face_helper.read_image(np_image)
        face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
        face_helper.align_warp_face()
        for cropped_face in face_helper.cropped_faces:
            cropped_face_t = _bgr_to_tensor(cropped_face / 255.0)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(device)
            with torch.no_grad():
                restored = restore_face(cropped_face_t)
            restored_face = _tensor_to_bgr(restored)
            face_helper.add_restored_face(restored_face)
        face_helper.get_inverse_affine(None)
        result = face_helper.paste_faces_to_input_image()
        if original_resolution != result.shape[0:2]:
            result = cv2.resize(
                result,
                (original_resolution[1], original_resolution[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        return result
    finally:
        face_helper.clean_all()


def restore_image(
    image: Image.Image,
    net: torch.nn.Module,
    *,
    model_info: EnhanceModel,
    options: RestoreOptions,
    device: torch.device,
) -> Image.Image:
    face_helper = _create_face_helper(device)
    np_image = _pil_to_bgr(image)

    if model_info.architecture == "CodeFormer":
        weight = float(options.codeformer_weight)

        def restore_face(cropped_face_t: torch.Tensor) -> torch.Tensor:
            return net(cropped_face_t, w=weight, adain=True)[0]
    else:

        def restore_face(cropped_face_t: torch.Tensor) -> torch.Tensor:
            return net(cropped_face_t, return_rgb=False)[0]

    restored_bgr = _restore_faces(np_image, face_helper, restore_face, device)
    restored = _bgr_to_pil(restored_bgr)

    visibility = float(options.visibility)
    if visibility < 1.0:
        restored = Image.blend(image.convert("RGB"), restored, visibility)
    return restored