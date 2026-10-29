"""RIFE frame interpolation via ComfyUI-Frame-Interpolation (Practical-RIFE)."""
from __future__ import annotations

import logging
import os
import sys
import types
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

CKPT_NAME_VER_DICT: dict[str, str] = {
    "rife40.pth": "4.0",
    "rife41.pth": "4.0",
    "rife42.pth": "4.2",
    "rife43.pth": "4.3",
    "rife44.pth": "4.3",
    "rife45.pth": "4.5",
    "rife46.pth": "4.6",
    "rife47.pth": "4.7",
    "rife48.pth": "4.7",
    "rife49.pth": "4.7",
    "sudo_rife4_269.662_testV1_scale1.pth": "4.0",
}

DEFAULT_VFI_SEARCH_PATHS: tuple[Path, ...] = (
    Path(__file__).resolve().parents[3] / "engines" / "ComfyUI-Frame-Interpolation",
)


class RifeUnavailable(RuntimeError):
    """Raised when RIFE dependencies or model weights are missing."""


def _install_comfy_shim(device) -> None:
    """Minimal comfy.model_management shim for ComfyUI-Frame-Interpolation imports."""
    import torch

    if "comfy.model_management" in sys.modules:
        return

    def soft_empty_cache() -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mm = types.ModuleType("comfy.model_management")
    mm.get_torch_device = lambda: device
    mm.soft_empty_cache = soft_empty_cache

    comfy = types.ModuleType("comfy")
    comfy.model_management = mm
    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = mm


def resolve_vfi_root(*, extra_roots: list[Path] | None = None) -> Path | None:
    env = os.environ.get("AIWF_RIFE_VFI_ROOT", "").strip()
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    if extra_roots:
        candidates.extend(extra_roots)
    candidates.extend(DEFAULT_VFI_SEARCH_PATHS)
    for root in candidates:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        if (resolved / "vfi_utils.py").is_file() and (resolved / "config.yaml").is_file():
            return resolved
    return None


def list_rife_checkpoints(vfi_root: Path | None = None) -> list[str]:
    names = sorted(CKPT_NAME_VER_DICT.keys(), key=lambda n: CKPT_NAME_VER_DICT[n])
    root = vfi_root or resolve_vfi_root()
    if root is None:
        return names
    ckpt_dir = root / "ckpts" / "rife"
    if not ckpt_dir.is_dir():
        return names
    local = {p.name for p in ckpt_dir.glob("*.pth")}
    return sorted(local or names, key=lambda n: CKPT_NAME_VER_DICT.get(n, "0"))


def _ensure_vfi_paths(vfi_root: Path) -> None:
    base = str(vfi_root)
    models = str(vfi_root / "vfi_models")
    for entry in (models, base):
        if entry not in sys.path:
            sys.path.insert(0, entry)


def _load_frames_from_video(path: Path, *, max_frames: int | None, on_progress: Callable[[int, int], None] | None):
    import numpy as np
    import torch

    from aiwf.infrastructure.video.processing import VideoProcessor

    info = VideoProcessor().probe(path)
    total = info.frame_count
    if max_frames is not None:
        total = min(total, max_frames) if total else max_frames

    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RifeUnavailable(f"Could not open video: {path}")

    frames: list[torch.Tensor] = []
    try:
        while True:
            if max_frames is not None and len(frames) >= max_frames:
                break
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(np.asarray(rgb, dtype=np.float32) / 255.0))
            if on_progress and total:
                on_progress(len(frames), total)
    finally:
        cap.release()

    if len(frames) < 2:
        raise RifeUnavailable("RIFE needs at least 2 frames in the input video.")
    stacked = torch.stack(frames, dim=0)
    return stacked, float(info.fps or 24.0), int(info.width), int(info.height)


def _load_rife_model(ckpt_name: str, *, device, vfi_root: Path):
    import torch

    from rife.rife_arch import IFNet
    from vfi_utils import load_file_from_github_release, preprocess_frames

    if ckpt_name not in CKPT_NAME_VER_DICT:
        raise RifeUnavailable(f"Unknown RIFE checkpoint: {ckpt_name}")
    model_path = load_file_from_github_release("rife", ckpt_name)
    model = IFNet(arch_ver=CKPT_NAME_VER_DICT[ckpt_name])
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model.to(device), preprocess_frames


def _interpolate_with_loaded_model(
    frames_nhwc: "torch.Tensor",
    *,
    model,
    preprocess,
    multiplier: int,
    scale_factor: float,
    fast_mode: bool,
    ensemble: bool,
    clear_cache_every_n_frames: int,
    on_synth_progress: Callable[[int, int], None] | None = None,
) -> "torch.Tensor":
    """Interpolate one in-memory chunk while reusing an already loaded model."""
    import torch

    from vfi_utils import generic_frame_loop, postprocess_frames

    frames = preprocess(frames_nhwc)

    def return_middle_frame(frame_0, frame_1, timestep, mdl, scale_list, in_fast_mode, in_ensemble):
        return mdl(frame_0, frame_1, timestep, scale_list, in_fast_mode, in_ensemble)

    scale_list = [8 / scale_factor, 4 / scale_factor, 2 / scale_factor, 1 / scale_factor]
    synth_total = max(0, (frames.shape[0] - 1) * max(1, int(multiplier) - 1))
    ticks = {"n": 0}

    def _counting_cb(frame_0, frame_1, timestep, mdl, scales, in_fast_mode, in_ensemble):
        out = return_middle_frame(
            frame_0,
            frame_1,
            timestep,
            mdl,
            scales,
            in_fast_mode,
            in_ensemble,
        )
        ticks["n"] += 1
        if on_synth_progress and synth_total:
            on_synth_progress(min(ticks["n"], synth_total), synth_total)
        return out

    with torch.inference_mode():
        out = generic_frame_loop(
            "RIFE_VFI",
            frames,
            clear_cache_every_n_frames,
            int(multiplier),
            _counting_cb,
            model,
            scale_list,
            fast_mode,
            ensemble,
            dtype=torch.float32,
        )
    return postprocess_frames(out)


def interpolate_tensor_frames(
    frames_nhwc: "torch.Tensor",
    *,
    ckpt_name: str,
    multiplier: int,
    scale_factor: float,
    fast_mode: bool,
    ensemble: bool,
    clear_cache_every_n_frames: int,
    device,
    vfi_root: Path,
    on_synth_progress: Callable[[int, int], None] | None = None,
) -> "torch.Tensor":
    """Compatibility helper for callers that already hold all frames in memory."""
    import torch

    model, preprocess = _load_rife_model(ckpt_name, device=device, vfi_root=vfi_root)
    try:
        return _interpolate_with_loaded_model(
            frames_nhwc,
            model=model,
            preprocess=preprocess,
            multiplier=multiplier,
            scale_factor=scale_factor,
            fast_mode=fast_mode,
            ensemble=ensemble,
            clear_cache_every_n_frames=clear_cache_every_n_frames,
            on_synth_progress=on_synth_progress,
        )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _resample_interpolated_frames(
    frames,
    *,
    input_frame_count: int,
    input_fps: float,
    virtual_output_fps: float,
    target_fps: float | None,
):
    """Legacy in-memory resampler retained for API/test compatibility."""
    if target_fps is None:
        return frames, virtual_output_fps
    safe_target = max(1.0, float(target_fps))
    safe_input_fps = max(1.0, float(input_fps))
    duration_seconds = max(0.0, (max(2, int(input_frame_count)) - 1) / safe_input_fps)
    target_count = max(2, int(round(duration_seconds * safe_target)) + 1)
    source_count = int(frames.shape[0])
    if source_count <= 1 or target_count == source_count:
        return frames, safe_target

    import torch

    indices = torch.linspace(
        0,
        source_count - 1,
        steps=target_count,
        device=frames.device,
    ).round().to(dtype=torch.long)
    indices = indices.clamp_(0, source_count - 1)
    return frames.index_select(0, indices), safe_target


def interpolate_video_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    ckpt_name: str = "rife47.pth",
    multiplier: int = 2,
    scale_factor: float = 1.0,
    fast_mode: bool = True,
    ensemble: bool = True,
    clear_cache_every_n_frames: int = 10,
    chunk_input_frames: int = 16,
    max_input_frames: int | None = None,
    target_fps: float | None = None,
    device=None,
    vfi_root: Path | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[Path, int, int, float, float, int, int]:
    """Interpolate and stream a video in overlapping chunks.

    Only ``chunk_input_frames`` source frames and one interpolated chunk are held
    at a time. The RIFE model is loaded once and reused for every chunk. Adjacent
    chunks overlap by one source frame; the duplicate boundary frame is dropped
    before it reaches the encoder.
    """
    import numpy as np
    import torch

    from aiwf.infrastructure.video.processing import VideoProcessor
    from aiwf.infrastructure.video.stream_writer import StreamingVideoWriter

    src = Path(input_path).expanduser().resolve()
    if not src.is_file():
        raise RifeUnavailable(f"Video not found: {src}")

    dest = Path(output_path).expanduser().resolve()
    if dest == src:
        raise RifeUnavailable("RIFE output must be different from the input video.")
    dest.parent.mkdir(parents=True, exist_ok=True)

    chunk_size = max(2, int(chunk_input_frames))
    vfi_root = vfi_root or resolve_vfi_root()
    if vfi_root is None:
        raise RifeUnavailable(
            "ComfyUI-Frame-Interpolation not found. Install it under Comfy custom_nodes "
            "or set AIWF_RIFE_VFI_ROOT to the folder containing vfi_utils.py and config.yaml."
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _install_comfy_shim(device)
    _ensure_vfi_paths(vfi_root)

    info = VideoProcessor().probe(src)
    in_fps = float(info.fps or 24.0)
    width = int(info.width)
    height = int(info.height)
    if width <= 0 or height <= 0:
        raise RifeUnavailable(f"Could not determine input dimensions for {src}.")

    reported_frames = max(0, int(info.frame_count))
    if max_input_frames is not None:
        expected_input = min(reported_frames, max_input_frames) if reported_frames else max_input_frames
    else:
        expected_input = reported_frames
    total_pairs = max(1, expected_input - 1) if expected_input else 1

    try:
        import cv2
    except Exception as exc:
        raise RifeUnavailable("RIFE video streaming needs OpenCV.") from exc

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RifeUnavailable(f"Could not open video: {src}")

    model = None
    writer: StreamingVideoWriter | None = None
    buffer: list[torch.Tensor] = []
    in_count = 0
    streamed_count = 0
    processed_pairs = 0
    first_chunk = True
    virtual_out_fps = in_fps * float(multiplier)

    def emit_chunk(chunk: list[torch.Tensor]) -> None:
        nonlocal writer, streamed_count, processed_pairs, first_chunk
        if len(chunk) < 2:
            return
        stacked = torch.stack(chunk, dim=0)
        out_tensor = _interpolate_with_loaded_model(
            stacked,
            model=model,
            preprocess=preprocess,
            multiplier=multiplier,
            scale_factor=scale_factor,
            fast_mode=fast_mode,
            ensemble=ensemble,
            clear_cache_every_n_frames=clear_cache_every_n_frames,
        )
        if not first_chunk:
            out_tensor = out_tensor[1:]
        if writer is None:
            writer = StreamingVideoWriter(
                dest,
                width=width,
                height=height,
                input_fps=virtual_out_fps,
                target_fps=target_fps,
                audio_source=src,
            )
        for frame in out_tensor:
            writer.write(frame)
            streamed_count += 1
        processed_pairs += len(chunk) - 1
        if on_progress:
            on_progress(min(processed_pairs, total_pairs), total_pairs)
        first_chunk = False
        del out_tensor, stacked

    try:
        model, preprocess = _load_rife_model(ckpt_name, device=device, vfi_root=vfi_root)
        while True:
            if max_input_frames is not None and in_count >= max_input_frames:
                break
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(np.asarray(rgb, dtype=np.float32) / 255.0)
            buffer.append(frame)
            in_count += 1
            if len(buffer) >= chunk_size:
                emit_chunk(buffer)
                buffer = [buffer[-1]]

        if len(buffer) >= 2:
            emit_chunk(buffer)
        if in_count < 2 or writer is None:
            raise RifeUnavailable("RIFE needs at least 2 readable frames in the input video.")
        writer.close()
        writer = None
    except Exception:
        if writer is not None:
            writer.abort()
        raise
    finally:
        cap.release()
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_fps = float(target_fps or virtual_out_fps)
    out_count = streamed_count
    try:
        output_info = VideoProcessor().probe(dest)
        if output_info.fps > 0:
            out_fps = float(output_info.fps)
        if output_info.frame_count > 0:
            out_count = int(output_info.frame_count)
    except Exception:
        if target_fps is not None:
            duration = max(0.0, (in_count - 1) / max(1.0, in_fps))
            out_count = max(2, int(round(duration * out_fps)) + 1)

    return dest, in_count, out_count, in_fps, out_fps, width, height
