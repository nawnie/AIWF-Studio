"""
aiwf/infrastructure/video/export.py

Video export engine — GPU-accelerated (NVENC) and software (libx264/libx265).

Flag: AIWF_NVENC=1   → use h264_nvenc / hevc_nvenc when available
      AIWF_HEVC=1    → prefer H.265 / HEVC over H.264

What NVENC is
-------------
NVENC is NVIDIA's hardware video encoder built into RTX/Quadro/Tesla GPUs.
It encodes H.264/H.265 video entirely on the GPU, offloading the encoder
from the CPU and running 3–5× faster than software libx264.

NVENC does NOT require CUDA to be installed — it uses the NVENC driver API
via ffmpeg's h264_nvenc encoder.

Detection
---------
On first encode call we probe ffmpeg for h264_nvenc support by running a
null encode.  If the probe fails we fall back to libx264 silently.
The probe result is cached for the process lifetime.

Codec selection table
---------------------
AIWF_NVENC | AIWF_HEVC | Codec selected
    0       |     0     | libx264 (software H.264)
    1       |     0     | h264_nvenc (GPU H.264) → libx264 fallback
    0       |     1     | libx265 (software H.265)
    1       |     1     | hevc_nvenc (GPU H.265) → libx265 fallback

All paths use subprocess with shell=False.  No shell=True anywhere.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

_NVENC_ENABLED = os.environ.get("AIWF_NVENC", "0") == "1"
_HEVC_ENABLED  = os.environ.get("AIWF_HEVC", "0") == "1"

# Cache of (nvenc_h264_ok, nvenc_hevc_ok)
_NVENC_PROBE_CACHE: dict[str, bool] = {}

PixelFormat = Literal["yuv420p", "yuv444p", "p010le"]
SUPPORTED_FRAME_EXTENSIONS = (".png", ".jpg", ".jpeg")
SUPPORTED_VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm")


# ---------------------------------------------------------------------------
# NVENC detection
# ---------------------------------------------------------------------------

def _probe_nvenc_codec(codec: str) -> bool:
    """Return True if ffmpeg supports *codec* on this machine."""
    if codec in _NVENC_PROBE_CACHE:
        return _NVENC_PROBE_CACHE[codec]

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        _NVENC_PROBE_CACHE[codec] = False
        return False

    # Probe: generate 1 frame of black video and encode it
    cmd = [
        ffmpeg, "-y",
        "-f", "lavfi", "-i", "color=black:s=64x64:r=1:d=1",
        "-vframes", "1",
        "-c:v", codec,
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        ok = result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        ok = False

    _NVENC_PROBE_CACHE[codec] = ok
    if ok:
        logger.info("NVENC probe: %s available", codec)
    else:
        logger.debug("NVENC probe: %s not available", codec)
    return ok


def nvenc_available() -> bool:
    return _probe_nvenc_codec("h264_nvenc")


def hevc_nvenc_available() -> bool:
    return _probe_nvenc_codec("hevc_nvenc")


# ---------------------------------------------------------------------------
# Codec selection
# ---------------------------------------------------------------------------

def select_codec() -> tuple[str, str]:
    """Return (video_codec, pixel_format) based on env flags and availability.

    Respects AIWF_NVENC and AIWF_HEVC flags with automatic fallback.
    """
    want_nvenc = _NVENC_ENABLED
    want_hevc  = _HEVC_ENABLED

    if want_hevc:
        if want_nvenc and _probe_nvenc_codec("hevc_nvenc"):
            return "hevc_nvenc", "p010le"
        return "libx265", "yuv420p"
    else:
        if want_nvenc and _probe_nvenc_codec("h264_nvenc"):
            return "h264_nvenc", "yuv420p"
        return "libx264", "yuv420p"


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

def frames_to_video(
    frames_dir: Path,
    output_path: Path,
    fps: float = 24.0,
    crf: int = 18,
    preset: str = "medium",
    audio_path: Path | None = None,
) -> Path:
    """Encode a directory of frame images (PNG/JPEG) into a video file.

    Parameters
    ----------
    frames_dir:
        Directory containing frame images named %05d.png (or %05d.jpg).
        Frames are sorted lexicographically.
    output_path:
        Destination video file.  Extension determines container format.
    fps:
        Output frame rate.
    crf:
        Constant Rate Factor (quality).  Lower = better.  18 is visually
        lossless for H.264.  For NVENC, mapped to ``-cq`` (similar meaning).
    preset:
        Encoder speed preset.  "fast"/"medium"/"slow" for libx264/x265.
        NVENC presets: "p1" (fast) .. "p7" (slow).
    audio_path:
        Optional audio file to mux into the output.  Copied without re-encoding.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg not found.  Install ffmpeg and ensure it is on PATH."
        )

    output_suffix = output_path.suffix.lower()
    if output_suffix not in SUPPORTED_VIDEO_EXTENSIONS:
        raise RuntimeError(
            f"Unsupported video output format: {output_suffix or '<none>'}. "
            f"Supported formats: {', '.join(SUPPORTED_VIDEO_EXTENSIONS)}"
        )

    codec, pix_fmt = select_codec()

    # Detect frame pattern
    frame_ext = _detect_frame_extension(frames_dir)
    pattern = str(frames_dir / f"%05d{frame_ext}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build ffmpeg command
    cmd = [
        ffmpeg, "-y",
        "-framerate", str(fps),
        "-i", pattern,
    ]

    if audio_path and audio_path.is_file():
        cmd += ["-i", str(audio_path), "-c:a", "copy", "-shortest"]

    # Video codec args. WebM needs VP9/VP8 rather than H.264/H.265.
    if output_suffix == ".webm":
        cmd += ["-c:v", "libvpx-vp9", "-crf", str(crf), "-b:v", "0", "-pix_fmt", "yuv420p"]
    elif "nvenc" in codec:
        # NVENC quality: -cq (similar to -crf for software encoders)
        cmd += ["-c:v", codec, "-cq", str(crf), "-pix_fmt", pix_fmt]
        # Map NVENC preset: medium → p4
        nvenc_preset_map = {
            "fast": "p2", "medium": "p4", "slow": "p6",
            "p1": "p1", "p2": "p2", "p3": "p3", "p4": "p4",
            "p5": "p5", "p6": "p6", "p7": "p7",
        }
        cmd += ["-preset", nvenc_preset_map.get(preset, "p4")]
    else:
        cmd += ["-c:v", codec, "-crf", str(crf), "-preset", preset, "-pix_fmt", pix_fmt]

    cmd.append(str(output_path))

    logger.info("Encoding video: %s → %s [%s]", frames_dir.name, output_path.name, codec)
    logger.debug("ffmpeg cmd: %s", cmd)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg encoding failed (codec={codec}):\n{result.stderr[-1000:]}"
        )

    logger.info("Video export complete: %s (%.1f MB)", output_path.name, output_path.stat().st_size / 1e6)
    return output_path


def _detect_frame_extension(frames_dir: Path) -> str:
    for ext in SUPPORTED_FRAME_EXTENSIONS:
        if sorted(frames_dir.glob(f"*{ext}")):
            return ext
    raise RuntimeError(
        f"No PNG, JPG, or JPEG frames found in {frames_dir}"
    )


def tensors_to_video(
    frames,          # list[PIL.Image] or list[torch.Tensor] or list[np.ndarray]
    output_path: Path,
    fps: float = 24.0,
    crf: int = 18,
    preset: str = "medium",
    audio_path: Path | None = None,
) -> Path:
    """Write a list of frame images to video without an intermediate frames-dir.

    Frames are written to a temp directory, then passed to ``frames_to_video``.
    """
    import tempfile
    from PIL import Image as _PIL

    with tempfile.TemporaryDirectory(prefix="aiwf_export_") as tmp:
        tmp_dir = Path(tmp)
        for i, frame in enumerate(frames):
            # Normalise frame to PIL
            pil = _normalise_frame_to_pil(frame, _PIL)

            pil.save(tmp_dir / f"{i:05d}.png")

        return frames_to_video(
            tmp_dir, output_path,
            fps=fps, crf=crf, preset=preset, audio_path=audio_path,
        )


def _normalise_frame_to_pil(frame, pil_module):
    import numpy as _np

    if hasattr(frame, "save"):
        return frame.convert("RGB")

    if hasattr(frame, "detach"):
        arr = frame.detach().cpu().float().numpy()
    elif hasattr(frame, "numpy"):
        arr = frame.numpy()
    else:
        arr = _np.asarray(frame)

    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        arr = _np.transpose(arr, (1, 2, 0))
    if arr.ndim == 2:
        arr = _np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3:
        raise RuntimeError(f"Unsupported video frame shape: {arr.shape}")
    if arr.shape[-1] == 1:
        arr = _np.concatenate([arr] * 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    elif arr.shape[-1] != 3:
        raise RuntimeError(f"Unsupported video frame channel count: {arr.shape[-1]}")

    if arr.dtype != _np.uint8:
        max_value = float(_np.nanmax(arr)) if arr.size else 0.0
        if max_value <= 1.0:
            arr = (arr * 255).clip(0, 255).astype(_np.uint8)
        else:
            arr = arr.clip(0, 255).astype(_np.uint8)
    return pil_module.fromarray(_np.ascontiguousarray(arr), mode="RGB")
