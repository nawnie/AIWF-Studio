"""Shared OpenCV-based video frame pipeline.

Decodes a video one frame at a time (streaming, so memory stays flat regardless
of clip length — friendly to 8 GB machines), hands each frame to a callback as a
PIL image, and re-encodes the result. Optionally remuxes the original audio with
FFmpeg when available. This module knows nothing about face swap or upscaling —
those live in their own services and are passed in via the frame callback.
"""
from __future__ import annotations

import functools
import logging
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

from aiwf.core.domain.video import VideoInfo, VideoProcessResult

logger = logging.getLogger(__name__)

# (frame, frame_index) -> processed frame (or None to keep the frame unchanged)
FrameCallback = Callable[[Image.Image, int], "Image.Image | None"]
ProgressCallback = Callable[[int, int], None]


class VideoError(RuntimeError):
    """Raised for missing / unsupported / corrupt video or encoder failure."""


def _frame_to_rgb_uint8(frame) -> np.ndarray:
    """Normalize PIL / numpy / torch video frames to HWC RGB uint8."""
    try:
        import torch as _torch

        if isinstance(frame, _torch.Tensor):
            t = frame.detach().cpu().float()
            if t.ndim == 4 and t.shape[0] == 1:
                t = t[0]
            if t.ndim == 3 and t.shape[0] in (1, 3, 4):
                t = t.permute(1, 2, 0)
            arr = t.numpy()
        else:
            arr = None
    except ImportError:
        arr = None

    if arr is None:
        if isinstance(frame, np.ndarray):
            arr = frame
        else:
            arr = np.asarray(frame.convert("RGB"))

    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3:
        raise VideoError(f"Unsupported video frame shape: {arr.shape}")
    if arr.shape[-1] == 1:
        arr = np.concatenate([arr] * 3, axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    elif arr.shape[-1] != 3:
        raise VideoError(f"Unsupported video frame channel count: {arr.shape[-1]}")

    if arr.dtype != np.uint8:
        max_value = float(np.nanmax(arr)) if arr.size else 0.0
        if max_value <= 1.0:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _require_cv2():
    try:
        import cv2

        return cv2
    except Exception as exc:  # pragma: no cover - environment check
        raise VideoError(
            "Video tools need OpenCV — install `opencv-python-headless`, then retry."
        ) from exc


@functools.lru_cache(maxsize=1)
def _resolve_ffmpeg() -> str | None:
    """Locate an ffmpeg binary from a *proper install*, not scavenged paths.

    Order: (1) ffmpeg on PATH, (2) the bundled binary from the ``imageio-ffmpeg``
    package (a real, pip-installed dependency — see requirements.txt). We do not
    reach into other applications' install trees.
    """
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).is_file():
            return exe
    except Exception:
        pass
    return None


def ffmpeg_available() -> bool:
    return _resolve_ffmpeg() is not None


def _transcode_to_h264(path: Path) -> bool:
    """Re-encode a finished video in-place to browser-playable H.264.

    OpenCV's ``mp4v`` fourcc writes MPEG-4 Part 2, which HTML5 ``<video>`` (and
    therefore gradio's preview / any browser) cannot decode — the clip then
    "fails to open" when loaded into the UI. H.264 + ``yuv420p`` + ``+faststart``
    (moov atom at the front for progressive play) is the universally supported
    combination. ffmpeg comes from the installed ``imageio-ffmpeg`` package (or
    PATH); if it is missing we leave the original file and warn.
    """
    ffmpeg = _resolve_ffmpeg()
    if ffmpeg is None:
        logger.warning(
            "ffmpeg not available — video left as mp4v and may not preview in the browser/gradio. "
            "Run `pip install imageio-ffmpeg` (it is in requirements.txt) and restart for H.264 output.",
        )
        return False
    tmp = path.with_name(f".{path.stem}.h264{path.suffix or '.mp4'}")
    try:
        subprocess.run(
            [
                ffmpeg, "-y", "-i", str(path),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "-an", str(tmp),
            ],
            capture_output=True, check=True, timeout=1800,
        )
        os.replace(str(tmp), str(path))
        return True
    except Exception as exc:
        logger.warning("H.264 transcode failed; leaving original codec: %s", exc)
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass
        return False


def _has_audio_stream(path: Path) -> bool:
    if shutil.which("ffprobe") is None:
        return False
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "a",
                "-show_entries", "stream=index", "-of", "csv=p=0", str(path),
            ],
            capture_output=True, text=True, timeout=30,
        )
        return bool(out.stdout.strip())
    except Exception:
        return False


class VideoProcessor:
    """Streaming frame transform over a video file."""

    def __init__(self, *, fourcc: str = "mp4v") -> None:
        self._fourcc = fourcc

    def probe(self, path: str | Path) -> VideoInfo:
        cv2 = _require_cv2()
        p = Path(path)
        if not p.is_file():
            raise VideoError(f"Video not found: {p}")
        cap = cv2.VideoCapture(str(p))
        try:
            if not cap.isOpened():
                raise VideoError(f"Could not open video (unsupported or corrupt): {p}")
            return VideoInfo(
                path=str(p),
                frame_count=max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))),
                fps=float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                has_audio=_has_audio_stream(p),
            )
        finally:
            cap.release()

    def process(
        self,
        input_path: str | Path,
        output_path: str | Path,
        frame_callback: FrameCallback,
        *,
        keep_audio: bool = False,
        on_progress: ProgressCallback | None = None,
        max_frames: int | None = None,
    ) -> VideoProcessResult:
        cv2 = _require_cv2()
        src = Path(input_path)
        if not src.is_file():
            raise VideoError(f"Video not found: {src}")
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp_video = out.with_name(f".{out.stem}.videoonly{out.suffix or '.mp4'}")

        cap = cv2.VideoCapture(str(src))
        writer = None
        fps = 24.0
        width = height = 0
        index = 0
        try:
            if not cap.isOpened():
                raise VideoError(f"Could not open video (unsupported or corrupt): {src}")
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 24.0
            total = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                processed = frame_callback(Image.fromarray(rgb), index)
                if processed is None:
                    processed = Image.fromarray(rgb)
                out_rgb = _frame_to_rgb_uint8(processed)
                out_bgr = np.ascontiguousarray(out_rgb[:, :, ::-1])
                if writer is None:
                    height, width = out_bgr.shape[:2]
                    writer = cv2.VideoWriter(
                        str(tmp_video),
                        cv2.VideoWriter_fourcc(*self._fourcc),
                        fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise VideoError("Could not open the video encoder (missing codecs?).")
                writer.write(out_bgr)
                index += 1
                if on_progress is not None:
                    on_progress(index, total)
                if max_frames is not None and index >= max_frames:
                    break
            if index == 0:
                raise VideoError(f"No readable frames in video: {src}")
        finally:
            cap.release()
            if writer is not None:
                writer.release()

        audio_muxed = False
        if keep_audio and ffmpeg_available() and _has_audio_stream(src):
            # _mux_audio re-encodes the video to H.264 while attaching audio.
            audio_muxed = self._mux_audio(src, tmp_video, out)
        if audio_muxed:
            if tmp_video.exists():
                tmp_video.unlink()
        else:
            # Publish only after the writer is closed. If remux/transcode fails,
            # users still get processed frames instead of losing the whole job
            # over optional audio/browser-codec polish.
            os.replace(str(tmp_video), str(out))
            # No audio path: cv2 wrote mp4v — transcode so it previews in-browser.
            _transcode_to_h264(out)

        return VideoProcessResult(
            output_path=str(out),
            frames_processed=index,
            fps=fps,
            width=width,
            height=height,
            audio_muxed=audio_muxed,
            message=f"{index} frame(s) at {width}x{height}, {fps:.0f} fps"
            + (" (audio kept)" if audio_muxed else ""),
        )

    def _mux_audio(self, src: Path, video_only: Path, final: Path) -> bool:
        """Copy the source's audio onto the processed video. Non-fatal."""
        ffmpeg = _resolve_ffmpeg()
        if ffmpeg is None:
            return False
        try:
            subprocess.run(
                [
                    ffmpeg, "-y", "-i", str(video_only), "-i", str(src),
                    # Re-encode video to browser-playable H.264 (the source is mp4v).
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    "-c:a", "aac",
                    "-map", "0:v:0", "-map", "1:a:0?", "-shortest", str(final),
                ],
                capture_output=True, check=True, timeout=1800,
            )
            return final.exists()
        except Exception as exc:
            logger.warning("Audio remux failed; writing video without audio: %s", exc)
            return False


def process_video_file(
    input_path: str | Path,
    output_path: str | Path,
    frame_callback: FrameCallback,
    *,
    keep_audio: bool = False,
    on_progress: ProgressCallback | None = None,
    max_frames: int | None = None,
    fourcc: str = "mp4v",
) -> VideoProcessResult:
    """Convenience wrapper: transform every frame of ``input_path`` with
    ``frame_callback`` and write to ``output_path``. ``max_frames`` caps the
    number of frames processed (handy for quick previews on small GPUs)."""
    return VideoProcessor(fourcc=fourcc).process(
        input_path,
        output_path,
        frame_callback,
        keep_audio=keep_audio,
        on_progress=on_progress,
        max_frames=max_frames,
    )


def write_frames(frames, output_path, *, fps: float = 24.0, fourcc: str = "mp4v") -> int:
    """Encode a list of PIL frames or numpy arrays (from VAE decode) into a video file (cv2).
    Returns frame count. Handles both to support diffusers video pipelines that may return np arrays.
    """
    cv2 = _require_cv2()
    usable = [f for f in frames if f is not None]
    if not usable:
        raise VideoError("No frames to encode.")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    def _to_rgb_uint8(frame):
        # torch.Tensor (C,H,W) or (H,W,C) from diffusers VAE decode
        try:
            import torch as _torch
            if isinstance(frame, _torch.Tensor):
                t = frame.detach().cpu().float()
                if t.ndim == 3 and t.shape[0] in (1, 3, 4):
                    t = t.permute(1, 2, 0)  # C,H,W → H,W,C
                arr = t.numpy()
                if arr.shape[-1] == 1:
                    arr = np.concatenate([arr] * 3, axis=-1)
                elif arr.shape[-1] == 4:
                    arr = arr[:, :, :3]
                if arr.max() <= 1.0:
                    arr = (arr * 255).clip(0, 255).astype(np.uint8)
                else:
                    arr = arr.clip(0, 255).astype(np.uint8)
                return arr
        except ImportError:
            pass
        if isinstance(frame, np.ndarray):
            arr = frame
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            if arr.shape[-1] == 4:
                arr = arr[:, :, :3]
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = (arr * 255).clip(0, 255).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)
            return arr
        else:
            # PIL Image
            return np.asarray(frame.convert("RGB"))

    first = _frame_to_rgb_uint8(usable[0])
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*fourcc), float(fps), (w, h))
    if not writer.isOpened():
        raise VideoError("Could not open the video encoder (missing codecs?).")
    try:
        for frame in usable:
            rgb = _frame_to_rgb_uint8(frame)
            if rgb.shape[:2] != (h, w):
                # Resize: since _to_rgb_uint8 already gave us a numpy array,
                # just use cv2 for any frame type.
                rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LANCZOS4)
            # cv2 expects contiguous BGR uint8
            bgr = rgb[:, :, ::-1] if rgb.shape[2] == 3 else rgb
            writer.write(np.ascontiguousarray(bgr))
    finally:
        writer.release()
    # cv2 wrote MPEG-4 Part 2 (mp4v); transcode to H.264 so the clip actually
    # plays in the browser / loads into gradio Video components.
    _transcode_to_h264(out)
    return len(usable)
