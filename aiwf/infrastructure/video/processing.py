"""Shared OpenCV-based video frame pipeline.

Decodes a video one frame at a time (streaming, so memory stays flat regardless
of clip length — friendly to 8 GB machines), hands each frame to a callback as a
PIL image, and re-encodes the result. Optionally remuxes the original audio with
FFmpeg when available. This module knows nothing about face swap or upscaling —
those live in their own services and are passed in via the frame callback.
"""
from __future__ import annotations

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


def _require_cv2():
    try:
        import cv2

        return cv2
    except Exception as exc:  # pragma: no cover - environment check
        raise VideoError(
            "Video tools need OpenCV — install `opencv-python-headless`, then retry."
        ) from exc


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


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
                out_rgb = np.asarray(processed.convert("RGB"))
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
            audio_muxed = self._mux_audio(src, tmp_video, out)
        if audio_muxed:
            if tmp_video.exists():
                tmp_video.unlink()
        else:
            os.replace(str(tmp_video), str(out))

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
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(video_only), "-i", str(src),
                    "-c:v", "copy", "-c:a", "aac",
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
    """Encode a list of PIL frames into a video file (cv2). Returns frame count.

    Used by generators (e.g. Wan I2V) that produce frames in memory rather than
    transforming an existing clip.
    """
    cv2 = _require_cv2()
    usable = [f for f in frames if f is not None]
    if not usable:
        raise VideoError("No frames to encode.")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    first = np.asarray(usable[0].convert("RGB"))
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*fourcc), float(fps), (w, h))
    if not writer.isOpened():
        raise VideoError("Could not open the video encoder (missing codecs?).")
    try:
        for frame in usable:
            rgb = frame.convert("RGB")
            if rgb.size != (w, h):
                rgb = rgb.resize((w, h))
            writer.write(np.ascontiguousarray(np.asarray(rgb)[:, :, ::-1]))
    finally:
        writer.release()
    return len(usable)
