from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from aiwf.infrastructure.video.processing import VideoError, _require_cv2, _resolve_ffmpeg


def resolve_video_path(video: Any) -> Path:
    """Normalize common Gradio video values to a local filesystem path."""
    if video is None:
        raise VideoError("Upload a video first.")

    raw_path: Any
    if isinstance(video, str | Path):
        raw_path = video
    elif isinstance(video, dict):
        raw_path = video.get("path") or video.get("name")
    else:
        raw_path = getattr(video, "path", None) or getattr(video, "name", None)

    if not raw_path:
        raise VideoError("Could not read the uploaded video path.")

    path = Path(raw_path)
    if not path.is_file():
        raise VideoError(f"Video not found: {path}")
    return path


def extract_last_frame(video: Any) -> Image.Image:
    """Return the final visible video frame as a PIL RGB image.

    The primary path seeks near the end of the file and reads one frame. It does
    not materialize the clip as a frame list, keeping Gradio callbacks light
    enough for long videos. A bounded ffmpeg tail decode is used only when
    OpenCV cannot seek/read the final frame for a container.
    """
    path = resolve_video_path(video)
    last_error: Exception | None = None

    try:
        frame = _extract_last_frame_cv2(path)
        if frame is not None:
            return frame
    except Exception as exc:
        last_error = exc

    try:
        return _extract_last_frame_ffmpeg(path)
    except Exception as exc:
        if last_error is not None:
            raise VideoError(f"Could not extract the last frame from video: {path}") from last_error
        raise VideoError(f"Could not extract the last frame from video: {path}") from exc


def _extract_last_frame_cv2(path: Path) -> Image.Image | None:
    cv2 = _require_cv2()
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise VideoError(f"Could not open video (unsupported or corrupt): {path}")

        frame_count = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        if frame_count > 0:
            # Some codecs fail exactly on the last reported frame, so walk a
            # small tail window backward instead of decoding from frame zero.
            start = max(0, frame_count - 1)
            stop = max(-1, frame_count - 16)
            for frame_index in range(start, stop, -1):
                frame = _read_cv2_frame_at(capture, cv2, frame_index)
                if frame is not None:
                    return _pil_from_bgr(cv2, frame)

        for ratio in (0.999, 0.995, 0.99, 0.95):
            capture.set(cv2.CAP_PROP_POS_AVI_RATIO, ratio)
            ok, frame = capture.read()
            if ok and frame is not None:
                return _pil_from_bgr(cv2, frame)
        return None
    finally:
        capture.release()


def _read_cv2_frame_at(capture: Any, cv2: Any, frame_index: int) -> Any | None:
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
    ok, frame = capture.read()
    return frame if ok and frame is not None else None


def _pil_from_bgr(cv2: Any, frame: Any) -> Image.Image:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb).convert("RGB")


def _extract_last_frame_ffmpeg(path: Path) -> Image.Image:
    ffmpeg = _resolve_ffmpeg()
    if ffmpeg is None:
        raise VideoError("ffmpeg is not available.")

    last_error: Exception | None = None
    with tempfile.TemporaryDirectory(prefix="aiwf-last-frame-") as tmp_dir:
        frame_path = Path(tmp_dir) / "frame.png"
        for tail_seconds in (5, 30, 120):
            try:
                subprocess.run(
                    [
                        ffmpeg,
                        "-y",
                        "-sseof",
                        f"-{tail_seconds}",
                        "-i",
                        str(path),
                        "-map",
                        "0:v:0",
                        "-an",
                        "-update",
                        "1",
                        str(frame_path),
                    ],
                    capture_output=True,
                    check=True,
                    timeout=180,
                )
                if frame_path.is_file() and frame_path.stat().st_size > 0:
                    with Image.open(frame_path) as image:
                        return image.convert("RGB").copy()
            except Exception as exc:
                last_error = exc
                if frame_path.exists():
                    frame_path.unlink(missing_ok=True)

    raise VideoError(f"No readable frames in video: {path}") from last_error
