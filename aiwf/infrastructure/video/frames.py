from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from aiwf.core.domain.video import VideoProcessResult

FrameProcessor = Callable[[Image.Image, int], Image.Image]
ProgressCallback = Callable[[int, int], None]


@dataclass(frozen=True)
class FrameSequenceResult:
    frames: list[Image.Image]
    frame_count: int
    fps: float
    width: int
    height: int


def process_frame_sequence(
    frames: Iterable[Image.Image],
    processor: FrameProcessor,
    *,
    fps: float = 24.0,
    total: int | None = None,
    on_progress: ProgressCallback | None = None,
) -> FrameSequenceResult:
    """Apply a PIL frame processor to an in-memory sequence.

    Tests use this directly; file-backed video processing uses the same callback
    contract, so frame order and progress behavior stay shared.
    """
    output: list[Image.Image] = []
    total_frames = int(total or 0)
    for index, frame in enumerate(frames):
        processed = processor(frame.convert("RGB"), index).convert("RGB")
        output.append(processed)
        if on_progress:
            on_progress(index + 1, total_frames)

    if output:
        width, height = output[0].size
    else:
        width, height = 0, 0
    return FrameSequenceResult(
        frames=output,
        frame_count=len(output),
        fps=float(fps or 24.0),
        width=width,
        height=height,
    )


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python-headless is required for video processing.") from exc
    return cv2


def _pil_from_bgr(frame) -> Image.Image:
    cv2 = _cv2()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _bgr_from_pil(image: Image.Image):
    import numpy as np

    cv2 = _cv2()
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def process_video_file(
    input_path: str | Path,
    output_path: str | Path,
    processor: FrameProcessor,
    *,
    on_progress: ProgressCallback | None = None,
    codec: str = "mp4v",
    max_frames: int | None = None,
) -> VideoProcessResult:
    """Read a video, process each frame as PIL RGB, and write a new video.

    Audio is not copied. This is intentionally frame-only infrastructure shared
    by Enhance and Face Swap; callers should communicate the no-audio limitation
    in their UI or docs until an ffmpeg muxing layer is added.
    """
    cv2 = _cv2()
    src = Path(input_path)
    dest = Path(output_path)
    if not src.is_file():
        raise FileNotFoundError(src)
    dest.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(src))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {src}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 24.0)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if max_frames is not None:
        total = min(total, max_frames) if total else max_frames

    writer = None
    frame_count = 0
    width = 0
    height = 0
    try:
        while True:
            if max_frames is not None and frame_count >= max_frames:
                break
            ok, raw = capture.read()
            if not ok:
                break
            processed = processor(_pil_from_bgr(raw), frame_count).convert("RGB")
            if writer is None:
                width, height = processed.size
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(str(dest), fourcc, fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Could not write video: {dest}")
            writer.write(_bgr_from_pil(processed))
            frame_count += 1
            if on_progress:
                on_progress(frame_count, total)
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if frame_count == 0:
        raise RuntimeError(f"No frames read from video: {src}")

    return VideoProcessResult.saved(
        dest,
        frame_count=frame_count,
        fps=fps,
        width=width,
        height=height,
        infotext="",
    )
