from __future__ import annotations

from aiwf.infrastructure.video.frames import process_frame_sequence
from aiwf.infrastructure.video.last_frame import extract_last_frame, resolve_video_path
from aiwf.infrastructure.video.processing import (
    FrameCallback,
    ProgressCallback,
    VideoError,
    VideoProcessor,
    ffmpeg_available,
    process_video_file,
    write_frames,
)

__all__ = [
    "FrameCallback",
    "ProgressCallback",
    "VideoError",
    "VideoProcessor",
    "ffmpeg_available",
    "process_video_file",
    "write_frames",
    "extract_last_frame",
    "resolve_video_path",
    "process_frame_sequence",
]
