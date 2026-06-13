from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from aiwf.infrastructure.video import VideoError, VideoProcessor


def _make_video(path: Path, frames: int = 6, size=(32, 24), fps: float = 8.0) -> None:
    import cv2

    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert w.isOpened(), "test environment cannot encode mp4v"
    for i in range(frames):
        w.write(np.full((size[1], size[0], 3), (i * 30) % 255, dtype=np.uint8))
    w.release()


def test_probe_reads_metadata(tmp_path: Path):
    v = tmp_path / "in.mp4"
    _make_video(v, frames=6)
    info = VideoProcessor().probe(v)
    assert info.frame_count > 0
    assert info.fps > 0
    assert info.width == 32 and info.height == 24


def test_process_identity_callback(tmp_path: Path):
    v = tmp_path / "in.mp4"
    _make_video(v, frames=6)
    out = tmp_path / "out.mp4"
    seen: list[int] = []

    def cb(img: Image.Image, idx: int):
        assert isinstance(img, Image.Image)
        seen.append(idx)
        return img

    res = VideoProcessor().process(v, out, cb)
    assert out.exists()
    assert res.frames_processed > 0
    assert seen == list(range(res.frames_processed))
    # output is itself a readable video
    assert VideoProcessor().probe(out).frame_count > 0


def test_process_transform_changes_dimensions(tmp_path: Path):
    v = tmp_path / "in.mp4"
    _make_video(v, frames=4)
    out = tmp_path / "out.mp4"
    res = VideoProcessor().process(v, out, lambda img, idx: img.resize((16, 16)))
    assert res.width == 16 and res.height == 16
    assert out.exists()


def test_progress_callback_called(tmp_path: Path):
    v = tmp_path / "in.mp4"
    _make_video(v, frames=5)
    out = tmp_path / "out.mp4"
    calls: list[tuple[int, int]] = []
    VideoProcessor().process(v, out, lambda img, idx: img, on_progress=lambda i, t: calls.append((i, t)))
    assert len(calls) >= 1
    assert calls[-1][0] == VideoProcessor().probe(out).frame_count


def test_missing_input_raises_readable_error(tmp_path: Path):
    with pytest.raises(VideoError):
        VideoProcessor().process(tmp_path / "nope.mp4", tmp_path / "o.mp4", lambda img, idx: img)


def test_probe_missing_raises(tmp_path: Path):
    with pytest.raises(VideoError):
        VideoProcessor().probe(tmp_path / "nope.mp4")


def test_corrupt_video_raises(tmp_path: Path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"this is not a video file")
    with pytest.raises(VideoError):
        VideoProcessor().process(bad, tmp_path / "o.mp4", lambda img, idx: img)


def test_process_video_file_wrapper_and_max_frames(tmp_path: Path):
    from aiwf.infrastructure.video import process_video_file

    v = tmp_path / "in.mp4"
    _make_video(v, frames=10)
    out = tmp_path / "out.mp4"
    res = process_video_file(v, out, lambda img, idx: img, max_frames=4)
    assert res.frames_processed == 4
    assert out.exists()
    assert res.message  # human-readable summary populated
    # result supports the enhance/faceswap usage pattern
    updated = res.model_copy(update={"infotext": "x", "message": "done " + res.message})
    assert updated.infotext == "x" and updated.message.startswith("done ")
