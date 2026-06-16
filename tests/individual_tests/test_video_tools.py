from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.enhance import UpscaleOptions
from aiwf.core.domain.faceswap import FaceSwapOptions
from aiwf.core.domain.video import VideoProcessResult
from aiwf.infrastructure.storage.filesystem import FilesystemImageStore
from aiwf.infrastructure.video import process_frame_sequence
from aiwf.services.enhance import EnhanceService
from aiwf.services.faceswap import FaceSwapService


def test_process_frame_sequence_preserves_order_and_reports_progress():
    frames = [Image.new("RGB", (4, 4), (i, 0, 0)) for i in range(3)]
    progress: list[tuple[int, int]] = []

    result = process_frame_sequence(
        frames,
        lambda frame, index: Image.new("RGB", frame.size, (index + 10, 0, 0)),
        fps=12,
        total=3,
        on_progress=lambda done, total: progress.append((done, total)),
    )

    assert result.frame_count == 3
    assert result.fps == 12
    assert [frame.getpixel((0, 0))[0] for frame in result.frames] == [10, 11, 12]
    assert progress == [(1, 3), (2, 3), (3, 3)]


def test_faceswap_video_processes_each_frame(tmp_path: Path):
    service = FaceSwapService(RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models"))
    source = Image.new("RGB", (8, 8), "blue")
    progress: list[tuple[int, int]] = []

    def fake_video(input_path, output_path, processor, *, on_progress=None, **_kwargs):
        assert Path(input_path).name == "clip.mp4"
        for index in range(2):
            processor(Image.new("RGB", (8, 8), "red"), index)
            if on_progress:
                on_progress(index + 1, 2)
        return VideoProcessResult.saved(
            output_path,
            frame_count=2,
            fps=24,
            width=8,
            height=8,
            infotext="",
        )

    with patch("aiwf.services.faceswap.process_video_file", side_effect=fake_video), patch.object(
        service,
        "swap",
        side_effect=lambda frame, src, options, restore_fn=None: frame,
    ) as swap_mock:
        result = service.swap_video(
            tmp_path / "clip.mp4",
            source,
            FaceSwapOptions(model_id="inswapper_128"),
            output_path=tmp_path / "out.mp4",
            on_progress=lambda done, total: progress.append((done, total)),
        )

    assert swap_mock.call_count == 2
    assert result.path == str(tmp_path / "out.mp4")
    assert result.frame_count == 2
    assert "Face swap video" in result.infotext
    assert progress == [(1, 2), (2, 2)]


def test_enhance_video_pipeline_processes_each_frame(tmp_path: Path):
    flags = RuntimeFlags(data_dir=tmp_path)
    settings = UserSettings(save_images=False)
    service = EnhanceService(flags, settings, MagicMock(), FilesystemImageStore(tmp_path / "outputs"))

    def fake_video(_input_path, output_path, processor, *, on_progress=None, **_kwargs):
        for index in range(3):
            processor(Image.new("RGB", (8, 8), "green"), index)
            if on_progress:
                on_progress(index + 1, 3)
        return VideoProcessResult.saved(
            output_path,
            frame_count=3,
            fps=30,
            width=16,
            height=16,
            infotext="",
        )

    with patch("aiwf.services.enhance.process_video_file", side_effect=fake_video), patch.object(
        service,
        "run_pipeline",
        side_effect=lambda frame, **_kwargs: (
            Image.new("RGB", (16, 16), "green"),
            "Upscale: test (2x)",
        ),
    ) as pipeline_mock:
        result = service.run_video_pipeline(
            tmp_path / "clip.mp4",
            output_path=tmp_path / "enhanced.mp4",
            upscale=UpscaleOptions(model_id="realesrgan-x2plus", scale=2),
        )

    assert pipeline_mock.call_count == 3
    assert result.frame_count == 3
    assert result.path == str(tmp_path / "enhanced.mp4")
    assert result.infotext == "Video: Upscale: test (2x)"
