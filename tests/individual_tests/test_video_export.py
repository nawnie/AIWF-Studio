"""tests/test_video_export.py — NVENC/video export tests (no ffmpeg required)."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from PIL import Image


class TestCodecSelection:
    def test_default_is_libx264(self) -> None:
        with patch.dict(os.environ, {"AIWF_NVENC": "0", "AIWF_HEVC": "0"}):
            from importlib import reload
            import aiwf.infrastructure.video.export as exp
            reload(exp)
            codec, pix = exp.select_codec()
        assert codec == "libx264"
        assert pix == "yuv420p"

    def test_hevc_flag_selects_libx265(self) -> None:
        with patch.dict(os.environ, {"AIWF_NVENC": "0", "AIWF_HEVC": "1"}):
            from importlib import reload
            import aiwf.infrastructure.video.export as exp
            reload(exp)
            codec, _ = exp.select_codec()
        assert codec == "libx265"

    def test_nvenc_unavailable_falls_back(self) -> None:
        with patch.dict(os.environ, {"AIWF_NVENC": "1", "AIWF_HEVC": "0"}):
            from importlib import reload
            import aiwf.infrastructure.video.export as exp
            reload(exp)
            exp._NVENC_PROBE_CACHE.clear()
            with patch.object(exp, "_probe_nvenc_codec", return_value=False):
                codec, _ = exp.select_codec()
        assert codec == "libx264"


class TestNVENCProbe:
    def test_probe_returns_false_without_ffmpeg(self) -> None:
        import shutil
        with patch.object(shutil, "which", return_value=None):
            from importlib import reload
            import aiwf.infrastructure.video.export as exp
            reload(exp)
            exp._NVENC_PROBE_CACHE.clear()
            result = exp._probe_nvenc_codec("h264_nvenc")
        assert result is False

    def test_probe_caches_result(self) -> None:
        from importlib import reload
        import aiwf.infrastructure.video.export as exp
        reload(exp)
        exp._NVENC_PROBE_CACHE.clear()
        exp._NVENC_PROBE_CACHE["h264_nvenc"] = True
        result = exp._probe_nvenc_codec("h264_nvenc")
        assert result is True


class TestFramesToVideo:
    def test_raises_without_ffmpeg(self, tmp_path: Path) -> None:
        import shutil
        frames = tmp_path / "frames"
        frames.mkdir()
        (frames / "00001.png").write_bytes(b"fake")
        with patch.object(shutil, "which", return_value=None):
            from aiwf.infrastructure.video.export import frames_to_video
            with pytest.raises(RuntimeError, match="ffmpeg not found"):
                frames_to_video(frames, tmp_path / "out.mp4")

    def test_raises_without_frames(self, tmp_path: Path) -> None:
        frames = tmp_path / "frames"
        frames.mkdir()
        import shutil
        with patch.object(shutil, "which", return_value="/usr/bin/ffmpeg"):
            from aiwf.infrastructure.video.export import frames_to_video
            with pytest.raises(RuntimeError, match="No PNG, JPG, or JPEG frames"):
                frames_to_video(frames, tmp_path / "out.mp4")

    def test_successful_encode(self, tmp_path: Path) -> None:
        frames = tmp_path / "frames"
        frames.mkdir()
        (frames / "00001.png").write_bytes(b"fake")

        output = tmp_path / "result.mp4"

        import shutil, aiwf.infrastructure.video.export as exp
        with patch.object(shutil, "which", return_value="/usr/bin/ffmpeg"):
            with patch("subprocess.run") as mock_run:
                mock_result = MagicMock()
                mock_result.returncode = 0
                mock_run.return_value = mock_result
                # Patch the stat call only on the output path via the module
                real_stat = output.__class__.stat
                def fake_stat(self_p, *a, **kw):
                    if self_p == output:
                        m = MagicMock(); m.st_size = 1024; return m
                    return real_stat(self_p, *a, **kw)
                with patch.object(output.__class__, "stat", fake_stat):
                    from aiwf.infrastructure.video.export import frames_to_video
                    frames_to_video(frames, output)

        # Verify shell=False: subprocess.run must be called with a list, not a string
        call_args = mock_run.call_args
        assert call_args is not None
        cmd = call_args[0][0]
        assert isinstance(cmd, list), "Must not use shell=True"

    @pytest.mark.parametrize("frame_ext", [".png", ".jpg", ".jpeg"])
    def test_frame_extension_matrix(self, tmp_path: Path, frame_ext: str) -> None:
        frames = tmp_path / f"frames_{frame_ext[1:]}"
        frames.mkdir()
        (frames / f"00001{frame_ext}").write_bytes(b"fake")
        output = tmp_path / "result.mp4"

        import shutil
        with patch.object(shutil, "which", return_value="/usr/bin/ffmpeg"):
            with patch("subprocess.run") as mock_run:
                mock_result = MagicMock(returncode=0)
                mock_run.return_value = mock_result
                real_stat = output.__class__.stat
                def fake_stat(self_p, *a, **kw):
                    if self_p == output:
                        m = MagicMock(); m.st_size = 1024; return m
                    return real_stat(self_p, *a, **kw)
                with patch.object(output.__class__, "stat", fake_stat):
                    from aiwf.infrastructure.video.export import frames_to_video
                    frames_to_video(frames, output)

        cmd = mock_run.call_args[0][0]
        assert str(frames / f"%05d{frame_ext}") in cmd

    @pytest.mark.parametrize(
        ("suffix", "expected_codec"),
        [
            (".mp4", "libx264"),
            (".mov", "libx264"),
            (".mkv", "libx264"),
            (".webm", "libvpx-vp9"),
        ],
    )
    def test_output_container_matrix(self, tmp_path: Path, suffix: str, expected_codec: str) -> None:
        frames = tmp_path / "frames"
        frames.mkdir()
        (frames / "00001.png").write_bytes(b"fake")
        output = tmp_path / f"result{suffix}"

        import shutil
        with patch.object(shutil, "which", return_value="/usr/bin/ffmpeg"):
            with patch("subprocess.run") as mock_run:
                mock_result = MagicMock(returncode=0)
                mock_run.return_value = mock_result
                real_stat = output.__class__.stat
                def fake_stat(self_p, *a, **kw):
                    if self_p == output:
                        m = MagicMock(); m.st_size = 1024; return m
                    return real_stat(self_p, *a, **kw)
                with patch.object(output.__class__, "stat", fake_stat):
                    from aiwf.infrastructure.video.export import frames_to_video
                    frames_to_video(frames, output)

        cmd = mock_run.call_args[0][0]
        assert cmd[-1] == str(output)
        assert cmd[cmd.index("-c:v") + 1] == expected_codec

    def test_unsupported_output_container_raises(self, tmp_path: Path) -> None:
        frames = tmp_path / "frames"
        frames.mkdir()
        (frames / "00001.png").write_bytes(b"fake")

        import shutil
        with patch.object(shutil, "which", return_value="/usr/bin/ffmpeg"):
            from aiwf.infrastructure.video.export import frames_to_video
            with pytest.raises(RuntimeError, match="Unsupported video output format"):
                frames_to_video(frames, tmp_path / "result.gif")


class TestTensorsToVideo:
    def test_accepts_pil_numpy_and_torch_frame_shapes(self, tmp_path: Path) -> None:
        torch = pytest.importorskip("torch")
        import numpy as np
        import aiwf.infrastructure.video.export as exp

        output = tmp_path / "mixed.mp4"
        frames = [
            Image.new("RGB", (8, 6), "red"),
            np.ones((6, 8, 3), dtype=np.float32),
            torch.ones(3, 6, 8),
            torch.ones(1, 3, 6, 8),
        ]

        def fake_frames_to_video(frames_dir, output_path, **kwargs):
            written = sorted(Path(frames_dir).glob("*.png"))
            assert len(written) == 4
            for frame_path in written:
                with Image.open(frame_path) as image:
                    assert image.mode == "RGB"
                    assert image.size == (8, 6)
            return Path(output_path)

        with patch.object(exp, "frames_to_video", side_effect=fake_frames_to_video):
            result = exp.tensors_to_video(frames, output, fps=12)

        assert result == output
