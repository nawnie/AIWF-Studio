from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.audio import AudioGenerationOptions
from aiwf.services.audio import AudioGenerationService, AudioUnavailable


def test_audio_options_defaults():
    opts = AudioGenerationOptions(prompt="calm synth score")
    assert opts.kind == "music"
    assert opts.model_id == "facebook/musicgen-small"
    assert opts.duration_seconds == 8.0


def test_audio_generation_missing_prompt_raises(tmp_path: Path):
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())

    with pytest.raises(AudioUnavailable, match="audio prompt"):
        service.generate(AudioGenerationOptions(prompt=""))


def test_audio_mux_builds_ffmpeg_command(tmp_path: Path):
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    video = tmp_path / "video.mp4"
    audio = tmp_path / "audio.wav"
    out = tmp_path / "out.mp4"
    video.write_bytes(b"video")
    audio.write_bytes(b"audio")
    captured = {}

    def fake_run(command, **_kwargs):
        captured["command"] = command
        out.write_bytes(b"muxed")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with patch("aiwf.services.audio._resolve_ffmpeg", return_value="ffmpeg"), patch(
        "aiwf.services.audio.subprocess.run",
        side_effect=fake_run,
    ):
        result = service.mux_audio(video, audio, output_path=out)

    assert captured["command"][:5] == ["ffmpeg", "-y", "-i", str(video), "-i"]
    assert str(audio) in captured["command"]
    assert "-shortest" in captured["command"]
    assert result.output_path == str(out)
