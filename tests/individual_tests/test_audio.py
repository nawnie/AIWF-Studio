from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.audio import AudioGenerationOptions, AudioGenerationResult
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
    assert captured["command"][captured["command"].index("-af") + 1] == "apad"
    assert "-shortest" in captured["command"]
    assert result.output_path == str(out)


def test_video_audio_builds_mmaudio_command(tmp_path: Path):
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    video = tmp_path / "clip.mp4"
    out = tmp_path / "sound.flac"
    video.write_bytes(b"video")
    engine = tmp_path / "engines" / "audio" / "MMAudio"
    engine.mkdir(parents=True)
    (engine / "demo.py").write_text("print('demo')", encoding="utf-8")
    python = tmp_path / "engines" / "audio" / ".venv" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    captured = {}

    def fake_run(command, **_kwargs):
        captured["command"] = command
        output_dir = Path(command[command.index("--output") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "clip.flac").write_bytes(b"audio")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with patch("aiwf.services.audio.subprocess.run", side_effect=fake_run):
        result = service.generate_video_audio(
            video,
            AudioGenerationOptions(
                prompt="cloth movement and footsteps",
                kind="video_audio",
                model_id="mmaudio:small_16k",
                duration_seconds=5,
                cfg_coef=4.5,
                steps=12,
                seed=123,
            ),
            output_path=out,
        )

    command = captured["command"]
    assert command[0] == str(python)
    assert command[1] == str(engine / "demo.py")
    assert command[command.index("--variant") + 1] == "small_16k"
    assert command[command.index("--video") + 1] == str(video)
    assert command[command.index("--num_steps") + 1] == "12"
    assert "--skip_video_composite" in command
    assert result.output_path == str(out)
    assert result.kind == "video_audio"
    assert out.read_bytes() == b"audio"


def test_video_audio_accepts_single_alternate_mmaudio_flac(tmp_path: Path):
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    video = tmp_path / "clip.mp4"
    out = tmp_path / "sound.flac"
    video.write_bytes(b"video")
    engine = tmp_path / "engines" / "audio" / "MMAudio"
    engine.mkdir(parents=True)
    (engine / "demo.py").write_text("print('demo')", encoding="utf-8")
    python = tmp_path / "engines" / "audio" / ".venv" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")

    def fake_run(command, **_kwargs):
        output_dir = Path(command[command.index("--output") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "mmaudio_output.flac").write_bytes(b"audio")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with patch("aiwf.services.audio.subprocess.run", side_effect=fake_run):
        result = service.generate_video_audio(
            video,
            AudioGenerationOptions(
                prompt="cloth movement",
                kind="video_audio",
                model_id="mmaudio:large_44k_v2",
                duration_seconds=5,
            ),
            output_path=out,
        )

    assert result.output_path == str(out)
    assert out.read_bytes() == b"audio"


def test_generate_for_video_clamps_probed_duration(tmp_path: Path):
    service = AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    video = tmp_path / "short.mp4"
    video.write_bytes(b"video")
    captured = {}

    def fake_generate_video_audio(video_path, options):
        captured["video_path"] = video_path
        captured["duration_seconds"] = options.duration_seconds
        return AudioGenerationResult(
            output_path=str(tmp_path / "sound.flac"),
            prompt=options.prompt,
            model_id=options.model_id,
            kind="video_audio",
            duration_seconds=options.duration_seconds,
        )

    service.generate_video_audio = fake_generate_video_audio

    with patch(
        "aiwf.services.audio.VideoProcessor.probe",
        return_value=SimpleNamespace(duration_seconds=0.25),
    ):
        result = service.generate_for_video(
            video,
            AudioGenerationOptions(
                prompt="room tone",
                kind="video_audio",
                model_id="mmaudio:small_16k",
            ),
        )

    assert captured["video_path"] == video
    assert captured["duration_seconds"] == 1.0
    assert result.duration_seconds == 1.0
