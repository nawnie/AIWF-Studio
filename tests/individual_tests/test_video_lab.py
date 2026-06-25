from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest


def _make_clip(path: Path, *, frames: int = 12, fps: int = 12, audio: bool = True) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is not installed")
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc2=size=128x96:rate={fps}",
    ]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"]
    cmd += ["-frames:v", str(frames), "-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd += [str(path)]
    subprocess.run(cmd, check=True, timeout=60)
    return path


def test_video_lab_preset_is_typed():
    from aiwf.services.video_lab import preset_settings

    settings = preset_settings("old_family_film")
    assert settings.deinterlace is True
    assert settings.deflicker is True
    assert settings.audio_cleanup is True
    assert settings.container == "mp4"


def test_video_lab_executes_atomic_ffmpeg_job(tmp_path):
    from aiwf.core.domain.video_lab import VideoLabSettings
    from aiwf.infrastructure.video.ffmpeg_core import probe_media
    from aiwf.services.video_lab import VideoLabService

    source = _make_clip(tmp_path / "source.mp4")
    service = VideoLabService(tmp_path / "outputs")
    settings = VideoLabSettings(
        preset="custom",
        denoise="light",
        sharpen="light",
        audio_normalize=True,
        codec="h264",
        container="mp4",
        quality=24,
    )
    plan = service.build_plan(source, settings, job_id="vlab_test")
    assert plan.command
    assert plan.command[0].lower().endswith(("ffmpeg", "ffmpeg.exe"))
    assert "shell=True" not in service.plan_text(plan)

    progress: list[float] = []
    result = service.execute(plan, on_progress=lambda value, _message: progress.append(value))
    output = Path(result.output_path)
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    output_probe = probe_media(output)

    assert output.is_file() and output.stat().st_size > 0
    assert manifest["status"] == "completed"
    assert output_probe.width == 128
    assert output_probe.height == 96
    assert output_probe.has_audio is True
    assert progress and progress[-1] == 1.0


def test_rife_video_path_uses_overlapping_chunks(tmp_path, monkeypatch):
    import torch

    from aiwf.infrastructure.rife import backend

    source = _make_clip(tmp_path / "five.mp4", frames=5, fps=5, audio=False)
    output = tmp_path / "rife.mp4"
    fake_root = tmp_path / "vfi"
    fake_root.mkdir()
    calls: list[int] = []

    monkeypatch.setattr(backend, "_load_rife_model", lambda *args, **kwargs: (object(), lambda value: value))

    def fake_interpolate(frames, *, multiplier, **_kwargs):
        calls.append(int(frames.shape[0]))
        out = []
        for index in range(frames.shape[0] - 1):
            first = frames[index]
            second = frames[index + 1]
            out.append(first)
            for middle in range(1, int(multiplier)):
                alpha = middle / float(multiplier)
                out.append(first * (1.0 - alpha) + second * alpha)
        out.append(frames[-1])
        return torch.stack(out)

    monkeypatch.setattr(backend, "_interpolate_with_loaded_model", fake_interpolate)

    result = backend.interpolate_video_file(
        source,
        output,
        ckpt_name="rife47.pth",
        multiplier=2,
        chunk_input_frames=3,
        device=torch.device("cpu"),
        vfi_root=fake_root,
    )

    assert output.is_file() and output.stat().st_size > 0
    assert result[1] == 5
    assert result[2] == 9
    assert calls == [3, 3]


def test_video_lab_cancellation_removes_partial(tmp_path):
    import sys
    import threading
    import time

    from aiwf.core.domain.video_lab import MediaProbe, VideoLabPlan, VideoLabSettings
    from aiwf.services.video_lab import VideoLabCancelled, VideoLabService

    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    output = tmp_path / "job" / "out.mp4"
    temporary = output.with_name(f".{output.stem}.partial{output.suffix}")
    worker = tmp_path / "slow_worker.py"
    worker.write_text(
        """
import pathlib, signal, sys, time
stop = False
def halt(*_args):
    global stop
    stop = True
signal.signal(signal.SIGTERM, halt)
pathlib.Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
pathlib.Path(sys.argv[1]).write_bytes(b'partial')
for i in range(100):
    print(f'out_time_us={i * 100000}', flush=True)
    time.sleep(0.03)
    if stop:
        sys.exit(3)
""",
        encoding="utf-8",
    )
    plan = VideoLabPlan(
        input_path=str(source),
        output_path=str(output),
        job_id="vlab_cancel",
        probe=MediaProbe(path=str(source), duration_seconds=10, width=64, height=64, fps=10),
        settings=VideoLabSettings(),
        command=[sys.executable, str(worker), str(temporary)],
        expected_duration_seconds=10,
    )
    service = VideoLabService(tmp_path)
    captured: dict[str, object] = {}

    def run():
        try:
            service.execute(plan)
        except Exception as exc:  # expected cancellation path
            captured["error"] = exc

    thread = threading.Thread(target=run)
    thread.start()
    deadline = time.time() + 5
    while service.active_job_id is None and time.time() < deadline:
        time.sleep(0.01)
    assert service.active_job_id == "vlab_cancel"
    message = service.cancel_active()
    thread.join(timeout=10)

    assert "vlab_cancel" in message
    assert isinstance(captured.get("error"), VideoLabCancelled)
    assert not temporary.exists()
    manifest = json.loads((output.parent / "job.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "cancelled"


def test_video_lab_stage_parameters_compile_into_filter_graph() -> None:
    from aiwf.core.domain.video_lab import MediaProbe, VideoLabSettings
    from aiwf.infrastructure.video.ffmpeg_core import build_filter_graph

    settings = VideoLabSettings(
        deinterlace=True,
        deinterlace_mode="send_field",
        deinterlace_parity="bff",
        deinterlace_scope="interlaced",
        stabilize=True,
        stabilize_radius_x=20,
        stabilize_radius_y=12,
        stabilize_edge="clamp",
        stabilize_block_size=16,
        stabilize_contrast=90,
        deflicker=True,
        deflicker_size=9,
        deflicker_mode="median",
        denoise="custom",
        denoise_luma_spatial=2.1,
        denoise_chroma_spatial=1.7,
        denoise_luma_temporal=7.2,
        denoise_chroma_temporal=6.8,
        sharpen="custom",
        sharpen_kernel=7,
        sharpen_amount=0.65,
        scale="custom",
        custom_width=1440,
        custom_height=1080,
        keep_aspect=True,
        audio_cleanup=True,
        audio_highpass_hz=55,
        audio_lowpass_hz=15000,
        audio_noise_reduction_db=14,
        audio_noise_floor_db=-58,
        audio_noise_type="vinyl",
        audio_track_noise=True,
        audio_normalize=True,
        audio_target_lufs=-14,
        audio_true_peak_db=-1,
        audio_lra=9,
    )
    filters = {
        "bwdif", "deshake", "deflicker", "hqdn3d", "unsharp",
        "highpass", "lowpass", "afftdn", "loudnorm",
    }
    video, audio, warnings = build_filter_graph(
        settings,
        MediaProbe(path="sample.mp4", has_audio=True),
        filter_names=filters,
    )
    joined_video = ",".join(video)
    joined_audio = ",".join(audio)
    assert "bwdif=mode=send_field:parity=bff:deint=interlaced" in joined_video
    assert "deshake=rx=20:ry=12:edge=clamp:blocksize=16:contrast=90" in joined_video
    assert "deflicker=size=9:mode=median" in joined_video
    assert "hqdn3d=2.1:1.7:7.2:6.8" in joined_video
    assert "unsharp=7:7:0.65:7:7:0.0" in joined_video
    assert "force_original_aspect_ratio=decrease" in joined_video
    assert "highpass=f=55" in joined_audio
    assert "afftdn=nr=14:nf=-58:nt=vinyl:tn=1" in joined_audio
    assert "loudnorm=I=-14:TP=-1:LRA=9" in joined_audio
    assert warnings == []


def test_video_lab_custom_resize_requires_a_dimension() -> None:
    from pydantic import ValidationError
    from aiwf.core.domain.video_lab import VideoLabSettings

    with pytest.raises(ValidationError, match="Custom resize needs"):
        VideoLabSettings(scale="custom")
