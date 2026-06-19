from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.vsr import VideoFxAigsOptions, VideoFxDenoiseOptions, VideoFxRelightOptions, VsrOptions
from aiwf.core.domain.video import VideoInfo
from aiwf.services.vsr import VsrInstallInfo, VsrService, VsrUnavailable


def test_vsr_options_defaults():
    opts = VsrOptions()
    assert opts.effect == "SuperRes"
    assert opts.scale == 2.0
    assert opts.mode == 3


def test_vsr_missing_sdk_raises_readable_error(tmp_path: Path):
    svc = VsrService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")

    with pytest.raises(VsrUnavailable, match="NVIDIA VideoFX needs"):
        svc.upscale(video, VsrOptions())


def test_vsr_detects_videofx_sdk_next_to_project_root(tmp_path: Path):
    project = tmp_path / "AIWF_Studio"
    sdk = tmp_path / "VideoFX"
    (sdk / "bin" / "models").mkdir(parents=True)
    (sdk / "features" / "nvvfxvideosuperres").mkdir(parents=True)
    (sdk / "nvvfx" / "include").mkdir(parents=True)
    (sdk / "bin" / "NVVideoEffects.dll").write_bytes(b"dll")
    (sdk / "bin" / "NVCVImage.dll").write_bytes(b"dll")
    (sdk / "bin" / "models" / "superres.trtpkg").write_bytes(b"model")
    (sdk / "features" / "install_feature.ps1").write_text("# installer", encoding="utf-8")
    (sdk / "nvvfx" / "include" / "nvVideoEffects.h").write_text("// header", encoding="utf-8")

    svc = VsrService(RuntimeFlags(data_dir=project), UserSettings())
    info = svc.install_info()

    assert info.sdk_root == sdk
    assert info.sdk_runtime_available
    assert not info.available
    assert info.model_dir == sdk / "bin" / "models"
    assert info.model_count == 1
    assert info.feature_names == ("nvvfxvideosuperres",)
    assert info.install_script == sdk / "features" / "install_feature.ps1"
    assert "Detected NVIDIA Video Effects SDK core" in svc.folder_help()
    assert "VideoEffectsApp.exe" in svc.folder_help()


def test_vsr_prefers_real_videofx_sdk_over_project_models_dir(tmp_path: Path):
    project = tmp_path / "AIWF_Studio"
    project_models = project / "models"
    app = project / "engines" / "nvidia-vfx-sdk-samples" / "build" / "apps" / "VideoEffectsApp" / "Release" / "VideoEffectsApp.exe"
    sdk = tmp_path / "VideoFX"
    project_models.mkdir(parents=True)
    app.parent.mkdir(parents=True)
    (sdk / "bin" / "models").mkdir(parents=True)
    (sdk / "features" / "nvvfxvideosuperres" / "bin").mkdir(parents=True)
    (sdk / "nvvfx" / "include").mkdir(parents=True)
    app.write_bytes(b"exe")
    (sdk / "bin" / "NVVideoEffects.dll").write_bytes(b"dll")
    (sdk / "bin" / "NVCVImage.dll").write_bytes(b"dll")
    (sdk / "bin" / "models" / "superres.trtpkg").write_bytes(b"model")
    (sdk / "features" / "nvvfxvideosuperres" / "bin" / "nvVFXVideoSuperRes.dll").write_bytes(b"dll")
    (sdk / "nvvfx" / "include" / "nvVideoEffects.h").write_text("// header", encoding="utf-8")

    svc = VsrService(RuntimeFlags(data_dir=project), UserSettings())
    info = svc.install_info()

    assert info.app_path == app
    assert info.sdk_root == sdk
    assert info.model_dir == sdk / "bin" / "models"
    assert info.model_dir != project_models
    assert sdk / "features" / "nvvfxvideosuperres" / "bin" in svc._candidate_path_entries(info)


def test_vsr_builds_video_effects_command(tmp_path: Path, monkeypatch):
    app = tmp_path / "sdk" / "samples" / "VideoEffectsApp" / "VideoEffectsApp.exe"
    models = tmp_path / "sdk" / "models"
    app.parent.mkdir(parents=True)
    models.mkdir(parents=True)
    app.write_bytes(b"exe")
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"in")
    out = tmp_path / "out.mp4"

    svc = VsrService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    monkeypatch.setattr(
        svc,
        "install_info",
        lambda: VsrInstallInfo(app_path=app, sdk_root=tmp_path / "sdk", model_dir=models),
    )
    monkeypatch.setattr(
        "aiwf.services.vsr.VideoProcessor.probe",
        lambda _self, path: VideoInfo(
            path=str(path),
            frame_count=2,
            fps=24.0,
            width=640 if Path(path) == src else 1280,
            height=480 if Path(path) == src else 960,
        ),
    )

    captured = {}

    def fake_run(command, **_kwargs):
        captured["command"] = command
        out.write_bytes(b"video")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with patch("aiwf.services.vsr.subprocess.run", side_effect=fake_run):
        result = svc.upscale(src, VsrOptions(scale=2.0), output_path=out)

    assert captured["command"][0] == str(app)
    assert f"--in_file={src}" in captured["command"]
    assert f"--out_file={out}" in captured["command"]
    assert "--effect=VideoSuperRes" in captured["command"]
    assert "--resolution=960" in captured["command"]
    assert "--mode=3" in captured["command"]
    assert f"--model_dir={models}" in captured["command"]
    assert result.output_width == 1280
    assert result.output_height == 960


def test_vsr_builds_upscale_pipeline_command(tmp_path: Path, monkeypatch):
    app = tmp_path / "sdk" / "samples" / "UpscalePipelineApp" / "UpscalePipelineApp.exe"
    models = tmp_path / "sdk" / "models"
    app.parent.mkdir(parents=True)
    models.mkdir(parents=True)
    app.write_bytes(b"exe")
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"in")
    out = tmp_path / "out.mp4"

    svc = VsrService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    monkeypatch.setattr(
        svc,
        "install_info",
        lambda: VsrInstallInfo(
            app_path=None,
            upscale_app_path=app,
            sdk_root=tmp_path / "sdk",
            model_dir=models,
        ),
    )
    monkeypatch.setattr(
        "aiwf.services.vsr.VideoProcessor.probe",
        lambda _self, path: VideoInfo(
            path=str(path),
            frame_count=2,
            fps=24.0,
            width=640 if Path(path) == src else 1280,
            height=480 if Path(path) == src else 960,
        ),
    )

    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["cwd"] = kwargs["cwd"]
        out.write_bytes(b"video")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with patch("aiwf.services.vsr.subprocess.run", side_effect=fake_run):
        result = svc.upscale(src, VsrOptions(effect="Upscale", scale=2.0, strength=0.7), output_path=out)

    assert captured["command"][0] == str(app)
    assert captured["cwd"] == str(app.parent)
    assert f"--in_file={src}" in captured["command"]
    assert f"--out_file={out}" in captured["command"]
    assert "--resolution=960" in captured["command"]
    assert "--upscale_strength=0.700" in captured["command"]
    assert f"--model_dir={models}" in captured["command"]
    assert result.output_width == 1280
    assert result.output_height == 960


def test_vsr_cleanup_forces_same_resolution(tmp_path: Path, monkeypatch):
    app = tmp_path / "sdk" / "samples" / "VideoEffectsApp" / "VideoEffectsApp.exe"
    models = tmp_path / "sdk" / "models"
    app.parent.mkdir(parents=True)
    models.mkdir(parents=True)
    app.write_bytes(b"exe")
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"in")
    out = tmp_path / "out.mp4"

    svc = VsrService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    monkeypatch.setattr(
        svc,
        "install_info",
        lambda: VsrInstallInfo(app_path=app, sdk_root=tmp_path / "sdk", model_dir=models),
    )
    monkeypatch.setattr(
        "aiwf.services.vsr.VideoProcessor.probe",
        lambda _self, path: VideoInfo(
            path=str(path),
            frame_count=2,
            fps=24.0,
            width=640,
            height=480,
        ),
    )

    captured = {}

    def fake_run(command, **_kwargs):
        captured["command"] = command
        out.write_bytes(b"video")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with patch("aiwf.services.vsr.subprocess.run", side_effect=fake_run):
        result = svc.upscale(src, VsrOptions(effect="Cleanup", scale=4.0, mode=12), output_path=out)

    assert "--effect=VideoSuperRes" in captured["command"]
    assert "--mode=12" in captured["command"]
    assert "--resolution=480" in captured["command"]
    assert result.output_width == 640
    assert result.output_height == 480
    assert "Cleanup" in result.infotext


def test_videofx_builds_denoise_command(tmp_path: Path, monkeypatch):
    app = tmp_path / "sdk" / "samples" / "DenoiseEffectApp" / "DenoiseEffectApp.exe"
    models = tmp_path / "sdk" / "models"
    app.parent.mkdir(parents=True)
    models.mkdir(parents=True)
    app.write_bytes(b"exe")
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"in")
    out = tmp_path / "out.mp4"

    svc = VsrService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    monkeypatch.setattr(
        svc,
        "install_info",
        lambda: VsrInstallInfo(
            app_path=None,
            denoise_app_path=app,
            sdk_root=tmp_path / "sdk",
            model_dir=models,
        ),
    )
    monkeypatch.setattr(
        "aiwf.services.vsr.VideoProcessor.probe",
        lambda _self, path: VideoInfo(
            path=str(path),
            frame_count=2,
            fps=24.0,
            width=512,
            height=288,
        ),
    )

    captured = {}

    def fake_run(command, **_kwargs):
        captured["command"] = command
        out.write_bytes(b"video")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with patch("aiwf.services.vsr.subprocess.run", side_effect=fake_run):
        result = svc.denoise(src, VideoFxDenoiseOptions(strength=0.7), output_path=out)

    assert captured["command"][0] == str(app)
    assert f"--in_file={src}" in captured["command"]
    assert f"--out_file={out}" in captured["command"]
    assert "--strength=0.700" in captured["command"]
    assert f"--model_dir={models}" in captured["command"]
    assert result.output_width == 512
    assert result.output_height == 288


def test_videofx_builds_aigs_command(tmp_path: Path, monkeypatch):
    app = tmp_path / "sdk" / "samples" / "AigsEffectApp" / "AigsEffectApp.exe"
    models = tmp_path / "sdk" / "models"
    bg = tmp_path / "background.png"
    app.parent.mkdir(parents=True)
    models.mkdir(parents=True)
    app.write_bytes(b"exe")
    bg.write_bytes(b"bg")
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"in")
    out = tmp_path / "out.mp4"

    svc = VsrService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    monkeypatch.setattr(
        svc,
        "install_info",
        lambda: VsrInstallInfo(
            app_path=None,
            aigs_app_path=app,
            sdk_root=tmp_path / "sdk",
            model_dir=models,
        ),
    )
    monkeypatch.setattr(
        "aiwf.services.vsr.VideoProcessor.probe",
        lambda _self, path: VideoInfo(
            path=str(path),
            frame_count=2,
            fps=24.0,
            width=512,
            height=288,
        ),
    )

    captured = {}

    def fake_run(command, **_kwargs):
        captured["command"] = command
        out.write_bytes(b"video")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with patch("aiwf.services.vsr.subprocess.run", side_effect=fake_run):
        result = svc.aigs(
            src,
            VideoFxAigsOptions(comp_mode=5, blur_strength=0.4, background_file=str(bg), cuda_graph=True),
            output_path=out,
        )

    assert captured["command"][0] == str(app)
    assert "--comp_mode=5" in captured["command"]
    assert "--blur_strength=0.400" in captured["command"]
    assert f"--bg_file={bg}" in captured["command"]
    assert "--cuda_graph=true" in captured["command"]
    assert f"--model_dir={models}" in captured["command"]
    assert result.output_width == 512
    assert result.output_height == 288


def test_videofx_builds_relight_command(tmp_path: Path, monkeypatch):
    app = tmp_path / "sdk" / "samples" / "RelightingEffectApp" / "RelightingEffectApp.exe"
    models = tmp_path / "sdk" / "models"
    hdr = app.parent / "Default.hdr"
    bg = tmp_path / "background.png"
    app.parent.mkdir(parents=True)
    models.mkdir(parents=True)
    app.write_bytes(b"exe")
    hdr.write_bytes(b"hdr")
    bg.write_bytes(b"bg")
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"in")
    out = tmp_path / "out.mp4"

    svc = VsrService(RuntimeFlags(data_dir=tmp_path), UserSettings())
    monkeypatch.setattr(
        svc,
        "install_info",
        lambda: VsrInstallInfo(
            app_path=None,
            relight_app_path=app,
            sdk_root=tmp_path / "sdk",
            model_dir=models,
        ),
    )
    monkeypatch.setattr(
        "aiwf.services.vsr.VideoProcessor.probe",
        lambda _self, path: VideoInfo(
            path=str(path),
            frame_count=2,
            fps=24.0,
            width=512,
            height=288,
        ),
    )

    captured = {}

    def fake_run(command, **_kwargs):
        captured["command"] = command
        out.write_bytes(b"video")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with patch("aiwf.services.vsr.subprocess.run", side_effect=fake_run):
        result = svc.relight(
            src,
            VideoFxRelightOptions(
                hdr_file=str(hdr),
                background_mode=3,
                background=str(bg),
                pan_degrees=-45,
                vfov_degrees=70,
                autorotate=True,
                rotation_rate=12,
            ),
            output_path=out,
        )

    assert captured["command"][0] == str(app)
    assert f"--in_hdr={hdr}" in captured["command"]
    assert "--bg_mode=3" in captured["command"]
    assert f"--in_bg={bg}" in captured["command"]
    assert "--pan=-45.000" in captured["command"]
    assert "--vfov=70.000" in captured["command"]
    assert "--autorotate=true" in captured["command"]
    assert "--rotation_rate=12.000" in captured["command"]
    assert f"--model_dir={models}" in captured["command"]
    assert result.output_width == 512
    assert result.output_height == 288
