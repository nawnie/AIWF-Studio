from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.sana_video import SANA_VIDEO_PIPELINE_I2V, SanaVideoRequest
from aiwf.services.sana_video import SanaVideoService, SanaVideoUnavailable


def _model_dir(root: Path) -> Path:
    model = root / "models" / "sana-video" / "Diffusers" / "SANA-Video_2B_480p_diffusers"
    model.mkdir(parents=True)
    (model / "model_index.json").write_text("{}", encoding="utf-8")
    return model


def test_sana_video_service_blocks_missing_model(tmp_path: Path):
    service = SanaVideoService(
        RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs"),
        UserSettings(),
    )

    with pytest.raises(SanaVideoUnavailable, match="model_index.json"):
        service.generate(SanaVideoRequest(prompt="slow camera move"))


def test_sana_video_service_exports_text_to_video(tmp_path: Path, monkeypatch):
    _model_dir(tmp_path)
    service = SanaVideoService(
        RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs"),
        UserSettings(),
    )
    captured = {}

    class FakePipe:
        vae = SimpleNamespace(enable_tiling=lambda: None, enable_slicing=lambda: None)

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            captured["model_path"] = path
            captured["load_kwargs"] = kwargs
            return cls()

        def to(self, device):
            captured["device"] = str(device)
            return self

        def set_progress_bar_config(self, **_kwargs):
            return None

        def __call__(self, **kwargs):
            captured["call_kwargs"] = kwargs
            frame = Image.new("RGB", (32, 32), "black")
            return SimpleNamespace(frames=[[frame, frame]])

    def fake_export(frames, output_path, *, fps):
        captured["frames"] = frames
        captured["fps"] = fps
        Path(output_path).write_bytes(b"video")

    monkeypatch.setattr("diffusers.SanaVideoPipeline", FakePipe)
    monkeypatch.setattr("diffusers.utils.export_to_video", fake_export)
    monkeypatch.setattr("aiwf.services.sana_video.VideoProcessor.probe", lambda self, path: SimpleNamespace(has_audio=False))

    result = service.generate(SanaVideoRequest(prompt="slow camera move", frames=2, fps=8, steps=1))

    assert result.output_path.endswith(".mp4")
    assert result.frames == 2
    assert result.has_audio is False
    assert captured["call_kwargs"]["prompt"] == "slow camera move"
    assert captured["fps"] == 8


def test_sana_video_i2v_requires_source_image(tmp_path: Path):
    _model_dir(tmp_path)
    service = SanaVideoService(
        RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs"),
        UserSettings(),
    )

    with pytest.raises(SanaVideoUnavailable, match="source image missing"):
        service.generate(SanaVideoRequest(prompt="animate this", pipeline=SANA_VIDEO_PIPELINE_I2V))
