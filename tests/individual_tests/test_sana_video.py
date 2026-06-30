from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.sana_video import (
    SANA_VIDEO_PIPELINE_I2V,
    SANA_VIDEO_QUANTIZATION_FP8,
    SANA_VIDEO_QUANTIZATION_BNB_INT8,
    SANA_VIDEO_VAE_TILING_ALWAYS,
    SanaVideoRequest,
)
from aiwf.services.sana_video import SanaVideoService, SanaVideoUnavailable, _SanaStageTracker


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

    import torch

    class FakeTransformer:
        def set_attention_backend(self, backend):
            captured["attention_backend"] = str(backend)

    class FakeTextEncoder:
        def to(self, device):
            captured.setdefault("text_encoder_devices", []).append(str(device))
            return self

    class FakeVae:
        def enable_tiling(self):
            captured["vae_tiling"] = True

        def enable_slicing(self):
            captured["vae_slicing"] = True

    class FakePipe:
        def __init__(self):
            self._execution_device = torch.device("cpu")
            self.transformer = FakeTransformer()
            self.text_encoder = FakeTextEncoder()
            self.vae = FakeVae()

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

        def encode_prompt(self, prompt, do_classifier_free_guidance, **kwargs):
            captured["encode_prompt"] = prompt
            captured["encode_guidance"] = do_classifier_free_guidance
            captured["encode_kwargs"] = kwargs
            return "prompt_embeds", "prompt_mask", "negative_embeds", "negative_mask"

        def __call__(self, **kwargs):
            captured["call_kwargs"] = kwargs
            callback = kwargs["callback_on_step_end"]
            captured["callback_output"] = callback(self, 0, 0, {})
            return SimpleNamespace(frames="fake-latents")

    def fake_export(frames, output_path, *, fps):
        captured["frames"] = frames
        captured["fps"] = fps
        Path(output_path).write_bytes(b"video")

    monkeypatch.setattr("diffusers.SanaVideoPipeline", FakePipe)
    monkeypatch.setattr("diffusers.utils.export_to_video", fake_export)
    monkeypatch.setattr("aiwf.services.sana_video.VideoProcessor.probe", lambda self, path: SimpleNamespace(has_audio=False))
    monkeypatch.setattr(
        service,
        "_decode_latents",
        lambda pipe, latents, request, tracker: ([Image.new("RGB", (32, 32), "black")] * int(request.frames), request.vae_tiling),
    )

    progress = []
    result = service.generate(
        SanaVideoRequest(prompt="slow camera move", frames=2, fps=8, steps=1, quantization="bf16"),
        on_progress=lambda *args: progress.append(args),
    )

    assert result.output_path.endswith(".mp4")
    assert result.frames == 2
    assert result.has_audio is False
    assert result.timings["load"] >= 0.0
    assert result.timings["encode"] >= 0.0
    assert result.timings["inference"] >= 0.0
    assert result.timings["decode"] >= 0.0
    assert result.receipt_path.endswith(".json")
    assert Path(result.receipt_path).is_file()
    latest = tmp_path / "_local" / "logs" / "sana_video_latest.json"
    assert latest.is_file()
    receipt_payload = json.loads(latest.read_text(encoding="utf-8"))
    assert receipt_payload["receipt_path"] == result.receipt_path
    assert receipt_payload["result"]["receipt_path"] == result.receipt_path
    assert result.progress[-1]["stage"] == "done"
    assert {event["stage"] for event in result.progress} >= {"load", "encode", "inference", "decode", "export", "done"}
    assert captured["encode_prompt"] == "slow camera move"
    assert captured["encode_kwargs"]["device"].type == "cpu"
    assert captured["text_encoder_devices"] == ["cpu"]
    assert captured["call_kwargs"]["prompt"] is None
    assert captured["call_kwargs"]["prompt_embeds"] == "prompt_embeds"
    assert captured["call_kwargs"]["negative_prompt_embeds"] == "negative_embeds"
    assert captured["call_kwargs"]["output_type"] == "latent"
    assert captured["callback_output"] == {}
    assert captured["fps"] == 8
    assert progress


def test_sana_video_moves_text_encoder_to_cuda_for_prompt_encode(tmp_path: Path, monkeypatch):
    service = SanaVideoService(
        RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs"),
        UserSettings(),
    )
    captured = {}

    class FakeTextEncoder:
        def to(self, device):
            captured.setdefault("devices", []).append(str(device))
            return self

    class FakeTensor:
        def __init__(self):
            self.device = "cpu"

        def to(self, *, device):
            self.device = str(device)
            return self

    class FakePipe:
        def __init__(self):
            self.text_encoder = FakeTextEncoder()
            self._execution_device = torch.device("cuda")

        def encode_prompt(self, prompt, guidance, **kwargs):
            captured["prompt"] = prompt
            captured["guidance"] = guidance
            captured["device"] = kwargs["device"]
            return (
                FakeTensor(),
                FakeTensor(),
                None,
                None,
            )

    import torch

    pipe = FakePipe()
    monkeypatch.setattr(service, "_execution_device", lambda _pipe: torch.device("cuda"))

    encoded = service._encode_prompt(pipe, SanaVideoRequest(prompt="fast encode", cfg_scale=6.0))

    assert captured["devices"] == ["cuda"]
    assert str(captured["device"]) == "cuda"
    assert str(encoded["prompt_embeds"].device) == "cuda"


def test_sana_video_i2v_requires_source_image(tmp_path: Path):
    _model_dir(tmp_path)
    service = SanaVideoService(
        RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs"),
        UserSettings(),
    )

    with pytest.raises(SanaVideoUnavailable, match="source image missing"):
        service.generate(SanaVideoRequest(prompt="animate this", pipeline=SANA_VIDEO_PIPELINE_I2V))


def test_sana_video_request_normalizes_quantization_and_vae_tiling():
    request = SanaVideoRequest(prompt="test", quantization="8bit", vae_tiling="yes")

    assert request.quantization == SANA_VIDEO_QUANTIZATION_BNB_INT8
    assert request.vae_tiling == SANA_VIDEO_VAE_TILING_ALWAYS


def test_sana_video_request_normalizes_fp8_quantization():
    request = SanaVideoRequest(prompt="test", quantization="fp8")

    assert request.quantization == SANA_VIDEO_QUANTIZATION_FP8


def test_sana_video_fp8_setup_applies_layerwise_casting_and_places_pipeline(tmp_path: Path, monkeypatch):
    service = SanaVideoService(
        RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs"),
        UserSettings(),
    )
    captured = {}

    class FakePipe:
        def to(self, device):
            captured["device"] = str(device)
            return self

    monkeypatch.setattr(service, "_device", lambda: "cuda")
    monkeypatch.setattr(
        SanaVideoService,
        "_apply_fp8_layerwise_casting",
        staticmethod(lambda pipe: captured.__setitem__("fp8_pipe", pipe)),
    )

    pipe = FakePipe()
    service._prepare_pipeline_after_load(pipe, SANA_VIDEO_QUANTIZATION_FP8)

    assert captured["fp8_pipe"] is pipe
    assert captured["device"] == "cuda"


def test_sana_video_service_uses_sage_attention_when_diffusers_supports_it(tmp_path: Path, monkeypatch):
    service = SanaVideoService(
        RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs"),
        UserSettings(),
    )
    captured = {}

    from diffusers.models import attention_dispatch as dispatch
    from diffusers.models.attention_dispatch import AttentionBackendName

    class FakeTransformer:
        def set_attention_backend(self, backend):
            captured["backend"] = backend

    monkeypatch.setattr(dispatch, "_CAN_USE_SAGE_ATTN", True, raising=False)
    monkeypatch.setattr(dispatch, "sageattn", object(), raising=False)

    status = service._apply_sage_attention(
        SimpleNamespace(transformer=FakeTransformer()),
        SanaVideoRequest(prompt="slow camera move"),
    )

    assert status == "diffusers.sage"
    assert captured["backend"] == AttentionBackendName.SAGE


def test_sana_video_decode_retries_with_vae_tiling_after_oom(tmp_path: Path, monkeypatch):
    service = SanaVideoService(
        RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs"),
        UserSettings(),
    )
    calls = {"decode": 0, "tiling": 0, "slicing": 0}

    class FakeVae:
        def enable_tiling(self):
            calls["tiling"] += 1

        def enable_slicing(self):
            calls["slicing"] += 1

    def fake_decode_once(_pipe, _latents, _request, **kwargs):
        calls["decode"] += 1
        if calls["decode"] == 1:
            raise RuntimeError("CUDA out of memory")
        assert kwargs["chunk_latent_frames"] == 1
        return ["frame"]

    monkeypatch.setattr(service, "_decode_latents_once", fake_decode_once)

    frames, tiling_mode = service._decode_latents(
        SimpleNamespace(vae=FakeVae()),
        object(),
        SanaVideoRequest(prompt="test", vae_tiling="auto"),
        _SanaStageTracker(),
    )

    assert frames == ["frame"]
    assert tiling_mode == "auto_retry_tiled_chunked"
    assert calls == {"decode": 2, "tiling": 1, "slicing": 1}


def test_sana_video_decode_falls_back_to_cpu_when_tiled_chunked_decode_ooms(tmp_path: Path, monkeypatch):
    service = SanaVideoService(
        RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs"),
        UserSettings(),
    )
    calls = {"decode": 0, "cpu": 0}

    def fake_decode_once(_pipe, _latents, _request, **kwargs):
        calls["decode"] += 1
        if calls["decode"] < 3:
            raise RuntimeError("CUDA out of memory")
        assert kwargs["chunk_latent_frames"] == 1
        assert kwargs["force_cpu"] is True
        return ["frame"]

    monkeypatch.setattr(service, "_decode_latents_once", fake_decode_once)
    monkeypatch.setattr(service, "_move_vae_to_cpu", lambda _pipe: calls.__setitem__("cpu", calls["cpu"] + 1))

    frames, tiling_mode = service._decode_latents(
        SimpleNamespace(vae=SimpleNamespace(enable_tiling=lambda **_kwargs: None)),
        object(),
        SanaVideoRequest(prompt="test", vae_tiling="auto"),
        _SanaStageTracker(),
    )

    assert frames == ["frame"]
    assert tiling_mode == "cpu_tiled_chunked"
    assert calls == {"decode": 3, "cpu": 1}


def test_sana_video_retries_native_attention_when_sage_rejects_attention_mask(tmp_path: Path, monkeypatch):
    _model_dir(tmp_path)
    service = SanaVideoService(
        RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs"),
        UserSettings(),
    )
    captured = {"denoise": 0, "backends": []}

    class FakeTransformer:
        def set_attention_backend(self, backend):
            captured["backends"].append(backend)

    class FakePipe:
        transformer = FakeTransformer()
        text_encoder = None

    def fake_load(_model_path, _request, *, image_to_video):
        assert image_to_video is False
        return FakePipe(), {"quantization": "bnb_int8", "attention_backend": "diffusers.sage"}

    def fake_run(_pipe, _request, *, source_image, prompt_inputs, tracker):
        assert source_image is None
        assert prompt_inputs == {"prompt_embeds": "embeds"}
        captured["denoise"] += 1
        if captured["denoise"] == 1:
            raise ValueError("`attn_mask` is not supported for sage attention")
        tracker.emit("inference", 0.82, "Denoising step 1/1", step=1, total=1)
        return "latents"

    def fake_export(_frames, output_path, *, fps):
        Path(output_path).write_bytes(b"video")

    monkeypatch.setattr(service, "_load_pipeline", fake_load)
    monkeypatch.setattr(service, "_encode_prompt", lambda _pipe, _request: {"prompt_embeds": "embeds"})
    monkeypatch.setattr(service, "_run_pipeline_to_latents", fake_run)
    monkeypatch.setattr(service, "_decode_latents", lambda _pipe, _latents, request, tracker: (["frame"], request.vae_tiling))
    monkeypatch.setattr("diffusers.utils.export_to_video", fake_export)
    monkeypatch.setattr("aiwf.services.sana_video.VideoProcessor.probe", lambda self, path: SimpleNamespace(has_audio=False))

    result = service.generate(SanaVideoRequest(prompt="slow camera move", frames=1, steps=1))

    assert captured["denoise"] == 2
    assert result.attention_backend == "native_after_sage_mask_retry"
    assert any(event["message"].startswith("Sage attention cannot handle") for event in result.progress)


def test_sana_video_export_flattens_single_video_frame_batch(tmp_path: Path, monkeypatch):
    captured = {}

    def fake_export(frames, output_path, *, fps):
        captured["frames"] = frames
        captured["fps"] = fps
        Path(output_path).write_bytes(b"video")

    monkeypatch.setattr("diffusers.utils.export_to_video", fake_export)

    output = tmp_path / "nested.mp4"
    frames = [[Image.new("RGB", (16, 16), "black"), Image.new("RGB", (16, 16), "white")]]

    SanaVideoService._export_frames(frames, output, fps=8)

    assert output.is_file()
    assert len(captured["frames"]) == 2
    assert all(hasattr(frame, "convert") for frame in captured["frames"])
    assert captured["fps"] == 8


def test_sana_video_writes_failure_receipt(tmp_path: Path, monkeypatch):
    _model_dir(tmp_path)
    service = SanaVideoService(
        RuntimeFlags(data_dir=tmp_path, models_dir=tmp_path / "models", output_dir=tmp_path / "outputs"),
        UserSettings(),
    )

    class FakePipe:
        text_encoder = None

    monkeypatch.setattr(
        service,
        "_load_pipeline",
        lambda _model_path, _request, *, image_to_video: (
            FakePipe(),
            {"quantization": "bnb_int8", "attention_backend": "native"},
        ),
    )
    monkeypatch.setattr(service, "_encode_prompt", lambda _pipe, _request: {"prompt_embeds": "embeds"})
    monkeypatch.setattr(service, "_run_pipeline_to_latents", lambda *_args, **_kwargs: "latents")
    monkeypatch.setattr(service, "_decode_latents", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("decode failed")))

    with pytest.raises(RuntimeError, match="decode failed"):
        service.generate(SanaVideoRequest(prompt="slow camera move", frames=1, steps=1))

    latest = tmp_path / "_local" / "logs" / "sana_video_latest.json"
    payload = json.loads(latest.read_text(encoding="utf-8"))

    assert payload["status"] == "error"
    assert payload["attention_backend"] == "native"
    assert payload["quantization"] == "bnb_int8"
    assert payload["timings"]["load"] >= 0.0
    assert payload["error"]["type"] == "RuntimeError"
    assert payload["error"]["message"] == "decode failed"
    assert payload["progress"][-1]["stage"] == "error"
