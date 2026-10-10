"""Commercial-safe audio by default (aiwf/services/audio_licenses.py and its enforcement).

AIWF Studio is sold commercially, so out of the box it must not offer or run audio models
whose licences forbid commercial use (MusicGen and MMAudio are CC-BY-NC 4.0). Research mode
brings them back, labeled. These tests pin that promise at every layer: the licence registry,
the audio service (pickers, generate, prepare, install), and the Pro API routes.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.audio import AudioGenerationOptions
from aiwf.services import audio_licenses
from aiwf.services.audio import AudioGenerationService, AudioLicenseBlocked, AudioUnavailable


@pytest.fixture(autouse=True)
def _no_research_mode_from_the_environment(monkeypatch):
    # UserSettings reads this variable; make sure a developer shell cannot flip the default
    monkeypatch.delenv("ALLOW_NONCOMMERCIAL_AUDIO_MODELS", raising=False)


def _service(tmp_path: Path, *, research: bool) -> AudioGenerationService:
    return AudioGenerationService(RuntimeFlags(data_dir=tmp_path), UserSettings(allow_noncommercial_audio_models=research))


# --- the registry ------------------------------------------------------------------------------
@pytest.mark.parametrize("model_id", [
    "facebook/musicgen-small", "facebook/musicgen-medium", "facebook/musicgen-melody", "facebook/musicgen-stereo-small",
    "mmaudio:small_16k", "mmaudio:large_44k_v2",
])
def test_musicgen_and_mmaudio_are_not_commercial(model_id: str) -> None:
    record = audio_licenses.license_for(model_id)
    assert record["license"] == "CC-BY-NC-4.0" and record["commercial"] == audio_licenses.NO
    assert not audio_licenses.allowed(model_id, research_mode=False)
    assert audio_licenses.allowed(model_id, research_mode=True)


@pytest.mark.parametrize(("model_id", "licence"), [
    ("acestep:1.5-turbo", "MIT"),
    ("moss-sfx:v2.0", "Apache-2.0"),
])
def test_replacement_models_are_commercial(model_id: str, licence: str) -> None:
    record = audio_licenses.license_for(model_id)
    assert record["license"] == licence and audio_licenses.commercial_ok(model_id)
    assert record["source"].startswith("https://huggingface.co/") and record["checked"] == audio_licenses.CHECKED


def test_unknown_models_are_never_treated_as_commercial() -> None:
    record = audio_licenses.license_for("someone/new-audio-model")
    assert record["license"] == "unknown" and not audio_licenses.allowed("someone/new-audio-model", research_mode=False)


def test_labels_and_refusal_text_name_the_licence() -> None:
    assert audio_licenses.short_label("acestep:1.5-turbo") == "MIT, commercial use OK"
    assert audio_licenses.short_label("mmaudio:small_16k") == "CC-BY-NC-4.0, non-commercial only"
    message = audio_licenses.blocked_message("facebook/musicgen-small")
    assert "CC-BY-NC-4.0" in message and "research" in message


# --- the audio service ---------------------------------------------------------------------------
def test_commercial_mode_offers_no_noncommercial_models(tmp_path: Path) -> None:
    service = _service(tmp_path, research=False)
    assert service.research_mode() is False
    # only the commercially licensed engines are offered; no MusicGen or MMAudio
    assert [m for _, m in service.music_model_choices()] == ["acestep:1.5-turbo"]
    assert [m for _, m in service.sfx_model_choices()] == ["moss-sfx:v2.0"]
    assert [m for _, m in service.video_audio_model_choices()] == ["events:moss-sfx"]


def test_prepare_accepts_event_soundtrack_as_video_audio_sfx(tmp_path: Path, monkeypatch) -> None:
    from contextlib import nullcontext

    service = _service(tmp_path, research=False)
    monkeypatch.setattr(service, "commercial_engine_missing", lambda _model_id: [])
    monkeypatch.setattr(service, "_gpu_tenant", lambda _reason: nullcontext())
    monkeypatch.setattr(service, "_release_image_models", lambda: None)
    monkeypatch.setattr(service, "unload", lambda: None)

    result = service.prepare(kind="sfx", model_id="events:moss-sfx")

    assert result["ready"] is True
    assert result["modelId"] == "events:moss-sfx"
    with pytest.raises(AudioUnavailable, match="Choose a supported music model"):
        service.prepare(kind="music", model_id="events:moss-sfx")


def test_research_mode_offers_them_labeled(tmp_path: Path) -> None:
    service = _service(tmp_path, research=True)
    labels = [label for label, model in service.music_model_choices() if model.startswith("facebook/")]
    assert labels and all("non-commercial (CC-BY-NC-4.0)" in label for label in labels)
    assert ("mmaudio:small_16k" in {model for _, model in service.video_audio_model_choices()})


@pytest.mark.parametrize(("kind", "model_id"), [("music", "facebook/musicgen-small"), ("sfx", "mmaudio:small_16k")])
def test_generate_and_prepare_refuse_noncommercial_models_before_any_work(tmp_path: Path, kind: str, model_id: str) -> None:
    service = _service(tmp_path, research=False)
    with pytest.raises(AudioLicenseBlocked):
        service.generate(AudioGenerationOptions(prompt="rain on a tin roof", kind=kind, model_id=model_id))
    with pytest.raises(AudioLicenseBlocked):
        service.prepare(kind=kind, model_id=model_id)
    assert not any(tmp_path.rglob("*.wav")) and not any(tmp_path.rglob("*.flac"))


def test_video_audio_refuses_mmaudio_in_commercial_mode(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not really a video")
    service = _service(tmp_path, research=False)
    with pytest.raises(AudioLicenseBlocked):
        service.generate_video_audio(video, AudioGenerationOptions(prompt="footsteps", kind="video_audio", model_id="mmaudio:small_16k"))


def test_installers_refuse_noncommercial_weights(tmp_path: Path) -> None:
    service = _service(tmp_path, research=False)
    with pytest.raises(AudioLicenseBlocked):
        service.install_musicgen_variant("small")
    with pytest.raises(AudioLicenseBlocked):
        service.install_mmaudio_variant("small_16k")


def test_status_reports_the_policy(tmp_path: Path) -> None:
    status = _service(tmp_path, research=False).setup_status(deep=False)
    assert status["researchMode"] is False and status["licenseNotice"].startswith("Commercial-safe mode")
    assert status["defaults"] == {"music": "acestep:1.5-turbo", "sfx": "moss-sfx:v2.0", "videoAudio": "events:moss-sfx"}
    assert status["licenses"]["facebook/musicgen-small"]["commercial"] == audio_licenses.NO
    research = _service(tmp_path, research=True).setup_status(deep=False)
    assert research["researchMode"] is True and research["defaults"]["videoAudio"] == "events:moss-sfx"


def test_event_soundtrack_readiness_requires_commercial_vision_model(tmp_path: Path, monkeypatch) -> None:
    import aiwf.services.audio as audio_module

    from aiwf.services.audio import EVENTS_MODEL_ID
    from aiwf.services.video_soundtrack import ChatDescriber

    service = _service(tmp_path, research=False)
    engine_python = tmp_path / "moss-python.exe"
    engine_python.write_bytes(b"python")
    weights = tmp_path / "weights"
    weights.mkdir()
    (weights / "config.json").write_text("{}", encoding="utf-8")
    spec = {
        "engine": "moss-sfx",
        "label": "MOSS-SoundEffect v2.0",
        "python": engine_python,
        "weights": weights,
        "files": ("config.json",),
    }
    monkeypatch.setattr(service, "_engine_spec", lambda model_id: spec if model_id == EVENTS_MODEL_ID else None)
    monkeypatch.setattr("aiwf.services.audio._resolve_ffmpeg", lambda: "ffmpeg")
    now = [100.0]
    monkeypatch.setattr(audio_module, "monotonic", lambda: now[0])
    calls = []

    def timed_out_probe(self, *, timeout):
        calls.append(timeout)
        raise TimeoutError("fixture chat model-list timeout")

    monkeypatch.setattr(ChatDescriber, "available_model", timed_out_probe)

    missing_vision = service.commercial_engine_missing(EVENTS_MODEL_ID)
    assert missing_vision == ["Qwen2.5-VL-7B vision model in the local chat engine"]
    assert service.commercial_engine_ready(EVENTS_MODEL_ID) is False
    assert service.available_video_audio_model_choices() == []
    assert service.commercial_engine_missing(EVENTS_MODEL_ID) == missing_vision
    assert calls == [audio_module._EVENTS_DESCRIBER_READINESS_TIMEOUT_SECONDS]

    # A cached negative probe expires promptly, then a commercial model makes the route ready.
    now[0] += audio_module._EVENTS_DESCRIBER_READINESS_CACHE_SECONDS + 0.01

    def available_probe(self, *, timeout):
        calls.append(timeout)
        return "Qwen2.5-VL-7B-Instruct"

    monkeypatch.setattr(ChatDescriber, "available_model", available_probe)
    assert service.commercial_engine_missing(EVENTS_MODEL_ID) == []
    assert service.commercial_engine_ready(EVENTS_MODEL_ID) is True
    assert service.available_video_audio_model_choices() == service.video_audio_model_choices()
    assert calls == [
        audio_module._EVENTS_DESCRIBER_READINESS_TIMEOUT_SECONDS,
        audio_module._EVENTS_DESCRIBER_READINESS_TIMEOUT_SECONDS,
    ]


def test_event_readiness_skips_chat_probe_until_local_prerequisites_are_ready(tmp_path: Path, monkeypatch) -> None:
    from aiwf.services.audio import EVENTS_MODEL_ID
    from aiwf.services.video_soundtrack import ChatDescriber

    service = _service(tmp_path, research=False)
    spec = {
        "engine": "moss-sfx",
        "label": "MOSS-SoundEffect v2.0",
        "python": tmp_path / "missing-python.exe",
        "weights": tmp_path / "weights",
        "files": ("config.json",),
    }
    monkeypatch.setattr(service, "_engine_spec", lambda model_id: spec if model_id == EVENTS_MODEL_ID else None)
    monkeypatch.setattr("aiwf.services.audio._resolve_ffmpeg", lambda: None)
    monkeypatch.setattr(
        ChatDescriber,
        "available_model",
        lambda self, **kwargs: pytest.fail("must not probe chat while local prerequisites are missing"),
    )

    missing = service.commercial_engine_missing(EVENTS_MODEL_ID)

    assert any("MOSS-SoundEffect v2.0 environment" in item for item in missing)
    assert "ffmpeg (needed to read video frames)" in missing
    assert str(spec["weights"] / "config.json") in missing
    assert all("Qwen2.5-VL-7B" not in item for item in missing)


def test_minimum_readiness_respects_commercial_policy(tmp_path: Path, monkeypatch) -> None:
    import aiwf.services.audio as audio_module
    from aiwf.services.audio_lab import AudioLabService

    monkeypatch.setattr(
        AudioLabService,
        "status",
        lambda _self, *, deep: SimpleNamespace(installed=True, python_path="audio-lab-python", message="ready"),
    )
    monkeypatch.setattr(audio_module, "_resolve_ffmpeg", lambda: "ffmpeg")

    commercial = _service(tmp_path / "commercial", research=False)
    research = _service(tmp_path / "research", research=True)
    monkeypatch.setattr(commercial, "_resolve_ffprobe", lambda _ffmpeg: "ffprobe")
    monkeypatch.setattr(research, "_resolve_ffprobe", lambda _ffmpeg: "ffprobe")

    commercial_status = commercial.setup_status(deep=False)
    research_status = research.setup_status(deep=False)

    assert commercial_status["labReady"] is True and commercial_status["muxReady"] is True
    assert commercial_status["musicReady"] is False and commercial_status["sfxReady"] is False
    assert commercial_status["videoAudioReady"] is False
    assert commercial_status["minimumReady"] is True
    assert research_status["labReady"] is True and research_status["muxReady"] is True
    assert research_status["minimumReady"] is False


def test_minimum_setup_skips_noncommercial_downloads_in_commercial_mode(tmp_path: Path, monkeypatch) -> None:
    import subprocess

    import aiwf.services.audio as audio_module

    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "bootstrap_audio_minimum.py").write_text("", encoding="utf-8")
    seen: list[list[str]] = []

    def fake_run(command, **kwargs):
        seen.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout='{"ok": true}', stderr="")

    monkeypatch.setattr(audio_module.subprocess, "run", fake_run)
    _service(tmp_path, research=False).install_minimum()
    assert "--commercial-only" in seen[0]
    seen.clear()
    _service(tmp_path, research=True).install_minimum()
    assert "--commercial-only" not in seen[0]


# --- the Pro API ----------------------------------------------------------------------------------
def _pro_client(tmp_path: Path, *, research: bool):
    from test_pro_api import _client, _ctx

    ctx = _ctx(tmp_path)
    ctx.settings.allow_noncommercial_audio_models = research
    saved = []
    ctx.save_settings = lambda: saved.append(ctx.settings.allow_noncommercial_audio_models)
    return ctx, saved, _client(ctx)


def test_pro_generate_refuses_musicgen_with_403_in_commercial_mode(tmp_path: Path) -> None:
    _, _, client = _pro_client(tmp_path, research=False)
    response = client.post("/api/pro/audio/generate", json={"prompt": "synth pulse", "kind": "music", "modelId": "facebook/musicgen-small"})
    assert response.status_code == 403 and "CC-BY-NC-4.0" in response.json()["detail"]


def test_pro_status_lists_licences_and_mode(tmp_path: Path) -> None:
    _, _, client = _pro_client(tmp_path, research=True)
    status = client.get("/api/pro/audio/status").json()
    assert status["researchMode"] is True
    by_id = {choice["id"]: choice for choice in status["models"]["music"]}
    assert by_id["facebook/musicgen-small"]["license"]["commercial"] == audio_licenses.NO
    assert by_id["acestep:1.5-turbo"]["license"]["commercial"] == audio_licenses.YES


def test_research_mode_switch_is_local_only_and_saved(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    ctx, saved, client = _pro_client(tmp_path, research=False)
    turned_on = client.post("/api/pro/audio/research-mode", json={"enabled": True})
    assert turned_on.status_code == 200 and turned_on.json()["researchMode"] is True and saved == [True]
    remote = TestClient(client.app, client=("192.168.1.40", 50000))
    assert remote.post("/api/pro/audio/research-mode", json={"enabled": False}).status_code == 403
    assert ctx.settings.allow_noncommercial_audio_models is True
