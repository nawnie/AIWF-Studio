"""Commercial-safe audio by default (aiwf/services/audio_licenses.py and its enforcement).

AIWF Studio is sold commercially, so out of the box it must not offer or run audio models
whose licences forbid commercial use (MusicGen and MMAudio are CC-BY-NC 4.0). Research mode
brings them back, labeled. These tests pin that promise at every layer: the licence registry,
the audio service (pickers, generate, prepare, install), and the Pro API routes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.audio import AudioGenerationOptions
from aiwf.services import audio_licenses
from aiwf.services.audio import AudioGenerationService, AudioLicenseBlocked


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
    assert service.video_audio_model_choices() == []


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
    assert status["defaults"] == {"music": "acestep:1.5-turbo", "sfx": "moss-sfx:v2.0", "videoAudio": ""}
    assert status["licenses"]["facebook/musicgen-small"]["commercial"] == audio_licenses.NO
    research = _service(tmp_path, research=True).setup_status(deep=False)
    assert research["researchMode"] is True and research["defaults"]["videoAudio"] == "mmaudio:small_16k"


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
