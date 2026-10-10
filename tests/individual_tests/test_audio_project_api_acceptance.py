from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from aiwf.core.domain.audio import AudioGenerationOptions
from aiwf.services import audio_licenses
from aiwf.services.audio_project import AudioProjectService
from test_pro_api import _AudioStub, _client, _ctx


def _audio_context(tmp_path: Path):
    ctx = _ctx(tmp_path)
    audio = _AudioStub(ctx.flags.output_dir)
    audio.installed = True
    # The real service factory reads the same flags/settings attributes as
    # AudioGenerationService; this stub only writes fixture bytes.
    audio.flags = ctx.flags
    audio.settings = ctx.settings
    ctx.audio = audio
    return ctx, audio, _client(ctx)


def _generate_fixture(client):
    response = client.post(
        "/api/pro/audio/generate",
        json={
            "prompt": "fixture-only soft piano",
            "kind": "music",
            "modelId": "facebook/musicgen-small",
            "durationSeconds": 8,
            "temperature": 1,
            "cfgCoef": 3,
            "topK": 250,
            "steps": 25,
            "seed": 731,
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["url"].startswith("/api/pro/outputs/")
    return response.json()


def test_generated_fixture_connects_real_route_to_project_service_and_recovery(tmp_path: Path):
    ctx, audio, client = _audio_context(tmp_path)
    generated = _generate_fixture(client)
    expected_license = audio_licenses.license_for(generated["modelId"])
    assert generated["license"] == expected_license
    source = Path(generated["outputPath"])
    source_bytes = source.read_bytes()
    service = AudioProjectService.from_audio_service(audio)
    options = AudioGenerationOptions(
        prompt=generated["prompt"],
        kind=generated["kind"],
        model_id=generated["modelId"],
        duration_seconds=generated["durationSeconds"],
        temperature=1,
        cfg_coef=3,
        top_k=250,
        steps=25,
        seed=731,
    )

    manifest = service.save_project(
        name="Route fixture",
        options=options,
        audio_path=source,
        sample_rate=generated["sampleRate"],
    )
    assert manifest.track.license == expected_license
    assert manifest.track.license_notice == audio_licenses.notice_for(generated["modelId"])
    asset = service.project_root / manifest.project_id / manifest.track.asset_ref
    assert asset.is_file()
    assert asset.read_bytes() == source_bytes
    assert source.read_bytes() == source_bytes

    project_dir = service.project_root / manifest.project_id
    primary = project_dir / "project.json"
    pending = project_dir / f".project.json.{uuid.uuid4().hex}.tmp"
    pending.write_bytes(primary.read_bytes())
    primary.unlink()
    recovered = service.load_project(manifest.project_id)
    assert recovered == manifest
    assert primary.is_file() and not pending.exists()

    unrelated = tmp_path / "unrelated-user-audio.wav"
    unrelated.write_bytes(b"preserve user file")
    with pytest.raises(ValueError, match="approved Studio audio output folder"):
        service.save_project(name="Rejected", options=options, audio_path=unrelated)
    assert unrelated.read_bytes() == b"preserve user file"
    assert source.read_bytes() == source_bytes


def test_audio_project_http_routes_save_load_recover_and_enforce_source_boundary(tmp_path: Path):
    ctx, audio, client = _audio_context(tmp_path)
    required_routes = {
        ("GET", "/api/pro/audio/projects"),
        ("POST", "/api/pro/audio/projects"),
        ("GET", "/api/pro/audio/projects/{project_id}"),
    }
    openapi_paths = client.app.openapi()["paths"]
    actual_routes = {
        (method.upper(), path)
        for path, operations in openapi_paths.items()
        for method in operations
    }
    assert ("POST", "/api/pro/audio/generate") in actual_routes
    assert required_routes.issubset(actual_routes), f"Missing audio project routes: {required_routes - actual_routes}"

    generated = _generate_fixture(client)
    expected_license = audio_licenses.license_for(generated["modelId"])
    assert generated["license"] == expected_license
    source = Path(generated["outputPath"])
    source_bytes = source.read_bytes()
    options = AudioGenerationOptions(
        prompt=generated["prompt"],
        kind=generated["kind"],
        model_id=generated["modelId"],
        duration_seconds=generated["durationSeconds"],
        temperature=1,
        cfg_coef=3,
        top_k=250,
        steps=25,
        seed=731,
    )
    request = {
        "name": "HTTP fixture project",
        "project_id": None,
        "audio_path": generated["outputPath"],
        "options": options.model_dump(mode="json"),
        "sample_rate": generated["sampleRate"],
        "license_notice": audio_licenses.notice_for(generated["modelId"]),
        "license": generated["license"],
        "consent_status": None,
    }
    saved_response = client.post("/api/pro/audio/projects", json=request)
    assert saved_response.status_code == 200, saved_response.text
    saved = saved_response.json()
    assert saved["project_id"]
    assert saved["track"]["sample_rate"] == generated["sampleRate"]
    assert saved["track"]["license"] == expected_license
    assert saved["track"]["license_notice"] == audio_licenses.notice_for(generated["modelId"])
    assert saved["audio_url"].startswith("/api/pro/outputs/")
    assert client.get(saved["audio_url"]).content == source_bytes
    assert source.read_bytes() == source_bytes

    listing = client.get("/api/pro/audio/projects")
    assert listing.status_code == 200, listing.text
    assert listing.json()["projects"][0]["project_id"] == saved["project_id"]
    assert listing.json()["projects"][0]["has_audio"] is True

    service = AudioProjectService.from_audio_service(audio)
    project_dir = service.project_root / saved["project_id"]
    primary = project_dir / "project.json"
    pending = project_dir / f".project.json.{uuid.uuid4().hex}.tmp"
    pending.write_bytes(primary.read_bytes())
    primary.unlink()
    loaded_response = client.get(f"/api/pro/audio/projects/{saved['project_id']}")
    assert loaded_response.status_code == 200, loaded_response.text
    assert loaded_response.json()["project_id"] == saved["project_id"]
    assert loaded_response.json()["track"]["license"] == expected_license
    assert loaded_response.json()["track"]["license_notice"] == saved["track"]["license_notice"]
    assert primary.is_file() and not pending.exists()
    assert client.get(loaded_response.json()["audio_url"]).content == source_bytes

    unrelated = tmp_path / "external-user-audio.wav"
    unrelated.write_bytes(b"preserve user file")
    rejected = client.post(
        "/api/pro/audio/projects",
        json={**request, "name": "Rejected external path", "project_id": None, "audio_path": str(unrelated)},
    )
    assert rejected.status_code == 422, rejected.text
    assert unrelated.read_bytes() == b"preserve user file"
    assert source.read_bytes() == source_bytes

    asset = project_dir / loaded_response.json()["track"]["asset_ref"]
    asset.unlink()
    missing = client.get(f"/api/pro/audio/projects/{saved['project_id']}")
    assert missing.status_code == 409, missing.text
    missing_row = next(
        row for row in client.get("/api/pro/audio/projects").json()["projects"]
        if row["project_id"] == saved["project_id"]
    )
    assert missing_row["audio_missing"] is True


@pytest.mark.parametrize("mismatch", ["model", "notice"])
def test_audio_project_http_rejects_inconsistent_license_attribution(tmp_path: Path, mismatch: str):
    ctx, _audio, client = _audio_context(tmp_path)
    generated = _generate_fixture(client)
    # Simulate changing the model selector to ACE-Step after the MusicGen render.
    options = AudioGenerationOptions(
        prompt=generated["prompt"],
        kind=generated["kind"],
        model_id="acestep:1.5-turbo" if mismatch == "model" else generated["modelId"],
        duration_seconds=generated["durationSeconds"],
    )
    license_notice = (
        audio_licenses.notice_for(generated["modelId"])
        if mismatch == "model"
        else audio_licenses.notice_for("acestep:1.5-turbo")
    )

    response = client.post(
        "/api/pro/audio/projects",
        json={
            "name": "Mismatched license fixture",
            "audio_path": generated["outputPath"],
            "options": options.model_dump(mode="json"),
            "sample_rate": generated["sampleRate"],
            "license_notice": license_notice,
            "license": generated["license"],
        },
    )

    assert response.status_code == 422
    assert "license" in response.json()["detail"].lower()
    assert client.get("/api/pro/audio/projects").json()["projects"] == []


def test_audio_project_http_keeps_unknown_artifact_unattributed(tmp_path: Path):
    ctx, _audio, client = _audio_context(tmp_path)
    source = ctx.flags.output_dir / "audio" / "unknown-fixture.wav"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"unknown audio fixture")
    options = AudioGenerationOptions(prompt="unknown fixture", model_id="custom:unknown")

    response = client.post(
        "/api/pro/audio/projects",
        json={
            "name": "Unknown artifact fixture",
            "audio_path": str(source),
            "options": options.model_dump(mode="json"),
            "sample_rate": 32000,
            "license_notice": None,
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["track"].get("license") is None
    assert response.json()["track"].get("license_notice") is None
