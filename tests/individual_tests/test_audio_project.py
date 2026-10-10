from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import pytest

from aiwf.core.domain.audio import AudioGenerationOptions
from aiwf.services import audio_licenses
from aiwf.services.audio_project import (
    AudioProjectAssetMissing,
    AudioProjectCorrupt,
    AudioProjectRecoveryError,
    AudioProjectService,
)


@pytest.fixture
def project_paths(tmp_path: Path) -> tuple[AudioProjectService, Path, Path]:
    sources = tmp_path / "outputs" / "audio"
    projects = tmp_path / "outputs" / "audio-projects"
    sources.mkdir(parents=True)
    service = AudioProjectService(projects, source_roots=[sources])
    return service, sources, projects


def test_round_trip_preserves_options_asset_and_existing_labels(project_paths):
    service, source_root, _ = project_paths
    source = source_root / "render.wav"
    source.write_bytes(b"fixture audio bytes")
    options = AudioGenerationOptions(prompt="soft piano", seed=44)

    saved = service.save_project(
        name="Idea 1",
        options=options,
        audio_path=source,
        license_notice=audio_licenses.notice_for(options.model_id),
        consent_status="user-confirmed",
        sample_rate=44100,
    )
    loaded = service.load_project(saved.project_id)
    saved_asset = (Path(project_paths[2]) / saved.project_id / loaded.track.asset_ref)

    assert loaded == saved
    assert loaded.options.prompt == "soft piano"
    assert loaded.options.seed == 44
    assert loaded.track.license_notice == audio_licenses.notice_for(options.model_id)
    assert loaded.track.consent_status == "user-confirmed"
    assert loaded.track.sample_rate == 44100
    assert saved_asset.read_bytes() == b"fixture audio bytes"
    assert source.read_bytes() == b"fixture audio bytes"


def test_save_without_explicit_license_notice_persists_authoritative_license(project_paths):
    service, source_root, projects = project_paths
    source = source_root / "render.flac"
    source.write_bytes(b"audio")
    saved = service.save_project(name="No labels", options=AudioGenerationOptions(), audio_path=source)
    manifest_text = (projects / saved.project_id / "project.json").read_text(encoding="utf-8")

    expected_license = audio_licenses.license_for("facebook/musicgen-small")
    assert saved.track.license == expected_license
    assert saved.track.license_notice == audio_licenses.notice_for("facebook/musicgen-small")
    assert json.loads(manifest_text)["track"]["license"] == expected_license
    assert "consent_status" not in manifest_text


@pytest.mark.parametrize("model_id", ["facebook/musicgen-small", "custom:unknown"])
def test_service_rejects_mismatched_notice_when_deriving_license(project_paths, model_id):
    service, source_root, projects = project_paths
    source = source_root / "render.wav"
    source.write_bytes(b"audio")

    with pytest.raises(ValueError, match="license notice does not match"):
        service.save_project(
            name="Mismatched notice",
            options=AudioGenerationOptions(model_id=model_id),
            audio_path=source,
            license_notice=audio_licenses.notice_for("acestep:default"),
        )
    assert service.list_projects() == []
    assert not list(projects.rglob("project.json"))


@pytest.mark.parametrize("model_id", ["facebook/musicgen-small", "custom:unknown"])
def test_service_accepts_matching_notice_when_deriving_license(project_paths, model_id):
    service, source_root, _ = project_paths
    source = source_root / "render.wav"
    source.write_bytes(b"audio")
    saved = service.save_project(
        name="Matching notice",
        options=AudioGenerationOptions(model_id=model_id),
        audio_path=source,
        license_notice=audio_licenses.notice_for(model_id),
    )
    loaded = service.load_project(saved.project_id)
    assert loaded.track.license == audio_licenses.license_for(model_id)
    assert loaded.track.license_notice == audio_licenses.notice_for(model_id)


def test_load_legacy_project_without_structured_license_record(project_paths):
    service, source_root, projects = project_paths
    source = source_root / "legacy.wav"
    source.write_bytes(b"audio")
    saved = service.save_project(name="Legacy", options=AudioGenerationOptions(), audio_path=source)
    manifest_path = projects / saved.project_id / "project.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["track"].pop("license")
    manifest["track"].pop("license_notice")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = service.load_project(saved.project_id)

    assert loaded.track.license is None
    assert loaded.track.license_notice is None
    asset = projects / saved.project_id / loaded.track.asset_ref
    resaved = service.save_project(
        name="Legacy",
        project_id=saved.project_id,
        options=loaded.options,
        audio_path=asset,
        license=None,
        license_notice=None,
    )
    assert resaved.track.license is None
    assert resaved.track.license_notice is None


def test_repeated_load_is_read_only_and_repeated_save_reuses_project_asset(project_paths):
    service, source_root, projects = project_paths
    source = source_root / "render.wav"
    source.write_bytes(b"audio")
    first = service.save_project(name="Repeat", options=AudioGenerationOptions(), audio_path=source)
    asset = projects / first.project_id / first.track.asset_ref
    before = asset.read_bytes()

    loaded_twice = service.load_project(first.project_id)
    loaded_thrice = service.load_project(first.project_id)
    updated = service.save_project(
        name="Repeat",
        options=AudioGenerationOptions(prompt="updated"),
        audio_path=asset,
        project_id=first.project_id,
    )

    assert loaded_twice == loaded_thrice == first
    assert updated.track.asset_ref == first.track.asset_ref
    assert updated.options.prompt == "updated"
    assert asset.read_bytes() == before
    assert len(list((projects / first.project_id / "assets").iterdir())) == 1


def test_missing_project_audio_reports_clear_error(project_paths):
    service, source_root, projects = project_paths
    source = source_root / "render.wav"
    source.write_bytes(b"audio")
    saved = service.save_project(name="Missing", options=AudioGenerationOptions(), audio_path=source)
    (projects / saved.project_id / saved.track.asset_ref).unlink()

    with pytest.raises(AudioProjectAssetMissing, match="missing audio"):
        service.load_project(saved.project_id)
    assert service.list_projects()[0]["has_audio"] is False


def test_malformed_schema_version_is_rejected(project_paths):
    service, _, projects = project_paths
    saved = service.save_project(name="Version", options=AudioGenerationOptions())
    path = projects / saved.project_id / "project.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AudioProjectCorrupt, match="Unsupported or malformed"):
        service.load_project(saved.project_id)


def test_manifest_path_traversal_is_rejected(project_paths):
    service, _, projects = project_paths
    saved = service.save_project(name="Traversal", options=AudioGenerationOptions())
    path = projects / saved.project_id / "project.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["track"] = {
        "track_id": uuid.uuid4().hex,
        "asset_ref": "assets/../../outside.wav",
        "source_name": "outside.wav",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AudioProjectCorrupt, match="Unsupported or malformed"):
        service.load_project(saved.project_id)


def test_project_id_path_traversal_is_rejected(project_paths):
    service, _, _ = project_paths
    with pytest.raises(ValueError, match="32-character hexadecimal"):
        service.load_project("../other-project")


def test_pending_manifest_is_recovered_only_after_validation(project_paths):
    service, _, projects = project_paths
    saved = service.save_project(name="Recovery", options=AudioGenerationOptions())
    project_dir = projects / saved.project_id
    pending = project_dir / f".project.json.{uuid.uuid4().hex}.tmp"
    pending.write_text(saved.model_dump_json(exclude_none=True), encoding="utf-8")
    (project_dir / "project.json").unlink()

    loaded = service.load_project(saved.project_id)

    assert loaded == saved
    assert (project_dir / "project.json").is_file()
    assert not pending.exists()


def test_ambiguous_recovery_leaves_pending_files_untouched(project_paths):
    service, _, projects = project_paths
    saved = service.save_project(name="Ambiguous", options=AudioGenerationOptions())
    project_dir = projects / saved.project_id
    (project_dir / "project.json").unlink()
    first = project_dir / f".project.json.{uuid.uuid4().hex}.tmp"
    second = project_dir / f".project.json.{uuid.uuid4().hex}.tmp"
    first.write_text(saved.model_dump_json(exclude_none=True), encoding="utf-8")
    second.write_text(saved.model_dump_json(exclude_none=True), encoding="utf-8")

    with pytest.raises(AudioProjectRecoveryError, match="More than one"):
        service.load_project(saved.project_id)
    assert first.exists() and second.exists()
    assert not (project_dir / "project.json").exists()


def test_external_or_arbitrary_input_is_not_copied(project_paths, tmp_path: Path):
    service, _, projects = project_paths
    unrelated = tmp_path / "user.wav"
    unrelated.write_bytes(b"keep exactly")

    with pytest.raises(ValueError, match="approved Studio audio output folder"):
        service.save_project(name="Refused", options=AudioGenerationOptions(), audio_path=unrelated)
    assert unrelated.read_bytes() == b"keep exactly"
    assert not list(projects.iterdir())


def test_save_copies_to_a_new_asset_path_without_touching_source(project_paths):
    service, source_root, projects = project_paths
    source = source_root / "same-name.wav"
    source.write_bytes(b"user output")

    saved = service.save_project(name="Copy", options=AudioGenerationOptions(), audio_path=source)
    asset = projects / saved.project_id / saved.track.asset_ref

    assert asset != source
    assert asset.read_bytes() == source.read_bytes()
    assert source.read_bytes() == b"user output"
    assert source.exists()


def test_asset_name_collision_does_not_replace_existing_file(project_paths, monkeypatch):
    from types import SimpleNamespace

    import aiwf.services.audio_project as audio_project_module

    service, source_root, projects = project_paths
    source = source_root / "render.wav"
    source.write_bytes(b"new generated audio")
    saved = service.save_project(name="Collision", options=AudioGenerationOptions())
    assets = projects / saved.project_id / "assets"
    assets.mkdir()
    asset_id = "a" * 32
    existing = assets / f"{asset_id}.wav"
    existing.write_bytes(b"unrelated existing asset")
    manifest_path = projects / saved.project_id / "project.json"
    original_manifest = manifest_path.read_bytes()
    monkeypatch.setattr(audio_project_module.uuid, "uuid4", lambda: SimpleNamespace(hex=asset_id))

    with pytest.raises(FileExistsError):
        service.save_project(
            name="Collision",
            options=AudioGenerationOptions(),
            audio_path=source,
            project_id=saved.project_id,
        )
    assert existing.read_bytes() == b"unrelated existing asset"
    assert manifest_path.read_bytes() == original_manifest


def test_asset_directory_symlink_cannot_redirect_writes(project_paths, tmp_path: Path):
    service, source_root, projects = project_paths
    source = source_root / "render.wav"
    source.write_bytes(b"audio")
    saved = service.save_project(name="Redirected", options=AudioGenerationOptions())
    project_dir = projects / saved.project_id
    outside = tmp_path / "outside-assets"
    outside.mkdir()
    assets = project_dir / "assets"
    try:
        os.symlink(outside, assets, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"directory symlink creation is unavailable: {exc}")

    with pytest.raises(AudioProjectCorrupt, match="symbolic links or junctions"):
        service.save_project(
            name="Redirected",
            options=AudioGenerationOptions(),
            audio_path=source,
            project_id=saved.project_id,
        )
    assert list(outside.iterdir()) == []
    assert source.read_bytes() == b"audio"


def test_asset_directory_replacement_before_open_is_detected(project_paths, tmp_path: Path, monkeypatch):
    import aiwf.services.audio_project as audio_project_module

    service, source_root, projects = project_paths
    source = source_root / "render.wav"
    source.write_bytes(b"audio")
    saved = service.save_project(name="Race", options=AudioGenerationOptions())
    project_dir = projects / saved.project_id
    assets = project_dir / "assets"
    assets.mkdir()
    outside = tmp_path / "outside-assets"
    outside.mkdir()
    asset_id = uuid.uuid4().hex
    destination = assets / f"{asset_id}.wav"
    destination.write_bytes(b"preserve unrelated asset")
    real_open = audio_project_module._open_exclusive_asset

    def replace_before_open(path):
        if Path(path) == destination:
            original_assets = assets.with_name("assets-original")
            assets.rename(original_assets)
            os.symlink(outside, assets, target_is_directory=True)
            fd = real_open(path)
            assets.unlink()
            original_assets.rename(assets)
            return fd
        return real_open(path)

    resolve_opened_path = audio_project_module._opened_file_path
    inspections = 0

    def count_inspection(fd):
        nonlocal inspections
        inspections += 1
        return resolve_opened_path(fd)

    monkeypatch.setattr(audio_project_module.uuid, "uuid4", lambda: type("Id", (), {"hex": asset_id})())
    monkeypatch.setattr(audio_project_module, "_open_exclusive_asset", replace_before_open)
    monkeypatch.setattr(audio_project_module, "_opened_file_path", count_inspection)
    with pytest.raises(AudioProjectCorrupt, match="destination changed during the save|Could not verify the opened"):
        service.save_project(
            name="Race",
            options=AudioGenerationOptions(),
            audio_path=source,
            project_id=saved.project_id,
        )
    if os.name == "nt":
        assert list(outside.iterdir()) == []
    else:
        leftovers = list(outside.iterdir())
        assert len(leftovers) == 1 and leftovers[0].stat().st_size == 0
    assert inspections == 1
    assert source.read_bytes() == b"audio"
    assert destination.read_bytes() == b"preserve unrelated asset"


def test_reused_asset_update_validates_metadata_before_manifest_write(project_paths):
    service, source_root, projects = project_paths
    source = source_root / "render.wav"
    source.write_bytes(b"audio")
    saved = service.save_project(name="Validated", options=AudioGenerationOptions(), audio_path=source)
    asset = projects / saved.project_id / saved.track.asset_ref
    original_manifest = (projects / saved.project_id / "project.json").read_bytes()

    with pytest.raises(ValueError):
        service.save_project(
            name="Validated",
            options=AudioGenerationOptions(prompt="change"),
            audio_path=asset,
            project_id=saved.project_id,
            sample_rate=384001,
        )

    assert (projects / saved.project_id / "project.json").read_bytes() == original_manifest
    assert service.load_project(saved.project_id) == saved


def test_invalid_track_metadata_does_not_leave_copied_asset(project_paths):
    service, source_root, projects = project_paths
    source = source_root / "render.wav"
    source.write_bytes(b"audio")

    with pytest.raises(ValueError):
        service.save_project(
            name="Invalid sample rate",
            options=AudioGenerationOptions(),
            audio_path=source,
            sample_rate=-1,
        )
    assert list(projects.iterdir()) == []


def test_failed_post_publish_temp_cleanup_keeps_manifest_and_audio(project_paths, monkeypatch):
    service, source_root, projects = project_paths
    source = source_root / "render.wav"
    source.write_bytes(b"audio")
    original_unlink = Path.unlink

    def fail_pending_unlink(path: Path, *args, **kwargs):
        if path.name.startswith(".project.json.") and path.name.endswith(".tmp"):
            raise PermissionError("simulated temp cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_pending_unlink)
    saved = service.save_project(name="Committed", options=AudioGenerationOptions(), audio_path=source)
    loaded = service.load_project(saved.project_id)

    assert loaded == saved
    assert (projects / saved.project_id / saved.track.asset_ref).read_bytes() == b"audio"


def test_pending_manifest_symlink_is_rejected(project_paths, tmp_path: Path):
    service, _, projects = project_paths
    saved = service.save_project(name="Linked pending", options=AudioGenerationOptions())
    project_dir = projects / saved.project_id
    (project_dir / "project.json").unlink()
    outside = tmp_path / "outside.json"
    outside.write_text(saved.model_dump_json(exclude_none=True), encoding="utf-8")
    pending = project_dir / f".project.json.{uuid.uuid4().hex}.tmp"
    try:
        os.symlink(outside, pending)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"file symlink creation is unavailable: {exc}")

    with pytest.raises(AudioProjectRecoveryError, match="regular files"):
        service.load_project(saved.project_id)
    assert outside.is_file()
    assert not (project_dir / "project.json").exists()


def test_audio_service_factory_uses_configured_audio_output(tmp_path: Path):
    from types import SimpleNamespace

    output_root = tmp_path / "outputs"
    audio_root = output_root / "custom-audio"
    audio_root.mkdir(parents=True)
    generated = audio_root / "clip.wav"
    generated.write_bytes(b"audio")
    audio_service = SimpleNamespace(
        flags=SimpleNamespace(resolved_output_dir=lambda: output_root),
        settings=SimpleNamespace(audio_output_subdir="custom-audio"),
    )

    service = AudioProjectService.from_audio_service(audio_service)
    saved = service.save_project(name="Configured", options=AudioGenerationOptions(), audio_path=generated)

    assert saved.track.source_name == "clip.wav"
    assert (output_root / "audio-projects" / saved.project_id / saved.track.asset_ref).is_file()


def test_manifest_failure_does_not_delete_replacement_directory_asset(project_paths, monkeypatch):
    service, source_root, projects = project_paths
    source = source_root / "render.wav"
    source.write_bytes(b"new generated audio")
    saved = service.save_project(name="Manifest failure race", options=AudioGenerationOptions())
    project_dir = projects / saved.project_id
    original_manifest = (project_dir / "project.json").read_bytes()
    assets = project_dir / "assets"
    moved_assets = project_dir / "assets-original"

    def replace_assets_then_fail(project_dir_arg, manifest, *, create_only):
        assert project_dir_arg == project_dir
        assert create_only is False
        asset_name = Path(manifest.track.asset_ref).name
        assets.rename(moved_assets)
        assets.mkdir()
        (assets / asset_name).write_bytes(b"unrelated replacement sentinel")
        raise OSError("simulated manifest publication failure")

    monkeypatch.setattr(service, "_write_manifest", replace_assets_then_fail)
    with pytest.raises(OSError, match="manifest publication failure"):
        service.save_project(
            name="Manifest failure race",
            options=AudioGenerationOptions(),
            audio_path=source,
            project_id=saved.project_id,
        )

    asset_name = next(moved_assets.iterdir()).name
    assert (assets / asset_name).read_bytes() == b"unrelated replacement sentinel"
    assert (moved_assets / asset_name).read_bytes() == b"new generated audio"
    assert (project_dir / "project.json").read_bytes() == original_manifest
    assert source.read_bytes() == b"new generated audio"


def test_partial_copy_cleanup_removes_its_unchanged_asset(project_paths, monkeypatch):
    import aiwf.services.audio_project as audio_project_module

    service, source_root, projects = project_paths
    source = source_root / "render.wav"
    source.write_bytes(b"new generated audio")
    saved = service.save_project(name="Interrupted copy", options=AudioGenerationOptions())
    assets = projects / saved.project_id / "assets"

    def fail_after_partial_copy(_source, destination_stream):
        destination_stream.write(b"partial copied bytes")
        raise OSError("simulated interrupted copy")

    monkeypatch.setattr(audio_project_module.shutil, "copyfileobj", fail_after_partial_copy)
    with pytest.raises(OSError, match="interrupted copy"):
        service.save_project(
            name="Interrupted copy",
            options=AudioGenerationOptions(),
            audio_path=source,
            project_id=saved.project_id,
        )

    if os.name == "nt":
        assert list(assets.iterdir()) == []
    else:
        assert len(list(assets.iterdir())) == 1
        assert next(assets.iterdir()).read_bytes() == b"partial copied bytes"
    assert source.read_bytes() == b"new generated audio"
    assert service.load_project(saved.project_id) == saved


def test_partial_copy_cleanup_does_not_delete_replacement_directory_asset(project_paths, tmp_path: Path, monkeypatch):
    from types import SimpleNamespace

    import aiwf.services.audio_project as audio_project_module

    service, source_root, projects = project_paths
    source = source_root / "render.wav"
    source.write_bytes(b"new generated audio")
    saved = service.save_project(name="Cleanup race", options=AudioGenerationOptions())
    project_dir = projects / saved.project_id
    assets = project_dir / "assets"
    assets.mkdir()
    asset_id = "b" * 32
    moved_assets = project_dir / "assets-original"
    destination = assets / f"{asset_id}.wav"
    monkeypatch.setattr(audio_project_module.uuid, "uuid4", lambda: SimpleNamespace(hex=asset_id))
    original_mark = audio_project_module._mark_open_asset_for_deletion
    deletion_attempts = 0

    def replace_before_cleanup(fd):
        nonlocal deletion_attempts
        deletion_attempts += 1
        destination.unlink()
        destination.write_bytes(b"unrelated replacement sentinel")
        return original_mark(fd)

    def fail_after_partial_copy(_source, destination_stream):
        destination_stream.write(b"partial copied bytes")
        raise OSError("simulated interrupted copy")

    monkeypatch.setattr(audio_project_module, "_mark_open_asset_for_deletion", replace_before_cleanup)
    monkeypatch.setattr(audio_project_module.shutil, "copyfileobj", fail_after_partial_copy)
    with pytest.raises(OSError, match="interrupted copy"):
        service.save_project(
            name="Cleanup race",
            options=AudioGenerationOptions(),
            audio_path=source,
            project_id=saved.project_id,
        )

    replacement_file = assets / f"{asset_id}.wav"
    if os.name == "nt":
        assert replacement_file.read_bytes() == b"unrelated replacement sentinel"
        assert deletion_attempts == 1
    else:
        assert replacement_file.read_bytes() == b"partial copied bytes"
        assert deletion_attempts == 0
    assert source.read_bytes() == b"new generated audio"
