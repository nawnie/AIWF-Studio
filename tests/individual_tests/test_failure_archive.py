from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from aiwf.core.domain.generation import GenerationRequest
from aiwf.services.failure_archive import FailureArchiveService


def test_failure_archive_saves_manifest_preview_source_and_index(tmp_path: Path):
    source = tmp_path / "bad.mp4"
    source.write_bytes(b"video")
    service = FailureArchiveService(tmp_path / "outputs")

    record = service.archive_failure(
        kind="video",
        stage="wan_video",
        request=GenerationRequest(prompt="dance", steps=4),
        error=RuntimeError("temporal fracture"),
        preview=Image.new("RGB", (8, 8), "red"),
        source_path=source,
        extra={"route": "5b"},
    )

    assert record.ok
    assert record.manifest_path.is_file()
    assert (record.archive_dir / "preview.png").is_file()
    assert (record.archive_dir / "source.mp4").read_bytes() == b"video"

    manifest = json.loads(record.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["kind"] == "video"
    assert manifest["stage"] == "wan-video"
    assert manifest["request"]["prompt"] == "dance"
    assert manifest["error"]["message"] == "temporal fracture"
    assert manifest["extra"]["route"] == "5b"

    index_lines = (tmp_path / "outputs" / "failures" / "index.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(index_lines) == 1
    assert json.loads(index_lines[0])["id"] == manifest["id"]


def test_failure_archive_can_mark_bad_image(tmp_path: Path):
    service = FailureArchiveService(tmp_path / "outputs")

    record = service.archive_bad_image(
        Image.new("RGB", (8, 8), "blue"),
        infotext="Steps: 8, Seed: 123",
        note="funny failure",
    )

    manifest = json.loads(record.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "bad_result"
    assert manifest["stage"] == "manual-bad-result"
    assert manifest["note"] == "funny failure"
    assert manifest["extra"]["infotext"] == "Steps: 8, Seed: 123"
