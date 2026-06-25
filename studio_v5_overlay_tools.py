#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

UPDATE_ID = "bbb1cae-studio-v5"
MANIFEST = "STUDIO-V5-OVERLAY.json"
ARCHIVE_MANIFEST = "STUDIO-V5-ARCHIVE-MANIFEST.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".restore", dir=str(target.parent))
    os.close(fd)
    temp = Path(temp_name)
    try:
        shutil.copy2(source, temp)
        os.replace(temp, target)
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def _load(root: Path) -> dict:
    path = root / MANIFEST
    if not path.is_file():
        raise RuntimeError(f"Overlay manifest is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("update_id") != UPDATE_ID:
        raise RuntimeError(f"Unexpected overlay manifest id: {payload.get('update_id')!r}")
    return payload


def _verify_archive_manifest(root: Path, failures: list[str]) -> int:
    manifest_path = root / ARCHIVE_MANIFEST
    if not manifest_path.is_file():
        failures.append(f"archive manifest is missing: {ARCHIVE_MANIFEST}")
        return 0
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        failures.append(f"archive manifest is invalid: {exc}")
        return 0
    entries = payload.get("files") or {}
    forbidden = ("frontend/", "aiwf/web/modern/", "aiwf/web/pro", "webui_modern", "webui_pro")
    for relative, expected in entries.items():
        normalized = str(relative).replace("\\", "/")
        if normalized.startswith("/") or "../" in f"/{normalized}/":
            failures.append(f"unsafe archive path: {relative}")
            continue
        if any(part in normalized for part in forbidden):
            failures.append(f"forbidden GUI-drift archive path: {relative}")
            continue
        if "/.venv/" in f"/{normalized}/" or normalized.endswith("/.venv"):
            failures.append(f"bundled virtual environment is forbidden: {relative}")
            continue
        path = root / normalized
        if not path.is_file():
            failures.append(f"archive payload missing: {relative}")
            continue
        expected_hash = expected.get("sha256") if isinstance(expected, dict) else None
        expected_size = expected.get("size") if isinstance(expected, dict) else None
        if expected_hash and _sha256(path) != expected_hash:
            failures.append(f"archive payload hash mismatch: {relative}")
        if expected_size is not None and path.stat().st_size != int(expected_size):
            failures.append(f"archive payload size mismatch: {relative}")
    return len(entries)


def verify(root: Path) -> None:
    payload = _load(root)
    failures: list[str] = []
    archive_count = _verify_archive_manifest(root, failures)
    forbidden = ("frontend/", "aiwf/web/modern/", "aiwf/web/pro", "webui_modern", "webui_pro")
    for relative, expected in (payload.get("overlay_files") or {}).items():
        normalized = relative.replace("\\", "/")
        if any(part in normalized for part in forbidden):
            failures.append(f"forbidden GUI-drift path in manifest: {relative}")
            continue
        if "/.venv/" in f"/{normalized}/" or normalized.endswith("/.venv"):
            failures.append(f"bundled virtual environment is forbidden: {relative}")
            continue
        path = root / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        actual = _sha256(path)
        if actual != expected:
            failures.append(f"hash mismatch: {relative}")
    required_markers = {
        "aiwf/web/tabs/image_workflow.py": "Processes to run",
        "aiwf/web/tabs/video_lab.py": "Processes to run",
        "aiwf/web/tabs/audio.py": "Install / Repair Audio Lab Engine",
        "aiwf/core/domain/segment_presets.py": "feather",
        "engines/audio_lab/runner.py": "process_job",
    }
    for relative, marker in required_markers.items():
        path = root / relative
        if not path.is_file() or marker not in path.read_text(encoding="utf-8", errors="ignore"):
            failures.append(f"required v5 marker missing: {relative} -> {marker}")
    if failures:
        raise RuntimeError("Studio v5 overlay verification failed:\n- " + "\n- ".join(failures))
    print(
        f"Studio v5 overlay verified: {len(payload['overlay_files'])} overlay file(s), "
        f"{archive_count} archive payload file(s), original Gradio Studio only, "
        "no bundled audio environment."
    )


def rollback(root: Path) -> None:
    payload = _load(root)
    backup_root = root / "AIWF_UPDATE_BACKUP"
    restored = 0
    for relative in payload.get("restore_files") or []:
        source = backup_root / relative
        target = root / relative
        if not source.is_file():
            raise RuntimeError(f"Static rollback backup is missing: {source}")
        _atomic_copy(source, target)
        restored += 1
    removed = 0
    for relative in payload.get("remove_on_rollback") or []:
        path = root / relative
        if path.is_file():
            path.unlink()
            removed += 1
    # Do not remove an installed Audio Lab environment automatically. It is
    # optional user-created state, not source overlay data.
    for relative in sorted(payload.get("prune_empty_dirs") or [], reverse=True):
        path = root / relative
        try:
            path.rmdir()
        except OSError:
            pass
    print(f"Studio v5 overlay rollback restored {restored} file(s) and removed {removed} new file(s).")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("verify", "rollback"))
    parser.add_argument("repo", nargs="?", default=".")
    args = parser.parse_args()
    root = Path(args.repo).expanduser().resolve()
    try:
        if args.action == "verify":
            verify(root)
        else:
            rollback(root)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
