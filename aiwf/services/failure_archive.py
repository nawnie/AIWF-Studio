from __future__ import annotations

import json
import shutil
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image


@dataclass(frozen=True)
class FailureArchiveRecord:
    archive_dir: Path
    manifest_path: Path
    artifacts: tuple[Path, ...]
    status: str
    kind: str
    stage: str
    ok: bool = True
    archive_errors: tuple[str, ...] = ()


class FailureArchiveService:
    """Keeps failed and user-marked bad generations for later review.

    The archive is intentionally append-only. Generation failures should not
    delete partial outputs, and this logger should never hide the original
    exception if saving the archive itself has a problem.
    """

    def __init__(self, output_dir: Path) -> None:
        self.root = Path(output_dir) / "failures"
        self.index_path = self.root / "index.jsonl"

    def archive_failure(
        self,
        *,
        kind: str,
        stage: str,
        request: Any = None,
        error: BaseException | str | None = None,
        preview: Image.Image | None = None,
        source_path: Any = None,
        note: str | None = None,
        extra: dict[str, Any] | None = None,
        status: str = "failed",
    ) -> FailureArchiveRecord:
        created_at = _utc_now()
        archive_id = uuid4().hex[:12]
        safe_kind = _slug(kind or "generation")
        safe_stage = _slug(stage or "unknown")
        archive_dir = self.root / created_at[:10] / f"{created_at[11:19].replace(':', '')}-{safe_kind}-{archive_id}"
        manifest_path = archive_dir / "manifest.json"
        archive_errors: list[str] = []
        artifacts: list[Path] = []
        artifact_entries: list[dict[str, Any]] = []

        try:
            archive_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            return FailureArchiveRecord(
                archive_dir=archive_dir,
                manifest_path=manifest_path,
                artifacts=(),
                status=status,
                kind=safe_kind,
                stage=safe_stage,
                ok=False,
                archive_errors=(f"mkdir failed: {exc}",),
            )

        if preview is not None:
            preview_path = archive_dir / "preview.png"
            try:
                preview.save(preview_path)
                artifacts.append(preview_path)
                artifact_entries.append({"kind": "preview_image", "path": preview_path.name})
            except Exception as exc:
                archive_errors.append(f"preview save failed: {exc}")

        src = _coerce_path(source_path)
        if src is not None and src.is_file():
            try:
                copied = _unique_path(archive_dir / f"source{src.suffix.lower() or '.bin'}")
                shutil.copy2(src, copied)
                artifacts.append(copied)
                artifact_entries.append(
                    {
                        "kind": "source_file",
                        "path": copied.name,
                        "source_name": src.name,
                    }
                )
            except Exception as exc:
                archive_errors.append(f"source copy failed: {exc}")
        elif source_path:
            archive_errors.append(f"source file missing: {source_path}")

        manifest = {
            "id": archive_id,
            "created_at": created_at,
            "status": status,
            "kind": safe_kind,
            "stage": safe_stage,
            "note": note or "",
            "request": _json_safe_request(request),
            "error": _error_payload(error),
            "artifacts": artifact_entries,
            "extra": _json_safe(extra or {}),
            "archive_errors": archive_errors,
        }

        ok = not archive_errors
        try:
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        except Exception as exc:
            archive_errors.append(f"manifest write failed: {exc}")
            ok = False

        try:
            self.root.mkdir(parents=True, exist_ok=True)
            with self.index_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(manifest, sort_keys=True) + "\n")
        except Exception as exc:
            archive_errors.append(f"index append failed: {exc}")
            ok = False

        return FailureArchiveRecord(
            archive_dir=archive_dir,
            manifest_path=manifest_path,
            artifacts=tuple(artifacts),
            status=status,
            kind=safe_kind,
            stage=safe_stage,
            ok=ok,
            archive_errors=tuple(archive_errors),
        )

    def archive_bad_image(
        self,
        image: Image.Image | None,
        *,
        infotext: str = "",
        note: str = "",
        request: Any = None,
    ) -> FailureArchiveRecord:
        return self.archive_failure(
            kind="image",
            stage="manual_bad_result",
            request=request,
            preview=image,
            note=note,
            extra={"infotext": infotext},
            status="bad_result",
        )

    def archive_bad_video(
        self,
        video_path: Any,
        *,
        note: str = "",
        request: Any = None,
        extra: dict[str, Any] | None = None,
    ) -> FailureArchiveRecord:
        return self.archive_failure(
            kind="video",
            stage="manual_bad_result",
            request=request,
            source_path=video_path,
            note=note,
            extra=extra,
            status="bad_result",
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _slug(value: str) -> str:
    chars = []
    for char in str(value).strip().lower():
        if char.isalnum():
            chars.append(char)
        elif char in {"-", "_", ".", " "}:
            chars.append("-")
    slug = "".join(chars).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "generation"


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(1, 10_000):
        candidate = path.with_name(f"{stem}-{index}{suffix}")
        if not candidate.exists():
            return candidate
    return path.with_name(f"{stem}-{uuid4().hex[:8]}{suffix}")


def _coerce_path(value: Any) -> Path | None:
    if value is None:
        return None
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        return Path(value).expanduser()
    if isinstance(value, dict):
        for key in ("path", "name", "value"):
            nested = _coerce_path(value.get(key))
            if nested is not None:
                return nested
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            nested = _coerce_path(item)
            if nested is not None:
                return nested
        return None
    path_attr = getattr(value, "path", None) or getattr(value, "name", None)
    if path_attr:
        return Path(str(path_attr)).expanduser()
    return None


def _json_safe_request(request: Any) -> Any:
    if request is None:
        return None
    dump = getattr(request, "model_dump", None)
    if callable(dump):
        try:
            return _json_safe(dump(mode="json"))
        except TypeError:
            return _json_safe(dump())
        except Exception:
            return repr(request)
    return _json_safe(request)


def _error_payload(error: BaseException | str | None) -> dict[str, Any] | None:
    if error is None:
        return None
    if isinstance(error, BaseException):
        return {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__))[-6000:],
        }
    return {"type": "Error", "message": str(error), "traceback": ""}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return _json_safe(dump(mode="json"))
        except TypeError:
            return _json_safe(dump())
        except Exception:
            return repr(value)
    return repr(value)
