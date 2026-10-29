from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from aiwf.core.domain.audio import AudioGenerationOptions
from aiwf.core.domain.audio_project import AudioProjectManifest, AudioProjectTrack, validate_project_id
from aiwf.services import audio_licenses


_AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav"}
_PENDING_MANIFEST = re.compile(r"^\.project\.json\.([0-9a-f]{32})\.tmp$")
_LICENSE_UNSET = object()


class AudioProjectError(RuntimeError):
    """Base error suitable for conversion to an API validation message."""


class AudioProjectNotFound(AudioProjectError):
    pass


class AudioProjectCorrupt(AudioProjectError):
    pass


class AudioProjectAssetMissing(AudioProjectError):
    pass


class AudioProjectRecoveryError(AudioProjectError):
    pass


class AudioProjectService:
    """Persist local project metadata and copied audio without replacing source files."""

    def __init__(self, project_root: str | Path, *, source_roots: Iterable[str | Path]) -> None:
        self.project_root = Path(project_root).expanduser().resolve()
        self.source_roots = tuple(Path(path).expanduser().resolve() for path in source_roots)

    @classmethod
    def from_audio_service(cls, audio_service) -> "AudioProjectService":
        output_root = Path(audio_service.flags.resolved_output_dir()).expanduser().resolve()
        audio_subdir = getattr(audio_service.settings, "audio_output_subdir", "audio")
        return cls(
            output_root / "audio-projects",
            source_roots=(output_root / audio_subdir, output_root / "audio-lab"),
        )

    def save_project(
        self,
        *,
        name: str,
        options: AudioGenerationOptions,
        audio_path: str | Path | None = None,
        project_id: str | None = None,
        license_notice: str | None = None,
        license: dict[str, Any] | None | object = _LICENSE_UNSET,
        consent_status: str | None = None,
        sample_rate: int = 0,
    ) -> AudioProjectManifest:
        clean_name = (name or "").strip()
        if not clean_name:
            raise ValueError("Enter a name for the audio project.")
        if len(clean_name) > 100:
            raise ValueError("Audio project names are limited to 100 characters.")

        is_new = project_id is None
        if is_new:
            project_id, project_dir = self._create_project_dir()
            previous = None
        else:
            project_id = validate_project_id(project_id)
            project_dir = self._project_dir(project_id)
            previous = self.load_project(project_id)

        if audio_path is None and previous and previous.track and options.model_id != previous.track.model_id:
            raise ValueError("The audio model cannot change while keeping a project track from another model.")

        track = previous.track if previous else None
        try:
            if audio_path is not None:
                license_record, resolved_license_notice = self._resolve_license_metadata(
                    options.model_id, license, license_notice
                )
                track = self._track_for_audio(
                    audio_path,
                    project_id,
                    project_dir,
                    options,
                    previous.track if previous else None,
                    license_notice=resolved_license_notice,
                    license_record=license_record,
                    consent_status=consent_status,
                    sample_rate=sample_rate,
                )

            now = _utc_now()
            manifest = AudioProjectManifest(
                project_id=project_id,
                name=clean_name,
                created_at=previous.created_at if previous else now,
                updated_at=now,
                options=options.model_copy(deep=True),
                track=track,
            )
            self._write_manifest(project_dir, manifest, create_only=is_new)
            return manifest
        except Exception:
            # A copied asset can become an unrelated path after a concurrent
            # assets-directory replacement. Preserve unreferenced output when
            # manifest publication fails rather than unlink by pathname.
            if is_new:
                self._remove_new_project_if_empty(project_dir)
            raise

    @staticmethod
    def _resolve_license_metadata(
        model_id: str,
        license: dict[str, Any] | None | object,
        license_notice: str | None,
    ) -> tuple[dict[str, Any] | None, str | None]:
        expected = audio_licenses.license_for(model_id)
        expected_notice = audio_licenses.notice_for(model_id)
        if license is _LICENSE_UNSET:
            if license_notice is not None and license_notice != expected_notice:
                raise ValueError("The license notice does not match the structured license record.")
            return expected, expected_notice
        if license is None:
            if license_notice is not None:
                raise ValueError("A license notice cannot be saved without its structured license record.")
            return None, None
        if not isinstance(license, dict) or license != expected:
            raise ValueError("The license record does not match the audio model attributed to this track.")
        if license_notice is not None and license_notice != expected_notice:
            raise ValueError("The license notice does not match the structured license record.")
        return expected, expected_notice

    def load_project(self, project_id: str) -> AudioProjectManifest:
        normalized = validate_project_id(project_id)
        manifest = self._read_manifest(normalized, recover=True)
        if manifest.track is None:
            return manifest
        asset_path = self._asset_path(self._project_dir(normalized), manifest.track.asset_ref)
        if not asset_path.is_file():
            raise AudioProjectAssetMissing(
                f"Audio project '{manifest.name}' refers to missing audio: {manifest.track.source_name}."
            )
        return manifest

    def list_projects(self) -> list[dict[str, object]]:
        if not self.project_root.exists():
            return []
        projects: list[dict[str, object]] = []
        for candidate in self.project_root.iterdir():
            if not candidate.is_dir() or candidate.is_symlink():
                continue
            try:
                project_id = validate_project_id(candidate.name)
                manifest = self._read_manifest(project_id, recover=False)
            except (ValueError, AudioProjectError):
                continue
            has_audio = False
            audio_missing = False
            if manifest.track is not None:
                has_audio = self._asset_path(candidate, manifest.track.asset_ref).is_file()
                audio_missing = not has_audio
            projects.append(
                {
                    "project_id": manifest.project_id,
                    "name": manifest.name,
                    "updated_at": manifest.updated_at,
                    "has_audio": has_audio,
                    "audio_missing": audio_missing,
                }
            )
        return sorted(projects, key=lambda item: str(item["updated_at"]), reverse=True)

    def recover_project(self, project_id: str) -> bool:
        """Recover one complete pending manifest only when the primary is absent."""
        normalized = validate_project_id(project_id)
        project_dir = self._project_dir(normalized)
        if not project_dir.exists():
            raise AudioProjectNotFound(f"Audio project not found: {normalized}.")
        primary = project_dir / "project.json"
        if _is_link_or_junction(primary):
            raise AudioProjectCorrupt("Audio project manifests cannot be symbolic links or junctions.")
        if primary.exists():
            self._read_manifest(normalized, recover=False)
            return False
        pending = self._pending_files(project_dir)
        if len(pending) != 1:
            if not pending:
                raise AudioProjectNotFound(f"Audio project not found: {normalized}.")
            raise AudioProjectRecoveryError("More than one pending project file exists; nothing was changed.")
        candidate = pending[0]
        if _is_link_or_junction(candidate) or not candidate.is_file():
            raise AudioProjectRecoveryError("Pending project files must be regular files; nothing was changed.")
        try:
            manifest = self._decode_manifest(candidate.read_text(encoding="utf-8"))
            self._validate_manifest_identity(normalized, manifest)
            self._validate_manifest_asset(project_dir, manifest)
        except AudioProjectError:
            raise
        except Exception as exc:
            raise AudioProjectRecoveryError("The pending audio project file is invalid; nothing was changed.") from exc
        try:
            # Publish without replacing a primary manifest that may have
            # appeared after the initial check. Same-directory hard linking
            # makes creation atomic and fails closed on a collision.
            if _is_link_or_junction(candidate) or _is_link_or_junction(primary):
                raise AudioProjectRecoveryError("A project manifest became a link during recovery.")
            os.link(candidate, primary)
            try:
                candidate.unlink()
            except OSError:
                # The primary is already a valid recovered manifest. A stale
                # pending copy is harmless and must not invalidate its assets.
                pass
        except AudioProjectRecoveryError:
            raise
        except FileExistsError as exc:
            raise AudioProjectRecoveryError("A project manifest appeared during recovery; nothing was replaced.") from exc
        except OSError as exc:
            raise AudioProjectRecoveryError(f"Could not recover the audio project: {exc}") from exc
        return True

    def _read_manifest(self, project_id: str, *, recover: bool) -> AudioProjectManifest:
        project_dir = self._project_dir(project_id)
        if not project_dir.is_dir():
            raise AudioProjectNotFound(f"Audio project not found: {project_id}.")
        path = project_dir / "project.json"
        if _is_link_or_junction(path):
            raise AudioProjectCorrupt("Audio project manifests cannot be symbolic links or junctions.")
        if not path.is_file():
            if recover:
                self.recover_project(project_id)
            else:
                raise AudioProjectNotFound(f"Audio project manifest not found: {project_id}.")
        try:
            manifest = self._decode_manifest(path.read_text(encoding="utf-8"))
        except AudioProjectError:
            raise
        except Exception as exc:
            raise AudioProjectCorrupt(f"Audio project '{project_id}' has an invalid project.json file.") from exc
        self._validate_manifest_identity(project_id, manifest)
        self._validate_manifest_asset(project_dir, manifest)
        return manifest

    @staticmethod
    def _decode_manifest(raw: str) -> AudioProjectManifest:
        try:
            return AudioProjectManifest.model_validate_json(raw)
        except Exception as exc:
            raise AudioProjectCorrupt(f"Unsupported or malformed audio project: {exc}") from exc

    @staticmethod
    def _validate_manifest_identity(project_id: str, manifest: AudioProjectManifest) -> None:
        if manifest.project_id != project_id:
            raise AudioProjectCorrupt("The audio project ID does not match its folder name.")

    def _validate_manifest_asset(self, project_dir: Path, manifest: AudioProjectManifest) -> None:
        if manifest.track is not None:
            self._asset_path(project_dir, manifest.track.asset_ref)

    def _project_dir(self, project_id: str) -> Path:
        normalized = validate_project_id(project_id)
        candidate = self.project_root / normalized
        if _is_link_or_junction(candidate):
            raise AudioProjectCorrupt("Audio project folders cannot be symbolic links.")
        resolved = candidate.resolve()
        if not _is_relative_to(resolved, self.project_root):
            raise ValueError("Audio project path escaped the managed project directory.")
        return candidate

    @staticmethod
    def _asset_path(project_dir: Path, asset_ref: str) -> Path:
        # Pydantic checks the portable POSIX form; resolve once more to catch a
        # replaced assets directory or symlink before touching a file.
        assets_dir = AudioProjectService._assets_dir(project_dir)
        asset = (project_dir / Path(*asset_ref.split("/"))).resolve()
        resolved_project = project_dir.resolve()
        if (
            assets_dir.parent != resolved_project
            or not _is_relative_to(asset, assets_dir)
            or asset.parent != assets_dir
        ):
            raise AudioProjectCorrupt("Audio asset reference escapes the project assets directory.")
        return asset

    @staticmethod
    def _assets_dir(project_dir: Path, *, create: bool = False) -> Path:
        project_resolved = project_dir.resolve()
        candidate = project_dir / "assets"
        if _is_link_or_junction(candidate):
            raise AudioProjectCorrupt("Audio project assets cannot be symbolic links or junctions.")
        if create:
            candidate.mkdir(exist_ok=True)
        if _is_link_or_junction(candidate):
            raise AudioProjectCorrupt("Audio project assets cannot be symbolic links or junctions.")
        if candidate.exists() and not candidate.is_dir():
            raise AudioProjectCorrupt("Audio project assets must be a directory inside the project.")
        resolved = candidate.resolve()
        if resolved.parent != project_resolved:
            raise AudioProjectCorrupt("Audio project assets escaped the project directory.")
        return resolved

    def _create_project_dir(self) -> tuple[str, Path]:
        self.project_root.mkdir(parents=True, exist_ok=True)
        for _ in range(5):
            project_id = uuid.uuid4().hex
            project_dir = self.project_root / project_id
            try:
                project_dir.mkdir()
            except FileExistsError:
                continue
            return project_id, project_dir
        raise AudioProjectError("Could not allocate a unique audio project ID.")

    def _track_for_audio(
        self,
        audio_path: str | Path,
        project_id: str,
        project_dir: Path,
        options: AudioGenerationOptions,
        previous_track: AudioProjectTrack | None,
        *,
        license_notice: str | None,
        license_record: dict[str, Any] | None,
        consent_status: str | None,
        sample_rate: int,
    ) -> AudioProjectTrack:
        try:
            source = Path(audio_path).expanduser().resolve(strict=True)
        except OSError as exc:
            raise AudioProjectAssetMissing("The selected audio file was not found.") from exc
        if not source.is_file():
            raise ValueError("Select an existing audio file to save in the project.")
        if source.suffix.lower() not in _AUDIO_EXTENSIONS:
            raise ValueError("Only common audio file formats can be added to an audio project.")

        if previous_track is not None:
            current_asset = self._asset_path(project_dir, previous_track.asset_ref)
            if source == current_asset:
                updated_track = previous_track.model_dump()
                updated_track.update(
                    {
                        "prompt": options.prompt,
                        "model_id": options.model_id,
                        "kind": options.kind,
                        "duration_seconds": options.duration_seconds,
                        "sample_rate": sample_rate if sample_rate > 0 else previous_track.sample_rate,
                        "license_notice": license_notice,
                        "license": license_record,
                        "consent_status": consent_status if consent_status is not None else previous_track.consent_status,
                    }
                )
                return AudioProjectTrack.model_validate(updated_track)

        if not any(_is_relative_to(source, root) for root in self.source_roots):
            raise ValueError("Audio file must be inside an approved Studio audio output folder.")

        self._assets_dir(project_dir, create=True)
        asset_dir_identity = self._asset_dir_identity(project_dir)
        asset_id = uuid.uuid4().hex
        asset_ref = f"assets/{asset_id}{source.suffix.lower()}"
        destination = self._asset_path(project_dir, asset_ref)
        track = AudioProjectTrack(
            track_id=asset_id,
            asset_ref=asset_ref,
            source_name=source.name,
            prompt=options.prompt,
            model_id=options.model_id,
            kind=options.kind,
            duration_seconds=options.duration_seconds,
            sample_rate=sample_rate,
            license_notice=license_notice,
            license=license_record,
            consent_status=consent_status,
        )
        try:
            # Exclusive creation makes collisions fail closed. On Windows the
            # handle also carries DELETE access so failed writes can be removed
            # by object identity rather than by a raceable pathname.
            fd = _open_exclusive_asset(destination)
            try:
                opened_path = _opened_file_path(fd)
                expected_path = destination.resolve()
                if os.path.normcase(str(opened_path)) != os.path.normcase(str(expected_path)):
                    raise AudioProjectCorrupt("Audio asset destination changed during the save; no audio was copied.")
                self._assert_asset_dir_identity(project_dir, asset_dir_identity)
            except Exception as exc:
                try:
                    _mark_open_asset_for_deletion(fd)
                except OSError:
                    pass
                os.close(fd)
                if isinstance(exc, AudioProjectCorrupt):
                    raise
                raise AudioProjectCorrupt("Could not verify the opened audio asset destination.") from exc

            try:
                with source.open("rb") as src, os.fdopen(fd, "wb", closefd=False) as dst:
                    shutil.copyfileobj(src, dst)
                    dst.flush()
                    os.fsync(dst.fileno())
            except Exception:
                try:
                    _mark_open_asset_for_deletion(fd)
                except OSError:
                    # Preserve ambiguous output if exact-handle deletion is
                    # unavailable; never unlink by a potentially replaced path.
                    pass
                raise
            finally:
                os.close(fd)
        except Exception:
            raise

        return track

    def _write_manifest(self, project_dir: Path, manifest: AudioProjectManifest, *, create_only: bool) -> None:
        temp = project_dir / f".project.json.{uuid.uuid4().hex}.tmp"
        target = project_dir / "project.json"
        data = (manifest.model_dump_json(indent=2, exclude_none=True) + "\n").encode("utf-8")
        fd = os.open(temp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        published = False
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            if create_only:
                # Hard-link publication is atomic and refuses to replace a
                # manifest that appeared after the project directory was made.
                os.link(temp, target)
                published = True
                try:
                    temp.unlink()
                except OSError:
                    # Publication already committed; a leftover temporary
                    # hard link is safer than reporting failure to the caller.
                    pass
            else:
                os.replace(temp, target)
                published = True
        except Exception:
            if not published:
                temp.unlink(missing_ok=True)
            raise

    def _asset_dir_identity(self, project_dir: Path) -> tuple[int, int]:
        assets_dir = self._assets_dir(project_dir)
        stat = assets_dir.stat()
        return stat.st_dev, stat.st_ino

    def _assert_asset_dir_identity(self, project_dir: Path, expected: tuple[int, int]) -> None:
        if self._asset_dir_identity(project_dir) != expected:
            raise AudioProjectCorrupt("Audio project assets changed during the save; no audio was copied.")

    def _pending_files(self, project_dir: Path) -> list[Path]:
        return [path for path in project_dir.iterdir() if _PENDING_MANIFEST.fullmatch(path.name)]

    @staticmethod
    def _remove_new_project_if_empty(project_dir: Path) -> None:
        try:
            assets_dir = project_dir / "assets"
            if assets_dir.is_dir() and not _is_link_or_junction(assets_dir) and not any(assets_dir.iterdir()):
                assets_dir.rmdir()
            for child in project_dir.iterdir():
                if child.is_file() and child.name.startswith(".") and child.name.endswith(".tmp"):
                    child.unlink(missing_ok=True)
            project_dir.rmdir()
        except OSError:
            # Preserve anything left by a failed operation for later inspection.
            pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _is_link_or_junction(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    return bool(is_junction and is_junction())


def _open_exclusive_asset(path: Path) -> int:
    """Create a new asset file, retaining delete rights on Windows."""
    if os.name != "nt":
        return os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)

    import ctypes
    import msvcrt

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32,
                            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
    create_file.restype = ctypes.c_void_p
    handle = create_file(str(path), 0x40000000 | 0x00010000, 0x1 | 0x2 | 0x4,
                         None, 1, 0x80, None)
    invalid = ctypes.c_void_p(-1).value
    if handle == invalid:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return msvcrt.open_osfhandle(int(handle), os.O_WRONLY | os.O_BINARY)
    except Exception:
        kernel32.CloseHandle(ctypes.c_void_p(handle))
        raise


def _mark_open_asset_for_deletion(fd: int) -> bool:
    """Mark the opened file object for deletion; do not resolve its pathname."""
    if os.name != "nt":
        return False

    import ctypes
    import msvcrt

    class _FileDispositionInfo(ctypes.Structure):
        _fields_ = [("DeleteFile", ctypes.c_ubyte)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    set_info = kernel32.SetFileInformationByHandle
    set_info.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
    set_info.restype = ctypes.c_int
    info = _FileDispositionInfo(1)
    if not set_info(msvcrt.get_osfhandle(fd), 4, ctypes.byref(info), ctypes.sizeof(info)):
        raise ctypes.WinError(ctypes.get_last_error())
    return True

def _opened_file_path(fd: int) -> Path:
    """Resolve the file represented by an open descriptor, not its pathname."""
    if os.name == "nt":
        import ctypes
        import msvcrt

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_final_path = kernel32.GetFinalPathNameByHandleW
        get_final_path.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32]
        get_final_path.restype = ctypes.c_uint32
        buffer = ctypes.create_unicode_buffer(32768)
        length = get_final_path(msvcrt.get_osfhandle(fd), buffer, len(buffer), 0)
        if length == 0 or length >= len(buffer):
            raise OSError(ctypes.get_last_error(), "Could not verify the opened audio asset destination.")
        value = buffer.value
        if value.startswith("\\\\?\\"):
            value = value[4:]
        return Path(value).resolve()

    descriptor_path = Path(f"/proc/self/fd/{fd}")
    if descriptor_path.exists():
        return Path(os.readlink(descriptor_path)).resolve()
    raise OSError("This platform cannot verify the opened audio asset destination.")
