"""Safely apply the installer's image-backend choice to the AES profile."""
from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import tempfile

SCHEMA = "aiwf-desktop-settings-v1"


def _reject_link(path: Path) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & reparse:
        raise ValueError(f"Refusing to migrate a linked settings or project path: {path}")


def _write_new(path: Path, content: bytes) -> None:
    """Create a migration file exclusively; never replace a user's existing file."""
    created = False
    try:
        with path.open("xb") as stream:
            created = True
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if created:
            path.unlink(missing_ok=True)
        raise


def migrate_legacy_profile(root: Path, profile: Path) -> int:
    """Copy old root-local folder settings and project records without overwriting."""
    source_launch = root / "launch.json"
    destination_launch = profile / "launch.json"
    source_projects = root / "_local" / "unified-projects"
    destination_projects = profile / "_local" / "unified-projects"
    for path in (source_launch, destination_launch, root / "_local", source_projects,
                 profile / "_local", destination_projects):
        _reject_link(path)

    pairs = []
    if source_launch.exists():
        if not source_launch.is_file():
            raise ValueError("The previous launch settings are not a regular file")
        pairs.append((source_launch, destination_launch))
    if source_projects.exists():
        if not source_projects.is_dir():
            raise ValueError("The previous project ledger is not a directory")
        for source in sorted(source_projects.glob("aiwfp-*.json")):
            _reject_link(source)
            if not source.is_file():
                raise ValueError(f"The previous project record is not a regular file: {source}")
            pairs.append((source, destination_projects / source.name))

    pending = []
    for source, destination in pairs:
        _reject_link(destination)
        content = source.read_bytes()
        if destination.exists():
            if not destination.is_file() or destination.read_bytes() != content:
                raise ValueError(f"Existing AES profile data differs; both copies were preserved: {destination}")
        else:
            pending.append((destination, content))
    for destination, content in pending:
        destination.parent.mkdir(parents=True, exist_ok=True)
        _write_new(destination, content)
        if destination.read_bytes() != content:
            raise OSError(f"Could not verify migrated AES profile data: {destination}")
    return len(pending)


def ensure_profile_launch(profile: Path) -> str:
    """Create default folder settings only when this profile has none."""
    launch = profile / "launch.json"
    _reject_link(launch)
    if launch.exists():
        if not launch.is_file():
            raise ValueError("AES folder settings are not a regular file")
        return "preserved"
    (profile / "models" / "Stable-diffusion").mkdir(parents=True, exist_ok=True)
    (profile / "outputs").mkdir(parents=True, exist_ok=True)
    data = {"models_dir": str(profile / "models"),
            "ckpt_dir": str(profile / "models" / "Stable-diffusion"),
            "output_dir": str(profile / "outputs")}
    _write_new(launch, (json.dumps(data, indent=2) + "\n").encode("utf-8"))
    return "created"


def configure_backend(profile: Path, backend: str, policy: str) -> str:
    """Apply an interactive choice, or keep an existing profile for silent setup."""
    if backend not in {"cpp", "both"} or policy not in {"apply", "preserve"}:
        raise ValueError("Invalid installer backend choice or settings policy")

    settings = profile / "settings.json"
    present = settings.exists() or settings.is_symlink()
    if present and policy == "preserve":
        return "preserved"
    if settings.is_symlink():
        raise ValueError("The existing settings file is a link; it was not changed")

    original = None
    if present:
        original = settings.read_bytes()
        try:
            data = json.loads(original.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("The existing settings file is invalid; it was not changed") from exc
        if not isinstance(data, dict) or data.get("schema") != SCHEMA:
            raise ValueError("The existing settings file has an unexpected format; it was not changed")
    else:
        data = {"schema": SCHEMA, "setup_completed": False}

    data["enabled_image_backends"] = ["python", "cpp"] if backend == "both" else [backend]
    data["image_backend_id"] = "cpp" if backend == "both" else backend
    payload = (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".aes-setup-", suffix=".tmp", dir=profile, delete=False) as temp:
            temp_path = Path(temp.name)
            temp.write(payload)
            temp.flush()
            os.fsync(temp.fileno())
        if settings.is_symlink():
            raise ValueError("The settings file became a link; it was not changed")
        if original is not None:
            if not settings.is_file() or settings.read_bytes() != original:
                raise ValueError("The settings file changed during installation; it was not overwritten")
        elif settings.exists():
            raise ValueError("A settings file appeared during installation; it was not overwritten")
        os.replace(temp_path, settings)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return "updated" if present else "created"
