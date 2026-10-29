"""Versioned compatibility metadata for the native app and engine API.

This module intentionally uses only the Python standard library. Its allowlist
defines the source files whose changes require the native shell and API to be
rebuilt together. Paths and hashes are root-independent and contain no user
data. The runtime drift guard rejects new API requests after an allowlisted
source change; it does not cancel an already-running generation. Stop the API
before changing its source and install from an immutable, matching checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping


NATIVE_API_PROTOCOL_VERSION = 2
BUILD_INFO_SCHEMA = "aiwf-native-build-info-v1"
FINGERPRINT_DOMAIN = b"AIWF_NATIVE_API_FINGERPRINT_V1\n"

# Keep this list stable and source-relative. Update intentionally when the
# native API contract gains another critical implementation or runtime file.
API_SOURCE_FILES = (
    "aiwf/api/security.py",
    "aiwf/bootstrap.py",
    "aiwf/credentials.py",
    "aiwf/infrastructure/download/stream.py",
    "aiwf/services/model_download.py",
    "aiwf/services/civitai_browser.py",
    "aiwf/core/config/launch.py",
    "aiwf/core/config/settings.py",
    "aiwf/core/domain/errors.py",
    "aiwf/core/domain/generation.py",
    "aiwf/core/domain/models.py",
    "aiwf/core/events/bus.py",
    "aiwf/core/events/types.py",
    "aiwf/core/interfaces/backend.py",
    "aiwf/engine_api.py",
    "aiwf/infrastructure/diffusers/backend.py",
    "aiwf/infrastructure/diffusers/checkpoints.py",
    "aiwf/infrastructure/diffusers/model_arch.py",
    "aiwf/infrastructure/diffusers/model_blocks.py",
    "aiwf/infrastructure/model_asset_summary.py",
    "aiwf/infrastructure/model_inventory.py",
    "aiwf/infrastructure/safetensors_metadata.py",
    "aiwf/infrastructure/sdcpp/backend.py",
    "aiwf/native_compat.py",
    "aiwf/services/aes_downloads.py",
    "aiwf/services/generation.py",
    "aiwf/services/image_contract.py",
    "aiwf/services/image_jobs.py",
    "aiwf/services/creative_jobs.py",
    "aiwf/services/workflows/aes_qwen21_edit.json",
    "aiwf/services/workflows/aes_moss_image.json",
    "aiwf/services/workflows/aes_wan22_i2v.json",
    "aiwf/services/workflows/aes_ltx25_i2v.json",
    "aiwf/services/model_files.py",
    "aiwf/services/native_image_backends.py",
    "aiwf/services/output_metadata.py",
    "aiwf/services/queue.py",
    "aiwf/services/setup_settings.py",
    "aiwf/services/unified_bridge.py",
    "aiwf/services/unified_contract.py",
    "aiwf/web/unified_api.py",
    "native/AIWF.Desktop/engine-api-preflight.py",
    "native/AIWF.Desktop/engine-api-requirements.txt",
)
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BUILD_INFO_FIELDS = {
    "schema",
    "protocol_version",
    "api_source_files",
    "api_source_fingerprint",
    "exe_sha256",
}


class NativeCompatibilityError(ValueError):
    """Raised when native build metadata does not match the running API."""


def _has_reparse_attribute(info: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(info.st_mode) or bool(getattr(info, "st_file_attributes", 0) & reparse_flag)


def _absolute_without_reparse(
    path: str | os.PathLike[str], *, label: str, allow_missing_leaf: bool = False
) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        raise NativeCompatibilityError(f"{label} must be an absolute path: {candidate}")

    current = Path(candidate.anchor)
    try:
        parts = candidate.parts[1:]
        for index, part in enumerate(parts):
            if part in ("", ".", ".."):
                raise NativeCompatibilityError(f"{label} contains an unsafe path component: {candidate}")
            current = current / part
            try:
                info = current.lstat()
            except FileNotFoundError:
                if allow_missing_leaf and index == len(parts) - 1:
                    return candidate
                raise
            if _has_reparse_attribute(info):
                raise NativeCompatibilityError(f"{label} must not cross a symlink or reparse point: {current}")
    except FileNotFoundError as exc:
        raise NativeCompatibilityError(f"{label} does not exist: {current}") from exc
    except OSError as exc:
        raise NativeCompatibilityError(f"Could not inspect {label}: {current}: {exc}") from exc
    return candidate


def _source_root(path: str | os.PathLike[str]) -> Path:
    root = _absolute_without_reparse(path, label="Studio source root")
    if not root.is_dir():
        raise NativeCompatibilityError(f"Studio source root is not a directory: {root}")
    return root


def _hash_regular_file(path: Path, *, label: str) -> str:
    checked = _absolute_without_reparse(path, label=label)
    try:
        if not checked.is_file():
            raise NativeCompatibilityError(f"{label} is not a regular file: {checked}")
        digest = hashlib.sha256()
        with checked.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError as exc:
        raise NativeCompatibilityError(f"Could not hash {label}: {checked}: {exc}") from exc


def compute_api_source_files(source_root: str | os.PathLike[str]) -> dict[str, str]:
    """Hash exactly the allowlisted API/runtime files in stable path order."""
    root = _source_root(source_root)
    result: dict[str, str] = {}
    for relative in sorted(API_SOURCE_FILES):
        candidate = root.joinpath(*relative.split("/"))
        result[relative] = _hash_regular_file(candidate, label=f"API source file {relative}")
    return result


def aggregate_fingerprint(file_hashes: Mapping[str, str]) -> str:
    """Return the canonical aggregate digest for an exact allowlist map."""
    if set(file_hashes) != set(API_SOURCE_FILES):
        missing = sorted(set(API_SOURCE_FILES) - set(file_hashes))
        extra = sorted(set(file_hashes) - set(API_SOURCE_FILES))
        raise NativeCompatibilityError(f"API source hash map differs from the allowlist (missing={missing}, extra={extra}).")

    digest = hashlib.sha256(FINGERPRINT_DOMAIN)
    for relative in sorted(API_SOURCE_FILES):
        file_digest = file_hashes[relative]
        if not isinstance(file_digest, str) or not _HEX_SHA256.fullmatch(file_digest):
            raise NativeCompatibilityError(f"Invalid SHA-256 for API source file {relative}.")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_digest))
        digest.update(b"\n")
    return digest.hexdigest()


def native_api_health_metadata(source_root: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Return public, nonsecret metadata used to check native/API compatibility."""
    root = Path(__file__).absolute().parents[1] if source_root is None else Path(source_root)
    files = compute_api_source_files(root)
    return {
        "protocol_version": NATIVE_API_PROTOCOL_VERSION,
        "api_source_fingerprint": aggregate_fingerprint(files),
    }


def make_build_info(
    source_root: str | os.PathLike[str],
    executable: str | os.PathLike[str],
) -> dict[str, Any]:
    """Build the portable manifest written beside the native executable."""
    files = compute_api_source_files(source_root)
    return {
        "schema": BUILD_INFO_SCHEMA,
        "protocol_version": NATIVE_API_PROTOCOL_VERSION,
        "api_source_files": files,
        "api_source_fingerprint": aggregate_fingerprint(files),
        "exe_sha256": _hash_regular_file(Path(executable), label="native executable"),
    }


def validate_health_metadata(build_info: Mapping[str, Any], health: Mapping[str, Any]) -> None:
    """Fail closed when running API health metadata is absent or mismatched."""
    if build_info.get("schema") != BUILD_INFO_SCHEMA:
        raise NativeCompatibilityError("Native build metadata is missing or has an unsupported schema.")
    protocol = build_info.get("protocol_version")
    if type(protocol) is not int or protocol != NATIVE_API_PROTOCOL_VERSION:
        raise NativeCompatibilityError("Native build metadata has an unsupported API protocol version.")
    fingerprint = build_info.get("api_source_fingerprint")
    if not isinstance(fingerprint, str) or not _HEX_SHA256.fullmatch(fingerprint):
        raise NativeCompatibilityError("Native build metadata has an invalid API source fingerprint.")
    if not isinstance(health, Mapping):
        raise NativeCompatibilityError("The running engine API returned legacy or invalid health metadata.")
    if type(health.get("protocol_version")) is not int:
        raise NativeCompatibilityError("The running engine API does not report a native API protocol version.")
    if health.get("protocol_version") != protocol:
        raise NativeCompatibilityError("The running engine API protocol does not match this native build.")
    running_fingerprint = health.get("api_source_fingerprint")
    if not isinstance(running_fingerprint, str) or not _HEX_SHA256.fullmatch(running_fingerprint):
        raise NativeCompatibilityError("The running engine API does not report a valid source fingerprint.")
    if not hmac.compare_digest(running_fingerprint, fingerprint):
        raise NativeCompatibilityError("The running engine API source does not match this native build.")


def read_build_info(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read and structurally validate a native build-info JSON file."""
    manifest_path = _absolute_without_reparse(path, label="native build-info file")
    try:
        parsed = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NativeCompatibilityError(f"Could not read native build-info file {manifest_path}: {exc}") from exc
    if not isinstance(parsed, dict) or set(parsed) != _BUILD_INFO_FIELDS:
        raise NativeCompatibilityError("Native build-info fields are missing, unknown, or invalid.")
    if parsed.get("schema") != BUILD_INFO_SCHEMA:
        raise NativeCompatibilityError("Native build-info schema is unsupported.")
    if type(parsed.get("protocol_version")) is not int or parsed["protocol_version"] != NATIVE_API_PROTOCOL_VERSION:
        raise NativeCompatibilityError("Native build-info protocol version is unsupported.")
    files = parsed.get("api_source_files")
    if not isinstance(files, dict):
        raise NativeCompatibilityError("Native build-info API source map is invalid.")
    if aggregate_fingerprint(files) != parsed.get("api_source_fingerprint"):
        raise NativeCompatibilityError("Native build-info API source map does not match its aggregate fingerprint.")
    if not isinstance(parsed.get("exe_sha256"), str) or not _HEX_SHA256.fullmatch(parsed["exe_sha256"]):
        raise NativeCompatibilityError("Native build-info executable SHA-256 is invalid.")
    return parsed


def write_build_info(
    source_root: str | os.PathLike[str],
    output: str | os.PathLike[str],
    executable: str | os.PathLike[str],
) -> dict[str, Any]:
    """Atomically write a build-info manifest beside a completed executable."""
    output_path = _absolute_without_reparse(output, label="native build-info output", allow_missing_leaf=True)
    if not output_path.parent.is_dir():
        raise NativeCompatibilityError(f"Native build-info output directory does not exist: {output_path.parent}")
    info = make_build_info(source_root, executable)
    encoded = (json.dumps(info, sort_keys=True, indent=2) + "\n").encode("utf-8")
    fd, temporary_name = tempfile.mkstemp(prefix=".native-build-info-", dir=str(output_path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, output_path)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return info


def verify_build_info(
    source_root: str | os.PathLike[str],
    build_info_path: str | os.PathLike[str],
    executable: str | os.PathLike[str],
) -> dict[str, Any]:
    """Verify source, manifest, and executable hashes before an install stages."""
    info = read_build_info(build_info_path)
    actual_files = compute_api_source_files(source_root)
    if info["api_source_files"] != actual_files:
        changed = [
            path for path in sorted(API_SOURCE_FILES)
            if info["api_source_files"].get(path) != actual_files.get(path)
        ]
        raise NativeCompatibilityError(f"Native build was produced from different API sources: {', '.join(changed)}")
    actual_fingerprint = aggregate_fingerprint(actual_files)
    if not hmac.compare_digest(info["api_source_fingerprint"], actual_fingerprint):
        raise NativeCompatibilityError("Native build API source fingerprint does not match the selected Studio root.")
    actual_executable = _hash_regular_file(Path(executable), label="native executable")
    if not hmac.compare_digest(info["exe_sha256"], actual_executable):
        raise NativeCompatibilityError("Native build executable hash does not match native-build-info.json.")
    return info


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate or validate AIWF Studio native/API compatibility metadata.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    write_parser = subparsers.add_parser("write", help="write native-build-info.json beside a built executable")
    write_parser.add_argument("--source-root", required=True)
    write_parser.add_argument("--output", required=True)
    write_parser.add_argument("--exe", required=True)

    verify_parser = subparsers.add_parser("verify", help="verify source and executable against native-build-info.json")
    verify_parser.add_argument("--source-root", required=True)
    verify_parser.add_argument("--build-info", required=True)
    verify_parser.add_argument("--exe", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "write":
            info = write_build_info(args.source_root, args.output, args.exe)
            print(f"Wrote {args.output} (protocol {info['protocol_version']}, API {info['api_source_fingerprint']}).")
        else:
            info = verify_build_info(args.source_root, args.build_info, args.exe)
            print(f"Verified {args.build_info} (protocol {info['protocol_version']}, API {info['api_source_fingerprint']}).")
    except NativeCompatibilityError as exc:
        print(f"Native API compatibility check failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
