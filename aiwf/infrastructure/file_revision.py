"""Lightweight identities for local model and support files."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path


_CONTENT_PROBE_BYTES = 1024 * 1024


def _content_probe(path: Path, size: int) -> str:
    """Read bounded content probes on each check, including on Windows."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            digest.update(handle.read(_CONTENT_PROBE_BYTES))
            if size > _CONTENT_PROBE_BYTES:
                handle.seek(max(0, size - _CONTENT_PROBE_BYTES))
                digest.update(handle.read(_CONTENT_PROBE_BYTES))
        return digest.hexdigest()[:16]
    except OSError:
        return "unreadable"


def local_file_revision(path: Path, stat_result: os.stat_result | None = None) -> tuple[int, int, int, int, str]:
    """Fingerprint metadata plus bounded content to catch same-size replacements.

    Small files are hashed in full. Large model files use first/last 1 MiB
    probes, so route checks stay bounded. Metadata and the sampled content
    invalidate dependent caches for common replacements; this is not a full
    content hash for large files.
    """
    stat = stat_result or path.stat()
    size = int(stat.st_size)
    mtime_ns = int(getattr(stat, "st_mtime_ns", 0))
    ctime_ns = int(getattr(stat, "st_ctime_ns", 0))
    inode = int(getattr(stat, "st_ino", 0))
    content = _content_probe(path, size)
    return (
        size,
        mtime_ns,
        ctime_ns,
        inode,
        content,
    )
