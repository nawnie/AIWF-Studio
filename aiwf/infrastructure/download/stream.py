from __future__ import annotations

import logging
import os
import urllib.request
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]


def _auth_headers() -> dict[str, str]:
    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def stream_download(
    url: str,
    dest: Path,
    *,
    on_progress: ProgressCallback | None = None,
    headers: dict[str, str] | None = None,
    chunk_size: int = 1024 * 256,
) -> Path:
    """Stream a remote file to ``dest`` via a ``.part`` temp file (idempotent)."""
    dest = dest.resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        return dest

    request_headers = dict(_auth_headers())
    if headers:
        request_headers.update(headers)

    request = urllib.request.Request(url, headers=request_headers)
    tmp = dest.with_suffix(dest.suffix + ".part")
    logger.info("Downloading %s -> %s", url, dest)
    try:
        with urllib.request.urlopen(request) as response:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            with open(tmp, "wb") as handle:
                while True:
                    block = response.read(chunk_size)
                    if not block:
                        break
                    handle.write(block)
                    done += len(block)
                    if on_progress:
                        on_progress(done, total)
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return dest