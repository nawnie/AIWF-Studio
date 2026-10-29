from __future__ import annotations

import logging
import os
import urllib.request
import urllib.parse

from aiwf.credentials import read_provider_key
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]


def _provider_host(url: str, hosts: tuple[str, ...]) -> bool:
    parsed = urllib.parse.urlsplit(url)
    return parsed.scheme.lower() == "https" and (parsed.hostname or "").lower() in hosts


def _auth_headers(url: str) -> dict[str, str]:
    if not _provider_host(url, ("huggingface.co", "www.huggingface.co")):
        return {}
    token = read_provider_key("huggingface")
    return {"Authorization": f"Bearer {token}"} if token else {}


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            old = urllib.parse.urlsplit(req.full_url)
            new = urllib.parse.urlsplit(newurl)
            if (old.scheme.lower(), old.netloc.lower()) != (new.scheme.lower(), new.netloc.lower()):
                redirected.remove_header("Authorization")
        return redirected


def safe_urlopen(request, timeout=None):
    return urllib.request.build_opener(_SafeRedirect()).open(request, timeout=timeout)


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

    request_headers = dict(_auth_headers(url))
    if headers:
        request_headers.update(headers)

    request = urllib.request.Request(url, headers=request_headers)
    tmp = dest.with_suffix(dest.suffix + ".part")
    logger.info("Downloading %s -> %s", url, dest)
    try:
        with safe_urlopen(request) as response:
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