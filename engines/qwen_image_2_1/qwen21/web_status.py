"""Tiny status server so AIWF Studio Pro's Qwen tab can see the desktop app.

The React tab (frontend/src/layouts/studio/QwenImageEditorLayout.tsx) polls
``http://127.0.0.1:7865/api/health`` and embeds ``/`` in an iframe. We serve a
health JSON, a small status page listing the latest outputs, and the output
files themselves. Nothing else; the editing UI stays native.
"""
from __future__ import annotations

import html
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


class StatusState:
    def __init__(self, output_dir: Path, status_provider: Callable[[], dict] | None = None) -> None:
        self.output_dir = output_dir
        self.status_provider = status_provider or (lambda: {})

    def recent_outputs(self, limit: int = 24) -> list[Path]:
        if not self.output_dir.is_dir():
            return []
        files = [p for p in self.output_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()]
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return files[:limit]


def _make_handler(state: StatusState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "Qwen21Studio/0.1"

        def log_message(self, *_args) -> None:  # quiet
            return

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/api/health":
                payload = {"status": "ok", "app": "qwen-image-2.1-studio"}
                payload.update(state.status_provider())
                self._send(200, json.dumps(payload).encode("utf-8"), "application/json")
                return
            if path.startswith("/outputs/"):
                name = Path(path[len("/outputs/"):]).name
                target = state.output_dir / name
                if target.is_file() and target.suffix.lower() in IMAGE_EXTS:
                    ctype = "image/png" if target.suffix.lower() == ".png" else "image/jpeg"
                    self._send(200, target.read_bytes(), ctype)
                    return
                self._send(404, b"not found", "text/plain")
                return
            if path in ("/", "/index.html"):
                items = "".join(
                    f'<a href="/outputs/{html.escape(p.name)}" target="_blank">'
                    f'<img src="/outputs/{html.escape(p.name)}" alt="{html.escape(p.name)}"></a>'
                    for p in state.recent_outputs()
                )
                status = state.status_provider()
                page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Qwen Image 2.1 Studio</title>
<style>body{{margin:0;background:#101214;color:#e6e6e6;font:14px system-ui}}header{{padding:12px 16px;border-bottom:1px solid #2a2e33}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:8px;padding:12px}}img{{width:100%;border-radius:6px;
background:repeating-conic-gradient(#333 0 25%,#222 0 50%) 0 0/16px 16px}}small{{color:#9aa}}</style></head><body>
<header><strong>Qwen Image 2.1 Studio</strong> &nbsp;<small>desktop app is running · {html.escape(json.dumps(status))}</small></header>
<div class="grid">{items or '<small style="padding:12px">No outputs yet.</small>'}</div></body></html>"""
                self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
                return
            self._send(404, b"not found", "text/plain")

    return Handler


class WebStatusServer:
    def __init__(self, state: StatusState, host: str = "127.0.0.1", port: int = 7865) -> None:
        self.state = state
        self.host, self.port = host, port
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        if self._server is not None:
            return True
        try:
            self._server = ThreadingHTTPServer((self.host, self.port), _make_handler(self.state))
        except OSError:
            self._server = None
            return False
        self._thread = threading.Thread(target=self._server.serve_forever, name="qwen21-web-status", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    @property
    def running(self) -> bool:
        return self._server is not None
