from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "static" / "second_gui"
DEFAULT_BACKEND_URL = os.environ.get("AIWF_SECOND_GUI_BACKEND", "http://127.0.0.1:7860")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")


def _read_json_body(handler: SimpleHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or 0)
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        decoded = raw.decode("utf-8")
        payload = json.loads(decoded)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _post_json(url: str, payload: dict[str, Any], timeout: float = 60.0) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec - local dev bridge
        raw = response.read().decode("utf-8", errors="replace")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}


def _get_url(url: str, timeout: float = 0.75) -> tuple[bool, str]:
    try:
        request = urllib.request.Request(url, method="GET", headers={"Accept": "text/html,application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec - local dev probe
            return response.status < 500, f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        # 404 still proves that a server is alive; connection refused does not.
        return exc.code < 500, f"HTTP {exc.code}"
    except Exception as exc:
        return False, exc.__class__.__name__


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


class SecondGuiHandler(SimpleHTTPRequestHandler):
    """Static preview shell plus tiny local JSON bridge.

    The second GUI is intentionally conservative. It can display the new visual
    shell immediately, but unfinished backend routes return explicit WIP payloads
    instead of pretending a feature is wired.
    """

    server_version = "AIWFSecondGUI/0.1"
    backend_url = DEFAULT_BACKEND_URL.rstrip("/")
    proxy_enabled = _env_bool("AIWF_SECOND_GUI_PROXY", False)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        print(f"[AIWF Second GUI] {self.address_string()} - {format % args}", flush=True)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        raw = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _send_wip(self, feature: str, detail: str = "Backend route is not wired yet.") -> None:
        self._send_json(
            {
                "ok": False,
                "wip": True,
                "feature": feature,
                "message": detail,
                "next": "Keep the UI clickable, but make the missing route obvious until the service exists.",
            },
            status=202,
        )

    def _runtime_status(self) -> dict[str, Any]:
        alive, note = _get_url(self.backend_url)
        return {
            "ok": True,
            "engine": "Local Engine",
            "engine_state": "Ready" if alive else "WIP bridge",
            "backend_url": self.backend_url,
            "backend_reachable": alive,
            "backend_note": note,
            "proxy_enabled": self.proxy_enabled,
            "backend": "AIWF Studio" if alive else "Second GUI shell",
            "device": os.environ.get("AIWF_SECOND_GUI_DEVICE", "Detected by main AIWF runtime"),
            "precision": os.environ.get("AIWF_SECOND_GUI_PRECISION", "WIP / runtime-reported"),
            "attention": os.environ.get("AIWF_SECOND_GUI_ATTENTION", "WIP / runtime-reported"),
            "max_resolution": os.environ.get("AIWF_SECOND_GUI_MAX_RES", "1024 x 1024"),
            "vram": os.environ.get("AIWF_SECOND_GUI_VRAM", "WIP"),
            "ram": os.environ.get("AIWF_SECOND_GUI_RAM", "WIP"),
            "storage": os.environ.get("AIWF_SECOND_GUI_STORAGE", "WIP"),
            "cpu": os.environ.get("AIWF_SECOND_GUI_CPU", "WIP"),
            "loaded_model": os.environ.get("AIWF_SECOND_GUI_MODEL", "sdxl-base-1.0 (Diffusers)"),
        }

    def _feature_list(self) -> dict[str, Any]:
        return {
            "ok": True,
            "features": [
                {"id": "image", "label": "Image", "state": "preview", "route": "/api/generate"},
                {"id": "video", "label": "Video", "state": "wip", "route": None},
                {"id": "inpaint", "label": "Inpaint", "state": "wip", "route": None},
                {"id": "models", "label": "Models", "state": "wip", "route": None},
                {"id": "data", "label": "Data", "state": "wip", "route": None},
                {"id": "batch", "label": "Batch", "state": "wip", "route": None},
                {"id": "workflows", "label": "Workflows", "state": "wip", "route": None},
                {"id": "logs", "label": "Logs", "state": "wip", "route": None},
            ],
        }

    def _handle_generate(self) -> None:
        request = _read_json_body(self)
        if not self.proxy_enabled:
            self._send_wip(
                "Image generation",
                "Second GUI is running. Set AIWF_SECOND_GUI_PROXY=1 to proxy Generate into the existing AIWF /sdapi/v1/txt2img adapter once the main backend is running.",
            )
            return

        sdapi_payload = {
            "prompt": str(request.get("prompt", "")),
            "negative_prompt": str(request.get("negative_prompt", "")),
            "steps": _coerce_int(request.get("steps"), 30),
            "cfg_scale": _coerce_float(request.get("cfg_scale"), 7.0),
            "sampler_name": str(request.get("sampler", "DPM++ 2M Karras")),
            "seed": _coerce_int(request.get("seed"), -1),
            "width": _coerce_int(request.get("width"), 1024),
            "height": _coerce_int(request.get("height"), 1024),
        }
        try:
            response = _post_json(f"{self.backend_url}/sdapi/v1/txt2img", sdapi_payload, timeout=300.0)
            self._send_json({"ok": True, "source": "sdapi", "response": response})
        except Exception as exc:
            self._send_json(
                {
                    "ok": False,
                    "wip": True,
                    "feature": "Image generation proxy",
                    "message": "Proxy attempted /sdapi/v1/txt2img but the backend did not accept it.",
                    "error": f"{exc.__class__.__name__}: {exc}",
                    "payload": sdapi_payload,
                },
                status=502,
            )

    def do_GET(self) -> None:  # noqa: N802 - stdlib API
        path = urllib.parse.urlparse(self.path).path
        if path in {"", "/"}:
            self.path = "/index.html"
            return super().do_GET()
        if path == "/api/runtime/status":
            return self._send_json(self._runtime_status())
        if path == "/api/features":
            return self._send_json(self._feature_list())
        if path.startswith("/api/wip"):
            return self._send_wip("Second GUI feature")
        return super().do_GET()

    def do_POST(self) -> None:  # noqa: N802 - stdlib API
        path = urllib.parse.urlparse(self.path).path
        if path == "/api/generate":
            return self._handle_generate()
        if path.startswith("/api/wip"):
            body = _read_json_body(self)
            return self._send_wip(str(body.get("feature", "Second GUI feature")))
        self._send_json({"ok": False, "error": "Not found"}, status=404)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AIWF Studio Second GUI preview shell.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("AIWF_SECOND_GUI_PORT", "8770")))
    parser.add_argument("--backend", default=DEFAULT_BACKEND_URL, help="Main AIWF backend URL for status/proxy checks.")
    parser.add_argument("--proxy", action="store_true", help="Proxy Generate to backend /sdapi/v1/txt2img.")
    parser.add_argument("--listen", action="store_true", help="Bind to 0.0.0.0 instead of 127.0.0.1.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser automatically.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not STATIC_DIR.exists():
        raise SystemExit(f"Second GUI static folder missing: {STATIC_DIR}")

    SecondGuiHandler.backend_url = str(args.backend).rstrip("/")
    SecondGuiHandler.proxy_enabled = bool(args.proxy or _env_bool("AIWF_SECOND_GUI_PROXY", False))

    host = "0.0.0.0" if args.listen else "127.0.0.1"
    url = f"http://127.0.0.1:{args.port}/"
    print("[AIWF Second GUI] Starting preview shell", flush=True)
    print(f"[AIWF Second GUI] Static root: {STATIC_DIR}", flush=True)
    print(f"[AIWF Second GUI] Backend: {SecondGuiHandler.backend_url}", flush=True)
    print(f"[AIWF Second GUI] Proxy generate: {SecondGuiHandler.proxy_enabled}", flush=True)
    print(f"[AIWF Second GUI] Open: {url}", flush=True)

    httpd = ThreadingHTTPServer((host, args.port), SecondGuiHandler)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[AIWF Second GUI] Shutdown requested", flush=True)
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
