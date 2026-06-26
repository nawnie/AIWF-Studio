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
DEFAULT_PROXY_MODE = os.environ.get("AIWF_SECOND_GUI_PROXY_MODE", "auto").strip().lower() or "auto"


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


def _request_json(method: str, url: str, payload: dict[str, Any] | None = None, timeout: float = 5.0) -> Any:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec - local dev bridge
            raw = response.read().decode("utf-8", errors="replace")
            if not raw.strip():
                return {}
            return json.loads(raw)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        detail = raw[:500] if raw else exc.reason
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def _try_get_json(base_url: str, path: str, timeout: float = 1.25) -> tuple[bool, Any, str]:
    url = f"{base_url.rstrip('/')}{path}"
    try:
        return True, _request_json("GET", url, timeout=timeout), "ok"
    except Exception as exc:
        return False, None, f"{exc.__class__.__name__}: {exc}"


def _post_json(url: str, payload: dict[str, Any], timeout: float = 60.0) -> dict[str, Any]:
    result = _request_json("POST", url, payload=payload, timeout=timeout)
    return result if isinstance(result, dict) else {"result": result}


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


def _format_gib(value: Any) -> str | None:
    try:
        number = float(value)
    except Exception:
        return None
    if number <= 0:
        return None
    return f"{number / (1024 ** 3):.1f} GB"


def _memory_label(free_bytes: Any, total_bytes: Any) -> str:
    total = _format_gib(total_bytes)
    free = _format_gib(free_bytes)
    if not total:
        return "WIP"
    if not free:
        return total
    try:
        used = float(total_bytes) - float(free_bytes)
        return f"{used / (1024 ** 3):.1f} / {float(total_bytes) / (1024 ** 3):.1f} GB"
    except Exception:
        return f"{free} free / {total}"


_SAMPLER_ALIASES = {
    "euler a": "euler_a",
    "euler_a": "euler_a",
    "euler": "euler",
    "ddim": "ddim",
    "unipc": "unipc",
    "uni_pc": "unipc",
    "dpm++ 2m karras": "dpmpp_2m",
    "dpmpp 2m karras": "dpmpp_2m",
    "dpm++ 2m": "dpmpp_2m",
    "dpmpp_2m": "dpmpp_2m",
}


def _normalize_sampler(value: Any) -> str:
    raw = str(value or "euler_a").strip()
    lowered = raw.lower().replace("-", " ").replace("_", " ")
    return _SAMPLER_ALIASES.get(lowered, raw)


def _first_text(item: dict[str, Any], keys: list[str], default: str = "") -> str:
    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def _normalize_models(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        title = _first_text(item, ["title", "model_name", "name", "id", "filename"], f"Model {index + 1}")
        model_id = _first_text(item, ["id", "key", "title", "model_name", "name", "filename"], title)
        normalized.append(
            {
                "id": model_id,
                "title": title,
                "path": _first_text(item, ["path", "filename"], ""),
                "hash": _first_text(item, ["hash", "sha256"], ""),
                "raw": item,
            }
        )
    return normalized


def _normalize_samplers(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        title = _first_text(item, ["label", "name", "title", "id"], "Sampler")
        sampler_id = _normalize_sampler(_first_text(item, ["id", "name", "label"], title))
        normalized.append({"id": sampler_id, "title": title, "raw": item})
    return normalized


def _safe_percent(value: Any, *, fraction: bool = False) -> int:
    try:
        number = float(value)
    except Exception:
        return 0
    if fraction and number <= 1:
        number *= 100
    return max(0, min(100, int(number)))


class SecondGuiHandler(SimpleHTTPRequestHandler):
    """Static preview shell plus tiny local JSON bridge.

    The second GUI is intentionally conservative. It can display the new visual
    shell immediately, but unfinished backend routes return explicit WIP payloads
    instead of pretending a feature is wired.
    """

    server_version = "AIWFSecondGUI/0.2"
    backend_url = DEFAULT_BACKEND_URL.rstrip("/")
    proxy_enabled = _env_bool("AIWF_SECOND_GUI_PROXY", False)
    proxy_mode = DEFAULT_PROXY_MODE if DEFAULT_PROXY_MODE in {"auto", "native", "sdapi"} else "auto"

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

    def _backend_health(self) -> tuple[bool, str]:
        ok, payload, note = _try_get_json(self.backend_url, "/api/v1/health", timeout=0.75)
        if ok:
            status = payload.get("status") if isinstance(payload, dict) else "ok"
            return True, f"/api/v1/health {status}"
        alive, root_note = _get_url(self.backend_url)
        return alive, root_note if alive else note

    def _backend_catalog(self) -> dict[str, Any]:
        native_models_ok, native_models, native_models_note = _try_get_json(self.backend_url, "/api/v1/models")
        sd_models_ok, sd_models, sd_models_note = _try_get_json(self.backend_url, "/sdapi/v1/sd-models")
        native_samplers_ok, native_samplers, native_samplers_note = _try_get_json(self.backend_url, "/api/v1/samplers")
        sd_samplers_ok, sd_samplers, sd_samplers_note = _try_get_json(self.backend_url, "/sdapi/v1/samplers")
        loras_ok, loras, loras_note = _try_get_json(self.backend_url, "/api/v1/loras", timeout=1.0)
        vae_ok, vaes, vaes_note = _try_get_json(self.backend_url, "/api/v1/vae", timeout=1.0)

        models = _normalize_models(native_models if native_models_ok else sd_models)
        samplers = _normalize_samplers(native_samplers if native_samplers_ok else sd_samplers)
        return {
            "ok": native_models_ok or sd_models_ok or native_samplers_ok or sd_samplers_ok,
            "backend_url": self.backend_url,
            "models": models,
            "samplers": samplers,
            "loras": loras if loras_ok and isinstance(loras, list) else [],
            "vaes": vaes if vae_ok and isinstance(vaes, list) else [],
            "notes": {
                "native_models": "ok" if native_models_ok else native_models_note,
                "sd_models": "ok" if sd_models_ok else sd_models_note,
                "native_samplers": "ok" if native_samplers_ok else native_samplers_note,
                "sd_samplers": "ok" if sd_samplers_ok else sd_samplers_note,
                "loras": "ok" if loras_ok else loras_note,
                "vaes": "ok" if vae_ok else vaes_note,
            },
        }

    def _progress_status(self) -> dict[str, Any]:
        native_ok, native, native_note = _try_get_json(self.backend_url, "/api/v1/progress", timeout=0.75)
        if native_ok and isinstance(native, dict):
            return {
                "ok": True,
                "source": "native",
                "state": str(native.get("state", "idle")),
                "progress": _safe_percent(native.get("progress", 0)),
                "message": str(native.get("message") or ""),
                "step": native.get("step", 0),
                "total_steps": native.get("total_steps", 0),
                "raw": native,
            }

        sd_ok, sd, sd_note = _try_get_json(self.backend_url, "/sdapi/v1/progress", timeout=0.75)
        if sd_ok and isinstance(sd, dict):
            state = sd.get("state") if isinstance(sd.get("state"), dict) else {}
            return {
                "ok": True,
                "source": "sdapi",
                "state": str(state.get("state") or state.get("job") or "idle"),
                "progress": _safe_percent(sd.get("progress", 0), fraction=True),
                "message": str(state.get("message") or ""),
                "step": state.get("step", 0),
                "total_steps": state.get("total_steps", 0),
                "raw": sd,
            }

        return {
            "ok": False,
            "source": "none",
            "state": "offline",
            "progress": 0,
            "message": native_note or sd_note,
            "step": 0,
            "total_steps": 0,
        }

    def _runtime_status(self) -> dict[str, Any]:
        alive, note = self._backend_health()
        memory_ok, memory, memory_note = _try_get_json(self.backend_url, "/sdapi/v1/memory", timeout=0.75)
        options_ok, options, _options_note = _try_get_json(self.backend_url, "/sdapi/v1/options", timeout=0.75)
        optimization_ok, optimization, _optimization_note = _try_get_json(
            self.backend_url,
            "/api/v1/optimization/status",
            timeout=0.75,
        )
        progress = self._progress_status()

        ram_label = "WIP"
        vram_label = os.environ.get("AIWF_SECOND_GUI_VRAM", "WIP")
        if memory_ok and isinstance(memory, dict):
            ram = memory.get("ram") if isinstance(memory.get("ram"), dict) else {}
            cuda = memory.get("cuda") if isinstance(memory.get("cuda"), dict) else {}
            system = cuda.get("system") if isinstance(cuda.get("system"), dict) else {}
            ram_label = _memory_label(ram.get("free"), ram.get("total"))
            detected_vram = _memory_label(system.get("free"), system.get("total"))
            if detected_vram != "WIP":
                vram_label = detected_vram

        loaded_model = os.environ.get("AIWF_SECOND_GUI_MODEL", "sdxl-base-1.0 (Diffusers)")
        if options_ok and isinstance(options, dict):
            loaded_model = str(options.get("sd_model_checkpoint") or loaded_model)

        profile_id = ""
        runtime_flags: dict[str, Any] = {}
        if optimization_ok and isinstance(optimization, dict):
            profile_id = str(optimization.get("profile_id") or optimization.get("requested_profile_id") or "")
            raw_flags = optimization.get("runtime_flags")
            runtime_flags = raw_flags if isinstance(raw_flags, dict) else {}

        attention = os.environ.get("AIWF_SECOND_GUI_ATTENTION") or runtime_flags.get("attention_backend") or profile_id
        precision = os.environ.get("AIWF_SECOND_GUI_PRECISION") or (
            "FP8" if runtime_flags.get("fp8") or runtime_flags.get("fp8_quant") else "FP16"
        )

        return {
            "ok": True,
            "engine": "Local Engine",
            "engine_state": "Ready" if alive else "WIP bridge",
            "backend_url": self.backend_url,
            "backend_reachable": alive,
            "backend_note": note if alive else memory_note,
            "proxy_enabled": self.proxy_enabled,
            "proxy_mode": self.proxy_mode,
            "backend": "AIWF Studio API" if alive else "Second GUI shell",
            "device": os.environ.get("AIWF_SECOND_GUI_DEVICE", "Detected by main AIWF runtime"),
            "precision": precision,
            "attention": attention or "WIP / runtime-reported",
            "max_resolution": os.environ.get("AIWF_SECOND_GUI_MAX_RES", "1024 x 1024"),
            "vram": vram_label,
            "ram": ram_label,
            "storage": os.environ.get("AIWF_SECOND_GUI_STORAGE", "WIP"),
            "cpu": os.environ.get("AIWF_SECOND_GUI_CPU", "WIP"),
            "loaded_model": loaded_model,
            "progress": progress,
            "queue_text": "1 task" if progress.get("state") in {"queued", "running"} else "0 tasks",
        }

    def _feature_list(self) -> dict[str, Any]:
        return {
            "ok": True,
            "features": [
                {"id": "image", "label": "Image", "state": "wired_proxy_optional", "route": "/api/generate"},
                {"id": "catalog", "label": "Model catalog", "state": "wired_proxy_optional", "route": "/api/catalog"},
                {"id": "progress", "label": "Progress", "state": "wired_proxy_optional", "route": "/api/progress"},
                {"id": "interrupt", "label": "Interrupt", "state": "wired_proxy_optional", "route": "/api/interrupt"},
                {"id": "video", "label": "Video", "state": "wip", "route": None},
                {"id": "inpaint", "label": "Inpaint", "state": "wip", "route": None},
                {"id": "data", "label": "Data", "state": "wip", "route": None},
                {"id": "batch", "label": "Batch", "state": "wip", "route": None},
                {"id": "workflows", "label": "Workflows", "state": "wip", "route": None},
                {"id": "logs", "label": "Logs", "state": "wip", "route": None},
            ],
        }

    def _native_payload(self, request: dict[str, Any]) -> dict[str, Any]:
        checkpoint_id = str(request.get("checkpoint_id") or "").strip() or None
        return {
            "prompt": str(request.get("prompt", "")),
            "negative_prompt": str(request.get("negative_prompt", "")),
            "steps": _coerce_int(request.get("steps"), 30),
            "cfg_scale": _coerce_float(request.get("cfg_scale"), 7.0),
            "sampler": _normalize_sampler(request.get("sampler", "euler_a")),
            "scheduler": str(request.get("scheduler", "automatic") or "automatic"),
            "seed": _coerce_int(request.get("seed"), -1),
            "width": _coerce_int(request.get("width"), 1024),
            "height": _coerce_int(request.get("height"), 1024),
            "batch_size": _coerce_int(request.get("batch_size"), 1),
            "batch_count": _coerce_int(request.get("batch_count"), 1),
            "checkpoint_id": checkpoint_id,
            "save_images": True,
        }

    def _sdapi_payload(self, request: dict[str, Any]) -> dict[str, Any]:
        checkpoint_id = str(request.get("checkpoint_id") or "").strip()
        payload = {
            "prompt": str(request.get("prompt", "")),
            "negative_prompt": str(request.get("negative_prompt", "")),
            "steps": _coerce_int(request.get("steps"), 30),
            "cfg_scale": _coerce_float(request.get("cfg_scale"), 7.0),
            "sampler_name": str(request.get("sampler_label") or request.get("sampler") or "DPM++ 2M Karras"),
            "seed": _coerce_int(request.get("seed"), -1),
            "width": _coerce_int(request.get("width"), 1024),
            "height": _coerce_int(request.get("height"), 1024),
            "batch_size": _coerce_int(request.get("batch_size"), 1),
            "n_iter": _coerce_int(request.get("batch_count"), 1),
        }
        if checkpoint_id:
            payload["override_settings"] = {"sd_model_checkpoint": checkpoint_id}
        return payload

    def _try_generate_native(self, request: dict[str, Any]) -> dict[str, Any]:
        payload = self._native_payload(request)
        response = _post_json(f"{self.backend_url}/api/v1/txt2img", payload, timeout=300.0)
        return {"ok": True, "source": "native", "response": response, "payload": payload}

    def _try_generate_sdapi(self, request: dict[str, Any]) -> dict[str, Any]:
        payload = self._sdapi_payload(request)
        response = _post_json(f"{self.backend_url}/sdapi/v1/txt2img", payload, timeout=300.0)
        return {"ok": True, "source": "sdapi", "response": response, "payload": payload}

    def _handle_generate(self) -> None:
        request = _read_json_body(self)
        if not self.proxy_enabled:
            self._send_wip(
                "Image generation",
                "Second GUI is running. Launch with --proxy to route Generate into the running AIWF backend. Missing backend routes stay marked WIP instead of fake-success.",
            )
            return

        errors: list[str] = []
        modes = ["native", "sdapi"] if self.proxy_mode == "auto" else [self.proxy_mode]
        for mode in modes:
            try:
                result = self._try_generate_native(request) if mode == "native" else self._try_generate_sdapi(request)
                self._send_json(result)
                return
            except Exception as exc:
                errors.append(f"{mode}: {exc.__class__.__name__}: {exc}")

        self._send_json(
            {
                "ok": False,
                "wip": True,
                "feature": "Image generation proxy",
                "message": "Proxy mode was enabled, but no backend generation route accepted the request.",
                "errors": errors,
                "mode": self.proxy_mode,
            },
            status=502,
        )

    def _handle_interrupt(self) -> None:
        if not self.proxy_enabled:
            self._send_wip("Interrupt", "Launch with --proxy before interrupt can call the main backend.")
            return

        errors: list[str] = []
        for path in ("/api/v1/interrupt", "/sdapi/v1/interrupt"):
            try:
                response = _post_json(f"{self.backend_url}{path}", {}, timeout=5.0)
                self._send_json({"ok": True, "source": path, "response": response})
                return
            except Exception as exc:
                errors.append(f"{path}: {exc.__class__.__name__}: {exc}")
        self._send_json({"ok": False, "wip": True, "feature": "Interrupt", "errors": errors}, status=502)

    def do_GET(self) -> None:  # noqa: N802 - stdlib API
        path = urllib.parse.urlparse(self.path).path
        if path in {"", "/"}:
            self.path = "/index.html"
            return super().do_GET()
        if path == "/api/runtime/status":
            return self._send_json(self._runtime_status())
        if path == "/api/features":
            return self._send_json(self._feature_list())
        if path == "/api/catalog":
            return self._send_json(self._backend_catalog())
        if path == "/api/progress":
            return self._send_json(self._progress_status())
        if path.startswith("/api/wip"):
            return self._send_wip("Second GUI feature")
        return super().do_GET()

    def do_POST(self) -> None:  # noqa: N802 - stdlib API
        path = urllib.parse.urlparse(self.path).path
        if path == "/api/generate":
            return self._handle_generate()
        if path == "/api/interrupt":
            return self._handle_interrupt()
        if path.startswith("/api/wip"):
            body = _read_json_body(self)
            return self._send_wip(str(body.get("feature", "Second GUI feature")))
        self._send_json({"ok": False, "error": "Not found"}, status=404)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the AIWF Studio Second GUI preview shell.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("AIWF_SECOND_GUI_PORT", "8770")))
    parser.add_argument("--backend", default=DEFAULT_BACKEND_URL, help="Main AIWF backend URL for status/proxy checks.")
    parser.add_argument("--proxy", action="store_true", help="Proxy Generate, catalog, progress, and interrupt to the main backend.")
    parser.add_argument(
        "--proxy-mode",
        choices=["auto", "native", "sdapi"],
        default=DEFAULT_PROXY_MODE if DEFAULT_PROXY_MODE in {"auto", "native", "sdapi"} else "auto",
        help="Generation route preference when --proxy is enabled.",
    )
    parser.add_argument("--listen", action="store_true", help="Bind to 0.0.0.0 instead of 127.0.0.1.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser automatically.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not STATIC_DIR.exists():
        raise SystemExit(f"Second GUI static folder missing: {STATIC_DIR}")

    SecondGuiHandler.backend_url = str(args.backend).rstrip("/")
    SecondGuiHandler.proxy_enabled = bool(args.proxy or _env_bool("AIWF_SECOND_GUI_PROXY", False))
    SecondGuiHandler.proxy_mode = str(args.proxy_mode or "auto").lower()

    host = "0.0.0.0" if args.listen else "127.0.0.1"
    url = f"http://127.0.0.1:{args.port}/"
    print("[AIWF Second GUI] Starting preview shell", flush=True)
    print(f"[AIWF Second GUI] Static root: {STATIC_DIR}", flush=True)
    print(f"[AIWF Second GUI] Backend: {SecondGuiHandler.backend_url}", flush=True)
    print(f"[AIWF Second GUI] Proxy generate/catalog/progress: {SecondGuiHandler.proxy_enabled}", flush=True)
    print(f"[AIWF Second GUI] Proxy mode: {SecondGuiHandler.proxy_mode}", flush=True)
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
