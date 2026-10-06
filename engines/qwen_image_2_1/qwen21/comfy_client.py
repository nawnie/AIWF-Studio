"""Minimal ComfyUI HTTP + WebSocket client (stdlib HTTP, websocket-client for progress).

Routes used (ComfyUI server):
  GET  /system_stats           server + device info
  GET  /object_info            node schemas (used for validation and model lists)
  GET  /models/{folder}        file list for a models subfolder
  POST /upload/image           multipart: image, overwrite, type, subfolder
  POST /prompt                 {"prompt": {...}, "client_id": "..."}
  GET  /history/{prompt_id}    outputs once finished
  GET  /view?filename=&subfolder=&type=
  POST /interrupt, POST /free
  WS   /ws?clientId=           status / executing / progress / executed / execution_error
"""
from __future__ import annotations

import json
import mimetypes
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib import error, parse, request


class ComfyError(RuntimeError):
    pass


@dataclass
class OutputImage:
    filename: str
    subfolder: str = ""
    type: str = "output"
    node_id: str = ""

    def view_params(self) -> dict[str, str]:
        return {"filename": self.filename, "subfolder": self.subfolder, "type": self.type}


@dataclass
class RunResult:
    prompt_id: str
    images: list[OutputImage] = field(default_factory=list)
    text_outputs: dict[str, list[str]] = field(default_factory=dict)  # e.g. rewritten prompts
    elapsed_s: float = 0.0
    error: str | None = None


ProgressCb = Callable[[int, int, str], None]  # value, max, node_id
StatusCb = Callable[[str], None]
PreviewCb = Callable[[bytes], None]


class ComfyClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8188", timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client_id = uuid.uuid4().hex

    # ------------------------------------------------------------------ HTTP
    def _url(self, path: str, params: dict[str, Any] | None = None) -> str:
        url = f"{self.base_url}{path}"
        if params:
            url += "?" + parse.urlencode(params)
        return url

    def _get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        try:
            with request.urlopen(self._url(path, params), timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            raise ComfyError(f"GET {path} -> HTTP {exc.code}: {body[:500]}") from exc
        except error.URLError as exc:
            raise ComfyError(f"ComfyUI not reachable at {self.base_url} ({exc.reason})") from exc

    def _post_json(self, path: str, payload: Any) -> Any:
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(self._url(path), data=data, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
                return json.loads(raw.decode("utf-8")) if raw else {}
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            raise ComfyError(f"POST {path} -> HTTP {exc.code}: {body[:2000]}") from exc
        except error.URLError as exc:
            raise ComfyError(f"ComfyUI not reachable at {self.base_url} ({exc.reason})") from exc

    # ------------------------------------------------------------------ info
    def is_up(self) -> bool:
        try:
            self.system_stats()
            return True
        except ComfyError:
            return False

    def system_stats(self) -> dict[str, Any]:
        return self._get_json("/system_stats")

    def server_summary(self) -> str:
        stats = self.system_stats()
        system = stats.get("system", {})
        lines = [f"ComfyUI {system.get('comfyui_version', '?')} | python {system.get('python_version', '?').split()[0]} "
                 f"| torch {system.get('pytorch_version', '?')}"]
        for dev in stats.get("devices", []):
            total = dev.get("vram_total", 0) / 1024**3
            free = dev.get("vram_free", 0) / 1024**3
            lines.append(f"{dev.get('name', 'device')} | VRAM {free:.1f} GB free of {total:.1f} GB")
        return "\n".join(lines)

    def comfyui_version(self) -> str:
        return str(self.system_stats().get("system", {}).get("comfyui_version", ""))

    def object_info(self) -> dict[str, Any]:
        return self._get_json("/object_info")

    def node_exists(self, class_type: str) -> bool:
        try:
            self._get_json(f"/object_info/{class_type}")
            return True
        except ComfyError:
            return False

    def list_models(self, folder: str) -> list[str]:
        """Files under models/<folder> (diffusion_models, text_encoders, vae, loras, model_patches)."""
        try:
            items = self._get_json(f"/models/{folder}")
            if isinstance(items, list):
                return sorted(str(i) for i in items)
        except ComfyError:
            pass
        # fallback: pull the combo list out of the loader node schema
        loader = {"diffusion_models": ("UNETLoader", "unet_name"), "text_encoders": ("CLIPLoader", "clip_name"),
                  "vae": ("VAELoader", "vae_name"), "loras": ("LoraLoaderModelOnly", "lora_name"),
                  "model_patches": ("ModelPatchLoader", "name")}.get(folder)
        if loader is None:
            return []
        try:
            info = self._get_json(f"/object_info/{loader[0]}")
            options = info[loader[0]]["input"]["required"][loader[1]][0]
            return sorted(str(o) for o in options)
        except (ComfyError, KeyError, IndexError, TypeError):
            return []

    # --------------------------------------------------------------- uploads
    def upload_image(self, path: str | Path, subfolder: str = "qwen21", overwrite: bool = True,
                     name: str | None = None) -> str:
        """Upload a file to ComfyUI's input folder. Returns the name LoadImage expects."""
        path = Path(path)
        if not path.is_file():
            raise ComfyError(f"upload: file not found {path}")
        filename = name or path.name
        boundary = f"----qwen21{uuid.uuid4().hex}"
        ctype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        body = bytearray()

        def part(field_name: str, value: str) -> None:
            body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field_name}\"\r\n\r\n{value}\r\n"
                        .encode("utf-8"))

        part("overwrite", "true" if overwrite else "false")
        part("type", "input")
        if subfolder:
            part("subfolder", subfolder)
        body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{filename}\"\r\n"
                    f"Content-Type: {ctype}\r\n\r\n".encode("utf-8"))
        body.extend(path.read_bytes())
        body.extend(f"\r\n--{boundary}--\r\n".encode("utf-8"))
        req = request.Request(self._url("/upload/image"), data=bytes(body), method="POST",
                              headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        try:
            with request.urlopen(req, timeout=max(self.timeout, 120)) as resp:
                info = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            raise ComfyError(f"upload failed: HTTP {exc.code} {exc.read().decode('utf-8', 'replace')[:300]}") from exc
        except error.URLError as exc:
            raise ComfyError(f"ComfyUI not reachable at {self.base_url} ({exc.reason})") from exc
        name_out = info.get("name", filename)
        sub = info.get("subfolder", subfolder)
        return f"{sub}/{name_out}" if sub else name_out

    # ------------------------------------------------------------------ jobs
    def queue_prompt(self, prompt: dict[str, Any]) -> str:
        resp = self._post_json("/prompt", {"prompt": prompt, "client_id": self.client_id})
        if "prompt_id" not in resp:
            raise ComfyError(f"queue failed: {json.dumps(resp)[:2000]}")
        return str(resp["prompt_id"])

    def history(self, prompt_id: str) -> dict[str, Any]:
        data = self._get_json(f"/history/{prompt_id}")
        return data.get(prompt_id, {}) if isinstance(data, dict) else {}

    def interrupt(self) -> None:
        self._post_json("/interrupt", {})

    def free(self, unload_models: bool = True, free_memory: bool = True) -> None:
        self._post_json("/free", {"unload_models": unload_models, "free_memory": free_memory})

    def view(self, image: OutputImage) -> bytes:
        try:
            with request.urlopen(self._url("/view", image.view_params()), timeout=max(self.timeout, 120)) as resp:
                return resp.read()
        except error.HTTPError as exc:
            raise ComfyError(f"view failed: HTTP {exc.code}") from exc
        except error.URLError as exc:
            raise ComfyError(f"ComfyUI not reachable at {self.base_url} ({exc.reason})") from exc

    def download_output(self, image: OutputImage, dest_dir: str | Path) -> Path:
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / image.filename
        dest.write_bytes(self.view(image))
        return dest

    @staticmethod
    def _collect_outputs(hist: dict[str, Any], result: RunResult) -> None:
        for node_id, out in (hist.get("outputs") or {}).items():
            for img in out.get("images", []) or []:
                result.images.append(OutputImage(img.get("filename", ""), img.get("subfolder", ""),
                                                 img.get("type", "output"), node_id))
            for key in ("text", "string", "value"):
                if key in out and isinstance(out[key], list):
                    result.text_outputs[node_id] = [str(t) for t in out[key]]

    def run(self, prompt: dict[str, Any], *, on_progress: ProgressCb | None = None,
            on_status: StatusCb | None = None, on_preview: PreviewCb | None = None,
            cancel: threading.Event | None = None, poll_interval: float = 1.0) -> RunResult:
        """Queue a prompt and wait for it. Uses the websocket for progress, falls back to polling."""
        started = time.time()
        prompt_id = self.queue_prompt(prompt)
        result = RunResult(prompt_id=prompt_id)
        if on_status:
            on_status(f"queued {prompt_id}")
        ws = None
        try:
            import websocket  # type: ignore

            ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
            ws = websocket.create_connection(f"{ws_url}/ws?clientId={self.client_id}", timeout=5)
        except Exception as exc:  # pragma: no cover - depends on environment
            if on_status:
                on_status(f"websocket unavailable ({exc}); polling history")
            ws = None

        finished = False
        while not finished:
            if cancel is not None and cancel.is_set():
                try:
                    self.interrupt()
                finally:
                    result.error = "cancelled"
                break
            if ws is not None:
                try:
                    message = ws.recv()
                except Exception:
                    message = None
                    time.sleep(poll_interval)
                if isinstance(message, (bytes, bytearray)) and len(message) > 8:
                    if on_preview and int.from_bytes(message[:4], "big") == 1:
                        on_preview(bytes(message[8:]))
                elif isinstance(message, str):
                    try:
                        event = json.loads(message)
                    except json.JSONDecodeError:
                        event = {}
                    etype, data = event.get("type"), event.get("data", {}) or {}
                    if data.get("prompt_id") not in (None, prompt_id) and etype != "status":
                        continue
                    if etype == "progress" and on_progress:
                        on_progress(int(data.get("value", 0)), int(data.get("max", 0)), str(data.get("node") or ""))
                    elif etype == "executing":
                        node = data.get("node")
                        if on_status:
                            on_status("sampling" if node is None else f"running node {node}")
                        if node is None and data.get("prompt_id") == prompt_id:
                            finished = True
                    elif etype == "execution_error":
                        result.error = f"{data.get('exception_type', 'error')}: {data.get('exception_message', '')}"
                        finished = True
                    elif etype in ("execution_success",):
                        finished = True
            else:
                time.sleep(poll_interval)
            hist = self.history(prompt_id)
            if hist:
                status = hist.get("status", {}) or {}
                if status.get("status_str") == "error":
                    msgs = [m for m in status.get("messages", []) if m and m[0] == "execution_error"]
                    if msgs:
                        detail = msgs[-1][1]
                        result.error = f"{detail.get('exception_type', 'error')}: {detail.get('exception_message', '')}"
                    else:
                        result.error = "execution error (see ComfyUI console)"
                    finished = True
                elif status.get("completed") or hist.get("outputs"):
                    finished = True
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        hist = self.history(prompt_id)
        self._collect_outputs(hist, result)
        result.elapsed_s = time.time() - started
        return result
