"""Curated local ComfyUI image and video jobs for the AES Studio preview.

The user chooses compatible files already registered with their own ComfyUI.
No model or node packages are installed. Only the four bundled graphs can run.
"""
from __future__ import annotations

import io
import json
import random
import re
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import httpx
from PIL import Image

from aiwf.services.image_contract import MAX_JOBS_KEPT, MAX_PROMPT_CHARS, MAX_SEED
from aiwf.services.image_jobs import ImageJobs, _execution_error, _rejection_message
from aiwf.services.unified_bridge import BridgeError, SCHEMA_VERSION, _utc_now

_DIR = Path(__file__).resolve().parent / "workflows"
_MAX_INPUT_BYTES = 24 * 1024 * 1024
_MAX_OUTPUT_BYTES = 256 * 1024 * 1024
_ALLOWED_INPUT = {".png", ".jpg", ".jpeg", ".webp"}

# field: (node ID, input name, node class, selectable input name)
_PRESETS: dict[str, dict[str, Any]] = {
    "qwen21-edit": {
        "label": "Qwen Image 2.1 edit", "template": "aes_qwen21_edit.json", "save": "461",
        "prompt": ("459:474", "prompt"), "seed": ("459:458", "seed"), "image": ("470", "image"),
        "fields": {"model": ("459:451", "unet_name", "UNETLoader", "unet_name"),
                   "encoder": ("459:453", "clip_name", "CLIPLoader", "clip_name"),
                   "vae": ("459:454", "vae_name", "VAELoader", "vae_name")},
        "suggest": {"model": ("qwen", "2.1"), "encoder": ("qwen", "vl"), "vae": ("qwen", "vae")},
    },
    "moss-image": {
        "label": "MOSS Image", "template": "aes_moss_image.json", "save": "7",
        "prompt": ("2", "text"), "seed": ("5", "seed"), "image": None,
        "fields": {"model": ("1", "unet_name", "UNETLoader", "unet_name"),
                   "encoder": ("8", "clip_name", "CLIPLoader", "clip_name"),
                   "vae": ("9", "vae_name", "VAELoader", "vae_name")},
        "suggest": {"model": ("moss", "sft"), "encoder": ("qwen", "3.5"), "vae": ("flux2", "vae")},
    },
    "wan22-i2v": {
        "label": "Wan 2.2 5B image to video", "template": "aes_wan22_i2v.json", "save": "58",
        "prompt": ("6", "text"), "seed": ("3", "seed"), "image": ("101", "image"),
        "fields": {"model": ("37", "unet_name", "UNETLoader", "unet_name"),
                   "encoder": ("38", "clip_name", "CLIPLoader", "clip_name"),
                   "vae": ("39", "vae_name", "VAELoader", "vae_name")},
        "suggest": {"model": ("wan", "2.2", "5b"), "encoder": ("umt5",), "vae": ("wan", "2.2", "vae")},
    },
    "ltx25-i2v": {
        "label": "LTX 2.5 image to video (GGUF)", "template": "aes_ltx25_i2v.json", "save": "1",
        "prompt": ("13", "text"), "seed": ("10", "noise_seed"), "image": ("25", "image"),
        "fields": {"model": ("17", "unet_name", "UnetLoaderGGUF", "unet_name"),
                   "encoder": ("20", "clip_name", "CLIPLoader", "clip_name"),
                   "vae": ("18", "vae_name", "VAELoader", "vae_name"),
                   "audio_vae": ("19", "vae_name", "VAELoader", "vae_name")},
        "suggest": {"model": ("ltx", "2.5", "distilled"), "encoder": ("ltx", "2.5"),
                    "vae": ("ltx", "2.5", "video"), "audio_vae": ("ltx", "2.5", "audio")},
    },
}


def _preset(name: str) -> dict[str, Any]:
    if name not in _PRESETS:
        raise BridgeError(422, "unknown_preset", "Choose one of the supported creative model workflows.")
    return _PRESETS[name]


def _graph(preset: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads((_DIR / preset["template"]).read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not value:
            raise ValueError("empty graph")
        return value
    except (OSError, ValueError, TypeError) as exc:
        raise BridgeError(500, "workflow_missing", "The AES Studio workflow is missing or damaged.") from exc


def _choice_list(info: dict[str, Any], node: str, field: str) -> list[str]:
    try:
        data = info[node]["input"]["required"][field]
        choices = data[0]
        if not isinstance(choices, list):
            return []
        return [v for v in choices if isinstance(v, str) and v and len(v) <= 1024]
    except (KeyError, IndexError, TypeError):
        return []


class CreativeJobs(ImageJobs):
    """Reuses the existing job watcher and job-specific cancellation behavior."""

    def __init__(self, *, client: Callable[[bool], httpx.Client], comfy_url: str) -> None:
        super().__init__(client=client, comfy_url=comfy_url, timeout_seconds=float("inf"))

    def inventory(self, name: str) -> dict[str, Any]:
        preset = _preset(name)
        graph = _graph(preset)
        try:
            with self._client(False) as client:
                response = client.get("/object_info")
                response.raise_for_status()
                info = response.json()
            if not isinstance(info, dict):
                raise ValueError("invalid inventory")
        except (httpx.HTTPError, ValueError) as exc:
            raise BridgeError(503, "comfy_unavailable", "Start your local ComfyUI installation, then refresh. AES Studio does not install it or its nodes.") from exc
        required = sorted({str(n.get("class_type", "")) for n in graph.values()})
        missing = [node for node in required if node not in info]
        fields, suggested = {}, {}
        for field, (_, _, node, item) in preset["fields"].items():
            choices = _choice_list(info, node, item)
            fields[field] = choices
            tokens = preset["suggest"].get(field, ())
            matches = [v for v in choices if all(t in v.casefold() for t in tokens)] if tokens else []
            suggested[field] = matches[0] if len(matches) == 1 else ""
        return {"schema_version": SCHEMA_VERSION, "preset": name, "label": preset["label"],
                "needs_image": bool(preset["image"]), "missing_nodes": missing,
                "fields": fields, "suggested": suggested}

    def start_creative(self, *, name: str, prompt: str, selections: dict[str, str], input_path: str | None,
                       output_root: Path | None, on_finished: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
        preset = _preset(name)
        prompt = prompt.strip()
        if not prompt or len(prompt) > MAX_PROMPT_CHARS:
            raise BridgeError(422, "invalid_prompt", f"Enter a prompt of 1 to {MAX_PROMPT_CHARS} characters.")
        if output_root is None or not Path(output_root).is_dir():
            raise BridgeError(503, "no_output_root", "The AES Studio output folder is unavailable.")
        inventory = self.inventory(name)
        if inventory["missing_nodes"]:
            raise BridgeError(503, "missing_nodes", "Missing ComfyUI nodes: " + ", ".join(inventory["missing_nodes"]))
        if set(selections) != set(preset["fields"]):
            raise BridgeError(422, "invalid_selection", "Choose every required model component.")
        for field, selected in selections.items():
            if selected not in inventory["fields"][field]:
                raise BridgeError(422, "invalid_selection", f"Refresh choices and choose an available {field.replace('_', ' ')}.")
        payload = None
        suffix = None
        if preset["image"]:
            try:
                path = Path(input_path or "")
                if not path.is_absolute() or path.suffix.lower() not in _ALLOWED_INPUT or not path.is_file():
                    raise ValueError("Choose a PNG, JPEG, or WebP image.")
                if path.stat().st_size > _MAX_INPUT_BYTES:
                    raise ValueError("The input image exceeds 24 MB.")
                payload = path.read_bytes()
                if not payload or len(payload) > _MAX_INPUT_BYTES:
                    raise ValueError("The input image is empty or exceeds 24 MB.")
                with Image.open(io.BytesIO(payload)) as image:
                    image.verify()
                suffix = path.suffix.lower()
            except (OSError, ValueError, Image.DecompressionBombError) as exc:
                raise BridgeError(422, "invalid_input_image", f"Could not use the selected image: {exc}") from exc
        graph = _graph(preset)
        seed = random.randint(0, MAX_SEED)
        graph[preset["prompt"][0]]["inputs"][preset["prompt"][1]] = prompt
        graph[preset["seed"][0]]["inputs"][preset["seed"][1]] = seed
        for field, (node, key, _, _) in preset["fields"].items():
            graph[node]["inputs"][key] = selections[field]
        try:
            with self._client(False) as client:
                if payload is not None:
                    filename = "aes-studio-" + uuid.uuid4().hex + suffix
                    upload = client.post("/upload/image", files={"image": (filename, payload, "application/octet-stream")}, data={"type": "input", "overwrite": "false"})
                    upload.raise_for_status()
                    uploaded = upload.json()
                    name_on_server, subfolder = uploaded.get("name"), uploaded.get("subfolder", "")
                    if name_on_server != filename or subfolder:
                        raise ValueError("Unexpected ComfyUI upload response")
                    graph[preset["image"][0]]["inputs"][preset["image"][1]] = filename
                response = client.post("/prompt", json={"prompt": graph, "client_id": "aes-studio-" + uuid.uuid4().hex[:12]})
        except (httpx.HTTPError, ValueError, AttributeError) as exc:
            raise BridgeError(503, "comfy_unavailable", "The selected image or workflow could not be sent to local ComfyUI.") from exc
        if response.status_code == 400:
            raise BridgeError(422, "workflow_rejected", _rejection_message(response))
        if response.status_code >= 400:
            raise BridgeError(502, "comfy_error", f"ComfyUI returned HTTP {response.status_code} when queuing the workflow.")
        try:
            prompt_id = response.json().get("prompt_id")
        except (ValueError, AttributeError):
            prompt_id = None
        if not isinstance(prompt_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", prompt_id):
            raise BridgeError(502, "comfy_bad_response", "ComfyUI did not return a usable job ID.")
        job = {"job_id": "img-" + uuid.uuid4().hex[:12], "prompt_id": prompt_id, "preset": name,
               "save_node": preset["save"], "state": "queued", "prompt": prompt, "seed": seed,
               "output_root": Path(output_root), "queue_position": None, "created_at": _utc_now(),
               "created_monotonic": time.monotonic(), "started_at": None, "finished_at": None,
               "error": None, "files": [], "on_finished": on_finished}
        with self._lock:
            self._jobs[job["job_id"]] = job
            self._prune_locked()
        result = {"schema_version": SCHEMA_VERSION, "job": self._view(job)}
        watcher = threading.Thread(target=self._watch_guarded, args=(job["job_id"],), name="creative-" + job["job_id"], daemon=True)
        try:
            watcher.start()
        except Exception:
            self._watch_guarded(job["job_id"])
        return result

    def cancel(self, job_id: str) -> dict[str, Any]:
        # Older ComfyUI versions may treat /interrupt as global even when a
        # prompt_id is sent. Never risk interrupting someone else's running job.
        job = self._get(job_id)
        if job["state"] in {"done", "failed", "cancelled"}:
            return {"schema_version": SCHEMA_VERSION, "job": self._view(job)}
        try:
            with self._client(False) as client:
                response = client.get("/queue")
                response.raise_for_status()
                queue = response.json()
                running = [item[1] for item in queue.get("queue_running", []) if len(item) > 1]
                pending = [item[1] for item in queue.get("queue_pending", []) if len(item) > 1]
                if job["prompt_id"] in running:
                    raise BridgeError(409, "already_running", "This job is already running. Stop it in ComfyUI if needed; AES Studio can safely remove only queued jobs.")
                if job["prompt_id"] in pending:
                    response = client.post("/queue", json={"delete": [job["prompt_id"]]})
                    response.raise_for_status()
                    response = client.get("/queue")
                    response.raise_for_status()
                    queue = response.json()
                    running = [item[1] for item in queue.get("queue_running", []) if len(item) > 1]
                    pending = [item[1] for item in queue.get("queue_pending", []) if len(item) > 1]
                    if job["prompt_id"] in running:
                        raise BridgeError(409, "already_running", "The job started before it could be removed. Stop it in ComfyUI if needed.")
                    if job["prompt_id"] not in pending:
                        self._finish(job_id, "cancelled", None)
        except (httpx.HTTPError, ValueError, AttributeError) as exc:
            raise BridgeError(503, "comfy_unavailable", "Could not check local ComfyUI to cancel the queued job.") from exc
        return {"schema_version": SCHEMA_VERSION, "job": self._view(job)}

    def _watch_guarded(self, job_id: str) -> None:
        try:
            self._watch(job_id)
            return
        except Exception:
            # Hold the GPU lease until queue inspection proves this job stopped.
            # Do not send a potentially global interrupt to an unknown version.
            pass
        while True:
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None or job["state"] in {"done", "failed", "cancelled"}:
                    return
                prompt_id = job["prompt_id"]
            try:
                with self._client(False) as client:
                    response = client.get("/queue")
                    response.raise_for_status()
                    queue = response.json()
                active = {item[1] for key in ("queue_running", "queue_pending")
                          for item in queue.get(key, []) if isinstance(item, (list, tuple)) and len(item) > 1}
                if prompt_id not in active:
                    self._finish(job_id, "failed", {"code": "watcher_failed", "message": "AES Studio lost job tracking after ComfyUI stopped reporting the job. Check ComfyUI's outputs."})
                    return
            except (httpx.HTTPError, ValueError, AttributeError):
                pass
            time.sleep(max(0.1, self._poll_seconds))

    def _complete_from_history(self, job_id: str, entry: dict[str, Any], client: httpx.Client) -> None:
        status = entry.get("status") if isinstance(entry.get("status"), dict) else {}
        state = str(status.get("status_str") or "").lower()
        if state in {"interrupted", "cancelled", "canceled"}:
            self._finish(job_id, "cancelled", None)
            return
        if state == "error":
            self._finish(job_id, "failed", {"code": "generation_failed", "message": _execution_error(status)})
            return
        with self._lock:
            job = self._jobs[job_id]
        outputs = entry.get("outputs") if isinstance(entry.get("outputs"), dict) else {}
        refs = (outputs.get(job["save_node"]) or {}).get("images") or []
        if not refs:
            self._finish(job_id, "failed", {"code": "no_output", "message": "ComfyUI finished without a saved result."})
            return
        try:
            files = [self._copy_output(job, ref, client) for ref in refs]
        except (httpx.HTTPError, OSError, ValueError, TypeError) as exc:
            self._finish(job_id, "failed", {"code": "save_failed", "message": f"ComfyUI generated a result, but AES Studio could not copy it: {type(exc).__name__}."})
            return
        with self._lock:
            job["files"] = files
        self._finish(job_id, "done", None)

    def _copy_output(self, job: dict[str, Any], ref: dict[str, Any], client: httpx.Client) -> dict[str, Any]:
        if not isinstance(ref, dict):
            raise ValueError("invalid output")
        filename, subfolder, kind = ref.get("filename"), ref.get("subfolder", ""), ref.get("type", "output")
        if not isinstance(filename, str) or not filename or '/' in filename or '\\' in filename:
            raise ValueError("invalid filename")
        if (not isinstance(subfolder, str) or subfolder.startswith(('/', '\\')) or ':' in subfolder
                or any(x in ('..', '.') for x in subfolder.replace('\\', '/').split('/')) or kind != 'output'):
            raise ValueError("invalid output location")
        suffix = Path(filename).suffix.lower()
        if suffix not in {'.png', '.jpg', '.jpeg', '.webp', '.mp4', '.webm', '.mov'}:
            raise ValueError("unsupported output type")
        root = Path(job['output_root']).resolve()
        folder = root / 'creative-models' / datetime.now().strftime('%Y-%m-%d')
        if root not in folder.resolve().parents:
            raise ValueError("invalid destination")
        folder.mkdir(parents=True, exist_ok=True)
        if root not in folder.resolve().parents:
            raise ValueError("invalid destination")
        target = folder / (job['preset'] + '-' + uuid.uuid4().hex[:12] + suffix)
        size = 0
        created = False
        try:
            with client.stream('GET', '/view', params={'filename': filename, 'subfolder': subfolder, 'type': 'output'}) as response:
                response.raise_for_status()
                with target.open('xb') as destination:
                    created = True
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if size > _MAX_OUTPUT_BYTES:
                            raise ValueError('output too large')
                        destination.write(chunk)
            if not size:
                raise ValueError('empty output')
        except Exception:
            if created:
                target.unlink(missing_ok=True)
            raise
        return {'relative_path': target.relative_to(root).as_posix(), 'size_bytes': size, 'kind': 'video' if suffix in {'.mp4','.webm','.mov'} else 'image'}

    def _view(self, job: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            return {'job_id': job['job_id'], 'preset': job['preset'], 'state': job['state'],
                    'queue_position': job['queue_position'], 'elapsed_seconds': round(time.monotonic()-job['created_monotonic'], 1),
                    'error': job['error'], 'cancel_requested': bool(job.get('cancel_requested')), 'files': list(job['files'])}
