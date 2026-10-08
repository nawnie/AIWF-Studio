"""Qwen Image 2.1 text-to-image jobs on the local ComfyUI engine (loopback only).

Image generation lives in the engine API so every surface behaves the same way:
the native Windows app, the React developer UI, the CLI and the MCP server all
call POST /api/pro/unified/images and then poll the job.

    prompt -> fill the bundled workflow (workflows/qwen_image_2_1_t2i.api.json)
           -> ComfyUI POST /prompt              queues the job on the GPU
           -> ComfyUI GET /queue, /history/ID   queued -> running -> done or failed
           -> ComfyUI GET /view                 PNG bytes
           -> saved under Studio's output folder in qwen-image/<date>/ with an
              A1111-style "parameters" text chunk, so the picture shows up in
              /outputs and can be cataloged into Dataset Studio like any Studio image.

What this module does not do: image editing, LoRAs, other model families, or
talking to a ComfyUI that is not on this machine (the bridge's client factory
refuses non-loopback addresses before any request is made).
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import random
import re
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import httpx

from aiwf.services.unified_bridge import SCHEMA_VERSION, BridgeError, _utc_now


# --- what a job can ask for ----------------------------------------------------------
# Short aspect names map to ComfyUI's ResolutionSelector options (checked against the
# live node catalog on 2026-10-07). Quality maps to the total megapixels generated.
ASPECT_RATIOS = {
    "1:1": "1:1 (Square)",
    "2:3": "2:3 (Portrait Photo)",
    "3:2": "3:2 (Photo)",
    "3:4": "3:4 (Portrait Standard)",
    "4:3": "4:3 (Standard)",
    "9:16": "9:16 (Portrait Widescreen)",
    "16:9": "16:9 (Widescreen)",
    "21:9": "21:9 (Ultrawide)",
}
QUALITY_MEGAPIXELS = {"draft": 0.6, "standard": 1.0}
MODEL_LABEL = "Qwen Image 2.1 (int8)"
MAX_PROMPT_CHARS = 2000
MAX_SEED = 2**53 - 1                  # largest seed that survives a JSON round trip in every client
MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_JOBS_KEPT = 50
logger = logging.getLogger(__name__)

# --- how the bundled workflow is filled ------------------------------------------------
WORKFLOW_PATH = Path(__file__).resolve().parent / "workflows" / "qwen_image_2_1_t2i.api.json"
SIZE_NODE, PROMPT_NODE, SAMPLER_NODE, SAVE_NODE = "13", "471", "476", "461"
NEGATIVE_PROMPT = "worst quality"
REQUIRED_FIXED_NODES = {
    "13": ("ResolutionSelector", {"aspect_ratio", "megapixels", "multiple"}),
    "461": ("SaveImageAdvanced", {"filename_prefix", "format", "images", "format.bit_depth", "format.input_color_space"}),
    "470": ("UNETLoader", {"unet_name", "weight_dtype"}),
    "471": ("TextEncodeQwenImage21", {"prompt", "negative_prompt", "resolution", "clip", "images"}),
    "472": ("CLIPLoader", {"clip_name", "type"}),
    "473": ("VAELoader", {"vae_name"}),
    "474": ("EmptyLatentImage", {"batch_size", "width", "height"}),
    "475": ("VAEDecode", {"samples", "vae"}),
    "476": ("KSampler", {"seed", "steps", "cfg", "sampler_name", "scheduler", "denoise", "model", "positive", "negative", "latent_image"}),
}
REQUIRED_WORKFLOW_LINKS = {
    ("461", "images"): ("475", 0),
    ("471", "clip"): ("472", 0),
    ("474", "width"): ("13", 0),
    ("474", "height"): ("13", 1),
    ("475", "samples"): ("476", 0),
    ("475", "vae"): ("473", 0),
    ("476", "model"): ("470", 0),
    ("476", "positive"): ("471", 0),
    ("476", "negative"): ("471", 1),
    ("476", "latent_image"): ("474", 0),
}

# --- how a job is watched ---------------------------------------------------------------
POLL_SECONDS = 1.0
JOB_TIMEOUT_SECONDS = 900.0           # a first run also loads ~20 GB of weights from disk
MISSING_POLLS_BEFORE_FAIL = 8         # ComfyUI forgot the job (restart or queue cleared)
FAILED_POLLS_BEFORE_FAIL = 15         # ComfyUI stopped answering
FINAL_STATES = {"done", "failed", "cancelled"}
_PROMPT_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_JOB_ID = re.compile(r"^img-[0-9a-f]{12}$")


class ImageJobs:
    """In-memory image job table plus one watcher thread per running job.

    client(slow) returns an httpx.Client already pointed at the loopback ComfyUI
    (the bridge builds it, so tests can inject a mock transport). on_done(job) is
    called once per finished job, which the bridge uses to write the project ledger.
    """

    def __init__(
        self,
        *,
        client: Callable[[bool], httpx.Client],
        comfy_url: str,
        on_done: Callable[[dict[str, Any]], None] | None = None,
        workflow_path: Path = WORKFLOW_PATH,
        poll_seconds: float = POLL_SECONDS,
        timeout_seconds: float = JOB_TIMEOUT_SECONDS,
    ) -> None:
        self._client = client
        self._comfy_url = comfy_url
        self._on_done = on_done
        self._workflow_path = Path(workflow_path)
        self._poll_seconds = poll_seconds
        self._timeout_seconds = timeout_seconds
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def readiness(self) -> dict[str, Any]:
        """Check the fixed workflow's node classes and model files without loading weights."""
        try:
            workflow = json.loads(self._workflow_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {"ready": False, "missingNodes": [], "missingModels": [], "reason": f"The bundled Qwen Image workflow could not be read: {exc}"}

        if not isinstance(workflow, dict) or not workflow:
            return {
                "ready": False,
                "missingNodes": [],
                "missingModels": [],
                "errorCode": "comfyui_workflow_invalid",
                "reason": "The bundled Qwen Image workflow is invalid: expected a non-empty node map.",
            }
        for node_id, node in workflow.items():
            if (
                not isinstance(node, dict)
                or not isinstance(node.get("class_type"), str)
                or not node["class_type"].strip()
                or not isinstance(node.get("inputs"), dict)
            ):
                return {
                    "ready": False,
                    "missingNodes": [],
                    "missingModels": [],
                    "errorCode": "comfyui_workflow_invalid",
                    "reason": f"The bundled Qwen Image workflow is invalid at node {node_id}.",
                }
        for node_id, (expected_type, expected_inputs) in REQUIRED_FIXED_NODES.items():
            node = workflow.get(node_id)
            if (
                not isinstance(node, dict)
                or node.get("class_type") != expected_type
                or not expected_inputs.issubset(node["inputs"])
            ):
                return {
                    "ready": False,
                    "missingNodes": [f"{node_id}:{expected_type}"],
                    "missingModels": [],
                    "errorCode": "comfyui_workflow_invalid",
                    "reason": f"The bundled Qwen Image workflow is invalid: node {node_id} must be {expected_type} with its required inputs.",
                }
        for (node_id, input_name), expected_ref in REQUIRED_WORKFLOW_LINKS.items():
            if workflow[node_id]["inputs"].get(input_name) != list(expected_ref):
                return {
                    "ready": False,
                    "missingNodes": [f"{node_id}.{input_name}"],
                    "missingModels": [],
                    "errorCode": "comfyui_workflow_invalid",
                    "reason": f"The bundled Qwen Image workflow has an invalid connection at {node_id}.{input_name}.",
                }
        node_ids = set(workflow)
        for node_id, node in workflow.items():
            for value in node["inputs"].values():
                if (
                    isinstance(value, list)
                    and len(value) == 2
                    and isinstance(value[0], str)
                    and isinstance(value[1], int)
                    and value[0] not in node_ids
                ):
                    return {
                        "ready": False,
                        "missingNodes": [f"{node_id}->{value[0]}"],
                        "missingModels": [],
                        "errorCode": "comfyui_workflow_invalid",
                        "reason": f"The bundled Qwen Image workflow references missing node {value[0]} from node {node_id}.",
                    }

        required_nodes = sorted({
            str(node.get("class_type") or "")
            for node in workflow.values()
            if isinstance(node, dict) and node.get("class_type")
        })
        model_inputs = {
            "UNETLoader": ("unet_name", "diffusion_models"),
            "CLIPLoader": ("clip_name", "text_encoders"),
            "VAELoader": ("vae_name", "vae"),
        }
        required_models: dict[str, set[str]] = {folder: set() for _key, folder in model_inputs.values()}
        for node in workflow.values():
            if not isinstance(node, dict):
                continue
            model_input = model_inputs.get(str(node.get("class_type") or ""))
            if model_input is None:
                continue
            input_name, folder = model_input
            filename = node["inputs"].get(input_name)
            if not isinstance(filename, str) or not filename.strip():
                return {
                    "ready": False,
                    "missingNodes": [],
                    "missingModels": [],
                    "errorCode": "comfyui_workflow_invalid",
                    "reason": f"The bundled Qwen Image workflow is invalid: {node['class_type']} has no {input_name}.",
                }
            required_models[folder].add(filename.replace("\\", "/").strip("/"))

        try:
            with self._client(False) as client:
                nodes_response = client.get("/object_info")
                if nodes_response.status_code != 200:
                    return {"ready": False, "missingNodes": required_nodes, "missingModels": [], "reason": f"ComfyUI node inventory returned HTTP {nodes_response.status_code}."}
                available_nodes = nodes_response.json()
                if not isinstance(available_nodes, dict):
                    return {"ready": False, "missingNodes": required_nodes, "missingModels": [], "reason": "ComfyUI returned an invalid node inventory."}
                missing_nodes = [name for name in required_nodes if name not in available_nodes]
                missing_models: list[str] = []
                for folder, expected in required_models.items():
                    if not expected:
                        continue
                    response = client.get(f"/models/{folder}")
                    if response.status_code != 200:
                        missing_models.extend(f"{folder}/{name}" for name in sorted(expected))
                        continue
                    available = response.json()
                    if not isinstance(available, list) or not all(isinstance(item, str) for item in available):
                        missing_models.extend(f"{folder}/{name}" for name in sorted(expected))
                        continue
                    normalized_available = {item.replace("\\", "/").strip("/").casefold() for item in available}
                    missing_models.extend(
                        f"{folder}/{name}"
                        for name in sorted(expected)
                        if name.casefold() not in normalized_available
                    )
        except (httpx.HTTPError, ValueError) as exc:
            return {
                "ready": False,
                "missingNodes": [],
                "missingModels": [],
                "errorCode": "comfyui_unreachable" if isinstance(exc, httpx.HTTPError) else "comfyui_invalid_response",
                "reason": f"ComfyUI setup could not be checked: {type(exc).__name__}.",
            }

        missing_nodes.sort()
        missing_models.sort()
        missing = []
        if missing_nodes:
            missing.append("workflow nodes: " + ", ".join(missing_nodes))
        if missing_models:
            missing.append("model files: " + ", ".join(missing_models))
        return {
            "ready": not missing,
            "missingNodes": missing_nodes,
            "missingModels": missing_models,
            "reason": (
                "Qwen Image 2.1 workflow nodes and model files are detected; loading and generation have not been verified."
                if not missing else "Missing " + "; ".join(missing) + "."
            ),
        }

    # ---------------------------------------------------------------------------------
    # start: validate, fill the workflow, queue it on ComfyUI, start watching
    # ---------------------------------------------------------------------------------
    def start(
        self,
        *,
        prompt: str,
        aspect_ratio: str = "1:1",
        quality: str = "draft",
        seed: int | None = None,
        project_id: str | None = None,
        output_root: Path | None,
        on_finished: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        text = (prompt or "").strip()
        if not text or len(text) > MAX_PROMPT_CHARS:
            raise BridgeError(422, "invalid_prompt", f"Describe the image in 1-{MAX_PROMPT_CHARS} characters.")
        if aspect_ratio not in ASPECT_RATIOS:
            raise BridgeError(422, "invalid_aspect_ratio", "Aspect ratio must be one of " + ", ".join(ASPECT_RATIOS) + ".")
        if quality not in QUALITY_MEGAPIXELS:
            raise BridgeError(422, "invalid_quality", "Quality must be draft or standard.")
        if seed is None:
            seed = random.randint(0, MAX_SEED)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SEED:
            raise BridgeError(422, "invalid_seed", f"Seed must be a whole number from 0 to {MAX_SEED}.")
        if output_root is None or not Path(output_root).is_dir():
            raise BridgeError(503, "no_output_root", "Studio's output folder is not available, so the image would have nowhere to go.")

        setup = self.readiness()
        if not setup["ready"]:
            code = str(setup.get("errorCode") or "comfyui_setup_incomplete")
            message = (
                f"The image engine (ComfyUI) is not running at {self._comfy_url}."
                if code == "comfyui_unreachable"
                else str(setup["reason"])
            )
            raise BridgeError(503, code, message)

        workflow = self._fill_workflow(text, ASPECT_RATIOS[aspect_ratio], QUALITY_MEGAPIXELS[quality], seed)
        try:
            with self._client(False) as client:
                response = client.post("/prompt", json={"prompt": workflow, "client_id": "aiwf-studio-" + uuid.uuid4().hex[:12]})
        except httpx.HTTPError as exc:
            raise BridgeError(503, "comfyui_unreachable", f"The image engine (ComfyUI) is not running at {self._comfy_url}.") from exc
        if response.status_code == 400:
            raise BridgeError(422, "workflow_rejected", _rejection_message(response))
        if response.status_code >= 400:
            raise BridgeError(502, "comfyui_error", f"ComfyUI answered HTTP {response.status_code} when queuing the image.")
        try:
            prompt_id = response.json().get("prompt_id")
        except (ValueError, AttributeError):
            prompt_id = None
        if not isinstance(prompt_id, str) or not _PROMPT_ID.fullmatch(prompt_id):
            raise BridgeError(502, "comfyui_bad_response", "ComfyUI queued the image but returned no job ID.")

        job = {
            "job_id": "img-" + uuid.uuid4().hex[:12],
            "prompt_id": prompt_id,
            "state": "queued",
            "prompt": text,
            "aspect_ratio": aspect_ratio,
            "quality": quality,
            "seed": seed,
            "project_id": project_id,
            "output_root": Path(output_root),
            "queue_position": None,
            "created_at": _utc_now(),
            "created_monotonic": time.monotonic(),
            "started_at": None,
            "finished_at": None,
            "error": None,
            "images": [],
            "on_finished": on_finished,
        }
        with self._lock:
            self._jobs[job["job_id"]] = job
            self._prune_locked()
        # the reply describes the job as queued; the watcher may move it on immediately afterwards
        view = self._view(job)
        watcher = threading.Thread(target=self._watch_guarded, args=(job["job_id"],), name=f"image-job-{job['job_id']}", daemon=True)
        try:
            watcher.start()
        except Exception:
            # Keep watching synchronously if the thread cannot be created;
            # returning and releasing the GPU lease would be unsafe.
            logger.exception("Could not create the watcher thread for %s; watching inline", job["job_id"])
            self._watch_guarded(job["job_id"])
        return {"schema_version": SCHEMA_VERSION, "job": view}

    # ---------------------------------------------------------------------------------
    # read side: status, saved file, cancel
    # ---------------------------------------------------------------------------------
    def status(self, job_id: str) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, "job": self._view(self._get(job_id))}

    def file(self, job_id: str, index: int) -> Path:
        job = self._get(job_id)
        images = job["images"]
        if not isinstance(index, int) or not 0 <= index < len(images):
            raise BridgeError(404, "image_not_found", "This job has no image with that number.")
        root = Path(job["output_root"]).resolve()
        path = (root / images[index]["relative_path"]).resolve()
        # the path came from this module, but it is checked again before any file is served
        if root not in path.parents or not path.is_file():
            raise BridgeError(404, "image_not_found", "The saved image is no longer in Studio's output folder.")
        return path

    def cancel(self, job_id: str) -> dict[str, Any]:
        job = self._get(job_id)
        if job["state"] in FINAL_STATES:
            return {"schema_version": SCHEMA_VERSION, "job": self._view(job)}
        try:
            with self._client(False) as client:
                queue_response = client.get("/queue")
                queue_response.raise_for_status()
                queue = queue_response.json()
                running = [item[1] for item in queue.get("queue_running", []) if len(item) > 1]
                pending = [item[1] for item in queue.get("queue_pending", []) if len(item) > 1]
                if job["prompt_id"] in pending:
                    deleted = client.post("/queue", json={"delete": [job["prompt_id"]]})
                    deleted.raise_for_status()
                    queue_response = client.get("/queue")
                    queue_response.raise_for_status()
                    queue = queue_response.json()
                    running = [item[1] for item in queue.get("queue_running", []) if len(item) > 1]
                    pending = [item[1] for item in queue.get("queue_pending", []) if len(item) > 1]
                    if job["prompt_id"] not in running and job["prompt_id"] not in pending:
                        self._finish(job["job_id"], "cancelled", None)
                        return {"schema_version": SCHEMA_VERSION, "job": self._view(job)}
                if job["prompt_id"] in running:
                    # interrupt only when this job is the one on the GPU, never someone else's work
                    interrupted = client.post("/interrupt", json={"prompt_id": job["prompt_id"]})
                    interrupted.raise_for_status()
        except (httpx.HTTPError, ValueError, AttributeError) as exc:
            raise BridgeError(503, "comfyui_unreachable", f"Could not reach ComfyUI at {self._comfy_url} to cancel the job.") from exc
        # Interrupt is asynchronous. Keep the GPU lease until the watcher sees
        # the job leave ComfyUI's queue or report a terminal history entry.
        with self._lock:
            if job["state"] not in FINAL_STATES:
                job["cancel_requested"] = True
        return {"schema_version": SCHEMA_VERSION, "job": self._view(job)}

    # ---------------------------------------------------------------------------------
    # the watcher: one thread per job, polling until a final state
    # ---------------------------------------------------------------------------------
    def _watch(self, job_id: str) -> None:
        missing = 0
        failures = 0
        last_timeout_stop_attempt = 0.0
        while True:
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None or job["state"] in FINAL_STATES:
                    return
                prompt_id = job["prompt_id"]
                age = time.monotonic() - job["created_monotonic"]
                timeout_requested = bool(job.get("timeout_requested"))
                if age > self._timeout_seconds and not timeout_requested:
                    job["timeout_requested"] = True
                    job["timeout_error"] = {
                        "code": "timeout",
                        "message": f"The image did not finish within {int(self._timeout_seconds)} seconds.",
                    }
                    timeout_requested = True
            if timeout_requested and time.monotonic() - last_timeout_stop_attempt >= max(2.0, self._poll_seconds):
                last_timeout_stop_attempt = time.monotonic()
                # A timeout does not prove ComfyUI stopped; request a stop and
                # retain the GPU lease until queue/history confirms termination.
                try:
                    with self._client(False) as client:
                        queue_response = client.get("/queue")
                        queue_response.raise_for_status()
                        queue = queue_response.json()
                        running = [item[1] for item in queue.get("queue_running", []) if len(item) > 1]
                        pending = [item[1] for item in queue.get("queue_pending", []) if len(item) > 1]
                        if prompt_id in pending:
                            response = client.post("/queue", json={"delete": [prompt_id]})
                            response.raise_for_status()
                        elif prompt_id in running:
                            response = client.post("/interrupt", json={"prompt_id": prompt_id})
                            response.raise_for_status()
                except Exception:
                    logger.debug("Could not stop timed-out ComfyUI prompt %s yet", prompt_id, exc_info=True)
            try:
                with self._client(True) as client:
                    entry = client.get(f"/history/{prompt_id}").json().get(prompt_id)
                    if isinstance(entry, dict):
                        with self._lock:
                            current = self._jobs.get(job_id)
                            timeout_error = current.get("timeout_error") if current else None
                        history_status = str((entry.get("status") or {}).get("status_str") or "").lower()
                        if timeout_error and history_status in {"interrupted", "cancelled", "canceled"}:
                            self._finish(job_id, "failed", timeout_error)
                            return
                        self._complete_from_history(job_id, entry, client)
                        return
                    queue = client.get("/queue").json()
                failures = 0
            except (httpx.HTTPError, ValueError, AttributeError):
                failures += 1
                if failures == FAILED_POLLS_BEFORE_FAIL:
                    # Connection loss leaves GPU activity unknown, so keep the
                    # lease and retry instead of marking the job terminal.
                    logger.warning("ComfyUI is unreachable while image job %s may still be active; retaining GPU lease", job_id)
                time.sleep(self._poll_seconds)
                continue

            # this block turns ComfyUI's queue into "running" or "queued, position N"
            running = [item[1] for item in queue.get("queue_running", []) if len(item) > 1]
            pending = sorted((item for item in queue.get("queue_pending", []) if len(item) > 1), key=lambda item: item[0])
            pending_ids = [item[1] for item in pending]
            with self._lock:
                if prompt_id in running:
                    missing = 0
                    if job["state"] == "queued":
                        job["state"], job["started_at"], job["queue_position"] = "running", _utc_now(), None
                elif prompt_id in pending_ids:
                    missing = 0
                    job["queue_position"] = pending_ids.index(prompt_id) + 1
                else:
                    missing += 1
            if missing >= MISSING_POLLS_BEFORE_FAIL:
                with self._lock:
                    current = self._jobs.get(job_id)
                    cancel_requested = bool(current and current.get("cancel_requested"))
                if cancel_requested:
                    self._finish(job_id, "cancelled", None)
                elif timeout_requested:
                    with self._lock:
                        current = self._jobs.get(job_id)
                        error = current.get("timeout_error") if current else None
                    self._finish(job_id, "failed", error or {"code": "timeout", "message": "The image timed out."})
                else:
                    self._finish(job_id, "failed", {"code": "job_lost", "message": "ComfyUI no longer has this job; it may have been restarted or its queue cleared."})
                return
            time.sleep(self._poll_seconds)

    def _watch_guarded(self, job_id: str) -> None:
        try:
            self._watch(job_id)
            return
        except Exception as exc:
            logger.exception("Unexpected watcher failure for %s; checking ComfyUI before releasing its GPU lease", job_id)

        # Do not release the shared GPU lock while the engine may still be
        # using the device. Retry queue inspection; once the job is absent,
        # record the watcher failure and let the normal completion callback
        # release the lease.
        while True:
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None or job["state"] in FINAL_STATES:
                    return
                prompt_id = job["prompt_id"]
            try:
                with self._client(False) as client:
                    response = client.get("/queue")
                    response.raise_for_status()
                    queue = response.json()
                active_ids = {
                    item[1]
                    for key in ("queue_running", "queue_pending")
                    for item in queue.get(key, [])
                    if isinstance(item, list) and len(item) > 1
                }
                if prompt_id not in active_ids:
                    self._finish(job_id, "failed", {"code": "watcher_failed", "message": "The image watcher failed after ComfyUI stopped reporting the job."})
                    return
                with self._client(False) as client:
                    if any(item[1] == prompt_id for item in queue.get("queue_pending", []) if isinstance(item, list) and len(item) > 1):
                        response = client.post("/queue", json={"delete": [prompt_id]})
                    else:
                        response = client.post("/interrupt", json={"prompt_id": prompt_id})
                    response.raise_for_status()
            except Exception:
                # Keep the lease while ComfyUI's activity cannot be confirmed.
                logger.debug("ComfyUI still cannot be checked for %s", job_id, exc_info=True)
            time.sleep(max(0.1, self._poll_seconds))

    def _complete_from_history(self, job_id: str, entry: dict[str, Any], client: httpx.Client) -> None:
        status = entry.get("status") if isinstance(entry.get("status"), dict) else {}
        if str(status.get("status_str") or "").lower() in {"interrupted", "cancelled", "canceled"}:
            self._finish(job_id, "cancelled", None)
            return
        if status.get("status_str") == "error":
            self._finish(job_id, "failed", {"code": "generation_failed", "message": _execution_error(status)})
            return
        outputs = entry.get("outputs") if isinstance(entry.get("outputs"), dict) else {}
        refs = (outputs.get(SAVE_NODE) or {}).get("images") or []
        if not refs:
            self._finish(job_id, "failed", {"code": "no_image", "message": "ComfyUI finished the job but saved no image."})
            return
        with self._lock:
            job = self._jobs[job_id]
        saved = []
        # this loop copies each finished image into Studio's output folder with its settings embedded
        for index, ref in enumerate(refs):
            try:
                saved.append(self._save_image(job, index, ref, client))
            except (httpx.HTTPError, OSError, ValueError) as exc:
                self._finish(job_id, "failed", {"code": "save_failed", "message": f"The image was generated but could not be saved: {type(exc).__name__}."})
                return
        with self._lock:
            job["images"] = saved
        self._finish(job_id, "done", None)

    def _save_image(self, job: dict[str, Any], index: int, ref: dict[str, Any], client: httpx.Client) -> dict[str, Any]:
        from PIL import Image
        from PIL.PngImagePlugin import PngInfo

        params = {"filename": str(ref.get("filename", "")), "subfolder": str(ref.get("subfolder", "")), "type": str(ref.get("type", "output"))}
        response = client.get("/view", params=params)
        if response.status_code != 200 or not response.headers.get("content-type", "").startswith("image/"):
            raise ValueError("ComfyUI did not return the image")
        payload = response.content
        if len(payload) > MAX_IMAGE_BYTES:
            raise ValueError("image too large")

        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            width, height = image.size
            info = PngInfo()
            info.add_text("parameters", _infotext(job, width, height))
            day = datetime.now().strftime("%Y-%m-%d")
            folder = Path(job["output_root"]) / "qwen-image" / day
            folder.mkdir(parents=True, exist_ok=True)
            stem = f"qwen-image-{datetime.now().strftime('%H%M%S')}-{job['seed']}-{index}"
            target = folder / f"{stem}.png"
            counter = 1
            while target.exists():
                target = folder / f"{stem}-{counter}.png"
                counter += 1
            image.save(target, format="PNG", pnginfo=info)
        return {
            "index": index,
            "relative_path": target.relative_to(Path(job["output_root"])).as_posix(),
            "width": width,
            "height": height,
            "size_bytes": target.stat().st_size,
        }

    # ---------------------------------------------------------------------------------
    # small helpers
    # ---------------------------------------------------------------------------------
    def _fill_workflow(self, prompt: str, aspect_label: str, megapixels: float, seed: int) -> dict[str, Any]:
        try:
            workflow = json.loads(self._workflow_path.read_text(encoding="utf-8"))
            workflow[SIZE_NODE]["inputs"]["aspect_ratio"] = aspect_label
            workflow[SIZE_NODE]["inputs"]["megapixels"] = megapixels
            workflow[PROMPT_NODE]["inputs"]["prompt"] = prompt
            workflow[PROMPT_NODE]["inputs"]["negative_prompt"] = NEGATIVE_PROMPT
            workflow[SAMPLER_NODE]["inputs"]["seed"] = seed
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise BridgeError(500, "workflow_missing", "AIWF Studio's Qwen Image workflow file is missing or damaged.") from exc
        # ComfyUI ignores _meta, but nothing beyond the graph itself needs to leave this process
        for node in workflow.values():
            node.pop("_meta", None)
        return workflow

    def _get(self, job_id: str) -> dict[str, Any]:
        if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id):
            raise BridgeError(422, "invalid_job_id", "Image job IDs look like img- followed by 12 hex characters.")
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise BridgeError(404, "job_not_found", "No image job with that ID (jobs are kept until the engine API restarts).")
        return job

    def _finish(self, job_id: str, state: str, error: dict[str, str] | None) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job["state"] in FINAL_STATES:
                return
            job["state"], job["error"], job["finished_at"], job["queue_position"] = state, error, _utc_now(), None
            snapshot = dict(job)
            on_finished = job.pop("on_finished", None)
        if self._on_done is not None:
            try:
                self._on_done(snapshot)
            except Exception:  # noqa: BLE001 - a ledger problem must not lose the finished image
                pass
        if callable(on_finished):
            try:
                on_finished(snapshot)
            except Exception:  # noqa: BLE001 - cleanup callbacks must not corrupt job state
                pass

    def _prune_locked(self) -> None:
        # keep the newest jobs; the oldest finished ones go first
        if len(self._jobs) <= MAX_JOBS_KEPT:
            return
        finished = [job_id for job_id, job in self._jobs.items() if job["state"] in FINAL_STATES]
        for job_id in finished[: len(self._jobs) - MAX_JOBS_KEPT]:
            del self._jobs[job_id]

    def _view(self, job: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            elapsed = time.monotonic() - job["created_monotonic"]
            return {
                "job_id": job["job_id"],
                "state": job["state"],
                "model": MODEL_LABEL,
                "prompt": job["prompt"][:300],
                "aspect_ratio": job["aspect_ratio"],
                "quality": job["quality"],
                "seed": job["seed"],
                "project_id": job["project_id"],
                "queue_position": job["queue_position"],
                "created_at": job["created_at"],
                "started_at": job["started_at"],
                "finished_at": job["finished_at"],
                "elapsed_seconds": round(elapsed, 1),
                "error": job["error"],
                "cancel_requested": bool(job.get("cancel_requested")),
                "images": [
                    {**image, "url": f"/api/pro/unified/images/{job['job_id']}/files/{image['index']}"}
                    for image in job["images"]
                ],
            }


def _rejection_message(response: httpx.Response) -> str:
    """Turn ComfyUI's 400 body into one sentence (usually a missing model file or node)."""
    try:
        body = response.json()
    except ValueError:
        return "ComfyUI refused the image workflow."
    error = body.get("error") if isinstance(body, dict) else None
    message = str(error.get("message", "")) if isinstance(error, dict) else ""
    detail = ""
    node_errors = body.get("node_errors") if isinstance(body, dict) else None
    if isinstance(node_errors, dict):
        # this loop picks the first concrete node error, e.g. a model file that is not installed
        for node in node_errors.values():
            errors = node.get("errors") if isinstance(node, dict) else None
            if errors and isinstance(errors[0], dict):
                detail = f"{node.get('class_type', 'node')}: {errors[0].get('details') or errors[0].get('message', '')}"
                break
    text = "ComfyUI refused the image workflow"
    if message:
        text += f": {message}"
    if detail:
        text += f" ({detail[:300]})"
    return text + "."


def _execution_error(status: dict[str, Any]) -> str:
    """Pull ComfyUI's execution_error text out of a failed history entry."""
    for message in status.get("messages") or []:
        if isinstance(message, list) and len(message) == 2 and message[0] == "execution_error" and isinstance(message[1], dict):
            node = message[1].get("node_type", "a node")
            text = str(message[1].get("exception_message", "")).strip().splitlines()
            return f"Generation failed in {node}: {text[0][:300] if text else 'no message'}."
    return "Generation failed in ComfyUI."


def _infotext(job: dict[str, Any], width: int, height: int) -> str:
    """A1111-style parameters text, the format Studio's /outputs listing already reads."""
    return (
        f"{job['prompt']}\n"
        f"Negative prompt: {NEGATIVE_PROMPT}\n"
        f"Steps: 30, Sampler: euler, Schedule type: Simple, CFG scale: 1, Seed: {job['seed']}, "
        f"Size: {width}x{height}, Model: {MODEL_LABEL}, Prompt hash: {hashlib.sha256(job['prompt'].encode('utf-8')).hexdigest()[:12]}"
    )
