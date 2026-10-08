"""Server-side bridge from AIWF Studio Pro to Dataset Studio, ReTrain and Qwen Chat.

The browser only ever talks to AIWF Pro (same origin). This module makes the
cross-app calls on the server, over loopback only, using each app's own
authenticated API:

    Studio outputs --catalog--> Dataset Studio  (POST /api/assets/import, labels, collection)
    Dataset Studio package --download--> verify revision --upload--> ReTrain
                                         (GET /api/retrain/exports/{name}/download,
                                          POST /api/retrain/datasets/import-package)
    ReTrain imported dataset --preflight (dry run only)--> plan receipt
    Project context card --explicit send--> Qwen Chat  (POST /v1/chat/completions)

Every action is recorded in a small JSON "project ledger" under
<data_dir>/_local/unified-projects/. That ledger's project ID is the shared
project identity: it is written into Dataset Studio as a tag on cataloged
assets and carried with every import, preflight and Qwen context card.

What this module deliberately does not do:
- start training (ReTrain's preflight route cannot, and no start route is called);
- send image outputs to ReTrain (Dataset Studio packages are text-only SFT data);
- send file contents, absolute paths or chat history to Qwen Chat;
- talk to any non-loopback address (configuration is rejected up front).
"""

from __future__ import annotations

import hashlib
import io
import ipaddress
import json
import os
import re
import tempfile
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import quote, urlparse

import httpx


# --- defaults for where each sibling app listens -----------------------------------
# These match each app's own launcher. Every one can be overridden by an
# environment variable so a test or an isolated profile can point elsewhere.
DEFAULT_DATASET_STUDIO_URL = "http://127.0.0.1:8796"          # Dataset Studio scripts/start-dataset-studio.ps1
DEFAULT_DATASET_STUDIO_STATE = Path(r"F:\Dataset Studio")     # its api-token.txt lives here
DEFAULT_RETRAIN_URL = "http://127.12.6.3:8787"               # ReTrain gui/api/app.py __main__ and Electron shell
DEFAULT_QWEN_CHAT_URL = "http://127.0.0.1:8080"               # F:\Ai_Models\llama.cpp\chat-start.cmd (active launcher)
DEFAULT_QWEN_KEY_FILE = Path.home() / ".llama-chat" / "api-key.txt"
DEFAULT_COMFYUI_URL = "http://127.0.0.1:8188"                  # F:\Ai_Models\llama.cpp\tools\start_comfyui_8188.cmd (image engine)

SCHEMA_VERSION = "1"
PACKAGE_SCHEMA = "retrain-gui-recipe-dataset-v1"
MAX_PACKAGE_BYTES = 512 * 1024 * 1024        # same cap as ReTrain's importer
MAX_CATALOG_OUTPUTS = 200
MAX_LEDGER_EVENTS = 500
MAX_QWEN_QUESTION_CHARS = 4000
MAX_QWEN_ANSWER_TOKENS = 768
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_PROJECT_ID = re.compile(r"^aiwfp-[0-9a-f]{16}$")
_MEDIA_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".avif", ".mp4", ".webm", ".mov", ".mkv"}

# preflight settings a caller may pass through to ReTrain; anything else is dropped
_PREFLIGHT_SETTING_KEYS = {
    "method", "tuneScope", "lastNLayers", "contextLength", "microBatch",
    "gradAccum", "loraRank", "precision", "optimizer", "scheduler",
}


class BridgeError(Exception):
    """A cross-app action failed in a way the UI should show to the user."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


# --- configuration ------------------------------------------------------------------
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _is_loopback_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    if parsed.hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        return False


def _first_key_line(path: Path) -> str:
    """Read a key file the way Qwen Chat's gateway does: first nonblank, noncomment line."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError:
        return ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    return ""


@dataclass(frozen=True)
class BridgeConfig:
    dataset_studio_url: str
    dataset_studio_token_file: Path
    retrain_url: str
    qwen_chat_url: str
    qwen_chat_key_file: Path
    projects_dir: Path
    timeout_seconds: float = 8.0
    # long-running calls (package transfer, preflight, chat completion)
    slow_timeout_seconds: float = 180.0
    # the image engine (Qwen Image 2.1 through ComfyUI); see aiwf/services/image_jobs.py
    comfyui_url: str = DEFAULT_COMFYUI_URL

    @classmethod
    def from_environment(cls, data_dir: Path) -> "BridgeConfig":
        state = Path(os.environ.get("DATASET_STUDIO_STATE", DEFAULT_DATASET_STUDIO_STATE))
        return cls(
            dataset_studio_url=os.environ.get("AIWF_DATASET_STUDIO_URL", DEFAULT_DATASET_STUDIO_URL).rstrip("/"),
            dataset_studio_token_file=Path(os.environ.get("AIWF_DATASET_STUDIO_TOKEN_FILE", state / "api-token.txt")),
            retrain_url=os.environ.get("AIWF_RETRAIN_API_URL", DEFAULT_RETRAIN_URL).rstrip("/"),
            qwen_chat_url=os.environ.get("AIWF_QWEN_CHAT_URL", DEFAULT_QWEN_CHAT_URL).rstrip("/"),
            qwen_chat_key_file=Path(os.environ.get("AIWF_QWEN_CHAT_KEY_FILE", DEFAULT_QWEN_KEY_FILE)),
            projects_dir=Path(data_dir) / "_local" / "unified-projects",
            comfyui_url=os.environ.get("AIWF_COMFYUI_URL", DEFAULT_COMFYUI_URL).rstrip("/"),
        )


# --- shared project ledger -------------------------------------------------------------
# One JSON file per project. Writes go to a temp file and are swapped in with
# os.replace, so a crash never leaves a half-written ledger behind.
#
# Windows refuses to open a file for a moment while os.replace swaps it, and refuses
# the swap while another handle has the file open; either side then sees
# PermissionError. Inside one process the ledger lock keeps reads and writes apart.
# Pro and the engine API are separate processes sharing this folder, so both sides
# also retry that transient error briefly instead of failing the request.
_SHARING_RETRIES = 12          # about half a second in total
_SHARING_RETRY_SECONDS = 0.04


def _read_json_retrying(path: Path) -> Any:
    for attempt in range(_SHARING_RETRIES):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except PermissionError:
            if attempt == _SHARING_RETRIES - 1:
                raise
            time.sleep(_SHARING_RETRY_SECONDS)
    return None   # not reached


def _replace_retrying(source: str, target: Path) -> None:
    for attempt in range(_SHARING_RETRIES):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == _SHARING_RETRIES - 1:
                raise
            time.sleep(_SHARING_RETRY_SECONDS)


class ProjectLedger:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        # reentrant: append() reads the record through get() while it holds the lock
        self._lock = threading.RLock()

    def _path(self, project_id: str) -> Path:
        if not isinstance(project_id, str) or not _PROJECT_ID.fullmatch(project_id):
            raise BridgeError(422, "invalid_project_id", "Project ID is not a valid unified project ID.")
        return self.root / f"{project_id}.json"

    def _write(self, record: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        target = self._path(record["project_id"])
        fd, temp_name = tempfile.mkstemp(prefix=".project-", suffix=".json", dir=self.root)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(record, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            _replace_retrying(temp_name, target)
        except BaseException:
            Path(temp_name).unlink(missing_ok=True)
            raise

    def create(self, name: str) -> dict[str, Any]:
        clean = (name or "").strip()
        if not clean or len(clean) > 120 or any(ord(ch) < 32 for ch in clean):
            raise BridgeError(422, "invalid_project_name", "Project name must be 1-120 printable characters.")
        record = {
            "schema_version": SCHEMA_VERSION,
            "project_id": f"aiwfp-{uuid.uuid4().hex[:16]}",
            "name": clean,
            "created_at": _utc_now(),
            "events": [],
        }
        with self._lock:
            self._write(record)
        return record

    def get(self, project_id: str) -> dict[str, Any]:
        path = self._path(project_id)
        try:
            with self._lock:
                record = _read_json_retrying(path)
        except FileNotFoundError as exc:
            raise BridgeError(404, "project_not_found", "That unified project does not exist.") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise BridgeError(500, "project_unreadable", "The project ledger could not be read.") from exc
        if record.get("project_id") != project_id:
            raise BridgeError(500, "project_unreadable", "The project ledger does not match its file name.")
        return record

    def list(self) -> list[dict[str, Any]]:
        if not self.root.is_dir():
            return []
        items = []
        # this loop summarizes every readable project; damaged files are skipped, not deleted
        for path in sorted(self.root.glob("aiwfp-*.json")):
            try:
                with self._lock:
                    record = _read_json_retrying(path)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(record, dict) or not _PROJECT_ID.fullmatch(str(record.get("project_id", ""))):
                continue
            items.append(_project_summary(record))
        items.sort(key=lambda item: item["created_at"], reverse=True)
        return items

    def append(self, project_id: str, kind: str, data: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            record = self.get(project_id)
            event = {"event_id": uuid.uuid4().hex[:12], "at": _utc_now(), "kind": kind, **data}
            events = list(record.get("events", []))
            events.append(event)
            record["events"] = events[-MAX_LEDGER_EVENTS:]
            self._write(record)
        return event


def _project_summary(record: dict[str, Any]) -> dict[str, Any]:
    events = record.get("events", [])
    counts: dict[str, int] = {}
    for event in events:
        counts[event.get("kind", "unknown")] = counts.get(event.get("kind", "unknown"), 0) + 1
    return {
        "project_id": record["project_id"],
        "name": record.get("name", ""),
        "created_at": record.get("created_at", ""),
        "event_counts": counts,
    }


# --- the bridge itself ---------------------------------------------------------------
@dataclass
class UnifiedBridge:
    config: BridgeConfig
    # tests inject an httpx.MockTransport here; production uses real loopback HTTP
    transport: httpx.BaseTransport | None = None
    ledger: ProjectLedger = field(init=False)
    # image job table, created on first use so startup never touches ComfyUI
    _images: Any = field(init=False, default=None)
    _images_lock: threading.Lock = field(init=False, default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.ledger = ProjectLedger(self.config.projects_dir)

    @property
    def images(self) -> Any:
        """The Qwen Image job runner (aiwf/services/image_jobs.py), sharing this bridge's loopback client."""
        with self._images_lock:
            if self._images is None:
                # imported here: image_jobs imports this module, so a top-level import would be circular
                from aiwf.services.image_jobs import ImageJobs

                self._images = ImageJobs(
                    client=lambda slow: self._client(self.config.comfyui_url, slow=slow),
                    comfy_url=self.config.comfyui_url,
                    on_done=self._record_image_job,
                )
            return self._images

    # this helper opens a client for one sibling app after re-checking it is loopback
    def _client(self, base_url: str, *, slow: bool = False, headers: dict[str, str] | None = None) -> httpx.Client:
        if not _is_loopback_url(base_url):
            raise BridgeError(503, "non_loopback_endpoint", f"Refusing to call a non-loopback endpoint: {base_url}")
        timeout = self.config.slow_timeout_seconds if slow else self.config.timeout_seconds
        return httpx.Client(base_url=base_url, timeout=timeout, transport=self.transport, trust_env=False, headers=headers or {})

    def _dataset_headers(self) -> dict[str, str]:
        token = _first_key_line(self.config.dataset_studio_token_file)
        if not token:
            raise BridgeError(503, "dataset_token_missing", "Dataset Studio's local API token file was not found or is empty.")
        return {"Authorization": f"Bearer {token}"}

    def _qwen_headers(self) -> dict[str, str]:
        key = _first_key_line(self.config.qwen_chat_key_file)
        if not key:
            raise BridgeError(503, "qwen_key_missing", "Qwen Chat's local API key file was not found or is empty.")
        return {"Authorization": f"Bearer {key}"}

    # ---------------------------------------------------------------------------------
    # capability status: one honest line per app, never a guess
    # ---------------------------------------------------------------------------------
    def status(self, *, studio_output_root: Path | None = None) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "checked_at": _utc_now(),
            "capabilities": {
                "dataset_studio": self._dataset_status(studio_output_root),
                "retrain": self._retrain_status(),
                "qwen_chat": self._qwen_status(),
                "comfyui": self._comfyui_status(),
            },
        }

    @staticmethod
    def _unreachable(name: str, url: str, exc: Exception) -> dict[str, Any]:
        reason = f"{name} is not running at {url}." if isinstance(exc, httpx.ConnectError) else f"{name} did not answer at {url}: {type(exc).__name__}."
        return {"available": False, "state": "not_running", "reason": reason}

    def _dataset_status(self, studio_output_root: Path | None) -> dict[str, Any]:
        url = self.config.dataset_studio_url
        try:
            headers = self._dataset_headers()
        except BridgeError as exc:
            return {"available": False, "state": "not_configured", "reason": exc.message}
        try:
            with self._client(url, headers=headers) as client:
                health = client.get("/api/health")
                if health.status_code == 401:
                    return {"available": False, "state": "auth_rejected", "reason": f"A protected service answered at {url} but rejected Dataset Studio's local token."}
                if health.status_code != 200:
                    return {"available": False, "state": "unexpected_response", "reason": f"{url}/api/health returned HTTP {health.status_code}; this is not a compatible Dataset Studio."}
                roots = [str(item) for item in health.json().get("allowed_roots", []) if isinstance(item, str)]
                listing = client.get("/api/retrain/exports")
        except BridgeError as exc:
            return {"available": False, "state": "not_configured", "reason": exc.message}
        except (httpx.HTTPError, ValueError) as exc:
            return self._unreachable("Dataset Studio", url, exc)
        package_listing = listing.status_code == 200
        # this block decides whether Studio's output folder is inside Dataset Studio's catalog roots
        catalog_outputs = False
        if studio_output_root is not None:
            resolved = Path(studio_output_root).resolve()
            catalog_outputs = any(_path_within(resolved, Path(root)) for root in roots)
        reasons = []
        if not package_listing:
            reasons.append("Dataset Studio is running but has no package list route; update it to list ReTrain packages.")
        if not catalog_outputs:
            reasons.append("Studio's output folder is not one of Dataset Studio's catalog roots, so outputs cannot be cataloged until Dataset Studio is started with that root.")
        return {
            "available": True,
            "state": "ready" if package_listing and catalog_outputs else "partial",
            "reason": " ".join(reasons) or "Dataset Studio is running; packages can be listed and Studio outputs can be cataloged.",
            "package_listing": package_listing,
            "catalog_studio_outputs": catalog_outputs,
            "endpoint": url,
        }

    def _retrain_status(self) -> dict[str, Any]:
        url = self.config.retrain_url
        try:
            with self._client(url) as client:
                response = client.get("/api/retrain/datasets/import-capability")
        except BridgeError as exc:
            return {"available": False, "state": "not_configured", "reason": exc.message}
        except httpx.HTTPError as exc:
            return self._unreachable("The ReTrain API", url, exc)
        if response.status_code == 401:
            return {"available": False, "state": "auth_rejected", "reason": f"A protected service answered at {url}; it is not the ReTrain API or needs credentials this bridge does not have."}
        if response.status_code == 404:
            return {"available": False, "state": "route_missing", "reason": f"A service at {url} answered, but it has no Dataset Studio package import route (not ReTrain, or an older ReTrain build)."}
        try:
            body = response.json()
        except ValueError:
            body = None
        if response.status_code != 200 or not isinstance(body, dict) or body.get("package_schema") != PACKAGE_SCHEMA:
            return {"available": False, "state": "unexpected_response", "reason": f"{url} returned an unrecognized import capability (HTTP {response.status_code})."}
        models = [
            {key: item.get(key) for key in ("model_id", "label", "family", "size_b", "text_capable", "local_weights_present")}
            for item in body.get("models", []) if isinstance(item, dict) and isinstance(item.get("model_id"), str)
        ]
        return {
            "available": True,
            "state": "ready",
            "reason": "ReTrain accepts Dataset Studio text packages and runs dry-run preflights. Training cannot be started from Studio.",
            "models": models,
            "training_start_available": False,
            "endpoint": url,
        }

    def _qwen_status(self) -> dict[str, Any]:
        url = self.config.qwen_chat_url
        try:
            headers = self._qwen_headers()
        except BridgeError as exc:
            return {"available": False, "state": "not_configured", "reason": exc.message}
        try:
            with self._client(url, headers=headers) as client:
                response = client.get("/v1/models")
        except BridgeError as exc:
            return {"available": False, "state": "not_configured", "reason": exc.message}
        except httpx.HTTPError as exc:
            return self._unreachable("Qwen Chat", url, exc)
        if response.status_code == 401:
            return {"available": False, "state": "auth_rejected", "reason": f"Qwen Chat answered at {url} but rejected the local API key."}
        try:
            data = response.json().get("data", []) if response.status_code == 200 else None
        except ValueError:
            data = None
        if not isinstance(data, list):
            return {"available": False, "state": "unexpected_response", "reason": f"{url}/v1/models returned HTTP {response.status_code}; not a compatible Qwen Chat endpoint."}
        models = []
        # this loop keeps only model IDs and whether llama.cpp reports them loaded
        for item in data:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                state = item.get("status", {}).get("value") if isinstance(item.get("status"), dict) else None
                models.append({"model_id": item["id"], "loaded": state == "loaded"})
        return {
            "available": True,
            "state": "ready",
            "reason": "Qwen Chat is running. Project context is sent only when you press Send.",
            "models": models,
            "endpoint": url,
        }

    def _comfyui_status(self) -> dict[str, Any]:
        """The image engine: verify ComfyUI and the fixed Qwen workflow prerequisites."""
        url = self.config.comfyui_url
        try:
            with self._client(url) as client:
                response = client.get("/system_stats")
        except BridgeError as exc:
            return {"available": False, "state": "not_configured", "reason": exc.message}
        except httpx.HTTPError as exc:
            return self._unreachable("ComfyUI", url, exc)
        try:
            body = response.json() if response.status_code == 200 else None
        except ValueError:
            body = None
        system = body.get("system") if isinstance(body, dict) else None
        if not isinstance(system, dict):
            return {"available": False, "state": "unexpected_response", "reason": f"{url}/system_stats returned HTTP {response.status_code}; not a compatible ComfyUI."}
        setup = self.images.readiness()
        if setup.get("errorCode") == "comfyui_unreachable":
            return {
                "available": False,
                "state": "not_running",
                "reason": f"ComfyUI stopped responding while its image workflow requirements were being checked at {url}.",
                "workflow_prerequisites_ready": False,
                "missing_nodes": list(setup["missingNodes"]),
                "missing_models": list(setup["missingModels"]),
                "version": system.get("comfyui_version"),
                "endpoint": url,
            }
        return {
            "available": True,
            "state": "ready" if setup["ready"] else "partial",
            "reason": "ComfyUI is running; " + str(setup["reason"]),
            "workflow_prerequisites_ready": bool(setup["ready"]),
            "missing_nodes": list(setup["missingNodes"]),
            "missing_models": list(setup["missingModels"]),
            "version": system.get("comfyui_version"),
            "endpoint": url,
        }

    # ---------------------------------------------------------------------------------
    # images: Qwen Image 2.1 through ComfyUI (job logic lives in image_jobs.py)
    # ---------------------------------------------------------------------------------
    def start_image_job(
        self,
        *,
        prompt: str,
        aspect_ratio: str,
        quality: str,
        seed: int | None,
        project_id: str | None,
        output_root: Path | None,
        on_finished: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        if project_id is not None:
            self.ledger.get(project_id)  # unknown or malformed project IDs fail before the GPU is used
        return self.images.start(
            prompt=prompt, aspect_ratio=aspect_ratio, quality=quality, seed=seed,
            project_id=project_id, output_root=output_root, on_finished=on_finished,
        )

    def image_status(self, job_id: str) -> dict[str, Any]:
        return self.images.status(job_id)

    def image_file(self, job_id: str, index: int) -> Path:
        return self.images.file(job_id, index)

    def cancel_image(self, job_id: str) -> dict[str, Any]:
        return self.images.cancel(job_id)

    def _record_image_job(self, job: dict[str, Any]) -> None:
        # the ledger keeps identifiers and counts only: the prompt is stored as a hash, never as text
        if not job.get("project_id") or job.get("state") != "done":
            return
        self.ledger.append(job["project_id"], "image_generated", {
            "job_id": job["job_id"],
            "model": "qwen-image-2.1",
            "images": [image["relative_path"] for image in job.get("images", [])],
            "seed": job.get("seed"),
            "aspect_ratio": job.get("aspect_ratio"),
            "quality": job.get("quality"),
            "prompt_sha256": hashlib.sha256(str(job.get("prompt", "")).encode("utf-8")).hexdigest(),
        })

    # ---------------------------------------------------------------------------------
    # projects (shared identity)
    # ---------------------------------------------------------------------------------
    def list_projects(self) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, "projects": self.ledger.list()}

    def create_project(self, name: str) -> dict[str, Any]:
        record = self.ledger.create(name)
        return {"schema_version": SCHEMA_VERSION, "project": _project_detail(record)}

    def get_project(self, project_id: str) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, "project": _project_detail(self.ledger.get(project_id))}

    # ---------------------------------------------------------------------------------
    # flow 1: Studio outputs -> Dataset Studio catalog, with project tag and provenance
    # ---------------------------------------------------------------------------------
    def catalog_outputs(
        self,
        project_id: str,
        relative_paths: Iterable[str],
        *,
        output_root: Path,
        read_provenance: Callable[[Path], dict[str, Any]],
    ) -> dict[str, Any]:
        project = self.ledger.get(project_id)
        root = Path(output_root).resolve()
        requested = list(dict.fromkeys(str(item) for item in relative_paths))
        if not requested:
            raise BridgeError(422, "no_outputs", "Select at least one Studio output.")
        if len(requested) > MAX_CATALOG_OUTPUTS:
            raise BridgeError(422, "too_many_outputs", f"Catalog at most {MAX_CATALOG_OUTPUTS} outputs at a time.")

        # this loop resolves each output inside Studio's output folder and records its provenance
        outputs: list[dict[str, Any]] = []
        for relative in requested:
            target = (root / relative).resolve()
            if not _path_within(target, root) or not target.is_file():
                raise BridgeError(404, "output_not_found", f"Studio output not found: {relative}")
            if target.suffix.lower() not in _MEDIA_EXTENSIONS:
                raise BridgeError(422, "unsupported_output", f"Only image or video outputs can be cataloged: {relative}")
            settings = read_provenance(target) or {}
            outputs.append({
                "relative_path": target.relative_to(root).as_posix(),
                "absolute_path": str(target),
                "sha256": _file_sha256(target),
                "size_bytes": target.stat().st_size,
                "prompt": str(settings.get("prompt", ""))[:2000],
                "seed": settings.get("seed"),
                "model": str(settings.get("modelName", ""))[:120],
            })

        project_tag = f"aiwf-project:{project_id}"
        collection_name = f"AIWF project: {project['name']} ({project_id})"[:160]
        try:
            with self._client(self.config.dataset_studio_url, slow=True, headers=self._dataset_headers()) as client:
                imported = _expect_json(client.post("/api/assets/import", json={"paths": [item["absolute_path"] for item in outputs]}), "Dataset Studio import")
                asset_ids = imported.get("asset_ids")
                if not isinstance(asset_ids, list) or len(asset_ids) != len(outputs) or not all(isinstance(item, int) for item in asset_ids):
                    raise BridgeError(502, "dataset_bad_response", "Dataset Studio returned an unexpected import result.")
                # tags carry the shared project ID and the source app into the catalog
                for value in (project_tag, "aiwf-source:studio"):
                    _expect_json(client.post("/api/assets/labels", json={"asset_ids": asset_ids, "kind": "tag", "value": value}), "Dataset Studio tagging")
                for asset_id, item in zip(asset_ids, outputs):
                    item["asset_id"] = asset_id
                    if item["model"]:
                        _expect_json(client.post(f"/api/assets/{asset_id}/labels", json={"kind": "tag", "value": f"aiwf-model:{item['model']}"[:160]}), "Dataset Studio tagging")
                    item["caption_written"] = self._import_caption(client, asset_id, item["prompt"])
                collection = _expect_json(client.post("/api/collections", json={"name": collection_name}), "Dataset Studio collection")
                _expect_json(client.post(f"/api/collections/{collection['id']}/assets", json={"asset_ids": asset_ids}), "Dataset Studio collection")
        except httpx.HTTPError as exc:
            raise BridgeError(503, "dataset_unreachable", f"Dataset Studio is not reachable at {self.config.dataset_studio_url}.") from exc

        # the ledger keeps hashes and generation settings; absolute paths stay out of it
        receipts = [{key: value for key, value in item.items() if key != "absolute_path"} for item in outputs]
        event = self.ledger.append(project_id, "dataset_catalog", {
            "collection": {"id": collection.get("id"), "name": collection.get("name")},
            "project_tag": project_tag,
            "outputs": receipts,
        })
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "cataloged",
            "project_id": project_id,
            "event_id": event["event_id"],
            "collection": {"id": collection.get("id"), "name": collection.get("name")},
            "project_tag": project_tag,
            "assets": [{"asset_id": item["asset_id"], "relative_path": item["relative_path"], "sha256": item["sha256"], "caption_written": item["caption_written"]} for item in receipts],
            "note": "Image and video assets are cataloged only. Dataset Studio's ReTrain packages are text-only, so these outputs cannot become ReTrain training rows.",
        }

    @staticmethod
    def _import_caption(client: httpx.Client, asset_id: int, prompt: str) -> bool:
        """Use the generation prompt as an 'import' caption only when the asset has no caption yet."""
        if not prompt:
            return False
        detail = client.get(f"/api/assets/{asset_id}")
        if detail.status_code != 200:
            return False
        body = detail.json()
        if body.get("caption"):
            return False
        response = client.put(f"/api/assets/{asset_id}/caption", json={"text": prompt[:8000], "source": "import", "expected_revision": body.get("revision", 0)})
        return response.status_code == 200

    # ---------------------------------------------------------------------------------
    # flow 2: explicit Dataset Studio package -> verified ReTrain import
    # ---------------------------------------------------------------------------------
    def list_packages(self) -> dict[str, Any]:
        try:
            with self._client(self.config.dataset_studio_url, headers=self._dataset_headers()) as client:
                body = _expect_json(client.get("/api/retrain/exports"), "Dataset Studio package list")
        except httpx.HTTPError as exc:
            raise BridgeError(503, "dataset_unreachable", f"Dataset Studio is not reachable at {self.config.dataset_studio_url}.") from exc
        packages = []
        # this loop forwards only display fields; Dataset Studio download URLs stay server-side
        for item in body.get("packages", []):
            if not isinstance(item, dict) or not isinstance(item.get("package_name"), str):
                continue
            entry = {"package_name": item["package_name"], "status": item.get("status", "invalid")}
            if entry["status"] == "ready" and _DIGEST.fullmatch(str(item.get("manifest_sha256", ""))):
                entry.update({
                    "manifest_sha256": item["manifest_sha256"],
                    "revision_id": item.get("revision_id"),
                    "created_at": item.get("created_at"),
                    "recipe_id": item.get("recipe_id"),
                    "modality": item.get("modality"),
                    "counts": item.get("counts", {}),
                })
            else:
                entry["status"] = "invalid"
                entry["reason"] = str(item.get("reason") or "Package manifest could not be verified.")
            packages.append(entry)
        return {"schema_version": SCHEMA_VERSION, "packages": packages}

    def import_package(self, project_id: str, package_name: str, manifest_sha256: str) -> dict[str, Any]:
        self.ledger.get(project_id)
        if not _DIGEST.fullmatch(manifest_sha256 or ""):
            raise BridgeError(422, "invalid_revision", "A 64-character package revision hash is required.")
        if not package_name or len(package_name) > 180:
            raise BridgeError(422, "invalid_package", "A package name is required.")

        with tempfile.SpooledTemporaryFile(max_size=16 * 1024 * 1024, mode="w+b") as archive:
            # this block downloads Dataset Studio's own hash-verified ZIP snapshot (bounded)
            try:
                with self._client(self.config.dataset_studio_url, slow=True, headers=self._dataset_headers()) as client:
                    with client.stream("GET", f"/api/retrain/exports/{quote(package_name, safe='')}/download") as response:
                        if response.status_code != 200:
                            response.read()
                            _raise_upstream(response, "Dataset Studio package download")
                        size = 0
                        for chunk in response.iter_bytes():
                            size += len(chunk)
                            if size > MAX_PACKAGE_BYTES:
                                raise BridgeError(413, "package_too_large", "The package exceeds the 512 MB transfer limit.")
                            archive.write(chunk)
            except httpx.HTTPError as exc:
                raise BridgeError(503, "dataset_unreachable", f"Dataset Studio is not reachable at {self.config.dataset_studio_url}.") from exc

            # this block confirms the bytes are the exact revision the user selected
            archive.seek(0)
            actual = _archive_revision(archive)
            if actual != manifest_sha256:
                raise BridgeError(409, "stale_revision", "The package on disk is not the revision you selected. Refresh the package list and select it again.")
            archive.seek(0, io.SEEK_END)
            length = archive.tell()
            archive.seek(0)

            # this block uploads the same bytes to ReTrain's strict importer
            try:
                with self._client(self.config.retrain_url, slow=True) as client:
                    response = client.post(
                        "/api/retrain/datasets/import-package",
                        content=_iter_file(archive),
                        headers={"Content-Type": "application/zip", "Content-Length": str(length)},
                    )
                    imported = _expect_json(response, "ReTrain import")
            except httpx.HTTPError as exc:
                raise BridgeError(503, "retrain_unreachable", f"The ReTrain API is not reachable at {self.config.retrain_url}.") from exc

        if imported.get("manifest_sha256") != manifest_sha256 or imported.get("dataset_id") != f"sha256-{manifest_sha256}":
            raise BridgeError(502, "revision_mismatch", "ReTrain stored a different revision than the one selected; nothing will be planned against it.")
        dataset = {
            "dataset_id": imported["dataset_id"],
            "package_name": imported.get("package_name"),
            "manifest_sha256": imported["manifest_sha256"],
            "revision_id": imported.get("revision_id"),
            "counts": imported.get("counts", {}),
            "modality": imported.get("modality"),
            "reused": bool(imported.get("reused")),
        }
        event = self.ledger.append(project_id, "retrain_import", {"dataset": dataset})
        return {"schema_version": SCHEMA_VERSION, "status": "imported", "project_id": project_id, "event_id": event["event_id"], "dataset": dataset}

    # ---------------------------------------------------------------------------------
    # flow 3: ReTrain preflight (dry run) against exactly that imported revision
    # ---------------------------------------------------------------------------------
    def preflight(self, project_id: str, dataset_id: str, manifest_sha256: str, model_id: str, settings: dict[str, Any] | None) -> dict[str, Any]:
        self.ledger.get(project_id)
        if not _DIGEST.fullmatch(manifest_sha256 or "") or dataset_id != f"sha256-{manifest_sha256}":
            raise BridgeError(422, "revision_binding", "The dataset ID does not match the selected package revision.")
        if not isinstance(model_id, str) or not model_id.strip():
            raise BridgeError(422, "model_required", "Choose a ReTrain model before running the preflight.")
        safe_settings = {key: value for key, value in (settings or {}).items() if key in _PREFLIGHT_SETTING_KEYS}
        try:
            with self._client(self.config.retrain_url, slow=True) as client:
                body = _expect_json(
                    client.post(f"/api/retrain/datasets/imported/{dataset_id}/preflight", json={"modelId": model_id, "settings": safe_settings}),
                    "ReTrain preflight",
                )
        except httpx.HTTPError as exc:
            raise BridgeError(503, "retrain_unreachable", f"The ReTrain API is not reachable at {self.config.retrain_url}.") from exc
        dataset = body.get("dataset") if isinstance(body.get("dataset"), dict) else {}
        if dataset.get("manifest_sha256") != manifest_sha256:
            raise BridgeError(502, "revision_mismatch", "ReTrain planned against a different revision; the result was discarded.")
        if body.get("start_enabled") is not False or body.get("execution_requested") is not False:
            raise BridgeError(502, "unsafe_preflight", "ReTrain's preflight did not confirm it is a non-executing dry run; the result was discarded.")
        plan = body.get("plan") if isinstance(body.get("plan"), dict) else {}
        result = {
            "dataset_id": dataset_id,
            "manifest_sha256": manifest_sha256,
            "model_id": model_id,
            "settings": safe_settings,
            "plan_status": plan.get("status"),
            **_browser_safe_plan(plan),
        }
        event = self.ledger.append(project_id, "retrain_preflight", {
            "dataset_id": dataset_id, "manifest_sha256": manifest_sha256, "model_id": model_id, "plan_status": plan.get("status"),
        })
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "preflight",
            "project_id": project_id,
            "event_id": event["event_id"],
            "start_enabled": False,
            "execution_requested": False,
            "message": str(body.get("message") or "Preflight only. Training was not started."),
            "result": result,
        }

    # ---------------------------------------------------------------------------------
    # model weights: what is on this PC, open vs gated on Hugging Face, key, downloads
    # (ReTrain owns the model folders and the downloader; Studio proxies it same-origin)
    # ---------------------------------------------------------------------------------
    def _retrain_json(self, method: str, path: str, action: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            with self._client(self.config.retrain_url, slow=method == "POST") as client:
                response = client.request(method, "/api/retrain/models/weights" + path, json=payload)
                return _expect_json(response, action)
        except httpx.HTTPError as exc:
            raise BridgeError(503, "retrain_unreachable", f"The ReTrain API is not reachable at {self.config.retrain_url}.") from exc

    def list_models(self, *, refresh: bool = False) -> dict[str, Any]:
        body = self._retrain_json("GET", "?refresh=true" if refresh else "", "ReTrain model catalog")
        models = []
        # this loop forwards display fields only; no folder paths leave ReTrain
        for item in body.get("models", []):
            if not isinstance(item, dict) or not isinstance(item.get("model_id"), str):
                continue
            entry = {key: item.get(key) for key in ("model_id", "label", "family", "size_b", "hf_repo", "present", "status", "text_capable", "download")}
            if isinstance(item.get("access"), dict):
                entry["access"] = {key: item["access"].get(key) for key in ("state", "gated", "has_key", "download_bytes", "file_count", "message")}
            models.append(entry)
        return {"schema_version": SCHEMA_VERSION, "has_key": bool(body.get("has_key")), "models": models}

    def save_hf_token(self, token: str) -> dict[str, Any]:
        # the token goes straight to ReTrain, which validates it with Hugging Face and stores it in the
        # standard Hugging Face token file; it is never logged, echoed or written to the project ledger
        body = self._retrain_json("POST", "/hf-token", "Hugging Face key", {"token": token})
        return {"schema_version": SCHEMA_VERSION, "ok": bool(body.get("ok")), "user": body.get("user", "")}

    def clear_hf_token(self) -> dict[str, Any]:
        self._retrain_json("POST", "/hf-token/clear", "Hugging Face key")
        return {"schema_version": SCHEMA_VERSION, "ok": True}

    def download_model(self, model_id: str) -> dict[str, Any]:
        job = self._retrain_json("POST", "/download", "Model download", {"modelId": model_id})
        return {"schema_version": SCHEMA_VERSION, "download": _download_view(job)}

    def download_status(self, job_id: str) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, "download": _download_view(self._download_job("GET", job_id, "", "Model download status"))}

    def cancel_download(self, job_id: str) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, "download": _download_view(self._download_job("POST", job_id, "/cancel", "Model download cancel"))}

    # an unknown download ID is the caller's mistake, not a ReTrain failure: say so with its own code,
    # the way unknown image jobs answer job_not_found
    def _download_job(self, method: str, job_id: str, suffix: str, action: str) -> dict[str, Any]:
        try:
            return self._retrain_json(method, f"/download/{quote(job_id, safe='')}{suffix}", action)
        except BridgeError as exc:
            if exc.status_code == 404:
                raise BridgeError(404, "download_not_found", "No model download with that ID (downloads are kept until ReTrain restarts).") from exc
            raise

    # ---------------------------------------------------------------------------------
    # flow 4: explicit project context -> Qwen Chat
    # ---------------------------------------------------------------------------------
    def qwen_context(self, project_id: str) -> dict[str, Any]:
        record = self.ledger.get(project_id)
        card = _context_card(record)
        return {"schema_version": SCHEMA_VERSION, "project_id": project_id, "context": card,
                "context_sha256": hashlib.sha256(card.encode("utf-8")).hexdigest()}

    def qwen_ask(self, project_id: str, model_id: str, question: str) -> dict[str, Any]:
        record = self.ledger.get(project_id)
        question = (question or "").strip()
        if not question or len(question) > MAX_QWEN_QUESTION_CHARS:
            raise BridgeError(422, "invalid_question", f"Ask a question of 1-{MAX_QWEN_QUESTION_CHARS} characters.")
        if not isinstance(model_id, str) or not model_id.strip():
            raise BridgeError(422, "model_required", "Choose a Qwen Chat model.")
        card = _context_card(record)
        payload = {
            "model": model_id,
            "stream": False,
            "max_tokens": MAX_QWEN_ANSWER_TOKENS,
            "messages": [
                {"role": "system", "content": "You are helping with an AIWF Studio project. The project context below was sent explicitly by the user. It contains identifiers and counts only, not file contents.\n\n" + card},
                {"role": "user", "content": question},
            ],
        }
        try:
            with self._client(self.config.qwen_chat_url, slow=True, headers=self._qwen_headers()) as client:
                body = _expect_json(client.post("/v1/chat/completions", json=payload), "Qwen Chat")
        except httpx.HTTPError as exc:
            raise BridgeError(503, "qwen_unreachable", f"Qwen Chat is not reachable at {self.config.qwen_chat_url}.") from exc
        try:
            answer = body["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise BridgeError(502, "qwen_bad_response", "Qwen Chat returned no answer.") from exc
        event = self.ledger.append(project_id, "qwen_context_sent", {
            "model_id": model_id,
            "context_sha256": hashlib.sha256(card.encode("utf-8")).hexdigest(),
            "question": question,
            "question_chars": len(question),
            "answer_chars": len(answer),
        })
        return {"schema_version": SCHEMA_VERSION, "project_id": project_id, "event_id": event["event_id"], "model_id": body.get("model", model_id), "answer": answer, "context_sent": card}

    def record_context_question(self, project_id: str, model_id: str, question: str, context_sha256: str) -> dict[str, Any]:
        """Record a submitted streaming question with the exact project context sent."""
        self.ledger.get(project_id)
        question = (question or "").strip()
        if not question or len(question) > MAX_QWEN_QUESTION_CHARS:
            raise BridgeError(422, "invalid_question", f"Ask a question of 1-{MAX_QWEN_QUESTION_CHARS} characters.")
        if not isinstance(model_id, str) or not model_id.strip():
            raise BridgeError(422, "model_required", "Choose a Qwen Chat model.")
        if not isinstance(context_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", context_sha256):
            raise BridgeError(422, "invalid_context_hash", "Include the SHA-256 hash of the project context sent.")
        event = self.ledger.append(project_id, "qwen_context_sent", {
            "model_id": model_id,
            "context_sha256": context_sha256,
            "question": question,
            "question_chars": len(question),
            "answer_chars": None,
            "stream_returned": True,
        })
        return {"schema_version": SCHEMA_VERSION, "project_id": project_id, "event_id": event["event_id"]}


# --- model family compatibility (static truth table) --------------------------------------
# Trained artifacts from these apps are not interchangeable. This table is what
# the UI shows so nobody tries to load an LLM adapter into an image pipeline.
MODEL_FAMILIES = [
    {
        "artifact": "ReTrain LoRA/QLoRA adapter",
        "family": "text LLM (Qwen2.5, LFM2.5, SmolLM3)",
        "produced_by": "ReTrain",
        "usable_in": ["ReTrain evaluation", "Hugging Face Transformers/PEFT with the same base model"],
        "not_usable_in": ["AIWF Studio image or video pipelines", "Qwen Chat (llama.cpp) until converted to GGUF and matched to the same base"],
    },
    {
        "artifact": "Image LoRA (SD/SDXL/Flux/Qwen-Image)",
        "family": "diffusion image model",
        "produced_by": "AIWF Studio engines (kohya) or external trainers",
        "usable_in": ["AIWF Studio pipelines of the same base family"],
        "not_usable_in": ["ReTrain", "Qwen Chat text models"],
    },
    {
        "artifact": "GGUF model / GGUF LoRA",
        "family": "llama.cpp runtime",
        "produced_by": "conversion from a Transformers checkpoint",
        "usable_in": ["Qwen Chat (llama.cpp) with a matching base"],
        "not_usable_in": ["ReTrain training", "AIWF Studio diffusion pipelines"],
    },
]


# --- browser-safe preflight plan ------------------------------------------------------------
# ReTrain's plan includes its full run config and gate details with absolute
# server paths (dataset folder, output root, TensorBoard folder). The browser
# gets only gate names/states, the VRAM estimate, dependency availability and
# notes, with any absolute path replaced by "<server path>".
_BACKSLASH = re.escape(chr(92))
_SERVER_PATH = re.compile("(?:[A-Za-z]:[" + _BACKSLASH + "/]|" + _BACKSLASH + _BACKSLASH + ").*")


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        return _SERVER_PATH.sub("<server path>", value)
    if isinstance(value, dict):
        return {key: _redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _browser_safe_plan(plan: dict[str, Any]) -> dict[str, Any]:
    validation = plan.get("validation") if isinstance(plan.get("validation"), dict) else {}
    estimate = validation.get("estimate") if isinstance(validation.get("estimate"), dict) else {}
    training_args = plan.get("training_args") if isinstance(plan.get("training_args"), dict) else {}
    return {
        "gates": [
            {"gate": str(item.get("gate", "")), "state": str(item.get("state", "")), "detail": _redact(str(item.get("detail", "")))}
            for item in validation.get("gates", []) if isinstance(item, dict)
        ],
        "estimate": _redact({key: estimate.get(key) for key in ("fit_state", "estimated_gb", "limit_gb", "headroom_gb", "percent", "warnings")}),
        "dependencies": [
            {"label": str(item.get("label") or item.get("package", "")), "available": bool(item.get("available"))}
            for item in validation.get("dependencies", []) if isinstance(item, dict)
        ],
        "notes": [_redact(str(item)) for item in validation.get("notes", [])],
        "summary": _redact(plan.get("summary", {})),
        "training_args": _redact({key: value for key, value in training_args.items() if key != "tensorboard_logdir"}),
    }


# --- small helpers -----------------------------------------------------------------------
def _path_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_file(stream: Any, chunk_size: int = 1024 * 1024):
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            return
        yield chunk


def _archive_revision(archive: Any) -> str:
    """Return the manifest revision hash inside a package ZIP after recomputing it."""
    try:
        with zipfile.ZipFile(archive) as bundle:
            info = bundle.getinfo("manifest.json")
            if info.file_size > 64 * 1024 * 1024:
                raise BridgeError(422, "invalid_package", "Package manifest is too large.")
            manifest = json.loads(bundle.read(info).decode("utf-8"))
    except (KeyError, zipfile.BadZipFile, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BridgeError(502, "invalid_package", "Dataset Studio returned a package without a readable manifest.") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != PACKAGE_SCHEMA:
        raise BridgeError(422, "invalid_package", "The selected package does not use the ReTrain text package schema.")
    unsigned = {key: value for key, value in manifest.items() if key != "revision"}
    digest = hashlib.sha256(json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    revision = manifest.get("revision") if isinstance(manifest.get("revision"), dict) else {}
    if revision.get("manifest_sha256") != digest:
        raise BridgeError(409, "stale_revision", "The package manifest no longer matches its revision hash.")
    return digest


def _download_view(job: dict[str, Any]) -> dict[str, Any]:
    keys = ("job_id", "model_id", "repo", "status", "total_bytes", "done_bytes", "files_total", "files_done", "message", "destination_name")
    return {key: job.get(key) for key in keys}


def _raise_upstream(response: httpx.Response, action: str) -> None:
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    message = detail if isinstance(detail, str) else response.text[:300] or f"HTTP {response.status_code}"
    # 403 is kept as 403 so the UI can tell "needs your Hugging Face key" apart from a failure
    status = response.status_code if response.status_code in {403, 404, 409, 413, 415, 422} else 502
    code = "upstream_auth" if response.status_code == 401 else "needs_permission" if response.status_code == 403 else "upstream_error"
    # a plain-text explanation from the sibling app is already written for people; show it as is
    text = message if isinstance(detail, str) and response.status_code in {403, 409, 422} else f"{action} failed (HTTP {response.status_code}): {message}"
    raise BridgeError(status, code, text)


def _expect_json(response: httpx.Response, action: str) -> dict[str, Any]:
    if response.status_code != 200:
        _raise_upstream(response, action)
    try:
        body = response.json()
    except ValueError as exc:
        raise BridgeError(502, "upstream_error", f"{action} returned a response that is not JSON.") from exc
    if not isinstance(body, dict):
        raise BridgeError(502, "upstream_error", f"{action} returned an unexpected JSON shape.")
    return body


def _project_detail(record: dict[str, Any]) -> dict[str, Any]:
    return {**_project_summary(record), "events": record.get("events", [])[-100:]}


def _context_card(record: dict[str, Any]) -> str:
    """Build the exact text sent to Qwen Chat: IDs, names, counts and hashes only."""
    lines = [
        f"Project: {record.get('name', '')}",
        f"Project ID: {record['project_id']}",
        f"Created: {record.get('created_at', '')}",
    ]
    cataloged: set[str] = set()
    models: set[str] = set()
    # keyed dicts keep one line per fact: repeated imports of a revision, or
    # repeated preflights of the same model on it, show only the latest result
    imports: dict[str, str] = {}
    preflights: dict[tuple[str, str], str] = {}
    generated = 0
    # this loop condenses the ledger into one line per meaningful fact
    for event in record.get("events", []):
        kind = event.get("kind")
        if kind == "image_generated":
            generated += len(event.get("images", []))
        elif kind == "dataset_catalog":
            outputs = event.get("outputs", [])
            cataloged.update(str(item.get("sha256")) for item in outputs)
            models.update(item.get("model") for item in outputs if item.get("model"))
        elif kind == "retrain_import":
            dataset = event.get("dataset", {})
            counts = dataset.get("counts", {})
            revision = str(dataset.get("manifest_sha256", ""))[:12]
            imports[revision] = f"- package '{dataset.get('package_name')}' revision {revision}: {counts.get('train', 0)} train / {counts.get('validation', 0)} validation text rows"
        elif kind == "retrain_preflight":
            revision = str(event.get("manifest_sha256", ""))[:12]
            key = (str(event.get("model_id")), revision)
            preflights[key] = f"- preflight with {key[0]} on revision {revision}: {event.get('plan_status')}"
    if generated:
        lines.append(f"Images generated with Qwen Image 2.1: {generated}")
    lines.append(f"Studio outputs cataloged in Dataset Studio: {len(cataloged)}")
    if models:
        lines.append("Image models used: " + ", ".join(sorted(models)))
    lines.append("ReTrain imports:" if imports else "ReTrain imports: none")
    lines.extend(imports.values())
    lines.append("ReTrain preflights (dry runs, no training started):" if preflights else "ReTrain preflights: none")
    lines.extend(preflights.values())
    return "\n".join(lines)
