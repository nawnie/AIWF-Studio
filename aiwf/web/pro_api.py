from __future__ import annotations

import base64
import asyncio
import heapq
import io
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError, model_validator

from aiwf.core.domain.errors import GenerationCancelledError
from aiwf.core.domain.generation import GenerationMode, GenerationRequest, JobRecord, JobState
from aiwf.core.domain.models import SCHEDULE_TYPES, normalize_schedule_id_for_sampler
from aiwf.core.domain.sana_video import SanaVideoRequest
from aiwf.core.infotext import normalize_sampler
from aiwf.infrastructure.diffusers.model_blocks import (
    is_non_selectable_image_asset_path,
    known_broken_selectable_image_asset,
)
from aiwf.services.model_download_catalog import QUICK_START_BUNDLES
from aiwf.services.pipeline_readiness import (
    READINESS_STATUSES,
    PipelineReadinessRecord,
    collect_pipeline_readiness,
    readiness_summary,
)

_RECENT_IMAGE_LIMIT = 8
_RECENT_SCAN_LIMIT = 400
_RECENT_MAX_SIDE = 512
_RECENT_MAX_BYTES = 2 * 1024 * 1024
_MAX_PRO_BATCH_IMAGES = 4
_PRO_SOURCE_IMAGE_MAX_BYTES = 15 * 1024 * 1024
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
_LOG_FILE_LIMIT = 12
_LOG_ROW_LIMIT = 80
_PRO_SANA_VIDEO_BACKEND_ENABLED = 1
_PRO_RESTART_EXIT_CODE = 75
_CAPABILITY_CACHE_TTL_SECONDS = 30.0
_SANA_VIDEO_SERVICES: dict[int, Any] = {}
_NVIDIA_SMI_CACHE: tuple[float, tuple[float, float] | None] = (0.0, None)
_PRO_VIDEO_JOBS: dict[int, dict[str, Any]] = {}
_PRO_VIDEO_JOBS_LOCK = threading.Lock()
logger = logging.getLogger(__name__)

_GRADIO_TOOL_TABS = [
    {"id": "studio", "label": "Studio", "group": "Create", "tab": "Image", "status": "ready", "summary": "Image, inpaint, ControlNet, LoRA, prompt tools"},
    {"id": "image_lab", "label": "Image Lab", "group": "Image", "tab": "Image Lab", "status": "ready", "summary": "XYZ plots, route maturity, batch runners"},
    {"id": "video", "label": "Wan / LTX Video", "group": "Video", "tab": "Video", "status": "ready", "summary": "Wan, LTX, post-processing chain"},
    {"id": "video_lab", "label": "Video Lab", "group": "Video", "tab": "Video Lab", "status": "ready", "summary": "Trim, stabilize, denoise, upscale, encode"},
    {"id": "rife", "label": "RIFE", "group": "Video", "tab": "RIFE", "status": "ready", "summary": "Frame interpolation for generated or uploaded video"},
    {"id": "audio_lab", "label": "Audio Lab", "group": "Audio", "tab": "Audio Lab", "status": "ready", "summary": "Audio cleanup, mixing, music and SFX generation"},
    {"id": "chat", "label": "Chat", "group": "Assistant", "tab": "Chat", "status": "gated", "summary": "Local chat workspace waits for the LLM worker/readiness route"},
    {"id": "model_manager", "label": "Model Manager", "group": "Models", "tab": "Models", "status": "ready", "summary": "Download, sort, inspect, convert, and fuse models"},
    {"id": "enhance", "label": "Enhance", "group": "Image", "tab": "Enhance", "status": "ready", "summary": "Upscale, restore, photo repair, face enhancement"},
    {"id": "segment", "label": "Segment", "group": "Image", "tab": "Segment", "status": "ready", "summary": "SAM masks, boxes, points, and workflow masks"},
    {"id": "reactor", "label": "ReActor", "group": "Image", "tab": "ReActor", "status": "ready", "summary": "Face swap for images and video stages"},
    {"id": "library", "label": "Library", "group": "Data", "tab": "Library", "status": "ready", "summary": "Saved output browsing and library search"},
    {"id": "pnginfo", "label": "PNG Info", "group": "Data", "tab": "PNG Info", "status": "ready", "summary": "Metadata import from saved images"},
    {"id": "history", "label": "History", "group": "Data", "tab": "History", "status": "ready", "summary": "Recent job receipts and output review"},
    {"id": "settings", "label": "Settings", "group": "System", "tab": "Settings", "status": "ready", "summary": "Paths, launch flags, UI defaults, and security"},
]

_ENGINE_LABELS = {
    "all": "All engines",
    "flux": "Flux",
    "flux2": "Flux 2",
    "sana_video": "Sana Video",
    "sd15": "Stable Diffusion 1.5",
    "sdxl": "Stable Diffusion XL",
    "sd35": "Stable Diffusion 3.5",
    "zimage": "Z-Image",
    "unknown": "Other",
}

_READINESS_NEEDS_WORK_STATUSES = (
    "metadata-only",
    "blocked-cleanly",
    "broken-runtime",
    "unsupported-no-route",
)

_READINESS_SORT_ORDER = {status: index for index, status in enumerate(READINESS_STATUSES)}


def _pro_sana_video_backend_enabled() -> bool:
    return bool(_PRO_SANA_VIDEO_BACKEND_ENABLED)


class ProGeneratePayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    mode: str = "image"
    prompt: str = ""
    negative_prompt: str = Field(default="", alias="negativePrompt")
    checkpoint_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("checkpointId", "checkpoint_id", "modelId", "model_id"),
    )
    checkpoint_title: str | None = Field(
        default=None,
        validation_alias=AliasChoices("checkpointTitle", "checkpoint_title", "title"),
    )
    sampler: str = "euler_a"
    scheduler: str = "automatic"
    steps: int = Field(default=20, ge=1, le=150)
    cfg_scale: float = Field(default=7.0, ge=0.0, le=30.0, alias="cfgScale")
    width: int = Field(default=512, ge=64, le=2048)
    height: int = Field(default=512, ge=64, le=2048)
    seed: int = -1
    batch_size: int = Field(default=1, ge=1, le=4, alias="batchSize")
    batch_count: int = Field(default=1, ge=1, le=4, alias="batchCount")
    frames: int = Field(default=81, ge=1, le=257)
    fps: float = Field(default=16.0, ge=1.0, le=60.0)
    source_image_path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("sourceImagePath", "source_image_path"),
    )
    source_image_data_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("sourceImageDataUrl", "source_image_data_url"),
    )
    source_image_name: str | None = Field(
        default=None,
        validation_alias=AliasChoices("sourceImageName", "source_image_name"),
    )
    sana_quantization: str = Field(
        default="auto",
        validation_alias=AliasChoices("sanaQuantization", "sana_quantization", "quantization"),
    )
    sana_vae_tiling: str = Field(
        default="auto",
        validation_alias=AliasChoices("sanaVaeTiling", "sana_vae_tiling", "vaeTiling", "vae_tiling"),
    )
    offload_text_encoder_after_encode: bool = Field(
        default=True,
        validation_alias=AliasChoices("offloadTextEncoderAfterEncode", "offload_text_encoder_after_encode"),
    )
    use_sage_attention: bool = Field(
        default=True,
        validation_alias=AliasChoices("useSageAttention", "use_sage_attention"),
    )
    generate_audio: bool = Field(default=False, validation_alias=AliasChoices("generateAudio", "generate_audio"))

    @model_validator(mode="after")
    def total_batch_must_be_bounded(self):
        if self.batch_size * self.batch_count > _MAX_PRO_BATCH_IMAGES:
            raise ValueError(f"batchSize * batchCount must be <= {_MAX_PRO_BATCH_IMAGES}")
        return self


def _dump_model(item: Any) -> dict[str, Any]:
    if hasattr(item, "model_dump"):
        return item.model_dump(mode="json")
    if isinstance(item, dict):
        return dict(item)
    return {
        key: value
        for key, value in vars(item).items()
        if not key.startswith("_") and isinstance(value, (str, int, float, bool, type(None), list, dict))
    }


def _checkpoint_payload(item: Any) -> dict[str, Any]:
    data = _dump_model(item)
    engine_id = _engine_id_for_architecture(str(data.get("architecture", "unknown")))
    return {
        "id": data.get("id", ""),
        "title": data.get("title", data.get("id", "")),
        "filename": data.get("filename", ""),
        "hash": data.get("hash"),
        "kind": data.get("kind", "checkpoint"),
        "architecture": data.get("architecture", "unknown"),
        "sizeBytes": int(data.get("size_bytes") or data.get("sizeBytes") or 0),
        "fileCount": int(data.get("file_count") or data.get("fileCount") or 0),
        "assetSummary": data.get("asset_summary") or data.get("assetSummary") or "",
        "engineId": engine_id,
        "engineLabel": _ENGINE_LABELS.get(engine_id, _ENGINE_LABELS["unknown"]),
    }


def _blocked_checkpoint_detail(item: Any) -> dict[str, str] | None:
    data = _dump_model(item)
    path = str(data.get("path") or data.get("filename") or "")
    if not path:
        return None
    known_block = known_broken_selectable_image_asset(path)
    if known_block is not None:
        return {
            "status": known_block.status,
            "reason": known_block.reason,
            "suggestedAction": known_block.suggested_action,
        }
    if is_non_selectable_image_asset_path(path):
        return {
            "status": "blocked-cleanly",
            "reason": "Auxiliary model asset is not selectable for normal image generation.",
            "suggestedAction": "Use this through the matching tool instead of the Generate model picker.",
        }
    return None


def _selectable_checkpoint_payloads(ctx: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selectable: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for item in _safe_list(ctx.generation.list_checkpoints):
        payload = _checkpoint_payload(item)
        block = _blocked_checkpoint_detail(item)
        if block is None:
            selectable.append(payload)
        else:
            blocked.append({**payload, **block})
    return selectable, blocked


def _sampler_payload(item: Any) -> dict[str, Any]:
    data = _dump_model(item)
    return {
        "id": data.get("id", ""),
        "label": data.get("label", data.get("id", "")),
        "family": data.get("family", "diffusers"),
        "supportsKarras": bool(data.get("supports_karras", False)),
    }


def _engine_id_for_architecture(architecture: str) -> str:
    normalized = (architecture or "").strip().lower()
    if normalized == "flux":
        return "flux"
    if normalized in {"flux2_klein", "flux2", "flux.2"}:
        return "flux2"
    if normalized in {"sana_video", "sana-video", "sanavideo"}:
        return "sana_video"
    if normalized == "z_image":
        return "zimage"
    if normalized in {"sd15", "sd1.5", "sd1", "inpaint"}:
        return "sd15"
    if normalized in {"sdxl", "sdxl_inpaint"}:
        return "sdxl"
    if normalized in {"sd35", "sd3", "stable-diffusion-3.5"}:
        return "sd35"
    return "unknown"


def _engine_summaries(checkpoints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for checkpoint in checkpoints:
        engine_id = str(checkpoint.get("engineId") or "unknown")
        counts[engine_id] = counts.get(engine_id, 0) + 1
    order = ["flux", "flux2", "sana_video", "sd15", "sdxl", "sd35", "zimage", "unknown"]
    return [
        {
            "id": engine_id,
            "label": _ENGINE_LABELS.get(engine_id, _ENGINE_LABELS["unknown"]),
            "count": counts.get(engine_id, 0),
        }
        for engine_id in order
        if counts.get(engine_id, 0) > 0
    ]


def _engine_id_for_catalog_entry(item: Any) -> str:
    text = " ".join(
        str(value or "")
        for value in (
            getattr(item, "key", ""),
            getattr(item, "title", ""),
            getattr(item, "category", ""),
            getattr(item, "repo_id", ""),
            getattr(item, "filename", ""),
        )
    ).lower()
    if "flux2" in text or "flux.2" in text:
        return "flux2"
    if "sana-video" in text or "sana_video" in text or "sanavideo" in text:
        return "sana_video"
    if "z-image" in text or "z_image" in text or "zimage" in text:
        return "zimage"
    if "sd35" in text or "sd3.5" in text or "stable-diffusion-3.5" in text or "sd 3.5" in text:
        return "sd35"
    if "sdxl" in text or "stable-diffusion-xl" in text:
        return "sdxl"
    if "sd15" in text or "sd1.5" in text or "stable-diffusion-v1-5" in text or "v1-5" in text:
        return "sd15"
    if "flux" in text:
        return "flux"
    return "unknown"


def _artifact_payload(item: Any) -> dict[str, str]:
    if isinstance(item, dict):
        path = item.get("path", "")
        infotext = item.get("infotext", "")
    else:
        path = getattr(item, "path", "")
        infotext = getattr(item, "infotext", "")
    return {"path": str(path), "infotext": str(infotext or "")}


def _image_to_data_url(
    image: Image.Image,
    *,
    max_side: int | None = None,
    max_bytes: int | None = None,
) -> str | None:
    out = image.copy()
    if max_side and max(out.size) > max_side:
        out.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    raw = buf.getvalue()
    if max_bytes is not None and len(raw) > max_bytes:
        return None
    return f"data:image/png;base64,{base64.b64encode(raw).decode('ascii')}"


def _path_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _image_path_to_data_url(path: Path) -> str | None:
    if path.suffix.lower() not in _IMAGE_EXTENSIONS:
        return None
    try:
        if path.stat().st_size > _RECENT_MAX_BYTES:
            return None
        with Image.open(path) as image:
            return _image_to_data_url(
                image.convert("RGB"),
                max_side=_RECENT_MAX_SIDE,
                max_bytes=_RECENT_MAX_BYTES,
            )
    except OSError:
        return None


def _job_state_value(job: Any) -> str:
    state = getattr(job, "state", "")
    return str(getattr(state, "value", state) or "").lower()


def _job_status(job: JobRecord | None) -> dict[str, Any]:
    if job is None:
        return {"state": "idle", "progress": 0, "message": ""}
    progress = getattr(job, "progress", None)
    state = getattr(job, "state", JobState.QUEUED)
    state_value = getattr(state, "value", str(state))
    return {
        "id": str(getattr(job, "id", "")),
        "state": state_value,
        "progress": progress.percent if progress else (100 if state == JobState.COMPLETED else 0),
        "step": progress.step if progress else 0,
        "totalSteps": progress.total_steps if progress else 0,
        "message": progress.message if progress else (getattr(job, "error", None) or ""),
        "hasResult": getattr(job, "result", None) is not None,
        "error": getattr(job, "error", None),
    }


def _pro_video_job_start(ctx: Any, request: Any) -> str:
    job_id = uuid4().hex
    with _PRO_VIDEO_JOBS_LOCK:
        _PRO_VIDEO_JOBS[id(ctx)] = {
            "id": job_id,
            "state": "running",
            "progress": 0,
            "step": 0,
            "totalSteps": max(1, int(getattr(request, "steps", 1) or 1)),
            "message": "Starting Sana video generation.",
            "hasResult": False,
            "error": "",
            "cancelRequested": False,
        }
    return job_id


def _pro_video_job_update(
    ctx: Any,
    job_id: str,
    *,
    progress: float,
    message: str,
    step: int = 0,
    total: int = 0,
) -> None:
    with _PRO_VIDEO_JOBS_LOCK:
        job = _PRO_VIDEO_JOBS.get(id(ctx))
        if not job or job.get("id") != job_id or job.get("state") != "running":
            return
        job.update(
            {
                "progress": max(0, min(100, int(round(float(progress) * 100)))),
                "step": max(0, int(step or 0)),
                "totalSteps": max(0, int(total or job.get("totalSteps") or 0)),
                "message": str(message),
            }
        )


def _pro_video_job_finish(ctx: Any, job_id: str, state: str, *, message: str = "", error: str = "") -> None:
    with _PRO_VIDEO_JOBS_LOCK:
        job = _PRO_VIDEO_JOBS.get(id(ctx))
        if not job or job.get("id") != job_id:
            return
        job.update(
            {
                "state": state,
                "progress": 100 if state == "completed" else int(job.get("progress") or 0),
                "message": str(message or job.get("message") or ""),
                "hasResult": state == "completed",
                "error": str(error or ""),
                "cancelRequested": False,
            }
        )


def _pro_video_job_status(ctx: Any) -> dict[str, Any] | None:
    with _PRO_VIDEO_JOBS_LOCK:
        job = dict(_PRO_VIDEO_JOBS.get(id(ctx)) or {})
    if not job:
        return None
    job.pop("cancelRequested", None)
    return job


def _pro_video_job_running(ctx: Any) -> bool:
    job = _pro_video_job_status(ctx)
    return bool(job and str(job.get("state") or "").lower() == "running")


def _request_pro_video_cancel(ctx: Any) -> str | None:
    with _PRO_VIDEO_JOBS_LOCK:
        job = _PRO_VIDEO_JOBS.get(id(ctx))
        if not job or str(job.get("state") or "").lower() != "running":
            return None
        job["cancelRequested"] = True
        job["message"] = "Stop requested. Sana will stop at the next safe checkpoint."
        return str(job.get("id") or "")


def _pro_video_cancel_requested(ctx: Any, job_id: str) -> bool:
    with _PRO_VIDEO_JOBS_LOCK:
        job = _PRO_VIDEO_JOBS.get(id(ctx))
        return bool(job and job.get("id") == job_id and job.get("cancelRequested"))


def _schedule_process_exit(exit_code: int, delay_seconds: float = 0.25) -> None:
    def _worker() -> None:
        time.sleep(max(0.0, float(delay_seconds)))
        os._exit(int(exit_code))

    threading.Thread(target=_worker, name="aiwf-pro-exit", daemon=True).start()


def _schedule_process_restart(delay_seconds: float = 0.25) -> None:
    root = Path(__file__).resolve().parents[2]
    launch_script = root / "launch_pro.py"
    forwarded_args = [arg for arg in sys.argv[1:] if arg != "--no-autolaunch"]
    command = [sys.executable, str(launch_script), *forwarded_args, "--no-autolaunch"]

    def _worker() -> None:
        time.sleep(max(0.0, float(delay_seconds)))
        try:
            subprocess.Popen(command, cwd=str(root))
        finally:
            os._exit(_PRO_RESTART_EXIT_CODE)

    threading.Thread(target=_worker, name="aiwf-pro-restart", daemon=True).start()


def _image_generation_running(ctx: Any) -> bool:
    generation = getattr(ctx, "generation", None)
    active_job = None
    if generation is not None and callable(getattr(generation, "active_job", None)):
        try:
            active_job = generation.active_job()
        except Exception:
            active_job = None
    if active_job is None:
        return False
    state_value = _job_state_value(active_job)
    return state_value not in {"", "idle", "completed"}


def _image_generation_pending(ctx: Any) -> bool:
    generation = getattr(ctx, "generation", None)
    if generation is None or not callable(getattr(generation, "pending_count", None)):
        return False
    try:
        return int(generation.pending_count()) > 0
    except Exception:
        return False


def _safe_recent_jobs(ctx: Any, limit: int) -> list[Any]:
    try:
        return list(ctx.generation.recent_jobs(limit))
    except Exception:
        return []


def _recent_terminal_image_job(ctx: Any) -> Any | None:
    for job in _safe_recent_jobs(ctx, 12):
        if _job_state_value(job) in {"failed", "cancelled", "canceled"}:
            return job
    return None


def _safe_output_root(ctx: Any) -> Path | None:
    flags = getattr(ctx, "flags", None)
    resolved = getattr(flags, "resolved_output_dir", None)
    if callable(resolved):
        try:
            return Path(resolved()).resolve()
        except OSError:
            return None
    return None


def _video_source_image_path(ctx: Any, payload: ProGeneratePayload) -> str | None:
    if payload.source_image_path:
        root = _safe_output_root(ctx)
        if root is None:
            raise HTTPException(status_code=500, detail="Output directory is not available for source image upload.")
        try:
            candidate = Path(payload.source_image_path).expanduser().resolve()
        except OSError as exc:
            raise HTTPException(status_code=422, detail="Source image path is not readable.") from exc
        if not _path_inside(candidate, root) or not candidate.is_file():
            raise HTTPException(status_code=422, detail="Source image path must point to an existing workspace output.")
        if candidate.suffix.lower() not in _IMAGE_EXTENSIONS:
            raise HTTPException(status_code=422, detail="Source image path must be PNG, JPEG, or WebP.")
        return str(candidate)
    data_url = (payload.source_image_data_url or "").strip()
    if not data_url:
        return None
    if "," not in data_url or ";base64" not in data_url.partition(",")[0].lower():
        raise HTTPException(status_code=422, detail="Source image must be a base64 image data URL.")
    header, encoded = data_url.split(",", 1)
    if not header.lower().startswith("data:image/"):
        raise HTTPException(status_code=422, detail="Source image must be PNG, JPEG, or WebP.")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Source image data is not valid base64.") from exc
    if len(raw) > _PRO_SOURCE_IMAGE_MAX_BYTES:
        raise HTTPException(status_code=413, detail="Source image is too large for Pro video upload.")
    root = _safe_output_root(ctx)
    if root is None:
        raise HTTPException(status_code=500, detail="Output directory is not available for source image upload.")
    input_dir = root / "pro-inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    target = input_dir / f"video-source-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')}.png"
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.convert("RGB").save(target, format="PNG")
    except OSError as exc:
        raise HTTPException(status_code=422, detail="Source image could not be opened.") from exc
    return str(target)


def _output_asset_path(ctx: Any, requested_path: str) -> Path:
    root = _safe_output_root(ctx)
    if root is None:
        raise HTTPException(status_code=404, detail="Output directory is not available.")
    target = (root / requested_path).resolve()
    if not _path_inside(target, root) or not target.is_file():
        raise HTTPException(status_code=404, detail="Output asset not found.")
    return target


def _output_asset_url(ctx: Any, path: str | Path) -> str:
    output_path = Path(path)
    root = _safe_output_root(ctx)
    if root is None:
        return str(output_path)
    try:
        resolved = output_path.resolve()
    except OSError:
        return str(output_path)
    if not _path_inside(resolved, root):
        return str(output_path)
    relative = resolved.relative_to(root).as_posix()
    return f"/api/pro/outputs/{quote(relative, safe='/')}"


def _sana_video_service(ctx: Any):
    service = getattr(ctx, "sana_video", None)
    if service is not None:
        return service
    key = id(ctx)
    service = _SANA_VIDEO_SERVICES.get(key)
    if service is None:
        from aiwf.services.sana_video import SanaVideoService

        devices = getattr(getattr(ctx, "generation", None), "backend", None)
        service = SanaVideoService(
            getattr(ctx, "flags", None),
            getattr(ctx, "settings", None),
            getattr(devices, "devices", None),
            supervisor=getattr(ctx, "supervisor", None),
        )
        _SANA_VIDEO_SERVICES[key] = service
    return service


def _sana_video_model_payload(ctx: Any) -> dict[str, Any]:
    if not _pro_sana_video_backend_enabled():
        return {
            "id": "",
            "title": "SANA-Video 2B 480p",
            "filename": "SANA-Video_2B_480p_diffusers",
            "hash": None,
            "kind": "video",
            "architecture": "sana_video",
            "engineId": "sana_video",
            "engineLabel": _ENGINE_LABELS["sana_video"],
            "backend": "Diffusers",
            "status": "Disabled",
        }
    try:
        service = _sana_video_service(ctx)
        model_path = service.default_model_path()
        ready = (model_path / "model_index.json").is_file()
    except Exception:
        model_path = Path("models/sana-video/Diffusers/SANA-Video_2B_480p_diffusers")
        ready = False
    return {
        "id": str(model_path),
        "title": "SANA-Video 2B 480p",
        "filename": model_path.name,
        "hash": None,
        "kind": "video",
        "architecture": "sana_video",
        "engineId": "sana_video",
        "engineLabel": _ENGINE_LABELS["sana_video"],
        "backend": "Diffusers",
        "status": "Ready" if ready else "Needs snapshot",
    }


def _recent_paths_from_disk(root: Path, *, limit: int) -> list[Path]:
    if not root.exists():
        return []
    heap: list[tuple[float, str, Path]] = []
    inspected = 0
    stack = [root]
    while stack and inspected < _RECENT_SCAN_LIMIT:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            if inspected >= _RECENT_SCAN_LIMIT:
                break
            try:
                if entry.is_dir():
                    if not entry.name.startswith("."):
                        stack.append(entry)
                    continue
                if entry.suffix.lower() not in _IMAGE_EXTENSIONS:
                    continue
                inspected += 1
                stat = entry.stat()
            except OSError:
                continue
            row = (stat.st_mtime, str(entry), entry)
            if len(heap) < limit:
                heapq.heappush(heap, row)
            else:
                heapq.heappushpop(heap, row)
    return [item[2] for item in sorted(heap, reverse=True)]


def _recent_output_images(ctx: Any, *, limit: int = _RECENT_IMAGE_LIMIT) -> list[dict[str, Any]]:
    root = _safe_output_root(ctx)
    seen_paths: set[str] = set()
    images: list[dict[str, Any]] = []

    def add_image(payload: dict[str, Any]) -> None:
        if len(images) < limit:
            images.append(payload)

    for job in _safe_recent_jobs(ctx, limit * 2):
        result = getattr(job, "result", None)
        if getattr(job, "state", None) != JobState.COMPLETED or result is None:
            continue
        artifacts = [_artifact_payload(item) for item in (getattr(result, "artifacts", []) or [])]
        for index, image in enumerate(getattr(result, "images", []) or []):
            artifact = artifacts[index] if index < len(artifacts) else {}
            artifact_path = artifact.get("path")
            if artifact_path:
                try:
                    seen_paths.add(str(Path(artifact_path).resolve()))
                except OSError:
                    pass
            data_url = _image_to_data_url(image, max_side=_RECENT_MAX_SIDE, max_bytes=_RECENT_MAX_BYTES)
            if data_url:
                add_image(
                    {
                        "source": "memory",
                        "dataUrl": data_url,
                        "path": artifact_path,
                        "seed": (getattr(result, "seeds", []) or [None])[index]
                        if index < len(getattr(result, "seeds", []) or [])
                        else None,
                        "infotext": (getattr(result, "infotexts", []) or [""])[index]
                        if index < len(getattr(result, "infotexts", []) or [])
                        else artifact.get("infotext", ""),
                    }
                )
            if len(images) >= limit:
                return images
        for artifact_data in artifacts:
            path = Path(artifact_data["path"])
            try:
                resolved_path = str(path.resolve())
            except OSError:
                continue
            if resolved_path in seen_paths:
                continue
            if not path.is_file() or (root is not None and not _path_inside(path, root)):
                continue
            seen_paths.add(resolved_path)
            data_url = _image_path_to_data_url(path)
            if data_url:
                add_image(
                    {
                        "source": "artifact",
                        "dataUrl": data_url,
                        "path": str(path),
                        "infotext": artifact_data["infotext"],
                    }
                )
            if len(images) >= limit:
                return images

    if root is None:
        return images
    for path in _recent_paths_from_disk(root, limit=limit):
        try:
            resolved = str(path.resolve())
        except OSError:
            continue
        if resolved in seen_paths:
            continue
        data_url = _image_path_to_data_url(path)
        if not data_url:
            continue
        add_image({"source": "disk", "dataUrl": data_url, "path": str(path), "infotext": ""})
        if len(images) >= limit:
            break
    return images


def _safe_jsonl_tail(path: Path, *, limit: int = _LOG_ROW_LIMIT) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            value = {"message": line[:1000]}
        if isinstance(value, dict):
            rows.append(
                {
                    "id": f"{path.name}-{index}",
                    "source": path.name,
                    "time": str(value.get("logged_at") or value.get("created_at") or value.get("time") or ""),
                    "title": str(value.get("action") or value.get("kind") or value.get("status") or path.stem),
                    "detail": str(value.get("detail") or value.get("message") or value.get("error") or value)[:1000],
                }
            )
    return rows


def _safe_text_tail(path: Path, *, limit: int = 24) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]
    except OSError:
        return []
    return [
        {
            "id": f"{path.name}-{index}",
            "source": path.name,
            "time": "",
            "title": path.stem,
            "detail": line[:1000],
        }
        for index, line in enumerate(lines)
        if line.strip()
    ]


def _log_root(ctx: Any) -> Path | None:
    return _safe_output_root(ctx)


def _sana_log_root(ctx: Any) -> Path | None:
    flags = getattr(ctx, "flags", None)
    data_dir = getattr(flags, "data_dir", None)
    if data_dir is None:
        return None
    try:
        return (Path(data_dir) / "_local" / "logs").resolve()
    except OSError:
        return None


def _sana_receipt_paths(ctx: Any, *, limit: int = 8) -> list[Path]:
    root = _sana_log_root(ctx)
    if root is None or not root.is_dir():
        return []
    paths: list[Path] = []
    latest = root / "sana_video_latest.json"
    if latest.is_file():
        paths.append(latest)
    candidates = []
    try:
        candidates = [path for path in root.glob("sana_video_*.json") if path.name != latest.name and path.is_file()]
    except OSError:
        candidates = []
    def modified_time(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    candidates.sort(key=modified_time, reverse=True)
    for path in candidates:
        if path not in paths:
            paths.append(path)
        if len(paths) >= limit:
            break
    return paths


def _latest_sana_receipt_path(ctx: Any) -> str:
    paths = _sana_receipt_paths(ctx, limit=1)
    return str(paths[0]) if paths else ""


def _failure_index_path(ctx: Any) -> str:
    root = _log_root(ctx)
    if root is None:
        return ""
    path = root / "failures" / "index.jsonl"
    return str(path) if path.is_file() else ""


def _pro_error_detail(message: str, **extra: Any) -> dict[str, Any]:
    detail: dict[str, Any] = {"message": str(message)}
    for key, value in extra.items():
        if value not in (None, "", [], {}):
            detail[key] = value
    return detail


def _log_files(ctx: Any) -> list[dict[str, Any]]:
    root = _log_root(ctx)
    candidates = []
    if root is not None:
        candidates.extend(
            [
                root / "client-events.jsonl",
                root / "client-errors.jsonl",
                root / "client-errors.log",
                root / "genlog" / "generation-log.jsonl",
                root / "failures" / "index.jsonl",
            ]
        )
    candidates.extend(_sana_receipt_paths(ctx))
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in candidates:
        if not path.is_file():
            continue
        try:
            stat = path.stat()
            resolved = str(path.resolve())
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        rows.append(
            {
                "name": path.name,
                "path": str(path),
                "sizeBytes": stat.st_size,
                "modifiedAt": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            }
        )
    rows.sort(key=lambda item: str(item.get("modifiedAt") or ""), reverse=True)
    return rows[:_LOG_FILE_LIMIT]


def _safe_json_file_event(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(value, dict):
        return []
    error = value.get("error")
    result = value.get("result")
    if isinstance(error, dict):
        detail = str(error.get("message") or error)
    elif isinstance(result, dict):
        detail = str(result.get("message") or result.get("output_path") or value)
    else:
        detail = str(value.get("message") or value.get("output_path") or value)
    return [
        {
            "id": f"{path.name}-{value.get('status', 'receipt')}",
            "source": path.name,
            "time": str(value.get("created_at") or ""),
            "title": f"Sana video {value.get('status') or 'receipt'}",
            "detail": detail[:1000],
        }
    ]


def _event_rows(ctx: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in _safe_recent_jobs(ctx, 24):
        status = _job_status(job)
        rows.append(
            {
                "id": status.get("id") or f"job-{len(rows)}",
                "source": "generation",
                "time": "",
                "title": str(status.get("state") or "job"),
                "detail": str(status.get("message") or status.get("error") or "Generation job receipt"),
            }
        )
    root = _log_root(ctx)
    if root is not None:
        rows.extend(_safe_jsonl_tail(root / "client-events.jsonl", limit=24))
        rows.extend(_safe_jsonl_tail(root / "client-errors.jsonl", limit=24))
        rows.extend(_safe_jsonl_tail(root / "genlog" / "generation-log.jsonl", limit=24))
        rows.extend(_safe_jsonl_tail(root / "failures" / "index.jsonl", limit=24))
        rows.extend(_safe_text_tail(root / "client-errors.log", limit=12))
    for path in _sana_receipt_paths(ctx):
        rows.extend(_safe_json_file_event(path))
    return rows[:_LOG_ROW_LIMIT]


def _recent_output_payload(ctx: Any) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for index, item in enumerate(_recent_output_images(ctx, limit=_RECENT_IMAGE_LIMIT)):
        path = str(item.get("path") or "")
        width = 0
        height = 0
        created_at = ""
        if path:
            try:
                path_obj = Path(path)
                if path_obj.is_file():
                    stat = path_obj.stat()
                    created_at = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
                    with Image.open(path_obj) as image:
                        width, height = image.size
            except OSError:
                pass
        outputs.append(
            {
                "id": f"recent-{index}-{path or item.get('source', 'memory')}",
                "url": item.get("dataUrl"),
                "thumbnailUrl": item.get("dataUrl"),
                "path": path,
                "prompt": str(item.get("infotext") or "Local output"),
                "width": width,
                "height": height,
                "createdAt": created_at,
                "mode": "image",
                "seed": item.get("seed"),
                "modelName": None,
                "status": "available",
                "source": item.get("source", "output"),
            }
        )
    return outputs


def _data_summary(ctx: Any) -> dict[str, Any]:
    output_root = _safe_output_root(ctx)
    checkpoints, blocked_checkpoints = _selectable_checkpoint_payloads(ctx)
    recent_outputs = _recent_output_payload(ctx)
    return {
        "outputRoot": str(output_root) if output_root is not None else "",
        "counts": {
            "checkpoints": len(checkpoints),
            "blockedCheckpoints": len(blocked_checkpoints),
            "recentOutputs": len(recent_outputs),
            "engines": len(_engine_summaries(checkpoints)),
        },
        "engines": _engine_summaries(checkpoints),
        "recentOutputs": recent_outputs,
    }


def _download_payload(ctx: Any) -> dict[str, Any]:
    service = getattr(ctx, "model_download", None)
    if service is None:
        return {
            "categories": [],
            "bundles": QUICK_START_BUNDLES,
            "catalog": [],
            "counts": {"categories": 0, "catalog": 0, "installed": 0},
        }

    categories: list[dict[str, Any]] = []
    for label, key in _safe_list(service.category_choices):
        destination = ""
        try:
            destination = str(service.destination_dir(key))
        except Exception:
            destination = ""
        categories.append({"key": key, "label": label, "destination": destination})

    catalog: list[dict[str, Any]] = []
    installed_count = 0
    for item in _safe_list(service.list_catalog):
        installed = False
        destination = ""
        try:
            installed = bool(service.is_catalog_installed(item))
        except Exception:
            installed = False
        try:
            destination = str(service.destination_dir(item.category))
        except Exception:
            destination = ""
        if installed:
            installed_count += 1
        engine_id = _engine_id_for_catalog_entry(item)
        catalog.append(
            {
                "key": item.key,
                "title": item.title,
                "category": item.category,
                "source": item.source,
                "sizeMb": item.size_mb,
                "repoId": item.repo_id,
                "filename": item.filename,
                "url": item.url,
                "notes": item.notes,
                "snapshot": item.snapshot,
                "installed": installed,
                "destination": destination,
                "engineId": engine_id,
                "engineLabel": _ENGINE_LABELS.get(engine_id, _ENGINE_LABELS["unknown"]),
            }
        )

    return {
        "categories": categories,
        "bundles": QUICK_START_BUNDLES,
        "catalog": catalog,
        "counts": {
            "categories": len(categories),
            "catalog": len(catalog),
            "installed": installed_count,
        },
    }


def _settings_payload(ctx: Any) -> dict[str, Any]:
    settings = getattr(ctx, "settings", None)
    flags = getattr(ctx, "flags", None)
    settings_path = getattr(ctx, "settings_path", None)
    launch_settings_path = getattr(ctx, "launch_settings_path", None)
    return {
        "paths": {
            "settings": str(settings_path or ""),
            "launch": str(launch_settings_path or ""),
            "models": str(flags.resolved_models_dir()) if flags is not None else "",
            "checkpoints": str(flags.resolved_ckpt_dir()) if flags is not None else "",
            "outputs": str(flags.resolved_output_dir()) if flags is not None else "",
        },
        "generationDefaults": _settings_defaults(ctx),
        "ui": {
            "accentPreset": getattr(settings, "accent_preset", "mint"),
            "galleryColumns": getattr(settings, "gallery_columns", 2),
            "galleryHeight": getattr(settings, "gallery_height", 480),
            "livePreview": getattr(settings, "enable_live_preview", True),
            "hiddenTabs": list(getattr(settings, "hidden_tabs", []) or []),
        },
        "runtime": {
            "listen": bool(getattr(flags, "listen", False)),
            "api": bool(getattr(flags, "api", False)),
            "genlog": bool(getattr(flags, "genlog", False)),
            "backend": getattr(flags, "inference_backend", "unknown") if flags is not None else "unknown",
            "attention": getattr(flags, "attention_backend", "unknown") if flags is not None else "unknown",
        },
    }


def _safe_count(ctx: Any, attr: str, method: str) -> int:
    service = getattr(ctx, attr, None)
    callable_obj = getattr(service, method, None)
    if not callable(callable_obj):
        return 0
    try:
        return len(list(callable_obj()))
    except Exception:
        return 0


def _safe_bool(ctx: Any, attr: str, method: str) -> bool:
    service = getattr(ctx, attr, None)
    callable_obj = getattr(service, method, None)
    if not callable(callable_obj):
        return False
    try:
        return bool(callable_obj())
    except Exception:
        return False


def _capability_status(count: int, *, optional_ready: bool = True) -> str:
    if count > 0:
        return "ready"
    return "available" if optional_ready else "needs-assets"


def _readiness_count_template() -> dict[str, int]:
    return {status: 0 for status in READINESS_STATUSES}


def _readiness_record_payload(record: PipelineReadinessRecord) -> dict[str, Any]:
    label = record.id
    if record.path:
        label = Path(record.path).name or record.id
    return {
        "id": record.id,
        "family": record.family,
        "assetType": record.asset_type,
        "path": record.path,
        "label": label,
        "status": record.status,
        "route": record.route,
        "reason": record.reason,
        "storage": record.storage,
        "quantization": record.quantization,
        "requiredVae": record.required_vae,
        "requiredTextEncoder": record.required_text_encoder,
        "tokenizer": record.tokenizer,
        "smokeCommand": record.smoke_command,
        "receiptPath": record.receipt_path,
        "suggestedAction": record.suggested_action,
    }


def _readiness_family_payload(records: list[PipelineReadinessRecord]) -> list[dict[str, Any]]:
    families: dict[str, dict[str, Any]] = {}
    for record in records:
        family = record.family or "unknown"
        item = families.setdefault(
            family,
            {
                "family": family,
                "counts": _readiness_count_template(),
                "total": 0,
            },
        )
        item["counts"][record.status] = item["counts"].get(record.status, 0) + 1
        item["total"] += 1
    return sorted(families.values(), key=lambda item: (-int(item["total"]), str(item["family"])))


def _readiness_payload(ctx: Any) -> dict[str, Any]:
    try:
        flags = getattr(ctx, "flags", None)
        if flags is None:
            raise RuntimeError("runtime flags are unavailable")
        records = collect_pipeline_readiness(
            flags,
            getattr(ctx, "settings", None),
            include_downloads=False,
            force_rescan=False,
        )
    except Exception as exc:
        return {
            "counts": _readiness_count_template(),
            "families": [],
            "working": [],
            "needsWork": [],
            "metadataOnlyCount": 0,
            "total": 0,
            "error": str(exc),
        }

    counts = readiness_summary(records)
    working = [record for record in records if record.status == "working"]
    needs_work = [record for record in records if record.status in _READINESS_NEEDS_WORK_STATUSES]
    needs_work.sort(
        key=lambda record: (
            _READINESS_SORT_ORDER.get(record.status, 99),
            record.family,
            record.asset_type,
            record.id.lower(),
        )
    )
    return {
        "counts": counts,
        "families": _readiness_family_payload(records),
        "working": [_readiness_record_payload(record) for record in working[:8]],
        "needsWork": [_readiness_record_payload(record) for record in needs_work[:10]],
        "metadataOnlyCount": counts.get("metadata-only", 0),
        "total": len(records),
        "error": "",
    }


def _cached_readiness_payload(ctx: Any) -> dict[str, Any]:
    now = time.monotonic()
    cached = getattr(ctx, "_pro_capability_cache", None)
    if isinstance(cached, dict):
        cached_at = float(cached.get("cached_at") or 0.0)
        payload = cached.get("payload")
        if payload is not None and now - cached_at < _CAPABILITY_CACHE_TTL_SECONDS:
            return payload
    payload = _readiness_payload(ctx)
    setattr(ctx, "_pro_capability_cache", {"cached_at": now, "payload": payload})
    return payload


def _capability_payload(ctx: Any) -> dict[str, Any]:
    checkpoints, blocked_checkpoints = _selectable_checkpoint_payloads(ctx)
    sana_model = _sana_video_model_payload(ctx)
    sana_enabled = _pro_sana_video_backend_enabled()
    sana_ready = sana_enabled and str(sana_model.get("status") or "").lower() == "ready"
    lora_count = _safe_count(ctx, "generation", "list_loras")
    controlnet_count = _safe_count(ctx, "controlnet", "list_models")
    controlnet_modules = _safe_count(ctx, "controlnet", "list_modules")
    sam_count = _safe_count(ctx, "segment", "list_models")
    upscaler_count = _safe_count(ctx, "enhance", "list_upscalers")
    restorer_count = _safe_count(ctx, "enhance", "list_restorers")
    reactor_model_count = _safe_count(ctx, "faceswap", "list_models")
    reactor_face_count = _safe_count(ctx, "faceswap", "list_face_models")
    wan_count = _safe_count(ctx, "wan", "list_local_models")
    wan_lora_count = _safe_count(ctx, "wan", "list_local_loras")
    wan_available = _safe_bool(ctx, "wan", "available")
    readiness = _cached_readiness_payload(ctx)
    readiness_counts = readiness.get("counts", {})
    llm_count = 0
    for family in readiness.get("families", []):
        if str(family.get("family", "")).lower() in {"llm", "llm-vl", "vision-language"}:
            llm_count += int(family.get("total", 0) or 0)

    tools = [
        {
            "id": "image-generation",
            "label": "Image generation",
            "group": "Create",
            "status": "ready" if checkpoints else "needs-assets",
            "count": len(checkpoints),
            "route": "create",
            "summary": "TXT2IMG, IMG2IMG, inpaint, samplers, schedulers, sizes, and seeds.",
            "details": [
                f"{len(checkpoints)} base models",
                f"{lora_count} LoRAs",
                "React Pro generate endpoint is wired.",
            ],
        },
        {
            "id": "controlnet",
            "label": "ControlNet",
            "group": "Image",
            "status": _capability_status(controlnet_count),
            "count": controlnet_count,
            "route": "create",
            "summary": "Gradio has multi-unit ControlNet controls; React needs the full request schema next.",
            "details": [f"{controlnet_count} models", f"{controlnet_modules} preprocessors"],
        },
        {
            "id": "segment",
            "label": "Segment / SAM",
            "group": "Image",
            "status": _capability_status(sam_count),
            "count": sam_count,
            "route": "modal:segmentation",
            "summary": "SAM masks are available in Gradio and exposed as a React tool popup.",
            "details": [f"{sam_count} SAM models", "Box, point, and text-prompt masks in Gradio."],
        },
        {
            "id": "enhance",
            "label": "Enhance",
            "group": "Image",
            "status": _capability_status(upscaler_count + restorer_count),
            "count": upscaler_count + restorer_count,
            "route": "tools",
            "summary": "Upscale, restore, old-photo repair, and video frame enhancement run from Gradio.",
            "details": [f"{upscaler_count} upscalers", f"{restorer_count} restorers"],
        },
        {
            "id": "reactor",
            "label": "ReActor",
            "group": "Image",
            "status": _capability_status(reactor_model_count),
            "count": reactor_model_count,
            "route": "modal:reactor",
            "summary": "Face swap can run from Gradio and video post-processing stages.",
            "details": [f"{reactor_model_count} swapper models", f"{reactor_face_count} saved face models"],
        },
        {
            "id": "video",
            "label": "Sana / Wan / LTX video",
            "group": "Video",
            "status": "ready" if sana_ready or (wan_available and wan_count > 0) else "available",
            "count": wan_count + (1 if sana_ready else 0),
            "route": "create",
            "summary": "Sana Video runs from React Pro; Wan, LTX, RIFE, and post stages remain visible in Gradio.",
            "details": [
                "Sana ready" if sana_ready else ("Sana backend disabled" if not sana_enabled else "Sana snapshot missing"),
                f"{wan_count} Wan models",
                f"{wan_lora_count} Wan LoRAs",
            ],
        },
        {
            "id": "audio",
            "label": "Audio Lab",
            "group": "Audio",
            "status": "available",
            "count": 0,
            "route": "tools",
            "summary": "Audio cleanup, mixing, music, SFX, and video-conditioned audio are in Gradio.",
            "details": ["Audio Lab has its own optional engine path."],
        },
        {
            "id": "llm-vl",
            "label": "LLM / vision-language",
            "group": "Assistant",
            "status": "not-wired",
            "count": llm_count,
            "route": "planned",
            "summary": "Tracked in the model readiness ledger; Pro does not expose a promoted chat worker yet.",
            "details": [
                f"{llm_count} candidate assets",
                f"{readiness_counts.get('unsupported-no-route', 0)} unsupported/no-route checks",
            ],
        },
        {
            "id": "data-tools",
            "label": "Library, PNG Info, History",
            "group": "Data",
            "status": "ready",
            "count": 3,
            "route": "data",
            "summary": "Data review tools remain available in Gradio while React Data catches up.",
            "details": ["Library search", "PNG metadata import", "History receipts"],
        },
    ]
    return {
        "gradioTabs": _GRADIO_TOOL_TABS,
        "tools": tools,
        "counts": {
            "gradioTabs": len(_GRADIO_TOOL_TABS),
            "reactRails": 7,
            "checkpoints": len(checkpoints),
            "blockedCheckpoints": len(blocked_checkpoints),
            "loras": lora_count,
            "controlnet": controlnet_count,
            "sam": sam_count,
            "reactor": reactor_model_count,
            "enhance": upscaler_count + restorer_count,
            "sanaVideo": 1 if sana_ready else 0,
            "wan": wan_count,
        },
        "readiness": readiness,
        "notes": [
            "React Pro now shows Gradio parity and asset readiness.",
            f"{len(blocked_checkpoints)} blocked model assets are hidden from the normal Generate picker.",
            "Heavy tool execution stays in Gradio until each Pro request path is typed and smoke-tested.",
        ],
    }


def _checkpoint_id_from_payload(ctx: Any, payload: ProGeneratePayload) -> str | None:
    if payload.checkpoint_id:
        return payload.checkpoint_id
    if not payload.checkpoint_title:
        return None
    needle = payload.checkpoint_title.strip().lower()
    for checkpoint in _safe_list(ctx.generation.list_checkpoints):
        data = _dump_model(checkpoint)
        candidates = (data.get("id"), data.get("title"), data.get("filename"))
        if any(str(candidate or "").lower() == needle for candidate in candidates):
            return str(data.get("id") or payload.checkpoint_title)
    return payload.checkpoint_title


def _resolve_checkpoint_for_generation_guard(ctx: Any, checkpoint_id: str | None) -> Any | None:
    generation = getattr(ctx, "generation", None)
    resolver = getattr(generation, "resolve_checkpoint", None)
    if callable(resolver):
        try:
            return resolver(checkpoint_id)
        except Exception:
            return None
    checkpoints = _safe_list(getattr(generation, "list_checkpoints", lambda: []))
    if checkpoint_id:
        needle = checkpoint_id.strip().lower()
        for checkpoint in checkpoints:
            data = _dump_model(checkpoint)
            candidates = (data.get("id"), data.get("title"), data.get("filename"))
            if any(str(candidate or "").lower() == needle for candidate in candidates):
                return checkpoint
    return checkpoints[0] if checkpoints else None


def _assert_checkpoint_selectable(ctx: Any, checkpoint_id: str | None) -> None:
    checkpoint = _resolve_checkpoint_for_generation_guard(ctx, checkpoint_id)
    if checkpoint is None:
        return
    block = _blocked_checkpoint_detail(checkpoint)
    if block is None:
        return
    data = _dump_model(checkpoint)
    label = str(data.get("title") or data.get("id") or data.get("filename") or "Selected model")
    raise HTTPException(
        status_code=422,
        detail=_pro_error_detail(
            f"{label} is blocked for normal Pro generation.",
            checkpointId=str(data.get("id") or checkpoint_id or ""),
            status=block["status"],
            reason=block["reason"],
            suggestedAction=block["suggestedAction"],
        ),
    )


def _safe_list(callable_obj) -> list[Any]:
    try:
        return list(callable_obj())
    except Exception:
        return []


def _format_gb(value: float) -> str:
    if value <= 0:
        return "0 GB"
    return f"{value:.1f} GB" if value < 10 else f"{value:.0f} GB"


def _format_file_size(path: str) -> str:
    if not path:
        return "Unknown"
    try:
        size = Path(path).stat().st_size
    except OSError:
        return "Unknown"
    gb = size / 1024**3
    if gb >= 1:
        return _format_gb(gb)
    return f"{max(1, round(size / 1024**2))} MB"


def _usage_metric(label: str, value: str, percent: float, tone: str = "neutral") -> dict[str, Any]:
    return {
        "label": label,
        "value": value,
        "percent": max(0, min(100, int(round(percent)))),
        "tone": tone,
    }


def _nvidia_gpu_utilization() -> tuple[float, float] | None:
    global _NVIDIA_SMI_CACHE
    now = time.monotonic()
    cached_at, cached = _NVIDIA_SMI_CACHE
    if now - cached_at < 0.9:
        return cached
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            encoding="utf-8",
            errors="replace",
            timeout=1.5,
        )
        first_line = (completed.stdout or "").splitlines()[0]
        gpu_value, memory_value = [float(part.strip()) for part in first_line.split(",", 1)]
        cached = (gpu_value, memory_value)
    except Exception:
        cached = None
    _NVIDIA_SMI_CACHE = (now, cached)
    return cached


def _runtime_resource_metrics(ctx: Any) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    try:
        torch = __import__("torch")
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            used = max(0, total - free)
            metrics.append(
                _usage_metric(
                    "VRAM",
                    f"{_format_gb(used / 1024**3)} / {_format_gb(total / 1024**3)}",
                    (used / total * 100) if total else 0,
                    "mint",
                )
            )
            gpu_utilization = _nvidia_gpu_utilization()
            if gpu_utilization is not None:
                gpu_percent, _memory_percent = gpu_utilization
                metrics.append(_usage_metric("GPU utilization", f"{gpu_percent:.0f}%", gpu_percent, "mint"))
            else:
                metrics.append(_usage_metric("GPU utilization", "Unavailable", 0, "neutral"))
        else:
            metrics.append(_usage_metric("VRAM", "CUDA unavailable", 0, "neutral"))
            metrics.append(_usage_metric("GPU utilization", "CUDA unavailable", 0, "neutral"))
    except Exception:
        metrics.append(_usage_metric("VRAM", "Unavailable", 0, "neutral"))
        metrics.append(_usage_metric("GPU utilization", "Unavailable", 0, "neutral"))

    try:
        psutil = __import__("psutil")
        ram = psutil.virtual_memory()
        cpu_percent = float(psutil.cpu_percent(interval=None))
        metrics.append(
            _usage_metric(
                "RAM",
                f"{_format_gb((ram.total - ram.available) / 1024**3)} / {_format_gb(ram.total / 1024**3)}",
                float(ram.percent),
                "blue",
            )
        )
        metrics.append(_usage_metric("CPU", f"{cpu_percent:.0f}%", cpu_percent, "blue"))
    except Exception:
        metrics.append(_usage_metric("RAM", "Unavailable", 0, "neutral"))
        metrics.append(_usage_metric("CPU", "Unavailable", 0, "neutral"))

    root = _safe_output_root(ctx) or Path.cwd()
    try:
        usage = shutil.disk_usage(root)
        used = usage.total - usage.free
        metrics.append(
            _usage_metric(
                "Storage",
                f"{_format_gb(used / 1024**3)} / {_format_gb(usage.total / 1024**3)}",
                (used / usage.total * 100) if usage.total else 0,
                "amber",
            )
        )
    except OSError:
        metrics.append(_usage_metric("Storage", "Unavailable", 0, "neutral"))

    return metrics


def _runtime_precision(flags: Any) -> str:
    if flags is None:
        return "Unknown"
    if bool(getattr(flags, "no_half", False)):
        return "FP32"
    if bool(getattr(flags, "fp8", False)) or bool(getattr(flags, "fluxfp8", False)) or bool(getattr(flags, "fp8_quant", False)):
        return "FP8/FP16"
    return "FP16"


def _runtime_loaded_model(ctx: Any) -> dict[str, Any]:
    generation = getattr(ctx, "generation", None)
    backend = getattr(generation, "backend", None)
    active = getattr(backend, "_active", None)
    if active is None:
        return {
            "name": "No model loaded",
            "type": "Text-to-Image",
            "baseModel": "None",
            "sizeOnDisk": "Unknown",
            "precision": "Unknown",
            "vae": "",
            "textEncoder": "",
            "unet": "",
            "loaded": False,
        }

    data = _dump_model(active)
    checkpoint_id = str(data.get("id") or data.get("title") or "")
    loaded = False
    if callable(getattr(backend, "is_checkpoint_loaded", None)):
        try:
            loaded = bool(backend.is_checkpoint_loaded(checkpoint_id))
        except Exception:
            loaded = False
    architecture = str(data.get("architecture") or "unknown")
    return {
        "name": str(data.get("title") or data.get("id") or "Loaded model"),
        "type": "Text-to-Image",
        "baseModel": architecture,
        "sizeOnDisk": _format_file_size(str(data.get("path") or "")),
        "precision": _runtime_precision(getattr(ctx, "flags", None)),
        "vae": str(getattr(getattr(ctx, "flags", None), "vae_path", "") or ""),
        "textEncoder": "Runtime owned",
        "unet": str(data.get("filename") or ""),
        "loaded": loaded,
    }


def _runtime_summary(ctx: Any) -> dict[str, Any]:
    flags = getattr(ctx, "flags", None)
    generation = getattr(ctx, "generation", None)
    backend = getattr(generation, "backend", None)
    devices = getattr(backend, "devices", None)
    active_job = None
    if generation is not None and callable(getattr(generation, "active_job", None)):
        try:
            active_job = generation.active_job()
        except Exception:
            active_job = None
    pending_count = 0
    if generation is not None and callable(getattr(generation, "pending_count", None)):
        try:
            pending_count = max(0, int(generation.pending_count()))
        except Exception:
            pending_count = 0
    video_job = _pro_video_job_status(ctx)
    video_job_state = str((video_job or {}).get("state") or "").lower()
    video_job_running = bool(video_job and video_job_state == "running")
    video_job_terminal = bool(video_job and video_job_state in {"failed", "cancelled", "canceled"})
    video_job_completed = bool(video_job and video_job_state == "completed")
    image_job_running = _image_generation_running(ctx)
    recent_terminal_image_job = _recent_terminal_image_job(ctx)
    if image_job_running:
        job_status = _job_status(active_job)
    elif video_job_running or video_job_terminal:
        job_status = video_job
    elif recent_terminal_image_job is not None:
        job_status = _job_status(recent_terminal_image_job)
    elif video_job_completed:
        job_status = video_job
    else:
        job_status = _job_status(None)
    try:
        torch_version = __import__("torch").__version__.split("+", 1)[0]
    except Exception:
        torch_version = "unavailable"
    device = "Unknown"
    if devices is not None and callable(getattr(devices, "describe", None)):
        try:
            device = devices.describe()
        except Exception:
            device = "Unknown"
    status = "idle"
    if image_job_running or video_job_running:
        status = "running"
    elif video_job_terminal:
        status = video_job_state or "failed"
    elif recent_terminal_image_job is not None:
        terminal_state = _job_state_value(recent_terminal_image_job)
        status = terminal_state or "failed"
    return {
        "status": status,
        "job": job_status,
        "device": device,
        "backend": getattr(flags, "inference_backend", backend.__class__.__name__ if backend is not None else "unknown"),
        "precision": _runtime_precision(flags),
        "attention": getattr(flags, "attention_backend", "unknown") if flags is not None else "unknown",
        "maxResolution": "2048 x 2048 request cap",
        "queueCount": (1 if image_job_running else 0) + (1 if video_job_running else 0) + pending_count,
        "resources": _runtime_resource_metrics(ctx),
        "loadedModel": _runtime_loaded_model(ctx),
        "python": platform.python_version(),
        "torch": torch_version,
        "port": getattr(ctx, "runtime_port", None),
        "listen": bool(getattr(flags, "listen", False)),
        "api": True,
        "localOnly": not bool(getattr(flags, "listen", False)),
    }


async def _runtime_sse_events(ctx: Any, request: Request):
    while True:
        if await request.is_disconnected():
            break
        payload = _runtime_summary(ctx)
        yield f"event: runtime\ndata: {json.dumps(payload, default=str)}\n\n"
        await asyncio.sleep(1.0)


def _settings_defaults(ctx: Any) -> dict[str, Any]:
    settings = getattr(ctx, "settings", None)
    return {
        "prompt": "",
        "negativePrompt": getattr(settings, "default_negative_prompt", "") or "",
        "useDefaultNegative": bool(getattr(settings, "use_default_negative", True)),
        "checkpointId": getattr(settings, "last_checkpoint_id", None),
        "sampler": getattr(settings, "default_sampler", "euler_a"),
        "scheduler": getattr(settings, "default_scheduler", "automatic"),
        "steps": int(getattr(settings, "default_steps", 20)),
        "cfgScale": float(getattr(settings, "default_cfg_scale", 7.0)),
        "width": int(getattr(settings, "default_width", 512)),
        "height": int(getattr(settings, "default_height", 512)),
        "seed": -1,
        "batchSize": 1,
        "batchCount": 1,
        "saveImages": bool(getattr(settings, "save_images", True)),
    }


def _generation_mode_from_payload(payload: ProGeneratePayload) -> GenerationMode:
    normalized = (payload.mode or "image").strip().lower()
    if normalized in {"image", "txt2img"}:
        return GenerationMode.TXT2IMG
    raise HTTPException(status_code=422, detail="React Pro generation currently supports image/txt2img mode.")


def _generation_request(ctx: Any, payload: ProGeneratePayload) -> GenerationRequest:
    sampler = normalize_sampler(payload.sampler) or "euler_a"
    scheduler = normalize_schedule_id_for_sampler(sampler, payload.scheduler)
    try:
        return GenerationRequest(
            mode=_generation_mode_from_payload(payload),
            prompt=payload.prompt,
            negative_prompt=payload.negative_prompt,
            checkpoint_id=_checkpoint_id_from_payload(ctx, payload),
            sampler=sampler,
            scheduler=scheduler,
            steps=payload.steps,
            cfg_scale=payload.cfg_scale,
            width=payload.width,
            height=payload.height,
            seed=payload.seed,
            batch_size=payload.batch_size,
            batch_count=payload.batch_count,
            enable_hr=False,
            controlnet_units=[],
            sdxl_refiner_enabled=False,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc


def _job_recent_output_payloads(job: Any) -> list[dict[str, Any]]:
    request = getattr(job, "request", None)
    result = getattr(job, "result", None)
    if result is None:
        return []
    images = list(getattr(result, "images", []) or [])
    seeds = list(getattr(result, "seeds", []) or [])
    infotexts = list(getattr(result, "infotexts", []) or [])
    artifacts = [_artifact_payload(item) for item in (getattr(result, "artifacts", []) or [])]
    mode = str(getattr(getattr(result, "mode", None), "value", getattr(result, "mode", "txt2img")))
    prompt = str(getattr(request, "prompt", "") or "")
    model_name = str(getattr(request, "checkpoint_id", "") or "")
    outputs: list[dict[str, Any]] = []
    for index, image in enumerate(images):
        data_url = _image_to_data_url(image)
        if not data_url:
            continue
        artifact = artifacts[index] if index < len(artifacts) else {}
        path = str(artifact.get("path") or "")
        created_at = datetime.now(timezone.utc).isoformat()
        if path:
            try:
                stat = Path(path).stat()
                created_at = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
            except OSError:
                pass
        width, height = getattr(image, "size", (0, 0))
        outputs.append(
            {
                "id": f"{getattr(job, 'id', 'job')}-{index}",
                "url": data_url,
                "thumbnailUrl": data_url,
                "path": path,
                "prompt": infotexts[index] if index < len(infotexts) and infotexts[index] else prompt,
                "width": width,
                "height": height,
                "createdAt": created_at,
                "mode": mode,
                "seed": seeds[index] if index < len(seeds) else None,
                "modelName": model_name,
                "status": "completed",
                "source": "generation",
            }
        )
    return outputs


def _generate_response(job: Any) -> dict[str, Any]:
    result = getattr(job, "result", None)
    if result is None:
        raise HTTPException(status_code=500, detail=getattr(job, "error", None) or "Generation failed")
    recent_outputs = _job_recent_output_payloads(job)
    encoded_images = [item["url"] for item in recent_outputs]
    return {
        "jobId": str(getattr(job, "id", getattr(result, "job_id", ""))),
        "status": "completed",
        "job": _job_status(job),
        "output": recent_outputs[0] if recent_outputs else None,
        "image": encoded_images[0] if encoded_images else None,
        "images": encoded_images,
        "recentOutputs": recent_outputs,
        "seeds": list(getattr(result, "seeds", []) or []),
        "infotexts": list(getattr(result, "infotexts", []) or []),
        "artifacts": [_artifact_payload(item) for item in (getattr(result, "artifacts", []) or [])],
        "message": f"Generated {len(recent_outputs)} image(s).",
    }


def _sana_video_request_from_payload(ctx: Any, payload: ProGeneratePayload) -> SanaVideoRequest:
    raw_model_path = str(payload.checkpoint_id or "")
    model_path = raw_model_path if any(marker in raw_model_path.lower() for marker in ("sana", "\\", "/", ":")) else ""
    source_image_path = _video_source_image_path(ctx, payload)
    try:
        return SanaVideoRequest(
            prompt=payload.prompt,
            negative_prompt=payload.negative_prompt,
            model_path=model_path,
            source_image_path=source_image_path,
            pipeline="image_to_video" if source_image_path else "text_to_video",
            width=payload.width,
            height=payload.height,
            frames=payload.frames,
            fps=payload.fps,
            seed=payload.seed,
            steps=min(int(payload.steps), 100),
            cfg_scale=payload.cfg_scale,
            quantization=payload.sana_quantization,
            vae_tiling=payload.sana_vae_tiling,
            offload_text_encoder_after_encode=payload.offload_text_encoder_after_encode,
            use_sage_attention=payload.use_sage_attention,
            generate_audio=payload.generate_audio,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc


def _sana_video_output_payload(ctx: Any, result: Any, payload: ProGeneratePayload) -> dict[str, Any]:
    output_path = str(getattr(result, "output_path", "") or "")
    path = Path(output_path)
    created_at = datetime.now(timezone.utc).isoformat()
    if path.is_file():
        try:
            created_at = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
        except OSError:
            pass
    return {
        "id": f"sana-video-{path.stem or 'output'}",
        "url": _output_asset_url(ctx, output_path),
        "thumbnailUrl": _output_asset_url(ctx, output_path),
        "path": output_path,
        "prompt": str(getattr(result, "infotext", "") or payload.prompt),
        "width": int(getattr(result, "width", payload.width) or payload.width),
        "height": int(getattr(result, "height", payload.height) or payload.height),
        "createdAt": created_at,
        "mode": "video",
        "seed": payload.seed,
        "modelName": payload.checkpoint_id or "SANA-Video 2B 480p",
        "status": "completed",
        "source": "sana-video",
    }


def _generate_sana_video_response(ctx: Any, payload: ProGeneratePayload) -> dict[str, Any]:
    if not _pro_sana_video_backend_enabled():
        raise HTTPException(status_code=503, detail="React Pro Sana Video backend is disabled in aiwf/web/pro_api.py.")
    request = _sana_video_request_from_payload(ctx, payload)
    job_id = _pro_video_job_start(ctx, request)
    progress: list[dict[str, Any]] = []

    def on_progress(
        stage: str,
        progress_value: float,
        message: str,
        step: int = 0,
        total: int = 0,
        seconds: float = 0.0,
    ) -> None:
        stage_text = str(stage)
        message_text = str(message)
        if stage_text == "error":
            _pro_video_job_finish(ctx, job_id, "failed", message=message_text, error=message_text)
        else:
            _pro_video_job_update(
                ctx,
                job_id,
                progress=float(progress_value),
                message=message_text,
                step=int(step or 0),
                total=int(total or 0),
            )
        progress.append(
            {
                "stage": stage_text,
                "progress": float(progress_value),
                "message": message_text,
                "step": int(step or 0),
                "total": int(total or 0),
                "seconds": float(seconds or 0.0),
            }
        )
        if stage_text != "error" and _pro_video_cancel_requested(ctx, job_id):
            _pro_video_job_finish(ctx, job_id, "cancelled", message="Sana video generation cancelled.")
            raise GenerationCancelledError("Sana video generation cancelled.")

    try:
        result = _sana_video_service(ctx).generate(request, on_progress=on_progress)
    except GenerationCancelledError as exc:
        _pro_video_job_finish(ctx, job_id, "cancelled", message=str(exc))
        raise HTTPException(status_code=499, detail=_pro_error_detail(str(exc), job=_pro_video_job_status(ctx))) from exc
    except Exception as exc:
        _pro_video_job_finish(ctx, job_id, "failed", message=str(exc), error=str(exc))
        receipt_path = _latest_sana_receipt_path(ctx)
        logger.exception(
            "Pro Sana video generation failed: job=%s model=%s size=%sx%s frames=%s steps=%s receipt=%s",
            job_id,
            request.model_path or "default",
            request.width,
            request.height,
            request.frames,
            request.steps,
            receipt_path or "",
        )
        raise HTTPException(
            status_code=500,
            detail=_pro_error_detail(str(exc), receiptPath=receipt_path, job=_pro_video_job_status(ctx)),
        ) from exc

    result_progress = list(getattr(result, "progress", None) or progress)
    output = _sana_video_output_payload(ctx, result, payload)
    _pro_video_job_finish(ctx, job_id, "completed", message=str(getattr(result, "message", "") or "Sana video complete."))
    return {
        "jobId": job_id,
        "status": "completed",
        "output": output,
        "video": output["url"],
        "recentOutputs": [output],
        "progress": result_progress,
        "timings": dict(getattr(result, "timings", {}) or {}),
        "receiptPath": str(getattr(result, "receipt_path", "") or ""),
        "attentionBackend": str(getattr(result, "attention_backend", "") or ""),
        "quantization": str(getattr(result, "quantization", "") or request.quantization),
        "vaeTiling": str(getattr(result, "vae_tiling", "") or request.vae_tiling),
        "message": str(getattr(result, "message", "") or "Sana video complete."),
    }


def build_router(ctx: Any) -> APIRouter:
    router = APIRouter(prefix="/api/pro")

    @router.get("/runtime")
    def runtime():
        return _runtime_summary(ctx)

    @router.get("/runtime/stream")
    async def runtime_stream(request: Request):
        return StreamingResponse(
            _runtime_sse_events(ctx, request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @router.get("/bootstrap")
    def bootstrap():
        checkpoints, blocked_checkpoints = _selectable_checkpoint_payloads(ctx)
        if _pro_sana_video_backend_enabled():
            sana_model = _sana_video_model_payload(ctx)
            if sana_model["id"] not in {str(item.get("id") or "") for item in checkpoints}:
                checkpoints.append(sana_model)
        samplers = [_sampler_payload(item) for item in _safe_list(ctx.generation.list_samplers)]
        return {
            "runtime": _runtime_summary(ctx),
            "settings": _settings_defaults(ctx),
            "checkpoints": checkpoints,
            "blockedCheckpoints": blocked_checkpoints,
            "counts": {
                "checkpoints": len(checkpoints),
                "blockedCheckpoints": len(blocked_checkpoints),
            },
            "engines": _engine_summaries(checkpoints),
            "samplers": samplers,
            "schedulers": [item.model_dump(mode="json") for item in SCHEDULE_TYPES],
            "recentImages": _recent_output_images(ctx),
        }

    @router.get("/data")
    def data():
        return _data_summary(ctx)

    @router.get("/downloads")
    def downloads():
        return _download_payload(ctx)

    @router.get("/logs")
    def logs():
        return {
            "runtime": _runtime_summary(ctx),
            "files": _log_files(ctx),
            "events": _event_rows(ctx),
        }

    @router.get("/settings")
    def settings():
        return _settings_payload(ctx)

    @router.get("/capabilities")
    def capabilities():
        return _capability_payload(ctx)

    @router.post("/restart")
    def restart():
        _schedule_process_restart()
        return {"status": "restart_requested"}

    @router.get("/outputs/{requested_path:path}")
    def output_asset(requested_path: str):
        return FileResponse(_output_asset_path(ctx, requested_path))

    @router.post("/generate")
    def generate(payload: ProGeneratePayload):
        if (payload.mode or "").strip().lower() in {"video", "sana", "sana_video"}:
            if _pro_video_job_running(ctx) or _image_generation_running(ctx) or _image_generation_pending(ctx):
                raise HTTPException(status_code=409, detail="A generation job is already running. Stop it or wait for it to finish.")
            return _generate_sana_video_response(ctx, payload)
        if _pro_video_job_running(ctx):
            raise HTTPException(status_code=409, detail="A Sana video job is already running. Stop it or wait for it to finish.")
        if _image_generation_running(ctx) or _image_generation_pending(ctx):
            raise HTTPException(status_code=409, detail="An image generation job is already running. Stop it or wait for it to finish.")
        request = _generation_request(ctx, payload)
        _assert_checkpoint_selectable(ctx, request.checkpoint_id)
        try:
            logger.info(
                "Pro image generation started: model=%s size=%sx%s steps=%s batch=%s prompt=%r",
                request.checkpoint_id or "default",
                request.width,
                request.height,
                request.steps,
                request.batch_size,
                request.prompt[:160],
            )
            job = ctx.generation.submit(request)
            logger.info(
                "Pro image generation finished: job=%s state=%s message=%s",
                job.id,
                getattr(job.state, "value", job.state),
                getattr(job.progress, "message", "") if getattr(job, "progress", None) else "",
            )
        except Exception as exc:
            failed_job = _recent_terminal_image_job(ctx)
            failure_log_path = _failure_index_path(ctx)
            logger.exception(
                "Pro image generation failed: model=%s size=%sx%s steps=%s batch=%s failure_log=%s",
                request.checkpoint_id or "default",
                request.width,
                request.height,
                request.steps,
                request.batch_size,
                failure_log_path or "",
            )
            raise HTTPException(
                status_code=500,
                detail=_pro_error_detail(str(exc), failureLogPath=failure_log_path, job=_job_status(failed_job)),
            ) from exc
        return _generate_response(job)

    @router.post("/interrupt")
    def interrupt():
        video_job_id = _request_pro_video_cancel(ctx)
        try:
            generation_interrupt = getattr(getattr(ctx, "generation", None), "interrupt", None)
            if callable(generation_interrupt):
                generation_interrupt()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"status": "interrupt_requested", "videoJobId": video_job_id or ""}

    return router
