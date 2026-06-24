from __future__ import annotations

import base64
import heapq
import io
import platform
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from PIL import Image
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError, model_validator

from aiwf.core.domain.generation import GenerationMode, GenerationRequest, JobRecord, JobState
from aiwf.core.domain.models import SCHEDULE_TYPES, normalize_schedule_id_for_sampler
from aiwf.core.infotext import normalize_sampler

_RECENT_IMAGE_LIMIT = 8
_RECENT_SCAN_LIMIT = 400
_RECENT_MAX_SIDE = 512
_RECENT_MAX_BYTES = 2 * 1024 * 1024
_MAX_PRO_BATCH_IMAGES = 4
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


class ProGeneratePayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    prompt: str = ""
    negative_prompt: str = Field(default="", alias="negativePrompt")
    checkpoint_id: str | None = Field(default=None, validation_alias=AliasChoices("checkpointId", "checkpoint_id"))
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
    return {
        "id": data.get("id", ""),
        "title": data.get("title", data.get("id", "")),
        "filename": data.get("filename", ""),
        "hash": data.get("hash"),
        "kind": data.get("kind", "checkpoint"),
        "architecture": data.get("architecture", "unknown"),
    }


def _sampler_payload(item: Any) -> dict[str, Any]:
    data = _dump_model(item)
    return {
        "id": data.get("id", ""),
        "label": data.get("label", data.get("id", "")),
        "family": data.get("family", "diffusers"),
        "supportsKarras": bool(data.get("supports_karras", False)),
    }


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


def _safe_recent_jobs(ctx: Any, limit: int) -> list[Any]:
    try:
        return list(ctx.generation.recent_jobs(limit))
    except Exception:
        return []


def _safe_output_root(ctx: Any) -> Path | None:
    flags = getattr(ctx, "flags", None)
    resolved = getattr(flags, "resolved_output_dir", None)
    if callable(resolved):
        try:
            return Path(resolved()).resolve()
        except OSError:
            return None
    return None


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


def _safe_list(callable_obj) -> list[Any]:
    try:
        return list(callable_obj())
    except Exception:
        return []


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
    return {
        "status": "running" if active_job is not None else "idle",
        "job": _job_status(active_job),
        "device": device,
        "backend": getattr(flags, "inference_backend", backend.__class__.__name__ if backend is not None else "unknown"),
        "python": platform.python_version(),
        "torch": torch_version,
        "port": getattr(ctx, "runtime_port", None),
        "listen": bool(getattr(flags, "listen", False)),
        "api": True,
        "localOnly": not bool(getattr(flags, "listen", False)),
    }


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


def _generation_request(ctx: Any, payload: ProGeneratePayload) -> GenerationRequest:
    sampler = normalize_sampler(payload.sampler) or "euler_a"
    scheduler = normalize_schedule_id_for_sampler(sampler, payload.scheduler)
    try:
        return GenerationRequest(
            mode=GenerationMode.TXT2IMG,
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


def _generate_response(job: Any) -> dict[str, Any]:
    result = getattr(job, "result", None)
    if result is None:
        raise HTTPException(status_code=500, detail=getattr(job, "error", None) or "Generation failed")
    images = getattr(result, "images", []) or []
    encoded_images = [_image_to_data_url(image) for image in images]
    encoded_images = [url for url in encoded_images if url]
    return {
        "jobId": str(getattr(job, "id", getattr(result, "job_id", ""))),
        "status": _job_status(job),
        "image": encoded_images[0] if encoded_images else None,
        "images": encoded_images,
        "seeds": list(getattr(result, "seeds", []) or []),
        "infotexts": list(getattr(result, "infotexts", []) or []),
        "artifacts": [_artifact_payload(item) for item in (getattr(result, "artifacts", []) or [])],
        "message": f"Generated {len(images)} image(s).",
    }


def build_router(ctx: Any) -> APIRouter:
    router = APIRouter(prefix="/api/pro")

    @router.get("/runtime")
    def runtime():
        return _runtime_summary(ctx)

    @router.get("/bootstrap")
    def bootstrap():
        checkpoints = [_checkpoint_payload(item) for item in _safe_list(ctx.generation.list_checkpoints)]
        samplers = [_sampler_payload(item) for item in _safe_list(ctx.generation.list_samplers)]
        return {
            "runtime": _runtime_summary(ctx),
            "settings": _settings_defaults(ctx),
            "checkpoints": checkpoints,
            "samplers": samplers,
            "schedulers": [item.model_dump(mode="json") for item in SCHEDULE_TYPES],
            "recentImages": _recent_output_images(ctx),
        }

    @router.post("/generate")
    def generate(payload: ProGeneratePayload):
        request = _generation_request(ctx, payload)
        try:
            job = ctx.generation.submit(request)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return _generate_response(job)

    return router
