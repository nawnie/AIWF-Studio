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

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from PIL import Image
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError, model_validator

from aiwf.core.config.launch import LaunchSettings, save_launch_settings
from aiwf.core.config.settings import normalize_vram_profile
from aiwf.core.domain.controlnet import ControlNetUnit
from aiwf.core.domain.errors import GenerationCancelledError
from aiwf.core.domain.generation import GenerationMode, GenerationRequest, JobRecord, JobState
from aiwf.core.domain.models import SCHEDULE_TYPES, normalize_schedule_id_for_sampler
from aiwf.core.domain.sana_video import SanaVideoRequest
from aiwf.core.infotext import normalize_sampler, parse_infotext
from aiwf.infrastructure.diffusers.model_blocks import (
    is_non_selectable_image_asset_path,
    known_broken_selectable_image_asset,
)
from aiwf.infrastructure.diffusers.model_arch import is_sd3_architecture
from aiwf.infrastructure.model_inventory import MODEL_EXTENSIONS, scan_and_write_model_inventory
from aiwf.infrastructure.model_sorter import SORT_INBOX_DIRNAME, reorganize_models, sort_inbox_models
from aiwf.services.model_download_catalog import CIVITAI_BROWSE_LINKS, QUICK_START_BUNDLES
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
_RECENT_INFOTEXT_MAX_CHARS = 20_000
_MAX_PRO_BATCH_IMAGES = 4
_PRO_SOURCE_IMAGE_MAX_BYTES = 15 * 1024 * 1024
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".avif"}
_LOG_FILE_LIMIT = 12
_LOG_ROW_LIMIT = 80
_PRO_SANA_VIDEO_BACKEND_ENABLED = 1
_PRO_RESTART_EXIT_CODE = 75
_CAPABILITY_CACHE_TTL_SECONDS = 30.0
_CAPABILITY_BACKGROUND_REFRESH_SECONDS = 300.0
_RUNTIME_RUNNING_TICK_SECONDS = 0.075
_RUNTIME_IDLE_TICK_SECONDS = 0.25
_RUNTIME_RESOURCE_CACHE_SECONDS = 0.35
_STARTUP_SPLASH_MIN_MS = 1800
_STARTUP_SPLASH_READY_HOLD_MS = 1200
_READINESS_SNAPSHOT_FILENAMES = (
    "pipeline_readiness_current_inventory.json",
    "pipeline_readiness_with_downloads_latest.json",
    "pipeline_readiness_latest.json",
)
_SANA_VIDEO_SERVICES: dict[int, Any] = {}
_WAN_SERVICES: dict[int, Any] = {}
_VSR_SERVICES: dict[int, Any] = {}
_RIFE_SERVICES: dict[int, Any] = {}
_AUDIO_SERVICES: dict[int, Any] = {}
_RUNTIME_RESOURCE_CACHE: dict[int, tuple[float, list[dict[str, Any]]]] = {}
_VIDEO_LAB_UPLOAD_MAX_BYTES = 2 * 1024 * 1024 * 1024
_VIDEO_LAB_UPLOAD_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".avi",
    ".m4v",
    ".wmv",
    ".flv",
    ".mpeg",
    ".mpg",
    ".ts",
    ".mts",
    ".m2ts",
    ".3gp",
    ".ogv",
}
_MODEL_UPLOAD_MAX_BYTES = 120 * 1024 * 1024 * 1024
_NVIDIA_SMI_CACHE: tuple[float, tuple[float, float] | None] = (0.0, None)
_JOB_PREVIEW_CACHE: dict[tuple[str, int, int], str] = {}
_JOB_PREVIEW_CACHE_LOCK = threading.Lock()
_JOB_PREVIEW_CACHE_LIMIT = 32
_SD35_LARGE_ACCESS_CACHE: tuple[float, bool] = (0.0, False)
_PRO_VIDEO_JOBS: dict[int, dict[str, Any]] = {}
_PRO_VIDEO_JOBS_LOCK = threading.Lock()
logger = logging.getLogger(__name__)

_GRADIO_TOOL_TABS = [
    {"id": "studio", "label": "Studio", "group": "Create", "tab": "Image", "status": "ready", "summary": "Image, inpaint, ControlNet, LoRA, prompt tools"},
    {"id": "image_lab", "label": "Image Lab", "group": "Image", "tab": "Image Lab", "status": "ready", "summary": "XYZ plots, route maturity, batch runners"},
    {"id": "video", "label": "Wan / LTX Video", "group": "Video", "tab": "Video", "status": "ready", "summary": "Wan, LTX, post-processing chain"},
    {"id": "sana_video", "label": "Sana Video", "group": "Video", "tab": "Sana Video", "status": "experimental", "summary": "Sana text-to-video and image-to-video"},
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
    "flux_fill": "Flux Fill (inpaint)",
    "flux2": "Flux 2",
    "sana_video": "Sana Video",
    "wan": "Wan Video",
    "sd15": "Stable Diffusion 1.5",
    "sdxl": "Stable Diffusion XL",
    "sd35": "Stable Diffusion 3.5",
    "zimage": "Z-Image",
    "qwen": "Qwen Image",
    "sana": "Sana",
    "unknown": "Other",
}

_READINESS_NEEDS_WORK_STATUSES = (
    "metadata-only",
    "blocked-cleanly",
    "broken-runtime",
    "unsupported-no-route",
)
_HIDDEN_V1_MODEL_ARCHITECTURES = {"anima", "qwen_image_nunchaku"}

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
    pipeline_backend: str = Field(
        default="aiwf",
        validation_alias=AliasChoices("pipelineBackend", "pipeline_backend", "backend"),
    )
    sampler: str = "euler_a"
    scheduler: str = "automatic"
    steps: int = Field(default=20, ge=1, le=150)
    cfg_scale: float = Field(default=7.0, ge=0.0, le=30.0, alias="cfgScale")
    width: int = Field(default=512, ge=64, le=2048)
    height: int = Field(default=512, ge=64, le=2048)
    seed: int = -1
    clip_skip: int = Field(default=1, ge=1, le=12, validation_alias=AliasChoices("clipSkip", "clip_skip"))
    batch_size: int = Field(default=1, ge=1, le=4, alias="batchSize")
    batch_count: int = Field(default=1, ge=1, le=4, alias="batchCount")
    enable_hr: bool = Field(
        default=False,
        validation_alias=AliasChoices("enableHr", "enable_hr", "enableHires", "enable_hires"),
    )
    hr_scale: float = Field(
        default=2.0,
        ge=1.0,
        le=4.0,
        validation_alias=AliasChoices("hrScale", "hr_scale", "hiresScale", "hires_scale"),
    )
    hr_steps: int = Field(
        default=20,
        ge=1,
        le=150,
        validation_alias=AliasChoices("hrSteps", "hr_steps", "hiresSteps", "hires_steps"),
    )
    hr_denoising_strength: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices(
            "hrDenoisingStrength",
            "hr_denoising_strength",
            "hiresDenoise",
            "hires_denoise",
        ),
    )
    hr_upscaler: str = Field(
        default="lanczos",
        validation_alias=AliasChoices("hrUpscaler", "hr_upscaler", "hiresUpscaler", "hires_upscaler"),
    )
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
    wan_runtime_mode: str = Field(
        default="fast_5b",
        validation_alias=AliasChoices("wanRuntimeMode", "wan_runtime_mode", "runtimeMode", "runtime_mode"),
    )
    high_noise_model_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("highNoiseModelId", "high_noise_model_id"),
    )
    low_noise_model_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("lowNoiseModelId", "low_noise_model_id"),
    )
    high_noise_steps: int = Field(default=20, ge=1, le=60, validation_alias=AliasChoices("highNoiseSteps", "high_noise_steps"))
    low_noise_steps: int = Field(default=1, ge=1, le=60, validation_alias=AliasChoices("lowNoiseSteps", "low_noise_steps"))
    boundary_ratio: float = Field(default=0.875, ge=0.0, le=1.0, validation_alias=AliasChoices("boundaryRatio", "boundary_ratio"))
    high_noise_lora_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("highNoiseLoraId", "high_noise_lora_id"),
    )
    high_noise_lora_scale: float = Field(default=1.0, ge=0.0, le=2.0, validation_alias=AliasChoices("highNoiseLoraScale", "high_noise_lora_scale"))
    low_noise_lora_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("lowNoiseLoraId", "low_noise_lora_id"),
    )
    low_noise_lora_scale: float = Field(default=1.0, ge=0.0, le=2.0, validation_alias=AliasChoices("lowNoiseLoraScale", "low_noise_lora_scale"))
    vae_id: str | None = Field(default=None, validation_alias=AliasChoices("vaeId", "vae_id"))
    text_encoder_path: str | None = Field(default=None, validation_alias=AliasChoices("textEncoderPath", "text_encoder_path"))
    wan_offload: str = Field(default="balanced", validation_alias=AliasChoices("wanOffload", "wan_offload", "offload"))
    wan_sigma_type: str = Field(default="simple", validation_alias=AliasChoices("wanSigmaType", "wan_sigma_type", "sigmaType", "sigma_type"))
    wan_sampler: str = Field(default="unipc", validation_alias=AliasChoices("wanSampler", "wan_sampler"))
    wan_flow_shift: float = Field(default=5.0, ge=0.5, le=25.0, validation_alias=AliasChoices("wanFlowShift", "wan_flow_shift", "flowShift", "flow_shift"))
    init_image_data_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("initImageDataUrl", "init_image_data_url", "initImage"),
    )
    mask_image_data_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("maskImageDataUrl", "mask_image_data_url", "maskImage"),
    )
    denoising_strength: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("denoisingStrength", "denoising_strength"),
    )
    mask_blur: int = Field(
        default=4,
        ge=0,
        le=64,
        validation_alias=AliasChoices("maskBlur", "mask_blur"),
    )
    inpaint_only_masked: bool = Field(
        default=False,
        validation_alias=AliasChoices("inpaintOnlyMasked", "inpaint_only_masked"),
    )
    inpaint_masked_padding: int = Field(
        default=32,
        ge=0,
        le=256,
        validation_alias=AliasChoices("inpaintMaskedPadding", "inpaint_masked_padding"),
    )
    inpaint_mask_content: str = Field(
        default="original",
        validation_alias=AliasChoices("inpaintMaskContent", "inpaint_mask_content"),
    )
    controlnet_units: list["ProControlNetUnitPayload"] = Field(
        default_factory=list,
        validation_alias=AliasChoices("controlnetUnits", "controlnet_units"),
    )

    @model_validator(mode="after")
    def total_batch_must_be_bounded(self):
        if self.batch_size * self.batch_count > _MAX_PRO_BATCH_IMAGES:
            raise ValueError(f"batchSize * batchCount must be <= {_MAX_PRO_BATCH_IMAGES}")
        return self


class ProControlNetUnitPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    enabled: bool = True
    model: str | None = None
    module: str = "none"
    weight: float = Field(default=1.0, ge=0.0, le=2.0)
    image: str | None = None
    mask: str | None = None
    resize_mode: str = Field(default="resize", validation_alias=AliasChoices("resizeMode", "resize_mode"))
    processor_res: int = Field(default=512, ge=64, le=4096, validation_alias=AliasChoices("processorRes", "processor_res"))
    threshold_a: float = Field(default=64.0, validation_alias=AliasChoices("thresholdA", "threshold_a"))
    threshold_b: float = Field(default=64.0, validation_alias=AliasChoices("thresholdB", "threshold_b"))
    guidance_start: float = Field(default=0.0, ge=0.0, le=1.0, validation_alias=AliasChoices("guidanceStart", "guidance_start"))
    guidance_end: float = Field(default=1.0, ge=0.0, le=1.0, validation_alias=AliasChoices("guidanceEnd", "guidance_end"))
    control_mode: str = Field(default="balanced", validation_alias=AliasChoices("controlMode", "control_mode"))


class ProMetadataImportPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    image_data_url: str = Field(
        validation_alias=AliasChoices("imageDataUrl", "image_data_url", "dataUrl", "data_url"),
    )
    filename: str = ""


class ProSettingsUpdatePayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    generation_defaults: dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("generationDefaults", "generation_defaults"),
    )
    ui: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    video: dict[str, Any] = Field(default_factory=dict)
    runtime: dict[str, Any] = Field(default_factory=dict)


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


def _checkpoint_payload(ctx: Any, item: Any) -> dict[str, Any]:
    data = _dump_model(item)
    checkpoint_id = str(data.get("id") or "")
    engine_id = _engine_id_for_architecture(str(data.get("architecture", "unknown")))
    size_bytes = int(data.get("size_bytes") or data.get("sizeBytes") or 0)
    # Rough resident-VRAM estimate: weights that live on the GPU plus a
    # working margin for activations/VAE. Large LM text encoders are kept in
    # system RAM by the backend, so they are intentionally not counted.
    est_vram_gb = round(size_bytes / 1024**3 + 2.5, 1) if size_bytes > 0 else 0.0
    return {
        "id": checkpoint_id,
        "title": data.get("title", data.get("id", "")),
        "filename": data.get("filename", ""),
        "hash": data.get("hash"),
        "kind": data.get("kind", "checkpoint"),
        "architecture": data.get("architecture", "unknown"),
        "sizeBytes": size_bytes,
        "fileCount": int(data.get("file_count") or data.get("fileCount") or 0),
        "assetSummary": data.get("asset_summary") or data.get("assetSummary") or "",
        "engineId": engine_id,
        "engineLabel": _ENGINE_LABELS.get(engine_id, _ENGINE_LABELS["unknown"]),
        "estVramGb": est_vram_gb,
        "heavyFor12Gb": bool(est_vram_gb > 12.0),
        "generationPreset": _generation_preset_payload(ctx, checkpoint_id),
    }


def _generation_preset_payload(ctx: Any, checkpoint_id: str) -> dict[str, Any]:
    if not checkpoint_id:
        return {}
    get_model_preset = getattr(getattr(ctx, "generation", None), "get_model_preset", None)
    if not callable(get_model_preset):
        return {}
    try:
        preset = get_model_preset(checkpoint_id)
    except Exception:
        return {}
    return dict(preset) if isinstance(preset, dict) else {}


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


def _runtime_checkpoint_block(ctx: Any, item: Any) -> dict[str, str] | None:
    data = _dump_model(item)
    architecture = str(data.get("architecture") or "")
    path = str(data.get("path") or data.get("filename") or "")
    if not path:
        return None
    if _is_hidden_v1_checkpoint_architecture(architecture):
        return {
            "status": "coming-soon",
            "reason": "This model family is blocked for the v1 app until its native runtime has a passing smoke receipt.",
            "suggestedAction": "Keep the files installed for sorting/research, but use a supported v1 image route for generation.",
        }
    if (
        is_sd3_architecture(architecture)
        and "large" in path.lower()
        and not _sd35_large_access_available()
    ):
        return {
            "status": "blocked-cleanly",
            "reason": "SD3.5 Large single-file checkpoints need gated Stability AI config files unless those files are cached locally.",
            "suggestedAction": "Sign in to Hugging Face with SD3.5 Large access, or provide a local diffusers pipeline folder for this model.",
        }
    return None


def _is_hidden_v1_checkpoint_architecture(architecture: str) -> bool:
    return (architecture or "").strip().lower() in _HIDDEN_V1_MODEL_ARCHITECTURES


def _huggingface_token() -> str:
    try:
        from huggingface_hub import get_token

        return str(get_token() or "")
    except Exception:
        return str(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or "")


def _sd35_large_access_available() -> bool:
    if _sd35_large_config_cached():
        return True
    token = _huggingface_token()
    if not token:
        return False
    global _SD35_LARGE_ACCESS_CACHE
    now = time.monotonic()
    cached_at, cached_value = _SD35_LARGE_ACCESS_CACHE
    if now - cached_at < 300.0:
        return cached_value
    try:
        import urllib.error
        import urllib.request

        request = urllib.request.Request(
            "https://huggingface.co/stabilityai/stable-diffusion-3.5-large/resolve/main/SD3.5L_example_workflow.json",
            headers={"Authorization": f"Bearer {token}"},
            method="HEAD",
        )
        with urllib.request.urlopen(request, timeout=2.5) as response:
            ok = 200 <= int(getattr(response, "status", 0) or 0) < 400
    except urllib.error.HTTPError as exc:
        ok = False if int(getattr(exc, "code", 0) or 0) in {401, 403, 404} else False
    except Exception:
        ok = False
    _SD35_LARGE_ACCESS_CACHE = (now, ok)
    return ok


def _sd35_large_config_cached() -> bool:
    try:
        from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
    except Exception:
        HUGGINGFACE_HUB_CACHE = os.environ.get("HUGGINGFACE_HUB_CACHE") or ""

    cache_roots = [
        Path(HUGGINGFACE_HUB_CACHE) if HUGGINGFACE_HUB_CACHE else None,
        Path(os.environ["HF_HOME"]) / "hub" if os.environ.get("HF_HOME") else None,
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    for root in cache_roots:
        if root is None:
            continue
        repo_dir = root / "models--stabilityai--stable-diffusion-3.5-large"
        if not repo_dir.is_dir():
            continue
        if any((repo_dir / "snapshots").glob("*/SD3.5L_example_workflow.json")):
            return True
    return False


def _selectable_checkpoint_payloads(ctx: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selectable: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for item in _safe_list(ctx.generation.list_checkpoints):
        payload = _checkpoint_payload(ctx, item)
        if _is_hidden_v1_checkpoint_architecture(str(payload.get("architecture") or "")):
            continue
        block = _blocked_checkpoint_detail(item) or _runtime_checkpoint_block(ctx, item)
        if block is None and str(payload.get("architecture") or "") == "sdxl_refiner":
            block = {
                "status": "blocked-cleanly",
                "reason": "This is the SDXL refiner, not a base checkpoint — it cannot generate on its own.",
                "suggestedAction": "Enable it as the refiner in generation settings instead of selecting it as the model.",
            }
        if block is None and payload.get("engineId") == "unknown":
            block = {
                "status": "blocked-cleanly",
                "reason": "This file is not a supported image/video checkpoint architecture.",
                "suggestedAction": "If this is a base model AIWF should support, report the filename; auxiliary weights belong in their tool-specific folders.",
            }
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
    if normalized == "flux_fill":
        return "flux_fill"
    if normalized in {"flux_kontext", "flux-kontext"}:
        return "flux"
    if normalized in {"flux2_klein", "flux2", "flux.2"}:
        return "flux2"
    if normalized in {"sana_video", "sana-video", "sanavideo"}:
        return "sana_video"
    if normalized in {"wan", "wan22", "wan2.2", "wan_video"}:
        return "wan"
    if normalized == "sana":
        return "sana"
    if normalized in {"qwen_image", "qwen_image_nunchaku"}:
        return "qwen"
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
    order = ["flux", "flux2", "sana_video", "wan", "sd15", "sdxl", "sd35", "zimage", "qwen", "sana", "unknown"]
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


def _artifact_payload(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        path = item.get("path", "")
        infotext = item.get("infotext", "")
        receipt_path = item.get("receipt_path") or item.get("receiptPath")
        metadata = item.get("metadata")
    else:
        path = getattr(item, "path", "")
        infotext = getattr(item, "infotext", "")
        receipt_path = getattr(item, "receipt_path", None)
        metadata = getattr(item, "metadata", None)
    payload: dict[str, Any] = {"path": str(path), "infotext": str(infotext or "")}
    if receipt_path:
        payload["receiptPath"] = str(receipt_path)
    if isinstance(metadata, dict) and metadata:
        payload["metadata"] = metadata
    return payload


def _clean_optional_text(value: Any) -> str:
    return str(value or "").strip()


def _optional_int(value: Any, *, minimum: int = 1) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= minimum else None


def _optional_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _settings_from_infotext(infotext: str) -> dict[str, Any]:
    text = _clean_optional_text(infotext)
    if not text:
        return {}
    try:
        params = parse_infotext(text)
    except Exception:
        logger.debug("Could not parse output infotext for Pro dock.", exc_info=True)
        return {}

    settings: dict[str, Any] = {}
    prompt = _clean_optional_text(params.get("Prompt"))
    if prompt:
        settings["prompt"] = prompt
    negative_prompt = _clean_optional_text(params.get("Negative prompt"))
    if negative_prompt:
        settings["negativePrompt"] = negative_prompt
    steps = _optional_int(params.get("Steps"))
    if steps is not None:
        settings["steps"] = steps
    cfg_scale = _optional_float(params.get("CFG scale"))
    if cfg_scale is not None:
        settings["cfgScale"] = cfg_scale
    seed = _optional_int(params.get("Seed"), minimum=0)
    if seed is not None:
        settings["seed"] = seed
    sampler = _clean_optional_text(params.get("Sampler"))
    if sampler:
        settings["sampler"] = sampler
    scheduler = _clean_optional_text(params.get("Schedule type"))
    if scheduler:
        settings["scheduler"] = scheduler
    model_name = _clean_optional_text(params.get("Model"))
    if model_name:
        settings["modelName"] = model_name
    width = _optional_int(params.get("Size-1") or params.get("Hires resize-1"))
    height = _optional_int(params.get("Size-2") or params.get("Hires resize-2"))
    if width is not None:
        settings["width"] = width
    if height is not None:
        settings["height"] = height
    return settings


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _settings_from_generation_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    pro_settings = metadata.get("pro_settings")
    if isinstance(pro_settings, dict):
        return dict(pro_settings)
    settings = metadata.get("settings")
    if not isinstance(settings, dict):
        return {}
    mapped: dict[str, Any] = {}
    key_map = {
        "prompt": "prompt",
        "negative_prompt": "negativePrompt",
        "checkpoint_id": "modelId",
        "width": "width",
        "height": "height",
        "steps": "steps",
        "cfg_scale": "cfgScale",
        "sampler": "sampler",
        "scheduler": "scheduler",
        "seed": "seed",
        "clip_skip": "clipSkip",
        "batch_size": "batchSize",
        "batch_count": "batchCount",
        "enable_hr": "enableHires",
        "hr_scale": "hiresScale",
        "hr_steps": "hiresSteps",
        "hr_denoising_strength": "hiresDenoise",
        "hr_upscaler": "hiresUpscaler",
        "vae_id": "vaeId",
        "denoising_strength": "denoisingStrength",
        "mask_blur": "maskBlur",
        "inpaint_only_masked": "inpaintOnlyMasked",
        "inpaint_masked_padding": "inpaintMaskedPadding",
        "inpaint_mask_content": "inpaintMaskContent",
        "save_images": "saveImages",
    }
    for source, target in key_map.items():
        if source in settings and settings[source] is not None:
            mapped[target] = settings[source]
    mode = str(settings.get("mode") or "").lower()
    if mode in {"inpaint", "img2img"}:
        mapped["mode"] = "inpaint" if mode == "inpaint" else "image"
    elif mode:
        mapped["mode"] = "image"
    return mapped


def _image_text_metadata_fields(image_text: dict[str, Any]) -> dict[str, Any]:
    aiwf_payload = _json_object(image_text.get("aiwf"))
    generation = _json_object(image_text.get("aiwf_generation"))
    if not generation and isinstance(aiwf_payload.get("generation"), dict):
        generation = dict(aiwf_payload["generation"])
    settings = _json_object(image_text.get("aiwf_generation_settings")) or _settings_from_generation_metadata(generation)
    receipt = _json_object(image_text.get("aiwf_generation_receipt"))
    if not receipt and isinstance(generation.get("receipt"), dict):
        receipt = dict(generation["receipt"])
    infotext = _clean_optional_text(image_text.get("parameters"))
    if not settings:
        settings = _settings_from_infotext(infotext)
    fields: dict[str, Any] = {}
    if generation:
        fields["metadata"] = generation
        fields["metadataSchema"] = str(generation.get("metadata_schema") or "aiwf.generation.v1")
        model = generation.get("model")
        if isinstance(model, dict):
            fields["modelName"] = str(model.get("title") or model.get("id") or model.get("filename") or "")
    if settings:
        fields["generationSettings"] = settings
        fields.update({key: value for key, value in settings.items() if key in {
            "prompt",
            "negativePrompt",
            "width",
            "height",
            "steps",
            "cfgScale",
            "sampler",
            "scheduler",
            "seed",
            "clipSkip",
            "modelName",
        }})
    if receipt:
        fields["generationReceipt"] = receipt
        elapsed = _optional_float(receipt.get("elapsed_seconds"))
        if elapsed is not None:
            fields["durationSeconds"] = round(elapsed, 2)
        steps_per_second = _optional_float(receipt.get("steps_per_second"))
        if steps_per_second is not None:
            fields["speed"] = f"{steps_per_second:.2f} steps/s"
    return fields


def _read_output_generation_metadata(path: Path) -> dict[str, Any]:
    try:
        with Image.open(path) as image:
            text = dict(getattr(image, "text", None) or {})
            info = getattr(image, "info", None) or {}
            for key in ("parameters", "aiwf", "aiwf_generation", "aiwf_generation_settings", "aiwf_generation_receipt"):
                if key not in text and key in info:
                    text[key] = info[key]
    except Exception:
        return {}
    return _image_text_metadata_fields(text)


def _read_output_infotext(path: Path, fallback: Any = "") -> str:
    text = _clean_optional_text(fallback)
    if text:
        return text[:_RECENT_INFOTEXT_MAX_CHARS]
    try:
        with Image.open(path) as image:
            image_text = getattr(image, "text", None) or {}
            text = _clean_optional_text(image_text.get("parameters"))
            if not text:
                image_info = getattr(image, "info", None) or {}
                text = _clean_optional_text(image_info.get("parameters"))
    except Exception:
        text = ""
    if text:
        return text[:_RECENT_INFOTEXT_MAX_CHARS]

    sidecar = path.with_suffix(".txt")
    try:
        if sidecar.is_file():
            return sidecar.read_text(encoding="utf-8", errors="replace")[:_RECENT_INFOTEXT_MAX_CHARS].strip()
    except OSError:
        return ""
    return ""


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


def _job_status(job: JobRecord | None, *, include_preview: bool = True) -> dict[str, Any]:
    if job is None:
        return {"state": "idle", "progress": 0, "message": "", "previewUrl": ""}
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
        "previewUrl": _job_preview_data_url(job) if (progress and include_preview) else "",
    }


def _job_preview_data_url(job: JobRecord) -> str:
    progress = getattr(job, "progress", None)
    image = getattr(progress, "current_image", None)
    if image is None:
        return ""
    key = (
        str(getattr(job, "id", "")),
        int(getattr(progress, "step", 0) or 0),
        int(getattr(progress, "total_steps", 0) or 0),
    )
    with _JOB_PREVIEW_CACHE_LOCK:
        cached = _JOB_PREVIEW_CACHE.get(key)
    if cached is not None:
        return cached
    data_url = _image_to_data_url(image, max_side=384, max_bytes=768 * 1024) or ""
    with _JOB_PREVIEW_CACHE_LOCK:
        _JOB_PREVIEW_CACHE[key] = data_url
        while len(_JOB_PREVIEW_CACHE) > _JOB_PREVIEW_CACHE_LIMIT:
            _JOB_PREVIEW_CACHE.pop(next(iter(_JOB_PREVIEW_CACHE)))
    return data_url


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
            popen_kwargs: dict[str, Any] = {"cwd": str(root)}
            if os.name == "nt" and "--terminal" not in forwarded_args:
                popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            subprocess.Popen(command, **popen_kwargs)
        finally:
            os._exit(_PRO_RESTART_EXIT_CODE)

    threading.Thread(target=_worker, name="aiwf-pro-restart", daemon=True).start()


def _open_support_terminal(ctx: Any) -> dict[str, str]:
    raise HTTPException(
        status_code=410,
        detail="Visible support terminals are disabled. Use the in-app Monitor and Logs views instead.",
    )


def _unload_generation_model(ctx: Any) -> dict[str, Any]:
    generation = getattr(ctx, "generation", None)
    backend = getattr(generation, "backend", None)
    unload = getattr(backend, "unload", None)
    if not callable(unload):
        raise HTTPException(status_code=501, detail="This backend does not expose a model unload action.")
    if _image_generation_running(ctx) or _image_generation_pending(ctx) or _pro_video_job_running(ctx):
        raise HTTPException(status_code=409, detail="A job is running. Stop it or wait before unloading models.")
    loaded_before = _runtime_loaded_model(ctx)
    try:
        unload()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not unload the current model: {exc}") from exc
    return {
        "status": "unloaded",
        "unloadedModel": loaded_before,
        "runtime": _runtime_summary(ctx),
    }


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


def _wan_service(ctx: Any):
    service = getattr(ctx, "wan", None)
    if service is not None:
        return service
    key = id(ctx)
    service = _WAN_SERVICES.get(key)
    if service is None:
        from aiwf.services.wan import WanService

        service = WanService(
            getattr(ctx, "flags", None),
            getattr(ctx, "settings", None),
            unload_image_models=getattr(getattr(getattr(ctx, "generation", None), "backend", None), "unload", None),
            supervisor=getattr(ctx, "supervisor", None),
            failure_archive=getattr(ctx, "failure_archive", None),
            genlog=getattr(ctx, "genlog", None),
        )
        _WAN_SERVICES[key] = service
    return service


def _wan_model_payloads(ctx: Any) -> list[dict[str, Any]]:
    try:
        service = _wan_service(ctx)
        labeled = service.list_local_models_labeled()
    except Exception:
        return []
    payloads: list[dict[str, Any]] = []
    for display, identifier in labeled:
        payloads.append(
            {
                "id": identifier,
                "title": display,
                "filename": identifier.rsplit("/", 1)[-1],
                "hash": None,
                "kind": "video",
                "architecture": "wan",
                "engineId": "wan",
                "engineLabel": _ENGINE_LABELS.get("wan", "Wan Video"),
                "backend": "Diffusers",
                "status": "Ready",
            }
        )
    return payloads


def _wan_model_ids(ctx: Any) -> set[str]:
    return {str(item.get("id") or "") for item in _wan_model_payloads(ctx)}


def _video_model_ids(ctx: Any) -> set[str]:
    ids = _wan_model_ids(ctx)
    try:
        sana_model = _sana_video_model_payload(ctx)
    except Exception:
        sana_model = {}
    sana_id = str(sana_model.get("id") or "")
    if sana_id:
        ids.add(sana_id)
    return ids


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


_RECENT_IMAGES_CACHE: dict[int, tuple[tuple[Any, ...], list[dict[str, Any]]]] = {}
_RECENT_IMAGES_CACHE_LOCK = threading.Lock()


def _recent_output_images(ctx: Any, *, limit: int = _RECENT_IMAGE_LIMIT) -> list[dict[str, Any]]:
    """Recent outputs with an identity cache.

    Encoding up to 8 images to base64 on every poll is the single most
    expensive part of the workspace refresh; the job list only changes when a
    generation finishes, so the encoded payload is reused until it does.
    """
    signature: tuple[Any, ...] = tuple(
        (str(getattr(job, "id", "")), str(getattr(getattr(job, "state", None), "value", "")))
        for job in _safe_recent_jobs(ctx, limit * 2)
    ) + (int(limit),)
    cache_key = id(ctx)
    with _RECENT_IMAGES_CACHE_LOCK:
        cached = _RECENT_IMAGES_CACHE.get(cache_key)
        if cached is not None and cached[0] == signature:
            return cached[1]
    images = _recent_output_images_uncached(ctx, limit=limit)
    with _RECENT_IMAGES_CACHE_LOCK:
        _RECENT_IMAGES_CACHE[cache_key] = (signature, images)
    return images


def _recent_output_images_uncached(ctx: Any, *, limit: int = _RECENT_IMAGE_LIMIT) -> list[dict[str, Any]]:
    root = _safe_output_root(ctx)
    seen_paths: set[str] = set()
    images: list[dict[str, Any]] = []

    def add_image(payload: dict[str, Any]) -> None:
        if len(images) < limit:
            images.append(payload)

    for job in _safe_recent_jobs(ctx, limit * 2):
        request = getattr(job, "request", None)
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
                infotext = (
                    (getattr(result, "infotexts", []) or [""])[index]
                    if index < len(getattr(result, "infotexts", []) or [])
                    else artifact.get("infotext", "")
                )
                elapsed_seconds = _generation_elapsed_seconds(result)
                metadata_fields = _image_text_metadata_fields(
                    {
                        "parameters": infotext,
                        "aiwf_generation": json.dumps(artifact.get("metadata", {}), sort_keys=True)
                        if isinstance(artifact.get("metadata"), dict)
                        else "",
                    }
                )
                seed = (getattr(result, "seeds", []) or [None])[index] if index < len(getattr(result, "seeds", []) or []) else None
                request_settings = _recent_generation_settings_payload(request, image, seed, str(getattr(getattr(result, "mode", None), "value", getattr(result, "mode", "txt2img"))))
                metadata_settings = metadata_fields.get("generationSettings")
                if isinstance(metadata_settings, dict):
                    request_settings = {**metadata_settings, **request_settings}
                add_image(
                    {
                        **metadata_fields,
                        "source": "memory",
                        "dataUrl": data_url,
                        "path": artifact_path,
                        "prompt": str(getattr(request, "prompt", "") or ""),
                        "negativePrompt": str(getattr(request, "negative_prompt", "") or ""),
                        "steps": int(getattr(request, "steps", 0) or 0) or None,
                        "cfgScale": (
                            float(getattr(request, "cfg_scale"))
                            if getattr(request, "cfg_scale", None) is not None
                            else None
                        ),
                        "sampler": str(getattr(request, "sampler", "") or ""),
                        "scheduler": str(getattr(request, "scheduler", "") or ""),
                        "durationSeconds": round(elapsed_seconds, 2) if elapsed_seconds > 0 else None,
                        "speed": _generation_speed_label(request, result),
                        "seed": seed,
                        "modelName": str(getattr(request, "checkpoint_id", "") or ""),
                        "infotext": infotext,
                        "receiptPath": artifact.get("receiptPath", ""),
                        "generationSettings": request_settings,
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
                infotext = _read_output_infotext(path, artifact_data["infotext"])
                add_image(
                    {
                        "source": "artifact",
                        "dataUrl": data_url,
                        "path": str(path),
                        "infotext": infotext,
                        "receiptPath": artifact_data.get("receiptPath", ""),
                        **_read_output_generation_metadata(path),
                        **_settings_from_infotext(infotext),
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
        infotext = _read_output_infotext(path)
        add_image(
            {
                "source": "disk",
                "dataUrl": data_url,
                "path": str(path),
                "infotext": infotext,
                **_read_output_generation_metadata(path),
                **_settings_from_infotext(infotext),
            }
        )
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


def _readiness_snapshot_paths(ctx: Any) -> list[Path]:
    root = _sana_log_root(ctx)
    if root is None:
        return []
    paths = [root / filename for filename in _READINESS_SNAPSHOT_FILENAMES]
    paths = [path for path in paths if path.is_file()]

    def modified(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    return sorted(paths, key=modified, reverse=True)


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


def _pro_startup_payload(ctx: Any) -> dict[str, Any]:
    now = time.time()
    started_at = float(getattr(ctx, "_pro_startup_started_at", 0.0) or 0.0)
    if started_at <= 0:
        started_at = now
        setattr(ctx, "_pro_startup_started_at", started_at)
    server_ready_at = float(getattr(ctx, "_pro_server_ready_at", 0.0) or 0.0)
    if server_ready_at <= 0:
        server_ready_at = now
        setattr(ctx, "_pro_server_ready_at", server_ready_at)
    window_ready_at = float(getattr(ctx, "_pro_window_ready_at", 0.0) or 0.0)
    window_ready = window_ready_at > 0
    return {
        "status": "window-ready" if window_ready else "server-ready",
        "serverReady": True,
        "windowReady": window_ready,
        "startedAt": datetime.fromtimestamp(started_at, timezone.utc).isoformat(),
        "serverReadyAt": datetime.fromtimestamp(server_ready_at, timezone.utc).isoformat(),
        "windowReadyAt": datetime.fromtimestamp(window_ready_at, timezone.utc).isoformat() if window_ready else "",
        "minSplashMs": _STARTUP_SPLASH_MIN_MS,
        "readyHoldMs": _STARTUP_SPLASH_READY_HOLD_MS,
    }


def _checkpoint_error_context(ctx: Any, checkpoint_id: str | None) -> dict[str, Any]:
    checkpoint = _resolve_checkpoint_for_generation_guard(ctx, checkpoint_id)
    if checkpoint is None:
        return {"requestedId": checkpoint_id or ""}
    data = _dump_model(checkpoint)
    path_value = str(data.get("path") or "")
    payload: dict[str, Any] = {
        "id": str(data.get("id") or checkpoint_id or ""),
        "title": str(data.get("title") or ""),
        "filename": str(data.get("filename") or ""),
        "path": path_value,
        "architecture": str(data.get("architecture") or ""),
        "kind": str(data.get("kind") or ""),
        "sizeBytes": int(data.get("size_bytes") or data.get("sizeBytes") or 0),
    }
    if path_value:
        path = Path(path_value)
        if path.is_file() and path.suffix.lower() in {".safetensors", ".gguf"}:
            try:
                from aiwf.infrastructure.model_header import read_model_info

                info = read_model_info(path)
                payload["header"] = {
                    "displayName": info.display_name,
                    "arch": info.arch,
                    "role": info.role,
                    "precision": info.precision,
                    "size": info.size_label(),
                    "tensorCount": info.tensor_count,
                    "metadata": {
                        str(key): str(value)
                        for key, value in list((info.raw_meta or {}).items())[:24]
                        if isinstance(value, (str, int, float, bool))
                    },
                }
            except Exception as exc:
                payload["header"] = {"error": str(exc)}
        elif path.is_dir():
            model_index = path / "model_index.json"
            payload["folder"] = {
                "modelIndex": str(model_index),
                "modelIndexExists": model_index.is_file(),
            }
            if model_index.is_file():
                try:
                    model_payload = json.loads(model_index.read_text(encoding="utf-8"))
                    payload["folder"]["className"] = str(model_payload.get("_class_name") or "")
                except Exception as exc:
                    payload["folder"]["error"] = str(exc)
    return payload


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
        width = width or _optional_int(item.get("width")) or 0
        height = height or _optional_int(item.get("height")) or 0
        outputs.append(
            {
                "id": f"recent-{index}-{path or item.get('source', 'memory')}",
                "url": item.get("dataUrl"),
                "thumbnailUrl": item.get("dataUrl"),
                "path": path,
                "prompt": str(item.get("prompt") or item.get("infotext") or "Local output"),
                "negativePrompt": item.get("negativePrompt"),
                "infotext": str(item.get("infotext") or ""),
                "width": width,
                "height": height,
                "createdAt": created_at,
                "mode": "image",
                "seed": item.get("seed"),
                "steps": item.get("steps"),
                "cfgScale": item.get("cfgScale"),
                "sampler": item.get("sampler"),
                "scheduler": item.get("scheduler"),
                "modelName": item.get("modelName"),
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

    catalog_items = list(_safe_list(service.list_catalog))
    visible_category_keys = {str(getattr(item, "category", "") or "") for item in catalog_items}
    categories: list[dict[str, Any]] = []
    for label, key in _safe_list(service.category_choices):
        if key not in visible_category_keys:
            continue
        destination = ""
        try:
            destination = str(service.destination_dir(key))
        except Exception:
            destination = ""
        categories.append({"key": key, "label": label, "destination": destination})

    catalog: list[dict[str, Any]] = []
    installed_count = 0
    for item in catalog_items:
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
        requires_auth = _catalog_entry_requires_auth(item)
        can_download = _catalog_entry_can_download(item)
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
                "hfUrl": _catalog_entry_hf_url(item),
                "requiresAuth": requires_auth,
                "canDownload": can_download,
                "comingSoon": bool(getattr(item, "coming_soon", False)),
            }
        )

    return {
        "categories": categories,
        "bundles": QUICK_START_BUNDLES,
        "catalog": catalog,
        "civitaiLinks": list(CIVITAI_BROWSE_LINKS),
        "counts": {
            "categories": len(categories),
            "catalog": len(catalog),
            "installed": installed_count,
        },
    }


def _catalog_entry_hf_url(item: Any) -> str:
    """Human-visitable download page for a catalog entry (HF repo or direct URL)."""
    repo_id = str(getattr(item, "repo_id", "") or "").strip()
    if repo_id:
        filename = str(getattr(item, "filename", "") or "").strip()
        if filename and not getattr(item, "snapshot", False):
            return f"https://huggingface.co/{repo_id}/blob/main/{filename}"
        return f"https://huggingface.co/{repo_id}"
    url = str(getattr(item, "url", "") or "").strip()
    if url.startswith("https://huggingface.co/"):
        return url.replace("/resolve/", "/blob/", 1)
    return url


def _catalog_entry_requires_auth(item: Any) -> bool:
    text = " ".join(
        str(value or "")
        for value in (
            getattr(item, "key", ""),
            getattr(item, "title", ""),
            getattr(item, "repo_id", ""),
            getattr(item, "filename", ""),
            getattr(item, "url", ""),
            getattr(item, "notes", ""),
        )
    ).lower()
    auth_tokens = (
        "gated",
        "token",
        "hf_token",
        "huggingface_token",
        "accepted hf",
        "accepted hugging face",
        "accept the hugging face gate",
        "requires accepted",
        "needs accepted",
        "may require accepted",
        "requires access",
        "needs access",
    )
    return any(token in text for token in auth_tokens)


def _catalog_entry_can_download(item: Any) -> bool:
    if bool(getattr(item, "coming_soon", False)):
        return False
    source = str(getattr(item, "source", "") or "").strip().lower()
    if source not in {"huggingface", "direct"}:
        return False
    if _catalog_entry_requires_auth(item):
        return False
    return bool(str(getattr(item, "repo_id", "") or getattr(item, "url", "") or "").strip())


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
            "showProgressEveryNSteps": int(getattr(settings, "show_progress_every_n_steps", 5)),
            "livePreviewDecoder": getattr(settings, "live_preview_decoder", "vae"),
            "livePreviewTitleProgress": bool(getattr(settings, "live_preview_title_progress", True)),
            "hiddenTabs": list(getattr(settings, "hidden_tabs", []) or []),
        },
        "output": {
            "imageFormat": getattr(settings, "image_format", "png"),
            "imageQuality": int(getattr(settings, "image_quality", 95)),
            "embedMetadata": bool(getattr(settings, "embed_metadata", True)),
            "saveGrid": bool(getattr(settings, "save_grid", False)),
            "saveSidecarTxt": bool(getattr(settings, "save_sidecar_txt", False)),
            "filenamePattern": getattr(settings, "filename_pattern", "[datetime]"),
            "saveBeforeHires": bool(getattr(settings, "save_before_hires", False)),
            "saveInterrupted": bool(getattr(settings, "save_interrupted", False)),
            "metadataIncludeModelHash": bool(getattr(settings, "metadata_include_model_hash", True)),
            "metadataIncludeVaeHash": bool(getattr(settings, "metadata_include_vae_hash", True)),
            "metadataIncludeLoraHashes": bool(getattr(settings, "metadata_include_lora_hashes", True)),
            "metadataIncludeAppVersion": bool(getattr(settings, "metadata_include_app_version", True)),
            "metadataIncludeOptimizationProfile": bool(getattr(settings, "metadata_include_optimization_profile", True)),
            "optimizationProfileId": getattr(settings, "optimization_profile_id", "balanced_sdpa_fp16"),
        },
        "video": {
            "wanHigh": getattr(settings, "last_wan_high", ""),
            "wanLow": getattr(settings, "last_wan_low", ""),
            "wanVae": getattr(settings, "last_wan_vae", ""),
            "wanTextEncoder": getattr(settings, "last_wan_text_encoder", ""),
            "wanOffload": getattr(settings, "last_wan_offload", "balanced"),
            "wanSampler": getattr(settings, "last_wan_sampler", "unipc"),
            "wanFlowShift": float(getattr(settings, "last_wan_flow_shift", 5.0)),
            "wanRuntimeMode": getattr(settings, "last_wan_runtime_mode", "fast_5b"),
            "ltxDtype": getattr(settings, "ltx_dtype", "bf16"),
            "ltxCpuOffload": getattr(settings, "ltx_cpu_offload", "auto"),
            "wanGroupOffloadStream": bool(getattr(settings, "wan_group_offload_stream", True)),
            "wanGroupOffloadBlocks": int(getattr(settings, "wan_group_offload_blocks", 4)),
            "ggufCudaKernels": bool(getattr(settings, "gguf_cuda_kernels", False)),
            "wanSageAttention": getattr(settings, "wan_sage_attention", "auto"),
            "wanNativeDenoise": bool(getattr(settings, "wan_native_denoise", True)),
            "wanManualVaeDecode": bool(getattr(settings, "wan_manual_vae_decode", False)),
            "wanVaeChunkFrames": int(getattr(settings, "wan_vae_chunk_frames", 4)),
            "wanGroupOffloadRecordStream": bool(getattr(settings, "wan_group_offload_record_stream", True)),
            "wanGroupOffloadLowCpuMem": bool(getattr(settings, "wan_group_offload_low_cpu_mem", True)),
            "wanResidentMinVramGb": int(getattr(settings, "wan_resident_min_vram_gb", 20)),
        },
        "runtime": {
            "port": int(getattr(flags, "port", 7860)) if flags is not None else 7860,
            "listen": bool(getattr(flags, "listen", False)),
            "share": bool(getattr(flags, "share", False)),
            "autolaunch": bool(getattr(flags, "autolaunch", False)),
            "api": bool(getattr(flags, "api", False)),
            "gerror": bool(getattr(flags, "gerror", False)),
            "genlog": bool(getattr(flags, "genlog", False)),
            "backend": getattr(flags, "inference_backend", "unknown") if flags is not None else "unknown",
            "onnxProvider": getattr(flags, "onnx_provider", "auto") if flags is not None else "auto",
            "attention": getattr(flags, "attention_backend", "unknown") if flags is not None else "unknown",
            "xformers": bool(getattr(flags, "xformers", False)),
            "optSdpAttention": bool(getattr(flags, "opt_sdp_attention", False)),
            "optSplitAttention": bool(getattr(flags, "opt_split_attention", False)),
            "asyncOffload": bool(getattr(flags, "async_offload", True)),
            "pinnedMemory": bool(getattr(flags, "pinned_memory", True)),
            "cudaMalloc": bool(getattr(flags, "cuda_malloc", False)),
            "vramProfile": flags.effective_vram_profile() if flags is not None else "normal",
            "medvram": bool(getattr(flags, "medvram", False)),
            "lowvram": bool(getattr(flags, "lowvram", False)),
            "highvram": bool(getattr(flags, "highvram", False)),
            "noHalf": bool(getattr(flags, "no_half", False)),
            "fp8": bool(getattr(flags, "fp8", False)),
            "fluxFp8": bool(getattr(flags, "fluxfp8", False)),
            "directml": bool(getattr(flags, "directml", False)),
            "cpu": bool(getattr(flags, "cpu", False)),
            "cudaGraphs": bool(getattr(flags, "cuda_graphs", False)),
            "torchao": bool(getattr(flags, "torchao", False)),
            "fp8Quant": bool(getattr(flags, "fp8_quant", False)),
            "torchCompile": bool(getattr(flags, "torch_compile", False)),
            "channelsLast": bool(getattr(flags, "channels_last", False)),
            "nvenc": bool(getattr(flags, "nvenc", False)),
            "hevc": bool(getattr(flags, "hevc", False)),
            "blockPrivateDownloadUrls": bool(getattr(flags, "block_private_download_urls", True)),
            "apiCorsOrigins": getattr(flags, "api_cors_origins", "") if flags is not None else "",
            "apiRateLimitPerMinute": int(getattr(flags, "api_rate_limit_per_minute", 0)) if flags is not None else 0,
            "theme": getattr(flags, "theme", "dark") if flags is not None else "dark",
            "modelsDir": str(getattr(flags, "models_dir", "") or "") if flags is not None else "",
            "checkpointDir": str(getattr(flags, "ckpt_dir", "") or "") if flags is not None else "",
            "outputDir": str(getattr(flags, "output_dir", "") or "") if flags is not None else "",
            "extraModelDirs": "\n".join(str(path) for path in flags.resolved_extra_model_dirs()) if flags is not None else "",
            "extraCheckpointDirs": "\n".join(str(path) for path in flags.resolved_extra_ckpt_dirs()) if flags is not None else "",
        },
    }


def _bool_setting(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _int_setting(value: Any, *, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))


def _float_setting(value: Any, *, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


_MISSING_SETTING = object()


def _payload_value(payload: dict[str, Any], camel: str, snake: str | None = None) -> Any:
    if camel in payload:
        return payload[camel]
    if snake and snake in payload:
        return payload[snake]
    return _MISSING_SETTING


def _choice_setting(value: Any, *, allowed: set[str], default: str) -> str:
    normalized = str(value or default).strip().lower().replace("-", "_")
    return normalized if normalized in allowed else default


def _text_setting(value: Any) -> str:
    return str(value or "").strip()


def _canonical_wan_runtime_mode(value: Any, *, default: str = "fast_5b") -> str:
    normalized = str(value or default).strip().lower().replace("-", "_")
    aliases = {
        "high_low": "native_high_low",
        "high_low_fp8": "native_high_low_fp8_experimental",
        "fp8_high_low": "native_high_low_fp8_experimental",
    }
    normalized = aliases.get(normalized, normalized)
    allowed = {"fast_5b", "native_high_low", "native_high_low_fp8_experimental"}
    return normalized if normalized in allowed else default


def _save_launch_profile(ctx: Any, launch: LaunchSettings) -> None:
    save_launch = getattr(ctx, "save_launch_settings", None)
    if callable(save_launch):
        save_launch(launch)
        return
    launch_path = getattr(ctx, "launch_settings_path", None)
    if launch_path:
        save_launch_settings(Path(launch_path), launch)


def _copy_runtime_flags(target: Any, source: Any) -> None:
    for field_name in type(source).model_fields:
        setattr(target, field_name, getattr(source, field_name))


def _apply_settings_update(ctx: Any, payload: ProSettingsUpdatePayload) -> dict[str, Any]:
    settings = getattr(ctx, "settings", None)
    if settings is None:
        raise HTTPException(status_code=500, detail="Settings are not available in this runtime.")

    generation = payload.generation_defaults or {}
    ui = payload.ui or {}
    output = payload.output or {}
    video = payload.video or {}
    runtime = payload.runtime or {}
    try:
        model_id = generation.get("checkpointId") or generation.get("checkpoint_id") or generation.get("modelId") or generation.get("model_id")
        if model_id is not None:
            setattr(settings, "last_checkpoint_id", str(model_id or ""))
        if "negativePrompt" in generation or "negative_prompt" in generation:
            setattr(settings, "default_negative_prompt", str(generation.get("negativePrompt") or generation.get("negative_prompt") or ""))
        if "useDefaultNegative" in generation or "use_default_negative" in generation:
            setattr(settings, "use_default_negative", _bool_setting(generation.get("useDefaultNegative", generation.get("use_default_negative"))))
        if "sampler" in generation:
            setattr(settings, "default_sampler", str(generation["sampler"] or "euler_a"))
        if "scheduler" in generation:
            setattr(settings, "default_scheduler", str(generation["scheduler"] or "automatic"))
        if "steps" in generation:
            setattr(settings, "default_steps", _int_setting(generation["steps"], minimum=1, maximum=150))
        if "cfgScale" in generation or "cfg_scale" in generation:
            setattr(settings, "default_cfg_scale", _float_setting(generation.get("cfgScale", generation.get("cfg_scale")), minimum=0.0, maximum=30.0))
        if "width" in generation:
            setattr(settings, "default_width", _int_setting(generation["width"], minimum=64, maximum=2048))
        if "height" in generation:
            setattr(settings, "default_height", _int_setting(generation["height"], minimum=64, maximum=2048))
        if "clipSkip" in generation or "clip_skip" in generation:
            setattr(settings, "default_clip_skip", _int_setting(generation.get("clipSkip", generation.get("clip_skip")), minimum=1, maximum=12))
        if "saveImages" in generation or "save_images" in generation:
            setattr(settings, "save_images", _bool_setting(generation.get("saveImages", generation.get("save_images"))))

        if "galleryColumns" in ui or "gallery_columns" in ui:
            setattr(settings, "gallery_columns", _int_setting(ui.get("galleryColumns", ui.get("gallery_columns")), minimum=1, maximum=8))
        if "galleryHeight" in ui or "gallery_height" in ui:
            setattr(settings, "gallery_height", _int_setting(ui.get("galleryHeight", ui.get("gallery_height")), minimum=160, maximum=1200))
        if "livePreview" in ui or "live_preview" in ui:
            setattr(settings, "enable_live_preview", _bool_setting(ui.get("livePreview", ui.get("live_preview"))))
        if "showProgressEveryNSteps" in ui or "show_progress_every_n_steps" in ui:
            setattr(settings, "show_progress_every_n_steps", _int_setting(ui.get("showProgressEveryNSteps", ui.get("show_progress_every_n_steps")), minimum=1, maximum=20))
        if "livePreviewDecoder" in ui or "live_preview_decoder" in ui:
            decoder = str(ui.get("livePreviewDecoder", ui.get("live_preview_decoder")) or "vae")
            setattr(settings, "live_preview_decoder", decoder if decoder == "vae" else "vae")
        if "livePreviewTitleProgress" in ui or "live_preview_title_progress" in ui:
            setattr(settings, "live_preview_title_progress", _bool_setting(ui.get("livePreviewTitleProgress", ui.get("live_preview_title_progress"))))
        if "accentPreset" in ui or "accent_preset" in ui:
            setattr(settings, "accent_preset", str(ui.get("accentPreset", ui.get("accent_preset")) or "mint"))
        if "hiddenTabs" in ui or "hidden_tabs" in ui:
            raw_tabs = ui.get("hiddenTabs", ui.get("hidden_tabs")) or []
            if isinstance(raw_tabs, list):
                setattr(settings, "hidden_tabs", [str(item) for item in raw_tabs])

        image_format = _payload_value(output, "imageFormat", "image_format")
        if image_format is not _MISSING_SETTING:
            setattr(settings, "image_format", _choice_setting(image_format, allowed={"png", "jpg", "jpeg", "webp"}, default="png"))
        image_quality = _payload_value(output, "imageQuality", "image_quality")
        if image_quality is not _MISSING_SETTING:
            setattr(settings, "image_quality", _int_setting(image_quality, minimum=10, maximum=100))
        for camel, snake, attr in (
            ("embedMetadata", "embed_metadata", "embed_metadata"),
            ("saveGrid", "save_grid", "save_grid"),
            ("saveSidecarTxt", "save_sidecar_txt", "save_sidecar_txt"),
            ("saveBeforeHires", "save_before_hires", "save_before_hires"),
            ("saveInterrupted", "save_interrupted", "save_interrupted"),
            ("metadataIncludeModelHash", "metadata_include_model_hash", "metadata_include_model_hash"),
            ("metadataIncludeVaeHash", "metadata_include_vae_hash", "metadata_include_vae_hash"),
            ("metadataIncludeLoraHashes", "metadata_include_lora_hashes", "metadata_include_lora_hashes"),
            ("metadataIncludeAppVersion", "metadata_include_app_version", "metadata_include_app_version"),
            (
                "metadataIncludeOptimizationProfile",
                "metadata_include_optimization_profile",
                "metadata_include_optimization_profile",
            ),
        ):
            value = _payload_value(output, camel, snake)
            if value is not _MISSING_SETTING:
                setattr(settings, attr, _bool_setting(value))
        filename_pattern = _payload_value(output, "filenamePattern", "filename_pattern")
        if filename_pattern is not _MISSING_SETTING:
            setattr(settings, "filename_pattern", str(filename_pattern or "[datetime]"))
        optimization_profile = _payload_value(output, "optimizationProfileId", "optimization_profile_id")
        if optimization_profile is not _MISSING_SETTING:
            setattr(settings, "optimization_profile_id", _text_setting(optimization_profile) or "balanced_sdpa_fp16")

        for camel, snake, attr in (
            ("wanHigh", "wan_high", "last_wan_high"),
            ("wanLow", "wan_low", "last_wan_low"),
            ("wanVae", "wan_vae", "last_wan_vae"),
            ("wanTextEncoder", "wan_text_encoder", "last_wan_text_encoder"),
        ):
            value = _payload_value(video, camel, snake)
            if value is not _MISSING_SETTING:
                setattr(settings, attr, _text_setting(value))
        wan_offload = _payload_value(video, "wanOffload", "wan_offload")
        if wan_offload is not _MISSING_SETTING:
            setattr(
                settings,
                "last_wan_offload",
                _choice_setting(
                    wan_offload,
                    allowed={"sequential", "group", "streamed", "model", "balanced", "resident", "none"},
                    default="balanced",
                ),
            )
        wan_sampler = _payload_value(video, "wanSampler", "wan_sampler")
        if wan_sampler is not _MISSING_SETTING:
            setattr(settings, "last_wan_sampler", _choice_setting(wan_sampler, allowed={"unipc", "euler", "heun"}, default="unipc"))
        wan_flow_shift = _payload_value(video, "wanFlowShift", "wan_flow_shift")
        if wan_flow_shift is not _MISSING_SETTING:
            setattr(settings, "last_wan_flow_shift", _float_setting(wan_flow_shift, minimum=0.0, maximum=20.0))
        wan_runtime_mode = _payload_value(video, "wanRuntimeMode", "wan_runtime_mode")
        if wan_runtime_mode is not _MISSING_SETTING:
            setattr(settings, "last_wan_runtime_mode", _choice_setting(wan_runtime_mode, allowed={"fast_5b", "high_low", "native_high_low", "native_high_low_fp8_experimental"}, default="fast_5b"))
        ltx_dtype = _payload_value(video, "ltxDtype", "ltx_dtype")
        if ltx_dtype is not _MISSING_SETTING:
            setattr(settings, "ltx_dtype", _choice_setting(ltx_dtype, allowed={"bf16", "fp16"}, default="bf16"))
        ltx_cpu_offload = _payload_value(video, "ltxCpuOffload", "ltx_cpu_offload")
        if ltx_cpu_offload is not _MISSING_SETTING:
            setattr(settings, "ltx_cpu_offload", _choice_setting(ltx_cpu_offload, allowed={"auto", "model", "none"}, default="auto"))
        wan_stream = _payload_value(video, "wanGroupOffloadStream", "wan_group_offload_stream")
        if wan_stream is not _MISSING_SETTING:
            setattr(settings, "wan_group_offload_stream", _bool_setting(wan_stream))
        wan_blocks = _payload_value(video, "wanGroupOffloadBlocks", "wan_group_offload_blocks")
        if wan_blocks is not _MISSING_SETTING:
            setattr(settings, "wan_group_offload_blocks", _int_setting(wan_blocks, minimum=1, maximum=40))
        gguf_kernels = _payload_value(video, "ggufCudaKernels", "gguf_cuda_kernels")
        if gguf_kernels is not _MISSING_SETTING:
            setattr(settings, "gguf_cuda_kernels", _bool_setting(gguf_kernels))
        wan_sage = _payload_value(video, "wanSageAttention", "wan_sage_attention")
        if wan_sage is not _MISSING_SETTING:
            setattr(settings, "wan_sage_attention", _choice_setting(wan_sage, allowed={"auto", "force", "off"}, default="auto"))
        for camel, snake, attr in (
            ("wanNativeDenoise", "wan_native_denoise", "wan_native_denoise"),
            ("wanManualVaeDecode", "wan_manual_vae_decode", "wan_manual_vae_decode"),
            ("wanGroupOffloadRecordStream", "wan_group_offload_record_stream", "wan_group_offload_record_stream"),
            ("wanGroupOffloadLowCpuMem", "wan_group_offload_low_cpu_mem", "wan_group_offload_low_cpu_mem"),
        ):
            value = _payload_value(video, camel, snake)
            if value is not _MISSING_SETTING:
                setattr(settings, attr, _bool_setting(value))
        wan_vae_chunk = _payload_value(video, "wanVaeChunkFrames", "wan_vae_chunk_frames")
        if wan_vae_chunk is not _MISSING_SETTING:
            setattr(settings, "wan_vae_chunk_frames", _int_setting(wan_vae_chunk, minimum=1, maximum=16))
        wan_resident_min = _payload_value(video, "wanResidentMinVramGb", "wan_resident_min_vram_gb")
        if wan_resident_min is not _MISSING_SETTING:
            setattr(settings, "wan_resident_min_vram_gb", _int_setting(wan_resident_min, minimum=8, maximum=96))
        # Apply immediately so the next pipeline load honors the new values
        # without an app restart.
        apply_video_perf_env = getattr(settings, "apply_video_perf_env", None)
        if callable(apply_video_perf_env):
            apply_video_perf_env()

        flags = getattr(ctx, "flags", None)
        if runtime and flags is not None:
            launch_data = LaunchSettings.from_runtime_flags(flags).model_dump()
            port = _payload_value(runtime, "port")
            if port is not _MISSING_SETTING:
                launch_data["port"] = _int_setting(port, minimum=1024, maximum=65535)
            api_rate = _payload_value(runtime, "apiRateLimitPerMinute", "api_rate_limit_per_minute")
            if api_rate is not _MISSING_SETTING:
                launch_data["api_rate_limit_per_minute"] = _int_setting(api_rate, minimum=0, maximum=6000)
            for camel, snake, field in (
                ("listen", None, "listen"),
                ("share", None, "share"),
                ("autolaunch", None, "autolaunch"),
                ("api", None, "api"),
                ("gerror", None, "gerror"),
                ("genlog", None, "genlog"),
                ("xformers", None, "xformers"),
                ("optSdpAttention", "opt_sdp_attention", "opt_sdp_attention"),
                ("optSplitAttention", "opt_split_attention", "opt_split_attention"),
                ("asyncOffload", "async_offload", "async_offload"),
                ("pinnedMemory", "pinned_memory", "pinned_memory"),
                ("cudaMalloc", "cuda_malloc", "cuda_malloc"),
                ("medvram", None, "medvram"),
                ("lowvram", None, "lowvram"),
                ("highvram", None, "highvram"),
                ("noHalf", "no_half", "no_half"),
                ("fp8", None, "fp8"),
                ("fluxFp8", "fluxfp8", "fluxfp8"),
                ("directml", None, "directml"),
                ("cpu", None, "cpu"),
                ("cudaGraphs", "cuda_graphs", "cuda_graphs"),
                ("torchao", None, "torchao"),
                ("fp8Quant", "fp8_quant", "fp8_quant"),
                ("torchCompile", "torch_compile", "torch_compile"),
                ("channelsLast", "channels_last", "channels_last"),
                ("nvenc", None, "nvenc"),
                ("hevc", None, "hevc"),
                ("blockPrivateDownloadUrls", "block_private_download_urls", "block_private_download_urls"),
            ):
                value = _payload_value(runtime, camel, snake)
                if value is not _MISSING_SETTING:
                    launch_data[field] = _bool_setting(value)
            vram_profile = _payload_value(runtime, "vramProfile", "vram_profile")
            if vram_profile is not _MISSING_SETTING:
                normalized_profile = normalize_vram_profile(str(vram_profile))
                launch_data["vram_profile"] = normalized_profile
                launch_data["cpu"] = normalized_profile == "cpu"
                launch_data["lowvram"] = normalized_profile == "low"
                launch_data["medvram"] = normalized_profile == "mid"
                launch_data["highvram"] = normalized_profile == "high"
            for camel, snake, field in (
                ("backend", "inference_backend", "inference_backend"),
                ("onnxProvider", "onnx_provider", "onnx_provider"),
                ("attention", "attention_backend", "attention_backend"),
                ("theme", None, "theme"),
                ("apiCorsOrigins", "api_cors_origins", "api_cors_origins"),
                ("modelsDir", "models_dir", "models_dir"),
                ("checkpointDir", "ckpt_dir", "ckpt_dir"),
                ("outputDir", "output_dir", "output_dir"),
                ("extraModelDirs", "extra_model_dirs", "extra_model_dirs"),
                ("extraCheckpointDirs", "extra_ckpt_dirs", "extra_ckpt_dirs"),
            ):
                value = _payload_value(runtime, camel, snake)
                if value is not _MISSING_SETTING:
                    launch_data[field] = _text_setting(value)
            launch = LaunchSettings.model_validate(launch_data)
            _save_launch_profile(ctx, launch)
            _copy_runtime_flags(flags, launch.to_runtime_flags(flags))
    except (TypeError, ValueError, ValidationError) as exc:
        raise HTTPException(status_code=422, detail=f"Settings update is invalid: {exc}") from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not save launch settings: {exc}") from exc

    save_settings = getattr(ctx, "save_settings", None)
    if callable(save_settings):
        try:
            save_settings()
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Could not save settings: {exc}") from exc

    return _settings_payload(ctx)


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


def _readiness_record_from_mapping(value: Any) -> PipelineReadinessRecord | None:
    if not isinstance(value, dict):
        return None

    def text(key: str) -> str:
        item = value.get(key, "")
        return "" if item is None else str(item)

    record_id = text("id")
    if not record_id:
        return None
    status = text("status") or "metadata-only"
    if status not in READINESS_STATUSES:
        status = "metadata-only"
    metadata = value.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    return PipelineReadinessRecord(
        id=record_id,
        family=text("family") or "unknown",
        asset_type=text("asset_type") or text("assetType"),
        path=text("path"),
        status=status,
        route=text("route"),
        reason=text("reason"),
        storage=text("storage"),
        quantization=text("quantization"),
        required_vae=text("required_vae") or text("requiredVae"),
        required_text_encoder=text("required_text_encoder") or text("requiredTextEncoder"),
        tokenizer=text("tokenizer"),
        smoke_command=text("smoke_command") or text("smokeCommand"),
        receipt_path=text("receipt_path") or text("receiptPath"),
        suggested_action=text("suggested_action") or text("suggestedAction"),
        metadata={str(key): "" if item is None else str(item) for key, item in metadata.items()},
    )


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


def _readiness_payload_from_records(
    records: list[PipelineReadinessRecord],
    *,
    error: str = "",
    source: str = "live",
) -> dict[str, Any]:
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
        "error": error,
        "source": source,
    }


def _readiness_payload_from_snapshot(ctx: Any) -> dict[str, Any] | None:
    for path in _readiness_snapshot_paths(ctx):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        raw_records = data.get("records") if isinstance(data, dict) else None
        if not isinstance(raw_records, list):
            continue
        records = [
            record
            for record in (_readiness_record_from_mapping(row) for row in raw_records)
            if record is not None
        ]
        if not records:
            continue
        try:
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
        except OSError:
            modified = ""
        message = f"Using cached readiness ledger from {path.name}"
        if modified:
            message += f" ({modified})"
        message += "; live refresh runs in the background."
        payload = _readiness_payload_from_records(records, source=str(path))
        payload["sourceMessage"] = message
        return payload
    return None


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
            "source": "live",
        }

    return _readiness_payload_from_records(records)


def _refresh_readiness_cache_in_background(ctx: Any) -> None:
    if getattr(ctx, "_pro_capability_refresh_running", False):
        return
    setattr(ctx, "_pro_capability_refresh_running", True)

    def _worker() -> None:
        try:
            payload = _readiness_payload(ctx)
            setattr(ctx, "_pro_capability_cache", {"cached_at": time.monotonic(), "payload": payload})
            setattr(ctx, "_pro_capability_refresh_at", time.monotonic())
        except Exception:
            logger.exception("Pro capability readiness refresh failed")
        finally:
            setattr(ctx, "_pro_capability_refresh_running", False)

    threading.Thread(target=_worker, name="aiwf-pro-capability-refresh", daemon=True).start()


def _cached_readiness_payload(ctx: Any) -> dict[str, Any]:
    now = time.monotonic()
    cached = getattr(ctx, "_pro_capability_cache", None)
    if isinstance(cached, dict):
        cached_at = float(cached.get("cached_at") or 0.0)
        payload = cached.get("payload")
        if payload is not None and now - cached_at < _CAPABILITY_CACHE_TTL_SECONDS:
            return payload
    snapshot_payload = _readiness_payload_from_snapshot(ctx)
    if snapshot_payload is not None:
        setattr(ctx, "_pro_capability_cache", {"cached_at": now, "payload": snapshot_payload})
        last_refresh_at = float(getattr(ctx, "_pro_capability_refresh_at", 0.0) or 0.0)
        if not _image_generation_running(ctx) and now - last_refresh_at > _CAPABILITY_BACKGROUND_REFRESH_SECONDS:
            _refresh_readiness_cache_in_background(ctx)
        return snapshot_payload
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
            "summary": "React Pro has one ControlNet unit; Gradio remains the advanced multi-unit surface.",
            "details": [f"{controlnet_count} models", f"{controlnet_modules} preprocessors", "One Pro unit, multi-unit in Gradio"],
        },
        {
            "id": "segment",
            "label": "Segment / SAM",
            "group": "Image",
            "status": _capability_status(sam_count),
            "count": sam_count,
            "route": "modal:segmentation",
            "summary": "Pro exposes quick mask routing; Gradio remains the full SAM and DINO workspace.",
            "details": [f"{sam_count} SAM models", "Box, point, and text-prompt masks in Gradio.", "Pro inpaint has quick auto-mask controls."],
        },
        {
            "id": "enhance",
            "label": "Enhance",
            "group": "Image",
            "status": _capability_status(upscaler_count + restorer_count),
            "count": upscaler_count + restorer_count,
            "route": "tools",
            "summary": "Pro can run quick face restore, upscale, and image VSR; Gradio keeps full old-photo and batch workflows.",
            "details": [f"{upscaler_count} upscalers", f"{restorer_count} restorers", "Video VSR is available from Pro Video Lab."],
        },
        {
            "id": "reactor",
            "label": "ReActor",
            "group": "Image",
            "status": _capability_status(reactor_model_count),
            "count": reactor_model_count,
            "route": "modal:reactor",
            "summary": "Pro can swap onto the current preview; Gradio keeps the advanced image and video ReActor workflow.",
            "details": [f"{reactor_model_count} swapper models", f"{reactor_face_count} saved face models", "Saved face and video-stage options stay in Gradio Lab."],
        },
        {
            "id": "video",
            "label": "Sana / Wan / LTX video",
            "group": "Video",
            "status": "ready" if sana_ready or (wan_available and wan_count > 0) else "available",
            "count": wan_count + (1 if sana_ready else 0),
            "route": "create",
            "summary": "Sana, Wan, LTX, RIFE, and post stages are visible in Gradio; Pro exposes Sana/Wan generation.",
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
    notes = [
        "React Pro now shows Gradio parity and asset readiness.",
        f"{len(blocked_checkpoints)} blocked model assets are hidden from the normal Generate picker.",
        "Heavy tool execution stays in Gradio until each Pro request path is typed and smoke-tested.",
    ]
    source_message = str(readiness.get("sourceMessage") or "")
    if source_message:
        notes.append(source_message)
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
        "notes": notes,
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
    block = _blocked_checkpoint_detail(checkpoint) or _runtime_checkpoint_block(ctx, checkpoint)
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


def _checkpoint_engine_id(ctx: Any, checkpoint_id: str | None) -> str:
    if not checkpoint_id:
        return "unknown"
    if checkpoint_id in _video_model_ids(ctx):
        return "wan" if checkpoint_id in _wan_model_ids(ctx) else "sana_video"
    checkpoint = _resolve_checkpoint_for_generation_guard(ctx, checkpoint_id)
    data = _dump_model(checkpoint) if checkpoint is not None else {}
    return _engine_id_for_architecture(str(data.get("architecture", "unknown")))


def _assert_image_route_checkpoint(ctx: Any, checkpoint_id: str | None) -> None:
    if _checkpoint_engine_id(ctx, checkpoint_id) in {"sana_video", "wan"}:
        raise HTTPException(
            status_code=422,
            detail=_pro_error_detail(
                "Video models are only available from the Video tab.",
                checkpointId=str(checkpoint_id or ""),
            ),
        )


def _assert_video_route_checkpoint(ctx: Any, checkpoint_id: str | None) -> None:
    if not checkpoint_id:
        return
    if _checkpoint_engine_id(ctx, checkpoint_id) not in {"sana_video", "wan"}:
        raise HTTPException(
            status_code=422,
            detail=_pro_error_detail(
                "Selected model is an image model. Choose a Wan or Sana Video model for video generation.",
                checkpointId=str(checkpoint_id),
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


_NVML_HANDLE: Any = None
_NVML_FAILED = False


def _nvml_gpu_utilization() -> tuple[float, float] | None:
    """In-process NVML query — microseconds instead of a subprocess spawn."""
    global _NVML_HANDLE, _NVML_FAILED
    if _NVML_FAILED:
        return None
    try:
        import pynvml

        if _NVML_HANDLE is None:
            pynvml.nvmlInit()
            _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
        rates = pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE)
        return (float(rates.gpu), float(rates.memory))
    except Exception:
        _NVML_FAILED = True
        _NVML_HANDLE = None
        return None


def _nvidia_gpu_utilization() -> tuple[float, float] | None:
    global _NVIDIA_SMI_CACHE
    now = time.monotonic()
    cached_at, cached = _NVIDIA_SMI_CACHE
    if now - cached_at < 0.9:
        return cached

    nvml = _nvml_gpu_utilization()
    if nvml is not None:
        _NVIDIA_SMI_CACHE = (now, nvml)
        return nvml

    # Subprocess fallback only when NVML is unavailable. Spawning nvidia-smi
    # every second on Windows is expensive and steals time from generation,
    # so the fallback refreshes far less often.
    if cached is not None and now - cached_at < 5.0:
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
    ctx_key = id(ctx)
    now = time.monotonic()
    cached = _RUNTIME_RESOURCE_CACHE.get(ctx_key)
    if cached is not None:
        cached_at, cached_metrics = cached
        if now - cached_at < _RUNTIME_RESOURCE_CACHE_SECONDS:
            return [dict(metric) for metric in cached_metrics]

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

    _RUNTIME_RESOURCE_CACHE[ctx_key] = (now, [dict(metric) for metric in metrics])
    return metrics


def _runtime_precision(flags: Any) -> str:
    if flags is None:
        return "Unknown"
    if bool(getattr(flags, "no_half", False)):
        return "FP32"
    if bool(getattr(flags, "fp8", False)) or bool(getattr(flags, "fluxfp8", False)) or bool(getattr(flags, "fp8_quant", False)):
        return "FP8/FP16"
    return "FP16"


def _runtime_text_encoder_label(backend: Any) -> str:
    pipe = getattr(backend, "_txt2img", None) or getattr(backend, "_inpaint", None)
    text_encoder = getattr(pipe, "text_encoder", None)
    precision = str(getattr(text_encoder, "_aiwf_precision", "") or "").strip()
    if precision:
        return f"Runtime owned ({precision})"
    return "Runtime owned"


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
        "textEncoder": _runtime_text_encoder_label(backend),
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
        # Finished jobs never need the live-preview data URL; without this the
        # runtime stream re-sends the last (up to ~768 KB) preview forever.
        job_status = _job_status(recent_terminal_image_job, include_preview=False)
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
        "gerror": bool(getattr(flags, "gerror", False)),
        "localOnly": not bool(getattr(flags, "listen", False)),
    }


async def _runtime_sse_events(ctx: Any, request: Request):
    """Adaptive runtime stream.

    - While a job runs, tick near the 75 ms UI budget so progress and previews feel
      immediate, but only send each preview frame ONCE per (job, step) — the
      base64 preview is by far the heaviest part of the payload.
    - While idle, tick slowly and skip emits entirely when nothing changed,
      so an idle Pro tab costs almost nothing on either side.
    """
    last_preview_key: tuple[str, int] | None = None
    last_idle_payload: str | None = None
    while True:
        if await request.is_disconnected():
            break
        payload = _runtime_summary(ctx)
        running = str(payload.get("status") or "") == "running"
        job = payload.get("job") or {}
        preview_url = str(job.get("previewUrl") or "")
        if preview_url:
            preview_key = (str(job.get("id") or ""), int(job.get("step") or 0))
            if preview_key == last_preview_key:
                # Same decoded frame as the previous tick — drop the heavy
                # data URL; the client keeps showing the frame it already has.
                job = {**job, "previewUrl": ""}
                payload = {**payload, "job": job}
            else:
                last_preview_key = preview_key

        if running:
            last_idle_payload = None
            yield f"event: runtime\ndata: {json.dumps(payload, default=str)}\n\n"
            await asyncio.sleep(_RUNTIME_RUNNING_TICK_SECONDS)
            continue

        serialized = json.dumps(payload, default=str)
        if serialized != last_idle_payload:
            last_idle_payload = serialized
            yield f"event: runtime\ndata: {serialized}\n\n"
        else:
            # SSE comment keeps proxies/browsers from timing the stream out.
            yield ": keepalive\n\n"
        await asyncio.sleep(_RUNTIME_IDLE_TICK_SECONDS)


def _settings_defaults(ctx: Any) -> dict[str, Any]:
    settings = getattr(ctx, "settings", None)
    checkpoint_id = getattr(settings, "last_checkpoint_id", None)
    if not _is_checkpoint_id_selectable(ctx, checkpoint_id):
        selectable, _blocked = _selectable_checkpoint_payloads(ctx)
        checkpoint_id = selectable[0]["id"] if selectable else None
    return {
        "prompt": "",
        "negativePrompt": getattr(settings, "default_negative_prompt", "") or "",
        "useDefaultNegative": bool(getattr(settings, "use_default_negative", True)),
        "checkpointId": checkpoint_id,
        "sampler": getattr(settings, "default_sampler", "euler_a"),
        "scheduler": getattr(settings, "default_scheduler", "automatic"),
        "steps": int(getattr(settings, "default_steps", 20)),
        "cfgScale": float(getattr(settings, "default_cfg_scale", 7.0)),
        "width": int(getattr(settings, "default_width", 512)),
        "height": int(getattr(settings, "default_height", 512)),
        "seed": -1,
        "clipSkip": int(getattr(settings, "default_clip_skip", 1)),
        "batchSize": 1,
        "batchCount": 1,
        "saveImages": bool(getattr(settings, "save_images", True)),
        "wanRuntimeMode": _canonical_wan_runtime_mode(getattr(settings, "last_wan_runtime_mode", "fast_5b")),
        "highNoiseModelId": "",
        "lowNoiseModelId": "",
        "highNoiseSteps": 20,
        "lowNoiseSteps": 1,
        "boundaryRatio": 0.875,
        "highNoiseLoraId": "",
        "highNoiseLoraScale": 1.0,
        "lowNoiseLoraId": "",
        "lowNoiseLoraScale": 1.0,
        "vaeId": "",
        "textEncoderPath": getattr(settings, "last_wan_text_encoder", "") or "",
        "wanOffload": getattr(settings, "last_wan_offload", "balanced") or "balanced",
        "wanSigmaType": getattr(settings, "last_wan_sigma_type", "simple") or "simple",
        "wanSampler": getattr(settings, "last_wan_sampler", "unipc") or "unipc",
        "wanFlowShift": float(getattr(settings, "last_wan_flow_shift", 5.0) or 5.0),
    }


def _is_checkpoint_id_selectable(ctx: Any, checkpoint_id: str | None) -> bool:
    if not checkpoint_id:
        return False
    checkpoint = _resolve_checkpoint_for_generation_guard(ctx, str(checkpoint_id))
    return (
        checkpoint is not None
        and _blocked_checkpoint_detail(checkpoint) is None
        and _runtime_checkpoint_block(ctx, checkpoint) is None
    )


def _generation_mode_from_payload(payload: ProGeneratePayload) -> GenerationMode:
    normalized = (payload.mode or "image").strip().lower()
    if normalized in {"image", "txt2img"}:
        return GenerationMode.TXT2IMG
    if normalized == "inpaint":
        return GenerationMode.INPAINT
    raise HTTPException(status_code=422, detail="React Pro generation currently supports image/txt2img and inpaint modes.")


def _controlnet_units_from_payload(payload: ProGeneratePayload) -> list[ControlNetUnit]:
    units: list[ControlNetUnit] = []
    for item in payload.controlnet_units or []:
        if not item.enabled:
            continue
        units.append(
            ControlNetUnit(
                enabled=True,
                model=item.model,
                module=item.module or "none",
                weight=item.weight,
                image=item.image,
                mask=item.mask,
                resize_mode=item.resize_mode or "resize",
                processor_res=item.processor_res,
                threshold_a=item.threshold_a,
                threshold_b=item.threshold_b,
                guidance_start=item.guidance_start,
                guidance_end=item.guidance_end,
                control_mode=item.control_mode or "balanced",
            )
        )
    return units


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
            clip_skip=payload.clip_skip,
            batch_size=payload.batch_size,
            batch_count=payload.batch_count,
            enable_hr=payload.enable_hr,
            hr_scale=payload.hr_scale,
            hr_steps=payload.hr_steps,
            hr_denoising_strength=payload.hr_denoising_strength,
            hr_upscaler=payload.hr_upscaler,
            denoising_strength=payload.denoising_strength,
            mask_blur=payload.mask_blur,
            inpaint_only_masked=payload.inpaint_only_masked,
            inpaint_masked_padding=payload.inpaint_masked_padding,
            inpaint_mask_content=payload.inpaint_mask_content,
            controlnet_units=_controlnet_units_from_payload(payload),
            sdxl_refiner_enabled=False,
            pipeline_backend=_normal_pipeline_backend(payload.pipeline_backend),
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc


def _normal_pipeline_backend(value: str | None) -> str:
    normalized = str(value or "aiwf").strip().lower().replace("-", "_")
    if normalized in {"dual", "both", "sdcpp", "stable_diffusion.cpp", "stable_diffusion_cpp"}:
        return "sdcpp"
    return "aiwf"


def _assert_requested_pipeline_backend(ctx: Any, payload: ProGeneratePayload) -> None:
    requested = _normal_pipeline_backend(payload.pipeline_backend)
    if requested != "sdcpp":
        return
    active_backend = str(getattr(getattr(ctx, "flags", None), "inference_backend", "") or "").strip().lower()
    backend_class = getattr(getattr(getattr(ctx, "generation", None), "backend", None), "__class__", type("", (), {})).__name__.lower()
    if (
        active_backend in {"sdcpp", "stable-diffusion.cpp", "stable_diffusion_cpp", "dual", "both"}
        or "sdcpp" in backend_class
        or "dual" in backend_class
    ):
        return
    raise HTTPException(
        status_code=409,
        detail=(
            "stable-diffusion.cpp is selected, but the running Pro backend is not the C++ backend. "
            "Restart Pro with --inference-backend dual (or sdcpp) to use this toggle."
        ),
    )


def _generation_elapsed_seconds(result: Any) -> float:
    try:
        return max(0.0, float(getattr(result, "elapsed_seconds", 0.0) or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _generation_speed_label(request: Any, result: Any) -> str:
    elapsed = _generation_elapsed_seconds(result)
    if elapsed <= 0:
        return ""
    try:
        steps = max(0, int(getattr(request, "steps", 0) or 0))
    except (TypeError, ValueError):
        steps = 0
    parts: list[str] = []
    if steps > 0:
        parts.append(f"{steps / elapsed:.2f} steps/s")
    images = len(getattr(result, "images", []) or [])
    if images > 1:
        parts.append(f"{images / elapsed:.2f} img/s")
    return " | ".join(parts)


def _recent_generation_settings_payload(request: Any, image: Any, seed: Any, mode: str) -> dict[str, Any]:
    width, height = getattr(image, "size", (0, 0))
    return {
        "mode": "inpaint" if mode == "inpaint" else "image",
        "prompt": str(getattr(request, "prompt", "") or ""),
        "negativePrompt": str(getattr(request, "negative_prompt", "") or ""),
        "modelId": str(getattr(request, "checkpoint_id", "") or ""),
        "width": int(getattr(request, "width", 0) or width or 0),
        "height": int(getattr(request, "height", 0) or height or 0),
        "steps": int(getattr(request, "steps", 0) or 0),
        "cfgScale": float(getattr(request, "cfg_scale", 0.0) or 0.0),
        "sampler": str(getattr(request, "sampler", "") or ""),
        "scheduler": str(getattr(request, "scheduler", "") or ""),
        "seed": seed if seed is not None else getattr(request, "seed", -1),
        "clipSkip": int(getattr(request, "clip_skip", 1) or 1),
        "batchSize": int(getattr(request, "batch_size", 1) or 1),
        "batchCount": int(getattr(request, "batch_count", 1) or 1),
        "enableHires": bool(getattr(request, "enable_hr", False)),
        "hiresScale": float(getattr(request, "hr_scale", 1.0) or 1.0),
        "hiresSteps": int(getattr(request, "hr_steps", 1) or 1),
        "hiresDenoise": float(getattr(request, "hr_denoising_strength", 0.0) or 0.0),
        "hiresUpscaler": str(getattr(request, "hr_upscaler", "") or ""),
        "denoisingStrength": float(getattr(request, "denoising_strength", 0.75) or 0.75),
        "maskBlur": int(getattr(request, "mask_blur", 4) or 4),
        "inpaintOnlyMasked": bool(getattr(request, "inpaint_only_masked", False)),
        "inpaintMaskedPadding": int(getattr(request, "inpaint_masked_padding", 32) or 32),
        "inpaintMaskContent": str(getattr(request, "inpaint_mask_content", "original") or "original"),
        "saveImages": bool(getattr(request, "save_images", True)),
    }


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
    negative_prompt = str(getattr(request, "negative_prompt", "") or "")
    model_name = str(getattr(request, "checkpoint_id", "") or "")
    elapsed_seconds = _generation_elapsed_seconds(result)
    speed = _generation_speed_label(request, result)
    outputs: list[dict[str, Any]] = []
    for index, image in enumerate(images):
        data_url = _image_to_data_url(image)
        if not data_url:
            continue
        artifact = artifacts[index] if index < len(artifacts) else {}
        path = str(artifact.get("path") or "")
        metadata_fields = _image_text_metadata_fields(
            {
                "parameters": infotexts[index] if index < len(infotexts) and infotexts[index] else "",
                "aiwf_generation": json.dumps(artifact.get("metadata", {}), sort_keys=True)
                if isinstance(artifact.get("metadata"), dict)
                else "",
            }
        )
        created_at = datetime.now(timezone.utc).isoformat()
        if path:
            try:
                stat = Path(path).stat()
                created_at = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
            except OSError:
                pass
        width, height = getattr(image, "size", (0, 0))
        seed = seeds[index] if index < len(seeds) else None
        request_settings = _recent_generation_settings_payload(request, image, seed, mode)
        metadata_settings = metadata_fields.get("generationSettings")
        if isinstance(metadata_settings, dict):
            request_settings = {**metadata_settings, **request_settings}
        outputs.append(
            {
                **metadata_fields,
                "id": f"{getattr(job, 'id', 'job')}-{index}",
                "url": data_url,
                "thumbnailUrl": data_url,
                "path": path,
                "prompt": prompt,
                "negativePrompt": negative_prompt,
                "infotext": infotexts[index] if index < len(infotexts) and infotexts[index] else "",
                "width": width,
                "height": height,
                "createdAt": created_at,
                "mode": mode,
                "seed": seed,
                "steps": int(getattr(request, "steps", 0) or 0) or None,
                "cfgScale": (
                    float(getattr(request, "cfg_scale"))
                    if getattr(request, "cfg_scale", None) is not None
                    else None
                ),
                "clipSkip": int(getattr(request, "clip_skip", 1) or 1),
                "sampler": str(getattr(request, "sampler", "") or ""),
                "scheduler": str(getattr(request, "scheduler", "") or ""),
                "durationSeconds": round(elapsed_seconds, 2) if elapsed_seconds > 0 else None,
                "speed": speed,
                "modelName": model_name,
                "receiptPath": artifact.get("receiptPath", ""),
                "generationSettings": request_settings,
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
    request = getattr(job, "request", None)
    elapsed_seconds = _generation_elapsed_seconds(result)
    try:
        steps = max(0, int(getattr(request, "steps", 0) or 0))
    except (TypeError, ValueError):
        steps = 0
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
        "timings": {
            "elapsedSeconds": round(elapsed_seconds, 3),
            "stepsPerSecond": round(steps / elapsed_seconds, 3) if elapsed_seconds > 0 and steps > 0 else 0,
        },
        "message": f"Generated {len(recent_outputs)} image(s).",
    }


def _decode_pro_image_data_url(data_url: str | None, label: str) -> Image.Image | None:
    value = (data_url or "").strip()
    if not value:
        return None
    if "," not in value or ";base64" not in value.partition(",")[0].lower():
        raise HTTPException(status_code=422, detail=f"{label} must be a base64 image data URL.")
    header, encoded = value.split(",", 1)
    if not header.lower().startswith("data:image/"):
        raise HTTPException(status_code=422, detail=f"{label} must be PNG, JPEG, or WebP.")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"{label} data is not valid base64.") from exc
    if len(raw) > _PRO_SOURCE_IMAGE_MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"{label} is too large.")
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except OSError as exc:
        raise HTTPException(status_code=422, detail=f"{label} could not be opened.") from exc
    return image


def _decode_data_url_bytes(data_url: str | None, label: str) -> bytes:
    value = (data_url or "").strip()
    if not value:
        raise HTTPException(status_code=422, detail=f"{label} is empty.")
    if "," not in value or ";base64" not in value.partition(",")[0].lower():
        raise HTTPException(status_code=422, detail=f"{label} must be a base64 data URL.")
    header, encoded = value.split(",", 1)
    if not header.lower().startswith("data:image/"):
        raise HTTPException(status_code=422, detail=f"{label} must be an image.")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"{label} data is not valid base64.") from exc
    if len(raw) > _PRO_SOURCE_IMAGE_MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"{label} is too large.")
    return raw


def _import_generation_metadata_from_image(payload: ProMetadataImportPayload) -> dict[str, Any]:
    raw = _decode_data_url_bytes(payload.image_data_url, "Image metadata import")
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            text = dict(getattr(image, "text", None) or {})
            info = getattr(image, "info", None) or {}
            for key in ("parameters", "aiwf", "aiwf_generation", "aiwf_generation_settings", "aiwf_generation_receipt"):
                if key not in text and key in info:
                    text[key] = info[key]
            width, height = image.size
    except OSError as exc:
        raise HTTPException(status_code=422, detail="Image metadata import could not be opened.") from exc

    infotext = _clean_optional_text(text.get("parameters"))
    fields = _image_text_metadata_fields(text)
    settings = fields.get("generationSettings")
    if not isinstance(settings, dict):
        settings = _settings_from_infotext(infotext)
    metadata = fields.get("metadata") if isinstance(fields.get("metadata"), dict) else {}
    receipt = fields.get("generationReceipt") if isinstance(fields.get("generationReceipt"), dict) else {}
    return {
        "status": "ok" if settings else "empty",
        "filename": payload.filename,
        "width": width,
        "height": height,
        "infotext": infotext,
        "settings": settings or {},
        "metadata": metadata,
        "receipt": receipt,
        "message": "Generation settings found." if settings else "No generation metadata was found in this image.",
    }


def _assert_inpaint_checkpoint_supported(ctx: Any, payload: ProGeneratePayload) -> None:
    checkpoint = _resolve_checkpoint_for_generation_guard(ctx, _checkpoint_id_from_payload(ctx, payload))
    data = _dump_model(checkpoint) if checkpoint is not None else {}
    engine_id = _engine_id_for_architecture(str(data.get("architecture", "unknown")))
    if engine_id not in {"sd15", "sdxl", "flux_fill"}:
        raise HTTPException(
            status_code=422,
            detail="Inpainting is supported for SD 1.5, SDXL, and Flux Fill checkpoints.",
        )


def _run_pro_image_generation(
    ctx: Any,
    request: GenerationRequest,
    init_images: list[Image.Image] | None = None,
    mask_images: list[Image.Image] | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    submit_streaming = getattr(ctx.generation, "submit_streaming", None)
    if not callable(submit_streaming):
        return ctx.generation.submit(request, init_images=init_images, mask_images=mask_images), []

    progress: list[dict[str, Any]] = []
    finished_job = None
    started_at = time.perf_counter()
    for item in submit_streaming(request, init_images=init_images, mask_images=mask_images):
        kind = item[0] if item else ""
        if kind == "done":
            finished_job = item[1]
        elif kind == "progress":
            _, step, total, message, _preview = item
            total_int = max(1, int(total or 1))
            step_int = max(0, int(step or 0))
            progress.append(
                {
                    "stage": "image",
                    "progress": min(1.0, max(0.0, step_int / total_int)),
                    "message": str(message or ""),
                    "step": step_int,
                    "total": total_int,
                    "seconds": round(time.perf_counter() - started_at, 3),
                }
            )
    if finished_job is None:
        raise RuntimeError("Generation finished without returning a job record.")
    return finished_job, progress


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
        "prompt": payload.prompt,
        "negativePrompt": payload.negative_prompt,
        "infotext": str(getattr(result, "infotext", "") or ""),
        "width": int(getattr(result, "width", payload.width) or payload.width),
        "height": int(getattr(result, "height", payload.height) or payload.height),
        "createdAt": created_at,
        "mode": "video",
        "seed": payload.seed,
        "steps": payload.steps,
        "cfgScale": payload.cfg_scale,
        "clipSkip": payload.clip_skip,
        "sampler": payload.sampler,
        "scheduler": payload.scheduler,
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


def _wan_video_request_from_payload(ctx: Any, payload: ProGeneratePayload):
    from aiwf.core.domain.wan import (
        OFFLOAD_MODES,
        SAMPLER_TYPES,
        SIGMA_TYPES,
        WAN_RUNTIME_FAST_5B,
        WAN_RUNTIME_HIGH_LOW,
        WAN_RUNTIME_HIGH_LOW_FP8,
        WAN_RUNTIME_MODES,
        WanI2VRequest,
    )

    service = _wan_service(ctx)
    settings = getattr(ctx, "settings", None)
    runtime_mode = _canonical_wan_runtime_mode(
        payload.wan_runtime_mode or getattr(settings, "last_wan_runtime_mode", "") or WAN_RUNTIME_FAST_5B,
        default=WAN_RUNTIME_FAST_5B,
    )
    if runtime_mode not in WAN_RUNTIME_MODES:
        runtime_mode = WAN_RUNTIME_FAST_5B
    sampler = str(payload.wan_sampler or getattr(settings, "last_wan_sampler", "") or "unipc").strip().lower()
    if sampler not in SAMPLER_TYPES:
        sampler = "unipc"
    sigma_type = str(payload.wan_sigma_type or getattr(settings, "last_wan_sigma_type", "") or "simple").strip().lower()
    if sigma_type not in SIGMA_TYPES:
        sigma_type = "simple"
    try:
        flow_shift = float(payload.wan_flow_shift if payload.wan_flow_shift is not None else getattr(settings, "last_wan_flow_shift", 5.0) or 5.0)
    except (TypeError, ValueError):
        flow_shift = 5.0
    offload = str(payload.wan_offload or getattr(settings, "last_wan_offload", "balanced") or "balanced").strip().lower()
    if offload not in OFFLOAD_MODES:
        offload = "balanced"
    vae_id = str(payload.vae_id or "").strip() or None
    if not vae_id:
        try:
            vae_id = service.preferred_vae(runtime_mode)
        except Exception:
            vae_id = None
    text_encoder = str(payload.text_encoder_path or getattr(settings, "last_wan_text_encoder", "") or "").strip()
    model_id = str(payload.checkpoint_id or "")
    if runtime_mode in {WAN_RUNTIME_HIGH_LOW, WAN_RUNTIME_HIGH_LOW_FP8}:
        model_id = model_id or str(payload.high_noise_model_id or "")
    try:
        return WanI2VRequest(
            prompt=payload.prompt,
            negative_prompt=payload.negative_prompt,
            width=payload.width,
            height=payload.height,
            num_frames=min(max(int(payload.frames), 5), 257),
            fps=max(1, int(round(payload.fps))),
            steps=min(int(payload.steps), 100),
            high_noise_steps=min(max(int(payload.high_noise_steps), 1), 60),
            low_noise_steps=min(max(int(payload.low_noise_steps), 1), 60),
            guidance_scale=min(max(float(payload.cfg_scale), 1.0), 20.0),
            sampler=sampler,
            sigma_type=sigma_type,
            flow_shift=flow_shift,
            seed=payload.seed,
            runtime_mode=runtime_mode,
            model_id=model_id,
            offload=offload,
            boundary_ratio=payload.boundary_ratio,
            high_noise_model_id=payload.high_noise_model_id,
            low_noise_model_id=payload.low_noise_model_id,
            high_noise_lora_id=payload.high_noise_lora_id,
            high_noise_lora_scale=payload.high_noise_lora_scale,
            low_noise_lora_id=payload.low_noise_lora_id,
            low_noise_lora_scale=payload.low_noise_lora_scale,
            vae_id=vae_id,
            text_encoder_path=text_encoder,
            offload_text_encoder_after_encode=payload.offload_text_encoder_after_encode,
            use_sage_attention=payload.use_sage_attention,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc


def _wan_video_output_payload(ctx: Any, result: Any, payload: ProGeneratePayload) -> dict[str, Any]:
    output_path = str(getattr(result, "output_path", "") or "")
    path = Path(output_path)
    created_at = datetime.now(timezone.utc).isoformat()
    if path.is_file():
        try:
            created_at = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
        except OSError:
            pass
    return {
        "id": f"wan-video-{path.stem or 'output'}",
        "url": _output_asset_url(ctx, output_path),
        "thumbnailUrl": _output_asset_url(ctx, output_path),
        "path": output_path,
        "prompt": payload.prompt,
        "negativePrompt": payload.negative_prompt,
        "infotext": "",
        "width": int(getattr(result, "width", payload.width) or payload.width),
        "height": int(getattr(result, "height", payload.height) or payload.height),
        "createdAt": created_at,
        "mode": "video",
        "seed": payload.seed,
        "steps": payload.steps,
        "cfgScale": payload.cfg_scale,
        "clipSkip": payload.clip_skip,
        "sampler": payload.sampler,
        "scheduler": payload.scheduler,
        "modelName": payload.checkpoint_id or "Wan 2.2",
        "status": "completed",
        "source": "wan-video",
    }


def _vsr_service(ctx: Any):
    service = getattr(ctx, "vsr", None)
    if service is not None:
        return service
    key = id(ctx)
    service = _VSR_SERVICES.get(key)
    if service is None:
        from aiwf.services.vsr import VsrService

        service = VsrService(
            getattr(ctx, "flags", None),
            getattr(ctx, "settings", None),
            supervisor=getattr(ctx, "supervisor", None),
        )
        _VSR_SERVICES[key] = service
    return service


def _rife_service(ctx: Any):
    service = getattr(ctx, "rife", None)
    if service is not None:
        return service
    key = id(ctx)
    service = _RIFE_SERVICES.get(key)
    if service is None:
        from aiwf.services.rife import RifeService

        backend = getattr(getattr(ctx, "generation", None), "backend", None)
        service = RifeService(
            getattr(ctx, "flags", None),
            getattr(ctx, "settings", None),
            getattr(backend, "devices", None),
            supervisor=getattr(ctx, "supervisor", None),
        )
        _RIFE_SERVICES[key] = service
    return service


def _audio_service(ctx: Any):
    service = getattr(ctx, "audio", None)
    if service is not None:
        return service
    key = id(ctx)
    service = _AUDIO_SERVICES.get(key)
    if service is None:
        from aiwf.services.audio import AudioService

        backend = getattr(getattr(ctx, "generation", None), "backend", None)
        service = AudioService(
            getattr(ctx, "flags", None),
            getattr(ctx, "settings", None),
            devices=getattr(backend, "devices", None),
            supervisor=getattr(ctx, "supervisor", None),
        )
        _AUDIO_SERVICES[key] = service
    return service


class ProVideoLabRunPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    op: str
    video_path: str = Field(default="", validation_alias=AliasChoices("videoPath", "video_path"))
    # VSR / upscale
    scale: float = Field(default=2.0, ge=1.0, le=4.0)
    mode: int = Field(default=0, ge=0, le=19)
    effect: str = "SuperRes"
    strength: float = Field(default=0.4, ge=0.0, le=1.0)
    # RIFE
    multiplier: int = Field(default=2, ge=2, le=8)
    target_fps: float | None = Field(
        default=None, ge=1.0, le=240.0, validation_alias=AliasChoices("targetFps", "target_fps")
    )
    # Audio
    audio_prompt: str = Field(default="", validation_alias=AliasChoices("audioPrompt", "audio_prompt"))
    audio_model: str = Field(default="", validation_alias=AliasChoices("audioModel", "audio_model"))
    # Extend (Wan i2v continuation)
    prompt: str = ""
    negative_prompt: str = Field(default="", validation_alias=AliasChoices("negativePrompt", "negative_prompt"))
    frames: int = Field(default=81, ge=5, le=257)
    steps: int = Field(default=8, ge=1, le=100)
    cfg_scale: float = Field(default=5.0, ge=1.0, le=20.0, validation_alias=AliasChoices("cfgScale", "cfg_scale"))
    seed: int = -1
    checkpoint_id: str = Field(default="", validation_alias=AliasChoices("checkpointId", "checkpoint_id", "modelId"))


class ProAutoMaskPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    image_data_url: str = Field(validation_alias=AliasChoices("imageDataUrl", "image_data_url", "image"))
    prompt: str = ""
    box_threshold: float = Field(default=0.3, ge=0.05, le=0.95, validation_alias=AliasChoices("boxThreshold", "box_threshold"))
    dilation: int = Field(default=8, ge=0, le=128)
    mask_blur: int = Field(default=4, ge=0, le=64, validation_alias=AliasChoices("maskBlur", "mask_blur"))
    feather: int = Field(default=6, ge=0, le=64)


class ProFaceSwapPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    target_image_data_url: str = Field(
        validation_alias=AliasChoices("targetImageDataUrl", "target_image_data_url", "target")
    )
    source_image_data_url: str = Field(
        validation_alias=AliasChoices("sourceImageDataUrl", "source_image_data_url", "source")
    )


class ProExtensionTogglePayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    extension_id: str = Field(validation_alias=AliasChoices("id", "extensionId", "extension_id"))
    enabled: bool = True


class ProVsrImagePayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    image_data_url: str = Field(validation_alias=AliasChoices("imageDataUrl", "image_data_url", "image"))
    scale: float = Field(default=2.0, ge=1.0, le=4.0)
    mode: int = Field(default=0, ge=0, le=19)
    effect: str = "SuperRes"
    strength: float = Field(default=0.4, ge=0.0, le=1.0)


class ProEnhanceImagePayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    image_data_url: str = Field(validation_alias=AliasChoices("imageDataUrl", "image_data_url", "image"))
    restore_enabled: bool = Field(default=True, validation_alias=AliasChoices("restoreEnabled", "restore_enabled"))
    restore_model: str = Field(default="", validation_alias=AliasChoices("restoreModel", "restore_model"))
    restore_visibility: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("restoreVisibility", "restore_visibility"),
    )
    codeformer_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("codeformerWeight", "codeformer_weight"),
    )
    upscale_enabled: bool = Field(default=False, validation_alias=AliasChoices("upscaleEnabled", "upscale_enabled"))
    upscale_model: str = Field(default="", validation_alias=AliasChoices("upscaleModel", "upscale_model"))
    upscale_scale: float = Field(default=2.0, ge=1.0, le=8.0, validation_alias=AliasChoices("upscaleScale", "upscale_scale"))
    tile_size: int = Field(default=256, ge=0, le=2048, validation_alias=AliasChoices("tileSize", "tile_size"))
    tile_overlap: int = Field(default=32, ge=0, le=512, validation_alias=AliasChoices("tileOverlap", "tile_overlap"))
    restore_first: bool = Field(default=True, validation_alias=AliasChoices("restoreFirst", "restore_first"))


def _video_lab_upload_root(ctx: Any) -> Path:
    flags = getattr(ctx, "flags", None)
    root = flags.resolved_output_dir() / "video-lab" / "uploads"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _model_upload_root(ctx: Any) -> Path:
    flags = getattr(ctx, "flags", None)
    if flags is None:
        raise HTTPException(status_code=500, detail="Runtime flags are unavailable.")
    root = flags.resolved_models_dir() / SORT_INBOX_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_upload_filename(filename: str | None, fallback: str) -> str:
    raw = Path(filename or fallback).name
    stem = Path(raw).stem
    suffix = Path(raw).suffix.lower()
    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in stem)[:120]
    safe_stem = safe_stem.strip("._") or fallback
    return f"{safe_stem}{suffix}"


def _sort_action_payload(action: Any) -> dict[str, Any]:
    return {
        "filename": str(getattr(action, "filename", "") or ""),
        "source": str(getattr(action, "source", "") or ""),
        "family": str(getattr(action, "family", "") or ""),
        "architecture": str(getattr(action, "architecture", "") or ""),
        "destSubdir": str(getattr(action, "dest_subdir", "") or ""),
        "status": str(getattr(action, "status", "") or ""),
        "reason": str(getattr(action, "reason", "") or ""),
    }


def _refresh_model_inventory_after_sort(ctx: Any) -> dict[str, Any]:
    flags = getattr(ctx, "flags", None)
    if flags is None:
        raise HTTPException(status_code=500, detail="Runtime flags are unavailable.")
    records = scan_and_write_model_inventory(flags)
    setattr(ctx, "_pro_capability_cache", None)
    return {"inventoryCount": len(records)}


def _model_sort_response(ctx: Any, actions: list[Any], *, uploaded_path: Path | None = None) -> dict[str, Any]:
    payload_actions = [_sort_action_payload(action) for action in actions]
    moved = sum(1 for item in payload_actions if item["status"] == "moved")
    left = sum(1 for item in payload_actions if item["status"] in {"left", "conflict", "error"})
    payload: dict[str, Any] = {
        "status": "completed",
        "uploadedPath": str(uploaded_path) if uploaded_path is not None else "",
        "actions": payload_actions,
        "counts": {
            "total": len(payload_actions),
            "moved": moved,
            "left": left,
        },
    }
    payload["counts"].update(_refresh_model_inventory_after_sort(ctx))
    return payload


def _video_lab_resolve_source(ctx: Any, video_path: str) -> Path:
    """Only accept sources inside the outputs tree so the API can't read arbitrary files."""
    value = (video_path or "").strip()
    if not value:
        raise HTTPException(status_code=422, detail="Upload or pick a source video first.")
    root = _safe_output_root(ctx)
    if root is None:
        raise HTTPException(status_code=500, detail="Output directory is not available.")
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / value
    try:
        candidate = candidate.resolve()
    except OSError as exc:
        raise HTTPException(status_code=422, detail=f"Video path could not be resolved: {exc}") from exc
    if not _path_inside(candidate, root) or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Source video was not found under the outputs directory.")
    return candidate


def _video_lab_probe(path: Path) -> dict[str, Any]:
    from aiwf.infrastructure.video import VideoProcessor

    try:
        info = VideoProcessor().probe(path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read video: {exc}") from exc
    return {
        "path": str(path),
        "width": int(getattr(info, "width", 0) or 0),
        "height": int(getattr(info, "height", 0) or 0),
        "fps": float(getattr(info, "fps", 0.0) or 0.0),
        "frameCount": int(getattr(info, "frame_count", 0) or 0),
        "durationSeconds": float(getattr(info, "duration_seconds", 0.0) or 0.0),
    }


def _video_lab_output(ctx: Any, output_path: str | Path, message: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "status": "completed",
        "outputPath": str(output_path),
        "url": _output_asset_url(ctx, output_path),
        "message": message,
        "probe": _video_lab_probe(Path(output_path)),
    }
    if extra:
        payload.update(extra)
    return payload


def _concat_videos_dropping_first_frame(first: Path, second: Path, dest: Path) -> None:
    """Concatenate two clips, dropping the second clip's first frame.

    The continuation clip starts from the exact last frame of the original
    (it was the i2v conditioning image), so frame 0 is dropped to avoid a
    visible stutter at the seam. Re-encodes to normalize codec/resolution.
    """
    from aiwf.infrastructure.video.processing import _resolve_ffmpeg

    ffmpeg = _resolve_ffmpeg()
    if ffmpeg is None:
        raise HTTPException(status_code=500, detail="ffmpeg is required to stitch the extended video.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(first),
        "-i",
        str(second),
        "-filter_complex",
        "[1:v]select=gte(n\\,1),setpts=PTS-STARTPTS[b];[0:v][b]concat=n=2:v=1:a=0[v]",
        "-map",
        "[v]",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dest),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=3600)
    if completed.returncode != 0 or not dest.is_file() or dest.stat().st_size <= 0:
        detail = (completed.stderr or completed.stdout or "").strip()[-800:]
        raise HTTPException(status_code=500, detail=f"Video stitch failed: {detail}")


def _video_lab_run_vsr(ctx: Any, src: Path, payload: ProVideoLabRunPayload) -> dict[str, Any]:
    from aiwf.core.domain.vsr import VsrOptions
    from aiwf.services.vsr import VsrUnavailable

    try:
        result = _vsr_service(ctx).upscale(
            src,
            VsrOptions(
                scale=float(payload.scale),
                mode=int(payload.mode),
                strength=float(payload.strength),
                effect=str(payload.effect or "SuperRes"),
            ),
        )
    except VsrUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _video_lab_output(ctx, result.output_path, result.message, {"infotext": result.infotext})


def _video_lab_run_rife(ctx: Any, src: Path, payload: ProVideoLabRunPayload) -> dict[str, Any]:
    from aiwf.core.domain.rife import RifeOptions
    from aiwf.services.rife import RifeUnavailable

    service = _rife_service(ctx)
    try:
        result = service.interpolate(
            src,
            RifeOptions(
                ckpt_name=service.default_checkpoint(),
                multiplier=int(payload.multiplier),
                target_fps=payload.target_fps,
            ),
        )
    except RifeUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _video_lab_output(ctx, result.output_path, getattr(result, "message", "RIFE interpolation complete."), {
        "infotext": getattr(result, "infotext", ""),
    })


def _video_lab_run_audio(ctx: Any, src: Path, payload: ProVideoLabRunPayload) -> dict[str, Any]:
    from aiwf.core.domain.audio import AudioGenerationOptions
    from aiwf.services.audio import AudioUnavailable

    prompt = (payload.audio_prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="Enter an audio prompt describing the soundtrack.")
    service = _audio_service(ctx)
    video_audio_choices = [model_id for _, model_id in (service.video_audio_model_choices() or [])]
    kind = "video_audio" if video_audio_choices else "music"
    model_id = (payload.audio_model or "").strip() or (
        video_audio_choices[0] if video_audio_choices else "facebook/musicgen-small"
    )
    options = AudioGenerationOptions(prompt=prompt, kind=kind, model_id=model_id)
    try:
        audio, muxed = service.generate_and_mux(src, options)
    except AudioUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _video_lab_output(
        ctx,
        muxed.output_path,
        f"Added {kind.replace('_', ' ')} audio -> {Path(muxed.output_path).name}",
        {"audioPath": audio.output_path, "infotext": audio.infotext},
    )


def _video_lab_run_extend(ctx: Any, src: Path, payload: ProVideoLabRunPayload) -> dict[str, Any]:
    """Extend a video: last frame -> Wan 5B i2v continuation -> stitch."""
    from aiwf.core.domain.wan import WAN_RUNTIME_FAST_5B, WanI2VRequest
    from aiwf.infrastructure.video.last_frame import extract_last_frame
    from aiwf.services.wan import WanUnavailable

    if not (payload.prompt or "").strip():
        raise HTTPException(status_code=422, detail="Describe the motion for the extension prompt.")
    checkpoint_id = (payload.checkpoint_id or "").strip()
    if not checkpoint_id:
        raise HTTPException(status_code=422, detail="Pick a Wan 5B model for the extension.")

    probe = _video_lab_probe(src)
    last_frame = extract_last_frame(src)
    settings = getattr(ctx, "settings", None)
    service = _wan_service(ctx)
    sampler = str(getattr(settings, "last_wan_sampler", "") or "unipc").strip().lower()
    if sampler not in {"unipc", "euler", "heun"}:
        sampler = "unipc"
    try:
        vae_id = service.preferred_vae(WAN_RUNTIME_FAST_5B)
    except Exception:
        vae_id = None
    try:
        request = WanI2VRequest(
            prompt=payload.prompt,
            negative_prompt=payload.negative_prompt,
            width=int(probe["width"]) or 480,
            height=int(probe["height"]) or 480,
            num_frames=min(max(int(payload.frames), 5), 257),
            fps=max(1, int(round(probe["fps"] or 16))),
            steps=min(int(payload.steps), 100),
            guidance_scale=float(payload.cfg_scale),
            sampler=sampler,
            flow_shift=float(getattr(settings, "last_wan_flow_shift", 5.0) or 5.0),
            seed=int(payload.seed),
            runtime_mode=WAN_RUNTIME_FAST_5B,
            model_id=checkpoint_id,
            offload=str(getattr(settings, "last_wan_offload", "balanced") or "balanced"),
            vae_id=vae_id,
            text_encoder_path=str(getattr(settings, "last_wan_text_encoder", "") or "").strip(),
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    try:
        result = service.generate(request, last_frame)
    except WanUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    continuation = Path(str(getattr(result, "output_path", "") or ""))
    if not continuation.is_file():
        raise HTTPException(status_code=500, detail="Wan continuation did not produce a video file.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    flags = getattr(ctx, "flags", None)
    dest = flags.resolved_output_dir() / "video-lab" / f"{src.stem}_extended_{stamp}.mp4"
    _concat_videos_dropping_first_frame(src, continuation, dest)
    return _video_lab_output(
        ctx,
        dest,
        f"Extended {src.name} by {request.normalized_frames()} generated frames.",
        {"continuationPath": str(continuation)},
    )


def _video_lab_status_payload(ctx: Any) -> dict[str, Any]:
    vsr = _vsr_service(ctx)
    info = vsr.install_info()
    rife = _rife_service(ctx)
    try:
        rife_checkpoints = rife.list_checkpoints()
    except Exception:
        rife_checkpoints = []
    audio = _audio_service(ctx)
    try:
        video_audio_models = [model_id for _, model_id in (audio.video_audio_model_choices() or [])]
    except Exception:
        video_audio_models = []
    return {
        "vsr": {
            "available": bool(info.available),
            "upscaleAvailable": bool(info.upscale_available),
            "denoiseAvailable": bool(info.denoise_available),
            "sdkRoot": str(info.sdk_root or ""),
            "modelCount": int(info.model_count),
            "features": list(info.feature_names or ()),
            "help": "" if info.available else vsr.folder_help(),
        },
        "rife": {
            "available": bool(rife_checkpoints),
            "checkpoints": rife_checkpoints,
        },
        "audio": {
            "videoAudioModels": video_audio_models,
        },
        "extend": {
            "available": True,
            "note": "Uses the Wan TI2V-5B route: the clip's last frame becomes the i2v conditioning image.",
        },
    }


def _generate_wan_video_response(ctx: Any, payload: ProGeneratePayload) -> dict[str, Any]:
    request = _wan_video_request_from_payload(ctx, payload)
    source_image = _decode_pro_image_data_url(payload.source_image_data_url, "Source image")
    source_path = payload.source_image_path
    if source_image is None and source_path:
        try:
            source_image = Image.open(source_path)
            source_image.load()
        except OSError as exc:
            raise HTTPException(status_code=422, detail="Wan source image could not be opened.") from exc
    if source_image is not None:
        source_image = source_image.convert("RGB")
    job_id = _pro_video_job_start(ctx, request)
    progress: list[dict[str, Any]] = []
    started_at = time.perf_counter()

    def on_progress(step, total, steps_per_second=None, message=None) -> None:
        total_int = max(1, int(total or 1))
        step_int = max(0, int(step or 0))
        message_text = str(message or "") or f"Video denoise {step_int}/{total_int}"
        ratio = min(0.99, step_int / total_int)
        _pro_video_job_update(ctx, job_id, progress=ratio, message=message_text, step=step_int, total=total_int)
        progress.append(
            {
                "stage": "video",
                "progress": ratio,
                "message": message_text,
                "step": step_int,
                "total": total_int,
                "seconds": round(time.perf_counter() - started_at, 3),
            }
        )

    def should_cancel() -> bool:
        return _pro_video_cancel_requested(ctx, job_id)

    try:
        result = _wan_service(ctx).generate(
            request,
            source_image,
            on_progress=on_progress,
            should_cancel=should_cancel,
        )
    except GenerationCancelledError as exc:
        _pro_video_job_finish(ctx, job_id, "cancelled", message=str(exc))
        raise HTTPException(status_code=499, detail=_pro_error_detail(str(exc), job=_pro_video_job_status(ctx))) from exc
    except Exception as exc:
        _pro_video_job_finish(ctx, job_id, "failed", message=str(exc), error=str(exc))
        logger.exception(
            "Pro Wan video generation failed: job=%s model=%s size=%sx%s frames=%s steps=%s",
            job_id,
            request.model_id or "default",
            request.width,
            request.height,
            request.num_frames,
            request.steps,
        )
        raise HTTPException(
            status_code=500,
            detail=_pro_error_detail(str(exc), job=_pro_video_job_status(ctx)),
        ) from exc

    output = _wan_video_output_payload(ctx, result, payload)
    message = str(getattr(result, "message", "") or "Wan video complete.")
    _pro_video_job_finish(ctx, job_id, "completed", message=message)
    return {
        "jobId": job_id,
        "status": "completed",
        "output": output,
        "video": output["url"],
        "recentOutputs": [output],
        "progress": progress,
        "timings": {},
        "message": message,
    }


def build_router(ctx: Any) -> APIRouter:
    if not getattr(ctx, "_pro_startup_started_at", None):
        setattr(ctx, "_pro_startup_started_at", time.time())
    router = APIRouter(prefix="/api/pro")

    @router.get("/startup")
    def startup():
        # CORS-open on purpose: the launcher's loading window is a file://
        # page that must read this payload to know the real backend is up.
        # Boot status carries nothing sensitive.
        return JSONResponse(
            _pro_startup_payload(ctx),
            headers={"Access-Control-Allow-Origin": "*", "Cache-Control": "no-store"},
        )

    @router.post("/startup/window-ready")
    def startup_window_ready():
        if not getattr(ctx, "_pro_window_ready_at", None):
            setattr(ctx, "_pro_window_ready_at", time.time())
        return _pro_startup_payload(ctx)

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
        existing_ids = {str(item.get("id") or "") for item in checkpoints}
        for wan_model in _wan_model_payloads(ctx):
            if wan_model["id"] not in existing_ids:
                checkpoints.append(wan_model)
                existing_ids.add(wan_model["id"])
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

    @router.post("/downloads/catalog/{key}")
    def download_catalog_model(key: str):
        service = getattr(ctx, "model_download", None)
        if service is None:
            raise HTTPException(status_code=500, detail="Model download service is unavailable.")
        entry = service.find_catalog(key)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Catalog entry '{key}' was not found.")
        if bool(getattr(entry, "coming_soon", False)):
            raise HTTPException(
                status_code=422,
                detail="This model family is coming soon and is hidden from v1 app downloads.",
            )
        if not _catalog_entry_can_download(entry):
            if _catalog_entry_requires_auth(entry):
                detail = "This model requires upstream access or authentication. Open the source page and accept access first."
            elif str(getattr(entry, "source", "") or "").strip().lower() == "civitai":
                detail = "Open this CivitAI page. Direct app download is not enabled for CivitAI catalog entries."
            else:
                detail = "This catalog entry is link-only and cannot be downloaded directly by the app."
            raise HTTPException(status_code=422, detail=detail)
        try:
            downloaded_path = service.download_catalog(key)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Catalog download failed: {exc}") from exc
        try:
            _refresh_model_inventory_after_sort(ctx)
        except HTTPException:
            raise
        except Exception:
            logger.exception("Could not refresh model inventory after catalog download")
        payload = _download_payload(ctx)
        payload["downloaded"] = {"key": key, "path": str(downloaded_path)}
        return payload

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

    @router.post("/settings")
    def save_settings(payload: ProSettingsUpdatePayload):
        return _apply_settings_update(ctx, payload)

    @router.get("/capabilities")
    def capabilities():
        return _capability_payload(ctx)

    @router.post("/models/reorganize")
    def models_reorganize():
        flags = getattr(ctx, "flags", None)
        if flags is None:
            raise HTTPException(status_code=500, detail="Runtime flags are unavailable.")
        actions = reorganize_models(flags)
        return _model_sort_response(ctx, actions)

    @router.post("/models/upload")
    async def models_upload(file: UploadFile = File(...)):
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in MODEL_EXTENSIONS:
            allowed = ", ".join(sorted(MODEL_EXTENSIONS))
            raise HTTPException(status_code=422, detail=f"Unsupported model file type '{suffix}'. Use {allowed}.")
        dest = _model_upload_root(ctx) / _safe_upload_filename(file.filename, "model")
        if dest.exists():
            raise HTTPException(status_code=409, detail=f"{SORT_INBOX_DIRNAME}/{dest.name} already exists.")
        written = 0
        try:
            with dest.open("wb") as handle:
                while True:
                    chunk = await file.read(16 * 1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > _MODEL_UPLOAD_MAX_BYTES:
                        raise HTTPException(status_code=413, detail="Model upload is larger than 120 GB.")
                    handle.write(chunk)
        except HTTPException:
            try:
                dest.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        except OSError as exc:
            try:
                dest.unlink(missing_ok=True)
            except OSError:
                pass
            raise HTTPException(status_code=500, detail=f"Could not store the model upload: {exc}") from exc
        finally:
            await file.close()
        flags = getattr(ctx, "flags", None)
        if flags is None:
            raise HTTPException(status_code=500, detail="Runtime flags are unavailable.")
        actions = sort_inbox_models(flags)
        payload = _model_sort_response(ctx, actions, uploaded_path=dest)
        payload["uploadedBytes"] = written
        return payload

    @router.post("/restart")
    def restart():
        _schedule_process_restart()
        return {"status": "restart_requested"}

    @router.post("/support/terminal")
    def support_terminal():
        return _open_support_terminal(ctx)

    @router.post("/models/unload")
    def models_unload():
        return _unload_generation_model(ctx)

    @router.get("/outputs/{requested_path:path}")
    def output_asset(requested_path: str):
        return FileResponse(_output_asset_path(ctx, requested_path))

    @router.post("/metadata/import")
    def metadata_import(payload: ProMetadataImportPayload):
        return _import_generation_metadata_from_image(payload)

    @router.post("/generate")
    def generate(payload: ProGeneratePayload):
        _assert_requested_pipeline_backend(ctx, payload)
        if (payload.mode or "").strip().lower() in {"video", "sana", "sana_video", "wan"}:
            if _pro_video_job_running(ctx) or _image_generation_running(ctx) or _image_generation_pending(ctx):
                raise HTTPException(status_code=409, detail="A generation job is already running. Stop it or wait for it to finish.")
            normalized_mode = (payload.mode or "").strip().lower()
            checkpoint_id = str(payload.checkpoint_id or "")
            _assert_video_route_checkpoint(ctx, checkpoint_id or None)
            if normalized_mode == "wan" or (checkpoint_id and checkpoint_id in _wan_model_ids(ctx)):
                return _generate_wan_video_response(ctx, payload)
            return _generate_sana_video_response(ctx, payload)
        if _pro_video_job_running(ctx):
            raise HTTPException(status_code=409, detail="A Sana video job is already running. Stop it or wait for it to finish.")
        if _image_generation_running(ctx) or _image_generation_pending(ctx):
            raise HTTPException(status_code=409, detail="An image generation job is already running. Stop it or wait for it to finish.")
        checkpoint_id = _checkpoint_id_from_payload(ctx, payload)
        _assert_image_route_checkpoint(ctx, checkpoint_id)
        _assert_checkpoint_selectable(ctx, checkpoint_id)
        request = _generation_request(ctx, payload)
        init_images: list[Image.Image] | None = None
        mask_images: list[Image.Image] | None = None
        if request.mode == GenerationMode.INPAINT:
            _assert_inpaint_checkpoint_supported(ctx, payload)
            init_image = _decode_pro_image_data_url(payload.init_image_data_url, "Init image")
            mask_image = _decode_pro_image_data_url(payload.mask_image_data_url, "Mask image")
            if init_image is None or mask_image is None:
                raise HTTPException(status_code=422, detail="Inpainting requires both an init image and a mask image.")
            init_images = [init_image.convert("RGB")]
            mask_images = [mask_image.convert("L")]
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
            job, progress = _run_pro_image_generation(ctx, request, init_images=init_images, mask_images=mask_images)
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
                detail=_pro_error_detail(
                    str(exc),
                    failureLogPath=failure_log_path,
                    job=_job_status(failed_job),
                    model=_checkpoint_error_context(ctx, checkpoint_id),
                ),
            ) from exc
        response = _generate_response(job)
        if progress:
            response["progress"] = progress
        return response

    @router.get("/video-lab/status")
    def video_lab_status():
        return _video_lab_status_payload(ctx)

    @router.post("/video-lab/upload")
    async def video_lab_upload(file: UploadFile = File(...)):
        suffix = Path(file.filename or "upload.mp4").suffix.lower()
        if suffix not in _VIDEO_LAB_UPLOAD_EXTENSIONS:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported video type '{suffix}'. Use mp4, mov, mkv, webm, or avi.",
            )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_stem = "".join(
            ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in Path(file.filename or "upload").stem
        )[:64] or "upload"
        dest = _video_lab_upload_root(ctx) / f"{safe_stem}_{stamp}{suffix}"
        written = 0
        try:
            with dest.open("wb") as handle:
                while True:
                    chunk = await file.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > _VIDEO_LAB_UPLOAD_MAX_BYTES:
                        raise HTTPException(status_code=413, detail="Video upload is larger than 2 GB.")
                    handle.write(chunk)
        except HTTPException:
            dest.unlink(missing_ok=True)
            raise
        except OSError as exc:
            dest.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail=f"Could not store the upload: {exc}") from exc
        probe = _video_lab_probe(dest)
        probe["url"] = _output_asset_url(ctx, dest)
        return probe

    @router.post("/video-lab/run")
    def video_lab_run(payload: ProVideoLabRunPayload):
        if _pro_video_job_running(ctx) or _image_generation_running(ctx) or _image_generation_pending(ctx):
            raise HTTPException(status_code=409, detail="A generation job is already running. Stop it or wait for it to finish.")
        src = _video_lab_resolve_source(ctx, payload.video_path)
        op = (payload.op or "").strip().lower()
        if op == "vsr":
            return _video_lab_run_vsr(ctx, src, payload)
        if op == "rife":
            return _video_lab_run_rife(ctx, src, payload)
        if op == "audio":
            return _video_lab_run_audio(ctx, src, payload)
        if op == "extend":
            return _video_lab_run_extend(ctx, src, payload)
        raise HTTPException(status_code=422, detail="op must be one of: vsr, rife, audio, extend.")

    @router.post("/segment/auto-mask")
    def segment_auto_mask(payload: ProAutoMaskPayload):
        from aiwf.core.domain.segment import SegmentRequest

        segment = getattr(ctx, "segment", None)
        if segment is None:
            raise HTTPException(status_code=503, detail="Segmentation service is not available in this runtime.")
        prompt = (payload.prompt or "").strip()
        if not prompt:
            raise HTTPException(status_code=422, detail="Enter a SAM + DINO prompt (e.g. 'person', 'face').")
        image = _decode_pro_image_data_url(payload.image_data_url, "Source image")
        if image is None:
            raise HTTPException(status_code=422, detail="Load an image into the inpaint canvas first.")
        try:
            mask, preview, _candidates, status = segment.segment(
                image.convert("RGB"),
                SegmentRequest(
                    text_prompt=prompt,
                    box_threshold=float(payload.box_threshold),
                    dilation=int(payload.dilation),
                    mask_blur=int(payload.mask_blur),
                    feather=int(payload.feather),
                ),
            )
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Auto mask failed: {exc}") from exc
        return {
            "status": status,
            "mask": _image_to_data_url(mask.convert("RGB")),
            "preview": _image_to_data_url(preview),
        }

    @router.post("/faceswap")
    def faceswap(payload: ProFaceSwapPayload):
        faceswap_service = getattr(ctx, "faceswap", None)
        if faceswap_service is None:
            raise HTTPException(status_code=503, detail="Face swap service is not available in this runtime.")
        target = _decode_pro_image_data_url(payload.target_image_data_url, "Target image")
        source = _decode_pro_image_data_url(payload.source_image_data_url, "Source face image")
        if target is None or source is None:
            raise HTTPException(status_code=422, detail="Provide both a target image and a source face image.")
        try:
            result = faceswap_service.swap(target.convert("RGB"), source.convert("RGB"))
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Face swap failed: {exc}") from exc
        return {
            "status": "completed",
            "image": _image_to_data_url(result),
            "width": result.width,
            "height": result.height,
            "message": "Face swap complete.",
        }

    @router.post("/enhance/image")
    def enhance_image(payload: ProEnhanceImagePayload):
        from aiwf.core.domain.enhance import RestoreOptions, UpscaleOptions

        enhance_service = getattr(ctx, "enhance", None)
        if enhance_service is None:
            raise HTTPException(status_code=503, detail="Enhance service is not available in this runtime.")
        image = _decode_pro_image_data_url(payload.image_data_url, "Source image")
        if image is None:
            raise HTTPException(status_code=422, detail="Provide a source image as a base64 data URL.")

        restore_opts = None
        if payload.restore_enabled:
            restore_model = (payload.restore_model or "").strip()
            if not restore_model:
                raise HTTPException(status_code=422, detail="Select or enter a face restoration model.")
            restore_opts = RestoreOptions(
                model_id=restore_model,
                visibility=float(payload.restore_visibility),
                codeformer_weight=float(payload.codeformer_weight),
            )

        upscale_opts = None
        if payload.upscale_enabled:
            upscale_model = (payload.upscale_model or "").strip()
            if not upscale_model:
                raise HTTPException(status_code=422, detail="Select or enter an upscaler model.")
            upscale_opts = UpscaleOptions(
                model_id=upscale_model,
                scale=float(payload.upscale_scale),
                tile_size=int(payload.tile_size),
                tile_overlap=int(payload.tile_overlap),
            )

        if restore_opts is None and upscale_opts is None:
            raise HTTPException(status_code=422, detail="Enable face restore, upscale, or both.")

        try:
            result, infotext = enhance_service.run_pipeline(
                image.convert("RGB"),
                restore=restore_opts,
                upscale=upscale_opts,
                restore_first=bool(payload.restore_first),
            )
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Enhance failed: {exc}") from exc

        flags = getattr(ctx, "flags", None)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        sub = getattr(getattr(ctx, "settings", None), "enhance_output_subdir", "extras-images")
        dest = flags.resolved_output_dir() / sub / f"pro_enhance_{stamp}.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        result.save(dest, format="PNG")
        return {
            "status": "completed",
            "outputPath": str(dest),
            "url": _output_asset_url(ctx, dest),
            "width": result.width,
            "height": result.height,
            "image": _image_to_data_url(result),
            "infotext": infotext,
            "message": infotext or "Enhance complete.",
        }

    @router.post("/vsr/image")
    def vsr_image(payload: ProVsrImagePayload):
        from aiwf.core.domain.vsr import VsrOptions
        from aiwf.services.vsr import VsrUnavailable

        image = _decode_pro_image_data_url(payload.image_data_url, "Source image")
        if image is None:
            raise HTTPException(status_code=422, detail="Provide a source image as a base64 data URL.")
        try:
            result = _vsr_service(ctx).upscale_image(
                image.convert("RGB"),
                VsrOptions(
                    scale=float(payload.scale),
                    mode=int(payload.mode),
                    strength=float(payload.strength),
                    effect=str(payload.effect or "SuperRes"),
                ),
            )
        except VsrUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        flags = getattr(ctx, "flags", None)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        sub = getattr(getattr(ctx, "settings", None), "vsr_output_subdir", "vsr-videos")
        dest = flags.resolved_output_dir() / sub / f"vsr_image_{stamp}.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        result.save(dest, format="PNG")
        return {
            "status": "completed",
            "outputPath": str(dest),
            "url": _output_asset_url(ctx, dest),
            "width": result.width,
            "height": result.height,
            "image": _image_to_data_url(result),
            "message": f"Upscaled to {result.width}x{result.height} via NVIDIA VSR.",
        }

    @router.get("/extensions")
    def extensions():
        registry = getattr(ctx, "plugins", None)
        plugins = registry.list_plugins() if registry is not None else []
        api_routes = {plugin_id for plugin_id, _router in getattr(registry, "api_routers", []) or []}
        flags = getattr(ctx, "flags", None)
        plugins_dir = str(flags.data_dir / "plugins") if flags is not None else ""
        disabled = list(getattr(getattr(ctx, "settings", None), "disabled_extensions", None) or [])
        return {
            "pluginsDir": plugins_dir,
            "disabled": disabled,
            "extensions": [
                {
                    "id": plugin.id,
                    "name": plugin.name,
                    "version": plugin.version,
                    "description": plugin.description,
                    "path": plugin.path,
                    "enabled": plugin.enabled,
                    "error": plugin.error,
                    "hasApi": plugin.id in api_routes,
                    "apiBase": f"/api/ext/{plugin.id}" if plugin.id in api_routes else "",
                }
                for plugin in plugins
            ],
        }

    @router.post("/extensions/toggle")
    def extensions_toggle(payload: ProExtensionTogglePayload):
        settings = getattr(ctx, "settings", None)
        if settings is None:
            raise HTTPException(status_code=500, detail="Settings are not available in this runtime.")
        extension_id = (payload.extension_id or "").strip()
        if not extension_id:
            raise HTTPException(status_code=422, detail="extension id is required")
        disabled = [str(item) for item in (getattr(settings, "disabled_extensions", None) or [])]
        normalized = {item.lower() for item in disabled}
        if payload.enabled and extension_id.lower() in normalized:
            disabled = [item for item in disabled if item.lower() != extension_id.lower()]
        elif not payload.enabled and extension_id.lower() not in normalized:
            disabled.append(extension_id)
        settings.disabled_extensions = disabled
        save_settings = getattr(ctx, "save_settings", None)
        if callable(save_settings):
            try:
                save_settings()
            except OSError as exc:
                raise HTTPException(status_code=500, detail=f"Could not save settings: {exc}") from exc
        return {
            "status": "saved",
            "disabled": disabled,
            "note": "Extension changes apply on the next app restart.",
        }

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
