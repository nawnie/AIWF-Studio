from __future__ import annotations

import base64
import asyncio
import hashlib
import heapq
import io
import json
import logging
import os
import platform
import re
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from urllib.parse import quote, urlparse
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from PIL import Image
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, ValidationError, model_validator

from aiwf import __version__ as AIWF_VERSION
from aiwf.core.config.launch import LaunchSettings, save_launch_settings
from aiwf.core.config.mobile_auth import ensure_mobile_token, load_mobile_auth, set_mobile_access_enabled
from aiwf.core.config.settings import RuntimeFlags, normalize_vram_profile
from aiwf.core.domain.audio import AudioGenerationOptions
from aiwf.core.domain.controlnet import ControlNetUnit
from aiwf.core.domain.errors import GenerationCancelledError
from aiwf.core.domain.errors import ModelNotFoundError
from aiwf.core.domain.generation import GenerationMode, GenerationRequest, JobRecord, JobState
from aiwf.core.domain.ltx import (
    LTX_GEMMA_BACKEND_HF_SAFETENSORS,
    LTX_PIPELINE_DIFFUSERS_2B,
    LTX_PIPELINE_DISTILLED,
    LTX_PIPELINE_ONE_STAGE,
    LTX_QUANTIZATION_MODES,
    LTX_OFFLOAD_MODES,
    LtxVideoRequest,
)
from aiwf.core.domain.models import SCHEDULE_TYPES, normalize_schedule_id_for_sampler
from aiwf.core.domain.sana_video import SanaVideoRequest
from aiwf.core.domain.workflow import WorkflowDefinition
from aiwf.core.infotext import normalize_sampler, parse_infotext
from aiwf.infrastructure.diffusers.model_blocks import (
    is_non_selectable_image_asset_path,
    known_broken_selectable_image_asset,
)
from aiwf.infrastructure.diffusers.model_arch import is_sd3_architecture
from aiwf.infrastructure.model_inventory import MODEL_EXTENSIONS, classify_model_file, model_inventory_roots, scan_and_write_model_inventory, scan_model_inventory_report
from aiwf.infrastructure.model_sorter import SORT_INBOX_DIRNAME, _is_confident, plan_model_reorganize, reorganize_models, sort_inbox_models
from aiwf.services.model_download_catalog import CIVITAI_BROWSE_LINKS, quick_start_bundles_for_platform
from aiwf.services.audio_project import (
    AudioProjectAssetMissing,
    AudioProjectCorrupt,
    AudioProjectError,
    AudioProjectNotFound,
    AudioProjectRecoveryError,
    AudioProjectService,
)
from aiwf.services.image_artifacts import image_artifact_dimensions
from aiwf.services.ltx_engine_setup import ltx_engine_install_status, start_ltx_engine_install
from aiwf.services.qwen_nunchaku_engine_setup import (
    qwen_nunchaku_engine_install_status,
    start_qwen_nunchaku_engine_install,
)
from aiwf.services.pipeline_readiness import (
    READINESS_STATUSES,
    PipelineReadinessRecord,
    collect_pipeline_readiness,
    readiness_summary,
)
from aiwf.services.pro_workflow_runs import ProWorkflowRunService
from aiwf.services.route_lifecycle import (
    begin_route_operation,
    clear_route_residency,
    confirm_route_residency,
    finish_route_operation,
    finish_route_preparation,
    lifecycle_snapshot,
    mark_route_running,
    select_route,
)
from aiwf.services.workflow_executor import validate_workflow

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
_RUNTIME_RUNNING_TICK_SECONDS = 0.15
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
_LTX_SERVICES: dict[int, Any] = {}
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
_PRO_WORKFLOW_RUN_SERVICES_LOCK = threading.RLock()
_PRO_MODEL_REORGANIZE_LOCK = threading.RLock()
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
    "flux2": "Flux.2 Klein",
    "krea2": "Krea 2",
    "sana_video": "Sana Video",
    "wan": "Wan Video",
    "ltx": "LTX Video",
    "sd15": "Stable Diffusion 1.5",
    "sdxl": "Stable Diffusion XL",
    "sd35": "Stable Diffusion 3.5",
    "zimage": "Z-Image",
    "qwen": "Qwen Image",
    "sana": "Sana",
    "unknown": "Other",
    "flux2_generic": "Flux.2 (generic)",
}

_READINESS_NEEDS_WORK_STATUSES = (
    "metadata-only",
    "blocked-cleanly",
    "broken-runtime",
    "unsupported-no-route",
)
_HIDDEN_V1_MODEL_ARCHITECTURES = {"anima"}

_READINESS_SORT_ORDER = {status: index for index, status in enumerate(READINESS_STATUSES)}


def _pro_sana_video_backend_enabled() -> bool:
    return bool(_PRO_SANA_VIDEO_BACKEND_ENABLED)


class ProWorkflowRunPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    workflow: WorkflowDefinition
    source_image_data_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("sourceImageDataUrl", "source_image_data_url"),
    )
    idempotency_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("idempotencyKey", "idempotency_key"),
        max_length=128,
    )


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
    sana_model_variant: str = Field(
        default="480p",
        validation_alias=AliasChoices("sanaModelVariant", "sana_model_variant", "modelVariant", "model_variant"),
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
    ltx_pipeline: str = Field(default=LTX_PIPELINE_ONE_STAGE, validation_alias=AliasChoices("ltxPipeline", "ltx_pipeline"))
    ltx_image_strength: float = Field(default=0.8, ge=0.0, le=1.0, validation_alias=AliasChoices("ltxImageStrength", "ltx_image_strength"))
    ltx_offload: str = Field(default="disk", validation_alias=AliasChoices("ltxOffload", "ltx_offload"))
    ltx_quantization: str = Field(default="fp8-cast", validation_alias=AliasChoices("ltxQuantization", "ltx_quantization"))
    ltx_enhance_prompt: bool = Field(default=False, validation_alias=AliasChoices("ltxEnhancePrompt", "ltx_enhance_prompt"))
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


class ProMobilePairingUpdatePayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    enabled: bool
    rotate: bool = False


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


class ProModelLoadPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    model_id: str = Field(validation_alias=AliasChoices("modelId", "model_id", "checkpointId", "checkpoint_id"))


class ProModelReorganizePayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    plan_id: str = Field(alias="planId", min_length=16, max_length=64)


class ProModelRootPlacementPreviewPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    scan_id: str = Field(alias="scanId", min_length=16, max_length=64)
    path: str = Field(min_length=1, max_length=4096)


class ProModelRootPlacementApplyPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    plan_id: str = Field(alias="planId", min_length=16, max_length=64)


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


def _checkpoint_asset_status(data: dict[str, Any]) -> str:
    raw_path = str(data.get("path") or "").strip()
    if not raw_path:
        return "unknown"
    path = Path(raw_path)
    if not path.is_absolute():
        return "unknown"
    try:
        return "present" if path.exists() else "missing"
    except OSError:
        return "unknown"


def _missing_checkpoint_detail(item: Any) -> dict[str, str] | None:
    data = _dump_model(item)
    if _checkpoint_asset_status(data) != "missing":
        return None
    label = str(data.get("title") or data.get("id") or data.get("filename") or "Selected model")
    return {
        "status": "missing-assets",
        "reason": f"{label} is listed in the model inventory, but its checkpoint file is no longer present.",
        "suggestedAction": "Restore the model file or refresh the Studio model inventory before generating.",
    }


def _checkpoint_payload(ctx: Any, item: Any) -> dict[str, Any]:
    data = _dump_model(item)
    checkpoint_id = str(data.get("id") or "")
    display_architecture = _checkpoint_display_architecture(data)
    setup_data = {**data, "architecture": display_architecture}
    if display_architecture.strip().lower() in {"flux", "flux_fill"}:
        backend = getattr(getattr(ctx, "generation", None), "backend", None)
        conditioning = getattr(backend, "_flux_prompt_conditioning", None)
        mode = str(getattr(conditioning, "mode", "") or "").strip().lower()
        setup_data["flux_conditioning_mode"] = mode if mode in {
            "teacher", "distillt5-control", "universal",
        } else "unknown"
    elif display_architecture.strip().lower() in {"qwen_image", "qwen-image"}:
        model_path = Path(str(data.get("path") or "")).expanduser()
        if model_path.is_dir():
            try:
                model_index = json.loads((model_path / "model_index.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                model_index = {}
            if isinstance(model_index, dict) and model_index.get("_class_name") == "QwenImage21Pipeline":
                setup_data["qwen_image_21_pipeline_available"] = _qwen_image_21_pipeline_available()
    setup_route = _setup_route_descriptor(setup_data)
    engine_id = _engine_id_for_architecture(display_architecture)
    size_bytes = int(data.get("size_bytes") or data.get("sizeBytes") or 0)
    # Rough resident-VRAM estimate: weights that live on the GPU plus a
    # working margin for activations/VAE. Large LM text encoders are kept in
    # system RAM by the backend, so they are intentionally not counted.
    est_vram_gb = round(size_bytes / 1024**3 + 2.5, 1) if size_bytes > 0 else 0.0
    engine_label = _checkpoint_engine_label(data, engine_id)
    if engine_id == "flux2" and engine_label == "Flux.2":
        # Keep Pro's display taxonomy aligned with the identity-derived label;
        # the inventory's broad Flux.2 detector may still report Klein.
        display_architecture = "flux2"
        engine_id = "unknown"
    return {
        "id": checkpoint_id,
        "title": data.get("title", data.get("id", "")),
        "filename": data.get("filename", ""),
        "hash": data.get("hash"),
        "kind": data.get("kind", "checkpoint"),
        "architecture": display_architecture,
        "sizeBytes": size_bytes,
        "fileCount": int(data.get("file_count") or data.get("fileCount") or 0),
        "assetSummary": data.get("asset_summary") or data.get("assetSummary") or "",
        "engineId": engine_id,
        "engineLabel": engine_label,
        "setupBundleKey": setup_route.get("setupBundleKey") if setup_route else None,
        "setupRoute": setup_route,
        "estVramGb": est_vram_gb,
        "heavyFor12Gb": bool(est_vram_gb > 12.0),
        "checkpointPathStatus": _checkpoint_asset_status(data),
        "routeStatus": "unknown",
        "generationPreset": _generation_preset_payload(ctx, checkpoint_id),
    }


def _recommended_setup_bundle_key(data: dict[str, Any]) -> str | None:
    from aiwf.services.model_setup_manifest import recommended_setup_bundle_key

    return recommended_setup_bundle_key(data)


def _setup_route_descriptor(data: dict[str, Any]) -> dict[str, Any] | None:
    from aiwf.services.model_setup_manifest import setup_route_descriptor

    return setup_route_descriptor(data)


def _checkpoint_engine_label(data: dict[str, Any], engine_id: str) -> str:
    architecture = _checkpoint_display_architecture(data).strip().lower()
    # A familiar token in an exported filename is weak evidence compared with
    # the inventory's recognized architecture. Keep identity heuristics for
    # otherwise generic/ambiguous Flux records only.
    if engine_id != "unknown" and architecture not in {
        "unknown", "flux", "flux2", "flux.2", "flux2_klein", "flux_kontext",
    }:
        return _ENGINE_LABELS.get(engine_id, _ENGINE_LABELS["unknown"])
    identity = " ".join(
        str(data.get(key) or "")
        # Use file identity only. Titles and parent directories can contain
        # inherited labels/folder names that do not describe the model itself.
        for key in ("filename", "id")
    ).lower().replace("_", "-")
    if architecture == "flux_kontext" or "kontext" in identity:
        return "Flux Kontext"
    explicit_klein = any(marker in identity for marker in ("klein", "f2k"))
    generic_flux2 = (
        architecture in {"flux2", "flux.2"}
        or "flux.2" in identity
        or "flux2" in identity.replace("-", "")
    )
    if generic_flux2 and not explicit_klein:
        return "Flux.2"
    return _ENGINE_LABELS.get(engine_id, _ENGINE_LABELS["unknown"])


def _checkpoint_display_architecture(data: dict[str, Any]) -> str:
    """Recover specific Flux variants from stable file identity when cached inventory is broad."""
    architecture = str(data.get("architecture") or "unknown").strip().lower()
    if architecture in {"flux", "flux2", "flux.2"}:
        # Do not inspect parent directories: a generic Flux file can live under
        # a folder named "Kontext" without being a Kontext checkpoint.
        identity = " ".join(str(data.get(key) or "") for key in ("filename", "id"))
        if "kontext" in identity.lower().replace("_", "-"):
            return "flux_kontext"
    return architecture


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
    architecture = _checkpoint_display_architecture(data)
    path = str(data.get("path") or data.get("filename") or "")
    if not path:
        return None
    if _is_hidden_v1_checkpoint_architecture(architecture):
        return {
            "status": "coming-soon",
            "reason": "This model family is blocked for the v1 app until its native runtime has a passing smoke receipt.",
            "suggestedAction": "Keep the files installed for sorting/research, but use a supported v1 image route for generation.",
        }
    if architecture.strip().lower() == "qwen_image_nunchaku":
        from aiwf.services.qwen_nunchaku import QwenNunchakuService

        status = QwenNunchakuService(getattr(ctx, "flags", None)).status(path)
        details = "; ".join(status.messages) if status.messages else "runtime and assets are present"
        return {
            "status": "blocked-runtime",
            "reason": (
                f"Qwen Nunchaku setup state: {details}. Generation remains blocked until this route passes "
                "an actual model load and generation smoke test."
            ),
            "suggestedAction": "Set up the Qwen Nunchaku model and isolated runtime, then use a validated image route until its generation smoke passes.",
        }
    if architecture.strip().lower() in {"flux", "flux_fill"}:
        if Path(path).is_dir():
            return {
                "status": "blocked-cleanly",
                "reason": "This generic Flux Diffusers folder is not loadable by the current single-file Flux route.",
                "suggestedAction": "Choose a supported Flux .gguf or .safetensors transformer file, or a full Flux Kontext Diffusers pipeline folder.",
            }
        backend = getattr(getattr(ctx, "generation", None), "backend", None)
        resolver = getattr(backend, "_validate_flux_prompt_assets", None) or getattr(
            backend, "_resolve_flux_component_paths", None
        )
        if callable(resolver):
            try:
                resolver()
            except ModelNotFoundError as exc:
                reason = str(exc)
                if "Flux" in reason and (
                    "missing required local assets" in reason
                    or "tokenizer" in reason.casefold()
                ):
                    return {
                        "status": "missing-assets",
                        "reason": reason,
                        "suggestedAction": "Install the listed Flux support models from Model Setup, or add their shared model root in Settings, then refresh the model list.",
                    }
    if architecture.strip().lower() in {"flux2", "flux.2"}:
        return {
            "status": "blocked-cleanly",
            "reason": "Generic Flux.2 is not the Flux.2 Klein model supported by this Pro runtime route.",
            "suggestedAction": "Use a verified Flux.2 Klein model, or wait until a dedicated generic Flux.2 runtime is implemented.",
        }
    if architecture.strip().lower() == "flux2_klein":
        backend = getattr(getattr(ctx, "generation", None), "backend", None)
        model_path = Path(path)
        try:
            if model_path.is_dir():
                from aiwf.infrastructure.diffusers.checkpoints import flux2_klein_missing_local_files

                complete = (model_path / "model_index.json").is_file() and not flux2_klein_missing_local_files(model_path)
                if not complete:
                    raise ModelNotFoundError(
                        f"Flux.2 Klein pipeline is incomplete at {model_path}; expected transformer, "
                        "text_encoder, tokenizer, scheduler and VAE local components."
                    )
            else:
                resolver = getattr(backend, "_resolve_component_dir", None)
                if callable(resolver):
                    resolver(architecture, item)
        except ModelNotFoundError as exc:
            return {
                "status": "missing-assets",
                "reason": str(exc),
                "suggestedAction": "Install the matching Flux.2 Klein setup from Model Setup, including its text encoder, tokenizer, scheduler and VAE components.",
            }
    single_file_config_architectures = {
        "sd15", "inpaint", "sdxl", "sdxl_inpaint", "sdxl_refiner", "sd35", "sd3",
        "stable-diffusion-3", "stable-diffusion-3.5",
    }
    if architecture.strip().lower() in single_file_config_architectures and Path(path).suffix.lower() in {".safetensors", ".ckpt", ".pt"}:
        backend = getattr(getattr(ctx, "generation", None), "backend", None)
        can_preload = getattr(backend, "can_preload_checkpoint_locally", None)
        if callable(can_preload):
            try:
                local_config_ready = bool(can_preload(str(data.get("id") or "") or None))
            except Exception as exc:
                logger.debug("Single-file Diffusers config readiness failed", exc_info=True)
                return {
                    "status": "blocked-runtime",
                    "reason": f"Local support config readiness could not be verified: {exc}",
                    "suggestedAction": "Check the Diffusers runtime and refresh model readiness.",
                }
            if not local_config_ready:
                normalized = architecture.strip().lower()
                family = "SD 1.5" if normalized in {"sd15", "inpaint"} else "SDXL" if normalized.startswith("sdxl") else "SD 3.5"
                return {
                    "status": "missing-assets",
                    "reason": f"{family} single-file checkpoint needs its local Diffusers config/tokenizer support snapshot.",
                    "suggestedAction": f"Install the {family} support config from Model Setup, then refresh model readiness.",
                }
    selected_architecture = architecture.strip().lower()
    if Path(path).is_dir() and selected_architecture in {"sd15", "sdxl", "sdxl_inpaint", "sd35", "sd3", "stable-diffusion-3", "stable-diffusion-3.5"}:
        family, accepted_classes = {
            "sd15": ("SD 1.5", {"StableDiffusionPipeline", "StableDiffusionImg2ImgPipeline", "StableDiffusionInpaintPipeline"}),
            "sdxl": ("SDXL", {"StableDiffusionXLPipeline", "StableDiffusionXLImg2ImgPipeline", "StableDiffusionXLInpaintPipeline"}),
            "sdxl_inpaint": ("SDXL", {"StableDiffusionXLInpaintPipeline"}),
            "sd35": ("SD 3.5", {"StableDiffusion3Pipeline"}),
            "sd3": ("SD 3.5", {"StableDiffusion3Pipeline"}),
            "stable-diffusion-3": ("SD 3.5", {"StableDiffusion3Pipeline"}),
            "stable-diffusion-3.5": ("SD 3.5", {"StableDiffusion3Pipeline"}),
        }[selected_architecture]
        block = _selected_diffusers_snapshot_block(
            Path(path), family=family, accepted_classes=accepted_classes
        )
        if block:
            return block
    if architecture.strip().lower() in {"z_image", "zimage"}:
        runtime_block = _z_image_runtime_block()
        if runtime_block:
            return runtime_block
        model_path = Path(path)
        if model_path.is_dir():
            from aiwf.infrastructure.diffusers.checkpoints import z_image_missing_local_files

            missing = z_image_missing_local_files(model_path)
            if missing:
                return {
                    "status": "missing-assets",
                    "reason": "Z-Image Diffusers snapshot is incomplete: " + ", ".join(str(item) for item in missing[:8]),
                    "suggestedAction": "Complete the Z-Image Diffusers folder or install the Z-Image Turbo setup from Model Setup, then refresh readiness.",
                }
            block = _selected_diffusers_snapshot_block(
                model_path,
                family="Z-Image",
                accepted_classes={"ZImagePipeline"},
            )
            if block:
                return block
        elif model_path.suffix.lower() in {".gguf", ".safetensors"}:
            backend = getattr(getattr(ctx, "generation", None), "backend", None)
            resolver = getattr(backend, "_resolve_component_dir", None)
            if not callable(resolver):
                return {
                    "status": "blocked-runtime",
                    "reason": "Z-Image support assets could not be verified for the selected checkpoint.",
                    "suggestedAction": "Check the Z-Image runtime, then refresh model readiness.",
                }
            try:
                resolver(architecture, item)
            except ModelNotFoundError as exc:
                return {
                    "status": "missing-assets",
                    "reason": str(exc),
                    "suggestedAction": "Install the Z-Image Turbo components from Model Setup, or add their shared model root in Settings, then refresh readiness.",
                }
            except Exception as exc:
                logger.debug("Z-Image component readiness check failed", exc_info=True)
                return {
                    "status": "blocked-runtime",
                    "reason": f"Z-Image component readiness could not be verified: {exc}",
                    "suggestedAction": "Check the Z-Image runtime and component folders, then refresh model readiness.",
                }
        else:
            return {
                "status": "blocked-cleanly",
                "reason": "Z-Image expects a complete Diffusers folder or a .gguf/.safetensors transformer file.",
                "suggestedAction": "Select a supported Z-Image model file or complete Diffusers folder.",
            }
    if architecture.strip().lower() in {"flux_kontext", "flux-kontext"}:
        model_path = Path(path)
        if model_path.is_dir():
            if not (model_path / "model_index.json").is_file():
                return {
                    "status": "blocked-cleanly",
                    "reason": "Flux Kontext expects a complete Diffusers pipeline folder or a verified GGUF transformer file.",
                    "suggestedAction": "Select a complete local Kontext Diffusers folder or the installed Kontext GGUF transformer.",
                }
            try:
                from aiwf.infrastructure.diffusers.checkpoints import flux_kontext_missing_local_files

                missing_files = flux_kontext_missing_local_files(model_path, limit=8)
            except Exception as exc:
                logger.debug("Flux Kontext local shard check failed", exc_info=True)
                missing_files = [model_path / "model_index.json"]
            if missing_files:
                return {
                    "status": "missing-assets",
                    "reason": "Flux Kontext pipeline has missing local files: " + ", ".join(str(item) for item in missing_files),
                    "suggestedAction": "Complete the Flux Kontext Diffusers snapshot before generating.",
                }
        elif model_path.suffix.lower() == ".gguf":
            try:
                has_selected_gguf = model_path.is_file() and model_path.stat().st_size > 0
            except OSError:
                has_selected_gguf = False
            if not has_selected_gguf:
                return {
                    "status": "missing-assets",
                    "reason": "The selected Flux Kontext GGUF is missing or empty.",
                    "suggestedAction": "Restore a nonempty local Flux Kontext GGUF transformer, then refresh readiness.",
                }
            backend = getattr(getattr(ctx, "generation", None), "backend", None)
            header_check = getattr(backend, "_is_flux_kontext_gguf_transformer", None)
            if not callable(header_check):
                return {
                    "status": "blocked-runtime",
                    "reason": "The runtime cannot verify the selected Flux Kontext GGUF header.",
                    "suggestedAction": "Check the Flux Kontext runtime and refresh model readiness.",
                }
            try:
                if not header_check(model_path):
                    return {
                        "status": "blocked-cleanly",
                        "reason": "The selected file is not a header-verified Flux Kontext GGUF transformer.",
                        "suggestedAction": "Select the local Flux Kontext GGUF transformer file.",
                    }
                resolver = getattr(backend, "_resolve_component_dir", None)
                if not callable(resolver):
                    return {
                        "status": "blocked-runtime",
                        "reason": "The runtime cannot verify local Flux Kontext companion components.",
                        "suggestedAction": "Check the Flux Kontext runtime and refresh model readiness.",
                    }
                resolver(architecture, item)
            except ModelNotFoundError as exc:
                return {
                    "status": "missing-assets",
                    "reason": str(exc),
                    "suggestedAction": "Install or complete the local Flux Kontext component snapshot, then refresh readiness.",
                }
            except Exception as exc:
                logger.debug("Flux Kontext GGUF readiness check failed", exc_info=True)
                return {
                    "status": "blocked-runtime",
                    "reason": f"Flux Kontext GGUF readiness could not be verified: {exc}",
                    "suggestedAction": "Check the Flux Kontext runtime and local components, then refresh readiness.",
                }
        else:
            return {
                "status": "blocked-cleanly",
                "reason": "Flux Kontext supports complete Diffusers pipeline folders or a verified GGUF transformer; this file format is unsupported.",
                "suggestedAction": "Select a complete local Kontext Diffusers folder or the installed Kontext GGUF transformer.",
            }
    if architecture.strip().lower() == "sana_video":
        model_path = Path(path).expanduser()
        if not model_path.is_absolute():
            model_path = (Path(getattr(getattr(ctx, "flags", None), "data_dir", ".")) / model_path).resolve()
        try:
            from aiwf.core.domain.sana_video import SanaVideoRequest
            from aiwf.services.pipeline_preflight import preflight_sana_video_pipeline

            preflight = preflight_sana_video_pipeline(
                ctx.flags,
                getattr(ctx, "settings", None),
                request=SanaVideoRequest(model_path=str(model_path)),
            )
            if not preflight.ok:
                model_installed = str(preflight.metadata.get("model_installed", "")).lower() == "true"
                detail = "; ".join(preflight.warnings) or preflight.message()
                return {
                    "status": "blocked-runtime" if model_installed else "missing-assets",
                    "reason": f"Sana Video is not ready for this model: {detail}",
                    "suggestedAction": (
                        "The Sana Video model files are already present. Enable or repair the installed Diffusers runtime for Sana Video, then refresh model readiness; reinstalling the model files is not needed."
                        if model_installed
                        else "Install the Sana Video setup from Model Setup, then refresh model readiness."
                    ),
                }
        except Exception as exc:
            logger.debug("Sana Video Pro preflight failed", exc_info=True)
            return {
                "status": "blocked-runtime",
                "reason": f"Sana Video readiness could not be verified for this model: {exc}",
                "suggestedAction": "Check the Sana Video runtime and model setup, then refresh model readiness.",
            }
    if architecture.strip().lower() in {"qwen_image", "qwen-image"}:
        model_path = Path(path)
        try:
            model_index = json.loads((model_path / "model_index.json").read_text(encoding="utf-8")) if model_path.is_dir() else {}
        except (OSError, ValueError):
            model_index = {}
        pipeline_name = str(model_index.get("_class_name") or "") if isinstance(model_index, dict) else ""
        if pipeline_name == "QwenImage21Pipeline" and not _qwen_image_21_pipeline_available():
            return {
                "status": "blocked-runtime",
                "reason": "This is a Qwen Image 2.1 pipeline, but the installed Diffusers runtime does not expose QwenImage21Pipeline.",
                "suggestedAction": "Install a compatible AIWF-supported Diffusers runtime or choose a Qwen Image pipeline supported by this installation.",
            }
        block = _selected_diffusers_snapshot_block(
            model_path,
            family="Qwen Image",
            accepted_classes={"QwenImagePipeline", "QwenImage21Pipeline"},
        )
        if block:
            return block
    if architecture.strip().lower() == "sana":
        block = _selected_diffusers_snapshot_block(
            Path(path),
            family="Sana",
            accepted_classes={"SanaPipeline", "SanaSprintPipeline"},
        )
        if block:
            return block
    if architecture.strip().lower() in {"krea2", "krea_2"}:
        model_path = Path(path)
        if not model_path.is_dir():
            return {
                "status": "blocked-cleanly",
                "reason": "Krea 2 split weights are not runnable by the current Pro loader; it needs a complete Diffusers pipeline folder.",
                "suggestedAction": "Use a complete Krea 2 Diffusers pipeline under models/krea2/Diffusers. Split-file generation support is not available yet.",
            }
        try:
            from aiwf.services.pipeline_preflight import preflight_krea2_pipeline

            preflight = preflight_krea2_pipeline(ctx.flags)
            resolved_folder = Path(preflight.metadata.get("diffusers_folder") or "").resolve()
            if preflight.ok and resolved_folder == model_path.resolve():
                return None
            detail = "; ".join(preflight.warnings) or "The Krea 2 pipeline preflight is not ready."
        except Exception as exc:
            logger.debug("Krea 2 Pro preflight failed", exc_info=True)
            detail = str(exc) or "Krea 2 preflight could not verify its local pipeline."
        return {
            "status": "blocked-cleanly",
            "reason": f"Krea 2 is not ready for Pro generation: {detail}",
            "suggestedAction": "Install a complete Krea 2 Diffusers pipeline and ensure the installed Diffusers runtime exposes Krea2Pipeline.",
        }
    if (
        is_sd3_architecture(architecture)
        and not Path(path).is_dir()
        and "large" in path.lower()
        and not _sd35_large_access_available()
    ):
        return {
            "status": "blocked-cleanly",
            "reason": "SD3.5 Large single-file checkpoints need gated Stability AI config files unless those files are cached locally.",
            "suggestedAction": "Sign in to Hugging Face with SD3.5 Large access, or provide a local diffusers pipeline folder for this model.",
        }
    return None


def _z_image_runtime_block() -> dict[str, str] | None:
    """Check the symbols imported by the selected Z-Image loader without loading weights."""
    try:
        import diffusers
        import transformers
    except Exception as exc:
        return {
            "status": "blocked-runtime",
            "reason": f"Z-Image runtime dependencies could not be imported: {exc}",
            "suggestedAction": "Install a compatible Diffusers and Transformers runtime, then refresh model readiness.",
        }
    missing = [
        f"diffusers.{name}"
        for name in ("ZImagePipeline", "ZImageTransformer2DModel")
        if not callable(getattr(diffusers, name, None))
    ]
    missing.extend(
        f"transformers.{name}"
        for name in ("AutoModel", "AutoTokenizer")
        if not callable(getattr(transformers, name, None))
    )
    if not missing:
        return None
    return {
        "status": "blocked-runtime",
        "reason": "Z-Image runtime is missing required classes: " + ", ".join(missing),
        "suggestedAction": "Install a compatible Diffusers and Transformers runtime, then refresh model readiness.",
    }


def _qwen_image_21_pipeline_available() -> bool:
    try:
        import diffusers

        return callable(getattr(diffusers, "QwenImage21Pipeline", None))
    except Exception:
        return False


def _selected_diffusers_snapshot_block(
    model_path: Path,
    *,
    family: str,
    accepted_classes: set[str],
) -> dict[str, str] | None:
    """Validate the selected snapshot itself, not another installed family copy."""
    if not model_path.is_dir() or not (model_path / "model_index.json").is_file():
        return {
            "status": "missing-assets",
            "reason": f"{family} requires a complete Diffusers folder with model_index.json at {model_path}.",
            "suggestedAction": f"Install or select a complete {family} Diffusers snapshot, then refresh model readiness.",
        }
    try:
        index = json.loads((model_path / "model_index.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        index = {}
    class_name = str(index.get("_class_name") or "") if isinstance(index, dict) else ""
    if class_name not in accepted_classes:
        expected = ", ".join(sorted(accepted_classes))
        return {
            "status": "blocked-cleanly",
            "reason": f"Selected folder is not a supported {family} pipeline (found {class_name or 'no _class_name'}; expected {expected}).",
            "suggestedAction": f"Select a {family} Diffusers folder with a supported pipeline class.",
        }
    try:
        from aiwf.infrastructure.diffusers.checkpoints import (
            diffusers_dir_has_required_local_files,
            flux_kontext_missing_local_files,
            missing_diffusers_local_files,
        )

        if family.strip().lower() == "flux kontext":
            missing = flux_kontext_missing_local_files(model_path, limit=8)
        else:
            missing = missing_diffusers_local_files(model_path, limit=8)
    except Exception as exc:
        logger.debug("%s selected snapshot check failed", family, exc_info=True)
        missing = [model_path / "model_index.json"]
    if missing:
        return {
            "status": "missing-assets",
            "reason": f"Selected {family} snapshot has missing local files: " + ", ".join(str(item) for item in missing),
            "suggestedAction": f"Complete the selected {family} Diffusers snapshot before generating.",
        }
    try:
        complete = diffusers_dir_has_required_local_files(model_path)
    except Exception:
        logger.debug("%s selected snapshot completeness check failed", family, exc_info=True)
        complete = False
    if not complete:
        return {
            "status": "missing-assets",
            "reason": f"Selected {family} Diffusers snapshot is missing required component configs or weights.",
            "suggestedAction": f"Complete the selected {family} Diffusers snapshot before generating.",
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
            block = _missing_checkpoint_detail(item)
        if block is None:
            payload["routeStatus"] = (
                "request-eligible" if payload["checkpointPathStatus"] == "present" else "unknown"
            )
            selectable.append(payload)
        else:
            if (
                str(payload.get("architecture") or "").strip().lower() == "sana_video"
                and block.get("status") == "blocked-runtime"
            ):
                # The route can be repaired by enabling/updating Diffusers;
                # reinstalling an already-complete snapshot is unnecessary.
                payload["setupBundleKey"] = None
                setup_route = payload.get("setupRoute")
                if isinstance(setup_route, dict):
                    payload["setupRoute"] = {**setup_route, "setupBundleKey": None}
            payload["routeStatus"] = "blocked"
            payload["readinessReason"] = block["reason"]
            blocked.append({**payload, **block, "routeStatus": "blocked"})
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
    if normalized == "flux2_klein":
        return "flux2"
    if normalized in {"krea2", "krea_2", "krea-2"}:
        return "krea2"
    if normalized in {"sana_video", "sana-video", "sanavideo"}:
        return "sana_video"
    if normalized in {"wan", "wan22", "wan2.2", "wan_video"}:
        return "wan"
    if normalized == "ltx":
        return "ltx"
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
        if engine_id == "unknown" and str(checkpoint.get("engineLabel") or "").strip().casefold() == "flux.2":
            engine_id = "flux2_generic"  # display taxonomy only; generic Flux.2 remains unroutable
        counts[engine_id] = counts.get(engine_id, 0) + 1
    order = ["flux", "flux2", "flux2_generic", "krea2", "sana_video", "wan", "ltx", "sd15", "sdxl", "sd35", "zimage", "qwen", "sana", "unknown"]
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
    if "krea2" in text or "krea-2" in text:
        return "krea2"
    if "qwen" in text:
        return "qwen"
    if "sana" in text:
        return "sana"
    if "wan" in text:
        return "wan"
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
        if image_artifact_dimensions(path) is None:
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


def _pro_video_job_start(
    ctx: Any,
    request: Any,
    *,
    message: str = "Starting video generation.",
    cancel_target: str = "",
) -> str:
    job_id = uuid4().hex
    with _PRO_VIDEO_JOBS_LOCK:
        current = _PRO_VIDEO_JOBS.get(id(ctx))
        if current and str(current.get("state") or "").lower() == "running":
            raise HTTPException(
                status_code=409,
                detail="A video generation job is already running. Stop it or wait for it to finish.",
            )
        _PRO_VIDEO_JOBS[id(ctx)] = {
            "id": job_id,
            "state": "running",
            "progress": 0,
            "step": 0,
            "totalSteps": max(1, int(getattr(request, "steps", 1) or 1)),
            "message": str(message or "Starting video generation."),
            "hasResult": False,
            "error": "",
            "cancelRequested": False,
            "cancelTarget": str(cancel_target or ""),
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
    job.pop("cancelTarget", None)
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
        job["message"] = "Stop requested. The active generation will stop at the next safe checkpoint."
        video_job_id = str(job.get("id") or "")
        cancel_target = str(job.get("cancelTarget") or "")
    if cancel_target == "ltx":
        service = getattr(ctx, "ltx", None) or _LTX_SERVICES.get(id(ctx))
        cancel = getattr(service, "cancel_active_generation", None)
        if callable(cancel):
            try:
                cancel()
            except Exception:
                logger.exception("Could not signal LTX cancellation")
    return video_job_id


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
    if _image_generation_running(ctx) or _image_generation_pending(ctx) or _pro_video_job_running(ctx) or _pro_workflow_runs_active(ctx):
        raise HTTPException(status_code=409, detail="A job is running. Stop it or wait before unloading models.")
    supervisor = getattr(ctx, "supervisor", None)
    if supervisor is None or not callable(getattr(supervisor, "request_switch", None)):
        raise HTTPException(status_code=503, detail="GPU tenant ownership is unavailable; the image model was not unloaded.")
    from aiwf.core.domain.engine import EngineSwitchRequest, EngineTenant

    tenant_job_id = f"image_unload_{uuid4().hex}"
    acquired = supervisor.request_switch(
        EngineSwitchRequest(
            target=EngineTenant.IMAGE,
            reason="Unload image model",
            job_id=tenant_job_id,
        )
    )
    if not acquired.ok:
        raise HTTPException(status_code=409, detail=f"GPU is busy; cannot unload the image model. {acquired.message}")
    loaded_before = _runtime_loaded_model(ctx)
    try:
        unload()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not unload the current model: {exc}") from exc
    finally:
        released = supervisor.request_switch(
            EngineSwitchRequest(
                target=EngineTenant.IDLE,
                reason="Image model unload complete",
                job_id=tenant_job_id,
            )
        )
        if not released.ok:
            raise HTTPException(status_code=500, detail=f"Image model unload finished but GPU ownership could not be released: {released.message}")
    return {
        "status": "unloaded",
        "unloadedModel": loaded_before,
        "runtime": _runtime_summary(ctx),
    }


def _release_image_model_for_video_route(ctx: Any) -> None:
    """Release a confirmed resident image model before a video route claims GPU memory."""
    loaded = _runtime_loaded_model(ctx)
    if not bool(loaded.get("loaded")):
        return
    model_id = str(loaded.get("id") or "")
    _unload_generation_model(ctx)
    if model_id:
        for route_state in lifecycle_snapshot(ctx):
            route_id = str(route_state.get("route") or "")
            if route_id.startswith("image.") and route_state.get("resident") is True:
                confirm_route_residency(
                    ctx,
                    route_id,
                    str(route_state.get("modelId") or model_id),
                    resident=False,
                )


def _unload_image_models_for_active_route(ctx: Any) -> Any:
    """Unload image weights under the caller's GPU tenant and reconcile lifecycle state."""
    backend = getattr(getattr(ctx, "generation", None), "backend", None)
    unload = getattr(backend, "unload", None)
    tracked = [
        state for state in lifecycle_snapshot(ctx)
        if str(state.get("route") or "").startswith("image.") and state.get("resident") is True
    ]
    if not callable(unload):
        if tracked:
            raise RuntimeError("The image backend cannot confirm release of its resident model.")
        return None
    result = unload()
    if result is False:
        raise RuntimeError("The image backend refused to release its resident model.")
    for state in tracked:
        confirm_route_residency(
            ctx,
            str(state.get("route") or ""),
            str(state.get("modelId") or ""),
            resident=False,
        )
    return result


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


def _pro_workflow_runs_active(ctx: Any) -> bool:
    service = getattr(ctx, "_pro_workflow_run_service", None)
    check = getattr(service, "has_active_runs", None)
    try:
        return bool(check()) if callable(check) else False
    except Exception:
        # Do not switch/unload models when workflow activity cannot be inspected.
        return True


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


def _validate_workflow_generation_targets(ctx: Any, workflow: WorkflowDefinition) -> None:
    for step in workflow.steps:
        step_type = getattr(step.type, "value", str(step.type))
        if step_type not in {"txt2img", "img2img", "inpaint"}:
            continue
        try:
            batch_size = int(step.params.get("batch_size", 1))
            batch_count = int(step.params.get("batch_count", 1))
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail="Generation workflow batch settings must be integers.")
        if batch_size != 1 or batch_count != 1:
            raise HTTPException(
                status_code=422,
                detail="Workflow generation currently supports one image per node (batch_size=1 and batch_count=1).",
            )
        checkpoint_id = str(step.params.get("checkpoint_id") or "").strip()
        if not checkpoint_id:
            raise HTTPException(
                status_code=422,
                detail="Generation workflow steps require an explicit installed checkpoint_id.",
            )
        _assert_checkpoint_selectable(ctx, checkpoint_id)
        _assert_image_route_checkpoint(ctx, checkpoint_id)
        engine_id = _checkpoint_engine_id(ctx, checkpoint_id)
        supported_image_engines = {"sd15", "sdxl", "flux_fill"} if step_type == "inpaint" else {"sd15", "sdxl", "sd35"}
        if engine_id not in supported_image_engines:
            raise HTTPException(
                status_code=422,
                detail=_pro_error_detail(
                    "This workflow runner supports SD 1.5, SDXL, and SD 3.5 for image generation; inpaint supports SD 1.5, SDXL, and Flux Fill.",
                    checkpointId=checkpoint_id,
                    status="unsupported-workflow-model-route",
                    suggestedAction="Use an eligible image route with SD 1.5, SDXL, or Flux Fill for inpaint, or SD 1.5, SDXL, or SD 3.5 for other generation nodes.",
                ),
            )
        pipeline_backend = step.params.get("pipeline_backend")
        if pipeline_backend:
            _assert_requested_pipeline_backend(
                ctx,
                ProGeneratePayload(pipeline_backend=str(pipeline_backend)),
            )


def _pro_workflow_run_service(ctx: Any) -> ProWorkflowRunService:
    service = getattr(ctx, "_pro_workflow_run_service", None)
    if service is not None:
        return service
    with _PRO_WORKFLOW_RUN_SERVICES_LOCK:
        service = getattr(ctx, "_pro_workflow_run_service", None)
        if service is not None:
            return service
        workflow_service = getattr(ctx, "workflows", None)
        if workflow_service is None:
            raise HTTPException(status_code=503, detail="Workflow execution is unavailable.")
        save_output = None
        image_store = getattr(getattr(ctx, "enhance", None), "store", None)
        if image_store is not None and callable(getattr(image_store, "save", None)):
            subdir = str(getattr(getattr(ctx, "settings", None), "workflow_output_subdir", "workflows"))
            save_output = lambda image, infotext: image_store.save(image, infotext, subdir)
        root = Path(ctx.flags.data_dir) / "_local" / "pro-workflow-runs"
        service = ProWorkflowRunService(
            root,
            workflow_service.run,
            save_output=save_output,
            output_root=_safe_output_root(ctx),
        )
        setattr(ctx, "_pro_workflow_run_service", service)
        return service


def _pro_workflow_run_payload(ctx: Any, record: dict[str, Any]) -> dict[str, Any]:
    root = _safe_output_root(ctx)

    def output_url(raw_path: Any) -> str | None:
        if not raw_path or root is None:
            return None
        try:
            candidate = Path(str(raw_path)).resolve()
            if not _path_inside(candidate, root) or not candidate.is_file():
                return None
            if candidate.suffix.lower() not in _IMAGE_EXTENSIONS or image_artifact_dimensions(candidate) is None:
                return None
            return _output_asset_url(ctx, candidate)
        except OSError:
            return None

    steps: list[dict[str, Any]] = []
    for step in record.get("steps", []):
        persisted_receipt = dict(step.get("receipt") or {})
        receipt = dict(persisted_receipt)
        image_path = receipt.pop("image_path", None)
        if image_path:
            receipt["image_url"] = output_url(image_path)
        receipt_seed = json.dumps(
            {
                "run_id": record.get("run_id"),
                "step_id": step.get("step_id"),
                "params_sha256": step.get("params_sha256"),
                "status": step.get("status"),
                "started_at": step.get("started_at"),
                "completed_at": step.get("completed_at"),
                "receipt": persisted_receipt,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        steps.append(
            {
                "stepId": step.get("step_id"),
                "type": step.get("type"),
                "label": step.get("label"),
                "status": step.get("status"),
                "generation": {
                    "checkpointId": (step.get("generation") or {}).get("checkpoint_id"),
                    "mode": (step.get("generation") or {}).get("mode"),
                    "route": (step.get("generation") or {}).get("route"),
                    "modelFamily": (step.get("generation") or {}).get("model_family"),
                    "resident": (step.get("generation") or {}).get("resident"),
                } if isinstance(step.get("generation"), dict) else None,
                "paramsSha256": step.get("params_sha256"),
                "startedAt": step.get("started_at"),
                "completedAt": step.get("completed_at"),
                "receiptId": hashlib.sha256(receipt_seed.encode("utf-8")).hexdigest()[:24],
                "receipt": receipt or None,
            }
        )

    output = None
    final_path = record.get("output_path")
    final_url = output_url(final_path)
    final_dimensions = image_artifact_dimensions(final_path) if final_path and final_url else None
    if final_url and final_dimensions:
        output = {"url": final_url, "width": final_dimensions[0], "height": final_dimensions[1]}
    workflow = record.get("workflow") or {}
    return {
        "runId": record.get("run_id"),
        "workflowName": workflow.get("name", "Workflow"),
        "status": record.get("status"),
        "createdAt": record.get("created_at"),
        "updatedAt": record.get("updated_at"),
        "startedAt": record.get("started_at"),
        "completedAt": record.get("completed_at"),
        "currentStep": record.get("current_step"),
        "steps": steps,
        "output": output,
        "summary": record.get("summary"),
        "error": record.get("error"),
        "recovery": record.get("recovery"),
    }




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
            unload_image_models=lambda: _unload_image_models_for_active_route(ctx),
        )
        _SANA_VIDEO_SERVICES[key] = service
    return service


def _release_cached_sana_video(ctx: Any) -> bool:
    """Release Sana's retained pipeline only while operation and VIDEO ownership are safe."""
    service = getattr(ctx, "sana_video", None) or _SANA_VIDEO_SERVICES.get(id(ctx))
    tracked_resident = [
        state for state in lifecycle_snapshot(ctx)
        if str(state.get("route") or "").startswith("video.sana.") and state.get("resident") is True
    ]
    if service is None:
        if tracked_resident:
            raise HTTPException(status_code=503, detail="Sana Video is marked resident but its runtime cannot confirm model release.")
        return True
    release = getattr(service, "release_cached_model_for_modality_switch", None)
    if not callable(release):
        if getattr(service, "_prepared_pipeline", None) is None and not tracked_resident:
            return True
        raise HTTPException(
            status_code=503,
            detail="The Sana Video runtime cannot safely release its selected pipeline before switching modalities.",
        )
    try:
        released = bool(release())
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not release the previous Sana Video model: {exc}") from exc
    if not released:
        raise HTTPException(
            status_code=409,
            detail="Sana Video is using its pipeline or another GPU tenant owns the device. Wait for that operation before switching modalities.",
        )
    for route_state in lifecycle_snapshot(ctx):
        route_id = str(route_state.get("route") or "")
        if route_id.startswith("video.sana."):
            confirm_route_residency(
                ctx,
                route_id,
                str(route_state.get("modelId") or ""),
                resident=False,
            )
    return True


def _unload_cached_ltx_model(ctx: Any) -> bool:
    """Release LTX 2B Diffusers residency before another route claims model memory."""
    tracked_resident = [
        state for state in lifecycle_snapshot(ctx)
        if str(state.get("route") or "").startswith("video.ltx.") and state.get("resident") is True
    ]
    service = getattr(ctx, "ltx", None) or _LTX_SERVICES.get(id(ctx))
    unload = getattr(service, "unload", None)
    unloaded = False
    if callable(unload):
        try:
            unloaded = bool(unload())
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Could not release the selected LTX model: {exc}") from exc
    if not unloaded:
        if tracked_resident:
            raise HTTPException(status_code=503, detail="LTX was marked resident, but its runtime did not confirm model release.")
        # The Diffusers cache is module-global and may outlive its service object.
        from aiwf.services.ltx_diffusers import unload_ltx2b_diffusers_cache

        try:
            unloaded = bool(unload_ltx2b_diffusers_cache())
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Could not release the selected LTX model: {exc}") from exc
    for route_state in lifecycle_snapshot(ctx):
        route_id = str(route_state.get("route") or "")
        if route_id.startswith("video.ltx.") and route_state.get("resident") is True:
            confirm_route_residency(ctx, route_id, str(route_state.get("modelId") or ""), resident=False)
    return unloaded


def _release_cached_audio_model(ctx: Any) -> bool:
    """Release resident MusicGen before another modality claims model memory."""
    service = getattr(ctx, "audio", None) or _AUDIO_SERVICES.get(id(ctx))
    if service is None:
        return True
    release = getattr(service, "release_cached_model_for_modality_switch", None)
    if not callable(release):
        if getattr(service, "_model", None) is None:
            return True
        raise HTTPException(
            status_code=503,
            detail="The audio runtime cannot safely release its selected model before switching modalities.",
        )
    try:
        released = bool(release())
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not release the selected audio model: {exc}") from exc
    if not released:
        raise HTTPException(
            status_code=409,
            detail="Audio still owns or is using its model. Wait for the active audio operation before switching modalities.",
        )
    for route_state in lifecycle_snapshot(ctx):
        route_id = str(route_state.get("route") or "")
        if route_id.startswith("audio.") and route_state.get("resident") is True:
            confirm_route_residency(
                ctx,
                route_id,
                str(route_state.get("modelId") or ""),
                resident=False,
            )
    return True


def _release_cached_wan_model(ctx: Any) -> bool:
    """Release a resident Wan pipeline before another route claims the GPU."""
    service = getattr(ctx, "wan", None) or _WAN_SERVICES.get(id(ctx))
    if service is None:
        return True
    release = getattr(service, "release_cached_model_for_modality_switch", None)
    if not callable(release):
        backend = getattr(service, "_backend", None)
        if backend is None or getattr(backend, "_pipe", None) is None:
            return True
        raise HTTPException(
            status_code=503,
            detail="The Wan runtime cannot safely release its selected pipeline before switching modalities.",
        )
    try:
        released = bool(release())
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Could not release the selected Wan pipeline: {exc}") from exc
    if not released:
        raise HTTPException(
            status_code=409,
            detail="Wan still owns or is using its pipeline. Wait for the active video operation before switching modalities.",
        )
    for route_state in lifecycle_snapshot(ctx):
        route_id = str(route_state.get("route") or "")
        if route_id.startswith("video.wan.") and route_state.get("resident") is True:
            confirm_route_residency(
                ctx,
                route_id,
                str(route_state.get("modelId") or ""),
                resident=False,
            )
    return True


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
            unload_image_models=lambda: _unload_image_models_for_active_route(ctx),
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
    available_ids = {identifier for _display, identifier in labeled}
    payloads: list[dict[str, Any]] = []
    for display, identifier in labeled:
        readiness = _wan_model_readiness(ctx, service, display, identifier)
        setup = _wan_setup_route_metadata(ctx, service, display, identifier, available_ids)
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
                **setup,
                **readiness,
            }
        )
    return payloads


def _wan_setup_route_metadata(
    ctx: Any,
    service: Any,
    display: str,
    identifier: str,
    available_ids: set[str],
) -> dict[str, Any]:
    """Expose a Wan setup action only for a complete, selected runtime route."""
    settings = getattr(ctx, "settings", None)
    runtime_mode = _canonical_wan_runtime_mode(getattr(settings, "last_wan_runtime_mode", "fast_5b"))
    if runtime_mode == "fast_5b":
        identity = f"{display} {identifier}".casefold().replace("_", "-")
        if "ti2v" not in identity or "5b" not in identity:
            return {}
        roots = list(getattr(service, "_wan_diffusers_roots", lambda: [])())
        model_root = getattr(service, "models_dir", None)
        if callable(model_root):
            try:
                roots.append(Path(model_root()))
            except Exception:
                pass
        raw_identifier = Path(identifier)
        candidates = [(raw_identifier, None)] if raw_identifier.is_absolute() else []
        for root in roots:
            root_path = Path(root)
            candidates.extend(((root_path / raw_identifier, root_path), (root_path / raw_identifier.name, root_path)))
        is_complete = getattr(service, "_is_full_fast_5b_diffusers_model", None)
        if callable(is_complete):
            for candidate, confinement_root in candidates:
                try:
                    resolved = candidate.resolve(strict=True)
                    if confinement_root is not None:
                        resolved.relative_to(Path(confinement_root).resolve(strict=True))
                except (OSError, RuntimeError, ValueError):
                    continue
                if is_complete(resolved):
                    return {
                        "setupBundleKey": "wan-ti2v-diffusers",
                        "setupRoute": _setup_route_descriptor({"routeKey": "pro.video.wan.ti2v-diffusers"}),
                    }
        return {}

    high = str(getattr(settings, "last_wan_high", "") or "").strip()
    low = str(getattr(settings, "last_wan_low", "") or "").strip()
    if not high or not low or high == low or not {high, low}.issubset(available_ids):
        return {}
    if identifier not in {high, low}:
        return {}
    pair_identity = f"{high} {low}".casefold().replace("_", "-")
    if not all(marker in pair_identity for marker in ("wan2.2", "i2v", "high", "low", "14b")):
        return {}
    high_suffix = Path(high).suffix.casefold()
    low_suffix = Path(low).suffix.casefold()
    if runtime_mode == "native_high_low_fp8_experimental":
        if high_suffix != ".safetensors" or low_suffix != ".safetensors" or "fp8" not in pair_identity:
            return {}
        route_key = "pro.video.wan.fp8-pair"
    elif high_suffix == ".gguf" and low_suffix == ".gguf":
        route_key = "pro.video.wan.gguf-pair"
    else:
        return {}
    descriptor = _setup_route_descriptor({"routeKey": route_key})
    if descriptor is None:
        return {}
    return {"setupBundleKey": descriptor["setupBundleKey"], "setupRoute": descriptor}


def _wan_model_readiness(ctx: Any, service: Any, display: str, identifier: str) -> dict[str, Any]:
    """Only call a Wan model Ready after validating its active route dependencies.

    The inventory item alone cannot prove that Wan's shared components, VAE,
    optional text encoder, and route-specific transformer selection are usable.
    For dual-transformer modes, preflight the configured pair and only apply
    that result to either member of that pair.
    """
    unknown = {
        "status": "Installed",
        "routeStatus": "unknown",
        "readinessReason": "Installed Wan weights are not verified for the active runtime route.",
        "suggestedAction": "Select a compatible Wan route and run its readiness check.",
    }
    preflight = getattr(service, "preflight", None)
    if not callable(preflight):
        return unknown
    settings = getattr(ctx, "settings", None)
    runtime_mode = _canonical_wan_runtime_mode(getattr(settings, "last_wan_runtime_mode", "fast_5b"))
    try:
        from aiwf.core.domain.wan import WanI2VRequest

        common = {
            "runtime_mode": runtime_mode,
            "vae_id": str(getattr(settings, "last_wan_vae", "") or ""),
            "text_encoder_path": str(getattr(settings, "last_wan_text_encoder", "") or ""),
            "offload": str(getattr(settings, "last_wan_offload", "balanced") or "balanced"),
        }
        if runtime_mode == "fast_5b":
            # A fast-route preflight against a 14B high/low file would report a
            # misleading incompatibility, so defer that file until its pair is
            # selected in the matching route.
            model_label = f"{display} {identifier}".casefold()
            if not any(token in model_label for token in ("5b", "ti2v", "wan diffusers")):
                return unknown
            request = WanI2VRequest(model_id=identifier, **common)
        else:
            high = str(getattr(settings, "last_wan_high", "") or "")
            low = str(getattr(settings, "last_wan_low", "") or "")
            if not high or not low or identifier not in {high, low}:
                return unknown
            request = WanI2VRequest(high_noise_model_id=high, low_noise_model_id=low, **common)
        result = preflight(request, image_present=True)
        if bool(getattr(result, "ok", False)):
            return {"status": "Ready", "routeStatus": "request-eligible"}
        errors = [str(error) for error in (getattr(result, "errors", ()) or ())]
        reason = "; ".join(errors[:4]) or "Wan route preflight did not confirm readiness."
        return {
            "status": "missing-assets",
            "routeStatus": "blocked",
            "readinessReason": reason,
            "suggestedAction": "Install or select the missing Wan transformer, components, VAE, or encoder, then refresh readiness.",
        }
    except Exception as exc:
        return {
            **unknown,
            "readinessReason": f"Wan route readiness could not be checked: {exc}",
        }


def _wan_model_ids(ctx: Any) -> set[str]:
    return {str(item.get("id") or "") for item in _wan_model_payloads(ctx)}


def _video_model_ids(ctx: Any) -> set[str]:
    ids = _wan_model_ids(ctx)
    try:
        ids.update(str(model.get("id") or "") for model in _sana_video_model_payloads(ctx) if model.get("id"))
    except Exception:
        pass
    ids.update(_ltx_model_ids())
    return ids


_LTX_PIPELINES = {
    "ltx:diffusers_2b": LTX_PIPELINE_DIFFUSERS_2B,
    "ltx:distilled": LTX_PIPELINE_DISTILLED,
    "ltx:one_stage": LTX_PIPELINE_ONE_STAGE,
}


def _ltx_model_ids() -> set[str]:
    return set(_LTX_PIPELINES)


def _ltx_model_payload(ctx: Any, model_id: str) -> dict[str, Any]:
    from aiwf.services.pipeline_preflight import preflight_ltx_pipeline

    pipeline = _LTX_PIPELINES[model_id]
    service = _ltx_service(ctx)
    checkpoint_resolver = getattr(service, "default_checkpoint_path", None)
    checkpoint_path = str(checkpoint_resolver(pipeline)) if callable(checkpoint_resolver) else ""
    request = LtxVideoRequest(
        pipeline=pipeline,
        checkpoint_path=checkpoint_path,
        gemma_backend=LTX_GEMMA_BACKEND_HF_SAFETENSORS,
    )
    try:
        preflight = preflight_ltx_pipeline(ctx.flags, getattr(ctx, "settings", None), request=request)
        missing = [item for item in preflight.items if not item.ok]
        ready = bool(preflight.ok)
        reason = "; ".join(item.message for item in missing) or "; ".join(preflight.warnings)
        if not reason and not ready:
            reason = preflight.message()
        bundle = {
            LTX_PIPELINE_DIFFUSERS_2B: "ltx-2b",
            LTX_PIPELINE_DISTILLED: "ltx23",
            LTX_PIPELINE_ONE_STAGE: "ltx23-one-stage",
        }[pipeline]
        setup_route = _setup_route_descriptor({
            "routeKey": {
                LTX_PIPELINE_DIFFUSERS_2B: "pro.video.ltx.diffusers-2b",
                LTX_PIPELINE_DISTILLED: "pro.video.ltx.distilled",
                LTX_PIPELINE_ONE_STAGE: "pro.video.ltx.one-stage",
            }[pipeline],
            "path": checkpoint_path,
        })
        if setup_route and setup_route.get("setupBundleKey"):
            bundle = str(setup_route["setupBundleKey"])
        runtime_blocked = bool(setup_route and setup_route.get("supportState") == "blocked-runtime")
        if runtime_blocked:
            ready = False
            reason = str(setup_route.get("limitation") or "This LTX route is blocked by the current runtime.")
        payload = {
            "id": model_id,
            "title": {
                LTX_PIPELINE_DIFFUSERS_2B: "LTX Video 0.9.5 2B",
                LTX_PIPELINE_DISTILLED: "LTX 2.3 Distilled",
                LTX_PIPELINE_ONE_STAGE: "LTX 2.3 One Stage",
            }[pipeline],
            "filename": Path(preflight.metadata.get("checkpoint_path") or "").name,
            "path": preflight.metadata.get("checkpoint_path", ""),
            "architecture": "ltx",
            "kind": "video",
            "engineId": "ltx",
            "engineLabel": _ENGINE_LABELS["ltx"],
            "backend": "LTX worker" if pipeline != LTX_PIPELINE_DIFFUSERS_2B else "Diffusers",
            "status": "Ready" if ready else "missing-assets" if any("missing" in item.message.lower() or "incomplete" in item.message.lower() for item in missing) else "blocked-runtime",
            "routeStatus": "request-eligible" if ready else "blocked",
            "setupBundleKey": None if ready or runtime_blocked else bundle,
            "setupRoute": setup_route,
            "pipeline": pipeline,
            "generationPreset": {
                "width": 512 if pipeline == LTX_PIPELINE_DIFFUSERS_2B else 832,
                "height": 320 if pipeline == LTX_PIPELINE_DIFFUSERS_2B else 480,
                "steps": 1 if pipeline == LTX_PIPELINE_DIFFUSERS_2B else 4,
                "frames": 9,
                "fps": 8,
            },
        }
        if not ready:
            payload["reason"] = reason
            payload["readinessReason"] = reason
            engine_missing = any(item.name == "engine worker" and not item.ok for item in missing)
            if runtime_blocked:
                payload["suggestedAction"] = (
                    "Choose the LTX 2B Diffusers route or select the supported FP8 one-stage checkpoint. "
                    "Installing the BF16 one-stage bundle will not remove this Windows runtime block."
                )
            else:
                payload["suggestedAction"] = (
                    f"Install or enable the LTX 2.3 worker from Gradio Settings, complete the {bundle} model assets, then refresh readiness."
                    if engine_missing and pipeline != LTX_PIPELINE_DIFFUSERS_2B
                    else f"Complete the {bundle} setup, then refresh model readiness."
                )
        return payload
    except Exception as exc:
        logger.debug("LTX Pro preflight failed for %s", model_id, exc_info=True)
        return {
            "id": model_id,
            "title": f"LTX {pipeline.replace('_', ' ').title()}",
            "filename": "",
            "path": "",
            "architecture": "ltx",
            "kind": "video",
            "engineId": "ltx",
            "engineLabel": _ENGINE_LABELS["ltx"],
            "status": "blocked-runtime",
            "routeStatus": "blocked",
            "setupBundleKey": None,
            "setupRoute": _setup_route_descriptor({"routeKey": {
                LTX_PIPELINE_DIFFUSERS_2B: "pro.video.ltx.diffusers-2b",
                LTX_PIPELINE_DISTILLED: "pro.video.ltx.distilled",
                LTX_PIPELINE_ONE_STAGE: "pro.video.ltx.one-stage",
            }[pipeline]}),
            "reason": f"LTX readiness could not be verified: {exc}",
            "suggestedAction": "Check the LTX worker/runtime and local assets, then refresh model readiness.",
            "pipeline": pipeline,
        }


def _ltx_model_payloads(ctx: Any) -> list[dict[str, Any]]:
    return [_ltx_model_payload(ctx, model_id) for model_id in _LTX_PIPELINES]


def _sana_video_model_payload(ctx: Any, variant: str = "480p") -> dict[str, Any]:
    variant = "720p" if str(variant).strip().lower() == "720p" else "480p"
    title = f"SANA-Video 2B {variant}"
    folder_name = f"SANA-Video_2B_{variant}_diffusers"
    if not _pro_sana_video_backend_enabled():
        return {
            "id": "",
            "title": title,
            "filename": folder_name,
            "modelVariant": variant,
            "hash": None,
            "kind": "video",
            "architecture": "sana_video",
            "engineId": "sana_video",
            "engineLabel": _ENGINE_LABELS["sana_video"],
            "backend": "Diffusers",
            "status": "Disabled",
            "setupRoute": _setup_route_descriptor({"routeKey": "pro.video.sana-video.720p" if variant == "720p" else "pro.video.sana-video.480p"}),
        }
    try:
        service = _sana_video_service(ctx)
        model_path = Path(service.default_model_path(variant)).expanduser().resolve()
        from aiwf.infrastructure.diffusers.checkpoints import sana_video_missing_local_files

        missing = sana_video_missing_local_files(model_path)
        text_to_video_runtime_ready = bool(service.runtime_available())
        image_to_video_runtime_ready = bool(service.runtime_available(image_to_video=True))
    except Exception:
        model_path = Path("models/sana-video/Diffusers") / folder_name
        missing = [model_path / "model_index.json"]
        text_to_video_runtime_ready = False
        image_to_video_runtime_ready = False
    text_to_video_ready = not missing and text_to_video_runtime_ready
    image_to_video_ready = not missing and image_to_video_runtime_ready
    # The picker represents the model as a route choice for either video
    # mode. Keep it selectable when at least one mode is usable; mode-specific
    # preflight still rejects the unavailable mode at generation time.
    ready = text_to_video_ready or image_to_video_ready
    payload = {
        "id": str(model_path),
        "title": title,
        "filename": model_path.name,
        "modelVariant": variant,
        "hash": None,
        "kind": "video",
        "architecture": "sana_video",
        "engineId": "sana_video",
        "engineLabel": _ENGINE_LABELS["sana_video"],
        "backend": "Diffusers",
        "status": "Ready" if ready else "missing-assets" if missing else "blocked-runtime",
        "routeStatus": "request-eligible" if ready else "blocked",
        "generationModes": {
            "textToVideo": text_to_video_ready,
            "imageToVideo": image_to_video_ready,
        },
        "setupBundleKey": f"sana-video{('-' + variant) if variant == '720p' else ''}" if missing else None,
        "setupRoute": _setup_route_descriptor({"routeKey": f"pro.video.sana-video.{variant}"}),
        "checkpointPathStatus": "present" if model_path.is_dir() else "missing",
    }
    if not ready:
        payload["readinessReason"] = (
            "Sana Video snapshot is incomplete: " + ", ".join(str(path) for path in missing[:8])
            if missing
            else "Sana Video pipeline classes are unavailable in the installed Diffusers runtime."
        )
        payload["reason"] = payload["readinessReason"]
        payload["suggestedAction"] = (
            "Install the Sana Video setup from Model Setup, then refresh model readiness."
            if missing
            else "The Sana Video model files are already present. Enable or repair the installed Diffusers runtime for Sana Video, then refresh model readiness; reinstalling the model files is not needed."
        )
    return payload


def _sana_video_model_payloads(ctx: Any) -> list[dict[str, Any]]:
    return [_sana_video_model_payload(ctx, variant) for variant in ("480p", "720p")]


def _ltx_service(ctx: Any):
    service = getattr(ctx, "ltx", None)
    if service is not None:
        return service
    key = id(ctx)
    service = _LTX_SERVICES.get(key)
    if service is None:
        from aiwf.services.ltx import LtxService

        service = LtxService(
            getattr(ctx, "flags", None),
            getattr(ctx, "settings", None),
            supervisor=getattr(ctx, "supervisor", None),
        )
        _LTX_SERVICES[key] = service
    return service


def _refresh_ltx_worker_registry(ctx: Any) -> None:
    service = getattr(ctx, "ltx", None) or _LTX_SERVICES.get(id(ctx))
    registry = getattr(service, "registry", None)
    repo_root = getattr(registry, "repo_root", None)
    if repo_root is None:
        return
    from aiwf.services.worker_tenant import WorkerTenantRegistry

    service.registry = WorkerTenantRegistry(repo_root)


def _ltx_video_request_from_payload(ctx: Any, payload: ProGeneratePayload) -> LtxVideoRequest:
    model_id = str(payload.checkpoint_id or f"ltx:{payload.ltx_pipeline}")
    pipeline = _LTX_PIPELINES.get(model_id)
    if pipeline is None:
        raise HTTPException(status_code=422, detail="Choose an LTX pipeline from the video model list.")
    if payload.ltx_offload not in LTX_OFFLOAD_MODES:
        raise HTTPException(status_code=422, detail=f"LTX offload must be one of {LTX_OFFLOAD_MODES}.")
    quantization = "" if payload.ltx_quantization == "none" else payload.ltx_quantization
    if quantization not in LTX_QUANTIZATION_MODES:
        raise HTTPException(status_code=422, detail="LTX quantization must be empty, fp8-cast, or fp8-scaled-mm.")
    source_image_path = _video_source_image_path(ctx, payload)
    service = _ltx_service(ctx)
    checkpoint_resolver = getattr(service, "default_checkpoint_path", None)
    checkpoint_path = str(checkpoint_resolver(pipeline)) if callable(checkpoint_resolver) else ""
    try:
        return LtxVideoRequest(
            prompt=payload.prompt,
            negative_prompt=payload.negative_prompt,
            source_image_path=source_image_path,
            pipeline=pipeline,
            checkpoint_path=checkpoint_path,
            width=payload.width,
            height=payload.height,
            num_frames=payload.frames,
            fps=payload.fps,
            seed=payload.seed,
            steps=min(int(payload.steps), 100),
            image_strength=payload.ltx_image_strength,
            offload=payload.ltx_offload,
            quantization=quantization,
            enhance_prompt=payload.ltx_enhance_prompt,
            gemma_backend=LTX_GEMMA_BACKEND_HF_SAFETENSORS,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc


def _ltx_video_output_payload(ctx: Any, result: Any, payload: ProGeneratePayload) -> dict[str, Any]:
    raw_path = str(getattr(result, "output_path", "") or "")
    path = Path(raw_path).expanduser()
    root = _safe_output_root(ctx)
    try:
        resolved = path.resolve()
    except OSError as exc:
        raise HTTPException(status_code=500, detail="LTX returned an unreadable output path.") from exc
    if root is None or not _path_inside(resolved, root) or not resolved.is_file():
        raise HTTPException(status_code=500, detail="LTX did not return a video file inside the configured output directory.")
    created_at = datetime.fromtimestamp(resolved.stat().st_mtime, timezone.utc).isoformat()
    output = {
        "id": f"ltx-video-{resolved.stem or 'output'}",
        "url": _output_asset_url(ctx, resolved),
        "thumbnailUrl": _output_asset_url(ctx, resolved),
        "path": str(resolved),
        "prompt": payload.prompt,
        "negativePrompt": payload.negative_prompt,
        "infotext": "",
        "width": payload.width,
        "height": payload.height,
        "createdAt": created_at,
        "mode": "video",
        "seed": payload.seed,
        "steps": payload.steps,
        "cfgScale": payload.cfg_scale,
        "clipSkip": payload.clip_skip,
        "sampler": payload.sampler,
        "scheduler": payload.scheduler,
        "modelName": payload.checkpoint_id or "ltx:one_stage",
        "status": "completed",
        "source": "ltx",
    }
    return output


def _generate_ltx_video_response(ctx: Any, payload: ProGeneratePayload) -> dict[str, Any]:
    request = _ltx_video_request_from_payload(ctx, payload)
    route_id = f"video.ltx.{request.pipeline}"
    from aiwf.services.pipeline_preflight import preflight_ltx_pipeline

    try:
        preflight = preflight_ltx_pipeline(ctx.flags, getattr(ctx, "settings", None), request=request)
    except Exception as exc:
        preflight = None
        preflight_error = f"LTX route preflight failed: {exc}"
    else:
        preflight_error = ""
    support_ids = [request.pipeline, str(payload.checkpoint_id or "")]
    if preflight is not None:
        support_ids.extend(
            str(item.path)
            for item in getattr(preflight, "items", ())
            if getattr(item, "path", None) is not None
        )
    if preflight is None or not preflight.ok:
        detail = preflight_error or str(
            getattr(preflight, "markdown", lambda: "LTX route preflight did not pass.")()
        )
        select_route(
            ctx, route=route_id, model_id=payload.checkpoint_id or request.pipeline,
            setup_ready=False, support_ids=support_ids, detail=detail,
        )
        missing_assets = preflight is not None and any(
            not item.ok and item.path is not None for item in getattr(preflight, "items", ())
        )
        raise HTTPException(status_code=422 if missing_assets else 503, detail=detail)
    _release_image_model_for_video_route(ctx)
    if request.pipeline != LTX_PIPELINE_DIFFUSERS_2B:
        _unload_cached_ltx_model(ctx)
    service = _ltx_service(ctx)
    begin_generation = getattr(service, "begin_generation", None)
    if callable(begin_generation):
        begin_generation()
    job_id = _pro_video_job_start(
        ctx, request, message="Starting LTX video generation.", cancel_target="ltx"
    )
    token = begin_route_operation(
      ctx, route=route_id, model_id=payload.checkpoint_id or request.pipeline,
      setup_ready=True, support_ids=support_ids, detail="LTX route and support assets passed preflight; preparing generation."
    )
    mark_route_running(ctx, route_id, token, "LTX generation is running.")
    try:
        def _on_progress(event: dict[str, Any]) -> None:
            kind = str(event.get("kind") or "").lower()
            message = str(event.get("message") or event.get("status") or "LTX generation is running.")
            if kind == "progress":
                _pro_video_job_update(
                    ctx, job_id,
                    progress=float(event.get("progress") or 0),
                    message=message,
                    step=int(event.get("step") or 0),
                    total=int(event.get("total") or event.get("total_steps") or 0),
                )
            elif kind == "status":
                _pro_video_job_update(ctx, job_id, progress=0, message=message)

        result = service.generate(request, on_progress=_on_progress)
        output = _ltx_video_output_payload(ctx, result, payload)
        events = list(getattr(result, "events", None) or [])
        message = str(getattr(result, "message", "") or "LTX video complete.")
        response = {
            "jobId": job_id,
            "status": "completed",
            "output": output,
            "video": output["url"],
            "recentOutputs": [output],
            "progress": events,
            "timings": {},
            "hasAudio": bool(getattr(result, "has_audio", False)),
            "audioMode": str(getattr(result, "audio_mode", "none") or "none"),
            "message": message,
        }
    except Exception as exc:
        cancelled = isinstance(exc, GenerationCancelledError) or _pro_video_cancel_requested(ctx, job_id)
        if cancelled:
            finish_route_operation(ctx, route_id, token, success=False, detail="LTX generation cancelled.")
            _pro_video_job_finish(ctx, job_id, "cancelled", message="LTX generation cancelled.")
            raise HTTPException(status_code=499, detail="LTX generation cancelled.") from exc
        finish_route_operation(ctx, route_id, token, success=False, detail=str(exc))
        _pro_video_job_finish(ctx, job_id, "failed", message=str(exc), error=str(exc))
        from aiwf.services.ltx import LtxUnavailable

        status_code = 503 if isinstance(exc, LtxUnavailable) else 500
        logger.exception("Pro LTX video generation failed: job=%s pipeline=%s", job_id, request.pipeline)
        raise HTTPException(status_code=status_code, detail=_pro_error_detail(str(exc), job=_pro_video_job_status(ctx))) from exc

    finish_route_operation(ctx, route_id, token, success=True, detail="LTX output file verified.")
    _pro_video_job_finish(ctx, job_id, "completed", message=message)
    return response


def _preflight_support_paths(preflight: Any) -> list[str]:
    """Include every locally resolved support asset in route revision tracking."""
    paths = [
        str(item.path)
        for item in getattr(preflight, "items", ())
        if getattr(item, "path", None) is not None
    ]
    # Wan preflight returns resolved assets as named dataclass fields instead
    # of the generic ``items`` shape used by image/video pipeline preflights.
    for field in (
        "model_id",
        "components_base",
        "high_noise_model",
        "low_noise_model",
        "vae",
        "text_encoder",
        "high_noise_lora",
        "low_noise_lora",
    ):
        value = getattr(preflight, field, None)
        if value:
            paths.append(str(value))
    return list(dict.fromkeys(paths))


def _sana_optional_audio_readiness(ctx: Any, request: SanaVideoRequest) -> tuple[bool, str, list[str]]:
    """Preflight optional MMAudio before claiming an audio-enabled Sana route ready."""
    if not request.generate_audio:
        return True, "", []
    model_id = str(request.audio_model_id or "")
    if not model_id.startswith("mmaudio:"):
        return False, "Sana video audio requires an installed MMAudio variant.", [model_id]
    variant = model_id.split(":", 1)[1]
    service = _audio_service(ctx)
    support_ids = [model_id]
    root_resolver = getattr(service, "_mmaudio_root", None)
    if callable(root_resolver):
        try:
            support_ids.append(str(root_resolver()))
        except Exception:
            pass
    cache_resolver = getattr(service, "_mmaudio_clip_hub_cache", None)
    if callable(cache_resolver):
        try:
            cache_path = cache_resolver()
            if cache_path is not None:
                support_ids.append(str(cache_path))
        except Exception:
            pass
    try:
        if not service._mmaudio_variant_ready(variant):
            return False, f"Sana video audio requires the complete MMAudio {variant} model bundle.", support_ids
        runtime_error = service._mmaudio_runtime_import_error()
        if runtime_error:
            return False, f"Sana video audio runtime is not ready: {runtime_error}", support_ids
        setup = service.setup_status(deep=False)
        if not setup.get("muxReady"):
            return False, "Sana video audio requires working FFmpeg and ffprobe to mux the soundtrack into the video.", support_ids
    except Exception as exc:
        return False, f"Sana video audio readiness could not be verified: {exc}", support_ids
    return True, "", support_ids


def _persist_last_checkpoint_id(ctx: Any, model_id: str) -> bool:
    """Persist only a successfully prepared/loaded route as the next startup default."""
    settings = getattr(ctx, "settings", None)
    if settings is None or not model_id:
        return False
    save_settings = getattr(ctx, "save_settings", None)
    if not callable(save_settings):
        return False
    previous = getattr(settings, "last_checkpoint_id", None)
    setattr(settings, "last_checkpoint_id", model_id)
    try:
        save_settings()
    except Exception:
        setattr(settings, "last_checkpoint_id", previous)
        logger.exception("Could not persist selected model %s as the startup default", model_id)
        return False
    return True


def _persist_last_audio_model_id(ctx: Any, model_id: str, *, kind: str, video_lab: bool = False) -> bool:
    """Remember a successfully prepared audio choice for the matching workspace."""
    settings = getattr(ctx, "settings", None)
    if settings is None or not model_id:
        return False
    field = "last_video_audio_model_id" if video_lab else (
        "last_audio_music_model_id" if kind == "music" else "last_audio_sfx_model_id"
    )
    previous = getattr(settings, field, "")
    previous_sana_audio = getattr(settings, "last_sana_audio_model_id", "")
    setattr(settings, field, model_id)
    if video_lab and model_id.startswith("mmaudio:"):
        setattr(settings, "last_sana_audio_model_id", model_id)
    save_settings = getattr(ctx, "save_settings", None)
    if not callable(save_settings):
        setattr(settings, field, previous)
        setattr(settings, "last_sana_audio_model_id", previous_sana_audio)
        return False
    try:
        save_settings()
    except Exception:
        setattr(settings, field, previous)
        setattr(settings, "last_sana_audio_model_id", previous_sana_audio)
        logger.exception("Could not persist selected audio model %s", model_id)
        return False
    return True


def _sana_video_audio_model_id(settings: Any) -> str:
    explicit = str(getattr(settings, "last_sana_audio_model_id", "") or "").strip()
    if explicit.startswith("mmaudio:"):
        return explicit
    legacy_video_choice = str(getattr(settings, "last_video_audio_model_id", "") or "").strip()
    if legacy_video_choice.startswith("mmaudio:"):
        return legacy_video_choice
    return "mmaudio:small_16k"


def _is_known_video_prepare_deferral(route_id: str, exc: Exception) -> bool:
    """Keep setup-ready only for route-specific, documented resource deferrals."""
    detail = str(exc).strip().casefold()
    if route_id.startswith("video.sana."):
        from aiwf.services.sana_video import SanaVideoUnavailable

        return isinstance(exc, SanaVideoUnavailable) and detail.startswith("sana video preparation deferred")
    if route_id.startswith("video.wan."):
        from aiwf.infrastructure.wan import WanUnavailable

        return isinstance(exc, WanUnavailable) and detail.startswith("wan preparation deferred")
    if route_id.startswith("video.ltx."):
        from aiwf.services.ltx import LtxUnavailable

        return isinstance(exc, LtxUnavailable) and detail.startswith((
            "ltx worker launch deferred",
            "ltx 2b loading deferred",
        ))
    return False


def _prepare_pro_video_route(ctx: Any, payload: ProGeneratePayload) -> dict[str, Any]:
    """Preflight the selected video route and load it when its backend supports residency."""
    checkpoint_id = str(payload.checkpoint_id or "")
    route_id = ""
    model_id = checkpoint_id
    support_ids: list[str] = []
    preflight = None
    detail = ""
    loaded = False
    resident: bool | None = None
    preparation_failed = False
    ready = False
    try:
        if checkpoint_id in _LTX_PIPELINES:
            request = _ltx_video_request_from_payload(ctx, payload)
            route_id = f"video.ltx.{request.pipeline}"
            model_id = checkpoint_id or request.pipeline
            support_ids = [request.pipeline, checkpoint_id]
            from aiwf.services.pipeline_preflight import preflight_ltx_pipeline

            preflight = preflight_ltx_pipeline(ctx.flags, getattr(ctx, "settings", None), request=request)
        elif (payload.mode or "").strip().lower() == "wan" or checkpoint_id in _wan_model_ids(ctx):
            request = _wan_video_request_from_payload(ctx, payload)
            route_id = f"video.wan.{request.runtime_mode}"
            model_id = request.model_id or request.runtime_mode
            support_ids = [
                request.runtime_mode,
                request.vae_id or "",
                request.text_encoder_path or "",
                request.high_noise_model_id or "",
                request.low_noise_model_id or "",
            ]
            service = _wan_service(ctx)
            preflight_fn = getattr(service, "preflight", None)
            if callable(preflight_fn):
                preflight = preflight_fn(
                    request,
                    image_present=bool(payload.source_image_path or payload.source_image_data_url),
                )
            else:
                detail = "Wan route readiness check is unavailable."
        else:
            request = _sana_video_request_from_payload(ctx, payload)
            route_id = f"video.sana.{request.model_variant}"
            model_id = request.model_path or request.model_variant
            support_ids = [request.model_variant, request.model_path or ""]
            if not _pro_sana_video_backend_enabled():
                detail = "React Pro Sana Video backend is disabled in aiwf/web/pro_api.py."
            else:
                from aiwf.services.pipeline_preflight import preflight_sana_video_pipeline

                preflight = preflight_sana_video_pipeline(
                    ctx.flags,
                    getattr(ctx, "settings", None),
                    request=request,
                )
        if preflight is not None:
            support_ids.extend(_preflight_support_paths(preflight))
            ready = bool(getattr(preflight, "ok", False))
            message = getattr(preflight, "markdown", None) or getattr(preflight, "message", None)
            detail = str(message() if callable(message) else message or "").strip()
            if not detail:
                detail = "Selected video route passed preflight." if ready else "Selected video route did not pass preflight."
            if ready and route_id.startswith("video.sana.") and request.generate_audio:
                audio_ready, audio_detail, audio_support_ids = _sana_optional_audio_readiness(ctx, request)
                support_ids.extend(audio_support_ids)
                if not audio_ready:
                    ready = False
                    resident = None
                    detail = audio_detail
            if ready:
                if route_id.startswith("video.ltx."):
                    _release_image_model_for_video_route(ctx)
                    if request.pipeline != LTX_PIPELINE_DIFFUSERS_2B:
                        _unload_cached_ltx_model(ctx)
                else:
                    _unload_cached_ltx_model(ctx)
                _release_cached_audio_model(ctx)
                if not route_id.startswith("video.wan."):
                    _release_cached_wan_model(ctx)
                if not route_id.startswith("video.sana."):
                    _release_cached_sana_video(ctx)
            if ready and route_id.startswith("video.wan."):
                try:
                    prepared = service.prepare(
                        request,
                        image_present=bool(payload.source_image_path or payload.source_image_data_url),
                        preflight=preflight,
                    )
                except Exception:
                    has_cached = getattr(getattr(service, "_backend", None), "has_cached_pipeline", None)
                    try:
                        previous_pipeline_evicted = callable(has_cached) and not bool(has_cached())
                    except Exception:
                        previous_pipeline_evicted = False
                    if previous_pipeline_evicted:
                        _clear_prior_video_family_residency(ctx, route_id)
                    raise
                _clear_prior_video_family_residency(ctx, route_id)
                resident_value = prepared.get("resident")
                resident = resident_value if isinstance(resident_value, bool) else None
                loaded = bool(prepared.get("loaded")) and resident is True
                if not loaded and resident is True:
                    resident = None
                detail = str(prepared.get("detail") or detail)
            if ready and route_id.startswith("video.ltx."):
                prepared = _ltx_service(ctx).prepare(request)
                loaded = bool(prepared.get("loaded")) and prepared.get("resident") is True
                resident_value = prepared.get("resident")
                resident = resident_value if isinstance(resident_value, bool) else None
                detail = str(prepared.get("detail") or detail)
            if ready and route_id.startswith("video.sana."):
                _clear_prior_video_family_residency(ctx, route_id)
                prepared = _sana_video_service(ctx).prepare(request)
                loaded = bool(prepared.get("loaded"))
                resident = loaded
                if loaded:
                    detail = (
                        f"Sana Video pipeline loaded and resident ({prepared.get('quantization') or 'default'}, "
                        f"attention={prepared.get('attentionBackend') or 'native'}). No generation was run."
                    )
                    for prior in lifecycle_snapshot(ctx):
                        prior_route = str(prior.get("route") or "")
                        if prior_route.startswith("video.sana.") and prior_route != route_id and prior.get("resident") is True:
                            confirm_route_residency(
                                ctx,
                                prior_route,
                                str(prior.get("modelId") or ""),
                                resident=False,
                            )
        else:
            ready = False
    except HTTPException:
        raise
    except Exception as exc:
        preflight_passed = bool(getattr(preflight, "ok", False))
        deferred = preflight_passed and _is_known_video_prepare_deferral(route_id, exc)
        ready = deferred
        preparation_failed = preflight_passed and not deferred
        resident = None
        if deferred:
            detail = str(exc)
        elif preflight_passed:
            detail = f"Video route preparation failed after setup checks: {exc}"
        else:
            detail = f"Selected video route readiness check failed: {exc}"

    if not route_id:
        raise HTTPException(status_code=422, detail="Choose a supported Wan, Sana Video, or LTX route before preparing it.")
    lifecycle = select_route(
        ctx,
        route=route_id,
        model_id=model_id,
        setup_ready=ready,
        support_ids=support_ids,
        detail=detail,
        resident=resident,
        failed=preparation_failed,
    )
    default_saved = _persist_last_checkpoint_id(ctx, checkpoint_id) if ready and checkpoint_id else False
    return {
        "ready": ready,
        "routeLifecycle": lifecycle,
        "loaded": loaded,
        "startupDefaultSaved": default_saved,
    }


def _assert_ltx_engine_not_installing() -> None:
    """Keep LTX worker setup from racing selected-route preparation or jobs."""
    status = ltx_engine_install_status()
    if bool(status.get("running")) or str(status.get("status") or "").strip().lower() == "running":
        raise HTTPException(
            status_code=409,
            detail="LTX engine setup is in progress. Wait for setup to finish before preparing or generating with LTX.",
        )


def _assert_qwen_nunchaku_engine_not_installing(ctx: Any) -> None:
    """Prevent Qwen Nunchaku weight/runtime use while its environment is changing."""
    status = qwen_nunchaku_engine_install_status(ctx.flags.data_dir)
    if bool(status.get("running")) or str(status.get("status") or "").strip().lower() == "running":
        raise HTTPException(
            status_code=409,
            detail="Qwen Nunchaku setup is in progress. Wait for setup to finish before loading or generating with Qwen Image 2.1.",
        )


def _clear_prior_video_family_residency(ctx: Any, active_route: str) -> None:
    """Invalidate residency receipts for sibling variants before a switch.

    Wan and Sana unload their prior cached pipeline before attempting a new
    variant. Clear the old receipt before that attempt so a failed replacement
    cannot leave the old route looking resident.
    """
    family_prefix = "video.wan." if active_route.startswith("video.wan.") else "video.sana."
    for prior in lifecycle_snapshot(ctx):
        prior_route = str(prior.get("route") or "")
        if prior_route.startswith(family_prefix) and prior_route != active_route and prior.get("resident") is True:
            confirm_route_residency(
                ctx,
                prior_route,
                str(prior.get("modelId") or ""),
                resident=False,
            )


def _require_loopback(request: Request) -> None:
    host = request.client.host if request.client else ""
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise HTTPException(
            status_code=403,
            detail="Mobile pairing details are only available from the local machine.",
        )


def _local_lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return str(sock.getsockname()[0])
    except OSError:
        return "127.0.0.1"


def _mobile_pairing_payload(ctx: Any, state: dict[str, Any]) -> dict[str, Any]:
    port = int(getattr(ctx, "runtime_port", 0) or getattr(getattr(ctx, "flags", None), "port", 0) or 0)
    enabled = bool(state.get("enabled")) and bool(state.get("token"))
    pairing_uri = (
        f"aiwf://pair?host={_local_lan_ip()}&port={port}&token={state.get('token', '')}&name=AIWF"
        if enabled
        else ""
    )
    return {
        "enabled": enabled,
        "token": str(state.get("token") or ""),
        "port": port,
        "pairingUri": pairing_uri,
    }


def _all_output_image_paths_from_disk(root: Path, *, scan_limit: int) -> list[Path]:
    if not root.exists():
        return []
    rows: list[tuple[float, Path]] = []
    inspected = 0
    stack = [root]
    while stack and inspected < scan_limit:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            if inspected >= scan_limit:
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
            rows.append((stat.st_mtime, entry))
    rows.sort(key=lambda row: row[0], reverse=True)
    return [row[1] for row in rows]


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
        "modelLoad": dict(getattr(ctx, "_pro_model_load_state", {}) or {"status": "not-started", "modelId": "", "detail": ""}),
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
            "bundles": quick_start_bundles_for_platform(),
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
        shared_snapshot_available = False
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
            if item.snapshot:
                try:
                    primary_target = service.snapshot_destination_for(item.category, item.repo_id)
                    primary_ready = bool(service._catalog_snapshot_ready(item, primary_target))
                    shared_snapshot_available = (
                        not primary_ready and bool(service._find_shared_catalog_snapshot(item))
                    )
                except Exception:
                    shared_snapshot_available = False
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
                "platformBlocked": bool(os.name == "nt" and item.key in {"fluxtrait-zimage-v2-q4", "fluxtrait-zimage-v2-q8"}),
                "platformBlockReason": (
                    "Z-Image GGUF loading is currently blocked on Windows; install the BF16 transformer bundle instead."
                    if os.name == "nt" and item.key in {"fluxtrait-zimage-v2-q4", "fluxtrait-zimage-v2-q8"} else ""
                ),
                "snapshot": item.snapshot,
                "installed": installed,
                "sharedSnapshotAvailable": shared_snapshot_available,
                "destination": destination,
                "engineId": engine_id,
                "engineLabel": _ENGINE_LABELS.get(engine_id, _ENGINE_LABELS["unknown"]),
                "catalogUrl": _catalog_entry_page_url(item),
                "hfUrl": _catalog_entry_hf_url(item),
                "requiresAuth": requires_auth,
                "canDownload": can_download,
                "comingSoon": bool(getattr(item, "coming_soon", False)),
            }
        )

    return {
        "categories": categories,
            "bundles": quick_start_bundles_for_platform(),
        "catalog": catalog,
        "civitaiLinks": list(CIVITAI_BROWSE_LINKS),
        "counts": {
            "categories": len(categories),
            "catalog": len(catalog),
            "installed": installed_count,
        },
    }


def _catalog_entry_page_url(item: Any) -> str:
    """Human-visitable source page for a catalog entry."""
    source = str(getattr(item, "source", "") or "").strip().lower()
    if source == "civitai":
        model_id = getattr(item, "civitai_model_id", None)
        version_id = getattr(item, "civitai_version_id", None)
        if model_id and version_id:
            return f"https://civitai.com/models/{model_id}?modelVersionId={version_id}"
        if model_id:
            return f"https://civitai.com/models/{model_id}"
        return ""
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


def _catalog_entry_hf_url(item: Any) -> str:
    """Backwards-compatible Hugging Face-only source page field."""
    source = str(getattr(item, "source", "") or "").strip().lower()
    if source != "huggingface":
        return ""
    return _catalog_entry_page_url(item)


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


def _catalog_entry_can_download(item: Any, *, has_hf_token: bool | None = None) -> bool:
    if bool(getattr(item, "coming_soon", False)):
        return False
    source = str(getattr(item, "source", "") or "").strip().lower()
    if source == "civitai":
        return bool(getattr(item, "civitai_model_id", None) or getattr(item, "civitai_version_id", None))
    if source not in {"huggingface", "direct"}:
        return False
    if _catalog_entry_requires_auth(item) and not (
        source == "huggingface"
        and (bool(_huggingface_token()) if has_hf_token is None else has_hf_token)
    ):
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
            "onnxModelDir": str(getattr(settings, "onnx_model_dir", "") or ""),
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
    model_roots_changed = False
    try:
        onnx_model_dir = _payload_value(runtime, "onnxModelDir", "onnx_model_dir")
        if onnx_model_dir is not _MISSING_SETTING:
            setattr(settings, "onnx_model_dir", _text_setting(onnx_model_dir))
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
                    if field in {"models_dir", "extra_model_dirs"}:
                        model_roots_changed = True
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

    if model_roots_changed:
        enhance_service = getattr(ctx, "enhance", None)
        invalidate = getattr(enhance_service, "invalidate_model_catalog", None)
        if callable(invalidate):
            invalidate()
        setattr(ctx, "_pro_capability_cache", None)

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
    sana_ready = sana_enabled and any(
        str(model.get("status") or "").lower() == "ready"
        for model in _sana_video_model_payloads(ctx)
    )
    ltx_models = _ltx_model_payloads(ctx)
    ltx_ready_count = sum(str(model.get("status") or "").lower() == "ready" for model in ltx_models)
    ltx_setup_count = len(ltx_models) - ltx_ready_count
    lora_count = _safe_count(ctx, "generation", "list_loras")
    controlnet_count = _safe_count(ctx, "controlnet", "list_models")
    controlnet_modules = _safe_count(ctx, "controlnet", "list_modules")
    sam_count = _safe_count(ctx, "segment", "list_models")
    enhance_service = getattr(ctx, "enhance", None)
    try:
        upscaler_models = list(enhance_service.list_upscalers()) if callable(getattr(enhance_service, "list_upscalers", None)) else []
    except Exception:
        upscaler_models = []
    try:
        restorer_models = list(enhance_service.list_restorers()) if callable(getattr(enhance_service, "list_restorers", None)) else []
    except Exception:
        restorer_models = []

    def installed_enhance_models(models):
        installed = 0
        for model in models:
            try:
                path = Path(str(model.path))
                installed += int(path.is_file() and path.stat().st_size > 0)
            except (AttributeError, OSError, TypeError, ValueError):
                continue
        return installed

    upscaler_count = installed_enhance_models(upscaler_models)
    restorer_count = installed_enhance_models(restorer_models)
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
            "status": "ready" if upscaler_count > 0 and restorer_count > 0 else "needs-assets",
            "count": upscaler_count + restorer_count,
            "route": "tools",
            "summary": "Pro can run quick face restore, upscale, and image VSR; Gradio keeps full old-photo and batch workflows.",
            "details": [
                f"{upscaler_count} of {len(upscaler_models)} upscalers installed",
                f"{restorer_count} of {len(restorer_models)} restorers installed",
                "Video VSR is available from Pro Video Lab.",
            ],
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
            "status": "ready" if sana_ready or ltx_ready_count or (wan_available and wan_count > 0) else "available",
            "count": wan_count + (1 if sana_ready else 0) + len(ltx_models),
            "route": "create",
            "summary": "Pro exposes Sana, Wan, and LTX generation; RIFE and post stages remain in Gradio.",
            "details": [
                "Sana ready" if sana_ready else ("Sana backend disabled" if not sana_enabled else "Sana snapshot missing"),
                f"{wan_count} Wan models",
                f"{ltx_ready_count} LTX pipelines preflight-ready",
                f"{ltx_setup_count} LTX pipelines need setup or runtime repair",
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
            checkpoint = resolver(checkpoint_id)
        except Exception:
            return None
        if checkpoint_id and checkpoint is not None:
            needle = checkpoint_id.strip().lower()
            data = _dump_model(checkpoint)
            candidates = (data.get("id"), data.get("title"), data.get("filename"))
            if not any(str(candidate or "").strip().lower() == needle for candidate in candidates):
                return None
        return checkpoint
    checkpoints = _safe_list(getattr(generation, "list_checkpoints", lambda: []))
    if checkpoint_id:
        needle = checkpoint_id.strip().lower()
        for checkpoint in checkpoints:
            data = _dump_model(checkpoint)
            candidates = (data.get("id"), data.get("title"), data.get("filename"))
            if any(str(candidate or "").lower() == needle for candidate in candidates):
                return checkpoint
    if checkpoint_id:
        return None
    return checkpoints[0] if checkpoints else None


def _assert_checkpoint_selectable(ctx: Any, checkpoint_id: str | None) -> None:
    checkpoint = _resolve_checkpoint_for_generation_guard(ctx, checkpoint_id)
    if checkpoint is None:
        if checkpoint_id:
            raise HTTPException(
                status_code=422,
                detail=_pro_error_detail(
                    "The selected model is not present in the current model inventory.",
                    checkpointId=checkpoint_id,
                    status="unavailable-model",
                    reason="The requested model ID could not be resolved to an installed checkpoint.",
                    suggestedAction="Refresh the model inventory and choose an installed model.",
                ),
            )
        return
    block = _blocked_checkpoint_detail(checkpoint) or _runtime_checkpoint_block(ctx, checkpoint)
    if block is None:
        block = _missing_checkpoint_detail(checkpoint)
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
    if checkpoint_id in _LTX_PIPELINES:
        return "ltx"
    if checkpoint_id in _video_model_ids(ctx):
        if checkpoint_id in _wan_model_ids(ctx):
            return "wan"
        if checkpoint_id in _ltx_model_ids():
            return "ltx"
        return "sana_video"
    checkpoint = _resolve_checkpoint_for_generation_guard(ctx, checkpoint_id)
    data = _dump_model(checkpoint) if checkpoint is not None else {}
    return _engine_id_for_architecture(str(data.get("architecture", "unknown")))


def _assert_image_route_checkpoint(ctx: Any, checkpoint_id: str | None) -> None:
    engine_id = _checkpoint_engine_id(ctx, checkpoint_id)
    if engine_id in {"sana_video", "wan", "ltx"}:
        raise HTTPException(
            status_code=422,
            detail=_pro_error_detail(
                "LTX models are only available with mode='video'." if engine_id == "ltx" else "Video models are only available from the Video tab.",
                checkpointId=str(checkpoint_id or ""),
            ),
        )


def _assert_video_route_checkpoint(
    ctx: Any,
    checkpoint_id: str | None,
    *,
    sana_model_variant: str = "480p",
) -> None:
    if not checkpoint_id:
        if not _pro_sana_video_backend_enabled():
            return
        default_model = _sana_video_model_payload(ctx, sana_model_variant)
        if str(default_model.get("status") or "").lower() != "ready":
            raise HTTPException(
                status_code=422,
                detail=_pro_error_detail(
                    "The default Sana Video model is not ready.",
                    checkpointId=str(default_model.get("id") or ""),
                    status=str(default_model.get("status") or "missing-assets"),
                    reason=str(default_model.get("reason") or "Sana Video local readiness could not be verified."),
                    suggestedAction=str(default_model.get("suggestedAction") or "Install the Sana Video setup and refresh model readiness."),
                ),
            )
        return
    engine_id = _checkpoint_engine_id(ctx, checkpoint_id)
    if engine_id not in {"sana_video", "wan", "ltx"}:
        raise HTTPException(
            status_code=422,
            detail=_pro_error_detail(
                "Selected model is an image model. Choose a Wan or Sana Video model, or an LTX pipeline, for video generation.",
                checkpointId=str(checkpoint_id),
            ),
        )
    if engine_id == "sana_video":
        checkpoint = _resolve_checkpoint_for_generation_guard(ctx, checkpoint_id)
        if checkpoint is not None:
            _assert_checkpoint_selectable(ctx, checkpoint_id)
            return
        default_model = _sana_video_model_payload(ctx)
        if checkpoint_id == str(default_model.get("id") or "") and str(default_model.get("status") or "").lower() != "ready":
            raise HTTPException(
                status_code=422,
                detail=_pro_error_detail(
                    "The selected Sana Video model is not ready.",
                    checkpointId=checkpoint_id,
                    status=str(default_model.get("status") or "missing-assets"),
                    reason=str(default_model.get("reason") or "Sana Video local readiness could not be verified."),
                    suggestedAction=str(default_model.get("suggestedAction") or "Install the Sana Video setup and refresh model readiness."),
                ),
            )
    if engine_id == "ltx":
        model = _ltx_model_payload(ctx, checkpoint_id)
        if str(model.get("status") or "").lower() != "ready":
            raise HTTPException(
                status_code=422,
                detail=_pro_error_detail(
                    "The selected LTX pipeline is not ready.",
                    checkpointId=checkpoint_id,
                    status=str(model.get("status") or "blocked-runtime"),
                    reason=str(model.get("reason") or "LTX local readiness could not be verified."),
                    suggestedAction=str(model.get("suggestedAction") or "Complete the LTX setup and refresh model readiness."),
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
            "id": "",
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
    status_check = getattr(backend, "is_checkpoint_loaded_for_status", None)
    if not callable(status_check):
        status_check = getattr(backend, "is_checkpoint_loaded", None)
    if callable(status_check):
        try:
            loaded = bool(status_check(checkpoint_id))
        except Exception:
            loaded = False
    architecture = str(data.get("architecture") or "unknown")
    return {
        "id": checkpoint_id,
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


def _assert_pro_model_load_idle(ctx: Any) -> None:
    state = getattr(ctx, "_pro_model_load_state", None)
    if isinstance(state, dict) and str(state.get("status") or "").strip().lower() in {"loading", "unloading"}:
        raise HTTPException(
            status_code=409,
            detail="An image model is still loading with its support assets. Wait for loading to finish, then retry.",
        )
    route_state = getattr(ctx, "_pro_route_prepare_state", None)
    if isinstance(route_state, dict) and route_state.get("status") == "preparing":
        raise HTTPException(
            status_code=409,
            detail="A generation route is being prepared. Wait for model switching to finish, then retry.",
        )


def _run_exclusive_pro_gpu_operation(ctx: Any, operation):
    """Serialize synchronous auxiliary GPU work against model switches/jobs."""
    from aiwf.services.model_startup import pro_model_load_lock

    lock = pro_model_load_lock(ctx)
    if not lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Another model operation is in progress. Retry when it finishes.")
    try:
        _assert_pro_model_load_idle(ctx)
        if _image_generation_running(ctx) or _image_generation_pending(ctx) or _pro_video_job_running(ctx) or _pro_workflow_runs_active(ctx):
            raise HTTPException(status_code=409, detail="A GPU generation job is active. Wait for it to finish, then retry.")
        return operation()
    finally:
        lock.release()


def _run_video_lab_gpu_with_audio_release(ctx: Any, operation):
    """Release resident MusicGen before Video Lab claims the GPU for VSR/RIFE."""
    _release_cached_audio_model(ctx)
    return operation()
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
        "modelLoad": dict(getattr(ctx, "_pro_model_load_state", {}) or {"status": "not-started", "modelId": "", "detail": ""}),
        "routeLifecycle": lifecycle_snapshot(ctx),
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

    - While a job runs, tick at the React apply cadence so progress and previews
      feel responsive without serializing payloads the client will drop.
      Preview frames still send only ONCE per (job, step) because the base64
      preview is by far the heaviest part of the payload.
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
    # Keep a saved model when its files still exist, even if support assets are
    # incomplete, so setup can explain how to repair that selection. If the
    # saved ID is no longer in the local catalog, choose a real ready image
    # route rather than asking the backend to resolve an unknown ID (which
    # historically could load the first unrelated checkpoint).
    selectable, _blocked = _selectable_checkpoint_payloads(ctx)
    if checkpoint_id and not _is_checkpoint_id_selectable(ctx, checkpoint_id):
        generation = getattr(ctx, "generation", None)
        list_checkpoints = getattr(generation, "list_checkpoints", None)
        saved_checkpoint = None
        known_checkpoints: list[Any] | None = None
        try:
            if callable(list_checkpoints):
                known_checkpoints = list_checkpoints()
                saved_checkpoint = next((
                    checkpoint
                    for checkpoint in known_checkpoints
                    if str(getattr(checkpoint, "id", "") or "") == str(checkpoint_id)
                ), None)
            saved_block = _runtime_checkpoint_block(ctx, saved_checkpoint) if saved_checkpoint else None
        except Exception:
            saved_block = None
        deliberately_unavailable = bool(
            saved_block and saved_block.get("status") in {"coming-soon", "blocked-cleanly"}
        )
        if saved_checkpoint is None and known_checkpoints is not None:
            checkpoint_id = _first_ready_image_checkpoint_id(selectable)
        elif saved_checkpoint is not None and deliberately_unavailable:
            checkpoint_id = _first_ready_image_checkpoint_id(selectable)
    # The picker also contains request-eligible video models. The initial
    # image default must come from a ready image route, regardless of inventory
    # ordering; explicitly saved video selections are preserved above.
    if not checkpoint_id:
        checkpoint_id = _first_ready_image_checkpoint_id(selectable)
    wan_runtime_mode = _canonical_wan_runtime_mode(getattr(settings, "last_wan_runtime_mode", "fast_5b"))
    wan_vae_id = str(getattr(settings, "last_wan_vae", "") or "").strip()
    wan_text_encoder = str(getattr(settings, "last_wan_text_encoder", "") or "").strip()
    try:
        wan_service = _wan_service(ctx)
        wan_vae_id = wan_vae_id or str(wan_service.preferred_vae(wan_runtime_mode) or "")
        wan_text_encoder = wan_text_encoder or str(wan_service.default_text_encoder() or "")
    except Exception:
        # Keep startup defaults usable when Wan support assets cannot be scanned;
        # the normal route preflight reports missing local dependencies.
        pass
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
        "wanRuntimeMode": wan_runtime_mode,
        "highNoiseModelId": str(getattr(settings, "last_wan_high", "") or ""),
        "lowNoiseModelId": str(getattr(settings, "last_wan_low", "") or ""),
        "highNoiseSteps": 20,
        "lowNoiseSteps": 1,
        "boundaryRatio": 0.875,
        "highNoiseLoraId": "",
        "highNoiseLoraScale": 1.0,
        "lowNoiseLoraId": "",
        "lowNoiseLoraScale": 1.0,
        "vaeId": wan_vae_id,
        "textEncoderPath": wan_text_encoder,
        "wanOffload": getattr(settings, "last_wan_offload", "balanced") or "balanced",
        "wanSigmaType": getattr(settings, "last_wan_sigma_type", "simple") or "simple",
        "wanSampler": getattr(settings, "last_wan_sampler", "unipc") or "unipc",
        "wanFlowShift": float(getattr(settings, "last_wan_flow_shift", 5.0) or 5.0),
    }


def _first_ready_image_checkpoint_id(selectable: list[dict[str, Any]]) -> str | None:
    """Pick an image model for the initial default, never a video route or blocked row."""
    for model in selectable:
        model_id = str(model.get("id") or "").strip()
        engine_id = str(model.get("engineId") or "").strip().lower()
        if (
            model_id
            and engine_id not in {"", "unknown", "wan", "sana_video", "ltx"}
            and model.get("routeStatus") == "request-eligible"
            and model.get("checkpointPathStatus") == "present"
        ):
            return model_id
    return None


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
        detail = getattr(job, "error", None) or "Generation failed"
        status_code = 409 if str(detail).startswith("Generation deferred:") else 500
        raise HTTPException(status_code=status_code, detail=detail)
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
        "verificationStatus": (
            "verified"
            if recent_outputs
            and all(
                bool(item.get("path")) and Path(str(item["path"])).is_file()
                for item in recent_outputs
            )
            else "unverified"
        ),
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
    resolved_checkpoint = _resolve_checkpoint_for_generation_guard(ctx, raw_model_path) if raw_model_path else None
    checkpoint_data = _dump_model(resolved_checkpoint) if resolved_checkpoint is not None else {}
    resolved_path = str(checkpoint_data.get("path") or "")
    model_path = resolved_path or (
        raw_model_path if any(marker in raw_model_path.lower() for marker in ("\\", "/", ":")) else ""
    )
    source_image_path = _video_source_image_path(ctx, payload)
    selected_text = " ".join(
        (
            raw_model_path,
            str(checkpoint_data.get("filename") or ""),
            Path(resolved_path).name if resolved_path else "",
            str(payload.checkpoint_title or checkpoint_data.get("title") or ""),
        )
    ).lower()
    variant = "720p" if "720p" in selected_text else payload.sana_model_variant
    try:
        return SanaVideoRequest(
            prompt=payload.prompt,
            negative_prompt=payload.negative_prompt,
            model_path=model_path,
            model_variant=variant,
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
            audio_model_id=_sana_video_audio_model_id(getattr(ctx, "settings", None)),
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc


def _sana_video_output_payload(ctx: Any, result: Any, payload: ProGeneratePayload) -> dict[str, Any]:
    output_path = str(getattr(result, "output_path", "") or "")
    path = Path(output_path)
    root = _safe_output_root(ctx)
    try:
        resolved = path.expanduser().resolve()
        output_stat = resolved.stat()
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Sana Video returned an unreadable output path.") from exc
    if root is None or not _path_inside(resolved, root) or not resolved.is_file() or output_stat.st_size <= 0:
        raise HTTPException(status_code=500, detail="Sana Video did not return a nonempty video file inside the configured output directory.")
    created_at = datetime.fromtimestamp(output_stat.st_mtime, timezone.utc).isoformat()
    return {
        "id": f"sana-video-{resolved.stem or 'output'}",
        "url": _output_asset_url(ctx, resolved),
        "thumbnailUrl": _output_asset_url(ctx, resolved),
        "path": str(resolved),
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


def _reconcile_musicgen_residency(ctx: Any, route_id: str, model_id: str, service: Any, *, detail: str) -> None:
    """Keep route residency truthful after MusicGen has had an execution chance."""
    if not model_id.startswith("facebook/musicgen-"):
        return
    parked_check = getattr(service, "musicgen_model_is_parked_on_cpu", None)
    try:
        parked = bool(callable(parked_check) and parked_check(model_id))
    except Exception:
        parked = False
    if parked:
        confirm_route_residency(ctx, route_id, model_id, resident=False)
    else:
        clear_route_residency(
            ctx,
            route_id,
            model_id,
            detail=f"{detail}; MusicGen CPU parking could not be confirmed, so residency is unknown.",
        )


def _reconcile_sana_audio_residency(ctx: Any, route_id: str, model_id: str, service: Any, *, detail: str) -> None:
    pipeline = getattr(service, "_prepared_pipeline", ...)
    pipeline_key = getattr(service, "_prepared_pipeline_key", ...)
    if pipeline is None and pipeline_key is None:
        confirm_route_residency(ctx, route_id, model_id, resident=False)
    else:
        clear_route_residency(ctx, route_id, model_id, detail=f"{detail}; Sana pipeline residency could not be confirmed.")


def _generate_sana_video_response(ctx: Any, payload: ProGeneratePayload) -> dict[str, Any]:
    if not _pro_sana_video_backend_enabled():
        raise HTTPException(status_code=503, detail="React Pro Sana Video backend is disabled in aiwf/web/pro_api.py.")
    request = _sana_video_request_from_payload(ctx, payload)
    route_id = f"video.sana.{request.model_variant}"
    from aiwf.services.pipeline_preflight import preflight_sana_video_pipeline

    try:
        preflight = preflight_sana_video_pipeline(
            ctx.flags, getattr(ctx, "settings", None), request=request,
        )
    except Exception as exc:
        preflight = None
        preflight_error = f"Sana Video route preflight failed: {exc}"
    else:
        preflight_error = ""
    support_ids = [request.model_variant, request.model_path or ""]
    if preflight is not None:
        support_ids.extend(_preflight_support_paths(preflight))
    audio_ready, audio_detail, audio_support_ids = _sana_optional_audio_readiness(ctx, request)
    support_ids.extend(audio_support_ids)
    if preflight is None or not preflight.ok or not audio_ready:
        detail = preflight_error or str(
            getattr(preflight, "markdown", lambda: "Sana Video route preflight did not pass.")()
        )
        if preflight is not None and preflight.ok and not audio_ready:
            detail = audio_detail
        select_route(
            ctx, route=route_id, model_id=request.model_path or request.model_variant,
            setup_ready=False, support_ids=support_ids, detail=detail,
        )
        missing_assets = preflight is not None and any(
            not item.ok and item.name == "local model snapshot"
            for item in getattr(preflight, "items", ())
        )
        raise HTTPException(status_code=422 if missing_assets else 503, detail=detail)
    job_id = _pro_video_job_start(ctx, request, message="Starting Sana video generation.")
    token = begin_route_operation(
        ctx, route=route_id, model_id=request.model_path or request.model_variant,
        setup_ready=True, support_ids=support_ids, detail="Sana route and support assets passed preflight; preparing generation."
    )
    mark_route_running(ctx, route_id, token, "Sana video generation is running.")
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

    service = None
    try:
        service = _sana_video_service(ctx)
        result = service.generate(request, on_progress=on_progress)
    except GenerationCancelledError as exc:
        if request.generate_audio:
            _reconcile_sana_audio_residency(ctx, route_id, request.model_path or request.model_variant, service, detail="Sana audio generation was cancelled")
        finish_route_operation(ctx, route_id, token, success=False, cancelled=True, detail=f"Cancelled: {exc}")
        _pro_video_job_finish(ctx, job_id, "cancelled", message=str(exc))
        raise HTTPException(status_code=499, detail=_pro_error_detail(str(exc), job=_pro_video_job_status(ctx))) from exc
    except Exception as exc:
        if request.generate_audio:
            _reconcile_sana_audio_residency(ctx, route_id, request.model_path or request.model_variant, service, detail="Sana audio generation failed")
        finish_route_operation(ctx, route_id, token, success=False, detail=str(exc))
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

    try:
        result_progress = list(getattr(result, "progress", None) or progress)
        output = _sana_video_output_payload(ctx, result, payload)
        message = str(getattr(result, "message", "") or "Sana video complete.")
        response = {
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
            "message": message,
        }
    except Exception as exc:
        if request.generate_audio:
            _reconcile_sana_audio_residency(ctx, route_id, request.model_path or request.model_variant, service, detail="Sana audio output verification failed")
        finish_route_operation(ctx, route_id, token, success=False, detail=str(exc))
        _pro_video_job_finish(ctx, job_id, "failed", message=str(exc), error=str(exc))
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finish_route_operation(ctx, route_id, token, success=True, detail="Sana video output file verified.")
    if request.generate_audio:
        _reconcile_sana_audio_residency(ctx, route_id, request.model_path or request.model_variant, service, detail="Sana video completed with audio")
    _pro_video_job_finish(ctx, job_id, "completed", message=message)
    return response


def _wan_vae_name_conflicts_with_runtime(vae_id: str, runtime_mode: str) -> bool:
    name = str(vae_id or "").replace("\\", "/").rsplit("/", 1)[-1].lower()
    looks_wan21 = bool(re.search(r"wan[_ .-]*2[._-]*1|wan21", name))
    looks_wan22 = bool(re.search(r"wan[_ .-]*2[._-]*2|wan22", name))
    if str(runtime_mode).strip().lower() == "fast_5b":
        return looks_wan21
    return looks_wan22


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
    # Preserve an explicit per-request VAE (preflight reports incompatibility).
    # A saved global VAE can belong to the previously selected Wan family, so
    # only reuse it when its known version matches the requested runtime.
    explicit_vae_id = str(payload.vae_id or "").strip()
    saved_vae_id = str(getattr(settings, "last_wan_vae", "") or "").strip()
    if saved_vae_id and _wan_vae_name_conflicts_with_runtime(saved_vae_id, runtime_mode):
        saved_vae_id = ""
    vae_id = explicit_vae_id or saved_vae_id or None
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
    root = _safe_output_root(ctx)
    try:
        resolved = path.expanduser().resolve()
        output_stat = resolved.stat()
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Wan returned an unreadable output path.") from exc
    if root is None or not _path_inside(resolved, root) or not resolved.is_file() or output_stat.st_size <= 0:
        raise HTTPException(status_code=500, detail="Wan did not return a nonempty video file inside the configured output directory.")
    created_at = datetime.fromtimestamp(output_stat.st_mtime, timezone.utc).isoformat()
    return {
        "id": f"wan-video-{resolved.stem or 'output'}",
        "url": _output_asset_url(ctx, resolved),
        "thumbnailUrl": _output_asset_url(ctx, resolved),
        "path": str(resolved),
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
        from aiwf.services.audio import AudioGenerationService

        backend = getattr(getattr(ctx, "generation", None), "backend", None)
        service = AudioGenerationService(
            getattr(ctx, "flags", None),
            getattr(ctx, "settings", None),
            devices=getattr(backend, "devices", None),
            supervisor=getattr(ctx, "supervisor", None),
            unload_image_models=lambda: _unload_image_models_for_active_route(ctx),
        )
        _AUDIO_SERVICES[key] = service
    return service


# Commercial-use policy for audio (aiwf/services/audio_licenses.py): a non-commercial model asked
# for outside research mode is refused with 403 and a sentence naming its licence, instead of the
# generic "choose an available model" that a filtered picker would otherwise produce.
def _raise_if_audio_license_blocked(ctx: Any, model_id: str) -> None:
    from aiwf.services import audio_licenses

    # the person's setting is the source of truth, whichever audio service object is in use
    research = bool(getattr(getattr(ctx, "settings", None), "allow_noncommercial_audio_models", False))
    if model_id and not audio_licenses.allowed(model_id, research_mode=research):
        raise HTTPException(status_code=403, detail=audio_licenses.blocked_message(model_id))


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


class ProAudioGeneratePayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    prompt: str = ""
    negative_prompt: str = Field(default="", validation_alias=AliasChoices("negativePrompt", "negative_prompt"))
    kind: str = "music"
    model_id: str = Field(
        default="facebook/musicgen-small",
        validation_alias=AliasChoices("modelId", "model_id", "audioModel", "audio_model"),
    )
    duration_seconds: float = Field(
        default=8.0,
        ge=1.0,
        le=120.0,
        validation_alias=AliasChoices("durationSeconds", "duration_seconds", "duration"),
    )
    temperature: float = Field(default=1.0, ge=0.1, le=2.0)
    cfg_coef: float = Field(
        default=3.0,
        ge=0.1,
        le=10.0,
        validation_alias=AliasChoices("cfgCoef", "cfg_coef", "guidance"),
    )
    top_k: int = Field(default=250, ge=0, le=1000, validation_alias=AliasChoices("topK", "top_k"))
    steps: int = Field(default=25, ge=1, le=200)
    seed: int = -1

    @model_validator(mode="after")
    def supported_kind(self):
        self.kind = str(self.kind or "music").strip().lower()
        if self.kind not in {"music", "sfx"}:
            raise ValueError("kind must be 'music' or 'sfx'; use Video Lab for video-conditioned audio.")
        return self


class ProAudioResearchModePayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    enabled: bool


class ProAudioPreparePayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    kind: str = "music"
    model_id: str = Field(
        default="facebook/musicgen-small",
        validation_alias=AliasChoices("modelId", "model_id", "audioModel", "audio_model"),
    )

    @model_validator(mode="after")
    def supported_kind(self):
        self.kind = str(self.kind or "").strip().lower()
        if self.kind not in {"music", "sfx"}:
            raise ValueError("kind must be 'music' or 'sfx'.")
        self.model_id = str(self.model_id or "").strip()
        if not self.model_id:
            raise ValueError("modelId is required.")
        return self


class ProAudioProjectSavePayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    name: str = Field(min_length=1, max_length=100)
    project_id: str | None = Field(default=None, validation_alias=AliasChoices("projectId", "project_id"))
    audio_path: str | None = Field(default=None, validation_alias=AliasChoices("audioPath", "audio_path"))
    options: AudioGenerationOptions = Field(default_factory=AudioGenerationOptions)
    sample_rate: int = Field(default=0, ge=0, le=384000)
    license_notice: str | None = Field(default=None, validation_alias=AliasChoices("licenseNotice", "license_notice"))
    license: dict[str, Any] | None = None
    consent_status: str | None = Field(default=None, validation_alias=AliasChoices("consentStatus", "consent_status"))


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


class ProEnhanceModelStatus(BaseModel):
    id: str
    title: str
    filename: str
    kind: Literal["upscaler", "restorer"]
    architecture: str
    scale: int
    installed: bool
    installAvailable: bool


class ProEnhanceModelsResponse(BaseModel):
    models: list[ProEnhanceModelStatus]


class ProEnhanceModelInstallResponse(BaseModel):
    model: ProEnhanceModelStatus
    path: str
    message: str


def _video_lab_upload_root(ctx: Any) -> Path:
    flags = getattr(ctx, "flags", None)
    root = flags.resolved_output_dir() / "video-lab" / "uploads"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _video_lab_upload_destination(root: Path, safe_stem: str, suffix: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return root / f"{safe_stem}_{stamp}_{uuid4().hex[:12]}{suffix}"


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
    # The Pro backend memoizes checkpoint, LoRA, embedding, and VAE catalogs
    # independently of the inventory file. Invalidate them too, or a sort or
    # upload can leave stale labels and selectable rows in the running app.
    backend = getattr(getattr(ctx, "generation", None), "backend", None)
    for name in ("invalidate_checkpoints", "invalidate_loras", "invalidate_embeddings", "invalidate_vaes"):
        invalidate = getattr(backend, name, None)
        if callable(invalidate):
            invalidate()
    invalidate_enhance = getattr(getattr(ctx, "enhance", None), "invalidate_model_catalog", None)
    if callable(invalidate_enhance):
        invalidate_enhance()
    setattr(ctx, "_pro_capability_cache", None)
    return {"inventoryCount": len(records)}


def _sort_source_revision(raw_source: str) -> list[tuple[str, int, int, int]]:
    """Fingerprint a planned file/folder without reading model weight contents."""
    source = Path(raw_source)
    entries: list[tuple[str, int, int, int]] = []
    if source.is_dir() and not source.is_symlink():
        try:
            for root, dirs, files in os.walk(source, followlinks=False):
                dirs[:] = sorted(name for name in dirs if not (Path(root) / name).is_symlink())
                for name in sorted(files):
                    path = Path(root) / name
                    try:
                        stat = path.stat()
                    except OSError:
                        entries.append((str(path.relative_to(source)), -1, -1, -1))
                    else:
                        entries.append((str(path.relative_to(source)), stat.st_size, stat.st_mtime_ns, getattr(stat, "st_ino", 0)))
        except OSError:
            return [("<unavailable>", -1, -1, -1)]
    else:
        try:
            stat = source.stat()
        except OSError:
            return [("<unavailable>", -1, -1, -1)]
        entries.append((source.name, stat.st_size, stat.st_mtime_ns, getattr(stat, "st_ino", 0)))
    return entries


def _model_reorganize_plan_id(actions: list[Any]) -> str:
    reviewed = []
    for action in actions:
        item = _sort_action_payload(action)
        item["sourceRevision"] = _sort_source_revision(item["source"])
        reviewed.append(item)
    encoded = json.dumps(reviewed, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _approved_model_reorganize_moves(actions: list[Any]) -> set[tuple[str, str, str, str]]:
    approved: set[tuple[str, str, str, str]] = set()
    for action in actions:
        if str(getattr(action, "status", "")) != "moved":
            continue
        source = Path(str(getattr(action, "source", "")))
        try:
            source = source.resolve(strict=True)
        except OSError:
            continue
        approved.add((
            str(source).casefold(),
            str(getattr(action, "family", "") or "").casefold(),
            str(getattr(action, "architecture", "") or "").casefold(),
            str(getattr(action, "dest_subdir", "") or "").replace("\\", "/").casefold(),
        ))
    return approved


def _model_sort_response(
    ctx: Any,
    actions: list[Any],
    *,
    uploaded_path: Path | None = None,
    planned: bool = False,
    plan_id: str = "",
) -> dict[str, Any]:
    payload_actions = [_sort_action_payload(action) for action in actions]
    moved = sum(1 for item in payload_actions if item["status"] == "moved")
    if planned:
        for item in payload_actions:
            if item["status"] == "moved":
                item["status"] = "would-move"
    left = sum(1 for item in payload_actions if item["status"] in {"left", "conflict", "error"})
    payload: dict[str, Any] = {
        "status": "planned" if planned else "completed",
        "planId": plan_id,
        "uploadedPath": str(uploaded_path) if uploaded_path is not None else "",
        "actions": payload_actions,
        "counts": {
            "total": len(payload_actions),
            "moved": 0 if planned else moved,
            "planned": moved if planned else 0,
            "left": left,
        },
    }
    # A plan is read-only. Only applied placement/upload operations should
    # rewrite inventory and invalidate backend caches.
    if not planned:
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
    available_choices = getattr(service, "available_video_audio_model_choices", service.video_audio_model_choices)
    video_audio_choices = [model_id for _, model_id in (available_choices() or [])]
    music_choices = [model_id for _, model_id in (service.music_model_choices() or [])]
    model_id = (payload.audio_model or "").strip() or (
        video_audio_choices[0] if video_audio_choices else music_choices[0] if music_choices else ""
    )
    if not model_id:
        raise HTTPException(
            status_code=409,
            detail="No commercial-safe soundtrack model is installed yet. Install one in Audio setup, "
                   "or turn on research mode in Audio settings for non-commercial work.",
        )
    _raise_if_audio_license_blocked(ctx, model_id)
    video_conditioned = model_id.startswith(("mmaudio:", "events:"))
    kind = "video_audio" if video_conditioned else "music"
    route_id = (
        f"audio.video.audio.{model_id}"
        if video_conditioned else f"audio.music.{model_id}"
    )
    commercial_engine = model_id.startswith(("acestep:", "moss-sfx:", "events:"))
    if commercial_engine:
        # ACE-Step, MOSS-SoundEffect and the event soundtrack run in their own environments
        variant = ""
        readiness_check = lambda _variant: bool(service.commercial_engine_ready(model_id))  # noqa: E731
    elif model_id.startswith("mmaudio:"):
        variant = model_id.split(":", 1)[1]
        readiness_check = getattr(service, "_mmaudio_variant_ready", None)
    elif model_id.startswith("facebook/musicgen-"):
        variant = model_id.removeprefix("facebook/musicgen-")
        readiness_check = getattr(service, "_musicgen_variant_ready", None)
    else:
        variant = ""
        readiness_check = None
    route_setup_ready = bool(callable(readiness_check) and readiness_check(variant))
    if model_id.startswith("mmaudio:"):
        runtime_check = getattr(service, "_mmaudio_runtime_import_error", None)
        route_setup_ready = route_setup_ready and callable(runtime_check) and not runtime_check()
    if model_id.startswith("facebook/musicgen-"):
        setup = service.setup_status(deep=False)
        route_setup_ready = route_setup_ready and bool(
            setup.get("musicDependenciesReady", setup.get("musicReady", False))
        )
    setup = service.setup_status(deep=False)
    if not setup.get("muxReady"):
        raise HTTPException(
            status_code=409,
            detail="Video audio requires working FFmpeg and ffprobe to mux the soundtrack into the video.",
        )
    support_paths = getattr(service, "model_support_paths", None)
    support_ids = support_paths(model_id) if callable(support_paths) else [model_id]
    token = begin_route_operation(
        ctx,
        route=route_id,
        model_id=model_id,
        setup_ready=route_setup_ready,
        support_ids=support_ids,
        detail="Checking the selected video-audio model and its exact support assets.",
    )
    if not route_setup_ready:
        raise HTTPException(status_code=409, detail="The selected video-audio model is not installed or its runtime dependencies are unavailable.")
    mark_route_running(ctx, route_id, token, "Video-audio generation and mux are running.")
    options = AudioGenerationOptions(prompt=prompt, kind=kind, model_id=model_id)
    try:
        audio, muxed = service.generate_and_mux(src, options)
    except AudioUnavailable as exc:
        finish_route_operation(ctx, route_id, token, success=False, detail=str(exc))
        _reconcile_musicgen_residency(ctx, route_id, model_id, service, detail="Video Lab audio failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        finish_route_operation(ctx, route_id, token, success=False, detail=str(exc))
        _reconcile_musicgen_residency(ctx, route_id, model_id, service, detail="Video Lab audio or mux failed")
        raise
    finish_route_operation(ctx, route_id, token, success=True, detail="Video-audio output and mux completed.")
    _reconcile_musicgen_residency(ctx, route_id, model_id, service, detail="Video Lab audio completed")
    return _video_lab_output(
        ctx,
        muxed.output_path,
        f"Added {kind.replace('_', ' ')} audio -> {Path(muxed.output_path).name}",
        {
            "audioPath": audio.output_path,
            "infotext": audio.infotext,
            "license": getattr(audio, "license", {}),
        },
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
        available_choices = getattr(audio, "available_video_audio_model_choices", audio.video_audio_model_choices)
        video_audio_models = [model_id for _, model_id in (available_choices() or [])]
        audio_setup = audio.setup_status(deep=False)
        video_audio_mux_ready = bool(audio_setup.get("muxReady"))
        audio_model_choices = []
        mmaudio_runtime_check = getattr(audio, "_mmaudio_runtime_import_error", None)
        musicgen_dependencies_ready = bool(
            audio_setup.get("musicDependenciesReady", audio_setup.get("musicReady", False))
        )
        for label, model_id in [*audio.video_audio_model_choices(), *audio.music_model_choices()]:
            if model_id.startswith(("events:", "acestep:")):
                # commercially licensed: its own environment and weights, not MMAudio's/MusicGen's
                conditioning_mode = "video-conditioned" if model_id.startswith("events:") else "prompt-only"
                installed = bool(audio.commercial_engine_ready(model_id))
                available = video_audio_mux_ready
                unavailable_reason = (
                    "Working FFmpeg and ffprobe are required to mux generated audio into the video."
                    if not video_audio_mux_ready else ""
                )
            elif model_id.startswith("mmaudio:"):
                conditioning_mode = "video-conditioned"
                variant = model_id.split(":", 1)[1]
                ready_check = getattr(audio, "_mmaudio_variant_ready", None)
                installed = bool(callable(ready_check) and ready_check(variant))
                runtime_ready = bool(callable(mmaudio_runtime_check) and not mmaudio_runtime_check())
                available = runtime_ready and video_audio_mux_ready
                unavailable_reason = (
                    "Install the minimum Audio setup to enable the MMAudio runtime." if not runtime_ready else
                    "Working FFmpeg and ffprobe are required to mux generated audio into the video."
                    if not video_audio_mux_ready else ""
                )
            elif model_id.startswith("facebook/musicgen-"):
                conditioning_mode = "prompt-only"
                variant = model_id.removeprefix("facebook/musicgen-")
                ready_check = getattr(audio, "_musicgen_variant_ready", None)
                installed = bool(callable(ready_check) and ready_check(variant))
                available = musicgen_dependencies_ready and video_audio_mux_ready
                unavailable_reason = (
                    "Install the minimum Audio setup before using MusicGen." if not musicgen_dependencies_ready else
                    "Working FFmpeg and ffprobe are required to mux generated audio into the video."
                    if not video_audio_mux_ready else ""
                )
            else:
                continue
            choice = {
                "label": label,
                "id": model_id,
                "conditioningMode": conditioning_mode,
                "available": available,
                "unavailableReason": unavailable_reason,
                "installed": installed,
                "installable": available,
                "ready": available and installed,
            }
            if model_id == "events:moss-sfx":
                choice["setupRoute"] = _setup_route_descriptor({"routeKey": "pro.audio.moss-sfx.v2"})
            audio_model_choices.append(choice)
    except Exception:
        video_audio_models = []
        audio_setup = {"videoAudioReady": False}
        audio_model_choices = []
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
            "modelChoices": audio_model_choices,
            "defaultModelId": str(getattr(getattr(ctx, "settings", None), "last_video_audio_model_id", "") or "") or "mmaudio:small_16k",
            "ready": bool(audio_setup.get("videoAudioReady")),
            "musicGenReady": bool(audio_setup.get("musicReady") and audio_setup.get("musicDependenciesReady") and audio_setup.get("muxReady")),
        },
        "extend": {
            "available": True,
            "note": "Uses the Wan TI2V-5B route: the clip's last frame becomes the i2v conditioning image.",
        },
    }


def _audio_status_payload(ctx: Any, *, deep: bool = False) -> dict[str, Any]:
    service = _audio_service(ctx)
    status = service.setup_status(deep=deep)
    settings = getattr(ctx, "settings", None)
    # a remembered choice counts only while the licence policy still offers it
    from aiwf.services import audio_licenses

    research = bool(getattr(settings, "allow_noncommercial_audio_models", False))

    def remembered(field: str, fallback: str) -> str:
        value = str(getattr(settings, field, "") or "")
        return value if value and audio_licenses.allowed(value, research_mode=research) else fallback

    status["defaults"] = {
        "music": remembered("last_audio_music_model_id", status["defaults"]["music"]),
        "sfx": remembered("last_audio_sfx_model_id", status["defaults"]["sfx"]),
        "videoAudio": remembered("last_video_audio_model_id", status["defaults"]["videoAudio"]),
    }
    mmaudio_runtime_check = getattr(service, "_mmaudio_runtime_import_error", None)
    mmaudio_component = next(
        (
            component
            for component in status.get("components", [])
            if isinstance(component, dict) and component.get("id") == "mmaudio-small-16k"
        ),
        None,
    )
    if status.get("runtimeChecksPerformed") and mmaudio_component is not None:
        mmaudio_runtime_error = str(mmaudio_component.get("error") or "")
        mmaudio_runtime_ready = not mmaudio_runtime_error
    elif callable(mmaudio_runtime_check):
        try:
            mmaudio_runtime_error = str(mmaudio_runtime_check() or "")
        except Exception as exc:
            mmaudio_runtime_error = str(exc) or "MMAudio runtime import check failed."
        mmaudio_runtime_ready = not mmaudio_runtime_error
    else:
        mmaudio_runtime_error = "MMAudio runtime readiness check is unavailable."
        mmaudio_runtime_ready = False

    def choices_for(choices: list[tuple[str, str]], *, route_kind: str) -> list[dict[str, Any]]:
        result = []
        for label, model_id in choices:
            available = model_id != "facebook/audiogen-medium"
            choice = {
                "label": label,
                "id": model_id,
                "available": available,
                "unavailableReason": "AudioGen requires AudioCraft, which is not installed in the shared Studio runtime."
                if not available else "",
            }
            if model_id.startswith(("acestep:", "moss-sfx:", "events:")):
                ready = bool(getattr(service, "commercial_engine_ready", lambda _m: False)(model_id))
                choice["installed"] = ready
                choice["installable"] = True
                choice["ready"] = ready
            elif model_id.startswith("mmaudio:"):
                variant = model_id.split(":", 1)[1]
                is_installed = getattr(service, "_mmaudio_variant_ready", None)
                choice["installed"] = bool(is_installed(variant)) if callable(is_installed) else (
                    bool(status.get("sfxReady")) if variant == "small_16k" else False
                )
                choice["available"] = available and mmaudio_runtime_ready
                choice["unavailableReason"] = (
                    f"MMAudio runtime is not ready: {mmaudio_runtime_error}"
                    if not mmaudio_runtime_ready
                    else ""
                )
                choice["installable"] = available
            elif model_id.startswith("facebook/musicgen-"):
                variant = model_id.removeprefix("facebook/musicgen-")
                is_installed = getattr(service, "_musicgen_variant_ready", None)
                choice["installed"] = bool(is_installed(variant)) if callable(is_installed) else (
                    bool(status.get("musicReady")) if variant == "small" else False
                )
                choice["installable"] = available
                if not bool(status.get("musicDependenciesReady", status.get("musicReady", False))):
                    choice["available"] = False
                    choice["unavailableReason"] = "Install the minimum Audio setup to add the shared MusicGen runtime dependencies."
            elif not available:
                choice["installed"] = False
                choice["installable"] = False
            if route_kind == "video.audio" and not bool(status.get("muxReady")):
                choice["available"] = False
                choice["unavailableReason"] = (
                    "Working FFmpeg and ffprobe are required to mux generated audio into the video."
                )
            route_key = f"audio.{route_kind}.{model_id}"
            support_paths = getattr(service, "model_support_paths", None)
            support_ids = support_paths(model_id) if callable(support_paths) else [model_id]
            musicgen_resident_probe = getattr(service, "_musicgen_model_is_resident", None)
            resident = (
                bool(musicgen_resident_probe(model_id))
                if model_id.startswith("facebook/musicgen-") and callable(musicgen_resident_probe)
                else None
            )
            lifecycle = select_route(
                ctx,
                route=route_key,
                model_id=model_id,
                setup_ready=bool(choice.get("available") and choice.get("installed")),
                support_ids=support_ids,
                resident=resident,
                detail=(
                    "Selected audio model and its exact variant assets are installed."
                    if choice.get("available") and choice.get("installed")
                    else str(choice.get("unavailableReason") or "Install this audio model variant before generation.")
                ),
            )
            choice["routeStatus"] = lifecycle.get("status")
            # every offered model says what its licence allows (pickers label non-commercial ones)
            from aiwf.services import audio_licenses

            choice["license"] = audio_licenses.license_for(model_id)
            choice["resident"] = lifecycle.get("resident")
            setup_route_key = (
                "pro.audio.acestep.1-5-turbo" if model_id.startswith("acestep:") else
                "pro.audio.moss-sfx.v2" if model_id.startswith(("moss-sfx:", "events:")) else
                f"pro.audio.musicgen.{model_id.removeprefix('facebook/musicgen-')}"
                if model_id.startswith("facebook/musicgen-") else
                f"pro.audio.mmaudio.{model_id.split(':', 1)[1].replace('_', '-')}"
                if model_id.startswith("mmaudio:") and route_kind != "video.audio" else
                None
            )
            if route_kind == "video.audio" and model_id.startswith("mmaudio:"):
                setup_route_key = "pro.video.audio.mmaudio"
            choice["setupRoute"] = _setup_route_descriptor({"routeKey": setup_route_key}) if setup_route_key else None
            result.append(choice)
        return result
    status["models"] = {
        "music": choices_for(service.music_model_choices(), route_kind="music"),
        "sfx": choices_for(service.sfx_model_choices(), route_kind="sfx"),
        "videoAudio": choices_for(service.video_audio_model_choices(), route_kind="video.audio"),
    }
    status["routeLifecycle"] = lifecycle_snapshot(ctx)
    return status


def _generate_audio_response(ctx: Any, payload: ProAudioGeneratePayload) -> dict[str, Any]:
    from aiwf.core.domain.audio import AudioGenerationOptions
    from aiwf.services.audio import AudioUnavailable

    prompt = (payload.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="Enter an audio prompt before generating.")
    if payload.kind not in {"music", "sfx"}:
        raise HTTPException(status_code=422, detail="Audio kind must be 'music' or 'sfx'.")
    service = _audio_service(ctx)
    _raise_if_audio_license_blocked(ctx, payload.model_id)
    model_choices = service.music_model_choices() if payload.kind == "music" else service.sfx_model_choices()
    allowed_model_ids = {str(model_id) for _, model_id in model_choices}
    if payload.model_id not in allowed_model_ids:
        kind_label = "music" if payload.kind == "music" else "sound effects"
        raise HTTPException(
            status_code=422,
            detail=f"Choose a {kind_label} model from the available {kind_label} models.",
        )
    if payload.model_id == "facebook/audiogen-medium":
        raise HTTPException(
            status_code=422,
            detail="AudioGen is unavailable because AudioCraft is not installed in the shared Studio runtime. Choose MMAudio or install AudioCraft in a supported isolated environment.",
        )
    options = AudioGenerationOptions(
        prompt=prompt,
        negative_prompt=payload.negative_prompt,
        kind=payload.kind,
        model_id=payload.model_id,
        duration_seconds=payload.duration_seconds,
        temperature=payload.temperature,
        cfg_coef=payload.cfg_coef,
        top_k=payload.top_k,
        steps=payload.steps,
        seed=payload.seed,
    )
    route_id = f"audio.{payload.kind}.{payload.model_id}"
    if payload.model_id.startswith("facebook/musicgen-"):
        variant = payload.model_id.removeprefix("facebook/musicgen-")
        readiness_check = getattr(service, "_musicgen_variant_ready", None)
    elif payload.model_id.startswith(("acestep:", "moss-sfx:")):
        variant = ""
        readiness_check = lambda _variant: bool(service.commercial_engine_ready(payload.model_id))  # noqa: E731
    elif payload.model_id.startswith("mmaudio:"):
        variant = payload.model_id.split(":", 1)[1]
        readiness_check = getattr(service, "_mmaudio_variant_ready", None)
    else:
        variant = ""
        readiness_check = None
    setup = service.setup_status(deep=False)
    if callable(readiness_check):
        route_setup_ready = bool(readiness_check(variant))
    else:
        route_setup_ready = bool(
            setup.get("musicReady") if payload.kind == "music" and variant == "small"
            else setup.get("sfxReady") if payload.kind == "sfx" and variant == "small_16k"
            else False
        )
    if payload.kind == "music" and not payload.model_id.startswith("acestep:"):
        route_setup_ready = route_setup_ready and bool(
            setup.get("musicDependenciesReady", setup.get("musicReady", False))
        )
    if payload.model_id.startswith("mmaudio:"):
        runtime_check = getattr(service, "_mmaudio_runtime_import_error", None)
        route_setup_ready = route_setup_ready and callable(runtime_check) and not runtime_check()
    token = begin_route_operation(
        ctx, route=route_id, model_id=payload.model_id, setup_ready=route_setup_ready,
        support_ids=(
            service.model_support_paths(payload.model_id)
            if callable(getattr(service, "model_support_paths", None))
            else [payload.model_id]
        ), detail=(
            "Exact audio model variant is installed; preparing generation."
            if route_setup_ready else "The selected audio variant is not installed; complete its setup before generation."
        )
    )
    if not route_setup_ready:
        raise HTTPException(
            status_code=409,
            detail=f"The selected {payload.kind} model is not installed yet. Install this model variant, then generate again.",
        )
    mark_route_running(ctx, route_id, token, "Audio generation is running.")
    try:
        result = service.generate(options)
    except AudioUnavailable as exc:
        finish_route_operation(ctx, route_id, token, success=False, detail=str(exc))
        _reconcile_musicgen_residency(ctx, route_id, payload.model_id, service, detail="Audio generation failed")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        finish_route_operation(ctx, route_id, token, success=False, detail=str(exc))
        _reconcile_musicgen_residency(ctx, route_id, payload.model_id, service, detail="Audio generation failed")
        raise
    try:
        output = Path(result.output_path).expanduser().resolve()
        output_stat = output.stat()
        output_root = _safe_output_root(ctx)
        if output_root is None or not _path_inside(output, output_root) or not output.is_file() or output_stat.st_size <= 0:
            raise HTTPException(status_code=500, detail="Audio generation did not produce a playable file inside the configured output directory.")
        response = {
            "status": "complete",
            "message": result.message,
            "outputPath": str(output),
            "url": _output_asset_url(ctx, output),
            "prompt": result.prompt,
            "kind": result.kind,
            "modelId": result.model_id,
            "durationSeconds": result.duration_seconds,
            "sampleRate": result.sample_rate,
            "infotext": result.infotext,
            "license": result.license,
        }
    except Exception as exc:
        finish_route_operation(ctx, route_id, token, success=False, detail=str(exc))
        _reconcile_musicgen_residency(ctx, route_id, payload.model_id, service, detail="Audio output verification failed")
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Audio generation output could not be verified: {exc}") from exc
    finish_route_operation(ctx, route_id, token, success=True, detail="Nonempty audio output file verified inside the configured output directory.")
    _reconcile_musicgen_residency(ctx, route_id, payload.model_id, service, detail="Audio generation completed")
    return response


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
    route_id = f"video.wan.{request.runtime_mode}"
    model_id = request.model_id or request.runtime_mode
    support_ids = [request.runtime_mode, request.vae_id or "", request.text_encoder_path or "", request.high_noise_model_id or "", request.low_noise_model_id or ""]
    service = _wan_service(ctx)
    preflight = getattr(service, "preflight", None)
    preflight_error = ""
    try:
        readiness = preflight(request, image_present=source_image is not None) if callable(preflight) else None
        if not callable(preflight):
            preflight_error = "Wan route readiness check is unavailable."
    except Exception as exc:
        readiness = None
        preflight_error = f"Wan route readiness check failed: {exc}"
    route_ready = bool(getattr(readiness, "ok", False))
    support_ids.extend(_preflight_support_paths(readiness))
    if not route_ready:
        message = str(getattr(readiness, "message", lambda: "")() or "").strip()
        detail = preflight_error or message or "Wan route and support assets are not ready."
        select_route(
            ctx, route=route_id, model_id=model_id, setup_ready=False,
            support_ids=support_ids, detail=detail,
        )
        status_code = 503 if preflight_error or not callable(preflight) else 422
        raise HTTPException(status_code=status_code, detail=detail)
    job_id = _pro_video_job_start(ctx, request, message="Starting Wan video generation.")
    token = begin_route_operation(
        ctx, route=route_id, model_id=model_id, setup_ready=route_ready,
        support_ids=support_ids, detail=(
            "Wan route and support assets passed preflight; preparing generation."
            if route_ready else "Wan route readiness was not confirmed for this exact model and support selection."
        )
    )
    mark_route_running(ctx, route_id, token, "Wan video generation is running.")
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
        result = service.generate(
            request,
            source_image,
            on_progress=on_progress,
            should_cancel=should_cancel,
        )
    except GenerationCancelledError as exc:
        finish_route_operation(ctx, route_id, token, success=False, cancelled=True, detail=f"Cancelled: {exc}")
        _pro_video_job_finish(ctx, job_id, "cancelled", message=str(exc))
        raise HTTPException(status_code=499, detail=_pro_error_detail(str(exc), job=_pro_video_job_status(ctx))) from exc
    except Exception as exc:
        finish_route_operation(ctx, route_id, token, success=False, detail=str(exc))
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

    try:
        output = _wan_video_output_payload(ctx, result, payload)
        message = str(getattr(result, "message", "") or "Wan video complete.")
        response = {
            "jobId": job_id,
            "status": "completed",
            "output": output,
            "video": output["url"],
            "recentOutputs": [output],
            "progress": progress,
            "timings": {},
            "message": message,
        }
    except Exception as exc:
        finish_route_operation(ctx, route_id, token, success=False, detail=str(exc))
        _pro_video_job_finish(ctx, job_id, "failed", message=str(exc), error=str(exc))
        if isinstance(exc, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finish_route_operation(ctx, route_id, token, success=True, detail="Wan video output file verified.")
    _pro_video_job_finish(ctx, job_id, "completed", message=message)
    return response


def build_router(ctx: Any) -> APIRouter:
    if not getattr(ctx, "_pro_startup_started_at", None):
        setattr(ctx, "_pro_startup_started_at", time.time())
    router = APIRouter(prefix="/api/pro")

    @router.get("/ping")
    def ping():
        # Keep discovery cheap and token-free. Every other non-loopback Pro
        # request stays closed until mobile access is enabled and paired.
        state = load_mobile_auth(Path(ctx.flags.data_dir))
        mobile_access_enabled = bool(state.get("enabled")) and bool(state.get("token"))
        return {
            "name": "AIWF Studio Pro",
            "version": AIWF_VERSION,
            "authRequired": True,
            "mobileAccessEnabled": mobile_access_enabled,
        }

    @router.get("/mobile-pairing")
    def mobile_pairing(request: Request):
        _require_loopback(request)
        data_dir = Path(ctx.flags.data_dir)
        ensure_mobile_token(data_dir)
        state = load_mobile_auth(data_dir)
        return _mobile_pairing_payload(ctx, state)

    @router.post("/mobile-pairing")
    def mobile_pairing_update(request: Request, payload: ProMobilePairingUpdatePayload):
        _require_loopback(request)
        state = set_mobile_access_enabled(
            Path(ctx.flags.data_dir),
            enabled=payload.enabled,
            rotate=payload.rotate,
        )
        return _mobile_pairing_payload(ctx, state)

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

    @router.get("/controlnet/models")
    def controlnet_models():
        from aiwf.services.model_download_catalog import MODEL_DOWNLOAD_CATALOG
        from aiwf.services.model_setup_manifest import CONTROLNET_SETUP_CATALOG_KEYS

        service = getattr(ctx, "controlnet", None)
        models = _safe_list(getattr(service, "list_models", lambda: []))
        catalog = {entry.key: entry for entry in MODEL_DOWNLOAD_CATALOG}
        setup_options = []
        for family, keys in CONTROLNET_SETUP_CATALOG_KEYS.items():
            for key in keys:
                entry = catalog.get(key)
                if entry is None:
                    continue
                model_id = Path(urlparse(entry.url).path).stem if entry.url else Path(entry.repo_id).name
                setup_options.append({
                    "key": key,
                    "family": family,
                    "label": entry.title,
                    "modelId": model_id,
                    "sizeMb": entry.size_mb,
                })
        return {
            "models": [
                {
                    "id": str(getattr(model, "id", "") or ""),
                    "title": str(getattr(model, "title", "") or ""),
                    "path": str(getattr(model, "path", "") or ""),
                }
                for model in models
                if str(getattr(model, "id", "") or "").strip()
            ],
            "setupOptions": setup_options,
        }

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
        for ltx_model in _ltx_model_payloads(ctx):
            if ltx_model["status"] == "Ready":
                checkpoints.append(ltx_model)
            else:
                blocked_checkpoints.append(ltx_model)
        if _pro_sana_video_backend_enabled():
            for sana_model in _sana_video_model_payloads(ctx):
                existing_sana_ids = {str(item.get("id") or "") for item in [*checkpoints, *blocked_checkpoints]}
                same_snapshot = any(
                    str(item.get("architecture") or "").strip().lower().replace("-", "_") == "sana_video"
                    and str(item.get("filename") or "").casefold() == str(sana_model.get("filename") or "").casefold()
                    for item in [*checkpoints, *blocked_checkpoints]
                )
                if sana_model["id"] not in existing_sana_ids and not same_snapshot:
                    if sana_model.get("routeStatus") == "blocked":
                        blocked_checkpoints.append(sana_model)
                    else:
                        checkpoints.append(sana_model)
        existing_ids = {str(item.get("id") or "") for item in [*checkpoints, *blocked_checkpoints]}
        for wan_model in _wan_model_payloads(ctx):
            if wan_model["id"] not in existing_ids:
                if wan_model.get("routeStatus") == "blocked":
                    blocked_checkpoints.append(wan_model)
                else:
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
            # Filter counts must describe the selectable rows shown by the
            # picker. Blocked models remain available through blockedCheckpoints
            # and the setup UI, but must not create an empty engine filter.
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
        if os.name == "nt" and key in {"fluxtrait-zimage-v2-q4", "fluxtrait-zimage-v2-q8"}:
            raise HTTPException(
                status_code=422,
                detail="Z-Image GGUF loading is currently blocked on Windows. Install the BF16 transformer bundle instead.",
            )
        copy_shared = getattr(service, "copy_shared_catalog_asset_to_primary", None)
        if callable(copy_shared):
            def copy_shared_and_refresh():
                copied = copy_shared(entry)
                if copied:
                    try:
                        _refresh_model_inventory_after_sort(ctx)
                    except Exception:
                        logger.exception("Could not refresh model inventory after shared-root asset import")
                return copied

            try:
                copied = _run_exclusive_pro_gpu_operation(ctx, copy_shared_and_refresh)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            if copied:
                payload = _download_payload(ctx)
                payload["catalogAction"] = {
                    "key": key,
                    "status": "copied_from_shared_root",
                    "source": copied.get("source"),
                    "path": copied.get("target"),
                }
                return payload
        if service.is_catalog_installed(entry):
            payload = _download_payload(ctx)
            payload["catalogAction"] = {"key": key, "status": "already_installed"}
            return payload
        def place_misplaced_and_refresh():
            placed = service.place_misplaced_catalog_asset(entry)
            if placed:
                try:
                    _refresh_model_inventory_after_sort(ctx)
                except Exception:
                    logger.exception("Could not refresh model inventory after catalog asset placement")
            return placed

        if _run_exclusive_pro_gpu_operation(ctx, place_misplaced_and_refresh):
            payload = _download_payload(ctx)
            payload["catalogAction"] = {"key": key, "status": "placed"}
            return payload
        if not _catalog_entry_can_download(entry):
            if _catalog_entry_requires_auth(entry):
                detail = "This model requires upstream access or authentication. Open the source page and accept access first."
            elif str(getattr(entry, "source", "") or "").strip().lower() == "civitai":
                detail = "Open this CivitAI page. Direct app download is not enabled for CivitAI catalog entries."
            else:
                detail = "This catalog entry is link-only and cannot be downloaded directly by the app."
            raise HTTPException(status_code=422, detail=detail)
        try:
            def download_and_refresh():
                path = service.download_catalog(key)
                try:
                    _refresh_model_inventory_after_sort(ctx)
                except Exception:
                    logger.exception("Could not refresh model inventory after catalog download")
                return path

            downloaded_path = _run_exclusive_pro_gpu_operation(ctx, download_and_refresh)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Catalog download failed: {exc}") from exc
        payload = _download_payload(ctx)
        payload["downloaded"] = {"key": key, "path": str(downloaded_path)}
        payload["inventoryRefresh"] = "complete-or-recoverable"
        return payload

    @router.post("/downloads/catalog/{key}/import-shared")
    def import_shared_catalog_snapshot(
        key: str, confirm: bool = False, source: str = "", size_bytes: int = -1
    ):
        if not confirm:
            raise HTTPException(status_code=400, detail="Confirm the estimated snapshot copy before importing.")
        service = getattr(ctx, "model_download", None)
        if service is None:
            raise HTTPException(status_code=500, detail="Model download service is unavailable.")
        entry = service.find_catalog(key)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Catalog entry '{key}' was not found.")
        copy_snapshot = getattr(service, "copy_shared_catalog_snapshot_to_primary", None)
        def import_snapshot_and_refresh():
            copied = (
                copy_snapshot(entry, expected_source=source, expected_size_bytes=size_bytes)
                if callable(copy_snapshot) else None
            )
            if copied:
                try:
                    _refresh_model_inventory_after_sort(ctx)
                except Exception:
                    logger.exception("Could not refresh inventory after shared snapshot import")
            return copied

        copied = _run_exclusive_pro_gpu_operation(ctx, import_snapshot_and_refresh)
        if copied is None:
            raise HTTPException(
                status_code=409,
                detail="The shared snapshot changed, is ambiguous, or there is not enough disk space. Refresh setup status and review the estimate.",
            )
        payload = _download_payload(ctx)
        payload["catalogAction"] = {
            "key": key,
            "status": "copied_snapshot_from_shared_root",
            "source": copied["source"],
            "path": copied["target"],
        }
        return payload

    @router.get("/downloads/catalog/{key}/shared-import-preview")
    def preview_shared_catalog_snapshot(key: str):
        service = getattr(ctx, "model_download", None)
        if service is None:
            raise HTTPException(status_code=500, detail="Model download service is unavailable.")
        entry = service.find_catalog(key)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"Catalog entry '{key}' was not found.")
        preview_shared = getattr(service, "preview_shared_catalog_snapshot_import", None)
        preview = preview_shared(entry) if callable(preview_shared) else None
        return {"preview": preview}

    @router.post("/downloads/bundles/{bundle_key}")
    def install_download_bundle(bundle_key: str):
        service = getattr(ctx, "model_download", None)
        keys = quick_start_bundles_for_platform().get(bundle_key)
        if service is None:
            raise HTTPException(status_code=500, detail="Model download service is unavailable.")
        if keys is None:
            raise HTTPException(status_code=404, detail=f"Install bundle '{bundle_key}' was not found.")
        results: list[dict[str, Any]] = []
        for key in keys:
            entry = service.find_catalog(key)
            if entry is None or bool(getattr(entry, "coming_soon", False)):
                results.append({"key": key, "status": "unavailable"})
                continue
            copy_shared = getattr(service, "copy_shared_catalog_asset_to_primary", None)
            if callable(copy_shared):
                def copy_shared_and_refresh():
                    copied = copy_shared(entry)
                    if copied:
                        try:
                            _refresh_model_inventory_after_sort(ctx)
                        except Exception:
                            logger.exception("Could not refresh model inventory after bundle shared-root import")
                    return copied

                try:
                    copied = _run_exclusive_pro_gpu_operation(ctx, copy_shared_and_refresh)
                except ValueError as exc:
                    results.append({"key": key, "status": "insufficient_space", "error": str(exc)})
                    continue
                if copied:
                    results.append({
                        "key": key,
                        "status": "copied_from_shared_root",
                        "source": copied.get("source"),
                        "path": copied.get("target"),
                    })
                    continue
            if service.is_catalog_installed(entry):
                results.append({"key": key, "status": "already_installed"})
                continue
            if bool(getattr(entry, "snapshot", False)):
                preview_snapshot = getattr(service, "preview_shared_catalog_snapshot_import", None)
                try:
                    preview = preview_snapshot(entry) if callable(preview_snapshot) else None
                except Exception as exc:
                    logger.exception("Could not inspect shared snapshot candidate for bundle item %s", key)
                    results.append({
                        "key": key,
                        "status": "shared_snapshot_preview_failed",
                        "error": str(exc),
                    })
                    continue
                if preview is not None:
                    preview_payload = {
                        "source": str(preview.get("source") or ""),
                        "target": str(preview.get("target") or ""),
                        "sizeBytes": int(preview.get("sizeBytes") or 0),
                        "requiredBytes": int(preview.get("requiredBytes") or 0),
                        "freeBytes": int(preview.get("freeBytes") or 0),
                        "enoughSpace": bool(preview.get("enoughSpace")),
                    }
                    results.append({
                        "key": key,
                        "status": (
                            "shared_snapshot_confirmation_required"
                            if preview_payload["enoughSpace"]
                            else "shared_snapshot_insufficient_space"
                        ),
                        "sharedSnapshotPreview": preview_payload,
                    })
                    # Snapshot imports require the same explicit size review
                    # and confirmation as the catalog-row flow. Never fall
                    # through to a network download while a verified shared
                    # copy is available.
                    continue
            def place_misplaced_and_refresh():
                placed = service.place_misplaced_catalog_asset(entry)
                if placed:
                    try:
                        _refresh_model_inventory_after_sort(ctx)
                    except Exception:
                        logger.exception("Could not refresh model inventory after bundle asset placement")
                return placed

            if _run_exclusive_pro_gpu_operation(ctx, place_misplaced_and_refresh):
                results.append({"key": key, "status": "placed"})
                continue
            if not _catalog_entry_can_download(entry):
                results.append({"key": key, "status": "manual_access_required"})
                continue
            try:
                def download_and_refresh():
                    path = service.download_catalog(key)
                    try:
                        _refresh_model_inventory_after_sort(ctx)
                    except Exception:
                        logger.exception("Could not refresh model inventory after bundle item download: %s", key)
                    return path

                path = _run_exclusive_pro_gpu_operation(ctx, download_and_refresh)
                results.append({"key": key, "status": "downloaded", "path": str(path)})
            except HTTPException as exc:
                if exc.status_code == 409:
                    results.append({"key": key, "status": "deferred-active-operation", "error": str(exc.detail)})
                else:
                    results.append({"key": key, "status": "failed", "error": str(exc.detail)})
            except Exception as exc:
                logger.exception("Bundle item download failed: %s", key)
                results.append({"key": key, "status": "failed", "error": str(exc)})
        try:
            _run_exclusive_pro_gpu_operation(ctx, lambda: _refresh_model_inventory_after_sort(ctx))
        except HTTPException as exc:
            if exc.status_code != 409:
                logger.exception("Could not refresh model inventory after bundle installation")
        except Exception:
            logger.exception("Could not refresh model inventory after bundle installation")
        payload = _download_payload(ctx)
        successful_statuses = {"already_installed", "copied_from_shared_root", "copied_snapshot_from_shared_root", "placed", "downloaded"}
        successful_count = sum(item.get("status") in successful_statuses for item in results)
        has_deferred = any(item.get("status") == "deferred-active-operation" for item in results)
        confirmation_count = sum(item.get("status") == "shared_snapshot_confirmation_required" for item in results)
        insufficient_space_count = sum(item.get("status") == "shared_snapshot_insufficient_space" for item in results)
        if has_deferred and successful_count == 0:
            installation_status = "deferred-active-operation"
        elif confirmation_count and successful_count + confirmation_count == len(keys):
            installation_status = "confirmation-required"
        elif insufficient_space_count and successful_count + insufficient_space_count == len(keys):
            installation_status = "insufficient-space"
        else:
            installation_status = "all-items-installed" if successful_count == len(keys) else ("partial" if successful_count else "blocked")
        payload["bundleInstall"] = {
            "key": bundle_key,
            "installationStatus": installation_status,
            "readiness": "refresh-required",
            "items": results,
        }
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
    def models_reorganize(payload: ProModelReorganizePayload):
        flags = getattr(ctx, "flags", None)
        if flags is None:
            raise HTTPException(status_code=500, detail="Runtime flags are unavailable.")
        plan_id = str(payload.plan_id or "")
        with _PRO_MODEL_REORGANIZE_LOCK:
            plans = getattr(ctx, "_pro_model_reorganize_plans", {})
            reviewed = plans.get(plan_id) if isinstance(plans, dict) else None
            if not reviewed or time.monotonic() - float(reviewed.get("createdAt", 0)) > 600:
                if isinstance(plans, dict):
                    plans.pop(plan_id, None)
                raise HTTPException(status_code=409, detail="The placement preview expired. Review a fresh plan before moving models.")
            approved_actions = list(reviewed.get("actions") or [])

            def apply_reviewed_plan():
                current_actions = plan_model_reorganize(flags)
                if _model_reorganize_plan_id(current_actions) != plan_id:
                    plans.pop(plan_id, None)
                    raise HTTPException(
                        status_code=409,
                        detail="The model files changed after the preview. Review the updated placement plan before moving anything.",
                    )
                return reorganize_models(
                    flags,
                    approved_moves=_approved_model_reorganize_moves(approved_actions),
                )

            def apply_and_refresh():
                actions = apply_reviewed_plan()
                return _model_sort_response(ctx, actions)

            response = _run_exclusive_pro_gpu_operation(ctx, apply_and_refresh)
            plans.pop(plan_id, None)
        return response

    @router.get("/models/reorganize/plan")
    def models_reorganize_plan():
        flags = getattr(ctx, "flags", None)
        if flags is None:
            raise HTTPException(status_code=500, detail="Runtime flags are unavailable.")
        with _PRO_MODEL_REORGANIZE_LOCK:
            actions = plan_model_reorganize(flags)
            plan_id = _model_reorganize_plan_id(actions)
            plans = getattr(ctx, "_pro_model_reorganize_plans", None)
            if not isinstance(plans, dict):
                plans = {}
                setattr(ctx, "_pro_model_reorganize_plans", plans)
            now = time.monotonic()
            plans[plan_id] = {"createdAt": now, "actions": list(actions)}
            for old_id, old_plan in list(plans.items()):
                if now - float(old_plan.get("createdAt", 0)) > 600:
                    plans.pop(old_id, None)
            while len(plans) > 8:
                plans.pop(next(iter(plans)))
            return _model_sort_response(ctx, actions, planned=True, plan_id=plan_id)

    def _model_inventory_scan_page(
        report: dict[str, object], *, scan_id: str, offset: int, limit: int, query: str
    ) -> dict[str, object]:
        all_assets = report.get("assets", [])
        if not isinstance(all_assets, list):
            all_assets = []
        needle = query.strip().casefold()
        if needle:
            matched_assets = []
            for asset in all_assets:
                if not isinstance(asset, dict):
                    continue
                searchable = " ".join(str(value) for key, value in asset.items() if key != "signals")
                signals = asset.get("signals")
                if isinstance(signals, dict):
                    searchable += " " + " ".join(str(value) for value in signals.values())
                if needle in searchable.casefold():
                    matched_assets.append(asset)
        else:
            matched_assets = all_assets
        total = len(matched_assets)
        page_assets = matched_assets[offset:offset + limit]
        next_offset = offset + len(page_assets)
        return {
            **report,
            "scanId": scan_id,
            "assets": page_assets,
            "offset": offset,
            "limit": limit,
            "matchedCount": total,
            "assetsTruncated": max(0, total - next_offset),
            "hasMore": next_offset < total,
            "nextOffset": next_offset if next_offset < total else None,
            "query": query,
        }

    @router.post("/models/scan")
    def models_scan(limit: int = Query(50, ge=1, le=250)):
        flags = getattr(ctx, "flags", None)
        if flags is None:
            raise HTTPException(status_code=500, detail="Runtime flags are unavailable.")

        def scan_and_invalidate():
            report = scan_model_inventory_report(flags, proposal_limit=None)
            scan_id = uuid4().hex
            scans = getattr(ctx, "_pro_model_inventory_scans", None)
            if not isinstance(scans, dict):
                scans = {}
                setattr(ctx, "_pro_model_inventory_scans", scans)
            now = time.monotonic()
            for old_id, old_scan in list(scans.items()):
                if now - float(old_scan.get("createdAt", 0)) > 600:
                    scans.pop(old_id, None)
            while len(scans) >= 4:
                scans.pop(next(iter(scans)))
            scans[scan_id] = {"createdAt": now, "report": report}
            backend = getattr(getattr(ctx, "generation", None), "backend", None)
            for name in ("invalidate_checkpoints", "invalidate_loras", "invalidate_embeddings", "invalidate_vaes"):
                invalidate = getattr(backend, name, None)
                if callable(invalidate):
                    invalidate()
            setattr(ctx, "_pro_capability_cache", None)
            return _model_inventory_scan_page(report, scan_id=scan_id, offset=0, limit=limit, query="")

        return _run_exclusive_pro_gpu_operation(ctx, scan_and_invalidate)

    @router.get("/models/scan/{scan_id}")
    def models_scan_page(
        scan_id: str,
        offset: int = Query(0, ge=0),
        limit: int = Query(50, ge=1, le=250),
        query: str = Query("", max_length=200),
    ):
        scans = getattr(ctx, "_pro_model_inventory_scans", None)
        entry = scans.get(scan_id) if isinstance(scans, dict) else None
        if not isinstance(entry, dict) or time.monotonic() - float(entry.get("createdAt", 0)) > 600:
            if isinstance(scans, dict):
                scans.pop(scan_id, None)
            raise HTTPException(status_code=404, detail="Model scan expired. Run Scan model roots again.")
        report = entry.get("report")
        if not isinstance(report, dict):
            raise HTTPException(status_code=404, detail="Model scan is unavailable. Run Scan model roots again.")
        return _model_inventory_scan_page(report, scan_id=scan_id, offset=offset, limit=limit, query=query)

    def _placement_reparse_point(path: Path) -> bool:
        try:
            info = path.lstat()
        except OSError:
            return False
        attributes = int(getattr(info, "st_file_attributes", 0))
        return stat.S_ISLNK(info.st_mode) or bool(attributes & 0x400)

    def _placement_is_relative(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def _placement_scan_asset(scan_id: str, path: str) -> tuple[dict[str, Any], dict[str, Any]]:
        scans = getattr(ctx, "_pro_model_inventory_scans", None)
        scan = scans.get(scan_id) if isinstance(scans, dict) else None
        if not isinstance(scan, dict) or time.monotonic() - float(scan.get("createdAt", 0)) > 600:
            if isinstance(scans, dict):
                scans.pop(scan_id, None)
            raise HTTPException(status_code=404, detail="Model scan expired. Scan model roots again before placement.")
        report = scan.get("report")
        if not isinstance(report, dict):
            raise HTTPException(status_code=404, detail="Model scan is unavailable. Scan model roots again before placement.")
        assets = report.get("assets")
        if not isinstance(assets, list):
            raise HTTPException(status_code=409, detail="The scanned proposal is no longer available.")
        asset = next((item for item in assets if isinstance(item, dict) and item.get("path") == path), None)
        if asset is None:
            raise HTTPException(status_code=404, detail="That exact path was not included in this scan.")
        if asset.get("placement") not in {"candidate", "reorganize-candidate", "manual-review"}:
            raise HTTPException(status_code=422, detail="Only scanned model files can be previewed for placement.")
        return scan, asset

    def _prepare_shared_asset_placement(flags: RuntimeFlags, asset: dict[str, Any]) -> dict[str, Any]:
        raw_path = str(asset.get("path") or "")
        source = Path(raw_path)
        if not source.is_absolute():
            raise HTTPException(status_code=422, detail="The scanned source path is not absolute.")
        try:
            source_resolved = source.resolve(strict=True)
            source_stat = source.lstat()
        except (OSError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail="The scanned source is no longer available.") from exc
        if str(source_resolved) != raw_path or _placement_reparse_point(source):
            raise HTTPException(status_code=422, detail="Symlink or reparse-point sources cannot be copied.")
        if not stat.S_ISREG(source_stat.st_mode):
            raise HTTPException(status_code=422, detail="Only regular model files can be copied from shared roots.")

        primary = flags.resolved_models_dir().resolve()
        try:
            source_resolved.relative_to(primary)
            raise HTTPException(status_code=422, detail="Files already inside the primary models root cannot use shared-root import.")
        except ValueError:
            pass
        extra_roots = [
            Path(root).resolve()
            for root in (*flags.resolved_extra_model_dirs(), *flags.resolved_extra_ckpt_dirs())
        ]
        shared_root = next((root for root in extra_roots if _placement_is_relative(source_resolved, root)), None)
        if shared_root is None:
            raise HTTPException(status_code=422, detail="The scanned source is outside configured extra model roots.")
        for root in extra_roots:
            if root == primary or _placement_is_relative(root, primary) or _placement_is_relative(primary, root):
                raise HTTPException(status_code=409, detail="Primary and shared model roots overlap; placement is disabled for safety.")

        record = classify_model_file(source_resolved, model_inventory_roots(flags))
        if record is None:
            raise HTTPException(status_code=409, detail="The scanned file is no longer a supported model asset.")
        confident, reason = _is_confident(record)
        if not confident or not record.should_move:
            raise HTTPException(status_code=422, detail=reason or "The asset no longer has a confident placement.")
        if (
            record.family != asset.get("family")
            or record.architecture != asset.get("architecture")
            or record.recommended_subdir != asset.get("recommendedSubdir")
        ):
            raise HTTPException(status_code=409, detail="The model classification changed after scanning. Scan model roots again.")

        relative_text = str(record.recommended_subdir or "").replace("\\", "/")
        relative = PurePosixPath(relative_text)
        if not relative_text or ":" in relative_text or relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise HTTPException(status_code=422, detail="The recommended destination is not a safe relative model folder.")
        destination = primary.joinpath(*relative.parts, source.name)
        cursor = primary
        try:
            for part in destination.parent.relative_to(primary).parts:
                cursor = cursor / part
                if cursor.exists() or cursor.is_symlink():
                    if _placement_reparse_point(cursor) or not cursor.is_dir():
                        raise HTTPException(status_code=422, detail="A destination parent is a symlink, reparse point, or non-folder.")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="The recommended destination escapes the primary models root.") from exc
        try:
            destination_resolved = destination.resolve(strict=False)
            destination_resolved.relative_to(primary)
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="The recommended destination escapes the primary models root.") from exc
        if any(_placement_is_relative(destination_resolved, root) for root in extra_roots):
            raise HTTPException(status_code=409, detail="The destination overlaps a shared model root.")

        try:
            disk_path = cursor
            while not disk_path.exists() and disk_path != disk_path.parent:
                disk_path = disk_path.parent
            free_bytes = shutil.disk_usage(disk_path).free
        except (OSError, RuntimeError, ValueError) as exc:
            if isinstance(exc, HTTPException):
                raise
            raise HTTPException(status_code=409, detail="Could not safely inspect destination folders or free space.") from exc
        collision = (
            destination.exists()
            or destination.is_symlink()
            or destination_resolved.exists()
            or destination_resolved.is_symlink()
        )
        size_bytes = int(source_stat.st_size)
        required_bytes = size_bytes + max(64 * 1024 * 1024, (size_bytes + 49) // 50)
        return {
            "source": source_resolved,
            "sourceStat": source_stat,
            "sourceRoot": shared_root,
            "destination": destination_resolved,
            "record": record,
            "sizeBytes": size_bytes,
            "requiredFreeBytes": required_bytes,
            "availableFreeBytes": int(free_bytes),
            "collision": collision,
        }

    @router.post("/models/scan/{scan_id}/placements/preview")
    def models_shared_placement_preview(scan_id: str, payload: ProModelRootPlacementPreviewPayload):
        if payload.scan_id != scan_id:
            raise HTTPException(status_code=422, detail="The scan ID in the path and body must match.")
        flags = getattr(ctx, "flags", None)
        if flags is None:
            raise HTTPException(status_code=500, detail="Runtime flags are unavailable.")
        scan, asset = _placement_scan_asset(scan_id, payload.path)
        details = _prepare_shared_asset_placement(flags, asset)
        def source_sha256(path: Path) -> str:
            digest = hashlib.sha256()
            with path.open("rb") as source_file:
                for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        source_digest = source_sha256(details["source"])
        plans = getattr(ctx, "_pro_shared_model_placement_plans", None)
        if not isinstance(plans, dict):
            plans = {}
            setattr(ctx, "_pro_shared_model_placement_plans", plans)
        now = time.monotonic()
        for old_id, old_plan in list(plans.items()):
            if now - float(old_plan.get("createdAt", 0)) > 300:
                plans.pop(old_id, None)
        while len(plans) >= 32:
            plans.pop(next(iter(plans)))
        plan_id = uuid4().hex
        plans[plan_id] = {
            "createdAt": now,
            "scanId": scan_id,
            "path": str(details["source"]),
            "sourceRoot": str(details["sourceRoot"]),
            "destination": str(details["destination"]),
            "sizeBytes": details["sizeBytes"],
            "mtimeNs": details["sourceStat"].st_mtime_ns,
            "inode": details["sourceStat"].st_ino,
            "sha256": source_digest,
            "family": details["record"].family,
            "architecture": details["record"].architecture,
            "recommendedSubdir": details["record"].recommended_subdir,
            "headerIdentifiers": dict(details["record"].header_identifiers),
            "collision": details["collision"],
        }
        can_apply = (
            not details["collision"]
            and details["availableFreeBytes"] >= details["requiredFreeBytes"]
        )
        return {
            "planId": plan_id,
            "scanId": scan_id,
            "source": str(details["source"]),
            "destination": str(details["destination"]),
            "sizeBytes": details["sizeBytes"],
            "requiredFreeBytes": details["requiredFreeBytes"],
            "availableFreeBytes": details["availableFreeBytes"],
            "collision": details["collision"],
            "canApply": can_apply,
            "status": "ready" if can_apply else "collision" if details["collision"] else "insufficient_space",
            "expiresInSeconds": 300,
        }

    @router.post("/models/scan/placements/apply")
    def models_shared_placement_apply(payload: ProModelRootPlacementApplyPayload):
        plans = getattr(ctx, "_pro_shared_model_placement_plans", None)
        plan = plans.get(payload.plan_id) if isinstance(plans, dict) else None
        if not isinstance(plan, dict) or time.monotonic() - float(plan.get("createdAt", 0)) > 300:
            if isinstance(plans, dict):
                plans.pop(payload.plan_id, None)
            raise HTTPException(status_code=409, detail="The placement preview expired. Review a fresh preview.")
        scan, asset = _placement_scan_asset(str(plan["scanId"]), str(plan["path"]))
        flags = getattr(ctx, "flags", None)
        if flags is None:
            raise HTTPException(status_code=500, detail="Runtime flags are unavailable.")

        def source_sha256(path: Path) -> str:
            digest = hashlib.sha256()
            with path.open("rb") as source_file:
                for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        def apply_copy():
            current = _prepare_shared_asset_placement(flags, asset)
            source_stat = current["sourceStat"]
            current_digest = source_sha256(current["source"])
            if (
                str(current["source"]) != plan["path"]
                or str(current["sourceRoot"]) != plan["sourceRoot"]
                or str(current["destination"]) != plan["destination"]
                or current["sizeBytes"] != plan["sizeBytes"]
                or source_stat.st_mtime_ns != plan["mtimeNs"]
                or source_stat.st_ino != plan["inode"]
                or current_digest != plan["sha256"]
                or current["record"].header_identifiers != plan["headerIdentifiers"]
            ):
                raise HTTPException(status_code=409, detail="The source or its classification changed after preview. Review a new placement preview.")
            if plan["collision"] or current["collision"]:
                raise HTTPException(status_code=409, detail="The destination already exists. No file was overwritten.")
            if current["availableFreeBytes"] < current["requiredFreeBytes"]:
                raise HTTPException(status_code=409, detail="There is not enough free space in the primary models volume.")

            source = current["source"]
            destination = current["destination"]
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                # Re-check root confinement and all created parents before staging.
                checked = _prepare_shared_asset_placement(flags, asset)
                if checked["destination"] != destination or checked["collision"]:
                    raise HTTPException(status_code=409, detail="The destination changed during placement. No file was overwritten.")
                with tempfile.TemporaryDirectory(prefix=".aiwf-placement-", dir=destination.parent) as staging_dir:
                    staged = Path(staging_dir) / source.name
                    shutil.copy2(source, staged)
                    staged_stat = staged.stat()
                    after_copy_stat = source.stat()
                    if (
                        staged_stat.st_size != current["sizeBytes"]
                        or after_copy_stat.st_size != source_stat.st_size
                        or after_copy_stat.st_mtime_ns != source_stat.st_mtime_ns
                        or after_copy_stat.st_ino != source_stat.st_ino
                    ):
                        raise HTTPException(status_code=409, detail="The source changed while it was being copied. Partial data was removed.")
                    staged_record = classify_model_file(staged, model_inventory_roots(flags))
                    staged_digest = source_sha256(staged)
                    after_copy_digest = source_sha256(source)
                    if (
                        staged_record is None
                        or staged_digest != plan["sha256"]
                        or after_copy_digest != plan["sha256"]
                        or staged_record.family != current["record"].family
                        or staged_record.architecture != current["record"].architecture
                        or staged_record.recommended_subdir != current["record"].recommended_subdir
                        or staged_record.header_identifiers != current["record"].header_identifiers
                    ):
                        raise HTTPException(status_code=409, detail="The copied file failed model-header verification. Partial data was removed.")
                    # Hard-link commit is atomic and fails if the destination already exists.
                    os.link(staged, destination)
            except HTTPException:
                raise
            except FileExistsError as exc:
                raise HTTPException(status_code=409, detail="The destination appeared during placement. No file was overwritten.") from exc
            except (OSError, RuntimeError, ValueError) as exc:
                raise HTTPException(status_code=409, detail=f"Could not safely copy the shared model asset: {exc}") from exc
            try:
                _refresh_model_inventory_after_sort(ctx)
            except Exception:
                logger.exception("Could not refresh inventory after shared-root model placement")
            plans.pop(payload.plan_id, None)
            return {
                "status": "copied_from_shared_root",
                "source": str(source),
                "destination": str(destination),
                "sourcePreserved": source.exists(),
                "inventoryRefresh": "complete-or-recoverable",
            }

        with _PRO_MODEL_REORGANIZE_LOCK:
            return _run_exclusive_pro_gpu_operation(ctx, apply_copy)

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
        def sort_uploaded_model():
            actions = sort_inbox_models(flags)
            return _model_sort_response(ctx, actions, uploaded_path=dest)

        try:
            payload = _run_exclusive_pro_gpu_operation(ctx, sort_uploaded_model)
        except HTTPException as exc:
            if exc.status_code == 409:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"The upload is safely staged at {SORT_INBOX_DIRNAME}/{dest.name}, "
                        "but automatic placement is waiting for the active model operation to finish. "
                        "Retry model sorting when the current generation or model switch is done."
                    ),
                ) from exc
            raise
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
        from aiwf.services.model_startup import pro_model_load_lock

        lock = pro_model_load_lock(ctx)
        if not lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="Another model operation is already in progress.")
        try:
            _assert_pro_model_load_idle(ctx)
            prior_loaded_model = str((getattr(ctx, "_pro_model_load_state", {}) or {}).get("modelId") or "")
            setattr(ctx, "_pro_model_load_state", {"status": "unloading", "modelId": "", "detail": "Unloading the active image model."})
            result = _unload_generation_model(ctx)
            if prior_loaded_model:
                confirm_route_residency(ctx, "image.txt2img", prior_loaded_model, resident=False)
            setattr(ctx, "_pro_model_load_state", {"status": "unloaded", "modelId": "", "detail": "No image model is loaded."})
            return result
        except HTTPException:
            if str((getattr(ctx, "_pro_model_load_state", {}) or {}).get("status") or "") == "unloading":
                setattr(ctx, "_pro_model_load_state", {"status": "failed", "modelId": "", "detail": "Model unload was not completed."})
            raise
        finally:
            lock.release()

    @router.post("/models/prepare")
    def models_prepare(payload: ProGeneratePayload):
        from aiwf.services.model_startup import pro_model_load_lock

        mode = (payload.mode or "").strip().lower()
        if mode not in {"video", "sana", "sana_video", "wan"}:
            raise HTTPException(status_code=422, detail="Route preparation currently accepts video models only.")
        lock = pro_model_load_lock(ctx)
        if not lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="Another model load or route preparation is already in progress.")
        try:
            _assert_pro_model_load_idle(ctx)
            if (
                _image_generation_running(ctx)
                or _image_generation_pending(ctx)
                or _pro_video_job_running(ctx)
                or _pro_workflow_runs_active(ctx)
            ):
                raise HTTPException(status_code=409, detail="Wait for active generation or workflow jobs to finish before preparing another route.")
            checkpoint_id = str(payload.checkpoint_id or "")
            engine_id = _checkpoint_engine_id(ctx, checkpoint_id) if checkpoint_id else "sana_video"
            if engine_id not in {"sana_video", "wan", "ltx"}:
                raise HTTPException(status_code=422, detail="Choose a Wan, Sana Video, or LTX model before preparing a video route.")
            if engine_id == "ltx":
                _assert_ltx_engine_not_installing()
            preparation_id = uuid4().hex
            setattr(ctx, "_pro_route_prepare_state", {"status": "preparing", "operationId": preparation_id})
            return _prepare_pro_video_route(ctx, payload)
        finally:
            route_state = getattr(ctx, "_pro_route_prepare_state", None)
            if isinstance(route_state, dict) and route_state.get("operationId") == locals().get("preparation_id"):
                setattr(ctx, "_pro_route_prepare_state", None)
            lock.release()

    @router.post("/engines/ltx/install")
    def install_ltx_engine():
        from aiwf.services.model_startup import pro_model_load_lock

        lock = pro_model_load_lock(ctx)
        if not lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="Another model operation is in progress. Retry LTX setup when it finishes.")
        try:
            _assert_pro_model_load_idle(ctx)
            if _image_generation_running(ctx) or _image_generation_pending(ctx) or _pro_video_job_running(ctx) or _pro_workflow_runs_active(ctx):
                raise HTTPException(status_code=409, detail="A generation or workflow job is active. Wait for it to finish before starting LTX setup.")
            try:
                return start_ltx_engine_install()
            except FileNotFoundError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except Exception as exc:
                logger.exception("Could not start the LTX engine installer")
                raise HTTPException(status_code=500, detail=f"Could not start LTX engine setup: {type(exc).__name__}: {exc}") from exc
        finally:
            lock.release()

    @router.get("/engines/ltx/install-status")
    def get_ltx_engine_install_status():
        status = ltx_engine_install_status()
        if status.get("status") == "finished" and status.get("exitCode") == 0:
            _refresh_ltx_worker_registry(ctx)
        return status

    @router.post("/engines/qwen_nunchaku/install")
    def install_qwen_nunchaku_engine():
        from aiwf.services.model_startup import pro_model_load_lock

        lock = pro_model_load_lock(ctx)
        if not lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="Another model operation is in progress. Retry Qwen setup when it finishes.")
        try:
            _assert_pro_model_load_idle(ctx)
            if _image_generation_running(ctx) or _image_generation_pending(ctx) or _pro_video_job_running(ctx) or _pro_workflow_runs_active(ctx):
                raise HTTPException(status_code=409, detail="A generation or workflow job is active. Wait for it to finish before starting Qwen setup.")
            current = qwen_nunchaku_engine_install_status(ctx.flags.data_dir)
            if bool(current.get("running")) or str(current.get("status") or "").strip().lower() == "running":
                raise HTTPException(status_code=409, detail="Qwen Nunchaku setup is already in progress.")
            from aiwf.services.qwen_nunchaku import clear_qwen_nunchaku_runtime_probe_cache

            clear_qwen_nunchaku_runtime_probe_cache()
            return start_qwen_nunchaku_engine_install(data_root=ctx.flags.data_dir)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Could not start the Qwen Nunchaku runtime installer")
            raise HTTPException(status_code=500, detail=f"Could not start Qwen Nunchaku setup: {type(exc).__name__}: {exc}") from exc
        finally:
            lock.release()

    @router.get("/engines/qwen_nunchaku/install-status")
    def get_qwen_nunchaku_engine_install_status():
        install = qwen_nunchaku_engine_install_status(ctx.flags.data_dir)
        from aiwf.services.qwen_nunchaku import QwenNunchakuService, clear_qwen_nunchaku_runtime_probe_cache

        if install.get("status") == "finished":
            clear_qwen_nunchaku_runtime_probe_cache()

        runtime = QwenNunchakuService(ctx.flags).status()
        return {**install, "runtimeReady": runtime.ready, "runtimeMessages": list(runtime.messages)}

    @router.post("/models/load")
    def models_load(payload: ProModelLoadPayload):
        from aiwf.services.model_startup import pro_model_load_lock

        lock = pro_model_load_lock(ctx)
        if not lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="Another image model load is already in progress.")
        try:
            _assert_pro_model_load_idle(ctx)
            return _load_pro_image_model(ctx, payload)
        finally:
            lock.release()

    def _load_pro_image_model(ctx: Any, payload: ProModelLoadPayload):
        model_id = payload.model_id.strip()
        setattr(ctx, "_pro_model_load_state", {
            "status": "loading", "modelId": model_id,
            "detail": "Checking model readiness and preparing its support assets.",
        })
        if not model_id or not _is_checkpoint_id_selectable(ctx, model_id):
            detail = _checkpoint_error_context(ctx, model_id)
            setattr(ctx, "_pro_model_load_state", {"status": "not-ready", "modelId": model_id, "detail": "The selected model is not ready for the Pro image route."})
            raise HTTPException(status_code=422, detail=_pro_error_detail(
                "The selected model is not ready for the Pro image route.",
                model=detail,
            ))
        engine_id = _checkpoint_engine_id(ctx, model_id)
        if engine_id in {"wan", "sana_video", "ltx"}:
            setattr(ctx, "_pro_model_load_state", {"status": "not-ready", "modelId": model_id, "detail": "Video models are loaded by their video route."})
            raise HTTPException(status_code=422, detail="Video models are prepared by their video route when a video job starts.")
        if engine_id == "qwen_nunchaku":
            _assert_qwen_nunchaku_engine_not_installing(ctx)
        if _image_generation_running(ctx) or _image_generation_pending(ctx) or _pro_video_job_running(ctx) or _pro_workflow_runs_active(ctx):
            setattr(ctx, "_pro_model_load_state", {"status": "deferred", "modelId": model_id, "detail": "A generation job is active."})
            raise HTTPException(status_code=409, detail="A generation job is active. Wait for it to finish before switching models.")
        generation = getattr(ctx, "generation", None)
        backend = getattr(generation, "backend", None)
        can_preload = getattr(backend, "can_preload_checkpoint_locally", None)
        if not callable(can_preload) or not can_preload(model_id):
            setattr(ctx, "_pro_model_load_state", {"status": "not-ready", "modelId": model_id, "detail": "The selected model's local files are incomplete for loading."})
            raise HTTPException(status_code=422, detail="The selected model's local files are incomplete for loading.")
        try:
            _release_cached_sana_video(ctx)
            _release_cached_wan_model(ctx)
            _release_cached_audio_model(ctx)
            _unload_cached_ltx_model(ctx)
        except HTTPException as exc:
            state = {
                "status": "deferred" if exc.status_code == 409 else "failed",
                "modelId": model_id,
                "detail": str(exc.detail or exc),
            }
            setattr(ctx, "_pro_model_load_state", state)
            raise
        except Exception as exc:
            setattr(ctx, "_pro_model_load_state", {
                "status": "failed",
                "modelId": model_id,
                "detail": f"Could not release cached generation models: {exc}",
            })
            raise
        from aiwf.services.model_startup import _gpu_headroom_status

        selectable, _blocked = _selectable_checkpoint_payloads(ctx)
        selected_model = next((item for item in selectable if str(item.get("id") or "") == model_id), None)
        headroom_issue = _gpu_headroom_status(ctx, selected_model or {"sizeBytes": 0})
        if headroom_issue:
            state = {"status": "deferred", "modelId": model_id, "detail": headroom_issue}
            setattr(ctx, "_pro_model_load_state", state)
            raise HTTPException(status_code=409, detail=headroom_issue)
        from aiwf.services.model_startup import pro_image_support_ids

        support_ids = pro_image_support_ids(
            ctx, model_id, engine_id, str((selected_model or {}).get("architecture") or "")
        )
        token = begin_route_operation(
            ctx,
            route="image.txt2img",
            model_id=model_id,
            setup_ready=True,
            support_ids=support_ids,
            detail="Loading the selected image model and its family-specific support assets.",
        )
        if not token:
            state = {
                "status": "deferred", "modelId": model_id,
                "detail": "Model loading deferred because the selected route changed during readiness checks.",
            }
            setattr(ctx, "_pro_model_load_state", state)
            raise HTTPException(status_code=409, detail=state["detail"])
        try:
            active_tenant = getattr(getattr(ctx, "supervisor", None), "active_tenant", None)
            tenant_value = str(getattr(active_tenant, "value", active_tenant) or "").strip().lower()
            if tenant_value not in {"", "idle", "none"}:
                setattr(ctx, "_pro_model_load_state", {"status": "deferred", "modelId": model_id, "detail": f"Cannot switch models while {tenant_value} owns the GPU."})
                raise HTTPException(status_code=409, detail=f"Cannot switch models while {tenant_value} owns the GPU.")
            setattr(ctx, "_pro_model_load_state", {
                "status": "loading", "modelId": model_id,
                "detail": "Loading the selected model and its family-specific support assets.",
            })
            mark_route_running(ctx, "image.txt2img", token, "Image model and support assets are loading.")
            generation.load_checkpoint(model_id)
            is_loaded = getattr(backend, "is_checkpoint_loaded", None)
            confirmed_loaded = bool(callable(is_loaded) and is_loaded(model_id))
            if not confirmed_loaded:
                finish_route_operation(
                    ctx, "image.txt2img", token, success=False,
                    detail="Image backend returned without confirming the selected model is resident.",
                )
                state = {
                    "status": "load-unconfirmed", "modelId": model_id,
                    "detail": "The image backend returned without confirming that the selected model is loaded.",
                }
                setattr(ctx, "_pro_model_load_state", state)
                raise HTTPException(status_code=503, detail=state["detail"])
            finish_route_operation(ctx, "image.txt2img", token, success=True, detail="Selected image model is resident and confirmed by its backend.", resident=True)
            confirm_route_residency(ctx, "image.txt2img", model_id, resident=True)
            startup_default_saved = _persist_last_checkpoint_id(ctx, model_id)
            state = {
                "status": "loaded", "modelId": model_id,
                "detail": (
                    "Selected model and its family-specific support assets are loaded and saved as the startup default. No generation was started."
                    if startup_default_saved else
                    "Selected model and its family-specific support assets are loaded for this session, but the startup default could not be saved. No generation was started."
                ),
            }
            setattr(ctx, "_pro_model_load_state", state)
            return {
                "status": "loaded",
                "modelLoad": state,
                "startupDefaultSaved": startup_default_saved,
                "runtime": _runtime_summary(ctx),
            }
        except HTTPException as exc:
            finish_route_operation(ctx, "image.txt2img", token, success=False, detail=str(exc.detail))
            if str((getattr(ctx, "_pro_model_load_state", {}) or {}).get("status") or "") == "loading":
                setattr(ctx, "_pro_model_load_state", {"status": "failed", "modelId": model_id, "detail": str(exc.detail)})
            raise
        except Exception as exc:
            finish_route_operation(ctx, "image.txt2img", token, success=False, detail=str(exc))
            state = {"status": "failed", "modelId": model_id, "detail": f"Could not load selected model: {exc}"}
            setattr(ctx, "_pro_model_load_state", state)
            raise HTTPException(status_code=409, detail=state["detail"]) from exc

    @router.get("/outputs/{requested_path:path}")
    def output_asset(requested_path: str, thumb: int | None = Query(default=None, ge=32, le=2048)):
        path = _output_asset_path(ctx, requested_path)
        if path.suffix.lower() in _IMAGE_EXTENSIONS and image_artifact_dimensions(path) is None:
            raise HTTPException(status_code=422, detail="Output image artifact is corrupt, mismatched, or too large.")
        if thumb and path.suffix.lower() in _IMAGE_EXTENSIONS:
            try:
                with Image.open(path) as image:
                    rendered = image.convert("RGB")
                    rendered.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
                    buf = io.BytesIO()
                    rendered.save(buf, format="JPEG", quality=85)
                return Response(
                    content=buf.getvalue(),
                    media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=3600"},
                )
            except OSError:
                pass
        return FileResponse(path)

    @router.get("/outputs-index")
    def outputs_index(
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=40, ge=1, le=200),
    ):
        root = _safe_output_root(ctx)
        if root is None:
            return {"items": [], "total": 0, "offset": offset, "limit": limit}
        scan_limit = max(_RECENT_SCAN_LIMIT, offset + limit + 1)
        all_paths = _all_output_image_paths_from_disk(root, scan_limit=scan_limit)
        total = len(all_paths)
        items: list[dict[str, Any]] = []
        for path in all_paths[offset : offset + limit]:
            try:
                stat = path.stat()
            except OSError:
                continue
            infotext = _read_output_infotext(path)
            items.append(
                {
                    "path": _output_asset_url(ctx, path),
                    "mtime": stat.st_mtime,
                    "sizeBytes": stat.st_size,
                    **_settings_from_infotext(infotext),
                }
            )
        return {"items": items, "total": total, "offset": offset, "limit": limit}

    @router.post("/metadata/import")
    def metadata_import(payload: ProMetadataImportPayload):
        return _import_generation_metadata_from_image(payload)

    @router.post("/workflows/runs", status_code=202)
    def workflow_run_submit(payload: ProWorkflowRunPayload):
        if len(payload.workflow.steps) > 16:
            raise HTTPException(status_code=422, detail="A workflow may contain at most 16 nodes.")
        step_ids = [step.id for step in payload.workflow.steps]
        if len(step_ids) != len(set(step_ids)):
            raise HTTPException(status_code=422, detail="Workflow node IDs must be unique.")
        seed_image = _decode_pro_image_data_url(payload.source_image_data_url, "Source image")
        try:
            validate_workflow(payload.workflow, has_seed_image=seed_image is not None)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        from aiwf.services.model_startup import pro_model_load_lock

        operation_lock = pro_model_load_lock(ctx)
        if not operation_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="A model operation is in progress. Retry the workflow after it finishes.")
        try:
            _assert_pro_model_load_idle(ctx)
            service = _pro_workflow_run_service(ctx)
            idempotency_key = (payload.idempotency_key or "").strip() or None
            previous = service.lookup_idempotency(
                payload.workflow,
                seed_image=seed_image,
                idempotency_key=idempotency_key,
            )
            if previous is not None:
                return _pro_workflow_run_payload(ctx, previous)
            _validate_workflow_generation_targets(ctx, payload.workflow)
            step_route_metadata: dict[str, dict[str, str | None]] = {}
            for step in payload.workflow.steps:
                if getattr(step.type, "value", str(step.type)) not in {"txt2img", "img2img", "inpaint"}:
                    continue
                checkpoint_id = str(step.params.get("checkpoint_id") or "").strip()
                model_family = _checkpoint_engine_id(ctx, checkpoint_id)
                step_route_metadata[step.id] = {
                    "route": f"image.{model_family}",
                    "model_family": model_family,
                }
            if step_route_metadata:
                # Workflow nodes run asynchronously and use the shared image
                # backend. Release retained models from other modalities while
                # this submission still owns the model-operation lock, before
                # the worker can start its first image node.
                _release_cached_audio_model(ctx)
                _release_cached_wan_model(ctx)
                _release_cached_sana_video(ctx)
                _unload_cached_ltx_model(ctx)
            record = service.submit(
                payload.workflow,
                seed_image=seed_image,
                idempotency_key=idempotency_key,
                step_route_metadata=step_route_metadata,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        finally:
            operation_lock.release()
        return _pro_workflow_run_payload(ctx, record)

    @router.get("/workflows/runs/{run_id}")
    def workflow_run_status(run_id: str):
        if len(run_id) != 32 or any(ch not in "0123456789abcdef" for ch in run_id.lower()):
            raise HTTPException(status_code=404, detail="Workflow run not found.")
        record = _pro_workflow_run_service(ctx).get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Workflow run not found.")
        return _pro_workflow_run_payload(ctx, record)

    @router.post("/workflows/runs/{run_id}/cancel")
    def workflow_run_cancel(run_id: str):
        if len(run_id) != 32 or any(ch not in "0123456789abcdef" for ch in run_id.lower()):
            raise HTTPException(status_code=404, detail="Workflow run not found.")
        record = _pro_workflow_run_service(ctx).cancel(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Workflow run not found.")
        return _pro_workflow_run_payload(ctx, record)

    @router.post("/generate")
    def generate(payload: ProGeneratePayload):
        from aiwf.services.model_startup import pro_model_load_lock

        lock = pro_model_load_lock(ctx)
        if not lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="A model operation is already in progress. Retry after it finishes.")
        try:
            return _generate_request(payload)
        finally:
            lock.release()

    def _generate_request(payload: ProGeneratePayload):
        _assert_pro_model_load_idle(ctx)
        if _pro_workflow_runs_active(ctx):
            raise HTTPException(status_code=409, detail="A workflow generation job is active. Wait for it to finish before starting another route.")
        _assert_requested_pipeline_backend(ctx, payload)
        if (payload.mode or "").strip().lower() in {"video", "sana", "sana_video", "wan"}:
            if _pro_video_job_running(ctx) or _image_generation_running(ctx) or _image_generation_pending(ctx):
                raise HTTPException(status_code=409, detail="A generation job is already running. Stop it or wait for it to finish.")
            normalized_mode = (payload.mode or "").strip().lower()
            checkpoint_id = str(payload.checkpoint_id or "")
            if checkpoint_id in _LTX_PIPELINES and normalized_mode != "video":
                raise HTTPException(status_code=422, detail="LTX generation uses mode='video'.")
            if checkpoint_id in _LTX_PIPELINES:
                _assert_ltx_engine_not_installing()
            is_wan_route = normalized_mode == "wan" or (checkpoint_id and checkpoint_id in _wan_model_ids(ctx))
            _assert_video_route_checkpoint(
                ctx,
                checkpoint_id or None,
                sana_model_variant=payload.sana_model_variant,
            )
            _release_cached_audio_model(ctx)
            if checkpoint_id != "ltx:diffusers_2b":
                _unload_cached_ltx_model(ctx)
            if is_wan_route:
                _release_cached_sana_video(ctx)
            else:
                _release_cached_wan_model(ctx)
                if checkpoint_id in _LTX_PIPELINES:
                    _release_cached_sana_video(ctx)
            if checkpoint_id in _LTX_PIPELINES:
                return _generate_ltx_video_response(ctx, payload)
            if is_wan_route:
                return _generate_wan_video_response(ctx, payload)
            return _generate_sana_video_response(ctx, payload)
        if _pro_video_job_running(ctx):
            raise HTTPException(status_code=409, detail="A Sana video job is already running. Stop it or wait for it to finish.")
        if _image_generation_running(ctx) or _image_generation_pending(ctx):
            raise HTTPException(status_code=409, detail="An image generation job is already running. Stop it or wait for it to finish.")
        checkpoint_id = _checkpoint_id_from_payload(ctx, payload)
        if _checkpoint_engine_id(ctx, checkpoint_id) == "qwen_nunchaku":
            _assert_qwen_nunchaku_engine_not_installing(ctx)
        _assert_image_route_checkpoint(ctx, checkpoint_id)
        _assert_checkpoint_selectable(ctx, checkpoint_id)
        request = _generation_request(ctx, payload)
        _release_cached_audio_model(ctx)
        _release_cached_wan_model(ctx)
        _release_cached_sana_video(ctx)
        _unload_cached_ltx_model(ctx)
        image_backend = getattr(getattr(ctx, "generation", None), "backend", None)
        is_checkpoint_loaded = getattr(image_backend, "is_checkpoint_loaded", None)
        try:
            already_loaded = bool(callable(is_checkpoint_loaded) and is_checkpoint_loaded(checkpoint_id))
        except Exception:
            already_loaded = False
        if not already_loaded:
            from aiwf.services.model_startup import _gpu_headroom_status

            selectable, _blocked = _selectable_checkpoint_payloads(ctx)
            selected_model = next(
                (item for item in selectable if str(item.get("id") or "") == checkpoint_id),
                {"sizeBytes": 0},
            )
            headroom_issue = _gpu_headroom_status(ctx, selected_model)
            if headroom_issue:
                raise HTTPException(status_code=409, detail=headroom_issue)
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

        selected_checkpoint = _resolve_checkpoint_for_generation_guard(ctx, checkpoint_id)
        selected_checkpoint_data = _dump_model(selected_checkpoint) if selected_checkpoint is not None else {}
        engine_id = str(selected_checkpoint_data.get("engineId") or _checkpoint_engine_id(ctx, checkpoint_id))
        architecture = str(selected_checkpoint_data.get("architecture") or "")
        from aiwf.services.model_startup import pro_image_support_ids
        from aiwf.services.route_lifecycle import select_route

        support_ids = pro_image_support_ids(ctx, checkpoint_id, engine_id, architecture)
        select_route(
            ctx,
            route="image.txt2img",
            model_id=checkpoint_id,
            setup_ready=True,
            support_ids=support_ids,
            resident=already_loaded,
            detail="Selected for image generation; checking backend residency after the request.",
        )
        ctx._pro_model_load_state = {
            "status": "loaded" if already_loaded else "loading",
            "modelId": checkpoint_id,
            "detail": "Selected image model is already resident." if already_loaded else "Loading the selected image model for generation.",
        }

        def reconcile_image_residency() -> None:
            try:
                resident: bool | None = bool(callable(is_checkpoint_loaded) and is_checkpoint_loaded(checkpoint_id))
            except Exception:
                resident = None
            from aiwf.services.route_lifecycle import clear_route_residency, confirm_route_residency

            if resident is None:
                clear_route_residency(
                    ctx, "image.txt2img", checkpoint_id,
                    detail="The image backend could not confirm residency after generation.",
                )
                status = "load-unconfirmed"
                detail = "Image generation finished, but model residency could not be confirmed."
            else:
                confirm_route_residency(ctx, "image.txt2img", checkpoint_id, resident=resident)
                status = "loaded" if resident else "not-ready"
                detail = (
                    "The image backend confirms the selected model is resident."
                    if resident else "The image backend confirms the selected model is not resident."
                )
            ctx._pro_model_load_state = {"status": status, "modelId": checkpoint_id, "detail": detail}

        try:
            logger.info(
                "Pro image generation started: model=%s size=%sx%s steps=%s batch=%s",
                request.checkpoint_id or "default",
                request.width,
                request.height,
                request.steps,
                request.batch_size,
            )
            job, progress = _run_pro_image_generation(ctx, request, init_images=init_images, mask_images=mask_images)
            logger.info(
                "Pro image generation finished: job=%s state=%s message=%s",
                job.id,
                getattr(job.state, "value", job.state),
                getattr(job.progress, "message", "") if getattr(job, "progress", None) else "",
            )
        except Exception as exc:
            reconcile_image_residency()
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
        reconcile_image_residency()
        response = _generate_response(job)
        if progress:
            response["progress"] = progress
        return response

    @router.get("/audio/status")
    def audio_status(deep: bool = Query(default=False)):
        return JSONResponse(
            _audio_status_payload(ctx, deep=deep),
            headers={"Cache-Control": "no-store"},
        )

    # research mode for audio: off by default; on brings back non-commercial models (MusicGen,
    # MMAudio), labeled. Only the person at this PC can change it, never a paired phone or agent.
    @router.post("/audio/research-mode")
    def audio_research_mode(payload: ProAudioResearchModePayload, request: Request):
        _require_loopback(request)
        settings = getattr(ctx, "settings", None)
        save_settings = getattr(ctx, "save_settings", None)
        if settings is None or not callable(save_settings):
            raise HTTPException(status_code=500, detail="User settings are unavailable.")
        previous = bool(getattr(settings, "allow_noncommercial_audio_models", False))
        settings.allow_noncommercial_audio_models = bool(payload.enabled)
        try:
            save_settings()
        except Exception as exc:
            settings.allow_noncommercial_audio_models = previous
            raise HTTPException(status_code=500, detail=f"Could not save the research-mode setting: {exc}") from exc
        return _audio_status_payload(ctx, deep=False)

    @router.post("/audio/setup/minimum")
    def audio_setup_minimum():
        from aiwf.services.audio import AudioUnavailable

        try:
            return _run_exclusive_pro_gpu_operation(
                ctx,
                lambda: _audio_service(ctx).install_minimum(),
            )
        except HTTPException:
            raise
        except AudioUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @router.post("/audio/setup/engine/{engine}")
    def audio_setup_commercial_engine(engine: str):
        from aiwf.services.audio import AudioUnavailable

        def install():
            _audio_service(ctx).install_commercial_engine(engine)
            return _audio_status_payload(ctx, deep=False)

        try:
            return _run_exclusive_pro_gpu_operation(ctx, install)
        except HTTPException:
            raise
        except AudioUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @router.post("/audio/setup/mmaudio/{variant}")
    def audio_setup_mmaudio_variant(variant: str):
        from aiwf.services.audio import AudioUnavailable

        def install():
            service = _audio_service(ctx)
            installer = getattr(service, "install_mmaudio_variant", None)
            if not callable(installer):
                raise AudioUnavailable("Selected MMAudio variant installation is unavailable in this runtime.")
            installer(variant)
            return _audio_status_payload(ctx, deep=False)

        try:
            return _run_exclusive_pro_gpu_operation(ctx, install)
        except HTTPException:
            raise
        except AudioUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @router.post("/audio/setup/musicgen/{variant}")
    def audio_setup_musicgen_variant(variant: str):
        from aiwf.services.audio import AudioUnavailable

        def install():
            service = _audio_service(ctx)
            installer = getattr(service, "install_musicgen_variant", None)
            if not callable(installer):
                raise AudioUnavailable("Selected MusicGen variant installation is unavailable in this runtime.")
            installer(variant)
            return _audio_status_payload(ctx, deep=False)

        try:
            return _run_exclusive_pro_gpu_operation(ctx, install)
        except HTTPException:
            raise
        except AudioUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    def _prepare_audio_route(payload: ProAudioPreparePayload, *, video_lab: bool = False):
        from aiwf.services.audio import AudioUnavailable

        if _pro_video_job_running(ctx) or _image_generation_running(ctx) or _image_generation_pending(ctx) or _pro_workflow_runs_active(ctx):
            raise HTTPException(status_code=409, detail="Wait for active image, video, or workflow work to finish before preparing an audio model.")
        service = _audio_service(ctx)
        _raise_if_audio_license_blocked(ctx, payload.model_id)
        prepare_kind = payload.kind
        if video_lab:
            video_audio_choices = service.video_audio_model_choices()
            video_audio_model_ids = {str(model_id) for _, model_id in video_audio_choices}
            if payload.model_id in video_audio_model_ids:
                # Video Lab's video-audio models use the SFX preparation path,
                # including events:moss-sfx, regardless of the client kind.
                choices = video_audio_choices
                prepare_kind = "sfx"
            elif payload.kind == "music":
                # Keep prompt-only music choices such as MusicGen available in
                # Video Lab, while excluding text-only SFX models.
                choices = service.music_model_choices()
            else:
                choices = []
        else:
            choices = service.music_model_choices() if payload.kind == "music" else service.sfx_model_choices()
        if payload.model_id not in {str(model_id) for _, model_id in choices}:
            raise HTTPException(status_code=422, detail=f"Choose a supported {payload.kind} model.")
        route_id = (
            f"audio.video.audio.{payload.model_id}"
            if video_lab and payload.model_id.startswith(("mmaudio:", "events:"))
            else f"audio.music.{payload.model_id}"
            if video_lab
            else f"audio.{payload.kind}.{payload.model_id}"
        )
        support_paths = getattr(service, "model_support_paths", None)
        support_ids = support_paths(payload.model_id) if callable(support_paths) else [payload.model_id]
        setup_ready = False
        if payload.model_id.startswith(("acestep:", "moss-sfx:", "events:")):
            setup_ready = bool(service.commercial_engine_ready(payload.model_id))
        elif payload.model_id.startswith("facebook/musicgen-"):
            variant = payload.model_id.removeprefix("facebook/musicgen-")
            is_ready = getattr(service, "_musicgen_variant_ready", None)
            setup = service.setup_status(deep=False)
            setup_ready = bool(callable(is_ready) and is_ready(variant)) and bool(
                setup.get("musicDependenciesReady", setup.get("musicReady", False))
            )
        elif payload.model_id.startswith("mmaudio:"):
            variant = payload.model_id.split(":", 1)[1]
            is_ready = getattr(service, "_mmaudio_variant_ready", None)
            runtime_check = getattr(service, "_mmaudio_runtime_import_error", None)
            setup_ready = bool(callable(is_ready) and is_ready(variant)) and bool(
                callable(runtime_check) and not runtime_check()
            )
        if setup_ready:
            _release_cached_sana_video(ctx)
            _release_cached_wan_model(ctx)
            _unload_cached_ltx_model(ctx)
            # A model-family switch may park the previous audio model before
            # the new variant finishes loading. Invalidate old residency
            # receipts before attempting that switch so a failed preparation
            # cannot leave the UI claiming the parked model is still loaded.
            for prior in lifecycle_snapshot(ctx):
                prior_route = str(prior.get("route") or "")
                if prior_route.startswith("audio.") and prior_route != route_id and prior.get("resident") is True:
                    confirm_route_residency(
                        ctx,
                        prior_route,
                        str(prior.get("modelId") or ""),
                        resident=False,
                    )
        token = begin_route_operation(
            ctx, route=route_id, model_id=payload.model_id, setup_ready=setup_ready,
            support_ids=support_ids,
            detail="Preparing the selected audio model." if setup_ready else "The selected audio model is not installed and ready.",
        )
        if not setup_ready:
            raise HTTPException(status_code=409, detail="The selected audio model is not installed and ready. Install it explicitly before preparation.")
        mark_route_running(ctx, route_id, token, "Preparing the selected audio model.")
        try:
            result = service.prepare(kind=prepare_kind, model_id=payload.model_id)
        except AudioUnavailable as exc:
            finish_route_operation(ctx, route_id, token, success=False, detail=str(exc))
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            finish_route_operation(ctx, route_id, token, success=False, detail=str(exc))
            raise
        finish_route_preparation(
            ctx, route_id, token,
            detail=str(result.get("detail") or "Selected audio route setup is ready."),
            resident=result.get("resident"),
        )
        result["startupDefaultSaved"] = _persist_last_audio_model_id(
            ctx, payload.model_id, kind=prepare_kind, video_lab=video_lab
        )
        return {**result, "routeStatus": next((item.get("status") for item in lifecycle_snapshot(ctx) if item.get("route") == route_id), "setup-ready")}

    @router.post("/audio/prepare")
    def audio_prepare(payload: ProAudioPreparePayload):
        from aiwf.services.model_startup import pro_model_load_lock

        lock = pro_model_load_lock(ctx)
        if not lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="Another model is being prepared or switched. Try again when it finishes.")
        preparation_id = uuid4().hex
        setattr(ctx, "_pro_route_prepare_state", {"status": "preparing", "operationId": preparation_id})
        try:
            return _prepare_audio_route(payload)
        finally:
            route_state = getattr(ctx, "_pro_route_prepare_state", None)
            if isinstance(route_state, dict) and route_state.get("operationId") == preparation_id:
                setattr(ctx, "_pro_route_prepare_state", None)
            lock.release()

    @router.post("/video-lab/prepare-audio")
    def video_lab_prepare_audio(payload: ProAudioPreparePayload):
        from aiwf.services.model_startup import pro_model_load_lock

        lock = pro_model_load_lock(ctx)
        if not lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="Another model is being prepared or switched. Try again when it finishes.")
        preparation_id = uuid4().hex
        setattr(ctx, "_pro_route_prepare_state", {"status": "preparing", "operationId": preparation_id})
        try:
            return _prepare_audio_route(payload, video_lab=True)
        finally:
            route_state = getattr(ctx, "_pro_route_prepare_state", None)
            if isinstance(route_state, dict) and route_state.get("operationId") == preparation_id:
                setattr(ctx, "_pro_route_prepare_state", None)
            lock.release()

    @router.post("/audio/generate")
    def audio_generate(payload: ProAudioGeneratePayload):
        from aiwf.services.model_startup import pro_model_load_lock

        lock = pro_model_load_lock(ctx)
        if not lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="A model operation is already in progress. Retry after it finishes.")
        try:
            _assert_pro_model_load_idle(ctx)
            if _pro_video_job_running(ctx) or _image_generation_running(ctx) or _image_generation_pending(ctx) or _pro_workflow_runs_active(ctx):
                raise HTTPException(status_code=409, detail="A generation job is already running.")
            _release_cached_sana_video(ctx)
            _release_cached_wan_model(ctx)
            _unload_cached_ltx_model(ctx)
            return _generate_audio_response(ctx, payload)
        finally:
            lock.release()

    def audio_project_service() -> AudioProjectService:
        return AudioProjectService.from_audio_service(_audio_service(ctx))

    def audio_project_payload(service: AudioProjectService, manifest: Any) -> dict[str, Any]:
        data = manifest.model_dump(mode="json", exclude_none=True)
        audio_url = ""
        if manifest.track is not None:
            asset = service._asset_path(service._project_dir(manifest.project_id), manifest.track.asset_ref)
            audio_url = _output_asset_url(ctx, asset)
            if not audio_url.startswith("/api/pro/outputs/"):
                raise HTTPException(status_code=404, detail="Audio project asset not found.")
        data["audio_url"] = audio_url
        return data

    def audio_project_failure(exc: Exception) -> None:
        if isinstance(exc, AudioProjectNotFound):
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if isinstance(exc, AudioProjectAssetMissing):
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if isinstance(exc, (AudioProjectCorrupt, AudioProjectRecoveryError, AudioProjectError, ValueError)):
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        raise exc

    @router.get("/audio/projects")
    def audio_projects_list():
        return {"projects": audio_project_service().list_projects()}

    @router.post("/audio/projects")
    def audio_projects_save(payload: ProAudioProjectSavePayload):
        service = audio_project_service()
        try:
            save_options = {
                "name": payload.name,
                "project_id": payload.project_id,
                "audio_path": payload.audio_path,
                "options": payload.options,
                "sample_rate": payload.sample_rate,
                "license_notice": payload.license_notice,
                "consent_status": payload.consent_status,
            }
            if "license" in payload.model_fields_set:
                save_options["license"] = payload.license
            elif payload.license_notice is not None:
                raise ValueError("A legacy save cannot set a license notice without its structured license record.")
            else:
                save_options["license"] = None
            manifest = service.save_project(**save_options)
            return audio_project_payload(service, manifest)
        except (AudioProjectError, ValueError) as exc:
            audio_project_failure(exc)

    @router.get("/audio/projects/{project_id}")
    def audio_projects_load(project_id: str):
        service = audio_project_service()
        try:
            return audio_project_payload(service, service.load_project(project_id))
        except (AudioProjectError, ValueError) as exc:
            audio_project_failure(exc)

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
        safe_stem = "".join(
            ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in Path(file.filename or "upload").stem
        )[:64] or "upload"
        upload_root = _video_lab_upload_root(ctx)
        dest: Path | None = None
        upload_handle = None
        for _ in range(8):
            candidate = _video_lab_upload_destination(upload_root, safe_stem, suffix)
            try:
                upload_handle = candidate.open("xb")
            except FileExistsError:
                continue
            except OSError as exc:
                raise HTTPException(status_code=500, detail=f"Could not store the upload: {exc}") from exc
            dest = candidate
            break
        if dest is None or upload_handle is None:
            raise HTTPException(status_code=500, detail="Could not allocate a unique upload path.")
        written = 0
        try:
            with upload_handle as handle:
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
        src = _video_lab_resolve_source(ctx, payload.video_path)
        op = (payload.op or "").strip().lower()
        if op == "vsr":
            operation = lambda: _run_video_lab_gpu_with_audio_release(
                ctx, lambda: _video_lab_run_vsr(ctx, src, payload)
            )
        elif op == "rife":
            operation = lambda: _run_video_lab_gpu_with_audio_release(
                ctx, lambda: _video_lab_run_rife(ctx, src, payload)
            )
        elif op == "audio":
            operation = lambda: _video_lab_run_audio(ctx, src, payload)
        elif op == "extend":
            operation = lambda: _video_lab_run_extend(ctx, src, payload)
        else:
            raise HTTPException(status_code=422, detail="op must be one of: vsr, rife, audio, extend.")
        result = _run_exclusive_pro_gpu_operation(ctx, operation)
        return result

    @router.post("/segment/auto-mask")
    def segment_auto_mask(payload: ProAutoMaskPayload):
        _assert_pro_model_load_idle(ctx)
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
            mask, preview, _candidates, status = _run_exclusive_pro_gpu_operation(
                ctx,
                lambda: segment.segment(
                    image.convert("RGB"),
                    SegmentRequest(
                        text_prompt=prompt,
                        box_threshold=float(payload.box_threshold),
                        dilation=int(payload.dilation),
                        mask_blur=int(payload.mask_blur),
                        feather=int(payload.feather),
                    ),
                ),
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Auto mask failed: {exc}") from exc
        return {
            "status": status,
            "mask": _image_to_data_url(mask.convert("RGB")),
            "preview": _image_to_data_url(preview),
        }

    @router.post("/faceswap")
    def faceswap(payload: ProFaceSwapPayload):
        _assert_pro_model_load_idle(ctx)
        faceswap_service = getattr(ctx, "faceswap", None)
        if faceswap_service is None:
            raise HTTPException(status_code=503, detail="Face swap service is not available in this runtime.")
        target = _decode_pro_image_data_url(payload.target_image_data_url, "Target image")
        source = _decode_pro_image_data_url(payload.source_image_data_url, "Source face image")
        if target is None or source is None:
            raise HTTPException(status_code=422, detail="Provide both a target image and a source face image.")
        try:
            result = _run_exclusive_pro_gpu_operation(
                ctx,
                lambda: faceswap_service.swap(target.convert("RGB"), source.convert("RGB")),
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Face swap failed: {exc}") from exc
        return {
            "status": "completed",
            "image": _image_to_data_url(result),
            "width": result.width,
            "height": result.height,
            "message": "Face swap complete.",
        }

    @router.get("/enhance/models", response_model=ProEnhanceModelsResponse)
    def enhance_models() -> ProEnhanceModelsResponse:
        enhance_service = getattr(ctx, "enhance", None)
        if enhance_service is None or not callable(getattr(enhance_service, "list_model_status", None)):
            raise HTTPException(status_code=503, detail="Enhance model inventory is not available in this runtime.")
        try:
            invalidate = getattr(enhance_service, "invalidate_model_catalog", None)
            if callable(invalidate):
                invalidate()
                setattr(ctx, "_pro_capability_cache", None)
            return ProEnhanceModelsResponse(models=enhance_service.list_model_status())
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Could not inspect Enhance models: {exc}") from exc

    @router.post("/enhance/models/{model_id}/install", response_model=ProEnhanceModelInstallResponse)
    def install_enhance_model(model_id: str) -> ProEnhanceModelInstallResponse:
        enhance_service = getattr(ctx, "enhance", None)
        if enhance_service is None or not callable(getattr(enhance_service, "prepare_model", None)):
            raise HTTPException(status_code=503, detail="Enhance model setup is not available in this runtime.")
        try:
            path = enhance_service.prepare_model(model_id)
            model = next(item for item in enhance_service.list_model_status() if item["id"] == model_id)
            setattr(ctx, "_pro_capability_cache", None)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except StopIteration as exc:
            raise HTTPException(status_code=404, detail=f"Unknown Enhance model: {model_id}") from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Enhance model setup failed: {exc}") from exc
        return ProEnhanceModelInstallResponse(
            model=model,
            path=str(path),
            message=f"{model['title']} is installed and ready for preflight.",
        )

    @router.post("/enhance/image")
    def enhance_image(payload: ProEnhanceImagePayload):
        _assert_pro_model_load_idle(ctx)
        from aiwf.core.domain.enhance import RestoreOptions, UpscaleOptions

        enhance_service = getattr(ctx, "enhance", None)
        if enhance_service is None:
            raise HTTPException(status_code=503, detail="Enhance service is not available in this runtime.")
        required_model_ids = []
        if payload.restore_enabled:
            required_model_ids.append(("restorer", payload.restore_model.strip()))
        if payload.upscale_enabled:
            required_model_ids.append(("upscaler", payload.upscale_model.strip()))
        if required_model_ids:
            inventory_method = getattr(enhance_service, "list_model_status", None)
            if not callable(inventory_method):
                raise HTTPException(status_code=503, detail="Enhance model readiness cannot be verified in this runtime.")
            try:
                model_inventory = {item["id"]: item for item in inventory_method()}
            except Exception as exc:
                raise HTTPException(status_code=503, detail=f"Could not verify Enhance model readiness: {exc}") from exc
            for expected_kind, model_id in required_model_ids:
                model = model_inventory.get(model_id)
                if model is None or model.get("kind") != expected_kind:
                    raise HTTPException(status_code=422, detail=f"Select a catalogued {expected_kind} model.")
                if not model.get("installed"):
                    raise HTTPException(
                        status_code=409,
                        detail=f"{model.get('title') or model_id} needs setup first. Install it from Enhance, then run again.",
                    )
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
            result, infotext = _run_exclusive_pro_gpu_operation(
                ctx,
                lambda: enhance_service.run_pipeline(
                    image.convert("RGB"),
                    restore=restore_opts,
                    upscale=upscale_opts,
                    restore_first=bool(payload.restore_first),
                ),
            )
        except HTTPException:
            raise
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
        _assert_pro_model_load_idle(ctx)
        from aiwf.core.domain.vsr import VsrOptions
        from aiwf.services.vsr import VsrUnavailable

        image = _decode_pro_image_data_url(payload.image_data_url, "Source image")
        if image is None:
            raise HTTPException(status_code=422, detail="Provide a source image as a base64 data URL.")
        try:
            result = _run_exclusive_pro_gpu_operation(
                ctx,
                lambda: _vsr_service(ctx).upscale_image(
                    image.convert("RGB"),
                    VsrOptions(
                        scale=float(payload.scale),
                        mode=int(payload.mode),
                        strength=float(payload.strength),
                        effect=str(payload.effect or "SuperRes"),
                    ),
                ),
            )
        except HTTPException:
            raise
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
