from __future__ import annotations

"""Shared Pro/Gradio wiring contract for generation settings and workflow blocks.

This module deliberately does not load model weights. It is a QA/wiring layer that
normalizes UI settings into a stable JSON packet, then wraps that packet into the
linear workflow code-block format used by Pipeline Atlas.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4
import json
import re

from aiwf.services.model_family_support import detect_precision_from_name, normalize_precision_label

PACKET_SCHEMA = "aiwf.studio-generation-packet.v1"
WORKFLOW_BLOCK_SCHEMA = "aiwf.workflow-code-block.payload.v1"
WORKFLOW_DOCUMENT_SCHEMA = "aiwf.workflow-code-blocks.v1"

BLOCKING_MODEL_STATUSES = {"broken-runtime", "blocked-cleanly", "unsupported-no-route"}
WARNING_MODEL_STATUSES = {"metadata-only", "needs-smoke", "experimental", "candidate"}
SELECTABLE_MODEL_STATUSES = {
    "working",
    "ready",
    "loaded",
    "supported",
    "supported-smoked",
    "supported-gated",
    "partial-supported",
    "supported-experimental-quants",
}

WAN_RUNTIME_DEFAULT = "fast_5b"
WAN_RUNTIME_MODES = {"fast_5b", "native_high_low", "native_high_low_fp8_experimental"}
WAN_OFFLOAD_MODES = {"sequential", "group", "streamed", "model", "balanced", "resident", "none"}
WAN_SIGMA_TYPES = {"simple", "beta", "exponential", "karras"}
WAN_SAMPLERS = {"unipc", "euler", "heun"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _string(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _clean_string(value: Any, default: str = "") -> str:
    text = _string(value, default).strip()
    return text if text else default


def _number(value: Any, default: float | int = 0) -> float | int:
    try:
        if isinstance(default, int) and not isinstance(default, bool):
            return int(round(float(value)))
        return float(value)
    except (TypeError, ValueError):
        return default


def _boolean(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disabled"}:
        return False
    return default


def _record(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:8]}"


def _slug(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-").lower()
    return safe or _safe_id("item")


def _first(settings: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in settings and settings[key] not in {None, ""}:
            return settings[key]
    return default


def infer_model_family(*, model_id: str = "", model_name: str = "", architecture: str = "", engine_id: str = "", backend: str = "") -> str:
    text = " ".join([model_id, model_name, architecture, engine_id, backend]).lower()
    if "wan" in text:
        return "wan"
    if "sana-video" in text or "sana_video" in text:
        return "sana_video"
    if "ltx" in text:
        return "ltx"
    if "flux2" in text or "flux.2" in text or "klein" in text:
        return "flux2_klein"
    if "flux" in text:
        return "flux"
    if "qwen" in text or "nunchaku" in text:
        return "qwen_image"
    if "z-image" in text or "zimage" in text or "z_image" in text:
        return "z_image"
    if "sana" in text:
        return "sana"
    if "sdxl" in text or "stable diffusion xl" in text:
        return "sdxl"
    if "sd3" in text or "sd 3" in text or "stable diffusion 3" in text:
        return "sd35"
    if "sd15" in text or "sd1.5" in text or "sd 1.5" in text or "stable diffusion 1.5" in text:
        return "sd15"
    if "onnx" in text:
        return "onnx"
    if "gguf" in text or "llm" in text or "vl" in text:
        return "llm_vl"
    return "unknown"


def route_for_packet(mode: str, family: str) -> str:
    normalized_mode = (mode or "image").strip().lower()
    normalized_family = (family or "unknown").strip().lower()
    if normalized_mode == "inpaint":
        return "flux-fill-inpaint" if normalized_family == "flux" else "inpaint"
    if normalized_mode in {"video", "i2v", "t2v"}:
        if normalized_family == "wan":
            return "wan-video"
        if normalized_family == "sana_video":
            return "sana-video"
        if normalized_family == "ltx":
            return "ltx-video"
        return "image-to-video"
    return {
        "flux2_klein": "flux2-klein-image",
        "qwen_image": "qwen-image",
        "z_image": "z-image",
        "sana": "sana-image",
        "flux": "flux-image",
    }.get(normalized_family, "image-generate")


def detect_precision_label(*values: Any) -> str:
    for value in values:
        text = _clean_string(value)
        if not text:
            continue
        direct = normalize_precision_label(text)
        if direct in {
            "FP32",
            "FP16",
            "BF16",
            "FP8",
            "INT8",
            "NF4",
            "FP4",
            "NVFP4",
            "INT4",
            "Q2_K",
            "Q3_K_S",
            "Q3_K_M",
            "Q3_K_L",
            "Q4_0",
            "Q4_1",
            "Q4_K_S",
            "Q4_K_M",
            "Q5_0",
            "Q5_K_S",
            "Q5_K_M",
            "Q6_K",
            "Q8_0",
            "IQ2",
            "IQ3",
            "IQ4",
        }:
            return direct
        detected = detect_precision_from_name(text)
        if detected:
            return detected
    return "auto"


def model_selection_gate(status: str | None, *, reason: str = "", suggested_action: str = "") -> dict[str, Any]:
    normalized = _clean_string(status, "metadata-only").lower()
    if normalized in BLOCKING_MODEL_STATUSES:
        return {
            "normalSelectable": False,
            "level": "block",
            "status": normalized,
            "reason": reason or "This asset is blocked from normal generation selection.",
            "suggestedAction": suggested_action or "Keep it visible for QA, but do not run it until the family route is implemented or fixed.",
        }
    if normalized in WARNING_MODEL_STATUSES:
        return {
            "normalSelectable": True,
            "requiresWarning": True,
            "level": "warn",
            "status": normalized,
            "reason": reason or "This asset is discovered but lacks a bounded smoke receipt.",
            "suggestedAction": suggested_action or "Use advanced/testing mode and capture a receipt before treating it as supported.",
        }
    return {
        "normalSelectable": True,
        "requiresWarning": normalized not in SELECTABLE_MODEL_STATUSES and normalized not in {"", "unknown"},
        "level": "pass" if normalized in SELECTABLE_MODEL_STATUSES else "info",
        "status": normalized,
        "reason": reason,
        "suggestedAction": suggested_action,
    }


def _wan_settings(settings: dict[str, Any]) -> dict[str, Any]:
    runtime_mode = _clean_string(_first(settings, "wanRuntimeMode", "wan_runtime_mode", "runtimeMode", "runtime_mode"), WAN_RUNTIME_DEFAULT).lower().replace("-", "_")
    runtime_mode = {
        "high_low": "native_high_low",
        "high_low_fp8": "native_high_low_fp8_experimental",
        "fp8_high_low": "native_high_low_fp8_experimental",
    }.get(runtime_mode, runtime_mode)
    if runtime_mode not in WAN_RUNTIME_MODES:
        runtime_mode = WAN_RUNTIME_DEFAULT
    offload = _clean_string(_first(settings, "wanOffload", "wan_offload", "offload"), "balanced")
    if offload not in WAN_OFFLOAD_MODES:
        offload = "balanced"
    sigma = _clean_string(_first(settings, "wanSigmaType", "wan_sigma_type", "sigmaType", "sigma_type"), "simple")
    if sigma not in WAN_SIGMA_TYPES:
        sigma = "simple"
    sampler = _clean_string(_first(settings, "wanSampler", "wan_sampler"), "unipc").lower()
    if sampler not in WAN_SAMPLERS:
        sampler = "unipc"
    return {
        "runtimeMode": runtime_mode,
        "highNoiseModelId": _clean_string(_first(settings, "highNoiseModelId", "high_noise_model_id")),
        "lowNoiseModelId": _clean_string(_first(settings, "lowNoiseModelId", "low_noise_model_id")),
        "vaeId": _clean_string(_first(settings, "vaeId", "vae_id")),
        "textEncoderPath": _clean_string(_first(settings, "textEncoderPath", "text_encoder_path")),
        "highNoiseLoraId": _clean_string(_first(settings, "highNoiseLoraId", "high_noise_lora_id")),
        "highNoiseLoraScale": _number(_first(settings, "highNoiseLoraScale", "high_noise_lora_scale"), 1.0),
        "lowNoiseLoraId": _clean_string(_first(settings, "lowNoiseLoraId", "low_noise_lora_id")),
        "lowNoiseLoraScale": _number(_first(settings, "lowNoiseLoraScale", "low_noise_lora_scale"), 1.0),
        "offload": offload,
        "boundaryRatio": _number(_first(settings, "boundaryRatio", "boundary_ratio"), 0.875),
        "highNoiseSteps": _number(_first(settings, "highNoiseSteps", "high_noise_steps"), 20),
        "lowNoiseSteps": _number(_first(settings, "lowNoiseSteps", "low_noise_steps"), 1),
        "sigmaType": sigma,
        "sampler": sampler,
        "flowShift": _number(_first(settings, "flowShift", "flow_shift"), 5.0),
        "offloadTextEncoderAfterEncode": _boolean(_first(settings, "offloadTextEncoderAfterEncode", "offload_text_encoder_after_encode"), True),
        "useSageAttention": _boolean(_first(settings, "useSageAttention", "use_sage_attention"), True),
    }


def build_studio_generation_packet(
    settings: dict[str, Any],
    *,
    model: dict[str, Any] | None = None,
    runtime: dict[str, Any] | None = None,
    source: str = "Studio",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize Pro or Gradio generation controls into one QA-safe packet."""

    model_record = _record(model)
    runtime_record = _record(runtime)
    mode = _clean_string(_first(settings, "mode"), "image").lower()
    model_id = _clean_string(_first(settings, "modelId", "model_id", "checkpointId", "checkpoint_id"))
    model_name = _clean_string(_first(settings, "modelName", "model_name", "checkpointTitle", "checkpoint_title"), model_id)
    architecture = _clean_string(model_record.get("architecture") or settings.get("architecture"))
    engine_id = _clean_string(model_record.get("engineId") or model_record.get("engine_id") or settings.get("engineId") or settings.get("engine_id"))
    backend = _clean_string(model_record.get("backend") or settings.get("backend"))
    family = infer_model_family(model_id=model_id, model_name=model_name, architecture=architecture, engine_id=engine_id, backend=backend)
    route = _clean_string(_first(settings, "route"), route_for_packet(mode, family))
    precision = detect_precision_label(
        _first(settings, "precision", "quantization"),
        model_record.get("precision"),
        model_record.get("quantization"),
        model_record.get("assetSummary"),
        model_name,
        model_id,
    )
    status = _clean_string(model_record.get("status") or settings.get("modelStatus") or settings.get("status"), "metadata-only")
    selection_gate = model_selection_gate(
        status,
        reason=_clean_string(model_record.get("reason") or settings.get("reason")),
        suggested_action=_clean_string(model_record.get("suggestedAction") or model_record.get("suggested_action") or settings.get("suggestedAction")),
    )
    packet: dict[str, Any] = {
        "schema": PACKET_SCHEMA,
        "capturedAt": _now(),
        "source": source,
        "mode": mode,
        "route": route,
        "family": family,
        "precision": precision,
        "prompt": {
            "positive": _string(_first(settings, "prompt")),
            "negative": _string(_first(settings, "negativePrompt", "negative_prompt")),
            "seed": _number(_first(settings, "seed"), -1),
        },
        "model": {
            "id": model_id,
            "name": model_name,
            "engineId": engine_id,
            "engineLabel": _clean_string(model_record.get("engineLabel") or model_record.get("engine_label") or settings.get("engineLabel")),
            "architecture": architecture,
            "backend": backend,
            "status": status,
            "reason": _clean_string(model_record.get("reason") or settings.get("reason")),
            "suggestedAction": _clean_string(model_record.get("suggestedAction") or model_record.get("suggested_action") or settings.get("suggestedAction")),
            "assetSummary": _clean_string(model_record.get("assetSummary") or model_record.get("asset_summary") or settings.get("assetSummary")),
            "estVramGb": _number(model_record.get("estVramGb") or model_record.get("est_vram_gb"), 0.0),
            "heavyFor12Gb": _boolean(model_record.get("heavyFor12Gb") or model_record.get("heavy_for_12gb"), False),
        },
        "selectionGate": selection_gate,
        "generation": {
            "width": _number(_first(settings, "width"), 512),
            "height": _number(_first(settings, "height"), 512),
            "aspectRatioId": _clean_string(_first(settings, "aspectRatioId", "aspect_ratio_id")),
            "steps": _number(_first(settings, "steps"), 20),
            "cfgScale": _number(_first(settings, "cfgScale", "cfg_scale"), 7.0),
            "sampler": _clean_string(_first(settings, "sampler"), "automatic"),
            "scheduler": _clean_string(_first(settings, "scheduler"), "automatic"),
            "clipSkip": _number(_first(settings, "clipSkip", "clip_skip"), 1),
            "batchSize": _number(_first(settings, "batchSize", "batch_size"), 1),
            "batchCount": _number(_first(settings, "batchCount", "batch_count"), 1),
            "saveImages": _boolean(_first(settings, "saveImages", "save_images"), True),
        },
        "imageTools": {
            "enableHires": _boolean(_first(settings, "enableHires", "enable_hires", "enableHr", "enable_hr"), False),
            "hiresScale": _number(_first(settings, "hiresScale", "hires_scale", "hrScale", "hr_scale"), 2.0),
            "hiresSteps": _number(_first(settings, "hiresSteps", "hires_steps", "hrSteps", "hr_steps"), 20),
            "hiresDenoise": _number(_first(settings, "hiresDenoise", "hires_denoise", "hrDenoisingStrength", "hr_denoising_strength"), 0.35),
            "hiresUpscaler": _clean_string(_first(settings, "hiresUpscaler", "hires_upscaler", "hrUpscaler", "hr_upscaler"), "lanczos"),
            "sourceImageName": _clean_string(_first(settings, "sourceImageName", "source_image_name")),
            "hasSourceImage": bool(_first(settings, "sourceImageDataUrl", "source_image_data_url", "initImageDataUrl", "init_image_data_url", "sourcePath", "source_path")),
        },
        "inpaint": {
            "hasMask": bool(_first(settings, "maskImageDataUrl", "mask_image_data_url", "maskPath", "mask_path")),
            "denoisingStrength": _number(_first(settings, "denoisingStrength", "denoising_strength"), 0.75),
            "maskBlur": _number(_first(settings, "maskBlur", "mask_blur"), 4),
            "inpaintOnlyMasked": _boolean(_first(settings, "inpaintOnlyMasked", "inpaint_only_masked"), False),
            "inpaintMaskedPadding": _number(_first(settings, "inpaintMaskedPadding", "inpaint_masked_padding"), 32),
            "inpaintMaskContent": _clean_string(_first(settings, "inpaintMaskContent", "inpaint_mask_content"), "original"),
        },
        "video": {
            "frames": _number(_first(settings, "frames", "num_frames"), 81),
            "fps": _number(_first(settings, "fps"), 16),
            "generateAudio": _boolean(_first(settings, "generateAudio", "generate_audio"), False),
            "sanaQuantization": _clean_string(_first(settings, "sanaQuantization", "sana_quantization"), "auto"),
            "sanaVaeTiling": _clean_string(_first(settings, "sanaVaeTiling", "sana_vae_tiling", "vaeTiling", "vae_tiling"), "auto"),
            "wan": _wan_settings(settings),
        },
        "runtime": {
            "state": _clean_string(runtime_record.get("state")),
            "backend": _clean_string(runtime_record.get("backend")),
            "device": _clean_string(runtime_record.get("device")),
            "precision": _clean_string(runtime_record.get("precision")),
            "attention": _clean_string(runtime_record.get("attention")),
            "queueCount": _number(runtime_record.get("queueCount") or runtime_record.get("queue_count"), 0),
        },
        "qa": {
            "validationOnly": True,
            "adapterTestingRequired": family in {"wan", "flux2_klein", "z_image", "qwen_image", "ltx"},
            "notes": [
                "This packet captures wiring; it does not certify runtime model or adapter behavior.",
            ],
        },
    }

    if family == "wan" or route == "wan-video":
        wan = packet["video"]["wan"]
        packet["sidecars"] = {
            "wanModelPack": {
                "runtimeMode": wan["runtimeMode"],
                "highNoiseModelId": wan["highNoiseModelId"],
                "lowNoiseModelId": wan["lowNoiseModelId"],
                "vaeId": wan["vaeId"],
                "textEncoderPath": wan["textEncoderPath"],
                "boundaryRatio": wan["boundaryRatio"],
                "highNoiseSteps": wan["highNoiseSteps"],
                "lowNoiseSteps": wan["lowNoiseSteps"],
            },
            "loraStack": {
                "entries": [
                    {"target": "high", "id": wan["highNoiseLoraId"], "scale": wan["highNoiseLoraScale"]},
                    {"target": "low", "id": wan["lowNoiseLoraId"], "scale": wan["lowNoiseLoraScale"]},
                ],
                "testingStatus": "needs family/precision smoke before support is declared",
            },
            "offloadPlan": {
                "mode": wan["offload"],
                "offloadTextEncoderAfterEncode": wan["offloadTextEncoderAfterEncode"],
                "useSageAttention": wan["useSageAttention"],
            },
        }
    if extra:
        packet["extra"] = extra
    return packet


def build_workflow_code_block(packet: dict[str, Any], *, order: int = 1, source: str | None = None, block_id: str | None = None) -> dict[str, Any]:
    family = _clean_string(packet.get("family"), "unknown")
    mode = _clean_string(packet.get("mode"), "image")
    route = _clean_string(packet.get("route"), "generation-request")
    precision = _clean_string(packet.get("precision"), "auto")
    label_mode = {"video": "Video", "inpaint": "Inpaint"}.get(mode, "Image")
    label = f"{label_mode} · {family}"
    width = _record(packet.get("generation")).get("width", "?")
    height = _record(packet.get("generation")).get("height", "?")
    steps = _record(packet.get("generation")).get("steps", "?")
    code_payload = {
        "schema": WORKFLOW_BLOCK_SCHEMA,
        "packet": packet,
    }
    return {
        "id": block_id or _safe_id("workflow-block"),
        "label": label,
        "kind": "generation",
        "nodeId": "generation-request",
        "source": source or _clean_string(packet.get("source"), "Studio"),
        "createdAt": _now(),
        "summary": f"{route} · {width}×{height} · {steps} steps · {precision}",
        "order": int(order),
        "classes": {"requires": [], "produces": ["artifact"]},
        "payload": code_payload,
        "code": json.dumps(code_payload, indent=2, ensure_ascii=False),
    }


def workflow_document_from_blocks(blocks: list[dict[str, Any]], *, label: str = "Workflow code block queue") -> dict[str, Any]:
    ordered = sorted(blocks, key=lambda item: int(_record(item).get("order") or 0))
    for index, block in enumerate(ordered, start=1):
        block["order"] = index
    return {
        "schema": WORKFLOW_DOCUMENT_SCHEMA,
        "id": "main",
        "label": label,
        "savedAt": _now(),
        "blocks": ordered,
        "routing": {
            "mode": "linear-code-blocks",
            "validationOnly": True,
            "note": "Blocks are movable settings snapshots. No graph wires are required.",
        },
    }


def validate_workflow_code_block_document(document: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    blocks = document.get("blocks")
    if not isinstance(blocks, list):
        return {"valid": False, "errors": ["workflow.blocks must be a list"], "blocks": []}
    seen: set[str] = set()
    results: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        if not isinstance(block, dict):
            errors.append(f"Block {index + 1} must be an object.")
            continue
        block_id = _clean_string(block.get("id"), f"block-{index + 1}")
        if block_id in seen:
            errors.append(f"Block {index + 1} duplicates id '{block_id}'.")
        seen.add(block_id)
        payload = block.get("payload")
        if not isinstance(payload, dict):
            code = block.get("code")
            if isinstance(code, str) and code.strip():
                try:
                    payload = json.loads(code)
                except json.JSONDecodeError:
                    errors.append(f"Block {index + 1} code is not valid JSON.")
                    payload = {}
            else:
                errors.append(f"Block {index + 1} needs an object payload or JSON code.")
                payload = {}
        packet = payload.get("packet") if isinstance(payload, dict) else None
        if isinstance(packet, dict):
            gate = packet.get("selectionGate")
            if isinstance(gate, dict) and gate.get("normalSelectable") is False:
                errors.append(f"Block {index + 1} model is blocked: {gate.get('reason') or gate.get('status')}")
        results.append({"id": block_id, "valid": True, "order": block.get("order") or index + 1})
    if not blocks:
        errors.append("Workflow has no code blocks.")
    return {"valid": not errors, "errors": errors, "blocks": results, "mode": "linear-code-blocks"}


def workflow_document_json_from_packet(packet: dict[str, Any], *, source: str | None = None) -> str:
    block = build_workflow_code_block(packet, source=source)
    return json.dumps(workflow_document_from_blocks([block]), indent=2, ensure_ascii=False)


def gradio_studio_workflow_json(**settings: Any) -> str:
    model = {
        "id": _clean_string(settings.get("modelId") or settings.get("checkpointId")),
        "name": _clean_string(settings.get("modelName") or settings.get("checkpointTitle") or settings.get("checkpointId")),
        "status": _clean_string(settings.get("modelStatus"), "metadata-only"),
    }
    extra = settings.get("extra") if isinstance(settings.get("extra"), dict) else None
    packet = build_studio_generation_packet(settings, model=model, source="Gradio Studio", extra=extra)
    return workflow_document_json_from_packet(packet, source="Gradio Studio")


def gradio_wan_workflow_json(**settings: Any) -> str:
    model_id = _clean_string(settings.get("modelId") or settings.get("checkpointId"))
    model = {
        "id": model_id,
        "name": _clean_string(settings.get("modelName"), model_id or "Wan video route"),
        "engineId": "wan",
        "architecture": "wan",
        "status": _clean_string(settings.get("modelStatus"), "metadata-only"),
    }
    normalized = {**settings, "mode": "video", "engineId": "wan", "route": "wan-video"}
    extra = settings.get("extra") if isinstance(settings.get("extra"), dict) else None
    packet = build_studio_generation_packet(normalized, model=model, source="Gradio Wan I2V", extra=extra)
    return workflow_document_json_from_packet(packet, source="Gradio Wan I2V")
