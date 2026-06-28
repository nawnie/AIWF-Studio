from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from aiwf.core.config.settings import RuntimeFlags, UserSettings

READINESS_STATUSES = (
    "working",
    "metadata-only",
    "blocked-cleanly",
    "broken-runtime",
    "unsupported-no-route",
)

MODEL_FILE_SUFFIXES = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx"}
DOWNLOAD_KEYWORDS = (
    "flux",
    "gemma",
    "ltx",
    "qwen",
    "sana",
    "vae",
    "wan",
    "z-image",
    "zimage",
)


@dataclass(frozen=True)
class PipelineReadinessRecord:
    id: str
    family: str
    asset_type: str
    path: str
    status: str
    route: str
    reason: str
    storage: str = ""
    quantization: str = ""
    required_vae: str = ""
    required_text_encoder: str = ""
    tokenizer: str = ""
    smoke_command: str = ""
    receipt_path: str = ""
    suggested_action: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def collect_pipeline_readiness(
    flags: RuntimeFlags,
    settings: UserSettings | None = None,
    *,
    include_downloads: bool = True,
    download_roots: Iterable[Path] | None = None,
    force_rescan: bool = False,
) -> list[PipelineReadinessRecord]:
    """Build a read-only readiness ledger for local model pipeline assets.

    This deliberately stays metadata/preflight only. It does not load model
    weights or run generation; the `smoke_command` and `receipt_path` fields
    tell the operator how to verify a route when a runtime smoke is needed.
    """

    user_settings = settings or UserSettings()
    records: list[PipelineReadinessRecord] = []
    records.extend(_preflight_records(flags, user_settings))
    records.extend(_registry_records(flags, user_settings))
    records.extend(_ltx_route_records(flags, user_settings))
    records.extend(_inventory_records(flags, force_rescan=force_rescan))
    if include_downloads:
        records.extend(_download_records(download_roots or (Path.home() / "Downloads",), flags))
    return sorted(records, key=lambda item: (item.family, item.asset_type, item.id.lower(), item.path.lower()))


def readiness_summary(records: Iterable[PipelineReadinessRecord]) -> dict[str, int]:
    summary = {status: 0 for status in READINESS_STATUSES}
    for record in records:
        summary[record.status] = summary.get(record.status, 0) + 1
    return summary


def classify_pipeline_asset(
    family: str,
    architecture: str,
    path: Path,
    *,
    source: str = "models",
    flags: RuntimeFlags | None = None,
    metadata: dict[str, str] | None = None,
) -> PipelineReadinessRecord:
    return _classify_model_record(family, architecture, path, source=source, metadata=metadata, flags=flags)


def _preflight_records(flags: RuntimeFlags, settings: UserSettings) -> list[PipelineReadinessRecord]:
    from aiwf.services.pipeline_preflight import (
        PipelinePreflightResult,
        preflight_diffusers_pipeline,
        preflight_image_runtime_pipelines,
        preflight_ltx_pipeline,
        preflight_onnx_pipeline,
        preflight_qwen_nunchaku_pipeline,
        preflight_sana_video_pipeline,
        preflight_wan_pipeline,
    )

    onnx_root = _onnx_root(flags, settings)
    preflights = [
        preflight_diffusers_pipeline(),
        preflight_image_runtime_pipelines(),
        preflight_qwen_nunchaku_pipeline(flags),
        preflight_sana_video_pipeline(flags, settings),
        preflight_wan_pipeline(flags, settings),
        preflight_ltx_pipeline(flags, settings),
        preflight_onnx_pipeline(onnx_root, provider_preference=flags.onnx_provider),
    ]
    return [_preflight_record(result, flags) for result in preflights]


def _preflight_record(result: "PipelinePreflightResult", flags: RuntimeFlags) -> PipelineReadinessRecord:
    route = _slug(result.pipeline)
    family = _pipeline_family(route)
    receipt = _latest_receipt(flags, family, route=route)
    status = "working" if result.ok and receipt else "metadata-only" if result.ok else "blocked-cleanly"
    blocking = next((item.message for item in result.items if not item.ok), "")
    metadata = {key: _scrub_value(value) for key, value in result.metadata.items()}
    metadata["warnings"] = " | ".join(result.warnings)
    return PipelineReadinessRecord(
        id=f"preflight:{route}",
        family=family,
        asset_type="pipeline",
        path=_first_item_path(result),
        status=status,
        route=route,
        reason=blocking or ("preflight passed; see receipt for last smoke output" if receipt else "preflight passed; no runtime smoke executed by this ledger"),
        smoke_command=_smoke_command_for_route(family, route),
        receipt_path=str(receipt) if receipt else "",
        metadata=metadata,
    )


def _registry_records(flags: RuntimeFlags, settings: UserSettings) -> list[PipelineReadinessRecord]:
    from aiwf.services.pipeline_registry import PipelineRegistry

    registry = PipelineRegistry(flags, settings)
    records = []
    for pipeline in [*registry.image_pipelines(), *registry.video_pipelines()]:
        family = _pipeline_family(pipeline.id)
        receipt = _latest_receipt(flags, family, route=pipeline.id)
        status = "working" if pipeline.ready and receipt else "metadata-only" if pipeline.ready else "blocked-cleanly"
        records.append(
            PipelineReadinessRecord(
                id=f"registry:{pipeline.id}",
                family=family,
                asset_type="pipeline",
                path="",
                status=status,
                route=pipeline.id,
                reason=pipeline.message,
                smoke_command=_smoke_command_for_route(family, pipeline.id),
                receipt_path=str(receipt) if receipt else "",
                metadata={
                    "label": pipeline.label,
                    "kind": pipeline.kind,
                    "engine": pipeline.engine,
                    "summary": pipeline.summary,
                },
            )
        )
    return records


def _ltx_route_records(flags: RuntimeFlags, settings: UserSettings) -> list[PipelineReadinessRecord]:
    from aiwf.core.domain.ltx import (
        LTX_GEMMA_BACKEND_GGUF,
        LTX_GEMMA_BACKEND_HF_SAFETENSORS,
        LTX_PIPELINE_ONE_STAGE,
        LtxVideoRequest,
    )
    from aiwf.services.ltx import LtxService, ltx_checkpoint_openability_error

    service = LtxService(flags, settings)
    records: list[PipelineReadinessRecord] = []
    records.append(_ltx2b_diffusers_route_record(flags))
    for backend, route, asset_type in (
        (LTX_GEMMA_BACKEND_HF_SAFETENSORS, "ltx-one-stage-hf-gemma", "pipeline"),
        (LTX_GEMMA_BACKEND_GGUF, "ltx-one-stage-heretic-gguf", "text_encoder"),
    ):
        request = LtxVideoRequest(pipeline=LTX_PIPELINE_ONE_STAGE, gemma_backend=backend)
        payload = service._resolve_request(request)
        checkpoint = Path(str(payload.get("checkpoint_path") or ""))
        gemma_root = Path(str(payload.get("gemma_root") or ""))
        gguf_path = Path(str(payload.get("gemma_gguf_path") or ""))
        engine_ready = False
        try:
            engine_ready = service.registry.status("ltx").ready
        except Exception:
            engine_ready = False
        openability_error = ltx_checkpoint_openability_error(checkpoint) if checkpoint.is_file() else ""
        route_ready = engine_ready and checkpoint.is_file() and not openability_error and gemma_root.exists()
        if backend == LTX_GEMMA_BACKEND_GGUF:
            route_ready = route_ready and gguf_path.is_file()
            records.append(
                PipelineReadinessRecord(
                    id=f"route:{route}",
                    family="ltx",
                    asset_type=asset_type,
                    path=str(gguf_path),
                    status="blocked-cleanly" if route_ready else "metadata-only",
                    route=route,
                    reason=(
                        "Heretic Q3 GGUF is present, but native LTX generation is blocked until a GGUF backend "
                        "can return every Gemma hidden-state layer and attention mask."
                        if route_ready
                        else openability_error
                        or "Heretic Q3 GGUF route is configured but engine/checkpoint/Gemma sidecar files are incomplete."
                    ),
                    storage="gguf",
                    quantization="Q3_K_M",
                    required_text_encoder="Gemma 3 hidden states from every layer plus attention mask",
                    tokenizer="Gemma tokenizer/processor sidecar folder",
                    smoke_command="venv\\Scripts\\python.exe scripts\\probe_ltx_runtime.py --gguf",
                    suggested_action=(
                        "Use this as a no-dequant probe route only; generation needs a native hidden-state adapter."
                    ),
                    metadata={
                        "checkpoint_path": str(checkpoint),
                        "gemma_root": str(gemma_root),
                        "engine_ready": str(engine_ready).lower(),
                        "checkpoint_openability": "blocked" if openability_error else "ok",
                    },
                )
            )
            continue
        records.append(
            PipelineReadinessRecord(
                id=f"route:{route}",
                family="ltx",
                asset_type=asset_type,
                path=str(checkpoint),
                status="metadata-only" if route_ready else "blocked-cleanly",
                route=route,
                reason=(
                    "One-stage LTX HF Gemma route is wired and ready for bounded generation smoke."
                    if route_ready
                    else openability_error
                    or "One-stage LTX HF Gemma route is missing engine, checkpoint, or Gemma sidecar files."
                ),
                storage="safetensors",
                quantization="runtime fp8-cast",
                required_text_encoder="google/gemma-3-12b-it-qat-q4_0-unquantized",
                tokenizer="Gemma tokenizer/processor sidecar folder",
                smoke_command="scripts\\run_ltx_smoketest.bat",
                suggested_action="Run 1-step/9-frame smoke first, then the 4-step usable default.",
                metadata={
                    "checkpoint_path": str(checkpoint),
                    "gemma_root": str(gemma_root),
                    "engine_ready": str(engine_ready).lower(),
                    "checkpoint_openability": "blocked" if openability_error else "ok",
                },
            )
        )
    return records


def _ltx2b_diffusers_route_record(flags: RuntimeFlags) -> PipelineReadinessRecord:
    from aiwf.services.ltx import LtxService

    service = LtxService(flags, UserSettings())
    checkpoint = service.default_checkpoint_path("diffusers_2b")
    t5_weights = service.default_t5_encoder_path()
    receipt = _latest_file(flags.resolved_output_dir() / "ltx-videos", "ltx2b*.mp4")
    route_ready = checkpoint.is_file() and t5_weights.is_file()
    status = "working" if route_ready and receipt else "metadata-only" if route_ready else "blocked-cleanly"
    missing = []
    if not checkpoint.is_file():
        missing.append("ltx-video-2b-v0.9.5.safetensors")
    if not t5_weights.is_file():
        missing.append("t5xxl_fp16.safetensors")
    reason = (
        "Local Diffusers LTX 2B route has a runtime smoke receipt."
        if status == "working"
        else "Local Diffusers LTX 2B route is wired; run the bounded 1-step/9-frame smoke."
        if route_ready
        else "Local Diffusers LTX 2B route is missing: " + ", ".join(missing)
    )
    return PipelineReadinessRecord(
        id="route:ltx-0.9.5-diffusers-local-t5xxl",
        family="ltx",
        asset_type="pipeline",
        path=str(checkpoint),
        status=status,
        route="ltx-0.9.5-diffusers-local-t5xxl",
        reason=reason,
        storage="safetensors",
        quantization="2B checkpoint + fp16 T5XXL",
        required_text_encoder="models\\flux\\Textencoder\\t5xxl_fp16.safetensors",
        tokenizer="google/t5-v1_1-xxl local tokenizer cache",
        smoke_command="venv\\Scripts\\python.exe scripts\\run_ltx2b_diffusers_smoke.py --json",
        receipt_path=str(receipt) if receipt else "",
        suggested_action="Use this as the practical local LTX smoke route while LTX-2/Gemma GGUF remains blocked.",
        metadata={
            "checkpoint_path": str(checkpoint),
            "t5_weights": str(t5_weights),
            "frame_contract": "8*k+1; 9 frames for the requested 8-frame smoke",
        },
    )


def _inventory_records(flags: RuntimeFlags, *, force_rescan: bool) -> list[PipelineReadinessRecord]:
    from aiwf.infrastructure.model_inventory import get_model_inventory

    records = get_model_inventory(flags, force_rescan=force_rescan)
    out: list[PipelineReadinessRecord] = []
    for record in records:
        if record.family not in {
            "checkpoint",
            "controlnet",
            "lora",
            "runtime_asset",
            "text_encoder",
            "vae",
            "wan",
            "ltx",
            "llm",
        }:
            continue
        path = Path(record.path)
        out.append(_classify_model_record(record.family, record.architecture, path, source="models", metadata=record.metadata))
    return out


def _download_records(download_roots: Iterable[Path], flags: RuntimeFlags) -> list[PipelineReadinessRecord]:
    records: list[PipelineReadinessRecord] = []
    for root in download_roots:
        if not root.exists():
            continue
        for path in _iter_download_model_files(root):
            family, architecture = _family_arch_from_download_path(path)
            if family == "unknown":
                continue
            records.append(_classify_model_record(family, architecture, path, source="downloads", flags=flags))
    return records


def _classify_model_record(
    family: str,
    architecture: str,
    path: Path,
    *,
    source: str,
    metadata: dict[str, str] | None = None,
    flags: RuntimeFlags | None = None,
) -> PipelineReadinessRecord:
    path_text = path.as_posix().lower()
    if family == "wan" or architecture == "wan":
        return _classify_wan(path, source=source, metadata=metadata)
    if (
        architecture == "ltx"
        or "ltx" in path_text
        or family in {"ltx", "runtime_asset", "vae", "text_encoder", "lora"} and "gemma" in path.name.lower()
    ):
        return _classify_ltx(path, source=source, metadata=metadata, flags=flags)
    if family == "llm" or _looks_like_llm(path):
        return _classify_llm(path, source=source, metadata=metadata)
    return _classify_image(path, family=family, architecture=architecture, source=source, metadata=metadata)


def _classify_wan(path: Path, *, source: str, metadata: dict[str, str] | None = None) -> PipelineReadinessRecord:
    from aiwf.services.wan_models import (
        wan_model_header_info,
        wan_model_quant_family,
        wan_model_stage_role,
        wan_model_storage_family,
    )

    lower = path.name.lower()
    storage = wan_model_storage_family(str(path))
    quant = wan_model_quant_family(str(path))
    role = wan_model_stage_role(str(path))
    header = wan_model_header_info(path) if path.suffix.lower() in {".safetensors", ".gguf"} and path.is_file() else None
    if header and header.ok:
        storage = header.storage or storage
        quant = header.quant or quant
        role = header.role or role

    route = "wan-gguf" if storage == "gguf" else "wan-diffusers"
    status = "metadata-only"
    reason = "Wan asset discovered; run a bounded smoke before marking this specific asset working."
    suggested = "Use as a selected model only after preflight resolves a matched VAE/text encoder and high/low pair."

    in_channels = header.in_channels if header and header.ok else 0
    size_gib = _size_gib(path)
    if in_channels == 52 or "fun_control" in lower or "fun-control" in lower:
        status = "unsupported-no-route"
        reason = "Fun-Control/control Wan uses a 52-channel patch embedding; the current high/low I2V route is 36-channel."
        suggested = "Add a dedicated Wan control-conditioning pipeline before exposing this as selectable."
    elif "animate" in lower:
        status = "unsupported-no-route"
        reason = "Wan Animate files need a dedicated Animate pipeline, not the current I2V high/low route."
        suggested = "Keep out of I2V dropdowns until an Animate route is implemented."
    elif "t2v_1.3b" in lower or ("t2v" in lower and "i2v" not in lower and "ti2v" not in lower):
        status = "unsupported-no-route"
        reason = "Wan T2V 1.3B is not wired to the current image-to-video service route."
        suggested = "Implement a Wan text-to-video request path and matching component defaults."
    elif path.suffix.lower() == ".safetensors" and 0 < size_gib < 1.5:
        status = "unsupported-no-route"
        reason = "This looks like a Wan LoRA/control adapter by size, not a base transformer checkpoint."
        suggested = "Route through LoRA/adapter attachment after the base Wan pipeline is selected."
    elif storage == "gguf" and quant in {"q3", "q6"}:
        status = "metadata-only"
        reason = "Wan GGUF Q3/Q6 variants are discovered but remain advanced/untested in bounded smokes."
        suggested = "Smoke Q4/Q5 first; test Q3/Q6 separately with explicit VRAM/error receipts."
    elif storage == "gguf" and quant in {"q4", "q5"} and role in {"high", "low"}:
        reason = "Wan GGUF Q4/Q5 high/low asset is in the supported experimental quantized route."
        suggested = "Pair high and low with the same storage and quantization tier for a 1-step/9-frame smoke."
    elif "ti2v_5b" in lower or (header and header.ok and header.size_class == "5b"):
        route = "wan-fast-5b"
        reason = "Wan TI2V 5B asset is in the supported fast single-transformer route."
        suggested = "Run the fast 5B smoke with Wan 2.2 VAE and shared Diffusers components."
    elif role in {"high", "low"}:
        reason = "Wan high/low transformer discovered; pair with the matching stage before runtime smoke."

    return PipelineReadinessRecord(
        id=f"asset:wan:{_slug(path.stem)}:{source}",
        family="wan",
        asset_type="model",
        path=str(path),
        status=status,
        route=route,
        reason=reason,
        storage=storage,
        quantization=quant,
        required_vae=(header.needs_vae if header and header.ok else ""),
        required_text_encoder="UMT5/Wan shared text_encoder",
        tokenizer="Wan shared tokenizer/spiece",
        smoke_command="venv\\Scripts\\python.exe scripts\\smoke_backend.py --video --video-gen",
        suggested_action=suggested,
        metadata={
            **_clean_metadata(metadata),
            "source": source,
            "role": role,
            "size_gib": f"{size_gib:.2f}" if size_gib else "",
            "header_ok": str(bool(header and header.ok)).lower(),
            "header_arch": header.arch if header and header.ok else "",
            "header_size_class": header.size_class if header and header.ok else "",
            "header_in_channels": str(in_channels or ""),
            "header_error": header.error if header and not header.ok else "",
        },
    )


def _classify_ltx(
    path: Path,
    *,
    source: str,
    metadata: dict[str, str] | None = None,
    flags: RuntimeFlags | None = None,
) -> PipelineReadinessRecord:
    lower = path.name.lower()
    suffix = path.suffix.lower()
    storage = "gguf" if suffix == ".gguf" else "safetensors" if suffix == ".safetensors" else suffix.lstrip(".")
    quant = _quant_from_name(path.name)
    route = "ltx-2.3"
    status = "metadata-only"
    reason = "LTX asset discovered; the isolated worker must preflight before runtime smoke."
    suggested = "Use the native LTX worker with repo-shaped Gemma, a native checkpoint, and the matching upscaler."
    required_text_encoder = "google/gemma-3-12b-it-qat-q4_0-unquantized"
    tokenizer = "Gemma tokenizer files in text_encoder repo folder"
    smoke_command = "scripts\\run_ltx_smoketest.bat"

    if suffix == ".gguf":
        status = "unsupported-no-route"
        reason = "LTX/Gemma GGUF is not wired into the current LTX worker route."
        suggested = "Add a separate GGUF/llama.cpp-style LTX text-encoder or model route before exposing this."
    elif "nvfp4" in lower or "fp4" in lower:
        status = "unsupported-no-route"
        reason = "LTX FP4/NVFP4 assets are present but the worker does not have a stable FP4/NVFP4 loading route."
        suggested = "Keep metadata-only until native FP4/NVFP4 loading is implemented and smoked."
    elif "fp8" in lower and "gemma" not in lower:
        status = "unsupported-no-route"
        reason = "LTX FP8 checkpoint files are present but the current worker route is native safetensors plus runtime fp8-cast/scaled-mm, not FP8 checkpoint loading."
        suggested = "Use BF16 or distilled safetensors first; add an explicit FP8 checkpoint loader later."
    elif "gemma" in lower and suffix == ".safetensors":
        status = "unsupported-no-route"
        reason = "Single-file Gemma safetensors is not the repo-shaped text encoder folder expected by the LTX worker."
        suggested = "Use the Gemma folder with config, tokenizer files, index JSON, and shards."
    elif lower == "ltx-video-2b-v0.9.5.safetensors":
        route = "ltx-0.9.5-diffusers-local-t5xxl"
        reason = "LTX 0.9.5 2B single-file checkpoint can run through Diffusers with a local T5XXL text encoder."
        suggested = "Run scripts\\run_ltx2b_diffusers_smoke.py for a bounded 1-step/9-frame smoke."
        required_text_encoder = "models\\flux\\Textencoder\\t5xxl_fp16.safetensors"
        tokenizer = "google/t5-v1_1-xxl local tokenizer cache"
        smoke_command = "venv\\Scripts\\python.exe scripts\\run_ltx2b_diffusers_smoke.py --json"
    elif _looks_like_ltx_default_checkpoint(path):
        reason = "Native LTX checkpoint is in the worker-supported safetensors family."
        suggested = "Pair with repo-shaped Gemma and the 1.1 spatial upscaler for the next bounded smoke."
    elif "distilled" in lower:
        reason = "LTX distilled checkpoint candidate found."
        suggested = _ltx_distilled_suggestion(path, flags)
    elif "upscaler" in lower:
        reason = "LTX spatial upscaler discovered."
        suggested = "Use with the distilled pipeline when the checkpoint is selected."
    elif "vae" in lower:
        reason = "LTX VAE asset discovered."
        suggested = "Keep as an auxiliary LTX asset; it is not a standalone video pipeline route."

    request_failure = _ltx_request_failure_hint(path)
    if request_failure and status == "metadata-only":
        status = "broken-runtime"
        reason = request_failure
        suggested = "Resolve the worker/runtime error, then rerun the bounded LTX smoke."

    return PipelineReadinessRecord(
        id=f"asset:ltx:{_slug(path.stem)}:{source}",
        family="ltx",
        asset_type="model",
        path=str(path),
        status=status,
        route=route,
        reason=reason,
        storage=storage,
        quantization=quant,
        required_vae="LTX video/audio VAE as requested by upstream pipeline",
        required_text_encoder=required_text_encoder,
        tokenizer=tokenizer,
        smoke_command=smoke_command,
        suggested_action=suggested,
        metadata={
            **_clean_metadata(metadata),
            "source": source,
            "size_gib": f"{_size_gib(path):.2f}" if path.exists() else "",
        },
    )


def _classify_llm(path: Path, *, source: str, metadata: dict[str, str] | None = None) -> PipelineReadinessRecord:
    storage = "gguf" if path.suffix.lower() == ".gguf" else path.suffix.lower().lstrip(".")
    return PipelineReadinessRecord(
        id=f"asset:llm:{_slug(path.stem)}:{source}",
        family="llm-vl",
        asset_type="model",
        path=str(path),
        status="unsupported-no-route" if storage == "gguf" else "metadata-only",
        route="llm-gguf",
        reason=(
            "GGUF LLM/VL assets are discovered, but AIWF does not yet have an isolated llama.cpp/llm_gguf worker."
            if storage == "gguf"
            else "LLM/VL asset is metadata-only until an AIWF inference/eval route is added."
        ),
        storage=storage,
        quantization=_quant_from_name(path.name),
        tokenizer="model-specific tokenizer assets required",
        smoke_command="",
        suggested_action="Add engines/llm_gguf with llama.cpp or llama-cpp-python, then add tokenizer/eval smoke receipts.",
        metadata={
            **_clean_metadata(metadata),
            "source": source,
            "size_gib": f"{_size_gib(path):.2f}" if path.exists() else "",
        },
    )


def _classify_image(
    path: Path,
    *,
    family: str,
    architecture: str,
    source: str,
    metadata: dict[str, str] | None = None,
) -> PipelineReadinessRecord:
    lower = path.name.lower()
    status = "metadata-only"
    route = _image_route_for_arch(architecture, path)
    reason = "Image model asset discovered; runtime smoke not executed by the readiness ledger."
    suggested = "Run the targeted image smoke for this checkpoint or pipeline family before marking it working."

    if "fluxedupfluxnsfw_110fp8" in lower:
        status = "broken-runtime"
        reason = "Known Flux FP8 selectable failure: checkpoint keys do not match the expected Flux loader schema."
        suggested = "Keep blocked until key mapping/loading support is fixed."
    elif "fluxfusion" in lower and path.suffix.lower() == ".gguf" and ("nf4" in lower or "ggufq4" in lower):
        status = "broken-runtime"
        reason = "Known Flux GGUF/NF4 mismatch: metadata/quantization does not match the current image route."
        suggested = "Do not expose as a normal Flux checkpoint until a compatible GGUF/NF4 route exists."
    elif lower == "4xbhi_dat2_multiblurjpg.safetensors":
        status = "broken-runtime"
        reason = "Known bad selectable: checkpoint is missing the expected CLIP text model."
        suggested = "Classify as an auxiliary/upscale asset instead of a txt2img checkpoint."
    elif path.suffix.lower() == ".gguf" and architecture in {"flux", "flux2-klein", "z-image"}:
        status = "unsupported-no-route"
        reason = "Image GGUF transformer route is not the default Diffusers checkpoint route."
        suggested = "Keep separate from normal checkpoint dropdowns until the matching GGUF loader is wired."

    return PipelineReadinessRecord(
        id=f"asset:image:{_slug(path.stem)}:{source}",
        family="image",
        asset_type=family,
        path=str(path),
        status=status,
        route=route,
        reason=reason,
        storage="gguf" if path.suffix.lower() == ".gguf" else "safetensors" if path.suffix.lower() == ".safetensors" else path.suffix.lower().lstrip("."),
        quantization=_quant_from_name(path.name),
        smoke_command="venv\\Scripts\\python.exe scripts\\smoke_backend.py --image --steps 1 --width 512 --height 512",
        suggested_action=suggested,
        metadata={
            **_clean_metadata(metadata),
            "source": source,
            "architecture": architecture,
            "size_gib": f"{_size_gib(path):.2f}" if path.exists() else "",
        },
    )


def _iter_download_model_files(root: Path) -> Iterable[Path]:
    for current, dir_names, file_names in os.walk(root):
        dir_names[:] = [name for name in dir_names if not name.startswith(".")]
        for filename in sorted(file_names, key=str.lower):
            path = Path(current) / filename
            if path.suffix.lower() not in MODEL_FILE_SUFFIXES:
                continue
            text = path.as_posix().lower()
            if any(keyword in text for keyword in DOWNLOAD_KEYWORDS):
                yield path


def _family_arch_from_download_path(path: Path) -> tuple[str, str]:
    text = path.as_posix().lower()
    if "wan" in text:
        return "wan", "wan"
    if "ltx" in text:
        return "ltx", "ltx"
    if "gemma" in text:
        return ("ltx", "ltx") if "ltx" in text or path.parent.name.lower() == "downloads" else ("llm", "llm")
    if any(token in text for token in ("flux", "qwen", "sana", "z-image", "zimage")):
        return "runtime_asset", _download_image_arch(path)
    return "unknown", "unknown"


def _download_image_arch(path: Path) -> str:
    lower = path.as_posix().lower()
    if "flux.2" in lower or "flux2" in lower:
        return "flux2-klein"
    if "flux" in lower:
        return "flux"
    if "qwen" in lower:
        return "qwen-image"
    if "sana" in lower:
        return "sana"
    if "z-image" in lower or "zimage" in lower:
        return "z-image"
    return "unknown"


def _pipeline_family(route: str) -> str:
    route = route.lower()
    if "wan" in route:
        return "wan"
    if "ltx" in route:
        return "ltx"
    if "sana-video" in route:
        return "video"
    if "qwen" in route or "sana" in route or "onnx" in route or "diffusers" in route or "image" in route:
        return "image"
    return "pipeline"


def _onnx_root(flags: RuntimeFlags, settings: UserSettings) -> Path:
    raw = (settings.onnx_model_dir or "").strip()
    if raw:
        path = Path(raw)
        return path.resolve() if path.is_absolute() else (flags.data_dir / path).resolve()
    return (flags.resolved_models_dir() / "onnx").resolve()


def _first_item_path(result) -> str:  # noqa: ANN001
    for item in result.items:
        if item.path is not None:
            return str(item.path)
    return ""


def _latest_receipt(flags: RuntimeFlags, family: str, *, route: str = "") -> Path | None:
    output_root = flags.resolved_output_dir()
    if family == "wan":
        return _latest_file(output_root / "video" / "wan", "*.mp4")
    if family == "ltx":
        pattern = "ltx2b*.mp4" if route in {"ltx-2b", "ltx-2b-diffusers"} else "ltx23*.mp4"
        return _latest_file(output_root / "ltx-videos", pattern)
    if family == "image":
        if route not in {"diffusers"}:
            return None
        candidates = [
            _latest_file(output_root / "txt2img-images", "*.png"),
            _latest_file(output_root / "img2img-images", "*.png"),
            _latest_file(output_root / "inpaint-images", "*.png"),
            _latest_file(output_root / "workflow-images", "*.png"),
        ]
        return max((path for path in candidates if path is not None), key=lambda p: p.stat().st_mtime, default=None)
    return None


def _latest_file(root: Path, pattern: str) -> Path | None:
    if not root.is_dir():
        return None
    files = [path for path in root.glob(pattern) if path.is_file()]
    return max(files, key=lambda path: path.stat().st_mtime, default=None)


def _smoke_command_for_family(family: str) -> str:
    if family == "wan":
        return "venv\\Scripts\\python.exe scripts\\smoke_backend.py --video --video-gen"
    if family == "ltx":
        return "scripts\\run_ltx_smoketest.bat"
    if family == "image":
        return "venv\\Scripts\\python.exe scripts\\smoke_backend.py --image --steps 1 --width 512 --height 512"
    return "venv\\Scripts\\python.exe scripts\\smoke_models_and_pipelines.py"


def _smoke_command_for_route(family: str, route: str) -> str:
    if family == "ltx" and route in {"ltx-2b", "ltx-2b-diffusers"}:
        return "venv\\Scripts\\python.exe scripts\\run_ltx2b_diffusers_smoke.py --json"
    return _smoke_command_for_family(family)


def _image_route_for_arch(architecture: str, path: Path) -> str:
    arch = str(architecture or "").lower()
    if arch in {"qwen-image", "qwen-image-nunchaku"}:
        return "qwen-nunchaku" if "nunchaku" in path.as_posix().lower() or arch.endswith("nunchaku") else "qwen-image"
    if arch in {"sana", "sana-video"}:
        return arch
    if arch in {"flux2-klein", "z-image"}:
        return arch
    if arch == "flux":
        return "flux"
    return "diffusers"


def _looks_like_llm(path: Path) -> bool:
    lower = path.as_posix().lower()
    return "/llm/" in lower.replace("\\", "/") or lower.endswith(".gguf") and any(token in lower for token in ("gemma", "qwen", "llama", "mistral"))


def _looks_like_ltx_default_checkpoint(path: Path) -> bool:
    lower = path.name.lower()
    return lower in {
        "ltx-2.3-22b-dev-bf16.safetensors",
        "ltx-2.3-22b-distilled-1.1.safetensors",
    }


def _ltx_distilled_suggestion(path: Path, flags: RuntimeFlags | None) -> str:
    if flags is None:
        return "Move under models\\ltx\\checkpoints or select it explicitly in the LTX request."
    expected = flags.resolved_models_dir() / "ltx" / "checkpoints" / "ltx-2.3-22b-distilled-1.1.safetensors"
    if path.resolve() == expected.resolve():
        return "Pair with repo-shaped Gemma and the 1.1 spatial upscaler for the next bounded smoke."
    return f"Move or copy to {expected}, or select this file explicitly as the LTX checkpoint."


def _ltx_request_failure_hint(path: Path) -> str:
    lower = path.name.lower()
    if "bf16" in lower:
        request = Path("outputs") / "ltx-videos" / "requests" / "ltx_bf16_disk_smoke.json"
        if request.is_file() and not any((Path("outputs") / "ltx-videos").glob("*.mp4")):
            return "Previous LTX BF16 smoke request exists but no LTX MP4 receipt was created."
    return ""


def _quant_from_name(filename: str) -> str:
    lower = filename.lower()
    for token in (
        "q2_k",
        "q3_k_s",
        "q3_k_m",
        "q4_k_s",
        "q4_k_m",
        "q4_0",
        "q5_k_s",
        "q5_k_m",
        "q5_0",
        "q6_k",
        "q8_0",
        "nf4",
        "nvfp4",
        "fp4",
        "fp8",
        "bf16",
        "fp16",
    ):
        if token in lower:
            return token.upper()
    match = re.search(r"(?:^|[_\-.])q([2-8])(?:[_\-.]|$)", lower)
    return f"Q{match.group(1)}" if match else ""


def _size_gib(path: Path) -> float:
    try:
        return path.stat().st_size / (1024**3)
    except OSError:
        return 0.0


def _slug(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return text or "unknown"


def _scrub_value(value: object) -> str:
    text = str(value)
    if text == "hf_safetensors":
        return text
    if "token" in text.lower() or text.startswith(("hf_", "hf-")):
        return "[redacted]"
    return text


def _clean_metadata(metadata: dict[str, str] | None) -> dict[str, str]:
    if not metadata:
        return {}
    allowed_prefixes = ("modelspec.",)
    allowed_keys = {
        "ss_base_model_version",
        "ss_sd_model_name",
        "ss_network_module",
    }
    skipped_fragments = ("tag_frequency", "dataset_dirs", "bucket_info", "training_comment")
    cleaned: dict[str, str] = {}
    for key, raw in metadata.items():
        if any(fragment in key for fragment in skipped_fragments):
            continue
        if not key.startswith(allowed_prefixes) and key not in allowed_keys:
            continue
        value = _scrub_value(raw)
        if len(value) > 160:
            value = value[:157].rstrip() + "..."
        cleaned[str(key)] = value
        if len(cleaned) >= 12:
            break
    return cleaned
