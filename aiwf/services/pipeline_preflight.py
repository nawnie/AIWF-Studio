from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.infrastructure.diffusers.checkpoints import diffusers_dir_has_required_local_files


@dataclass(frozen=True)
class PipelineCheckItem:
    name: str
    ok: bool
    message: str
    path: Path | None = None


@dataclass(frozen=True)
class PipelinePreflightResult:
    pipeline: str
    ok: bool
    items: tuple[PipelineCheckItem, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)

    def markdown(self) -> str:
        mark = "OK" if self.ok else "Blocked"
        lines = [f"**{mark}: {self.pipeline} pipeline**"]
        lines.extend(
            f"- {'OK' if item.ok else 'Missing'} **{item.name}:** {item.message}"
            for item in self.items
        )
        if self.warnings:
            lines.append("\n**Warnings**")
            lines.extend(f"- {warning}" for warning in self.warnings)
        return "\n".join(lines)


_ONNX_REQUIRED_MODELS = {
    "text_encoder": Path("text_encoder") / "model.onnx",
    "unet": Path("unet") / "model.onnx",
    "vae_decoder": Path("vae_decoder") / "model.onnx",
}

_TOKENIZER_HINTS = (
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer_config.json",
)


def preflight_onnx_pipeline(
    model_dir: str | Path,
    *,
    provider_preference: str = "auto",
    available_providers: list[str] | None = None,
) -> PipelinePreflightResult:
    """Check whether an ONNX txt2img model folder is usable.

    This function is intentionally light: it never creates ONNX Runtime
    sessions and it does not import torch or diffusers.
    """
    root = Path(model_dir).expanduser().resolve()
    items: list[PipelineCheckItem] = []
    warnings: list[str] = []
    metadata: dict[str, str] = {"model_dir": str(root), "provider_preference": provider_preference}

    items.append(
        PipelineCheckItem(
            "model folder",
            root.is_dir(),
            str(root) if root.exists() else f"Folder does not exist: {root}",
            root,
        )
    )

    for name, rel in _ONNX_REQUIRED_MODELS.items():
        path = root / rel
        items.append(
            PipelineCheckItem(
                name,
                path.is_file(),
                str(path) if path.is_file() else f"Expected {rel.as_posix()}",
                path,
            )
        )

    tokenizer_dir = root / "tokenizer"
    tokenizer_files = [tokenizer_dir / hint for hint in _TOKENIZER_HINTS]
    tokenizer_ok = tokenizer_dir.is_dir() and any(path.is_file() for path in tokenizer_files)
    items.append(
        PipelineCheckItem(
            "tokenizer",
            tokenizer_ok,
            str(tokenizer_dir) if tokenizer_ok else "Expected local tokenizer assets in tokenizer/",
            tokenizer_dir,
        )
    )

    providers = available_providers
    if providers is None:
        providers = _load_available_onnx_providers(warnings)
    metadata["available_providers"] = ", ".join(providers) if providers else ""
    provider_ok, provider_message = _provider_status(provider_preference, providers)
    items.append(PipelineCheckItem("execution provider", provider_ok, provider_message))

    if provider_preference in {"directml", "cpu"}:
        warnings.append("DirectML and CPU are compatibility paths. NVIDIA CUDA remains the preferred ONNX provider on this machine.")
    warnings.append("ONNX is optional. Diffusers remains the required baseline pipeline.")

    return PipelinePreflightResult(
        pipeline="ONNX",
        ok=all(item.ok for item in items),
        items=tuple(items),
        warnings=tuple(warnings),
        metadata=metadata,
    )


def preflight_diffusers_pipeline() -> PipelinePreflightResult:
    """Check the mandatory Diffusers baseline without importing heavy models."""
    warnings: list[str] = []
    items = [
        _import_check("diffusers", "Required baseline inference package."),
        _import_check("transformers", "Required for tokenizers/text encoders; must stay <5."),
        _import_check("torch", "Required runtime tensor package."),
        _import_check("safetensors", "Preferred local model file format."),
    ]
    try:
        import transformers

        version = tuple(int(part) for part in transformers.__version__.split(".")[:2])
        if version >= (5, 0):
            items.append(
                PipelineCheckItem(
                    "transformers version",
                    False,
                    f"transformers {transformers.__version__} is unsupported; use >=4.44,<5.",
                )
            )
        else:
            items.append(PipelineCheckItem("transformers version", True, transformers.__version__))
    except Exception as exc:
        warnings.append(f"Could not inspect transformers version: {exc}")

    return PipelinePreflightResult(
        pipeline="Diffusers",
        ok=all(item.ok for item in items),
        items=tuple(items),
        warnings=tuple(warnings),
    )


def preflight_image_runtime_pipelines() -> PipelinePreflightResult:
    """Check optional image pipeline classes without loading model weights."""
    required = [
        _diffusers_attr_check("Flux2KleinPipeline", "Required for Flux.2 Klein full-folder/single-transformer routes."),
        _diffusers_attr_check("ZImagePipeline", "Required for Z-Image full-folder/single-transformer routes."),
        _diffusers_attr_check("QwenImagePipeline", "Required for Qwen Image full-folder routes."),
        _diffusers_attr_check("SanaPipeline", "Required for standard Sana full-folder routes."),
        _diffusers_attr_check("SanaSprintPipeline", "Required for Sana Sprint full-folder routes."),
    ]
    optional = [
        _diffusers_attr_check("Krea2Pipeline", "Required for future Krea 2 Diffusers-folder routes."),
    ]
    warnings = []
    if not optional[0].ok:
        warnings.append("Krea 2 is blocked until the installed Diffusers package exposes Krea2Pipeline.")
    return PipelinePreflightResult(
        pipeline="Image Runtime Families",
        ok=all(item.ok for item in required),
        items=tuple([*required, *optional]),
        warnings=tuple(warnings),
    )


def preflight_krea2_pipeline(flags: RuntimeFlags | str | Path) -> PipelinePreflightResult:
    runtime_flags = flags if isinstance(flags, RuntimeFlags) else RuntimeFlags(data_dir=Path(flags))
    models = runtime_flags.resolved_models_dir()
    diffusers_folder, incomplete_diffusers_folder = _find_krea2_diffusers_folder(models)
    transformer = _first_existing(
        models / "krea2" / "UNet",
        (
            "krea2_turbo_fp8_scaled.safetensors",
            "krea2_turbo_nvfp4.safetensors",
            "krea2_turbo_bf16.safetensors",
            "krea2_raw_fp8_scaled.safetensors",
        ),
    )
    text_encoder = _first_existing(
        models / "krea2" / "Textencoder",
        ("qwen3vl_4b_fp8_scaled.safetensors", "qwen3vl_4b_bf16.safetensors"),
    )
    vae = _first_existing(
        models / "krea2" / "VAE",
        ("qwen_image_vae.safetensors",),
    ) or _first_existing(models / "VAE", ("qwen_image_vae.safetensors",))
    class_check = _diffusers_attr_check("Krea2Pipeline", "Required for Krea 2 Diffusers-folder loading in AIWF.")
    diffusers_message = (
        str(diffusers_folder)
        if diffusers_folder
        else f"Expected a complete Krea 2 Diffusers folder under {models / 'krea2' / 'Diffusers'}"
    )
    if incomplete_diffusers_folder and not diffusers_folder:
        diffusers_message = f"Incomplete Krea 2 Diffusers folder under {incomplete_diffusers_folder}"
    items = [
        class_check,
        PipelineCheckItem(
            "Krea 2 Diffusers folder",
            diffusers_folder is not None,
            diffusers_message,
            diffusers_folder or incomplete_diffusers_folder,
        ),
        PipelineCheckItem(
            "Krea 2 split transformer",
            transformer is not None,
            str(transformer) if transformer else f"Expected Krea 2 transformer under {models / 'krea2' / 'UNet'}",
            transformer,
        ),
        PipelineCheckItem(
            "Qwen3-VL text encoder",
            text_encoder is not None,
            str(text_encoder) if text_encoder else f"Expected qwen3vl_4b_* sidecar under {models / 'krea2' / 'Textencoder'}",
            text_encoder,
        ),
        PipelineCheckItem(
            "Qwen Image VAE",
            vae is not None,
            str(vae) if vae else "Expected qwen_image_vae.safetensors under models/krea2/VAE or models/VAE.",
            vae,
        ),
    ]
    warnings: list[str] = []
    if transformer and text_encoder and vae and diffusers_folder is None:
        warnings.append(
            "Krea 2 split files are present, but AIWF still needs a split-file loader before they can generate."
        )
    if incomplete_diffusers_folder and diffusers_folder is None:
        warnings.append("Finish or remove the incomplete Krea 2 Diffusers snapshot before selecting it.")
    if not class_check.ok:
        warnings.append("Install diffusers>=0.39.0, then restart AIWF before running Krea 2.")
    if not warnings:
        warnings.append("Use the Krea 2 Turbo preset first; Raw is heavier and should be smoked separately.")
    return PipelinePreflightResult(
        pipeline="Krea 2",
        ok=class_check.ok and diffusers_folder is not None,
        items=tuple(items),
        warnings=tuple(warnings),
        metadata={
            "default_profile": runtime_flags.effective_vram_profile(),
            "recommended_bundle": "krea2 for the runnable Turbo Diffusers folder; krea2-low/mid/high are split-file staging bundles",
            "diffusers_folder": str(diffusers_folder) if diffusers_folder else "",
            "split_transformer": str(transformer) if transformer else "",
            "split_text_encoder": str(text_encoder) if text_encoder else "",
            "vae": str(vae) if vae else "",
        },
    )


def preflight_anima_pipeline(flags: RuntimeFlags | str | Path) -> PipelinePreflightResult:
    runtime_flags = flags if isinstance(flags, RuntimeFlags) else RuntimeFlags(data_dir=Path(flags))
    models = runtime_flags.resolved_models_dir()
    transformer = _first_existing(
        models / "anima" / "UNet",
        ("anima-base-v1.0.safetensors", "anima-preview3-base.safetensors"),
    )
    text_encoder = _first_existing(models / "anima" / "Textencoder", ("qwen_3_06b_base.safetensors",))
    vae = _first_existing(
        models / "anima" / "VAE",
        ("qwen_image_vae.safetensors",),
    ) or _first_existing(models / "VAE", ("qwen_image_vae.safetensors",))
    items = [
        PipelineCheckItem(
            "native Anima loader",
            False,
            "AIWF does not yet include a native Anima loader for split files; upstream release is ComfyUI-native.",
        ),
        PipelineCheckItem(
            "Anima transformer",
            transformer is not None,
            str(transformer) if transformer else f"Expected Anima transformer under {models / 'anima' / 'UNet'}",
            transformer,
        ),
        PipelineCheckItem(
            "Qwen 0.6B text encoder",
            text_encoder is not None,
            str(text_encoder) if text_encoder else f"Expected qwen_3_06b_base.safetensors under {models / 'anima' / 'Textencoder'}",
            text_encoder,
        ),
        PipelineCheckItem(
            "Qwen Image VAE",
            vae is not None,
            str(vae) if vae else "Expected qwen_image_vae.safetensors under models/anima/VAE or models/VAE.",
            vae,
        ),
    ]
    return PipelinePreflightResult(
        pipeline="Anima",
        ok=all(item.ok for item in items),
        items=tuple(items),
        warnings=("Anima is tracked as a low/mid-VRAM target, but generation is blocked until the native split-file loader exists.",),
        metadata={
            "default_profile": runtime_flags.effective_vram_profile(),
            "recommended_bundle": "anima for base setup; anima-low/mid/high mirror VRAM setup profiles",
        },
    )


def preflight_sana_video_pipeline(
    flags: RuntimeFlags | str | Path,
    settings: UserSettings | None = None,
    request=None,  # noqa: ANN001
) -> PipelinePreflightResult:
    from aiwf.core.domain.sana_video import SanaVideoRequest
    from aiwf.services.sana_video import SanaVideoService

    runtime_flags = flags if isinstance(flags, RuntimeFlags) else RuntimeFlags(data_dir=Path(flags))
    service = SanaVideoService(runtime_flags, settings or UserSettings())
    base_request = request or SanaVideoRequest()
    if not isinstance(base_request, SanaVideoRequest):
        base_request = SanaVideoRequest.model_validate(base_request)
    model_path = service.default_model_path()
    if base_request.model_path:
        model_path = Path(base_request.model_path)
        if not model_path.is_absolute():
            model_path = (runtime_flags.data_dir / model_path).resolve()

    items = [
        _diffusers_attr_check("SanaVideoPipeline", "Required for Sana text-to-video routes."),
        _diffusers_attr_check("SanaImageToVideoPipeline", "Required for Sana image-to-video routes."),
    ]
    model_index = model_path / "model_index.json"
    warnings = []
    if not model_index.is_file():
        warnings.append(
            f"Sana video model snapshot is not installed at {model_path}; runtime is available once the folder is downloaded."
        )
    warnings.append("Sana video exports silent MP4s; use the MMAudio post-process route when generated audio is requested.")
    return PipelinePreflightResult(
        pipeline="Sana Video",
        ok=all(item.ok for item in items),
        items=tuple(items),
        warnings=tuple(warnings),
        metadata={
            "model_path": str(model_path),
            "model_installed": str(model_index.is_file()).lower(),
            "default_repo": "Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
            "sage_attention": service.sage_status(),
            "bitsandbytes": service.bitsandbytes_status(),
            "default_quantization": base_request.quantization,
            "vae_tiling": base_request.vae_tiling,
        },
    )


def preflight_wan_pipeline(
    flags: RuntimeFlags | str | Path,
    settings: UserSettings | None = None,
    request=None,  # noqa: ANN001
) -> PipelinePreflightResult:
    from aiwf.core.domain.wan import WAN_RUNTIME_FAST_5B, WanI2VRequest
    from aiwf.services.wan import WanService

    runtime_flags = flags if isinstance(flags, RuntimeFlags) else RuntimeFlags(data_dir=Path(flags))
    base_request = request or WanI2VRequest(runtime_mode=WAN_RUNTIME_FAST_5B)
    service = WanService(runtime_flags, settings or UserSettings())
    backend_ok = service.available()
    result = service.preflight(base_request, image_present=True)

    items: list[PipelineCheckItem] = [
        PipelineCheckItem(
            "Wan backend",
            backend_ok,
            "diffusers Wan runtime is importable"
            if backend_ok
            else "update diffusers (>=0.35) and install ftfy, then restart",
        )
    ]
    model_path = Path(result.model_id) if result.model_id else None
    items.append(
        PipelineCheckItem(
            "fast 5B transformer",
            bool(model_path and model_path.exists()),
            str(model_path) if model_path and model_path.exists() else "local Wan TI2V 5B model did not resolve",
            model_path,
        )
    )
    components_path = Path(result.components_base) if result.components_base else None
    items.append(
        PipelineCheckItem(
            "shared components",
            bool(components_path and components_path.is_dir()),
            str(components_path)
            if components_path and components_path.is_dir()
            else "missing local text_encoder/tokenizer/scheduler component base",
            components_path,
        )
    )
    vae_path = Path(result.vae) if result.vae else None
    items.append(
        PipelineCheckItem(
            "Wan VAE",
            bool(vae_path and vae_path.exists()),
            str(vae_path) if vae_path and vae_path.exists() else "missing Wan 2.2 VAE for fast 5B",
            vae_path,
        )
    )
    if result.text_encoder:
        text_encoder_path = Path(result.text_encoder)
        items.append(
            PipelineCheckItem(
                "text encoder override",
                text_encoder_path.exists(),
                str(text_encoder_path)
                if text_encoder_path.exists()
                else f"selected text encoder is not local: {text_encoder_path}",
                text_encoder_path,
            )
        )
    else:
        explicit_text_encoder = bool(str(getattr(base_request, "text_encoder_path", "") or "").strip())
        items.append(
            PipelineCheckItem(
                "text encoder",
                not explicit_text_encoder,
                "using component base text_encoder"
                if not explicit_text_encoder
                else "explicit text encoder did not resolve",
            )
        )

    for index, error in enumerate(result.errors[:8], start=1):
        items.append(PipelineCheckItem(f"preflight error {index}", False, error))

    metadata = {
        "runtime_mode": str(base_request.runtime_mode),
        "model_id": str(result.model_id or ""),
        "components_base": str(result.components_base or ""),
        "vae": str(result.vae or ""),
        "text_encoder": str(result.text_encoder or ""),
        "steps": str(base_request.steps),
        "cfg_scale": str(base_request.guidance_scale),
        "sampler": str(base_request.sampler),
        "scheduler": str(base_request.sigma_type),
        "offload": str(base_request.offload),
    }
    return PipelinePreflightResult(
        pipeline="Wan fast 5B",
        ok=not result.errors and all(item.ok for item in items),
        items=tuple(items),
        warnings=tuple(result.warnings),
        metadata=metadata,
    )


def preflight_qwen_nunchaku_pipeline(flags: RuntimeFlags | str | Path) -> PipelinePreflightResult:
    from aiwf.services.qwen_nunchaku import QwenNunchakuService

    runtime_flags = flags if isinstance(flags, RuntimeFlags) else RuntimeFlags(data_dir=Path(flags))
    service = QwenNunchakuService(runtime_flags)
    status = service.status()
    base_blockers = tuple(message for message in status.messages if message.startswith("base components"))
    items = [
        PipelineCheckItem(
            "engine python",
            status.python_exe.is_file(),
            str(status.python_exe) if status.python_exe.is_file() else f"Missing engine runtime: {status.python_exe}",
            status.python_exe,
        ),
        PipelineCheckItem(
            "runner script",
            status.runner_script.is_file(),
            str(status.runner_script) if status.runner_script.is_file() else f"Missing runner: {status.runner_script}",
            status.runner_script,
        ),
        PipelineCheckItem(
            "base components",
            status.base_dir.is_dir() and not base_blockers,
            "; ".join(base_blockers)
            if base_blockers
            else str(status.base_dir)
            if status.base_dir.is_dir()
            else f"Missing Qwen Image base folder: {status.base_dir}",
            status.base_dir,
        ),
        PipelineCheckItem(
            "transformer",
            status.transformer_path.is_file(),
            str(status.transformer_path)
            if status.transformer_path.is_file()
            else f"Missing Nunchaku transformer: {status.transformer_path}",
            status.transformer_path,
        ),
    ]
    return PipelinePreflightResult(
        pipeline="Qwen Nunchaku",
        ok=all(item.ok for item in items),
        items=tuple(items),
        metadata={
            "python_exe": str(status.python_exe),
            "runner_script": str(status.runner_script),
            "base_dir": str(status.base_dir),
            "transformer_path": str(status.transformer_path),
            "storage_mode": "single_transformer_safetensors_plus_base_components",
        },
    )


def preflight_ltx_pipeline(
    flags: RuntimeFlags | str | Path,
    settings: UserSettings | None = None,
    request=None,  # noqa: ANN001
) -> PipelinePreflightResult:
    from aiwf.core.domain.ltx import (
        LTX_GEMMA_BACKEND_GGUF,
        LTX_PIPELINE_DIFFUSERS_2B,
        LTX_PIPELINE_DISTILLED,
        LTX_PIPELINE_ONE_STAGE,
        LtxVideoRequest,
    )
    from aiwf.services.ltx import (
        LtxService,
        ltx_checkpoint_openability_error,
        ltx_checkpoint_requires_no_offload,
        ltx_native_checkpoint_runtime_blocker,
    )

    runtime_flags = flags if isinstance(flags, RuntimeFlags) else RuntimeFlags(data_dir=Path(flags))
    service = LtxService(runtime_flags, settings or UserSettings())
    base_request = request or service.default_launch_request()
    if not isinstance(base_request, LtxVideoRequest):
        base_request = LtxVideoRequest.model_validate(base_request)

    payload = service._resolve_request(base_request)
    selected_pipeline = str(payload.get("pipeline") or base_request.pipeline)
    checkpoint = Path(str(payload.get("checkpoint_path") or ""))
    t5_encoder = Path(str(payload.get("t5_encoder_path") or ""))
    gemma_root = Path(str(payload.get("gemma_root") or ""))
    gemma_backend = str(payload.get("gemma_backend") or "")
    gemma_gguf = Path(str(payload.get("gemma_gguf_path") or ""))
    upsampler = Path(str(payload.get("spatial_upsampler_path") or ""))

    if selected_pipeline == LTX_PIPELINE_DIFFUSERS_2B:
        items = [
            PipelineCheckItem(
                "checkpoint",
                checkpoint.is_file(),
                str(checkpoint) if checkpoint.is_file() else f"missing LTX 2B checkpoint: {checkpoint}",
                checkpoint,
            ),
            PipelineCheckItem(
                "T5XXL text encoder",
                t5_encoder.is_file(),
                str(t5_encoder) if t5_encoder.is_file() else f"missing T5XXL text encoder: {t5_encoder}",
                t5_encoder,
            ),
        ]
    else:
        status = service.registry.status("ltx")
        items = [
            PipelineCheckItem(
                "engine worker",
                status.ready,
                "ready via isolated worker" if status.ready else "; ".join(status.messages),
                status.worker_script,
            ),
            PipelineCheckItem(
                "checkpoint",
                checkpoint.is_file(),
                str(checkpoint) if checkpoint.is_file() else f"missing LTX checkpoint: {checkpoint}",
                checkpoint,
            ),
            PipelineCheckItem(
                "Gemma tokenizer/processor",
                gemma_root.exists(),
                str(gemma_root) if gemma_root.exists() else f"missing Gemma root: {gemma_root}",
                gemma_root,
            ),
        ]
        if checkpoint.is_file():
            openability_error = ltx_checkpoint_openability_error(checkpoint)
            items.append(
                PipelineCheckItem(
                    "checkpoint openability",
                    not openability_error,
                    openability_error or "shallow safetensors open check passed",
                    checkpoint,
                )
            )
            runtime_blocker = ltx_native_checkpoint_runtime_blocker(checkpoint)
            items.append(
                PipelineCheckItem(
                    "native worker compatibility",
                    not runtime_blocker,
                    runtime_blocker or "native worker compatibility check passed",
                    checkpoint,
                )
            )
    if gemma_backend == LTX_GEMMA_BACKEND_GGUF:
        items.append(
            PipelineCheckItem(
                "Gemma GGUF",
                gemma_gguf.is_file() and gemma_gguf.suffix.lower() == ".gguf",
                str(gemma_gguf)
                if gemma_gguf.is_file() and gemma_gguf.suffix.lower() == ".gguf"
                else f"missing Gemma GGUF file: {gemma_gguf}",
                gemma_gguf,
            )
        )
    if selected_pipeline == LTX_PIPELINE_DISTILLED:
        items.append(
            PipelineCheckItem(
                "spatial upscaler",
                upsampler.is_file(),
                str(upsampler) if upsampler.is_file() else f"missing LTX spatial upscaler: {upsampler}",
                upsampler,
            )
        )

    warnings: list[str] = []
    distilled_missing = not service.default_checkpoint_path(LTX_PIPELINE_DISTILLED).is_file()
    if base_request.pipeline != selected_pipeline or (selected_pipeline == LTX_PIPELINE_ONE_STAGE and distilled_missing):
        warnings.append(
            "Distilled checkpoint is missing, so the launch default falls back to the installed one-stage checkpoint."
        )
    if selected_pipeline == LTX_PIPELINE_ONE_STAGE and checkpoint.is_file():
        size_gib = checkpoint.stat().st_size / (1024**3)
        if ltx_checkpoint_requires_no_offload(checkpoint):
            warnings.append(
                "Selected LTX FP8 checkpoint uses offload=none on this Windows runtime; CPU/disk streaming "
                "was isolated as the access-violation path."
            )
        elif size_gib > 12:
            warnings.append(
                f"Selected LTX checkpoint is {size_gib:.1f} GiB; keep CPU offload enabled on 8-12 GB cards."
            )
    if str(payload.get("offload") or "").lower() == "none" and not ltx_checkpoint_requires_no_offload(checkpoint):
        warnings.append("LTX offload is disabled; use CPU offload for consumer GPUs.")
    if gemma_backend == LTX_GEMMA_BACKEND_GGUF:
        warnings.append(
            "Native Gemma GGUF is path-checked only; LTX generation still needs a GGUF backend "
            "that returns every Gemma hidden-state layer."
        )

    metadata = {
        "requested_pipeline": str(base_request.pipeline),
        "selected_pipeline": selected_pipeline,
        "checkpoint_path": str(checkpoint),
        "t5_encoder_path": str(t5_encoder),
        "t5_tokenizer": str(payload.get("t5_tokenizer") or ""),
        "gemma_root": str(gemma_root),
        "gemma_backend": gemma_backend,
        "gemma_gguf_path": str(gemma_gguf) if str(gemma_gguf) != "." else "",
        "spatial_upsampler_path": str(upsampler),
        "offload": str(payload.get("offload") or ""),
        "quantization": str(payload.get("quantization") or ""),
        "steps": str(payload.get("steps") or ""),
    }
    return PipelinePreflightResult(
        pipeline="LTX 2B" if selected_pipeline == LTX_PIPELINE_DIFFUSERS_2B else "LTX 2.3",
        ok=all(item.ok for item in items),
        items=tuple(items),
        warnings=tuple(warnings),
        metadata=metadata,
    )


def _import_check(module_name: str, message: str) -> PipelineCheckItem:
    try:
        __import__(module_name)
        return PipelineCheckItem(module_name, True, message)
    except Exception as exc:
        return PipelineCheckItem(module_name, False, f"{message} Import failed: {exc}")


def _diffusers_attr_check(attr_name: str, message: str) -> PipelineCheckItem:
    try:
        import diffusers

        ok = hasattr(diffusers, attr_name)
        return PipelineCheckItem(attr_name, ok, message if ok else f"{message} Missing in installed diffusers.")
    except Exception as exc:
        return PipelineCheckItem(attr_name, False, f"{message} Import failed: {exc}")


def _first_existing(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = root / name
        if path.is_file():
            return path
    return None


def _find_krea2_diffusers_folder(models: Path) -> tuple[Path | None, Path | None]:
    root = models / "krea2" / "Diffusers"
    incomplete: Path | None = None
    if not root.exists():
        return None, None
    for index_path in sorted(root.rglob("model_index.json"), key=lambda item: str(item).lower()):
        folder = index_path.parent
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            incomplete = incomplete or folder
            continue
        class_name = str(payload.get("_class_name") or "").lower().replace("_", "")
        folder_text = folder.as_posix().lower().replace("_", "")
        if "krea2pipeline" not in class_name and "krea2" not in folder_text and "krea-2" not in folder_text:
            continue
        if diffusers_dir_has_required_local_files(folder):
            return folder, incomplete
        incomplete = incomplete or folder
    return None, incomplete


def _load_available_onnx_providers(warnings: list[str]) -> list[str]:
    try:
        from aiwf.infrastructure.onnx.session import get_available_providers

        return list(get_available_providers())
    except Exception as exc:
        warnings.append(f"ONNX Runtime provider probe failed: {exc}")
        return []


def _provider_status(preference: str, providers: list[str]) -> tuple[bool, str]:
    provider_set = set(providers)
    if not providers:
        return False, "onnxruntime is not installed or no providers are available."
    if preference == "auto":
        for provider in ("CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"):
            if provider in provider_set:
                return True, f"auto will use {provider}."
        return False, f"No supported provider in: {', '.join(providers)}"
    required = {
        "cuda": "CUDAExecutionProvider",
        "directml": "DmlExecutionProvider",
        "cpu": "CPUExecutionProvider",
    }.get(preference)
    if required is None:
        return False, f"Unknown provider preference: {preference}"
    if required not in provider_set:
        return False, f"{required} is not available. Installed providers: {', '.join(providers)}"
    return True, f"{required} is available."
