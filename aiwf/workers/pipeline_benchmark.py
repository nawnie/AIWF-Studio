from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image

from aiwf import __version__
from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.optimization import (
    BenchmarkMemory,
    BenchmarkOutputs,
    BenchmarkReceipt,
    BenchmarkTiming,
    GpuCapability,
    OptimizationRequest,
    PipelineKind,
)
from aiwf.services.optimization import CapabilityDetector, OptimizationPlanner


_PACKAGES = (
    "torch",
    "torchvision",
    "diffusers",
    "transformers",
    "accelerate",
    "safetensors",
    "sageattention",
    "triton-windows",
    "gguf",
    "kernels",
    "torchao",
    "xformers",
    "flash-attn",
)

_FLAG_ENV = (
    "AIWF_CHANNELS_LAST",
    "AIWF_TORCH_COMPILE",
    "AIWF_TORCHAO",
    "AIWF_FP8",
    "AIWF_WAN_SAGE_ATTENTION",
    "AIWF_USE_SAGE_ATTENTION",
    "AIWF_WAN_GGUF_RUNTIME",
    "AIWF_WAN_MANUAL_VAE_DECODE",
    "AIWF_WAN_VAE_CHUNK_FRAMES",
    "DIFFUSERS_GGUF_CUDA_KERNELS",
)


def _log(message: str) -> None:
    print(f"[AIWF benchmark] {message}", flush=True)


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in _PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "MISSING"
    return versions


def _runtime_info() -> dict[str, Any]:
    return {
        "app_version": __version__,
        "python": platform.python_version(),
        "python_executable": os.sys.executable,
        "platform": platform.platform(),
        "packages": _package_versions(),
        "env_flags": {name: os.environ.get(name, "") for name in _FLAG_ENV},
    }


def _pipeline_kind_from_config(config: dict[str, Any]) -> PipelineKind:
    kind = str(config.get("kind") or "").strip().lower()
    if kind == "img2img":
        return PipelineKind.IMG2IMG
    if kind == "wan_i2v":
        return PipelineKind.VIDEO
    return PipelineKind.TXT2IMG


def _optimization_sections(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("optimization") if isinstance(config.get("optimization"), dict) else {}
    request_data = config.get("request") if isinstance(config.get("request"), dict) else {}
    profile_id = str(
        raw.get("profile_id")
        or config.get("optimization_profile_id")
        or "balanced_sdpa_fp16"
    )
    opt_request = OptimizationRequest(
        profile_id=profile_id,
        pipeline_kind=_pipeline_kind_from_config(config),
        width=request_data.get("width"),
        height=request_data.get("height"),
        batch_size=int(request_data.get("batch_size") or 1),
        experimental_flags=dict(raw.get("experimental_flags") or {}),
        fast_method=raw.get("fast_method"),
    )
    capability_report = CapabilityDetector().detect(include_gpu=True)
    plan = OptimizationPlanner().resolve(opt_request, capabilities=capability_report)
    return {
        "capability_report": capability_report.model_dump(mode="json"),
        "optimization_profile": plan.effective_profile.model_dump(mode="json"),
        "optimization_decisions": [decision.model_dump(mode="json") for decision in plan.decisions],
        "optimization_blocked": plan.blocked,
        "optimization_fallback_profile_id": plan.fallback_profile_id,
    }


def _numeric_timing(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    if parsed != parsed or parsed < 0:
        return None
    return parsed


def _sum_timing(*values: Any) -> float | None:
    total = 0.0
    found = False
    for value in values:
        parsed = _numeric_timing(value)
        if parsed is None:
            continue
        total += parsed
        found = True
    return round(total, 3) if found else None


def _typed_receipt(
    *,
    benchmark_id: str,
    created_at: str,
    runtime: dict[str, Any],
    config: dict[str, Any],
    optimization: dict[str, Any],
    status: str,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    result = result or {}
    config_request = config.get("request") if isinstance(config.get("request"), dict) else {}
    result_request = result.get("request") if isinstance(result.get("request"), dict) else {}
    request = {**config_request, **result_request}
    gpu_data = optimization.get("capability_report", {}).get("gpu", {})
    timing = BenchmarkTiming(
        load_time_s=result.get("load_seconds"),
        first_generation_time_s=result.get("elapsed_seconds"),
        steady_state_times_s=(
            [float(result["elapsed_seconds"])]
            if isinstance(result.get("elapsed_seconds"), (int, float))
            else []
        ),
        median_time_s=result.get("elapsed_seconds"),
        denoise_time_s=result.get("denoise_seconds"),
        prompt_encode_time_s=result.get("prompt_encode_seconds"),
        preprocess_time_s=_sum_timing(
            result.get("preprocess_seconds"),
            result.get("image_encode_seconds"),
            result.get("latent_prepare_seconds"),
        ),
        vae_decode_time_s=result.get("vae_decode_seconds"),
        postprocess_time_s=_sum_timing(
            result.get("video_postprocess_seconds"),
            result.get("offload_cleanup_seconds"),
            result.get("postprocess_seconds"),
            result.get("video_write_seconds"),
        ),
    )
    outputs = BenchmarkOutputs(
        image_paths=[str(result["output_path"])] if result.get("output_path") else [],
    )
    receipt = BenchmarkReceipt(
        receipt_id=benchmark_id,
        created_at=created_at,
        aiwf={
            "app_version": str(runtime.get("app_version") or __version__),
            "profile_registry_version": str(
                optimization.get("optimization_profile", {}).get("profile_version") or ""
            ),
        },
        system={
            "python": runtime.get("python"),
            "platform": runtime.get("platform"),
            "python_executable": runtime.get("python_executable"),
            "capability_report_id": optimization.get("capability_report", {}).get("report_id"),
        },
        dependencies=dict(runtime.get("packages") or {}),
        gpu=GpuCapability.model_validate(gpu_data or {}),
        model=dict(config.get("model") or {}),
        pipeline={
            "kind": result.get("kind") or config.get("kind"),
            "scheduler_class": request.get("scheduler"),
            "sampler": request.get("sampler"),
        },
        optimization_profile=dict(optimization.get("optimization_profile") or {}),
        generation={
            "seed": request.get("seed"),
            "width": request.get("width"),
            "height": request.get("height"),
            "steps": request.get("steps"),
            "batch_size": request.get("batch_size"),
            "config": config,
        },
        timing=timing,
        memory=BenchmarkMemory(oom=status == "failed" and "out of memory" in (error or "").lower()),
        outputs=outputs,
        status=status,
    )
    payload = receipt.model_dump(mode="json")
    if error:
        payload["error"] = error
    if result:
        payload["result"] = result
    return payload


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _receipt_path(out_dir: Path, benchmark_id: str) -> Path:
    return out_dir / f"{benchmark_id}.json"


def _write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")


def _install_benchmark_diagnostics(out_dir: Path) -> str | None:
    try:
        from aiwf.dev.diagnostics import install_standalone_dev_diagnostics

        diag = install_standalone_dev_diagnostics(out_dir)
        return str(diag.log_path)
    except Exception as exc:
        _log(f"Diagnostics disabled: {exc}")
        return None


def _flags_from_config(config: dict[str, Any]) -> RuntimeFlags:
    raw = dict(config.get("flags") or {})
    if config.get("data_dir") and "data_dir" not in raw:
        raw["data_dir"] = config["data_dir"]
    flags = RuntimeFlags.model_validate(raw)
    from aiwf.runtime.bootstrap_env import apply_runtime_env

    apply_runtime_env(cuda_malloc=bool(getattr(flags, "cuda_malloc", True)))
    return flags


def _load_image(config: dict[str, Any], *keys: str) -> Image.Image:
    for key in keys:
        value = config.get(key)
        if value:
            return Image.open(value).convert("RGB")
    raise ValueError(f"Missing image path; expected one of: {', '.join(keys)}")


def run_img2img_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    from aiwf.core.domain.generation import GenerationRequest
    from aiwf.infrastructure.diffusers.backend import DiffusersBackend
    from aiwf.infrastructure.torch.devices import DeviceManager

    flags = _flags_from_config(config)
    devices = DeviceManager(flags)
    backend = DiffusersBackend(flags, devices)
    request_data = dict(config.get("request") or {})
    request_data.setdefault("mode", "img2img")
    request_data.setdefault("save_images", False)
    request = GenerationRequest.model_validate(request_data)
    image = _load_image(config, "init_image", "image")

    started = time.perf_counter()
    result = backend.generate(request, init_images=[image], preview_every_n_steps=0)
    elapsed = time.perf_counter() - started
    steps = max(1, int(request.steps))
    return {
        "kind": "img2img",
        "elapsed_seconds": elapsed,
        "units": steps,
        "units_label": "steps",
        "steps_per_second": steps / elapsed if elapsed > 0 else None,
        "iterations_per_second": steps / elapsed if elapsed > 0 else None,
        "image_count": len(result.images),
        "request": request.model_dump(mode="json"),
    }


def run_wan_i2v_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    from aiwf.core.domain.wan import WanI2VRequest
    from aiwf.services.wan import WanService

    flags = _flags_from_config(config)
    settings = UserSettings()
    service = WanService(flags, settings)
    request = WanI2VRequest.model_validate(dict(config.get("request") or {}))
    image = _load_image(config, "init_image", "image")

    started = time.perf_counter()
    result = service.generate(request, image)
    elapsed = time.perf_counter() - started
    frames = max(1, int(result.frame_count))
    return {
        "kind": "wan_i2v",
        "elapsed_seconds": elapsed,
        "units": frames,
        "units_label": "frames",
        "frames_per_second": frames / elapsed if elapsed > 0 else None,
        "step_count": int(result.step_count),
        "load_seconds": result.load_seconds,
        "preprocess_seconds": result.preprocess_seconds,
        "prompt_encode_seconds": result.prompt_encode_seconds,
        "image_encode_seconds": result.image_encode_seconds,
        "latent_prepare_seconds": result.latent_prepare_seconds,
        "denoise_seconds": result.denoise_seconds,
        "high_denoise_seconds": result.high_denoise_seconds,
        "low_denoise_seconds": result.low_denoise_seconds,
        "pipeline_seconds": result.pipeline_seconds,
        "pipeline_overhead_seconds": result.pipeline_overhead_seconds,
        "vae_decode_seconds": result.vae_decode_seconds,
        "manual_vae_decode": result.manual_vae_decode,
        "vae_decode_chunk_frames": result.vae_decode_chunk_frames,
        "video_postprocess_seconds": result.video_postprocess_seconds,
        "offload_cleanup_seconds": result.offload_cleanup_seconds,
        "postprocess_seconds": result.postprocess_seconds,
        "video_write_seconds": result.video_write_seconds,
        "steps_per_second": result.steps_per_second,
        "iterations_per_second": result.iterations_per_second,
        "fp8_linear_layers": result.fp8_linear_layers,
        "fp8_fast_mm_calls": result.fp8_fast_mm_calls,
        "fp8_fallback_calls": result.fp8_fallback_calls,
        "fp8_fallback_layers": result.fp8_fallback_layers,
        "fp8_fallback_reasons": result.fp8_fallback_reasons,
        "fp8_strict_mode": result.fp8_strict_mode,
        "fp8_native_available": result.fp8_native_available,
        "cache_mode": result.cache_mode,
        "output_path": result.output_path,
        "request": request.model_dump(mode="json"),
        "capabilities": service.acceleration_capabilities(),
    }


def run_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    kind = str(config.get("kind") or "").strip().lower()
    if kind == "probe":
        return run_probe(config)
    if kind == "img2img":
        return run_img2img_benchmark(config)
    if kind == "wan_i2v":
        return run_wan_i2v_benchmark(config)
    raise ValueError("Benchmark kind must be 'probe', 'img2img', or 'wan_i2v'.")


def run_probe(config: dict[str, Any] | None = None) -> dict[str, Any]:
    from aiwf.infrastructure.torch.wan_perf import describe_wan_acceleration_capabilities

    config = config or {}
    return {
        "kind": "probe",
        "label": str(config.get("label") or ""),
        "wan_capabilities": describe_wan_acceleration_capabilities(),
    }


def run_with_receipt(config: dict[str, Any], out_dir: Path) -> tuple[int, Path]:
    benchmark_id = f"pipeline-benchmark-{_timestamp()}-{uuid4().hex[:8]}"
    path = _receipt_path(out_dir, benchmark_id)
    created_at = datetime.now(timezone.utc).isoformat()
    runtime = _runtime_info()
    optimization = _optimization_sections(config)
    diagnostics_log = _install_benchmark_diagnostics(out_dir)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_id": benchmark_id,
        "created_at": created_at,
        "status": "running",
        "runtime": runtime,
        "config": config,
        "diagnostics_log": diagnostics_log,
        "typed_receipt": _typed_receipt(
            benchmark_id=benchmark_id,
            created_at=created_at,
            runtime=runtime,
            config=config,
            optimization=optimization,
            status="running",
        ),
    }
    receipt.update(optimization)
    _write_receipt(path, receipt)
    try:
        result = run_benchmark(config)
    except Exception as exc:
        receipt["status"] = "failed"
        receipt["error"] = str(exc)
        receipt["typed_receipt"] = _typed_receipt(
            benchmark_id=benchmark_id,
            created_at=created_at,
            runtime=runtime,
            config=config,
            optimization=optimization,
            status="failed",
            error=str(exc),
        )
        _write_receipt(path, receipt)
        _log(f"FAILED: {exc}")
        return 1, path
    receipt["status"] = "completed"
    receipt["result"] = result
    receipt["typed_receipt"] = _typed_receipt(
        benchmark_id=benchmark_id,
        created_at=created_at,
        runtime=runtime,
        config=config,
        optimization=optimization,
        status="completed",
        result=result,
    )
    _write_receipt(path, receipt)
    _log(f"Receipt written: {path}")
    return 0, path


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m aiwf.workers.pipeline_benchmark")
    parser.add_argument("--config", help="Path to benchmark JSON config.")
    parser.add_argument("--probe", action="store_true", help="Write a no-model accelerator capability receipt.")
    parser.add_argument("--label", default="", help="Optional label for --probe receipts.")
    parser.add_argument("--out", default="outputs/benchmarks", help="Directory for receipt JSON.")
    args = parser.parse_args()
    if args.probe:
        config = {"kind": "probe", "label": args.label}
    elif args.config:
        config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    else:
        parser.error("Either --config or --probe is required.")
    rc, _path = run_with_receipt(config, Path(args.out))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
