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


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _receipt_path(out_dir: Path, benchmark_id: str) -> Path:
    return out_dir / f"{benchmark_id}.json"


def _write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")


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
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_id": benchmark_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "runtime": _runtime_info(),
        "config": config,
    }
    _write_receipt(path, receipt)
    try:
        result = run_benchmark(config)
    except Exception as exc:
        receipt["status"] = "failed"
        receipt["error"] = str(exc)
        _write_receipt(path, receipt)
        _log(f"FAILED: {exc}")
        return 1, path
    receipt["status"] = "completed"
    receipt["result"] = result
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
