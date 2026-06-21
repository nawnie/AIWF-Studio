from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from aiwf.core.config.settings import RuntimeFlags
from aiwf.core.domain.models import Checkpoint, LoraInfo
from aiwf.core.domain.worker import WorkerCommand
from aiwf.infrastructure.safetensors_metadata import read_safetensors_metadata


MODEL_OP_EXTENSIONS = {".safetensors", ".ckpt", ".pt", ".bin", ".pth", ".gguf", ".onnx"}
IMAGE_MODEL_ARCHES = {"sd15", "sdxl", "sdxl_inpaint", "inpaint"}
TRANSFORMER_MODEL_ARCHES = {"wan", "flux", "flux2_klein", "z_image"}
JOB_OUTPUT_DIRNAME = "model-ops"
DTYPE_EXPORT_QUANTS = {"fp16", "bf16"}
FP8_READY_EXPORT_QUANTS = {"aiwf_fp8_ready"}
RECEIPT_ONLY_QUANTS = {"fp8", "int8", "nvfp4"}


@dataclass(frozen=True)
class ModelAsset:
    path: Path
    family: str
    storage: str
    architecture: str = "unknown"
    dtype_hint: str = "unknown"
    quant_hint: str = "none"
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def exists(self) -> bool:
        return self.path.exists()


@dataclass(frozen=True)
class PreflightResult:
    ok: bool
    title: str
    messages: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    command: WorkerCommand | None = None
    receipt_path: Path | None = None

    def markdown(self) -> str:
        mark = "OK" if self.ok else "Blocked"
        lines = [f"**{mark}: {self.title}**"]
        lines.extend(f"- {message}" for message in self.messages)
        if self.warnings:
            lines.append("\n**Warnings**")
            lines.extend(f"- {warning}" for warning in self.warnings)
        if self.receipt_path is not None:
            lines.append(f"\n**Receipt:** `{self.receipt_path}`")
        if self.command is not None:
            lines.append(f"\n**Worker:** `{self.command.name}`")
        return "\n".join(lines)


def sanitize_output_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "aiwf_model_output"


def inspect_model_asset(path: str | Path, *, architecture: str | None = None) -> ModelAsset:
    resolved = Path(path).expanduser().resolve()
    suffix = resolved.suffix.lower()
    metadata = read_safetensors_metadata(resolved)
    storage = "unknown"
    family = "unknown"
    dtype_hint = "unknown"
    quant_hint = "none"

    if resolved.is_dir():
        if (resolved / "model_index.json").is_file():
            storage = "diffusers"
            family = "image"
        elif any((resolved / sub / "model.onnx").is_file() for sub in ("unet", "text_encoder", "vae_decoder")):
            storage = "onnx-folder"
            family = "image"
        else:
            storage = "folder"
    elif suffix == ".safetensors":
        storage = "safetensors"
    elif suffix in {".ckpt", ".pt", ".pth", ".bin"}:
        storage = suffix.lstrip(".")
    elif suffix == ".gguf":
        storage = "gguf"
        family = "llm-or-quantized"
    elif suffix == ".onnx":
        storage = "onnx"

    lower_name = resolved.name.lower()
    meta_arch = (
        metadata.get("modelspec.architecture")
        or metadata.get("ss_base_model_version")
        or metadata.get("ss_sd_model_name")
        or ""
    ).lower()
    combined = f"{lower_name} {meta_arch}"

    detected_arch = (architecture or "").strip().lower() or "unknown"
    if detected_arch == "unknown":
        compact = combined.replace("_", "").replace("-", "").replace(" ", "")
        if "z-image" in combined or "zimage" in compact:
            detected_arch = "z_image"
        elif "flux.2" in combined or "flux2" in compact or "klein" in combined:
            detected_arch = "flux2_klein"
        elif "sdxl" in combined or "xl" in combined:
            detected_arch = "sdxl"
        elif "sd1" in combined or "sd 1" in combined or "v1-5" in combined or "1.5" in combined:
            detected_arch = "sd15"
        elif "wan" in combined:
            detected_arch = "wan"
        elif "flux" in combined:
            detected_arch = "flux"

    if family in {"unknown", "llm-or-quantized"} and detected_arch in TRANSFORMER_MODEL_ARCHES:
        family = "video-or-transformer"
    elif family == "unknown":
        if "lora" in combined or metadata.get("ss_network_module"):
            family = "lora"
        elif "vae" in combined:
            family = "vae"
        elif storage != "unknown":
            family = "image"

    for key, value in metadata.items():
        text = f"{key} {value}".lower()
        if "float8" in text or "fp8" in text:
            quant_hint = "fp8"
        elif "fp4" in text or "nvfp4" in text:
            quant_hint = "nvfp4"
        if "bf16" in text or "bfloat16" in text:
            dtype_hint = "bf16"
        elif "fp16" in text or "float16" in text:
            dtype_hint = "fp16"
        elif "fp32" in text or "float32" in text:
            dtype_hint = "fp32"

    return ModelAsset(
        path=resolved,
        family=family,
        storage=storage,
        architecture=detected_arch,
        dtype_hint=dtype_hint,
        quant_hint=quant_hint,
        metadata=metadata,
    )


def write_model_op_receipt(path: str | Path, payload: dict[str, Any]) -> Path:
    receipt = Path(path).expanduser().resolve()
    receipt.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "app": "AIWF Studio",
        "created_at": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    receipt.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return receipt


class ModelOpsService:
    """Preflight and command builder for model mixing, conversion, and quant jobs.

    This layer is intentionally conservative: unsupported or quality-sensitive
    paths become receipts/preflight messages instead of silent destructive
    exports. Several paths work as maintenance workflows but are not optimized
    runtime accelerators until receipts and benchmark evidence prove that.
    """

    def __init__(self, flags: RuntimeFlags) -> None:
        self.flags = flags
        self.models_dir = flags.resolved_models_dir()
        self.output_dir = flags.resolved_output_dir() / JOB_OUTPUT_DIRNAME

    def default_output_path(self, output_name: str, suffix: str = ".safetensors") -> Path:
        return self.output_dir / f"{sanitize_output_name(output_name)}{suffix}"

    def preflight_checkpoint_blend(
        self,
        left: Checkpoint | None,
        right: Checkpoint | None,
        *,
        ratio: float,
        output_name: str,
    ) -> PreflightResult:
        errors: list[str] = []
        warnings: list[str] = []
        if left is None or right is None:
            errors.append("Select two checkpoints.")
        if left is not None and right is not None:
            if left.id == right.id:
                errors.append("Choose two different checkpoints.")
            if left.architecture != right.architecture:
                errors.append(f"Architecture mismatch: {left.architecture} vs {right.architecture}.")
            if left.architecture not in IMAGE_MODEL_ARCHES:
                errors.append(f"Unsupported checkpoint architecture: {left.architecture}.")
            for ckpt in (left, right):
                if Path(ckpt.path).suffix.lower() != ".safetensors":
                    errors.append(f"{ckpt.filename} is not safetensors; first pass only blends safetensors.")
        cleaned = sanitize_output_name(output_name)
        output = self.default_output_path(cleaned)
        if left is not None and output.resolve() == Path(left.path).resolve():
            errors.append("Output path matches the first source checkpoint.")
        if right is not None and output.resolve() == Path(right.path).resolve():
            errors.append("Output path matches the second source checkpoint.")
        ratio = max(0.0, min(1.0, float(ratio)))
        warnings.append("Large checkpoint blends can use significant RAM; this runs in a worker process.")
        if errors:
            return PreflightResult(False, "Checkpoint blend is not ready", tuple(errors), tuple(warnings))

        receipt = output.with_suffix(output.suffix + ".receipt.json")
        command = self._worker_command(
            "checkpoint-blend",
            [
                "--left", str(left.path),
                "--right", str(right.path),
                "--ratio", str(ratio),
                "--output", str(output),
                "--receipt", str(receipt),
                "--left-arch", left.architecture,
                "--right-arch", right.architecture,
            ],
        )
        messages = (
            f"Blend {left.filename} and {right.filename}.",
            f"Ratio: {ratio:.2f} keeps {(1-ratio):.0%} first / {ratio:.0%} second.",
            f"Output: `{output}`",
        )
        return PreflightResult(True, "Checkpoint blend ready", messages, tuple(warnings), command, receipt)

    def preflight_lora_fuse(
        self,
        base: Checkpoint | None,
        loras: Iterable[LoraInfo],
        *,
        weights: str,
        output_name: str,
    ) -> PreflightResult:
        selected = list(loras)
        errors: list[str] = []
        warnings: list[str] = []
        if base is None:
            errors.append("Select a base checkpoint.")
        elif base.architecture not in IMAGE_MODEL_ARCHES:
            errors.append(f"Unsupported base architecture: {base.architecture}.")
        if not selected:
            errors.append("Select at least one LoRA.")
        parsed_weights = _parse_weight_list(weights, len(selected))
        if parsed_weights is None:
            errors.append("Weights must be numbers matching the selected LoRAs, or a single number.")
            parsed_weights = [1.0 for _ in selected]
        output = self.default_output_path(output_name, suffix="")
        if output.suffix:
            output = output.with_suffix("")
        warnings.append("LoRA fusion uses Diffusers/PEFT in a worker; dependencies are optional and checked at runtime.")
        warnings.append("The first pass saves a Diffusers folder, not a single-file checkpoint.")
        if errors:
            return PreflightResult(False, "LoRA fuse is not ready", tuple(errors), tuple(warnings))

        receipt = output / "aiwf_model_op_receipt.json"
        args = [
            "--base", str(base.path),
            "--base-arch", base.architecture,
            "--output", str(output),
            "--receipt", str(receipt),
        ]
        for lora, weight in zip(selected, parsed_weights):
            args.extend(["--lora", str(lora.path), "--weight", str(weight)])
        command = self._worker_command("lora-fuse", args)
        messages = (
            f"Base: {base.filename}",
            f"LoRAs: {', '.join(lora.filename for lora in selected)}",
            f"Output folder: `{output}`",
        )
        return PreflightResult(True, "LoRA fuse ready", messages, tuple(warnings), command, receipt)

    def preflight_conversion(
        self,
        *,
        source_path: str,
        operation: str,
        output_name: str,
        architecture: str,
    ) -> PreflightResult:
        asset = inspect_model_asset(source_path, architecture=architecture)
        errors: list[str] = []
        warnings = _platform_notes()
        if not asset.exists:
            errors.append(f"Source does not exist: {asset.path}")
        if asset.storage == "unknown":
            errors.append(f"Unsupported source format: {asset.path}")

        output_suffix = "" if operation in {"single-to-diffusers", "onnx-export"} else ".safetensors"
        output = self.default_output_path(output_name, suffix=output_suffix)
        if operation == "single-to-diffusers":
            if asset.storage not in {"safetensors", "ckpt"}:
                errors.append("Single-file to Diffusers requires a safetensors or ckpt source.")
            output = output.with_suffix("") if output.suffix else output
            messages = (f"Export Diffusers folder to `{output}`.",)
        elif operation == "onnx-export":
            if asset.storage != "diffusers":
                errors.append("ONNX export currently requires a Diffusers model folder.")
            errors.append("ONNX export is blocked until a stable local Optimum exporter path is wired.")
            output = output.with_suffix("") if output.suffix else output
            messages = (f"Target ONNX folder would be `{output}`.",)
        elif operation == "diffusers-to-single":
            if asset.storage != "diffusers":
                errors.append("Diffusers folder to single-file requires a Diffusers model folder.")
            errors.append("Diffusers-folder to single-file export is preflight-only until a stable local converter is selected.")
            messages = ("This operation is documented but blocked in this first pass.",)
        else:
            errors.append(f"Unknown conversion operation: {operation}")
            messages = ()

        if errors:
            return PreflightResult(False, "Conversion blocked", tuple(errors), tuple(warnings))

        receipt = (output / "aiwf_model_op_receipt.json") if output_suffix == "" else output.with_suffix(output.suffix + ".receipt.json")
        command = self._worker_command(
            "convert",
            [
                "--operation", operation,
                "--source", str(asset.path),
                "--output", str(output),
                "--architecture", asset.architecture,
                "--receipt", str(receipt),
            ],
        )
        return PreflightResult(True, "Conversion ready", messages, tuple(warnings), command, receipt)

    def preflight_quantization(
        self,
        *,
        source_path: str,
        target: str,
        quant: str,
        output_name: str,
        architecture: str,
    ) -> PreflightResult:
        asset = inspect_model_asset(source_path, architecture=architecture)
        quant = (quant or "").strip().lower()
        target = (target or "").strip().lower()
        errors: list[str] = []
        warnings = _platform_notes()
        messages: list[str] = []
        if not asset.exists:
            errors.append(f"Source does not exist: {asset.path}")
        if quant == "nvfp4":
            warnings.append(
                "NVFP4 is treated as compression/storage here. RTX 4070 Ti SUPER is Ada, not Blackwell, so do not expect native FP4 speedups."
            )
        if quant == "fp8":
            warnings.append(
                "Plain FP8 export is storage-oriented unless the runtime also keeps scales and executes FP8 matmuls."
            )
        if quant == "aiwf_fp8_ready":
            warnings.append(
                "AIWF FP8-ready export writes scaled FP8 linear weights plus sidecar scales for AIWF native FP8 runtimes."
            )
        if target == "vae":
            errors.append("VAE quantization is preflight-only until decode-quality validation is implemented.")
        if quant in DTYPE_EXPORT_QUANTS | FP8_READY_EXPORT_QUANTS and asset.storage != "safetensors":
            errors.append(f"{quant.upper()} export currently supports single-file safetensors only.")
        elif quant in RECEIPT_ONLY_QUANTS and asset.storage not in {"safetensors", "diffusers"}:
            errors.append(f"{quant.upper()} receipt jobs expect safetensors or a Diffusers folder.")
        if quant == "gguf":
            warnings.append("GGUF conversion belongs to the future llama.cpp/chat lane and is not executed from image quantization yet.")
            errors.append("GGUF image-model quantization is not enabled in this pass.")
        elif quant not in DTYPE_EXPORT_QUANTS | FP8_READY_EXPORT_QUANTS | RECEIPT_ONLY_QUANTS:
            errors.append(f"Unknown quantization choice: {quant or 'none'}")

        purpose = {
            "fp16": "compatibility and broad runtime support",
            "bf16": "NVIDIA Ada reliability and reduced overflow risk",
            "fp8": "storage-size experiment; runtime speed still depends on FP8 kernels",
            "aiwf_fp8_ready": "scaled-FP8 runtime package for Wan and other transformer-heavy models",
            "nvfp4": "storage compression only on this machine",
            "int8": "VRAM/file-size experiment through optional torchao",
        }.get(quant, "unknown")
        messages.append(f"Purpose: {purpose}.")

        if errors:
            return PreflightResult(False, "Quantization blocked", tuple(errors + messages), tuple(warnings))

        output = self.default_output_path(output_name)
        receipt = output.with_suffix(output.suffix + ".receipt.json")
        if quant in DTYPE_EXPORT_QUANTS:
            messages.append("Worker will write a converted safetensors copy; non-floating tensors are preserved.")
            title = "Quantization job ready"
        elif quant in FP8_READY_EXPORT_QUANTS:
            messages.append(
                "Worker will write an AIWF FP8-ready safetensors package; 2D linear `.weight` tensors become FP8 "
                "and matching `.weight_scale` sidecars are written."
            )
            messages.append(
                "Non-linear tensors, biases, embeddings, norms, and integer tensors are preserved to limit quality risk."
            )
            messages.append(
                "Runtime speed still requires the AIWF native FP8 loader and strict no-fallback execution."
            )
            title = "AIWF FP8-ready export job ready"
        else:
            messages.append(
                "Worker will write a receipt only; model export is held until quality and runtime validation land."
            )
            title = "Quantization receipt job ready"
        messages.append(f"Output: `{output}`")
        command = self._worker_command(
            "quantize",
            [
                "--source", str(asset.path),
                "--output", str(output),
                "--target", target,
                "--quant", quant,
                "--architecture", asset.architecture,
                "--receipt", str(receipt),
            ],
        )
        return PreflightResult(True, title, tuple(messages), tuple(warnings), command, receipt)

    def _worker_command(self, op: str, args: list[str]) -> WorkerCommand:
        name = f"model-ops-{op}"
        return WorkerCommand(
            args=[sys.executable, "-m", "aiwf.workers.model_ops", op, *args],
            cwd=self.flags.data_dir,
            env={},
            name=name,
        )


def _parse_weight_list(raw: str, expected: int) -> list[float] | None:
    if expected <= 0:
        return []
    text = (raw or "").strip()
    if not text:
        return [1.0 for _ in range(expected)]
    try:
        values = [float(part.strip()) for part in text.replace(";", ",").split(",") if part.strip()]
    except ValueError:
        return None
    if len(values) == 1:
        return values * expected
    if len(values) != expected:
        return None
    return values


def _platform_notes() -> list[str]:
    # TODO: add AMD ROCm/DirectML, Intel XPU/OpenVINO, and CPU quant backends
    # after the NVIDIA-first path has preflight, receipts, and benchmarks.
    return [
        "NVIDIA path is prioritized for this pass. TODO: add AMD, Intel, and CPU quant backends later.",
        "Optional engines and converters must never become mandatory boot dependencies.",
    ]
