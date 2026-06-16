from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aiwf.infrastructure.safetensors_metadata import read_safetensors_metadata
from aiwf.services.model_ops import write_model_op_receipt


def _log(message: str) -> None:
    print(f"[AIWF model-ops] {message}", flush=True)


def _read_safetensors(path: Path) -> dict[str, Any]:
    from safetensors.torch import load_file

    return load_file(str(path), device="cpu")


def _save_safetensors(state: dict[str, Any], path: Path, metadata: dict[str, str]) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(path), metadata=metadata)


def checkpoint_blend(args: argparse.Namespace) -> int:
    left = Path(args.left)
    right = Path(args.right)
    output = Path(args.output)
    ratio = max(0.0, min(1.0, float(args.ratio)))
    _log(f"Loading first checkpoint: {left}")
    left_state = _read_safetensors(left)
    _log(f"Loading second checkpoint: {right}")
    right_state = _read_safetensors(right)

    left_keys = set(left_state)
    right_keys = set(right_state)
    if left_keys != right_keys:
        missing = sorted(left_keys.symmetric_difference(right_keys))[:10]
        raise RuntimeError(f"Checkpoint keys differ; first mismatches: {missing}")

    _log("Blending tensors on CPU")
    merged: dict[str, Any] = {}
    for key, left_tensor in left_state.items():
        right_tensor = right_state[key]
        if tuple(left_tensor.shape) != tuple(right_tensor.shape):
            raise RuntimeError(f"Shape mismatch at {key}: {tuple(left_tensor.shape)} vs {tuple(right_tensor.shape)}")
        if not left_tensor.is_floating_point():
            merged[key] = left_tensor
            continue
        dtype = left_tensor.dtype
        blended = left_tensor.float().mul(1.0 - ratio).add(right_tensor.float(), alpha=ratio)
        merged[key] = blended.to(dtype=dtype)

    metadata = {
        "aiwf.operation": "checkpoint_blend",
        "aiwf.left": str(left),
        "aiwf.right": str(right),
        "aiwf.ratio": str(ratio),
        "aiwf.left_architecture": args.left_arch,
        "aiwf.right_architecture": args.right_arch,
    }
    _log(f"Saving merged checkpoint: {output}")
    _save_safetensors(merged, output, metadata)
    write_model_op_receipt(
        args.receipt,
        {
            "operation": "checkpoint_blend",
            "sources": [str(left), str(right)],
            "output": str(output),
            "ratio": ratio,
            "warnings": ["CPU blend can use significant RAM for large checkpoints."],
        },
    )
    _log(f"Receipt written: {args.receipt}")
    return 0


def lora_fuse(args: argparse.Namespace) -> int:
    base = Path(args.base)
    output = Path(args.output)
    loras = [Path(path) for path in args.lora or []]
    weights = [float(value) for value in args.weight or []]
    if len(loras) != len(weights):
        raise RuntimeError("Each --lora must have a matching --weight.")

    _log("Importing Diffusers lazily")
    from diffusers import DiffusionPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline

    if args.base_arch.startswith("sdxl"):
        pipe_cls = StableDiffusionXLPipeline
    elif args.base_arch in {"sd15", "inpaint"}:
        pipe_cls = StableDiffusionPipeline
    else:
        pipe_cls = DiffusionPipeline

    _log(f"Loading base model: {base}")
    if base.is_dir():
        pipe = pipe_cls.from_pretrained(str(base), torch_dtype="auto", local_files_only=True)
    else:
        pipe = pipe_cls.from_single_file(str(base), torch_dtype="auto", local_files_only=True)

    adapter_names: list[str] = []
    for idx, (lora, weight) in enumerate(zip(loras, weights), start=1):
        name = f"aiwf_lora_{idx}"
        _log(f"Loading LoRA {idx}: {lora} (weight={weight:g})")
        pipe.load_lora_weights(str(lora.parent), weight_name=lora.name, adapter_name=name)
        adapter_names.append(name)
    pipe.set_adapters(adapter_names, adapter_weights=weights)
    _log("Fusing LoRA adapters")
    pipe.fuse_lora(adapter_names=adapter_names, lora_scale=1.0)
    pipe.unload_lora_weights()

    _log(f"Saving fused Diffusers folder: {output}")
    output.mkdir(parents=True, exist_ok=True)
    pipe.save_pretrained(str(output), safe_serialization=True)
    write_model_op_receipt(
        args.receipt,
        {
            "operation": "lora_fuse",
            "base": str(base),
            "loras": [str(path) for path in loras],
            "weights": weights,
            "output": str(output),
            "warnings": ["Output is a Diffusers folder in this first pass."],
        },
    )
    _log(f"Receipt written: {args.receipt}")
    return 0


def convert(args: argparse.Namespace) -> int:
    source = Path(args.source)
    output = Path(args.output)
    _log(f"Preparing conversion: {args.operation}")
    if args.operation == "single-to-diffusers":
        from diffusers import DiffusionPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline

        pipe_cls = StableDiffusionXLPipeline if args.architecture.startswith("sdxl") else StableDiffusionPipeline
        if args.architecture not in {"sd15", "sdxl", "sdxl_inpaint", "inpaint"}:
            pipe_cls = DiffusionPipeline
        _log(f"Loading single-file model: {source}")
        pipe = pipe_cls.from_single_file(str(source), torch_dtype="auto", local_files_only=True)
        _log(f"Saving Diffusers folder: {output}")
        output.mkdir(parents=True, exist_ok=True)
        pipe.save_pretrained(str(output), safe_serialization=True)
        write_model_op_receipt(
            args.receipt,
            {
                "operation": args.operation,
                "source": str(source),
                "output": str(output),
                "architecture": args.architecture,
            },
        )
        return 0
    if args.operation == "onnx-export":
        raise RuntimeError(
            "ONNX export requires Optimum's exporter. Install/enable optimum, then wire this worker to "
            "`optimum-cli export onnx` for the selected model."
        )
    raise RuntimeError(f"Unsupported conversion operation in worker: {args.operation}")


def quantize(args: argparse.Namespace) -> int:
    source = Path(args.source)
    output = Path(args.output)
    quant = str(args.quant).lower()
    if quant in {"fp16", "bf16"}:
        if source.suffix.lower() != ".safetensors":
            raise RuntimeError(f"{quant.upper()} export currently supports safetensors sources only.")
        import torch

        dtype = torch.float16 if quant == "fp16" else torch.bfloat16
        _log(f"Loading safetensors source: {source}")
        state = _read_safetensors(source)
        _log(f"Converting floating tensors to {quant.upper()}")
        converted = {
            key: tensor.to(dtype=dtype) if tensor.is_floating_point() else tensor
            for key, tensor in state.items()
        }
        metadata = read_safetensors_metadata(source)
        metadata.update(
            {
                "aiwf.operation": "quantize_dtype_export",
                "aiwf.source": str(source),
                "aiwf.target": args.target,
                "aiwf.quant": quant,
                "aiwf.architecture": args.architecture,
            }
        )
        _log(f"Saving converted checkpoint: {output}")
        _save_safetensors(converted, output, metadata)
        write_model_op_receipt(
            args.receipt,
            {
                "operation": "quantize_dtype_export",
                "source": str(source),
                "output": str(output),
                "target": args.target,
                "quant": quant,
                "architecture": args.architecture,
                "warnings": ["Only floating tensors were converted; integer and metadata tensors were preserved."],
            },
        )
        _log(f"Receipt written: {args.receipt}")
        return 0
    if quant in {"nvfp4", "fp8", "int8"}:
        _log(f"{quant.upper()} receipt requested for {source}")
        write_model_op_receipt(
            args.receipt,
            {
                "operation": "quantize_receipt_only",
                "source": str(source),
                "proposed_output": str(output),
                "target": args.target,
                "quant": quant,
                "architecture": args.architecture,
                "warnings": [
                    "This receipt records intent only; destructive quantized export is disabled until quality validation lands.",
                    "NVFP4 is storage/compression only on RTX 4070 Ti SUPER, not a speed promise.",
                ],
            },
        )
        _log(f"Receipt written: {args.receipt}")
        return 0
    raise RuntimeError(f"Unsupported quantization target: {args.quant}")


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m aiwf.workers.model_ops")
    sub = parser.add_subparsers(dest="op", required=True)

    blend = sub.add_parser("checkpoint-blend")
    blend.add_argument("--left", required=True)
    blend.add_argument("--right", required=True)
    blend.add_argument("--ratio", required=True)
    blend.add_argument("--output", required=True)
    blend.add_argument("--receipt", required=True)
    blend.add_argument("--left-arch", required=True)
    blend.add_argument("--right-arch", required=True)

    fuse = sub.add_parser("lora-fuse")
    fuse.add_argument("--base", required=True)
    fuse.add_argument("--base-arch", required=True)
    fuse.add_argument("--output", required=True)
    fuse.add_argument("--receipt", required=True)
    fuse.add_argument("--lora", action="append", default=[])
    fuse.add_argument("--weight", action="append", default=[])

    conv = sub.add_parser("convert")
    conv.add_argument("--operation", required=True)
    conv.add_argument("--source", required=True)
    conv.add_argument("--output", required=True)
    conv.add_argument("--architecture", required=True)
    conv.add_argument("--receipt", required=True)

    quant = sub.add_parser("quantize")
    quant.add_argument("--source", required=True)
    quant.add_argument("--output", required=True)
    quant.add_argument("--target", required=True)
    quant.add_argument("--quant", required=True)
    quant.add_argument("--architecture", required=True)
    quant.add_argument("--receipt", required=True)

    args = parser.parse_args()
    try:
        if args.op == "checkpoint-blend":
            return checkpoint_blend(args)
        if args.op == "lora-fuse":
            return lora_fuse(args)
        if args.op == "convert":
            return convert(args)
        if args.op == "quantize":
            return quantize(args)
    except Exception as exc:
        _log(f"ERROR: {exc}")
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
