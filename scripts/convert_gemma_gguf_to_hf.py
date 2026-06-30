"""
Convert an unsloth-style llama.cpp GGUF Gemma-3-12B-it checkpoint (e.g.
`gemma-3-12b-it-heretic-Q3_K_M.gguf`, 5.7GB) into the HF-style layout that
AIWF's LTX worker / ltx-core's Gemma loader expects.

Why this exists
----------------
ltx-core's Gemma loader globs `gemma_root` for `*.safetensors` and runs every
tensor through GEMMA_LLM_KEY_OPS, which expects HF Gemma3ForConditionalGeneration
key names (language_model.model.layers.<N>..., vision_tower..., etc). GGUF
files use llama.cpp's own naming + block-quantized tensor storage, neither of
which match.

Key mapping (verified against this file's actual tensor names -- see
inspection done in chat, matches llama.cpp's convert_hf_to_gguf.py Gemma3
converter):
    token_embd.weight              -> language_model.model.embed_tokens.weight
    output_norm.weight             -> language_model.model.norm.weight
    blk.N.attn_norm.weight         -> language_model.model.layers.N.input_layernorm.weight
    blk.N.attn_q/k/v/output.weight -> language_model.model.layers.N.self_attn.{q,k,v,o}_proj.weight
    blk.N.attn_q_norm/k_norm       -> language_model.model.layers.N.self_attn.{q,k}_norm.weight
    blk.N.ffn_gate/up/down.weight  -> language_model.model.layers.N.mlp.{gate,up,down}_proj.weight
    blk.N.ffn_norm.weight          -> language_model.model.layers.N.pre_feedforward_layernorm.weight
    blk.N.post_attention_norm      -> language_model.model.layers.N.post_attention_layernorm.weight
    blk.N.post_ffw_norm            -> language_model.model.layers.N.post_feedforward_layernorm.weight

IMPORTANT -- this GGUF is TEXT-ONLY (626 tensors, no vision_model.* /
multi_modal_projector.* tensors at all -- the vision tower ships separately
as a clip.cpp-format mmproj-*.gguf for these unsloth/heretic repos, which is
NOT converted by this script). By default this script copies the official
Gemma vision_tower/multi_modal_projector tensors into a sidecar safetensors
file so the upstream LTX builder can initialize without meta tensors. That
sidecar is a loader compatibility fix, not a quantized Heretic vision model.
Plain text-to-video has been smoke-tested; image-conditioned prompt enhancement
remains unverified.

Dequantization uses the `gguf` pip package's own dequantize() (handles
Q2_K..Q8_0, IQ*, K-quants -- the same code llama.cpp's Python tooling uses),
not a hand-rolled reimplementation, since K-quant superblock math is easy to
get subtly wrong by hand. bf16 packing is done via numpy bit manipulation
(round-to-nearest-even) since this script has no torch dependency and can run
in any plain Python + numpy environment.

Same caveat as the fp8/nvfp4 paths: once dequantized, the output is full
bf16 size (~24GB) when loaded. Only the *download* was smaller.

Usage:
    python convert_gemma_gguf_to_hf.py \
        --src "/path/to/gemma-3-12b-it-heretic-Q3_K_M.gguf" \
        --dst "/path/to/gemma-3-12b-q3km-converted" \
        --copy-processor-from "/path/to/old/gemma-3-12b-it-qat-q4_0-unquantized"

Requires: pip install gguf numpy
"""
import argparse
import json
import shutil
import struct
import sys
import time
from pathlib import Path

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

PROCESSOR_FILES = (
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.json",
    "added_tokens.json",
    "tokenizer.json",
    "tokenizer.model",
)

DEFAULT_SRC = Path("models/LLM/GGUF/gemma-3-12b-it-heretic-Q3_K_M.gguf")
DEFAULT_DST = Path("models/ltx/text_encoder/gemma-3-12b-heretic-q3km-converted")
DEFAULT_COPY_PROCESSOR_FROM = Path("models/ltx/text_encoder/gemma-3-12b-it-qat-q4_0-unquantized")
DEFAULT_RECEIPT = Path("_local/logs/ltx_heretic_q3_gguf_conversion_plan_latest.json")
VISION_SIDECAR = "vision_projector.safetensors"
VISION_PREFIXES = ("vision_tower.", "multi_modal_projector.")


def f32_to_bf16_bytes(arr: np.ndarray) -> bytes:
    """Round float32 -> bfloat16 (round-to-nearest-even) and return raw bytes."""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    bits = arr.view(np.uint32)
    rounding_bias = ((bits >> 16) & 1) + 0x7FFF
    rounded = (bits.astype(np.uint64) + rounding_bias) >> 16
    return rounded.astype(np.uint16).tobytes()


def remap_key(name: str) -> str | None:
    if name == "output_norm.weight":
        return "language_model.model.norm.weight"
    if name == "token_embd.weight":
        return "language_model.model.embed_tokens.weight"
    if name.startswith("blk."):
        parts = name.split(".", 2)  # ["blk", "<N>", "<rest>"]
        n = parts[1]
        rest = parts[2]
        mapping = {
            "attn_norm.weight": "self_attn_input_layernorm",  # placeholder, fixed below
        }
        suffix_map = {
            "attn_norm.weight": "input_layernorm.weight",
            "attn_q.weight": "self_attn.q_proj.weight",
            "attn_k.weight": "self_attn.k_proj.weight",
            "attn_v.weight": "self_attn.v_proj.weight",
            "attn_output.weight": "self_attn.o_proj.weight",
            "attn_q_norm.weight": "self_attn.q_norm.weight",
            "attn_k_norm.weight": "self_attn.k_norm.weight",
            "ffn_gate.weight": "mlp.gate_proj.weight",
            "ffn_up.weight": "mlp.up_proj.weight",
            "ffn_down.weight": "mlp.down_proj.weight",
            "ffn_norm.weight": "pre_feedforward_layernorm.weight",
            "post_attention_norm.weight": "post_attention_layernorm.weight",
            "post_ffw_norm.weight": "post_feedforward_layernorm.weight",
        }
        if rest in suffix_map:
            return f"language_model.model.layers.{n}.{suffix_map[rest]}"
        print(f"WARNING: unrecognized blk tensor suffix, skipping: {name}", file=sys.stderr)
        return None
    # non-weight metadata tensors (shouldn't appear in r.tensors, but just in case)
    print(f"WARNING: unrecognized tensor name, skipping: {name}", file=sys.stderr)
    return None


def _size_gib(num_bytes: int | float) -> float:
    return round(float(num_bytes) / (1024**3), 3)


def _free_bytes_for(path: Path) -> int:
    target = path if path.exists() else path.parent
    while not target.exists() and target != target.parent:
        target = target.parent
    return shutil.disk_usage(target).free


def _build_plan(reader: GGUFReader) -> list[tuple[str, object]]:  # noqa: ANN401
    plan = []
    for tensor in reader.tensors:
        out_key = remap_key(tensor.name)
        if out_key is None:
            continue
        plan.append((out_key, tensor))
    return plan


def _output_shape(tensor: object) -> list[int]:  # noqa: ANN401
    shape = [int(dim) for dim in tensor.shape]
    if len(shape) == 2:
        return [shape[1], shape[0]]
    return shape


def _header_for_plan(plan: list[tuple[str, object]]) -> tuple[dict[str, dict[str, object]], list[int], int]:  # noqa: ANN401
    new_header: dict[str, dict[str, object]] = {}
    offset = 0
    sizes = []
    for out_key, tensor in plan:
        n_elems = 1
        for dim in tensor.shape:
            n_elems *= int(dim)
        nbytes = n_elems * 2  # bf16
        shape = _output_shape(tensor)
        new_header[out_key] = {"dtype": "BF16", "shape": shape, "data_offsets": [offset, offset + nbytes]}
        sizes.append(nbytes)
        offset += nbytes
    return new_header, sizes, offset


def _processor_status(copy_from: Path | None) -> list[dict[str, object]]:
    if copy_from is None:
        return [{"filename": name, "present": False, "source": ""} for name in PROCESSOR_FILES]
    return [
        {
            "filename": name,
            "present": (copy_from / name).is_file(),
            "source": str(copy_from / name),
        }
        for name in PROCESSOR_FILES
    ]


def _vision_sidecar_plan(copy_from: Path | None) -> dict[str, object]:
    if copy_from is None:
        return {"source": "", "tensor_count": 0, "size_gib": 0, "available": False}
    tensor_count = 0
    total_bytes = 0
    for shard in sorted(copy_from.glob("*.safetensors")):
        with shard.open("rb") as handle:
            header_len = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_len))
        for key, info in header.items():
            if key == "__metadata__" or not key.startswith(VISION_PREFIXES):
                continue
            tensor_count += 1
            start, end = info["data_offsets"]
            total_bytes += int(end) - int(start)
    return {
        "source": str(copy_from),
        "tensor_count": tensor_count,
        "size_gib": _size_gib(total_bytes),
        "available": tensor_count > 0,
    }


def _copy_vision_sidecar(copy_from: Path, dst_dir: Path, *, overwrite: bool = False) -> Path:
    out = dst_dir / VISION_SIDECAR
    if out.exists() and not overwrite:
        raise FileExistsError(f"Vision sidecar already exists, pass --overwrite to replace: {out}")

    from safetensors import safe_open
    from safetensors.torch import save_file

    tensors = {}
    for shard in sorted(copy_from.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith(VISION_PREFIXES):
                    tensors[key] = handle.get_tensor(key).clone().contiguous()
    if not tensors:
        raise RuntimeError(f"No vision/projector tensors found under {copy_from}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, out)
    return out


def _summary_payload(
    *,
    src: Path,
    dst_dir: Path,
    dst: Path,
    copy_processor_from: Path | None,
    reader: GGUFReader,
    plan: list[tuple[str, object]],  # noqa: ANN401
    output_bytes: int,
) -> dict[str, object]:
    free_bytes = _free_bytes_for(dst_dir)
    processor_files = _processor_status(copy_processor_from)
    missing_processor = [item["filename"] for item in processor_files if not item["present"]]
    return {
        "ok": bool(src.is_file() and plan and free_bytes > output_bytes and not missing_processor),
        "mode": "dry-run",
        "src": str(src),
        "dst_dir": str(dst_dir),
        "dst": str(dst),
        "source_size_gib": _size_gib(src.stat().st_size) if src.is_file() else 0,
        "tensor_count": len(reader.tensors),
        "planned_tensors": len(plan),
        "transposed_2d_tensors": sum(1 for _, tensor in plan if len(tensor.shape) == 2),
        "output_size_gib": _size_gib(output_bytes),
        "free_gib": _size_gib(free_bytes),
        "processor_copy_from": str(copy_processor_from) if copy_processor_from is not None else "",
        "processor_files": processor_files,
        "missing_processor_files": missing_processor,
        "vision_sidecar": _vision_sidecar_plan(copy_processor_from),
        "vision_sidecar_path": str(dst_dir / VISION_SIDECAR),
        "conversion_ready": bool(src.is_file() and plan and free_bytes > output_bytes and not missing_processor),
        "source_text_only": True,
        "vision_sidecar_required": True,
        "generation_ready_after_conversion": (
            "plain text-to-video when model.safetensors and vision_projector.safetensors are present; "
            "prompt enhancement with images remains unverified"
        ),
        "notes": (
            "Converts the smallest Heretic Q3 GGUF download into HF-shaped BF16 safetensors for LTX. "
            "Runtime size is full BF16; this does not preserve Q3 memory savings."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--dst", default=str(DEFAULT_DST))
    ap.add_argument("--copy-processor-from", default=str(DEFAULT_COPY_PROCESSOR_FROM))
    ap.add_argument("--copy-vision-from", default=str(DEFAULT_COPY_PROCESSOR_FROM))
    ap.add_argument(
        "--skip-vision-sidecar",
        action="store_true",
        help="Do not copy vision_tower/multi_modal_projector sidecar tensors into the destination.",
    )
    ap.add_argument(
        "--vision-sidecar-only",
        action="store_true",
        help="Only create vision_projector.safetensors in an existing converted destination.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Inspect GGUF metadata and estimate output without writing weights.")
    ap.add_argument("--receipt", default=str(DEFAULT_RECEIPT), help="Optional JSON receipt path for dry-run or conversion summary.")
    ap.add_argument("--overwrite", action="store_true", help="Allow replacing an existing destination model.safetensors.")
    args = ap.parse_args()

    src = Path(args.src)
    dst_dir = Path(args.dst)
    dst = dst_dir / "model.safetensors"
    copy_processor_from = Path(args.copy_processor_from) if args.copy_processor_from else None
    copy_vision_from = Path(args.copy_vision_from) if args.copy_vision_from else copy_processor_from

    if args.vision_sidecar_only:
        if copy_vision_from is None:
            raise ValueError("--vision-sidecar-only requires --copy-vision-from")
        sidecar = _copy_vision_sidecar(copy_vision_from, dst_dir, overwrite=args.overwrite)
        payload = {
            "ok": True,
            "mode": "vision-sidecar",
            "dst_dir": str(dst_dir),
            "vision_sidecar_path": str(sidecar),
            "vision_sidecar_size_gib": _size_gib(sidecar.stat().st_size),
            "vision_sidecar": _vision_sidecar_plan(copy_vision_from),
        }
        if args.receipt:
            receipt = Path(args.receipt)
            receipt.parent.mkdir(parents=True, exist_ok=True)
            receipt.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if not src.is_file():
        raise FileNotFoundError(f"GGUF source missing: {src}")
    print(f"Reading GGUF {src} ...")
    reader = GGUFReader(str(src))
    print(f"{len(reader.tensors)} tensors found")

    # ---- Pass 1: dequantize everything into memory-mapped plan (shapes only) ----
    # We dequantize tensor-by-tensor in pass 2 directly (streaming), but need
    # final byte sizes up front to build the header. Since GGUF tensors are
    # already memory-mapped, computing shape/dtype without materializing is cheap.
    plan = _build_plan(reader)

    print(f"{len(plan)} tensors will be written")

    # Need shapes to build header -- dequantize() requires loading the block
    # data anyway to know elem count for quantized types, but element shape is
    # derivable from t.shape directly (already in HF/torch order per our
    # verification: dequantize() output shape == reduce via t.shape correctly).
    new_header, sizes, offset = _header_for_plan(plan)

    header_bytes = json.dumps(new_header).encode("utf-8")
    pad = (-len(header_bytes)) % 8
    header_bytes += b" " * pad
    header_len = len(header_bytes)

    print(f"New checkpoint size: {offset / (1024**3):.2f} GiB (+ {header_len} byte header)")
    summary = _summary_payload(
        src=src,
        dst_dir=dst_dir,
        dst=dst,
        copy_processor_from=copy_processor_from,
        reader=reader,
        plan=plan,
        output_bytes=offset + header_len + 8,
    )
    if args.receipt:
        receipt = Path(args.receipt)
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote receipt: {receipt}")
    if args.dry_run:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["conversion_ready"] else 1

    if dst.exists() and not args.overwrite:
        raise FileExistsError(f"Destination already exists, pass --overwrite to replace: {dst}")

    dst_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with open(dst, "wb") as fdst:
        fdst.write(struct.pack("<Q", header_len))
        fdst.write(header_bytes)

        for i, (out_key, t) in enumerate(plan):
            arr = dequantize(t.data, t.tensor_type)
            arr = arr.reshape([int(dim) for dim in t.shape])
            if arr.ndim == 2:
                arr = arr.T
            arr = np.ascontiguousarray(arr)
            fdst.write(f32_to_bf16_bytes(arr))
            del arr
            if i % 50 == 0 or i == len(plan) - 1:
                elapsed = time.time() - t0
                print(f"  [{i + 1}/{len(plan)}] {out_key} -- {elapsed:.1f}s elapsed")

    print(f"Wrote {dst} in {time.time() - t0:.1f}s")

    if copy_processor_from is not None:
        src_dir = copy_processor_from
        for fname in PROCESSOR_FILES:
            src_file = src_dir / fname
            if src_file.exists():
                shutil.copy2(src_file, dst_dir / fname)
                print(f"Copied {fname}")
            else:
                print(f"  (skip, not found in source: {fname})")
    else:
        print("No --copy-processor-from given -- tokenizer/processor files must be added manually.")

    if not args.skip_vision_sidecar:
        if copy_vision_from is None:
            print("No --copy-vision-from given -- vision/projector tensors will remain unavailable.")
        else:
            sidecar = _copy_vision_sidecar(copy_vision_from, dst_dir, overwrite=args.overwrite)
            print(f"Copied vision/projector sidecar -> {sidecar}")

    print(f"\nDone. New gemma_root: {dst_dir}")
    print(
        "NOTE: the Heretic GGUF source is text-only. This folder uses the official Gemma "
        "vision/projector sidecar for loader compatibility; image-conditioned prompt enhancement remains unverified."
    )
    summary["mode"] = "conversion"
    summary["wrote_model"] = str(dst)
    summary["elapsed_seconds"] = round(time.time() - t0, 3)
    sidecar = dst_dir / VISION_SIDECAR
    if sidecar.is_file():
        summary["vision_sidecar_written"] = str(sidecar)
        summary["vision_sidecar_size_gib"] = _size_gib(sidecar.stat().st_size)
    if args.receipt:
        Path(args.receipt).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
