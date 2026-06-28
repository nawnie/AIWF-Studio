"""
Convert the ComfyUI-format fp8 Gemma-3-12B text encoder checkpoint
(`gemma_3_12B_it_fp8_e4m3fn.safetensors`, from GitMylo/LTX-2-comfy_gemma_fp8_e4m3fn)
into the HF-style layout that AIWF's LTX worker / ltx-core expects.

Why this exists
----------------
ltx-core's Gemma loader (`ltx_pipelines.utils.blocks.PromptEncoder`) globs
`gemma_root` for `*.safetensors` and runs every tensor through
`GEMMA_LLM_KEY_OPS`, which expects HF `Gemma3ForConditionalGeneration` key
names:
    language_model.model.layers.<N>...
    vision_tower.vision_model...
    multi_modal_projector...

ComfyUI's checkpoint strips those prefixes down to:
    model.layers.<N>...
    vision_model...
    multi_modal_projector...   (already matches)

It also stores the big Linear weight matrices (q/k/v/o_proj, mlp
gate/up/down/fc1/fc2) as plain `float8_e4m3fn` (no block/tensor scale --
this is *not* NVFP4, just a straight 8-bit float cast of each weight).
ltx-core has no fp8 matmul path, so those tensors are widened to bf16 here.
Norms/biases/embeddings are already bf16 in the source file and are copied
through unchanged.

There's also a `spiece_model` tensor: ComfyUI embeds the raw sentencepiece
tokenizer.model bytes as a U8 tensor inside the checkpoint. This script
extracts it back out to a real `tokenizer.model` file in the destination
folder, so the converted folder is usable without needing the old
unquantized folder's tokenizer files at all (though we'll still copy the
small processor/config files over if a source folder is given, since the
image processor needs `preprocessor_config.json`).

IMPORTANT — this does NOT make the text encoder smaller to load.
Once the fp8 matrices are widened to bf16 here, the output is full bf16
size (~24GB), same as a stock unquantized Gemma-3-12B checkpoint. The only
win versus downloading the unquantized HF repo directly is a smaller
*download* (~13GB fp8 vs ~24GB bf16) -- VRAM/RAM footprint when loaded is
unchanged. For an actual smaller footprint, a real quantized (q4/q4_0/
nvfp4) variant with a working dequant kernel is still needed (see
ltx-nvfp4-checkpoint-unsupported memory for why naive quantized loads fail).

Usage:
    python remap_gemma_fp8_comfy.py ^
        --src "F:\\AIWF_Studio\\models\\ltx\\text_encoder\\gemma-3-12b-fp8-comfy\\gemma_3_12B_it_fp8_e4m3fn.safetensors" ^
        --dst "F:\\AIWF_Studio\\models\\ltx\\text_encoder\\gemma-3-12b-remapped" ^
        --copy-processor-from "F:\\AIWF_Studio\\models\\ltx\\text_encoder\\gemma-3-12b-it-qat-q4_0-unquantized"

Streams tensor-by-tensor (reads source bytes directly, writes output bytes
directly) so peak memory stays at "a couple of tensors", not "the whole
12B-parameter model".
"""
import argparse
import json
import shutil
import struct
import sys
import time
from pathlib import Path

import torch

DTYPE_MAP = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}

# Files the Gemma loader's module_ops_from_gemma_root() / AutoImageProcessor
# need that aren't weights. tokenizer.model is reconstructed from the
# embedded `spiece_model` tensor, so it's not in this list.
PROCESSOR_FILES = (
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.json",
    "added_tokens.json",
)

SPIECE_KEY = "spiece_model"


def read_header(path: Path):
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    data_start = 8 + header_len
    return header, data_start


def remap_key(key: str) -> str | None:
    """Return the HF-style key name, or None if this key shouldn't be written
    out as a model weight (e.g. the embedded tokenizer blob)."""
    if key == SPIECE_KEY:
        return None
    if key.startswith("model."):
        return "language_model." + key
    if key.startswith("vision_model."):
        return "vision_tower." + key
    if key.startswith("multi_modal_projector."):
        return key
    # Unknown key -- pass through unchanged rather than silently dropping it,
    # but this should not happen for this checkpoint.
    print(f"WARNING: unrecognized key prefix, passing through unchanged: {key}", file=sys.stderr)
    return key


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to gemma_3_12B_it_fp8_e4m3fn.safetensors")
    ap.add_argument("--dst", required=True, help="Destination gemma_root directory to create")
    ap.add_argument(
        "--copy-processor-from",
        default=None,
        help="Optional existing gemma_root (e.g. the old unquantized folder) to copy "
        "preprocessor/tokenizer config files from. Not required for tokenizer.model "
        "(reconstructed from the embedded spiece_model tensor).",
    )
    ap.add_argument(
        "--keep-fp8",
        action="store_true",
        help="Write F8_E4M3 weights through unchanged instead of widening to bf16. "
        "ltx-core has no fp8 matmul path today, so only use this if/when that lands.",
    )
    args = ap.parse_args()

    src = Path(args.src)
    dst_dir = Path(args.dst)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "model.safetensors"

    print(f"Reading header from {src} ...")
    header, data_start = read_header(src)
    meta = header.get("__metadata__")

    keys = [k for k in header if k != "__metadata__"]
    n_fp8 = sum(1 for k in keys if header[k]["dtype"] == "F8_E4M3")
    print(f"{len(keys)} tensors total, {n_fp8} are F8_E4M3"
          f" ({'kept as fp8' if args.keep_fp8 else 'will be widened to bf16'})")

    # ---- Pass 1: build the new header (renamed keys, possibly new dtype/size) ----
    new_header = {}
    write_order = []  # list of (src_key, out_key, info)
    offset = 0

    for k in keys:
        out_key = remap_key(k)
        if out_key is None:
            continue
        info = header[k]
        nbytes = info["data_offsets"][1] - info["data_offsets"][0]
        out_dtype = info["dtype"]
        out_nbytes = nbytes
        if info["dtype"] == "F8_E4M3" and not args.keep_fp8:
            out_dtype = "BF16"
            out_nbytes = nbytes * 2  # 1 byte/elem -> 2 bytes/elem
        new_header[out_key] = {
            "dtype": out_dtype,
            "shape": info["shape"],
            "data_offsets": [offset, offset + out_nbytes],
        }
        write_order.append((k, out_key, info))
        offset += out_nbytes

    if meta is not None:
        new_header["__metadata__"] = meta

    header_bytes = json.dumps(new_header).encode("utf-8")
    pad = (-len(header_bytes)) % 8
    header_bytes += b" " * pad
    header_len = len(header_bytes)

    print(f"New checkpoint size: {offset / (1024**3):.2f} GiB (+ {header_len} byte header)")

    # ---- Pass 2: stream tensors through, widening fp8 -> bf16 as needed ----
    t0 = time.time()
    n_done = 0
    with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
        fdst.write(struct.pack("<Q", header_len))
        fdst.write(header_bytes)

        def read_raw(info):
            start, end = info["data_offsets"]
            fsrc.seek(data_start + start)
            return fsrc.read(end - start)

        for i, (src_key, out_key, info) in enumerate(write_order):
            if info["dtype"] == "F8_E4M3" and not args.keep_fp8:
                buf = bytearray(read_raw(info))
                t = torch.frombuffer(buf, dtype=torch.float8_e4m3fn)
                t = t.reshape(info["shape"]).to(torch.bfloat16).contiguous()
                fdst.write(t.view(torch.uint8).numpy().tobytes())
            else:
                fdst.write(read_raw(info))
            n_done += 1
            if i % 200 == 0 or i == len(write_order) - 1:
                elapsed = time.time() - t0
                print(f"  [{i + 1}/{len(write_order)}] tensors written -- {elapsed:.1f}s elapsed")

    print(f"Wrote {dst} in {time.time() - t0:.1f}s")

    # ---- Extract the embedded tokenizer ----
    if SPIECE_KEY in header:
        info = header[SPIECE_KEY]
        with open(src, "rb") as fsrc:
            fsrc.seek(data_start + info["data_offsets"][0])
            spiece_bytes = fsrc.read(info["data_offsets"][1] - info["data_offsets"][0])
        tok_path = dst_dir / "tokenizer.model"
        tok_path.write_bytes(spiece_bytes)
        print(f"Extracted embedded tokenizer -> {tok_path} ({len(spiece_bytes)} bytes)")
    else:
        print("WARNING: no spiece_model tensor found -- tokenizer.model will need to be supplied manually.",
              file=sys.stderr)

    # ---- Copy over the small processor/config files ----
    if args.copy_processor_from:
        src_dir = Path(args.copy_processor_from)
        for fname in PROCESSOR_FILES:
            src_file = src_dir / fname
            if src_file.exists():
                shutil.copy2(src_file, dst_dir / fname)
                print(f"Copied {fname}")
            else:
                print(f"  (skip, not found in source: {fname})")
    else:
        print("No --copy-processor-from given -- you'll need preprocessor_config.json "
              "and tokenizer_config.json in the destination folder before this gemma_root is usable.")

    print(f"\nDone. New gemma_root: {dst_dir}")


if __name__ == "__main__":
    main()
