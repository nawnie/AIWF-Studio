"""
Convert the ComfyUI-format NVFP4 Gemma-3-12B-it checkpoint
(`gemma_3_12B_it_fp4_mixed.safetensors`, found in Downloads -- 9.01GB) into
the HF-style layout that AIWF's LTX worker / ltx-core's Gemma loader expects.

Why this exists
----------------
This checkpoint is the NVFP4 sibling of the fp8 ComfyUI checkpoint already
handled by remap_gemma_fp8_comfy.py, and uses the same key-prefix convention:
    model.<...>                    -> language_model.model.<...>
    vision_model.<...>             -> vision_tower.vision_model.<...>
    multi_modal_projector.<...>    -> multi_modal_projector.<...>  (unchanged)
    spiece_model                   -> extracted to tokenizer.model (not a weight)

Inspected header (2040 tensors):
  - Only `model.*` (language-model) Linear layers are NVFP4-quantized: each
    quantized layer is a {weight (U8, packed 2x4bit), weight_scale (F8_E4M3,
    per-16-block), weight_scale_2 (F32, per-tensor), comfy_quant (U8[19],
    ComfyUI-internal quant-method metadata, dropped on conversion)} tuple --
    302 such layers.
  - vision_model.* and multi_modal_projector.* (439 tensors) are plain BF16/F32,
    NOT quantized -- meaning, unlike the GGUF route which ships the vision
    tower in a completely different ggml/clip.cpp format, this checkpoint's
    vision tower is already HF-shaped and just needs the key-prefix rename.
  - Norms/embeddings in the language model are also plain BF16, copied through
    unchanged (renamed).

Dequantization reuses ComfyUI's pure-PyTorch NVFP4 kernel (`comfy_kitchen.
dequantize_nvfp4`), the same one used by convert_nvfp4_to_bf16.py for the
main LTX transformer checkpoint. No Blackwell GPU needed for plain
dequantization (only the *accelerated matmul* path needs SM>=10.0).

IMPORTANT -- same caveat as the fp8 path: once dequantized to bf16, the
output is full bf16 size (~24GB) when loaded. This script only wins on
*download* size (9GB vs 23.5GB) and reuses tooling we already have working
(comfy_kitchen in ComfyUI's venv) -- it does not reduce VRAM footprint.

Run with ComfyUI's venv Python (has comfy_kitchen installed):
    F:\\ComfyUI\\venv\\Scripts\\python.exe F:\\AIWF_Studio\\scripts\\convert_gemma_nvfp4_to_hf.py ^
        --src "C:\\Users\\Shawn\\Downloads\\gemma_3_12B_it_fp4_mixed.safetensors" ^
        --dst "F:\\AIWF_Studio\\models\\ltx\\text_encoder\\gemma-3-12b-nvfp4-converted"

Streams tensor-by-tensor so peak memory stays low, not "the whole 12B model".
"""
import argparse
import json
import struct
import sys
import time
from pathlib import Path

import torch

try:
    import comfy_kitchen as ck
except ImportError:
    print(
        "ERROR: comfy_kitchen is not importable. Run this script with ComfyUI's\n"
        "venv Python, e.g. F:\\ComfyUI\\venv\\Scripts\\python.exe ...",
        file=sys.stderr,
    )
    sys.exit(1)

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

SPIECE_KEY = "spiece_model"

# Files the Gemma loader's module_ops_from_gemma_root() / AutoImageProcessor
# need that aren't weights. tokenizer.model is reconstructed from the
# embedded spiece_model tensor, so it's not in this list.
PROCESSOR_FILES = (
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.json",
    "added_tokens.json",
)


def read_header(path: Path):
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    data_start = 8 + header_len
    return header, data_start


def remap_key(key: str) -> str | None:
    """Return the HF-style key name, or None if this key shouldn't be written
    out as a model weight (embedded tokenizer blob)."""
    if key == SPIECE_KEY:
        return None
    if key.startswith("model."):
        return "language_model." + key
    if key.startswith("vision_model."):
        return "vision_tower." + key
    if key.startswith("multi_modal_projector."):
        return key
    print(f"WARNING: unrecognized key prefix, passing through unchanged: {key}", file=sys.stderr)
    return key


def find_quant_layers(header: dict):
    """Group .weight / .weight_scale / .weight_scale_2 triples by layer prefix.
    (comfy_quant is intentionally ignored here -- it's ComfyUI-internal
    quant-method metadata not needed once we've dequantized to bf16.)"""
    layers = {}
    for k, info in header.items():
        if k == "__metadata__":
            continue
        if k.endswith(".weight_scale_2"):
            layers.setdefault(k[: -len(".weight_scale_2")], {})["weight_scale_2"] = k
        elif k.endswith(".weight_scale"):
            layers.setdefault(k[: -len(".weight_scale")], {})["weight_scale"] = k
        elif k.endswith(".weight") and info.get("dtype") == "U8":
            layers.setdefault(k[: -len(".weight")], {})["weight"] = k

    complete = {
        p: d for p, d in layers.items() if {"weight", "weight_scale", "weight_scale_2"} <= d.keys()
    }
    return complete


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to gemma_3_12B_it_fp4_mixed.safetensors")
    ap.add_argument("--dst", required=True, help="Destination gemma_root directory to create")
    ap.add_argument(
        "--copy-processor-from",
        default=None,
        help="Optional existing gemma_root (e.g. the old unquantized folder) to copy "
        "preprocessor/tokenizer config files from. Not required for tokenizer.model "
        "(reconstructed from the embedded spiece_model tensor).",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                     help="Device to run dequantization on (default: cuda if available, else cpu)")
    args = ap.parse_args()

    src = Path(args.src)
    dst_dir = Path(args.dst)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "model.safetensors"
    device = torch.device(args.device)

    print(f"Reading header from {src} ...")
    header, data_start = read_header(src)
    meta = header.get("__metadata__")

    quant_layers = find_quant_layers(header)
    quant_keys_used = set()
    for d in quant_layers.values():
        quant_keys_used.update(d.values())
    # also drop comfy_quant tensors -- not needed post-dequant
    comfy_quant_keys = {k for k in header if k != "__metadata__" and k.endswith(".comfy_quant")}

    skip_keys = quant_keys_used | comfy_quant_keys | {SPIECE_KEY}
    plain_keys = [k for k in header if k != "__metadata__" and k not in skip_keys]

    print(f"Found {len(quant_layers)} NVFP4-quantized language-model layers, "
          f"{len(plain_keys)} plain tensors (incl. full vision tower + projector)")
    print(f"Dequantizing on device: {device}")

    # ---- Pass 1: build the new header ----
    new_header = {}
    write_order = []  # list of (out_key, kind, payload)
    offset = 0

    for k in plain_keys:
        out_key = remap_key(k)
        if out_key is None:
            continue
        info = header[k]
        nbytes = info["data_offsets"][1] - info["data_offsets"][0]
        new_header[out_key] = {"dtype": info["dtype"], "shape": info["shape"], "data_offsets": [offset, offset + nbytes]}
        write_order.append((out_key, "plain", info))
        offset += nbytes

    for prefix in sorted(quant_layers):
        wk = quant_layers[prefix]["weight"]
        out_key = remap_key(wk)
        w_info = header[wk]
        packed_shape = w_info["shape"]  # [out_features, in_features // 2]
        unpacked_shape = [packed_shape[0], packed_shape[1] * 2]
        nbytes = unpacked_shape[0] * unpacked_shape[1] * 2  # bf16 = 2 bytes/elem
        new_header[out_key] = {"dtype": "BF16", "shape": unpacked_shape, "data_offsets": [offset, offset + nbytes]}
        write_order.append((out_key, "quant", prefix))
        offset += nbytes

    if meta is not None:
        new_header["__metadata__"] = meta

    header_bytes = json.dumps(new_header).encode("utf-8")
    pad = (-len(header_bytes)) % 8
    header_bytes += b" " * pad
    header_len = len(header_bytes)

    print(f"New checkpoint size: {offset / (1024**3):.2f} GiB (+ {header_len} byte header)")

    # ---- Pass 2: stream tensors through, dequantizing nvfp4 layers ----
    t0 = time.time()
    n_quant_done = 0
    with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
        fdst.write(struct.pack("<Q", header_len))
        fdst.write(header_bytes)

        def read_raw(info):
            start, end = info["data_offsets"]
            fsrc.seek(data_start + start)
            return fsrc.read(end - start)

        def load_tensor(info):
            buf = bytearray(read_raw(info))
            dtype = DTYPE_MAP[info["dtype"]]
            t = torch.frombuffer(buf, dtype=dtype)
            shape = info["shape"] if info["shape"] else []
            return t.reshape(shape).clone()

        for i, (out_key, kind, payload) in enumerate(write_order):
            if kind == "plain":
                fdst.write(read_raw(payload))
            else:
                prefix = payload
                w_info = header[quant_layers[prefix]["weight"]]
                s_info = header[quant_layers[prefix]["weight_scale"]]
                s2_info = header[quant_layers[prefix]["weight_scale_2"]]

                w_raw = load_tensor(w_info).to(device)
                block_scale = load_tensor(s_info).to(device)
                tensor_scale = load_tensor(s2_info).to(device)

                dq = ck.dequantize_nvfp4(w_raw, tensor_scale, block_scale, output_type=torch.bfloat16)
                dq = dq.contiguous().to("cpu")

                raw_bytes = dq.view(torch.uint8).numpy().tobytes()
                fdst.write(raw_bytes)

                n_quant_done += 1
                del w_raw, block_scale, tensor_scale, dq

            if i % 200 == 0 or i == len(write_order) - 1:
                elapsed = time.time() - t0
                print(f"  [{i + 1}/{len(write_order)}] tensors written "
                      f"({n_quant_done} dequantized) -- {elapsed:.1f}s elapsed")

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
        import shutil
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
