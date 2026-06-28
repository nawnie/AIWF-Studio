"""
Convert an NVFP4-quantized safetensors checkpoint (NVIDIA ModelOpt / TensorRT-LLM
convention, as used by the LTX-2 nvfp4 release) into a plain full-precision
(bf16) safetensors checkpoint that any normal diffusers/LTX loader can read.

Why this exists
----------------
`ltx-2.3-22b-dev-nvfp4.safetensors` stores each quantized Linear layer as three
tensors:
    <layer>.weight         uint8   packed 4-bit (2 values per byte)
    <layer>.weight_scale   float8_e4m3fn   per-block scale (block_size=16),
                            stored in a cuBLAS "swizzled"/blocked tile layout
    <layer>.weight_scale_2 float32 scalar  per-tensor scale
ltx-core/ltx-pipelines 1.1.6 has no NVFP4 unpacking support, so loading this
checkpoint directly fails with hundreds of shape-mismatch errors (see
ltx-nvfp4-checkpoint-unsupported memory).

ComfyUI ships a pure-PyTorch ("eager") NVFP4 dequantization kernel in the
`comfy_kitchen` package that requires NO Blackwell GPU / CUDA extension --
only the *accelerated matmul* path needs SM>=10.0, plain dequantization is
just bit ops + an embedding lookup + a multiply. This script reuses that
kernel (via ComfyUI's venv, where comfy_kitchen is already installed) to
dequantize every NVFP4 layer once, up front, into bf16, and writes a new
safetensors file with the same key names as a normal (non-quantized)
checkpoint. AIWF's existing LTX worker can then load that file with zero
engine changes.

Run with ComfyUI's venv Python (has comfy_kitchen installed), e.g.:
    F:\\ComfyUI\\venv\\Scripts\\python.exe F:\\AIWF_Studio\\scripts\\convert_nvfp4_to_bf16.py ^
        --src "F:\\AIWF_Studio\\models\\ltx\\checkpoints\\ltx-2.3-22b-dev-nvfp4.safetensors" ^
        --dst "F:\\AIWF_Studio\\models\\ltx\\checkpoints\\ltx-2.3-22b-dev-bf16.safetensors"

The script streams tensor-by-tensor (reads source bytes directly, writes
output bytes directly) so peak memory stays at "a couple of layers", not
"the whole 22B-parameter model" -- important since the dequantized output is
~2x the size of the nvfp4 file (roughly 44GB for this checkpoint).
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


def read_header(path: Path):
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    data_start = 8 + header_len
    return header, data_start


def find_quant_layers(header: dict):
    """Group .weight / .weight_scale / .weight_scale_2 triples by layer prefix."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Path to nvfp4 safetensors checkpoint")
    ap.add_argument("--dst", required=True, help="Path to write converted bf16 safetensors checkpoint")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                     help="Device to run dequantization on (default: cuda if available, else cpu)")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    device = torch.device(args.device)

    print(f"Reading header from {src} ...")
    header, data_start = read_header(src)
    meta = header.get("__metadata__")

    quant_layers = find_quant_layers(header)
    quant_keys_used = set()
    for d in quant_layers.values():
        quant_keys_used.update(d.values())

    plain_keys = [k for k in header if k != "__metadata__" and k not in quant_keys_used]

    print(f"Found {len(quant_layers)} NVFP4-quantized layers, {len(plain_keys)} plain tensors")
    print(f"Dequantizing on device: {device}")

    # ---- Pass 1: build the new header (shapes/dtypes/offsets) ----
    new_header = {}
    write_order = []  # list of (out_key, kind, payload)
    offset = 0

    for k in plain_keys:
        info = header[k]
        nbytes = info["data_offsets"][1] - info["data_offsets"][0]
        new_header[k] = {"dtype": info["dtype"], "shape": info["shape"], "data_offsets": [offset, offset + nbytes]}
        write_order.append((k, "plain", info))
        offset += nbytes

    for prefix in sorted(quant_layers):
        wk = quant_layers[prefix]["weight"]
        w_info = header[wk]
        packed_shape = w_info["shape"]  # [out_features, in_features // 2]
        unpacked_shape = [packed_shape[0], packed_shape[1] * 2]
        nbytes = unpacked_shape[0] * unpacked_shape[1] * 2  # bf16 = 2 bytes/elem
        new_header[wk] = {"dtype": "BF16", "shape": unpacked_shape, "data_offsets": [offset, offset + nbytes]}
        write_order.append((wk, "quant", prefix))
        offset += nbytes

    if meta is not None:
        new_header["__metadata__"] = meta

    header_bytes = json.dumps(new_header).encode("utf-8")
    pad = (-len(header_bytes)) % 8
    header_bytes += b" " * pad
    header_len = len(header_bytes)

    print(f"New checkpoint size: {offset / (1024**3):.2f} GiB (+ {header_len} byte header)")
    dst.parent.mkdir(parents=True, exist_ok=True)

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

        for i, (key, kind, payload) in enumerate(write_order):
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

                # reinterpret bf16 storage as raw bytes without numpy (numpy has no
                # native bf16 dtype on most installs)
                raw_bytes = dq.view(torch.uint8).numpy().tobytes()
                fdst.write(raw_bytes)

                n_quant_done += 1
                del w_raw, block_scale, tensor_scale, dq

            if i % 200 == 0 or i == len(write_order) - 1:
                elapsed = time.time() - t0
                print(f"  [{i + 1}/{len(write_order)}] tensors written "
                      f"({n_quant_done} dequantized) -- {elapsed:.1f}s elapsed")

    print(f"Done in {time.time() - t0:.1f}s -> {dst}")


if __name__ == "__main__":
    main()
