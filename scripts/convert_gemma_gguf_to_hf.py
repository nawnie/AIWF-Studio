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
NOT converted by this script). The resulting gemma_root will have no vision
tower. This is expected to be fine for LTX's plain text-to-video prompt
encoding (GemmaTextEncoder.encode() only runs the language-model forward
pass), but will NOT support image-conditioned prompt enhancement
(enhance_i2v()) -- unverified until the Builder actually loads this folder.

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--copy-processor-from", default=None)
    args = ap.parse_args()

    src = Path(args.src)
    dst_dir = Path(args.dst)
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / "model.safetensors"

    print(f"Reading GGUF {src} ...")
    reader = GGUFReader(str(src))
    print(f"{len(reader.tensors)} tensors found")

    # ---- Pass 1: dequantize everything into memory-mapped plan (shapes only) ----
    # We dequantize tensor-by-tensor in pass 2 directly (streaming), but need
    # final byte sizes up front to build the header. Since GGUF tensors are
    # already memory-mapped, computing shape/dtype without materializing is cheap.
    plan = []  # (out_key, tensor)
    for t in reader.tensors:
        out_key = remap_key(t.name)
        if out_key is None:
            continue
        plan.append((out_key, t))

    print(f"{len(plan)} tensors will be written")

    # Need shapes to build header -- dequantize() requires loading the block
    # data anyway to know elem count for quantized types, but element shape is
    # derivable from t.shape directly (already in HF/torch order per our
    # verification: dequantize() output shape == reduce via t.shape correctly).
    new_header = {}
    offset = 0
    sizes = []
    for out_key, t in plan:
        n_elems = 1
        for d in t.shape:
            n_elems *= int(d)
        nbytes = n_elems * 2  # bf16
        shape = [int(d) for d in t.shape]
        new_header[out_key] = {"dtype": "BF16", "shape": shape, "data_offsets": [offset, offset + nbytes]}
        sizes.append(nbytes)
        offset += nbytes

    header_bytes = json.dumps(new_header).encode("utf-8")
    pad = (-len(header_bytes)) % 8
    header_bytes += b" " * pad
    header_len = len(header_bytes)

    print(f"New checkpoint size: {offset / (1024**3):.2f} GiB (+ {header_len} byte header)")

    t0 = time.time()
    with open(dst, "wb") as fdst:
        fdst.write(struct.pack("<Q", header_len))
        fdst.write(header_bytes)

        for i, (out_key, t) in enumerate(plan):
            arr = dequantize(t.data, t.tensor_type)
            arr = arr.reshape(new_header[out_key]["shape"])
            fdst.write(f32_to_bf16_bytes(arr))
            del arr
            if i % 50 == 0 or i == len(plan) - 1:
                elapsed = time.time() - t0
                print(f"  [{i + 1}/{len(plan)}] {out_key} -- {elapsed:.1f}s elapsed")

    print(f"Wrote {dst} in {time.time() - t0:.1f}s")

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
        print("No --copy-processor-from given -- tokenizer/processor files must be added manually.")

    print(f"\nDone. New gemma_root: {dst_dir}")
    print("NOTE: this gemma_root has NO vision tower (text-only GGUF source). "
          "Image-conditioned prompt enhancement will not work; plain text encoding should.")


if __name__ == "__main__":
    main()
