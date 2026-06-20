# Wan Local Components

AIWF keeps Wan generation local-only. The generation workflow must not download model files.

For Wan 2.2 high/low transformer pairs, AIWF needs these pieces:

- High-noise transformer: selected from `models/wan/Safetensor`, `models/wan/GGUF`, or another scanned Wan model folder.
- Low-noise transformer: selected from `models/wan/Safetensor`, `models/wan/GGUF`, or another scanned Wan model folder.
- Text encoder: UMT5 XXL weights, stored in the local component base under `text_encoder/model.safetensors`.
- Tokenizer: T5 tokenizer files under `tokenizer/`.
- Scheduler: `scheduler/scheduler_config.json`.
- VAE: selected from `models/VAE`; Wan commonly uses `wan2.1_vae.safetensors`.

Current local component base:

```text
models/wan/Diffusers/Wan2.2-TI2V-5B-Diffusers/
|-- model_index.json
|-- scheduler/scheduler_config.json
|-- text_encoder/config.json
|-- text_encoder/model.safetensors
`-- tokenizer/
    |-- special_tokens_map.json
    |-- spiece.model
    |-- tokenizer.json
    `-- tokenizer_config.json
```

Component source mapping:

- `text_encoder/model.safetensors` can be copied from a compatible local Wan
  UMT5 text encoder file.
- `tokenizer/` can be copied from a compatible Wan Diffusers or WanVideoWrapper
  tokenizer folder.
- `scheduler/scheduler_config.json` matches the Wan Diffusers UniPC flow
  scheduler config.
- `models/VAE/wan2.1_vae.safetensors` and `models/VAE/wan2.2_vae.safetensors`
  should live in the configured VAE/model folders.

Do not require hard links or junctions for these components. Copy the small
shared config/tokenizer files into the local component base, and add large model
libraries through Settings -> Model paths.

The future "Install base components for video" button should install or verify exactly this base set. Transformer high/low files and LoRAs remain user-selectable model files, not part of the shared base.

## Maintainer Notes

Keep user-facing Wan routes separate in UI, validation, and docs:

- 5B safetensors, 14B safetensors, and GGUF are separate video routes with
  different memory, speed, and compatibility expectations.
- FP8 safetensors and GGUF should not be mixed in one high/low transformer
  pair. Treat them as different loader/runtime paths, not interchangeable file
  formats.
- Some current paths work because the compatibility layer accepts them, but
  that does not mean they are optimized. Avoid claiming speed wins without
  measured step time, VRAM/RAM, fallback count, and attention backend evidence.
- If temporal jumps, odd reference drift, or unexpected frame-to-frame behavior
  appear, investigate conditioning, latent handoff, and reference-image handling
  first before assuming the selected checkpoint is corrupt.
- Wan S2V is not the same plan as post-Wan audio. The MVP audio route is
  MMAudio after visual generation; S2V should stay framed as a separate future
  generation route.

## Engineering Priorities

Current priority is compatibility and validation, not native FP8 execution.

1. Preflight validation before generation: block known-bad setups with clear missing-file guidance.
2. Streaming tensor loading: inspect and remap tensors without loading whole Wan checkpoints into RAM when possible.
3. Universal ingestion: format detector, compatibility adapter, validation layer, runtime loader.
4. Quantization abstraction: keep FP8, BF16, GGUF, and future formats behind loader/runtime boundaries.
5. Diagnostics dashboard: show model type, quantization type, missing components, expected memory, attention backend, and compatibility status.

Deferred research: native FP8 execution layers, custom FP8 attention, CUDA kernels, and Tensor Core-specific FP8 paths.
