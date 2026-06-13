# Wan Local Components

AIWF keeps Wan generation local-only. The generation workflow must not download model files.

For Wan 2.2 high/low transformer pairs, AIWF needs these pieces:

- High-noise transformer: selected from `models/wan/Safetensor`, `models/wan/GGUF`, or another scanned Wan model folder.
- Low-noise transformer: selected from `models/wan/Safetensor`, `models/wan/GGUF`, or another scanned Wan model folder.
- Text encoder: UMT5 XXL weights, stored in the local component base under `text_encoder/model.safetensors`.
- Tokenizer: T5 tokenizer files under `tokenizer/`.
- Scheduler: `scheduler/scheduler_config.json`.
- VAE: selected from `models/VAE`; Wan commonly uses `wan_2.1_vae.safetensors`.

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

Current local source mapping:

- `text_encoder/model.safetensors` is the tuned `models/Clip/nsfw_wan_umt5-xxl_fp8_scaled.safetensors`.
- `tokenizer/` came from `F:/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper/configs/T5_tokenizer/`.
- `scheduler/scheduler_config.json` matches the Wan Diffusers UniPC flow scheduler config.
- `models/VAE/wan_2.1_vae.safetensors` and `models/VAE/wan2.2_vae.safetensors` came from `F:/ComfyUI/models/vae/`.

The future "Install base components for video" button should install or verify exactly this base set. Transformer high/low files and LoRAs remain user-selectable model files, not part of the shared base.

## Engineering Priorities

Current priority is compatibility and validation, not native FP8 execution.

1. Preflight validation before generation: block known-bad setups with clear missing-file guidance.
2. Streaming tensor loading: inspect and remap tensors without loading whole Wan checkpoints into RAM when possible.
3. Universal ingestion: format detector, compatibility adapter, validation layer, runtime loader.
4. Quantization abstraction: keep FP8, BF16, GGUF, and future formats behind loader/runtime boundaries.
5. Diagnostics dashboard: show model type, quantization type, missing components, expected memory, attention backend, and compatibility status.

Deferred research: native FP8 execution layers, custom FP8 attention, CUDA kernels, and Tensor Core-specific FP8 paths.
