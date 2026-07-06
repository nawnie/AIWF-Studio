# AIWF Studio model family support matrix

This document is generated from the implementation patch, not from marketing copy. The patch scanned the extracted archive and merged v9 visual polish before adding this matrix.

## Source pass

- Text files indexed: **546**
- Source/document lines indexed: **142,835**
- Heavy model weights are **not** loaded by this matrix. Runtime status comes from the existing `pipeline_readiness` ledger.
- Endpoint: `GET /api/pro/model-families`
- UI: left rail **Families**

## Precision vocabulary

`FP32`, `FP16`, `BF16`, `FP8`, `INT8`, `NF4`, `FP4`, `NVFP4`, `INT4`, `Q2_K`, `Q3_K_S`, `Q3_K_M`, `Q3_K_L`, `Q4_0`, `Q4_1`, `Q4_K_S`, `Q4_K_M`, `Q5_0`, `Q5_K_S`, `Q5_K_M`, `Q6_K`, `Q8_0`, `IQ2`, `IQ3`, `IQ4`

## Family table

| Family | Status | Storage | Precision/quant notes | Loader routes | Main gaps |
| --- | --- | --- | --- | --- | --- |
| **Stable Diffusion 1.5** | `supported` | .safetensors, .ckpt, .pt, Diffusers folder | FP32 (runtime), FP16 (supported), FP8 storage (optional) | diffusers (supported), inpaint (supported), controlnet (supported) | Quantized single-file SD checkpoints beyond FP8 storage are not a dedicated route. |
| **Stable Diffusion XL** | `supported` | .safetensors, .ckpt, .pt, Diffusers folder | FP32 (runtime), FP16 (supported), BF16 (runtime), FP8 storage (optional) | diffusers (supported), sdxl-refiner (supported), inpaint (supported) | SD1.5 embeddings/LoRAs are skipped or should be labeled incompatible for SDXL. |
| **Stable Diffusion 3.5** | `supported-gated` | .safetensors, Diffusers folder | BF16 (preferred), FP16 (fallback), FP32 (runtime) | diffusers (supported), inpaint (supported) | SD3.5 Large single-file checkpoints need gated config files cached or account access. |
| **Flux / Flux Fill / Flux Kontext** | `supported-experimental-quants` | .safetensors transformer, .gguf transformer, FluxKontext Diffusers folder | BF16 (preferred), FP16 (fallback), FP8 (partial), NF4 (supported), FP4 (supported), GGUF Q4/Q5/Q8 (experimental) | flux (supported), flux-fill (supported), flux-kontext (folder-only) | Known bad Flux FP8/GGUF-NF4 assets are blocked from normal selection.<br>Kontext is folder-only, not raw single-file. |
| **Flux.2 Klein** | `experimental` | .safetensors transformer, .gguf transformer, Diffusers folder | BF16 (preferred), FP16 (fallback), FP8 (runtime), NF4 (text-encoder), GGUF Q4/Q5/Q8 (experimental), INT8 (missing) | flux2-klein (experimental) | Needs newer Diffusers/Transformers stack.<br>int8/convrot is listed as missing until loader support and receipts exist. |
| **Z-Image** | `experimental-blocked-on-windows-gguf` | .safetensors transformer, .gguf transformer, Diffusers folder | BF16 (preferred), FP16 (fallback), NF4 (text-encoder), GGUF Q4/Q5/Q8 (platform-limited), FP8 (candidate) | z-image (experimental) | Z-Image GGUF is blocked on Windows in model_blocks.py due fused kernel availability and VRAM paging. |
| **Qwen Image / Nunchaku** | `partial` | Diffusers folder, Nunchaku .safetensors transformer | BF16 (preferred), FP16 (fallback), INT4 (supported-sidecar), GGUF (missing) | qwen-image (supported-when-folder-installed), qwen-nunchaku (sidecar) | LLM/VL GGUF rows remain metadata-only until a local worker/API route exists. |
| **Sana / Sana Sprint** | `supported-smoked` | Diffusers folder with model_index.json | BF16 (preferred), FP16 (fallback), FP8/int8/4bit (not-a-current-route) | sana (supported-smoked) | Requires full folder, not loose single-file transformer. |
| **Sana Video** | `supported-smoked-silent` | Diffusers folder | BF16 (preferred), FP16 (fallback), bitsandbytes (optional) | sana-video (supported-smoked) | Generated audio is not part of the main smoke; use audio/mux post-process. |
| **Wan Video** | `supported-plus-sidecars` | Diffusers folder, .safetensors transformer, .gguf high/low transformers | BF16 (preferred), FP16 (fallback), FP8 (supported-experimental), GGUF Q4/Q5 (experimental-supported), GGUF Q3/Q6 (metadata-only) | wan-fast-5b (supported), wan-high-low-fp8 (experimental), wan-gguf (experimental) | T2V 1.3B, Animate, and Fun-Control/control are explicitly unsupported by the current I2V route. |
| **LTX Video** | `partial-supported` | .safetensors checkpoint, HF-shaped Gemma folder, Gemma GGUF metadata probe, T5XXL safetensors | BF16 (preferred), FP16 (fallback), FP8 (supported-smoked), FP8 scaled-mm (configured), NVFP4/FP4 (blocked), Gemma Q3_K_M GGUF (blocked-probe-only) | ltx-2b-diffusers (supported-smoked), ltx-one-stage-hf-gemma (supported-smoked), ltx-one-stage-heretic-gguf (blocked-probe-only) | Native Gemma GGUF cannot generate until a backend returns every hidden-state layer and attention mask.<br>BF16 22B on Windows is blocked unless explicitly retested. |
| **ONNX Image** | `blocked-until-folder` | ONNX folder | FP16/FP32 (provider-dependent), INT8 (not-declared) | onnx (blocked-cleanly) | models/onnx folder and provider-specific receipts required. |
| **LLM / VL GGUF** | `metadata-only` | .gguf | Q2/Q3/Q4/Q5/Q6/Q8 (metadata-only), FP16/BF16 (metadata-only) | llm-vl-worker (missing) | Keep as metadata-only until a GGUF worker/API route exists. |

## Loader evidence modules

### Stable Diffusion 1.5

Classic Diffusers single-file/folder image route with txt2img, img2img, inpaint, embeddings, ControlNet, LoRA, and optional FP8 UNet storage.

Evidence modules:
- `aiwf.infrastructure.diffusers.backend`
- `aiwf.infrastructure.diffusers.model_arch`

### Stable Diffusion XL

SDXL base/refiner/inpaint support with single-file or folder loading, dual-encoder embeddings, LoRA, ControlNet, and optional FP8 UNet storage.

Evidence modules:
- `aiwf.infrastructure.diffusers.backend`
- `aiwf.infrastructure.diffusers.model_arch`

### Stable Diffusion 3.5

SD3.5 Diffusers route with txt2img/img2img/inpaint classes and explicit gated-config access checks for Large single-file checkpoints.

Evidence modules:
- `aiwf.web.pro_api`
- `aiwf.infrastructure.diffusers.backend`

### Flux / Flux Fill / Flux Kontext

Single-transformer Flux route with shared CLIP-L/T5/AE sidecars, GGUF support, bitsandbytes 4-bit safetensors loader, Flux Fill inpaint, and Kontext folder route.

Evidence modules:
- `aiwf.infrastructure.diffusers.backend`
- `aiwf.infrastructure.diffusers.flux_bnb_loader`
- `aiwf.infrastructure.quant.bnb_nf4_format`

### Flux.2 Klein

Flux.2 Klein route uses new Diffusers Flux2KleinPipeline when available, single transformer or folder, Qwen3 text encoder components, NF4 encoder fallback, and GGUF transformer loading.

Evidence modules:
- `aiwf.infrastructure.diffusers.backend`
- `aiwf.infrastructure.diffusers.model_arch`

### Z-Image

Z-Image route uses ZImagePipeline, ZImageTransformer2DModel, component folder, Qwen3 text encoder with NF4 option, and GGUF transformer loading when platform kernels allow it.

Evidence modules:
- `aiwf.infrastructure.diffusers.backend`
- `aiwf.infrastructure.diffusers.model_blocks`

### Qwen Image / Nunchaku

Full Qwen Image Diffusers folder route plus isolated Qwen Nunchaku Lightning sidecar runtime for SVDQ-int4 single transformer.

Evidence modules:
- `aiwf.services.qwen_nunchaku`
- `aiwf.infrastructure.diffusers.backend`
- `aiwf.services.pipeline_preflight`

### Sana / Sana Sprint

Sana and Sana Sprint full Diffusers-folder image routes, with bounded Sana Sprint smoke evidence in the QA matrix.

Evidence modules:
- `aiwf.infrastructure.diffusers.backend`
- `aiwf.services.pipeline_registry`

### Sana Video

SANA-Video 2B 480p Diffusers route with quantization/tiling settings and silent MP4 smoke evidence; audio is a post-process lane.

Evidence modules:
- `aiwf.services.sana_video`
- `aiwf.core.domain.sana_video`
- `aiwf.services.pipeline_preflight`

### Wan Video

Wan fast 5B I2V plus experimental high/low FP8 and GGUF model-pair routes with explicit VAE, text encoder, LoRA, offload, sampler, and sigma controls.

Evidence modules:
- `aiwf.core.domain.wan`
- `aiwf.services.wan`
- `aiwf.infrastructure.wan.pipeline`
- `aiwf.services.wan_models`

### LTX Video

LTX 2B Diffusers route and LTX 2.3 isolated worker route with HF/converted Gemma support; native Gemma GGUF and FP4/NVFP4 remain blocked until a hidden-state-capable backend exists.

Evidence modules:
- `aiwf.core.domain.ltx`
- `aiwf.services.ltx`
- `aiwf.services.ltx_diffusers`
- `aiwf.services.pipeline_preflight`

### ONNX Image

Optional ONNX image route with provider preflight for CUDA/DirectML/CPU; blocked until the expected model folder exists.

Evidence modules:
- `aiwf.services.pipeline_preflight`
- `aiwf.infrastructure.onnx`

### LLM / VL GGUF

GGUF LLM/VL assets are inventoried and can support future chat/VL or text-encoder probes, but no native worker/API route is complete in this code pass.

Evidence modules:
- `aiwf.infrastructure.model_header`
- `aiwf.services.pipeline_readiness`

