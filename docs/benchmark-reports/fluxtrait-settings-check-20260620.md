# Fluxtrait Settings Check - June 20, 2026

The downloaded Fluxtrait files are not one model family. The CivitAI model entry is:

`FLUXTRAIT [FLUX.2 Klein / FLUX / Z-Image] for Portrait & detailed skin`

Local files found under `models/flux/GGUF/`:

| File | CivitAI family | Intended settings from model page | AIWF status |
| --- | --- | --- | --- |
| `fluxtraitFLUX2KleinFLUXZ_klein9bV2Q4KM.gguf` | Flux.2 Klein 9B | Euler, CFG `1`, `10+` steps, max `15` | Routed as Flux.2 Klein |
| `fluxtraitFLUX2KleinFLUXZ_klein9bV2Q5KM.gguf` | Flux.2 Klein 9B | Euler, CFG `1`, `10+` steps, max `15` | Routed as Flux.2 Klein |
| `fluxtraitFLUX2KleinFLUXZ_klein9bV2Q6K.gguf` | Flux.2 Klein 9B | Euler, CFG `1`, `10+` steps, max `15` | Routed as Flux.2 Klein |
| `fluxtraitFLUX2KleinFLUXZ_klein9bV2Q80.gguf` | Flux.2 Klein 9B | Euler, CFG `1`, `10+` steps, max `15` | Routed as Flux.2 Klein |
| `fluxtraitFLUX2KleinFLUXZ_zImageV2GgufQ4.gguf` | ZImageTurbo | Euler, CFG `1`, `8+` steps | Routed as Z-Image |

Do not benchmark these through the old Flux.1 split route. AIWF's Flux.1 path uses `FluxPipeline` with CLIP-L, T5-XXL, and Flux VAE. Flux.2 Klein now has a separate `Flux2KleinPipeline` route with Qwen/Qwen3 components, and Z-Image now has a separate `ZImagePipeline` route.

Current local blocker:

- The GGUF diffusion files are present.
- Matching Flux2 Klein / Z-Image component stacks are not present in `models/`.
- Hugging Face access to `black-forest-labs/FLUX.2-klein-9B` returned `401 Unauthorized` without accepted-gated-model credentials.

Implemented guardrails:

- Model inventory classifies Klein and Z-Image GGUF/safetensors files as `flux2_klein` and `z_image`, not plain `flux`.
- Download categories install Klein assets under `models/flux2/...` and Z-Image assets under `models/z-image/...`.
- The backend refuses to run these routes without the matching component snapshot instead of falling back to Flux.1.
- Model profiles set CFG `1`, Euler, and family-specific step defaults before generation.

Remaining validation:

- Benchmark the quants only after the correct route and components are available.

Sources checked:

- CivitAI: https://civitai.com/models/2086049/fluxtrait-flux2-klein-flux-z-image-for-portrait-and-detailed-skin
- Flux.2 Klein 4B model card: https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
- Z-Image Turbo model card: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
- ComfyUI Flux.2 Klein note: https://blog.comfy.org/p/flux2-klein-4b-fast-local-image-editing
