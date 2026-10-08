# Civitai Resource Support

This page tracks what AIWF Studio can currently use or train from common Civitai resource categories. Civitai model type is catalog metadata; it does not mean every item is a standalone generator or a trainable target. Civitai's `supportsGeneration` field describes its own generation service, not local AIWF compatibility.

The category list below follows the current Civitai developer and generator documentation. Civitai's live `ModelType` enum remains authoritative and may change. This is a Studio route map, not a promise to accept every Civitai upload or file format.

## Capability map

| Civitai resource type | Generate or use in Studio | Train in Studio | Local verification state |
| --- | --- | --- | --- |
| Checkpoint | Partial: local image and video families use separate loaders and accepted formats. | Partial: ED2 image full training and Kohya image LoRA training are available from the legacy Training tab for supported bases. | Validate each family and exact checkpoint. Qwen Image 2.1 has no local generation smoke. |
| LORA / LoCon / DoRA | Partial: classic SD-family adapters and Wan stage adapters are family-specific; several new transformer-image families remain blocked. | Kohya supports SD 1.x, SDXL, and Flux. | Validate each base, adapter format, and Wan stage. Qwen Image 2.1 LoRA training is not wired. |
| TextualInversion | Limited to compatible embedding loaders; a Civitai download is not auto-imported as a runnable asset. | No Studio textual-inversion trainer route. | Validate each loader and base family. |
| Hypernetwork | No native route. | No trainer route. | Not tested on this PC; not implemented. |
| AestheticGradient | No native route. | No trainer route. | Not tested on this PC; not implemented. |
| Controlnet | Partial: compatible SD 1.x / SDXL image conditioning routes. | No trainer route. | Validate the exact control model, base, and preprocessor. |
| VAE | Partial: external VAE selection on compatible classic image routes; newer families use their own components. | No trainer route. | Validate architecture and base compatibility. |
| MotionModule | No generic Civitai MotionModule loader. Wan, LTX, and Sana Video use separate native routes. | No MotionModule or video-LoRA trainer route. | Not tested on this PC; not implemented. |
| Upscaler | Partial: selected image upscale and restoration workers run as post-processing. | No trainer route. | Validate the exact worker and asset format. |
| Poses | Pose data may be conditioning input for a compatible route; it is not a standalone generator. | Not a model-training target. | Validate the consuming route and pose format. |
| Wildcards | Prompt assets are not model weights; Civitai import and wildcard expansion are not wired. | Not a model-training target. | Not tested on this PC; import route is not implemented. |
| Workflows | AIWF has a native workflow builder, but it does not import and execute arbitrary Civitai packages. | Not a model-training target. | Civitai workflow import is not implemented or tested on this PC. |
| Detection | Some local detection and segmentation features exist; Civitai Detection packs have no universal loader. | No Civitai Detection trainer route. | Validate the exact worker and model format. |
| Other | No generic loader. Identify its architecture, role, and format first. | No generic trainer. | Not tested on this PC; classify and validate before enabling. |

The same information is served at `GET /api/pro/civitai/support` and shown beside the model-family view in Pro.

## Qwen Image 2.1

AIWF now recognizes a `QwenImage21Pipeline` Diffusers folder, selects a separate Qwen Image 2.1 inference path, and provides text-to-image plus image-conditioned editing calls. Its profile uses the upstream defaults of 40 steps and true CFG 1.0, preserves the pipeline's native block-causal attention (including bypassing the shared Sage SDPA wrapper), aligns output dimensions to 32-pixel blocks, and keeps classic Qwen LoRA application disabled pending compatibility evidence. Live previews remain disabled until the packed-latent decoder is implemented. The classic `denoising_strength` slider is not an equivalent Qwen 2.1 control and is not used by its image-conditioned route.

This is implemented code, not validated support on the current PC. The Studio environment has Diffusers 0.39.0 and does not expose `QwenImage21Pipeline`. The app reports the missing class as a clear runtime block. No model was downloaded and no GPU generation was run. The current GPU query reported 16,376 MiB total and 2,266 MiB free; the Qwen 2.1 full model card lists a 33.1 GB snapshot, so local memory placement and performance remain unverified.

The upstream Diffusers main-branch docs document Qwen Image 2.1 inference and image-conditioned editing. The Diffusers DreamBooth LoRA example is for `Qwen/Qwen-Image`; it does not establish Qwen Image 2.1 training compatibility. Therefore Qwen Image 2.1 training and runtime LoRA remain unavailable in Studio until a compatible trainer/adapter path is implemented and validated.

## Training coverage still needed for the combined app

- The available ED2 and Kohya image-trainer surfaces live in the legacy Gradio Training tab, not the primary Pro React interface.
- Studio has Wan/LTX/Sana video generation routes, but no native video full-training or video-LoRA training lane.
- There is no single trainer that accepts every Civitai checkpoint, adapter, sidecar, or utility resource.
- Qwen Image 2.1 requires the upstream Diffusers pipeline class before even inference can run in this installation.

These are implementation gaps, not PC-only validation gaps. The UI should keep them distinct from routes that are implemented but not tested on this machine.

## Sources and code paths

- [Civitai Models API](https://github.com/civitai/civitai-developer-docs/blob/main/site/reference/models.md): model `type` taxonomy and `supportsGeneration` semantics.
- [Civitai generator resource notes](https://github.com/civitai/civitai/blob/main/docs/features/generator.md): documented resource categories and the narrower set accepted by Civitai's own generator flow.
- [Qwen Image 2.1 Diffusers API](https://huggingface.co/docs/diffusers/main/api/pipelines/qwenimage21): pipeline class, image-conditioning API, 40-step and no-guidance defaults, and native attention details.
- [Qwen Image 2.1 model card](https://huggingface.co/Qwen/Qwen-Image-2.1): model files, model-card size, and license.
- [Diffusers Qwen Image LoRA example](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_qwen.md): Qwen Image LoRA training example; this does not cover 2.1.
- Local route policy: `agents_runtime.md`, `docs/agent-workflows/MODEL_FAMILY_ATTENTION.md`, and `docs/LORA_PIPELINE_STRATEGY.md`.
- Runtime implementation: `aiwf/infrastructure/diffusers/backend.py`, `aiwf/core/model_profile.py`, `aiwf/services/pipeline_preflight.py`, and `aiwf/services/model_family_support.py`.
