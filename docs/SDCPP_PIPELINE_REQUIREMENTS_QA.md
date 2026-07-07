# stable-diffusion.cpp Pipeline Requirements QA

This page defines how AIWF Studio should treat the stable-diffusion.cpp backend lane during QA.

## Backend role

The sd.cpp backend is a fallback and speed lane, not a replacement for every AIWF/Diffusers pipeline.

Use sd.cpp when:

- the model is a supported single-file checkpoint or GGUF route;
- the user wants lower dependency pressure;
- the workflow can be represented cleanly as sd-cli arguments;
- a failure should stay isolated from the Diffusers runtime.

Use the existing AIWF/Diffusers pipelines when:

- the model requires Python-only pipeline glue;
- the route depends on custom model-family code already written in AIWF;
- the feature needs complex image-edit adapters not yet mapped to sd-cli;
- video/audio routing needs a richer pipeline than the current sd.cpp CLI bridge.

## Minimum QA routes

### Image

- SD 1.5 txt2img, 512x512, 6 to 20 steps.
- SDXL txt2img, 1024x1024, 6 to 30 steps.
- img2img with init image and strength.
- inpaint with init image and mask.
- ControlNet with control image and configured ControlNet model path.
- LoRA by setting `--lora-model-dir` and using the syntax supported by sd.cpp.

### Split assets

The sd.cpp profile UI maps these fields into CLI flags:

| Profile field | sd-cli flag |
|---|---|
| `clipL` | `--clip_l` |
| `clipG` | `--clip_g` |
| `clipVision` | `--clip_vision` |
| `t5xxl` | `--t5xxl` |
| `llm` | `--llm` |
| `llmVision` | `--llm_vision` |
| `diffusionModel` | `--diffusion-model` |
| `highNoiseDiffusionModel` | `--high-noise-diffusion-model` |
| `uncondDiffusionModel` | `--uncond-diffusion-model` |
| `vae` | `--vae` |
| `taesd` | `--taesd` |
| `controlNet` | `--control-net` |
| `loraModelDir` | `--lora-model-dir` |
| `tensorTypeRules` | `--tensor-type-rules` |
| `modelArgs` | `--model-args` |
| `extraSampleArgs` | `--extra-sample-args` |

These are passed through `AIWF_SDCPP_EXTRA_ARGS` for the current subprocess bridge.

## Video

Upstream sd.cpp has video support, but AIWF should keep video QA conservative until the exact model asset layout is known.

QA order:

1. Prove still-image SD1.5 and SDXL first.
2. Prove split-asset image routes.
3. Prove one small video model through direct sd-cli smoke testing.
4. Only then expose video as a normal Pro generation route.

## UI policy

- No live engine hot-swap. The backend object is created at boot.
- The profile UI saves settings; launch/restart selects the backend.
- Keep the Pro theme coherent. Do not introduce the common synthetic mint/teal palette into new sd.cpp surfaces.
- Failure messages should tell the user whether to switch to Diffusers, install/build sd-cli, or provide missing split-asset paths.

## Fallback policy

If sd.cpp fails because an asset layout or flag combination is unsupported, AIWF should:

1. record the command and error;
2. keep the output folder and logs;
3. suggest the matching Diffusers route when one exists;
4. never silently fall back mid-generation without telling the user.

Logs first. Tiny fireworks later.
