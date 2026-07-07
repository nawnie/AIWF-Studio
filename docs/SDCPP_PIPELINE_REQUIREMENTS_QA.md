# stable-diffusion.cpp Pipeline Requirements QA

This document defines the QA lane for the AIWF `sdcpp` backend profile.

## Backend routing rule

AIWF should choose the backend at Pro launch time:

```text
AIWF Pro UI -> Pro API -> GenerationService -> backend profile
```

Supported profile ids:

- `diffusers`
- `sdcpp`
- `onnx`

The backend object is created at boot. Switching profile requires relaunching Pro.

## sd.cpp route matrix

| Route | QA status | Notes |
|---|---:|---|
| SD1.5 txt2img | Ready for smoke | Single-file `.safetensors`, `.ckpt`, `.pt`, `.pth`, or `.gguf`. |
| SDXL txt2img | Ready for smoke | First 1024 test should use low steps. |
| img2img | Wired | Uses init image and strength. |
| inpaint | Wired | Uses init image and mask. |
| LoRA | Directory-wired | `--lora-model-dir` is exposed; prompt syntax still follows sd.cpp behavior. |
| ControlNet | Wired for QA | Path and image handoff are exposed. |
| Flux/Qwen split assets | Argument-mapped | Requires component paths in the sd.cpp QA profile. |
| Video | Experimental smoke | Use sd.cpp video smoke route until main video-tab parity is proven. |

## Split asset arguments

The sd.cpp QA profile maps profile fields into CLI flags:

| Profile field | CLI flag |
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

## Fallback policy

Use AIWF native Diffusers pipelines when:

- sd.cpp does not support the model family.
- A split-asset model requires component mapping that is not known yet.
- Video output format or motion behavior differs from the AIWF route.
- A workflow needs AIWF-only offload, overlay, or pipeline receipts.

Use sd.cpp when:

- A single-file SD/SDXL model smoke test passes.
- Lower dependency surface matters.
- CUDA/Vulkan/CPU fallback is being compared.
- GGUF route testing is the goal.

## QA order

1. Diffusers profile launches unchanged.
2. sd.cpp profile launches and reports `StableDiffusionCppBackend`.
3. SD1.5 512 smoke test.
4. SDXL 1024 smoke test.
5. Inpaint smoke test.
6. LoRA directory smoke test.
7. ControlNet smoke test.
8. Flux/Qwen split-asset smoke test.
9. Video smoke test.
10. Compare logs and history between Diffusers and sd.cpp outputs.

Logs first. Panic later.
