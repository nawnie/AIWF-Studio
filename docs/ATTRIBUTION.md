# Third-party attribution

## Segment Anything integration

Aiwf's **Segment** tab and workflow `segment` / `inpaint` steps are inspired by and aligned with the excellent Stable Diffusion WebUI extension:

**[sd-webui-segment-anything](https://github.com/continue-revolution/sd-webui-segment-anything)**
Copyright (c) continue-revolution and contributors
License: AGPL-3.0 (see the upstream repository)

We do **not** copy the extension's A1111-specific code. Instead, Aiwf implements a native service layer that:

- Uses the same SAM checkpoint filenames and `{models}/sam/` layout recommended upstream
- Supports text-prompt â†’ box â†’ mask workflows (Grounding DINO-tiny via Hugging Face `transformers` in our build)
- Exposes masks to Studio inpaint and JSON workflow chains

Please star and support the original project if you find segmentation useful in diffusion workflows.

### SAM model downloads (upstream README)

| Model | Size | URL |
|-------|------|-----|
| sam_vit_h_4b8939.pth | 2.56 GB | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth |
| sam_vit_l_0b3195.pth | 1.25 GB | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth |
| sam_vit_b_01ec64.pth | 375 MB | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth |

Place files in `{models_dir}/sam/` without renaming (same requirement as the extension).

### Related projects (credit)

- [Segment Anything (Meta)](https://github.com/facebookresearch/segment-anything)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) â€” text-to-box detection in the original extension


## ControlNet integration

ControlNet conditioning in Aiwf is a clean-room reimplementation. The preprocessor
vocabulary (canny, depth, openpose, lineart, scribble, â€¦) follows the familiar
[sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet) naming so
existing mental models carry over. Model loading uses the diffusers
`ControlNetModel` / `StableDiffusion(XL)ControlNetPipeline` APIs.

Downloadable models (fp16 SD1.5 ControlNet-v1.1) come from
[comfyanonymous/ControlNet-v1-1_fp16_safetensors](https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors).

## Face Swap (ReActor) integration

The Face Swap tab is a clean-room reimplementation inspired by
[sd-webui-reactor](https://github.com/Gourieff/sd-webui-reactor). It uses
[insightface](https://github.com/deepinsight/insightface) for face analysis and the
`inswapper_128.onnx` model run via `onnxruntime`. The model is downloaded from the
[Gourieff/ReActor](https://huggingface.co/datasets/Gourieff/ReActor) assets repo and
stored in `{models_dir}/insightface/`.

InsightFace code is MIT licensed, but InsightFace-trained models and inswapper
series face-swap models need separate license review for anything beyond local
personal use. Keep this feature optional and user-installed unless we have
explicit permission for broader redistribution or commercial use.

Face swapping must only be used with the consent of the people depicted and in
compliance with applicable law.

## MMAudio video-audio post-processing

AIWF's VAP path can optionally call
[MMAudio](https://github.com/hkchengrex/MMAudio) after a video has been
generated. The MMAudio checkout and virtual environment live under
`engines/audio/` and are not vendored into the AIWF repo.

MMAudio code is MIT licensed. The released checkpoints are hosted on Hugging
Face under CC-BY-NC 4.0, so AIWF must treat MMAudio-backed output as
non-commercial unless the user has separate permission.

## LTX 2.3 video generation

AIWF can optionally call [Lightricks LTX-2](https://github.com/Lightricks/LTX-2)
from an isolated worker venv under `engines/ltx/`. LTX model weights and the
Gemma text encoder are user-installed assets fetched from Hugging Face, not
vendored or redistributed by AIWF:

- LTX 2.3 model repository: https://huggingface.co/Lightricks/LTX-2.3
- Gemma text encoder repository: https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized

Users are responsible for accepting and following the upstream licenses and
model terms for any downloaded LTX/Gemma assets.

## NVIDIA Video Effects / VFX SDK integration

AIWF's optional VideoFX/VSR post-processing can call locally installed NVIDIA
Video Effects / VFX SDK sample applications and runtime components when the user
configures those paths. NVIDIA describes the VFX SDK as an AI-powered video
processing SDK with features including video super resolution, relighting, and
AI green screen, powered by NVIDIA GPUs with Tensor Cores:

- NVIDIA VFX SDK User Guide: https://docs.nvidia.com/maxine/vfx/index.html
- NVIDIA Video Effects SDK System Guide: https://docs.nvidia.com/deeplearning/maxine/vfx-sdk-system-guide/index.html
- NVIDIA Broadcast SDK resources: https://www.nvidia.com/en-us/geforce/broadcasting/broadcast-sdk/resources/

AIWF does not vendor or redistribute NVIDIA SDK binaries, models, or samples.
Users must install and license NVIDIA components separately.

## SageAttention optimization research

AIWF can probe SageAttention-style attention acceleration for Wan/video work.
[SageAttention](https://github.com/thu-ml/SageAttention) is Apache-2.0, but it
is an optimization dependency, not a required feature dependency. Keep it in the
shared SDK cache or a disposable copied venv until local Windows/NVIDIA tests
prove installability, output quality, and speed for this project.
