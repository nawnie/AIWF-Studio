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

Face swapping must only be used with the consent of the people depicted and in
compliance with applicable law.
