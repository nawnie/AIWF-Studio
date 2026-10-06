# Qwen Image 2.1 Studio (ComfyUI backend)

Desktop app (PySide6) for **Qwen-Image 2.1** that uses a running ComfyUI as the
inference engine and wraps **ai-toolkit** / **DiffSynth-Studio** for LoRA training.
Built for the 16 GB RTX 4070 Ti SUPER ("Jarvis"); everything defaults to the
int8 ConvRot files that are known to run there without offload flags.

Why an app on top of ComfyUI instead of a new engine: ComfyUI core already has
the model's attention/KV-cache code, RGBA VAE, reference-image splicing, Fun
ControlNet and prompt-enhancer nodes, and the Comfy-Org quantized files. A
PySide6 or C++ frontend cannot make the DiT sample faster; what it adds is a
task-shaped UI (10 reference slots, paint-to-mark, alpha preview, LoRA stack,
accelerator presets, training) and validation before anything hits the GPU.

## What the app does

| Tab | Native Qwen-Image 2.1 features exposed |
| --- | --- |
| **Generate** | text to image at ~1 MP or native 2K (7 aspect ratios), CFG 1 default, negative prompt when CFG > 1, RGBA/transparent output (prompt hint button), LoRA stack, accelerator LoRAs (Turbo8 8-step, Pruna 8/5, Viggle 6, PAI Fun-Acc 4), prompt enhancer (official PE model via TextGenerate), APG+FreSca "Fix" guidance, Fun ControlNet Union control maps, live sampler preview, workflow export |
| **Edit** | up to 10 reference images (`image_1` is edited; `<image2>`… tokens), paint circles/strokes on image_1 for local edits ("the red area"), reference resolution budget, canvas = image_1 or custom, background removal / subject extraction to real alpha PNG, KV-cache device/dtype for memory-starved multi-ref edits, Fun ControlNet inpaint (your strokes become the mask), prompt enhancer that sees the references |
| **Train LoRA** | dataset scan (captions, alpha, control-folder matching), caption tools, ai-toolkit YAML with the 16 GB preset (int8 ConvRot DiT + encoder, low_vram, layer offloading, cached latents/embeddings), edit-LoRA training from 1-3 reference folders, RGBA training flag, DiffSynth official recipe as an alternative, live log + progress |
| **Settings** | ComfyUI URL, launch command with a UTF-8 environment (`PYTHONUTF8=1`, `PYTHONIOENCODING=utf-8`), model-file check against the server, trainer paths, Pro-tab bridge |

Every run is validated twice: offline against the node signatures in
`qwen21/presets.py`, then against the live `/object_info` (missing nodes,
missing model files). A `*.receipt.json` with the spec and the exact API prompt
is written next to each output.

## Install (Windows, PowerShell from the repo root)

```powershell
# app venv (PySide6 + websocket-client + Pillow + PyYAML; no torch)
.\scripts\bootstrap_qwen21.ps1

# optional: clone ai-toolkit into engines\qwen_image_2_1\ai-toolkit and build its venv (torch cu130)
.\scripts\bootstrap_qwen21.ps1 -WithAiToolkit
```

Run: `Qwen Image 2.1 Studio.bat` (repo root) or
`engines\qwen_image_2_1\.venv\Scripts\python.exe engines\qwen_image_2_1\app.py`.

ComfyUI must be **0.37.0 or newer** (TextEncodeQwenImage21 / QwenImage21Cache
landed there; Fun ControlNet on 2026-09-28). Start your F: install, not the
bundled desktop app, with a UTF-8 console; the Settings tab's *Launch ComfyUI*
button does that for a command you enter once.

## Model files (ComfyUI `models/`)

From https://huggingface.co/Comfy-Org/Qwen-Image-2.1 (Qwen Research License):

| folder | file | GB | 16 GB pick |
| --- | --- | --- | --- |
| diffusion_models | `qwen_image_2.1_int8_convrot.safetensors` | 7.26 | **yes** |
| diffusion_models | `qwen_image_2.1_bf16.safetensors` | 14.23 | no (offloads) |
| text_encoders | `qwen3vl_8b_int8_convrot.safetensors` | 9.35 | **yes** |
| text_encoders | `qwen3vl_8b_w4a8.safetensors` | 6.31 | headroom option |
| text_encoders | `qwen3vl_8b_bf16.safetensors` | 17.53 | no |
| vae | `qwen_image_2.1_vae_bf16.safetensors` | 0.68 | **yes** |
| text_encoders | `qwen3.5_9b_qwen_image_2.1_pe_t2i.int8_convrot.safetensors` | 9.47 | optional (prompt enhancer) |
| text_encoders | `qwen3.5_9b_qwen_image_2.1_pe_i2i.int8_convrot.safetensors` | 9.47 | optional (edit enhancer) |
| model_patches | `qwen_image_2.1_fun_controlnet_union_int8_convrot.safetensors` | 3.78 | optional (ControlNet) |

Accelerator LoRAs go in `models/loras` (links and settings in `qwen21/presets.py`):
Turbo8 `chriswritescode/Turbo8-LoRA-Qwen-Image-2.1` (recommended), Pruna
`NidAll/pruna-image-2.1-comfyui-loras`, Viggle `Viggle/Qwen-Image-2.1-viggle-turbo`,
Alibaba PAI `alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs` (experimental in ComfyUI).

### Why int8 ConvRot and not fp8 / GGUF / Nunchaku

* Unsloth's measurements on 2.1: INT8 LPIPS 0.064 / SSIM 0.936 vs FP8 LPIPS 0.112 / SSIM 0.899, at the same ~7.2 GB. fp8 was not faster in their tests. So fp8 only loses quality.
* The text encoder is the real VRAM problem (17.5 GB bf16). ComfyUI loads it, encodes, and evicts it before the DiT runs, so the int8 encoder (9.35 GB) and int8 DiT (7.26 GB) never need to coexist. Measured on a 4060 Ti 16 GB with exactly these files: ~20 s per 1024² image at 25 steps, ~60 s per edit, peak ~15 GB, no offload flags.
* GGUF Q4_K_M (4.6 GB) is for 8-12 GB cards; it needs the ComfyUI-GGUF custom nodes plus a patched Qwen3-VL loader and one publisher reports tensor-shape problems with Q8_0. Not needed at 16 GB.
* There is no official Nunchaku/SVDQuant build for 2.1 (the repo's `engines/qwen_nunchaku` sidecar is for the old 20B Qwen-Image Lightning).
* For speed, use a distilled LoRA on top of the int8 base: Turbo8 (8 steps, no CFG) keeps editing, up to 10 references and RGBA working; dense small text degrades. Keep 25-40 steps for finals.

Settings → *Check which files the server sees* prints this table with present/missing per file.

## Sampling defaults (from the official templates and docs)

* 25 steps, CFG 1.0, euler / simple, denoise 1.0; 40 steps for finals; CFG 2 for dense prompts (negative prompt only matters above 1); CFG 5+ degrades.
* Reference images: `resolution` 1024 budget by default. Image_1's size decides the canvas cost (12 MP at `0` is ~20x slower than 1 MP).
* Transparency: say so in the prompt ("This is an RGBA image with transparency…", or "Remove the background, and output a PNG image"); the VAE carries alpha and the PNG saver keeps it.
* Local edits: paint a mark on image_1 and reference it ("change only the red area"); the model reads painted annotations natively.
* KV cache (`QwenImage21Cache`): `auto`/`default` is lossless; `cpu` + `int8` frees VRAM for 6-10 reference edits at little speed cost.

## Training a LoRA (16 GB)

GUI: Train tab → dataset folder (images + same-name `.txt`), optional reference
folders for an **edit LoRA**, 16 GB preset, *Start training*. CLI:

```powershell
# character/style LoRA (ai-toolkit, int8 ConvRot DiT + encoder, layer offload, rank 16, 2000 steps)
python engines\qwen_image_2_1\train_lora.py --name shawn_v1 --dataset D:\data\shawn --trigger ohwx_shawn --prepend-trigger

# edit LoRA: targets in --dataset, references (same basenames) in --control-dir (up to 3)
python engines\qwen_image_2_1\train_lora.py --name outfit_swap --dataset D:\data\targets --control-dir D:\data\refs

# transparent-asset LoRA
python engines\qwen_image_2_1\train_lora.py --name cutouts --dataset D:\data\rgba_pngs --rgba

# write the YAML only / use the official DiffSynth recipe
python engines\qwen_image_2_1\train_lora.py --name t --dataset D:\data\x --write-config-only
python engines\qwen_image_2_1\train_lora.py --trainer diffsynth --name t --dataset D:\data\x
```

What the 16 GB preset does (ai-toolkit `arch: qwen_image_2`): `quantize`/`quantize_te`
with `convrot8`, `low_vram`, `layer_offloading` 100 % for DiT and encoder,
`cache_latents_to_disk`, `cache_text_embeddings` (encoder unloaded after caching),
batch 1, rank/alpha 16, 1024 buckets, adamw8bit, lr 1e-4, flowmatch with `shift`
timesteps, no sampling during training. A published 12 GB RTX 4070 run with these
settings took ~3.5 h for 2000 steps on 31 images; expect faster on the 4070 Ti SUPER.
Trigger words are baked into captions because cached embeddings ignore
`trigger_word`. The resulting `.safetensors` loads with the core LoRA node;
copy it to `ComfyUI/models/loras` and refresh.

DiffSynth-Studio is the Qwen team's own trainer (bf16, `--lora_base_model dit`,
rank 32). It is the reference recipe; on 16 GB it needs gradient-checkpoint
offload and an fp8 text encoder and is not verified here.

Caveat: musubi-tuner's Qwen-Image 2.1 support (PR #1145) was still unmerged on
2026-10-04, so it is not wired in.

## AIWF Studio Pro bridge

The Pro React tab *Qwen Image Editor* polls `127.0.0.1:7865/api/health` and
iframes `/`. While this app runs it answers that health check and shows the
latest outputs on that page, so the Pro tab reports "Desktop editor connected".
Disable it in Settings if the port is taken.

## Files

```
engines/qwen_image_2_1/
  app.py                  GUI entry point
  train_lora.py           CLI trainer wrapper (JSONL events with --jsonl)
  requirements.txt        PySide6, websocket-client, Pillow, PyYAML
  qwen21/presets.py       model files, resolutions, sampler presets, accelerators, node signatures
  qwen21/workflow_builder.py  GenerationSpec -> ComfyUI API prompt, validator, export
  qwen21/comfy_client.py  HTTP/WS client (upload, queue, progress, history, view)
  qwen21/training/        dataset scan, ai-toolkit YAML, DiffSynth command, subprocess runner
  qwen21/ui/              PySide6 widgets and tabs
  workflows/              16 validated reference API workflows (+ .spec.json)
  prompts/                prompt-enhancer system prompts (MIT, from Comfy-Org templates)
scripts/bootstrap_qwen21.ps1, scripts/export_qwen21_workflows.py
tests/individual_tests/test_qwen21_app.py
```

Sources are listed in `docs/QWEN_IMAGE_2_1.md`.
