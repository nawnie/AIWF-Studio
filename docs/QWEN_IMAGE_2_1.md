# Qwen-Image 2.1 on a 16 GB card: research notes and decisions

Compiled 2026-10-06 for the `engines/qwen_image_2_1` app. Facts below are from
the sources listed at the end; numbers we could not reproduce locally are marked
as published claims.

## The model

* Released 2026-09-20 by the Qwen team under the Qwen Research License (non-commercial without a separate license).
* 7B single-stream DiT, 32 layers, mixed-granularity attention with prefix KV-cache reuse; replaces the 20B MMDiT of Qwen-Image / Qwen-Image-Edit-25xx.
* Text encoder: Qwen3-VL 8B (vision-language, so references are *seen* as well as spliced in as VAE latents).
* VAE: 64 latent channels, 16× spatial compression, RGBA. Transparency is native output, not a matting pass.
* One checkpoint does text-to-image and editing. Editing accepts up to 10 reference images (ComfyUI wires 16 slots), local edits via painted circles/areas or masks, identity preservation, subject extraction, native 2K (2048², 2400×1792, 2528×1696, 2752×1536 and portrait variants).
* Diffusers class `QwenImage21Pipeline`, `image=` takes a list of PIL images; 40 steps default. DiffSynth-Studio has `QwenImage21Pipeline` with `edit_image=[...]`, `cfg_scale`, 7 GB minimum with disk offload.

## ComfyUI support (core, no custom nodes)

* ComfyUI ≥ 0.37.0 (PR #16400). Nodes: `TextEncodeQwenImage21` (clip, prompt, negative_prompt, vae, resolution, images.image_1..16; outputs positive, negative, latent sized from image_1), `QwenImage21Cache` (KV cache device auto/gpu/cpu/off, dtype default/int8/int4), standard `UNETLoader` / `CLIPLoader(type=qwen_image)` / `VAELoader` / `KSampler` / `VAEDecode` / `SaveImageAdvanced`.
* Fun ControlNet Union (Alibaba PAI, converted by kijai, merged 2026-09-28): `ModelPatchLoader` + `ZImageFunControlnet` ("Apply Fun ControlNet"; model, model_patch, vae, strength, image, inpaint_image, mask, start/end percent). Eight control types + inpainting in one 3.78 GB int8 file.
* Prompt enhancers: Qwen3.5-9B fine-tunes (`pe_t2i`, `pe_i2i`) loaded through `CLIPLoader` and run with the `TextGenerate` node; the official templates ship the system prompts (MIT) which we bundle in `engines/qwen_image_2_1/prompts/`.
* Official templates: text-to-image (1024², 25 steps, CFG 1, euler/simple), image edit (two references, `<image1>`/`<image2>` tokens, KV cache auto), background removal (prompt "Remove the background, and output a PNG image").
* Distilled few-step LoRAs load with the core `LoraLoaderModelOnly`; those with fixed sigma schedules need `ManualSigmas` + `SamplerCustom` (both core).

## Quantization decision for the RTX 4070 Ti SUPER (16 GB)

| component | pick | size | why |
| --- | --- | --- | --- |
| DiT | `qwen_image_2.1_int8_convrot` | 7.26 GB | Unsloth: LPIPS 0.064 / SSIM 0.936 vs bf16; fp8 at the same size is 0.112 / 0.899 and was not faster |
| text encoder | `qwen3vl_8b_int8_convrot` | 9.35 GB | encoded once then evicted; measured peak ~15 GB at 1024² on a 4060 Ti 16 GB with no offload |
| text encoder (headroom) | `qwen3vl_8b_w4a8` | 6.31 GB | for 6-10 references or native 2K edits; quality cost of 4-bit VL encoding is unpublished |
| VAE | `qwen_image_2.1_vae_bf16` | 0.68 GB | only option, carries alpha |
| ControlNet | `fun_controlnet_union_int8_convrot` | 3.78 GB | half the bf16 patch |

Rejected: bf16 DiT (14.23 GB, forces offload on 16 GB); fp8 (worse than int8 at equal size, no speed win); GGUF Q4_K_M (4.6 GB, quality loss, needs ComfyUI-GGUF plus a patched Qwen3-VL text-encoder loader, Q8_0 shape issues reported); Nunchaku/SVDQuant (no official 2.1 build; the existing `engines/qwen_nunchaku` is for the old 20B Lightning model).

Published 16 GB and 12 GB numbers with int8 files: 4060 Ti 16 GB — ~20 s per 1024² image at 25 steps, ~60 s per edit, 13-14 GB typical, 15 GB peak. RTX 4070 12 GB — ~10 s at 1024², ~90 s at 2048². Treat these as the envelope until a receipt from Jarvis replaces them.

## Speed: distilled LoRAs

| adapter | steps | sampler path | notes |
| --- | --- | --- | --- |
| Turbo8 (`chriswritescode/Turbo8-LoRA-Qwen-Image-2.1`, 1.36 GB, rank 128) | 8, CFG 1 | KSampler euler/simple + ModelSamplingFlux max_shift 0.6935 base_shift 0.5 | covers edit, 10 refs, RGBA, extraction; dense text degrades; community ranked it above Viggle 6-step for edits |
| Pruna 8 / 5 (`NidAll/pruna-image-2.1-comfyui-loras`, 0.34 GB) | 8 / 5, CFG 1 | ManualSigmas + SamplerCustom, exact sigmas required | v0.1, 1K training, ≤3 references |
| Viggle Turbo v0.3 (`Viggle/Qwen-Image-2.1-viggle-turbo`, r128 0.7 GB) | 6, CFG 1 | sigmas 1, .9375, .875, .75, .5, .25 | official path uses Viggle's custom node (unmerged LoRA, shift_terminal null); core LoRA node is an approximation |
| Alibaba PAI Fun-Acc (`alibaba-pai/Qwen-Image-2.1-Fun-Acc-LoRAs`, 0.35 GB) | 4, CFG 1 | sigmas 1, .9169867, .7861579, .549491, 0 | official PDD distill for Diffusers/VideoX-Fun; ComfyUI key layout may need a conversion |

Community quality LoRAs worth knowing: e-n-v-y "Qwen-Image 2.1 Fix" (washed-out colors / blotching; its workflow uses CFG 3, 20 steps, seeds_2 + sgm_uniform, APG 1/10/0.3, FreSca 1/2/8 and a specific negative prompt, all available as a preset in the app), ausboss "Edit Consistency" and "Outfit Swap Consistency" (reduce drift during edits).

## Environment on Jarvis

* ComfyUI from the F: install, not the bundled desktop app, started with `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8` (Settings → Launch ComfyUI does this). Keep ComfyUI current (≥ 0.37; 0.39.0 at time of writing).
* App venv: `scripts/bootstrap_qwen21.ps1` → PySide6, websocket-client, Pillow, PyYAML. No torch in the app venv; all inference is ComfyUI's.
* Training venv: ai-toolkit's own (`python -m venv venv; pip install torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130; pip install -r requirements.txt`), created by `bootstrap_qwen21.ps1 -WithAiToolkit`. ComfyUI and training should not run at the same time on 16 GB.

## LoRA training

* **ai-toolkit** (`arch: qwen_image_2`): one arch for T2I and edit; a dataset `control_path` (1-3 folders, same basenames) turns it into reference-guided edit training. Quantization `convrot8` for DiT and text encoder, `low_vram`, `layer_offloading` (100 % both), cached latents and text embeddings. A 12 GB RTX 4070 run (31 images at 1024, rank 16, lr 1e-4, 2000 steps, adamw8bit, flowmatch/shift) took ~3.5 h. `name_or_path` defaults to the Comfy-Org repack; the extension pulls configs/processor from `Qwen/Qwen-Image-2.1`. RGBA training via `model_kwargs.rgba`. Output is a ComfyUI-loadable LoRA.
* **DiffSynth-Studio**: official `examples/qwen_image_21/model_training/train.py` (`--lora_base_model dit --lora_rank 32 --use_gradient_checkpointing`, metadata.csv `image,prompt`; edit mode adds `--data_file_keys image,edit_image --extra_inputs edit_image` with metadata.json). Low-VRAM flags: `--use_gradient_checkpointing_offload`, `--fp8_models` on the frozen text encoder, `--quant_options ...:bitsandbytes_nf4`, `--initialize_model_on_cpu`. bf16 trainable DiT is 14 GB, so 16 GB is tight; unverified here.
* **musubi-tuner**: Qwen-Image 2.1 support exists as PR #1145 (new `qwen_image_21_*` scripts, loads ComfyUI single-file checkpoints) but was unmerged on 2026-10-04; not wired in.

## Sources

* Model card / repo: https://huggingface.co/Qwen/Qwen-Image-2.1 · https://github.com/QwenLM/Qwen-Image-2.1
* ComfyUI files + templates: https://huggingface.co/Comfy-Org/Qwen-Image-2.1 · https://github.com/Comfy-Org/workflow_templates (image_qwen_image_2_1_t2i / _image_edit / _background_removal) · https://blog.comfy.org/p/qwen-image-21-in-comfyui-open-weight · https://docs.comfy.org/tutorials/image/qwen/qwen-image-2-1
* ComfyUI core sources read for node schemas: comfy_extras/nodes_qwen.py, nodes_model_patch.py, nodes_custom_sampler.py, nodes_model_advanced.py, nodes_apg.py, nodes_fresca.py, nodes_textgen.py, nodes_images.py, nodes.py (0.39.0); Fun ControlNet PR https://github.com/Comfy-Org/ComfyUI/pull/16519
* Quantization: https://unsloth.ai/docs/models/qwen-image-2.1 · https://talkaiwith.substack.com/p/qwen-image-21-every-quantisation · https://github.com/alesha-pro/tools/blob/main/qwen-image-2.1/LOW-VRAM.md · https://github.com/wildminder/awesome-qwen-image
* 16 GB / 12 GB reports: https://huggingface.co/Qwen/Qwen-Image-2.1/discussions/35 (4060 Ti 16 GB) · https://github.com/kuraneko1/qwen21-fast-comfyui (RTX 4070 12 GB) · https://smeltcore.com/recipes/qwen-image-2-1-on-any-rtx-4070-board-int8-comfyui-install-in-12-gb-2k-and-image-editing/
* Accelerators: https://huggingface.co/chriswritescode/Turbo8-LoRA-Qwen-Image-2.1 · https://comfyui-wiki.com/en/news/2026-09-27-qwen-image-2-1-turbo8-lora · https://comfyui-wiki.com/en/news/2026-09-23-pruna-qwen-image-2-1 · https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo · https://comfyui-wiki.com/en/news/2026-09-24-qwen-image-2-1-fun-acc-lora · https://comfyui-wiki.com/en/news/2026-09-23-qwen-image-2-1-fix
* ControlNet: https://comfyui-wiki.com/en/news/2026-09-24-qwen-image-2-1-fun-controlnet-union
* Training: https://github.com/ostris/ai-toolkit (extensions_built_in/diffusion_models/qwen_image_2, config/examples) · https://note.com/sepiablue/n/n9372886b9e5d (12 GB ai-toolkit run) · https://github.com/modelscope/DiffSynth-Studio (examples/qwen_image_21) · https://diffsynth-studio-doc.readthedocs.io/en/latest/Model_Details/Qwen-Image-2.1.html · https://github.com/kohya-ss/musubi-tuner/pull/1145
