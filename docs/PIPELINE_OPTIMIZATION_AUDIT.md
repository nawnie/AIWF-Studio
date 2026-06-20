# Pipeline Optimization Audit

Date: 2026-06-15

Scope: image-to-image and image-to-video inference under the current AIWF Studio `venv`.

## Current venv capability matrix

| Capability | Local state | Practical effect |
| --- | --- | --- |
| PyTorch CUDA | `torch 2.6.0+cu124`, `torchvision 0.21.0+cu124` | Good baseline for SDPA, bf16/fp16, Ada FP8 primitives. |
| Diffusers | `diffusers 0.38.0` | Matches current docs version and supports Wan/GGUF APIs upstream. |
| Transformers | `transformers 4.57.6` | Still below 5, compatible with the project rule. |
| xFormers | missing | `--xformers` cannot activate unless installed; PyTorch SDPA remains available. |
| SageAttention | `sageattention 1.0.6`, `triton-windows 3.7.0.post26` | AIWF's Wan fallback hook can use it, but Diffusers native SAGE dispatch is disabled in this venv. |
| flash-attn | missing | Diffusers FLASH attention backend is disabled. |
| torchao | missing | AIWF TorchAO helper is code-only until the package is installed. |
| GGUF | `gguf 0.19.0` | Wan GGUF file parsing/runtime hooks are available. |
| Diffusers GGUF kernels | missing `kernels` | Upstream optimized GGUF CUDA kernels cannot activate. |
| ONNX GPU | `onnxruntime 1.23.2`, no `onnxruntime-gpu` | CPU ONNX only; not a speed path for local CUDA image pipelines. |

## External baselines checked

- Hugging Face Diffusers memory docs: CPU offload saves memory but is very slow; model offload is faster because whole components stay on GPU while active; group offload with CUDA streams can overlap transfer and compute but increases CPU memory pressure. Source: https://huggingface.co/docs/diffusers/en/optimization/memory
- Hugging Face Diffusers speed docs: PyTorch SDPA is enabled by default on PyTorch 2.x; `torch.compile` and `channels_last` are recommended speed experiments for UNet/VAE-heavy pipelines. Source: https://huggingface.co/docs/diffusers/en/optimization/fp16
- Hugging Face Diffusers img2img docs: SDXL image-to-image can be created from an existing text-to-image pipeline with `from_pipe` to avoid duplicate memory. Source: https://huggingface.co/docs/diffusers/en/using-diffusers/sdxl
- Hugging Face Diffusers Wan docs: Wan 2.2 I2V is supported in Diffusers; examples emphasize dtype, offload, quantization config, and component-level memory management. Source: https://huggingface.co/docs/diffusers/en/api/pipelines/wan
- Hugging Face Diffusers GGUF docs: Diffusers supports GGUF transformer loading and optional optimized CUDA kernels through the `kernels` package plus `DIFFUSERS_GGUF_CUDA_KERNELS=true`. Source: https://huggingface.co/docs/diffusers/en/quantization/gguf
- ComfyUI-WanVideoWrapper: popular Wan workflows use FP8 scaled models, GGUF loading, SageAttention, TeaCache-style caching, and torch.compile experiments; the repo notes Windows/Triton cache issues after model-code updates. Source: https://github.com/kijai/ComfyUI-WanVideoWrapper
- ComfyUI-GGUF: common Comfy GGUF path uses a dedicated GGUF UNet loader and the `gguf` package; LoRA support is marked experimental. Source: https://github.com/city96/ComfyUI-GGUF
- SageAttention: current upstream recommends SageAttention 2.x for newer kernels; the installed `1.0.6` is explicitly the older Triton v1 path. Source: https://github.com/thu-ml/SageAttention
- TeaCache4Wan2.1: reports large Wan2.1 speedups in a training-free cache with quality trade-offs and task-specific thresholds. Source: https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4Wan2.1/README.md

## Image-to-image: Diffusers pipelines

AIWF's current image pipeline is already above the basic Diffusers baseline:

- `DiffusersBackend` builds `StableDiffusionImg2ImgPipeline` / `StableDiffusionXLImg2ImgPipeline` from the loaded txt2img components instead of loading a second checkpoint. This matches the memory intent of Diffusers `from_pipe`.
- Model placement supports full CUDA, model CPU offload, sequential CPU offload, and automatic SDXL model offload on low VRAM.
- SDXL VAE slicing and tiling are enabled to reduce decode spikes.
- UNet channels-last is applied through `apply_attention_optimizations`.
- Local single-file loading uses cached config and `local_files_only=True`, so startup preload does not try remote component downloads.

Image pipeline gaps:

| Gap | Evidence | Recommendation |
| --- | --- | --- |
| `torch_compile` flag is not wired into `DiffusersBackend`. | `aiwf/infrastructure/quantization/torchao_quant.py` has helpers, but no runtime caller outside tests. | Add a flag-gated compile path for UNet, then benchmark same checkpoint/size before enabling broadly. |
| TorchAO int8/fp8 helpers are not usable in current venv. | `torchao` package is missing. | Keep disabled; install only for an explicit experiment with artifact receipts and rollback. |
| Channels-last setting is misleading. | `apply_attention_optimizations` applies channels-last unconditionally, while settings expose `AIWF_CHANNELS_LAST`. | Either document that image UNet channels-last is always on, or make the setting actually gate it. |
| VAE channels-last is not applied. | Local code applies channels-last to UNet only; Diffusers docs show UNet and VAE channels-last for SDXL compile path. | Low-risk experiment: apply VAE channels-last for SD/SDXL image pipelines and benchmark decode time. |
| xFormers is absent. | Package probe shows missing `xformers`. | Do not chase xFormers first; PyTorch SDPA is the modern default. Only install if a benchmark proves it helps this GPU/torch pair. |

## Image-to-video: Wan pipeline

AIWF's Wan path is heavily optimized and closer to a Comfy-style custom runtime than a plain Diffusers pipeline:

- Local-only component base, tokenizer, scheduler, and VAE loading.
- Mandatory dual high/low Wan 2.2 I2V transformer pair with high/low boundary handling.
- Native Comfy scaled-FP8 safetensors path using `_scaled_mm` when available.
- GGUF runtime path that mmaps quantized weights and replaces linear layers for on-the-fly dequant.
- CPU cache and high/low transformer swapping to avoid keeping both 14B stages in VRAM.
- Optional pinned-memory path with fail-safe fallback to disk-sequential mode when pinning is unsafe.
- Background low-stage preload while the high stage denoises.
- Wan attention bootstrap: CUDA SDPA flags, Diffusers SAGE/FLASH when actually callable, then SageAttention global fallback, then torch SDPA.
- Temporal chunked denoise with overlap blending for long clips on 16 GB GPUs.
- Structured throughput traces for video runs.

Wan pipeline gaps:

| Gap | Evidence | Recommendation |
| --- | --- | --- |
| Native Diffusers SAGE and FLASH backends are disabled in this venv. | Probe: `_CAN_USE_SAGE_ATTN=False`, `sageattn_callable=False`, `_CAN_USE_FLASH_ATTN=False`, `flash_attn_callable=False`. Installed SageAttention is `1.0.6`. | Benchmark current fallback first. Then test `sageattention>=2.1.1` or `2.2.0` in a copied venv, because it changes kernel behavior. |
| `torch.compile` is not wired for Wan transformers. | Roadmap marks compile deferred; local code does not compile Wan transformer modules. | Treat as a flagged experiment only. Kijai notes Windows/Triton cache and VRAM spikes after compile changes, so add cache cleanup guidance and shape-specific benchmarks. |
| TeaCache is not implemented. | No local TeaCache/step-cache code found; external TeaCache reports meaningful Wan I2V speedups with quality trade-offs. | Highest-impact research lane after Sage v2: implement as an optional sampler cache with visible quality/speed controls and default off. |
| Diffusers group offload/use_stream is not used. | AIWF has custom high/low cache/offload, not Diffusers group offload hooks. | Do not replace the custom path. Test group offload only for non-FP8/non-GGUF or fallback modes where current cache gives poor results. |
| GGUF runtime is functional but may be slower than upstream optimized kernels/custom ops. | AIWF does per-layer on-the-fly dequant; Diffusers GGUF docs mention optional CUDA kernels; `kernels` is missing locally. | Add a GGUF benchmark matrix: AIWF GGUF runtime vs FP8 safetensors vs optional `kernels` package path, same prompt/frames/steps. |
| Text encoder GGUF path appears load-oriented, not optimized runtime-oriented. | Local code supports UMT5 `.gguf`, but transformer GGUF has the richer mmap linear runtime. | Leave text encoder GGUF as memory/storage feature until measured; prioritize transformer step speed first. |

## Priority recommendation

1. Benchmark current baseline before changing anything.
   - Img2img: SD1.5 and SDXL, same init image, size, steps, sampler, batch size.
   - Wan: FP8 high/low and GGUF high/low, same prompt/image, frames, high/low steps, offload mode.

2. Fix the image pipeline settings mismatch.
   - Either gate channels-last on `AIWF_CHANNELS_LAST`, or rename/document it as always-on for image UNets.
   - Add a VAE channels-last experiment behind the existing flag.

3. Test SageAttention 2.x in a copied venv.
   - Current venv has the older SageAttention v1 path.
   - Acceptance gate: Diffusers native SAGE dispatch becomes callable and beats current fallback on the same Wan job without quality regressions or crashes.

4. Add a Wan TeaCache research branch.
   - Default off.
   - Controls: threshold, start percent, end percent, max skip or retention mode if implemented.
   - Acceptance gate: side-by-side videos plus throughput traces, because speedups trade against visual fidelity.

5. Evaluate GGUF optimized kernels.
   - Install `kernels` only in a copied/test venv first.
   - Acceptance gate: GGUF speed improves enough to justify the dependency, and visual differences are acceptable.

6. Only then revisit torch.compile.
   - It can help image UNet/VAE and video transformers, but on Windows it can cause first-run compile delay, shape recompiles, and Triton cache issues.
   - Acceptance gate: cached second-run throughput improves for fixed sizes, and fallback remains clean.

## Bottom line

AIWF is already doing the big structural work for local Wan on a 16 GB class GPU: FP8, GGUF, staged high/low loading, model swap, async preload, and temporal chunking. The largest remaining gaps are not simple code toggles; they are benchmark-gated accelerator lanes: SageAttention 2.x, TeaCache, Diffusers/Comfy-style GGUF kernel acceleration, and carefully scoped `torch.compile`.

## Implementation update: 2026-06-15

Implemented repo-local accelerator plumbing without changing the main Studio `venv`:

- `AIWF_CHANNELS_LAST` now gates image UNet/VAE channels-last instead of applying it unconditionally.
- `AIWF_TORCH_COMPILE` is wired for image UNet and VAE decode, and is skipped when CPU offload is active or expected.
- Wan acceleration capability reporting now checks whether optional packages actually import, not just whether package metadata exists.
- `python -m aiwf.workers.pipeline_benchmark` can run real img2img/Wan benchmarks and no-model accelerator probes, writing JSON receipts under `outputs/benchmarks/`.
- Created copied test environment `engines/pipeline_accel/.venv-test` and ignored it in git.

Copied-venv accelerator probe results:

| Experiment | Result |
| --- | --- |
| `sageattention==2.2.0` | Deferred: current upstream documents this install path, but it still needs a copied-venv Windows/CUDA retest before promotion. The previous local probe only reached the older `1.0.6` fallback. |
| `kernels==0.15.2` | Installed in copied venv, but it requires `huggingface-hub>=1.10.0`, which conflicts with `transformers 4.57.6`; after restoring HF Hub `0.36.2`, `kernels` is present but not importable. |
| `torchao` latest | Blocked: `torchao 0.17.0` reports incompatible Torch expectations and fails importing with current Torch/Triton. |
| `torchao==0.10.0` | Still not importable with current Torch/Triton. |
| Main venv | Unchanged: SageAttention fallback and AIWF GGUF runtime available; Diffusers SAGE/FLASH, GGUF CUDA kernels, and TorchAO unavailable. |

Probe receipts written:

- `outputs/benchmarks/pipeline-benchmark-20260615T232725Z-38fb95eb.json` — main `venv`.
- `outputs/benchmarks/pipeline-benchmark-20260615T232725Z-caffb46b.json` — copied accelerator test venv.

Current accelerator conclusion: stay on the main venv's existing Wan SageAttention fallback and AIWF GGUF runtime. Do not promote `kernels` or TorchAO into the main venv with the current Torch/Diffusers/Transformers stack.

## Benchmark update: 2026-06-15

Wan I2V smoke benchmark, same source image/prompt/seed, 320x320, 5 frames, 2 total denoise steps, sequential offload:

| Runtime | Receipt | Result |
| --- | --- | --- |
| Main `venv`, Comfy FP8 safetensors high/low | `pipeline-benchmark-20260615T234436Z-4648c1eb.json` | Completed in 119.05s end-to-end, 0.0420 frames/s. Denoise progress reported ~41s for 2 steps. |
| Main `venv`, GGUF Q3 high/low | `pipeline-benchmark-20260615T234647Z-52ce286d.json` | Completed in 108.20s end-to-end, 0.0462 frames/s. Denoise progress reported ~21s for 2 steps. |
| Copied accelerator test venv, GGUF Q3 high/low | `pipeline-benchmark-20260615T234912Z-7b25f2fc.json` | Failed preflight: `kernels` import failure breaks Diffusers Wan import; TorchAO also fails importing against current Triton. |

Observed bottleneck:

- `offload="model"` at 384x384 OOMed on the 16 GB GPU before completing even a 5-frame/2-step smoke.
- `offload="sequential"` completes, but most wall time is load/offload/CPU movement, not just attention kernels.
- FP8 safetensors currently emit repeated `_scaled_mm` layout failures and fall back to bf16 linear per layer. That is the clearest local speed gap; fixing the FP8 matmul layout in AIWF is more promising than creating a venv2 around currently broken `kernels`/TorchAO packages.

Decision: do **not** implement/promote `venv2` yet. Keep the copied venv as a disposable test lane only. The next high-value implementation is an AIWF runtime fix for FP8 `_scaled_mm` layout/weight orientation, then rerun the same receipts. After that, revisit SageAttention 2.x or GGUF CUDA kernels only if compatible Windows wheels become available.
