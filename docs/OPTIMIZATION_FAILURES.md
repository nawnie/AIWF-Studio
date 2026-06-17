# Optimization Failures

This file records optimization lanes that have already failed or produced
misleading results in the local Windows/NVIDIA environment. Do not retry these
as default-path work without a new compatibility reason and fresh receipts.

## Current failures and blocked lanes

| ID | Status | Summary | Decision |
| --- | --- | --- | --- |
| `kernels_windows_import` | Blocked | `kernels==0.15.2` installed in a copied accelerator venv but broke Diffusers Wan imports after dependency conflicts. | Do not promote `kernels` into the main venv. |
| `torchao_torch_triton` | Blocked | TorchAO latest and `0.10.0` failed to import cleanly with the current Torch/Triton stack. | Keep TorchAO disabled unless a copied venv proves compatibility. |
| `sageattention_2_windows` | Unavailable | SageAttention 2.x was not available from the current Windows/PyPI lane; installed path is older SageAttention 1.x. | Benchmark current fallback first; revisit only when compatible wheels exist. |
| `wan_fp8_scaled_mm_fallback` | Instrumented | Wan FP8 safetensors repeatedly fell back from `_scaled_mm` to bf16 linear. | Rerun FP8 receipts and inspect `wan.fp8_scaled_mm_fallback` tensor metadata before changing dependencies or layouts. |
| `wan_model_offload_16gb_oom` | Observed | Wan model offload OOMed on a 16 GB GPU at 384x384 smoke settings. | Treat sequential offload as the current fit path for smoke runs; benchmark before claiming usability. |
| `wan_720square_81f_diffusers_timeout` | Observed | 720x720, 81-frame, 4-step FP8 safetensor runs did not finish under AIWF's current Diffusers/sequential path, even without LoRA and with a `chunk_size=64`, `chunk_overlap=0` experiment. | Do not treat 720-square Comfy parity as solved. Focus next on stage timing, duplicate low-transformer load removal, and Comfy-style block swap/context/cache behavior. |

## Latest observations

- 2026-06-16 sequential-offload Wan smoke receipts at 128x128, 5 frames, and 2
  total steps completed for both FP8 safetensors and GGUF Q4. The worker trace
  recorded no `wan.fp8_scaled_mm_fallback` rows, so the earlier fallback was not
  reproduced by the tiny smoke run.
- 2026-06-16 sequential-offload Wan receipts at 384x384, 5 frames, and 2 total
  steps also completed for both FP8 safetensors and GGUF Q4. The worker trace
  again recorded no `wan.fp8_scaled_mm_fallback` rows. GGUF Q4 was faster in
  this matched short run, but the run is too small to promote defaults.
- 2026-06-16 720x720 fit smokes completed for both FP8 safetensors and GGUF Q4
  at 5 frames / 2 steps. This confirms 720-square safetensor fit is possible in
  AIWF and the blocker is not simply model format.
- 2026-06-16 720x720 4-step LightX2V LoRA isolation completed at 5 frames
  (`113.68s`) and 21 frames (`375.44s`). LoRA is not the immediate crash source.
  The 21-frame run used the correct `low_vram_model_offload` profile in the
  receipt.
- 2026-06-16 duplicate low-noise preload was removed from the Wan generation
  path. Post-fix 720x720 4-step LoRA receipts completed at 5 frames (`100.98s`)
  and 21 frames (`367.01s`). The fix removes a clear redundant load and adds
  useful stage traces, but it is not enough to close the Comfy parity gap.
- 2026-06-16 Wan progress and receipts were updated to report denoise
  throughput as `it/s` in the UI and `steps_per_second` in generated benchmark
  data. A fresh 720x720, 5-frame, 2-step FP8 safetensor metric smoke completed
  with `steps_per_second=0.03631` (`27.54 s/it`), `denoise_seconds=55.081`,
  and `frames_per_second=0.0523` in receipt
  `pipeline-benchmark-20260616T231803Z-c116603b.json`.
- 2026-06-17 aggregate stage timing landed. The 720x720, 5-frame, 2-step FP8
  safetensor stage-timing smoke completed with `steps_per_second=0.040803`,
  `denoise_seconds=49.016`, high/low split `28.062s / 20.953s`,
  `pipeline_overhead_seconds=32.332`, and `video_write_seconds=2.726` in
  receipt `pipeline-benchmark-20260617T001822Z-09ca83c8.json`. The next
  bottleneck is not just denoise; it is also the post-step Diffusers pipeline
  overhead before MP4 write.
- 2026-06-17 internal Diffusers hook timing identified the biggest post-step
  overhead in the 720x720, 5-frame, 2-step FP8 safetensor smoke: VAE decode
  took `24.799s`, while video postprocess took `0.072s`, offload cleanup
  `0.006s`, and MP4 write `1.834s`. Receipt:
  `pipeline-benchmark-20260617T002445Z-7de4c744.json`. Next optimization work
  should target Wan VAE decode before spending time on the writer.
- 2026-06-17 `AIWF_WAN_MANUAL_VAE_DECODE=1` completed without OOM but should
  not be promoted as a speed default. With `AIWF_WAN_VAE_CHUNK_FRAMES=4`, VAE
  decode slowed to `49.419s` and total receipt time to `114.19s`
  (`pipeline-benchmark-20260617T003048Z-a95d9a66.json`). With
  `AIWF_WAN_VAE_CHUNK_FRAMES=8`, VAE decode was `23.055s`, close to default,
  but total receipt time was still slightly slower at `91.29s`
  (`pipeline-benchmark-20260617T003312Z-6b0ccb40.json`). Treat manual decode as
  a fit/fallback experiment, not a speed optimization.
- 2026-06-16 Shawn's Comfy workflows show the real target lane: 720-side Wan
  2.2 I2V, CFG 1, Euler/simple or Euler/Normal, 4 steps, flow shift 5,
  16 fps, 81 frames, Dasiwa high/low safetensors, FP8 UMT5, Wan 2.1 VAE, and
  matching high/low 4-step LoRAs. Comfy reports roughly 120-140s for 720-side
  81-frame runs on the local class of hardware, while AIWF did not complete the
  equivalent 81-frame Diffusers path within the tested timeouts.

## Rules

- Failed optional dependency lanes stay out of the main venv.
- A copied test venv is disposable evidence, not a product feature.
- New accelerator claims require successful receipts, Windows verification, and a fallback path.
- Quality-changing or numerically different paths must remain visible in metadata and receipts.
