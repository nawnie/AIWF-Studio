# Wan 720x720 81-frame benchmark report

Date: 2026-06-17 UTC

## Objective

Run the current Wan I2V pipelines at the user's requested stress target:

- Resolution: 720x720
- Frames: 81
- FPS: 16, approximately 5 seconds
- Steps: 8 total, split as 4 high-noise steps and 4 low-noise steps
- Guidance: 1.0
- Image guidance: 1.0
- Sampler: FlowMatch Euler
- Sigma schedule: beta
- Flow shift: 5
- Seed: 123456
- Offload: sequential
- Temporal chunking: chunk 16, overlap 8

Benchmark configs in this folder:

- `wan_fp8_default_720square_81f_8step.json`
- `wan_fp8_manual_decode_720square_81f_8step.json`
- `wan_gguf_720square_81f_8step.json`

The completed GGUF sample video was copied here for review:

- `artifacts/wan_gguf_q4_720square_81f_8step.mp4`

The benchmark input image is also included:

- `artifacts/wan_input_720x720.png`

## Result Matrix

| Pipeline | Status | Result |
| --- | --- | --- |
| FP8 safetensor, default decode | Stalled and manually stopped | Reached high step 4, swapped to low-noise transformer, then produced no step 5 callback after extended observation. |
| GGUF Q4, default decode | Completed | Generated 81 frames at 720x720. Output copied to `artifacts/wan_gguf_q4_720square_81f_8step.mp4`. |
| FP8 safetensor, manual VAE decode | Stalled and manually stopped | Reproduced the same low-stage stall before VAE decode, so manual decode cannot fix this failure. |

## Completed GGUF Q4 Stats

Source receipt:

- `outputs/benchmarks/wan_720square_81f_8step_20260617/gguf_q4/pipeline-benchmark-20260617T014230Z-7c1aae79.json`

Key numbers:

- Total elapsed: 561.046 s
- Pipeline time: 543.364 s
- Denoise time: 494.880 s
- High denoise: 273.637 s
- Low denoise: 221.242 s
- VAE decode: 46.948 s
- Video postprocess: 1.490 s
- Video write: 2.567 s
- Frames per second: 0.144373
- Steps per second: 0.016166
- Iterations per second: 0.016166
- Seconds per iteration: 61.86 s/it at final callback
- Output size: 458,251 bytes

GGUF per-step trace:

| Step | Stage | it/s | s/it |
| --- | --- | ---: | ---: |
| 1 | high | 0.009901 | 100.999 |
| 2 | high | 0.012614 | 79.276 |
| 3 | high | 0.013873 | 72.085 |
| 4 | high | 0.014618 | 68.409 |
| 5 | low | 0.014915 | 67.045 |
| 6 | low | 0.015451 | 64.722 |
| 7 | low | 0.015850 | 63.090 |
| 8 | low | 0.016166 | 61.860 |

The successful GGUF run used about 6.3 GB VRAM near step 5 and released back to about 327 MB after completion.

## FP8 Safetensor Default Stall

Source trace:

- `outputs/benchmarks/wan_720square_81f_8step_20260617/fp8_default/dev-trace.log`
- Receipt remained `running` because the process was externally stopped after the stall.

High-stage progress before stall:

| Step | Stage | it/s | s/it |
| --- | --- | ---: | ---: |
| 1 | high | 0.009533 | 104.900 |
| 2 | high | 0.012346 | 80.996 |
| 3 | high | 0.013651 | 73.253 |
| 4 | high | 0.014380 | 69.539 |

Last trace event:

```text
High stage done - swapping to low-noise transformer.
```

After that event, no step 5 callback arrived. During the stall, GPU utilization stayed near 100% and VRAM was about 16,031 MiB of 16,376 MiB.

## FP8 Safetensor Manual Decode Stall

Source trace:

- `outputs/benchmarks/wan_720square_81f_8step_20260617/fp8_manual_decode/dev-trace.log`
- Receipt remained `running` because the process was externally stopped after the stall.

High-stage progress before stall:

| Step | Stage | it/s | s/it |
| --- | --- | ---: | ---: |
| 1 | high | 0.009602 | 104.143 |
| 2 | high | 0.012393 | 80.692 |
| 3 | high | 0.013661 | 73.202 |
| 4 | high | 0.014381 | 69.538 |

Last trace event:

```text
High stage done - swapping to low-noise transformer.
```

No step 5 callback arrived after more than 7 minutes. GPU utilization stayed near 100% and VRAM was about 16,027 MiB of 16,376 MiB.

This run never reached VAE decode. The manual VAE decode experiment is therefore not relevant to the full 81-frame FP8 failure. It may still be useful as a fallback for post-denoise memory issues, but it is not the bottleneck here.

## Key Findings

1. GGUF Q4 is currently the only tested pipeline that completed the full 720x720, 81-frame, 8-step request on the RTX 4070 Ti SUPER.
2. The FP8 safetensor pipeline is not simply OOM-crashing. It enters a long or stuck low-stage execution state after high step 4 with near-full VRAM and no callback progress.
3. The FP8 default and FP8 manual-decode runs have nearly identical high-stage throughput, then fail at the same high-to-low boundary. That points away from VAE decode and toward low-transformer execution, device placement, memory pressure, or the temporal chunk wrapper.
4. In the completed GGUF run, denoise dominates total time: 494.880 s of 561.046 s, about 88.2%. VAE decode is still meaningful at 46.948 s, but it is not the primary speed limit for this full run.
5. Once GGUF reaches the low stage, the low stage is faster than the high stage. The low-stage work took 221.242 s vs 273.637 s for high.
6. The current progress callback only fires after a denoise step completes. For the FP8 low-stage stall, we need instrumentation inside the low-stage step and inside each temporal chunk.

## Programmer Investigation Plan

Highest-value next changes:

1. Add per-stage memory snapshots around the high-to-low boundary:
   - before releasing high transformer
   - after releasing high transformer
   - before low transformer device move
   - after low transformer device move
   - before first low chunk
   - after each low chunk
2. Add start/end timing inside the temporal chunk denoise wrapper, including chunk index, chunk frame range, transformer stage, current step index, and torch CUDA memory.
3. Verify FP8 low transformer device placement and dtype after the swap. The symptom looks like the low transformer is technically running but under severe memory pressure or fallback behavior.
4. Compare this path against the Comfy workflow settings for:
   - block swap count
   - CPU offload strategy
   - context window / context overlap
   - cache strategy such as TeaCache, MagCache, or EasyCache
   - exact sampler and schedule settings
5. Test FP8 at 81 frames with chunk sizes 8, 12, 16, and 24 and overlaps 4 and 8. Keep the same 720x720 and 4 high / 4 low split so the failure signature stays comparable.
6. Do not spend more time on manual VAE decode for this specific failure until FP8 reaches step 5. The failure occurs before decode.

## Reproduction

Run GGUF:

```powershell
$env:AIWF_DEV_TRACE='1'
$env:AIWF_WAN_GGUF_RUNTIME='1'
.\venv\Scripts\python.exe -m aiwf.workers.pipeline_benchmark --config docs\benchmarks\wan_720square_81f_8step_20260617\wan_gguf_720square_81f_8step.json --out outputs\benchmarks\wan_720square_81f_8step_20260617\gguf_q4
```

Run FP8 default:

```powershell
$env:AIWF_DEV_TRACE='1'
Remove-Item Env:AIWF_WAN_GGUF_RUNTIME -ErrorAction SilentlyContinue
Remove-Item Env:AIWF_WAN_MANUAL_VAE_DECODE -ErrorAction SilentlyContinue
.\venv\Scripts\python.exe -m aiwf.workers.pipeline_benchmark --config docs\benchmarks\wan_720square_81f_8step_20260617\wan_fp8_default_720square_81f_8step.json --out outputs\benchmarks\wan_720square_81f_8step_20260617\fp8_default
```

Run FP8 manual VAE decode:

```powershell
$env:AIWF_DEV_TRACE='1'
$env:AIWF_WAN_MANUAL_VAE_DECODE='1'
$env:AIWF_WAN_VAE_CHUNK_FRAMES='81'
Remove-Item Env:AIWF_WAN_GGUF_RUNTIME -ErrorAction SilentlyContinue
.\venv\Scripts\python.exe -m aiwf.workers.pipeline_benchmark --config docs\benchmarks\wan_720square_81f_8step_20260617\wan_fp8_manual_decode_720square_81f_8step.json --out outputs\benchmarks\wan_720square_81f_8step_20260617\fp8_manual_decode
```
