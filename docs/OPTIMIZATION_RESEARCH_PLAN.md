# Optimization Research Plan

Prepared from `aiwf_studio_deep_research_2026-06-16/`.

## Current stance

AIWF Studio should stabilize optimization planning before promoting more speed
paths. The default lane remains PyTorch SDPA with fp16 and conservative
scheduler/memory behavior. Anything that changes quality, dependencies, graph
shape, or model semantics stays explicit and receipt-gated.

## Source package

Use these extracted research files as the working reference:

- `roadmap.md` - implementation order.
- `compatibility_matrix.md` / `.csv` - planner conflicts and denylist rules.
- `default_generation_profiles.md` - Safe, Balanced, Quality, Low VRAM, Fast Mode.
- `experimental_feature_flags.md` - flags, predicates, conflicts, receipt fields.
- `benchmark_protocol.md` - benchmark suites and promotion criteria.
- `benchmark_receipt_schema.json` - target receipt shape.
- `optimization_profile_schema.json` - target profile shape.

## Implemented foundation

- `OptimizationProfile`, `CapabilityReport`, `OptimizationPlan`, and
  `BenchmarkReceipt` domain models.
- Lazy `CapabilityDetector` for core package versions, optional feature
  availability, and GPU facts.
- `OptimizationPlanner` with conservative defaults and first denylist rules.
- `BenchmarkReceiptService` for reproducible profile/capability receipts.
- Profile selection in generation diagnostics and saved metadata.
- Settings UI profile controls.
- Benchmark worker receipt sections for capability and optimization profile data.
- `OptimizationDiagnosticsService` for read-only status, failures, and promotion gates.
- Planner denylist rules for Flash/Sage availability, ONNX GPU gating,
  FP4/NVFP4 hardware gating, VAE tiling visibility, and unknown Fast Mode
  recipes.
- Nested typed benchmark receipt payloads in pipeline benchmark receipts while
  preserving legacy receipt keys.
- Wan FP8 `_scaled_mm` fallback diagnostics with tensor metadata only: shapes,
  strides, dtypes, device type/index, padding, and exception class/message.
- Benchmark worker diagnostics: `pipeline_benchmark` installs a standalone
  `DevDiagnostics` writer so worker-only fallback traces are captured beside
  receipt JSON.
- Wan status telemetry: pipeline/service status messages are mirrored to
  `dev-trace.log` as `wan.status`, so killed or timed-out benchmark processes
  still leave stage evidence.
- Wan denoise throughput telemetry: Video progress now reports `it/s`
  (`s/it` included for very slow runs), and Wan result/benchmark receipts now
  include `step_count`, `denoise_seconds`, and `steps_per_second` alongside
  frame throughput.
- Wan aggregate stage timing: result/benchmark receipts now carry load,
  preprocess, denoise, high/low denoise split, pipeline overhead, postprocess,
  and video write timing fields.
- Wan internal Diffusers hook timing: during a single pipeline call AIWF now
  temporarily times `encode_prompt`, `encode_image`, `prepare_latents`,
  `vae.decode`, `postprocess_video`, and `maybe_free_model_hooks`, then restores
  the original methods. These fields explain which part of
  `pipeline_overhead_seconds` is prompt/image prep, decode/postprocess, or
  offload cleanup without changing generation behavior.
- Flag-gated Wan manual VAE decode experiment: set
  `AIWF_WAN_MANUAL_VAE_DECODE=1` to request latent output from Diffusers and
  decode through AIWF's temporal chunk helper (`AIWF_WAN_VAE_CHUNK_FRAMES`,
  default 4). Receipts record `manual_vae_decode` and
  `vae_decode_chunk_frames`. This is not default until local receipts prove it
  improves speed or fit without quality regressions.

## Local smoke receipts -- 2026-06-16

Receipt folder:
`outputs/benchmarks/wan_fp8_gguf_20260616/`.

| Lane | Receipt | Status | Elapsed | Throughput | Notes |
| --- | --- | --- | --- | --- | --- |
| Wan capability probe | `pipeline-benchmark-20260616T180217Z-e5faf248.json` | completed | n/a | n/a | GGUF runtime available; sageattention fallback available; diffusers FLASH/SAGE dispatch unavailable; torchao unavailable. |
| Wan FP8 safetensors smoke | `pipeline-benchmark-20260616T180426Z-47caa2bd.json` | completed | 57.62s receipt / 51.79s model trace | 0.087 fps receipt / 0.097 fps model trace | 128x128, 5 frames, 2 steps, sequential offload. No `wan.fp8_scaled_mm_fallback` trace rows recorded. |
| Wan GGUF Q4 smoke | `pipeline-benchmark-20260616T180537Z-53813d52.json` | completed | 50.50s receipt / 43.83s model trace | 0.099 fps receipt / 0.114 fps model trace | Same request/settings with `AIWF_WAN_GGUF_RUNTIME=1`. |

These are smoke receipts, not promotion-quality benchmarks. They validate that
both local Wan transformer formats can complete the same tiny request through
the application path and that the FP8 fallback seen earlier did not reproduce
under this sequential-offload smoke configuration.

## Local 384 Wan receipts -- 2026-06-16

Receipt folder:
`outputs/benchmarks/wan_fp8_gguf_384_20260616/`.

Matched settings: 384x384, 5 frames, 2 total steps, sequential offload, same
seed, same FP8 UMT5 text encoder, same Wan 2.1 VAE.

| Lane | Receipt | Status | Elapsed | Throughput | Notes |
| --- | --- | --- | --- | --- | --- |
| Wan FP8 safetensors 384 | `pipeline-benchmark-20260616T185427Z-36d229ea.json` | completed | 75.92s receipt / 69.98s model trace | 0.066 fps receipt / 0.071 fps model trace | No `wan.fp8_scaled_mm_fallback` trace rows recorded. |
| Wan GGUF Q4 384 | `pipeline-benchmark-20260616T185554Z-d075654b.json` | completed | 57.38s receipt / 50.92s model trace | 0.087 fps receipt / 0.098 fps model trace | Same request/settings with `AIWF_WAN_GGUF_RUNTIME=1`. |

This still is not a promotion-quality benchmark because it uses only one short
run per lane and only 2 denoise steps. It does show that sequential-offload
GGUF Q4 is currently faster than the local FP8 safetensor pair for the matched
384x384 smoke case, while FP8 did not trigger `_scaled_mm` fallback telemetry.

## Local 720-square Wan receipts -- 2026-06-16

Receipt folders:

- `outputs/benchmarks/wan_fp8_gguf_720square_20260616/`
- `outputs/benchmarks/wan_fp8_720square_lora_isolation_20260616/`
- `outputs/benchmarks/wan_fp8_720square_preloadfix_20260616/`
- `outputs/benchmarks/wan_fp8_720square_comfyfit_20260616/`
- `outputs/benchmarks/wan_fp8_720square_stage_timing_20260616/`
- `outputs/benchmarks/wan_fp8_720square_hook_timing_20260617/`
- `outputs/benchmarks/wan_fp8_720square_manual_vae_decode_20260617/`

Comfy reference settings were read from a local ComfyUI `user/default/workflows`
folder,
especially `DRAFTS\wan video gen.json`, `wan loop2.json`, and
`1_video_wan2_2_14B_i2v.json`. Shawn's recorded go-to settings are 720-side
Wan 2.2 I2V, CFG 1, Euler/Normal or Euler/simple, 4 steps, flow/sigma shift 5,
16 fps, 81 frames, Dasiwa high/low safetensors, FP8 UMT5 text encoder,
Wan 2.1 VAE, and the matching 4-step high/low LightX2V LoRAs.

| Lane | Receipt | Status | Elapsed | Throughput | Notes |
| --- | --- | --- | --- | --- | --- |
| Wan FP8 safetensors 720-square fit smoke | `pipeline-benchmark-20260616T193359Z-fcb161c0.json` | completed | 76.00s | 0.066 fps | 720x720, 5 frames, 2 steps, sequential offload, no LoRA. This proves safetensors can fit at 720-square in AIWF. |
| Wan GGUF Q4 720-square fit smoke | `pipeline-benchmark-20260616T193734Z-4059143a.json` | completed | 62.19s | 0.080 fps | Same 720x720 request with `AIWF_WAN_GGUF_RUNTIME=1`; no fallback trace rows observed. |
| Wan FP8 safetensors 720-square metric smoke | `pipeline-benchmark-20260616T231803Z-c116603b.json` | completed | 95.54s receipt / 89.28s model trace | 0.052 fps receipt / 0.056 fps model trace; 0.036 steps/s | Same 720x720, 5-frame, 2-step request after `it/s` and `steps_per_second` wiring. Typed receipt records `denoise_time_s=55.081`, equivalent to about 27.54 s/it. |
| Wan FP8 safetensors 720-square stage-timing smoke | `pipeline-benchmark-20260617T001822Z-09ca83c8.json` | completed | 99.50s receipt / 95.18s model trace | 0.050 fps receipt / 0.053 fps model trace; 0.041 steps/s | Same 720x720, 5-frame, 2-step request after aggregate stage timing. Load 10.93s, preprocess 0.07s, denoise 49.02s (high 28.06s / low 20.95s), pipeline overhead 32.33s, video write 2.73s. |
| Wan FP8 safetensors 720-square internal-hook smoke | `pipeline-benchmark-20260617T002445Z-7de4c744.json` | completed | 90.50s receipt / 86.42s model trace | 0.055 fps receipt / 0.058 fps model trace; 0.039 steps/s | Same 720x720, 5-frame, 2-step request with internal Diffusers hook timing. Load 8.62s, prompt encode 2.55s, latent prep 6.74s, denoise 50.75s, VAE decode 24.80s, video postprocess 0.07s, offload cleanup 0.01s, video write 1.83s. |
| Wan FP8 safetensors 720-square manual VAE decode chunk=4 | `pipeline-benchmark-20260617T003048Z-a95d9a66.json` | completed | 114.19s | 0.044 fps; 0.040 steps/s | `AIWF_WAN_MANUAL_VAE_DECODE=1`, `AIWF_WAN_VAE_CHUNK_FRAMES=4`. No OOM, but slower: VAE decode 49.42s because the 5-frame smoke splits into two decode chunks. |
| Wan FP8 safetensors 720-square manual VAE decode chunk=8 | `pipeline-benchmark-20260617T003312Z-6b0ccb40.json` | completed | 91.29s | 0.055 fps; 0.039 steps/s | Same manual path with no temporal split for the 5-frame smoke. VAE decode 23.06s, close to default 24.80s, but total receipt time is still slightly slower than default. Keep this as a fit fallback, not a speed default. |
| Wan FP8 safetensors + 4-step LoRA isolation | `pipeline-benchmark-20260616T210640Z-100a05fd.json` | completed | 113.68s receipt / 106.15s model trace | 0.044 fps receipt / 0.047 fps model trace | 720x720, 5 frames, 4 steps, high/low LightX2V LoRAs. LoRA loading is not the immediate failure source. |
| Wan FP8 safetensors + 4-step LoRA scaling probe | `pipeline-benchmark-20260616T211015Z-5a9892fd.json` | completed | 375.44s receipt / 368.86s model trace | 0.056 fps receipt / 0.057 fps model trace | 720x720, 21 frames, 4 steps, corrected profile ID `low_vram_model_offload`. Denoise progress was about 95s, so load/offload/reload overhead is a major part of the wall time. |
| Wan FP8 safetensors + 4-step LoRA post-preload-fix smoke | `pipeline-benchmark-20260616T212023Z-2f76c923.json` | completed | 100.98s receipt / 96.22s model trace | 0.050 fps receipt / 0.052 fps model trace | Same 5-frame LoRA case after skipping duplicate low-noise preload. Logs confirm the second low load is gone. |
| Wan FP8 safetensors + 4-step LoRA post-preload-fix scaling probe | `pipeline-benchmark-20260616T212227Z-7b36c6a4.json` | completed | 367.01s receipt / 360.55s model trace | 0.057 fps receipt / 0.058 fps model trace | Same 21-frame LoRA case after skipping duplicate low-noise preload. Improvement is real but small; low-stage/decode/sampler behavior remains the dominant gap. |
| Wan FP8 safetensors no-LoRA Comfy target | `pipeline-benchmark-20260616T194115Z-01a17b02.json` | killed externally | >3600s wall timeout | n/a | 720x720, 81 frames, 4 steps, no LoRA. Process did not OOM quickly; receipt remained `running` because the process was killed outside Python. |
| Wan FP8 safetensors no-LoRA chunk experiment | `pipeline-benchmark-20260616T204341Z-9315495a.json` | killed externally | >1200s wall timeout | n/a | Same 81-frame target with `chunk_size=64`, `chunk_overlap=0`; still did not reach Comfy-like throughput. |

Interpretation: 720-square fit is not the blocker. AIWF can run both local FP8
safetensors and GGUF at 720x720. The blocker is speed parity for longer clips.
Comfy's WanVideoWrapper path uses its own sampler, context windows, block
swapping with optional prefetch, Comfy RoPE options, and cache hooks such as
TeaCache/MagCache/EasyCache. AIWF's current Diffusers path uses full pipeline
orchestration plus temporal chunk wrapping and repeated high/low transformer
placement. The next optimization should target Comfy-style block-level memory
movement and sampler/context behavior before claiming 81-frame 720-side parity.

## Next implementation steps

1. Investigate and prototype Wan VAE decode optimization. The 720-square
   internal-hook smoke shows VAE decode, not MP4 writing or offload cleanup, is
   the largest post-step overhead.
2. Keep `AIWF_WAN_MANUAL_VAE_DECODE=1` as a fit/fallback experiment only.
   It did not beat the default 720-square 5-frame path; chunking below the clip
   length slowed decode materially.
3. Use internal hook timing on 21-frame and 81-frame targets to confirm whether
   VAE decode continues to dominate beyond the 5-frame smoke.
4. Investigate why the low-stage/decode portion dominates post-fix 21-frame
   runs after the duplicate low preload was removed.
5. Prototype a flag-gated Comfy-parity Wan path: block swap/prefetch first,
   then context windows, then cache methods. Keep it receipt-gated.
6. Rerun the 720x720 81-frame 4-step safetensor target only after stage timing
   and the next low-stage/decode optimization land.
7. Promote only locally benchmarked changes into defaults.

## Diagnostics

Use `GET /api/v1/optimization/status` or the Settings optimization diagnostics
panel to see whether a profile is recorded only, runtime-active, blocked, or a
promotion candidate. Known failed lanes are tracked in
`docs/OPTIMIZATION_FAILURES.md`.
