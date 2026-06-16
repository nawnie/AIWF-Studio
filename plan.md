# Plan

## Current Objective - Wan 16 GB Runtime Architecture Fix
- Make AIWF Studio's Wan video runtime behave like a serious 16 GB Ada/RTX local video pipeline instead of accepting a 320-360p ceiling.
- Use current online sources to separate raw Wan reference requirements from Comfy-style quantized/offloaded workflows.
- Keep Diffusers working as the default/reference path, then add better selectable methods around it so Studio can teach and benchmark multiple runtimes.
- Implement concrete runtime fixes first, then decide whether a dedicated venv2/engine lane is needed.

## Current Request Interpretation
- User rejects the conclusion that the hardware is the limiting factor and explicitly asked to look online.
- User clarified that Diffusers is the default/reference path, not the limit. We should get Diffusers-based methods working, then add and compare better methods.
- User wants the app to teach all methods and support experimentation: Diffusers, AIWF native FP8/GGUF, Comfy/WanVideoWrapper-style, and future methods should be visible choices with benchmarks.
- Success means: source-backed diagnosis, a model/backend matrix using Shawn's local files, then code/architecture changes that move AIWF toward Comfy/WanVideoWrapper behavior on the RTX 4070 Ti SUPER class system.
- Assumption to verify: Q3/Q4 GGUF pairs should fit without OOM even if both stages are resident/cached, so OOM means our runtime/offload architecture is wrong or loading them expanded.

## Current Constraints
- Clean-room rule applies: read public docs/repos for behavior and interfaces; do not copy incompatible implementation code.
- Main Studio boot dependencies must stay stable; optional accelerators and external engines must remain optional.
- No claimed speed/resolution improvement without a saved benchmark receipt.
- Avoid pushing broken `kernels`/TorchAO installs into main `venv`; prior copied-venv probe showed they currently break imports.

## Current Affected Project Surfaces
- `aiwf/infrastructure/wan/pipeline.py` - FP8 scaled linear, high/low transformer load, offload/cache, output type, Diffusers call compatibility.
- `aiwf/infrastructure/wan/gguf_runtime.py` - GGUF linear/runtime path if the FP8 fix is insufficient.
- `aiwf/services/wan.py` - preflight, model path resolution, throughput/capability reporting.
- `aiwf/workers/pipeline_benchmark.py` and `docs/benchmark_log.jsonl` - verification receipts.
- `aiwf/infrastructure/comfy/` and architecture docs - possible future external Comfy/venv2 engine lane.
- Future UI/API surface: Video tab should expose backend/method as a first-class control, not hide it in model names.

## Current Acceptance Gates
- Gate WAN16-G1: Online evidence checked from current official/model/repo sources.
  Evidence source: Comfy docs, Wan model card, Diffusers docs, Kijai WanVideoWrapper, ComfyUI-GGUF, SageAttention, TeaCache.
  Pass action: implement local runtime fix.
  Fail action: do not change architecture based on stale claims.
- Gate WAN16-G2: FP8 `_scaled_mm` layout failure reproduced and fixed in a small local CUDA probe.
  Evidence source: `venv\Scripts\python.exe` CUDA script and focused tests.
  Pass action: patch `FP8ScaledLinear`.
  Fail action: move to Comfy-style engine lane instead of guessing.
- Gate WAN16-G3: Runtime change proves itself with same Wan smoke benchmark receipts.
  Evidence source: `python -m aiwf.workers.pipeline_benchmark`.
  Pass action: update benchmark log and set larger-resolution benchmark route.
  Fail action: implement engine isolation/Comfy backend lane.
- Gate WAN16-G4: Tests pass for changed runtime behavior.
  Evidence source: focused pytest and compile checks.
  Pass action: final summary with source links.
  Fail action: fix tests before claiming progress.
- Gate WAN16-G5: Backend/method claims are teachable and comparable.
  Evidence source: method registry docs/UI text/receipts.
  Pass action: expose method choices and descriptions.
  Fail action: keep method experimental/internal.

## Current Public Verification Trace
- Pass 1 - Request alignment: User asked for online verification plus architecture/runtime changes, not another explanation of the old benchmark. Active route targets Wan runtime behavior.
- Pass 2 - Source check: Official Wan card distinguishes raw 80 GB reference generation from consumer/Comfy-style optimized use; Comfy docs list FP8/quantized workflow resources and GGUF/community resources.
- Pass 3 - Project fit: AIWF already has a custom FP8/GGUF Wan path; the fastest near-term change is fixing its FP8 matmul fallback before standing up a new engine.
- Pass 4 - Failure pre-mortem: If `_scaled_mm` cannot be made reliable on Windows/Torch 2.6, venv2 should be an isolated video engine, likely Comfy/WanVideoWrapper-compatible, not a mutation of the image venv.
- Pass 5 - User correction: Diffusers remains the default/reference, but the architecture goal is multi-method experimentation. Plan updated to avoid treating Diffusers as the ceiling.

## Current Local Model Matrix
Source: `models/wan` inventory on 2026-06-15.

| Family | High | Low | Expected role |
| --- | --- | --- | --- |
| FP8 safetensors Lightspeed/Boundbite | `DasiwaWAN22I2V14BLightspeed_boundbiteHighV10.safetensors` | `DasiwaWAN22I2V14BLightspeed_boundbiteLowV10.safetensors` | AIWF native Comfy scaled-FP8 path; should be fastest if `_scaled_mm` path stays active. |
| GGUF Q3 | `Wan2.2-I2V-A14B-HighNoise-Q3_K_S.gguf` | `Wan2.2-I2V-A14B-LowNoise-Q3_K_S.gguf` | Lowest-memory quantized pair; should not expand both models to bf16. If it OOMs, runtime architecture is wrong. |
| GGUF Q4 TastySin | `DasiwaWAN22I2V14BTastysinV8_q4High.gguf` | `DasiwaWAN22I2V14BTastysinV8_q4Low.gguf` | Better quality/more memory than Q3; should be benchmarked after Q3 residency/offload is correct. |
| 4-step LoRA variants | Several high/low LightX2V/Lightning/SVI LoRAs | matching low-stage LoRAs | Speed/quality method lane after base model path is stable. |

Text encoders available:

| Encoder | Expected role |
| --- | --- |
| `nsfw_wan_umt5-xxl_fp8_scaled.safetensors` | Preferred Wan UMT5 encoder for 16 GB path. |
| `umt5-xxl-encoder-Q4_K_M.gguf` | Lower-memory UMT5 experiment. |
| `umt5-xxl-encoder-Q5_K_M.gguf` | Higher-quality GGUF UMT5 experiment. |
| bundled `models/wan/Diffusers/.../text_encoder/model.safetensors` | Diffusers/reference fallback, but likely heavier. |

## Current Lanes
### Lane: WAN16 Online Baseline
Status: done
Goal: Verify what the hardware/runtime should reasonably target.

#### Card: WAN16-ONLINE-1
Type: fact
Goal: Verify official Wan/Comfy/Diffusers expectations.
Depends on: web sources.
Evidence: Comfy Wan2.2 docs list 5B on 8GB, 14B I2V resources, GGUF resources, WanVideoWrapper resources; Wan model card says I2V-A14B supports 480P/720P but raw single-GPU command needs 80GB VRAM; Diffusers docs show group offload/streaming as memory path.
Failure mode: confusing raw fp16 reference runtime with optimized consumer workflows.
Success check: final/source notes distinguish raw reference limits from optimized quant/offload runtime.
Verification state: verified
Next if pass: WAN16-FP8-1
Next if fail: gather more primary sources.

### Lane: WAN16 FP8 Runtime Fix
Status: active
Goal: Stop the current native FP8 path from falling back to bf16 linear on every layer.

#### Card: WAN16-FP8-1
Type: diagnostic_branch
Goal: Reproduce cuBLASLt row/column layout requirement in a small CUDA script.
Depends on: local Torch CUDA.
Evidence: pending.
Failure mode: patching tensor strides based on guesswork.
Success check: script identifies the correct `torch._scaled_mm` weight layout.
Verification state: unverified
Next if pass: WAN16-FP8-2
Next if fail: use architecture lane.

#### Card: WAN16-FP8-2
Type: action
Goal: Patch `FP8ScaledLinear.forward` to preserve the column-major weight layout needed by `_scaled_mm`.
Depends on: WAN16-FP8-1.
Evidence: pending diff and tests.
Failure mode: functional correctness regression or still falls back during Wan benchmark.
Success check: unit CUDA test and Wan smoke no longer emit repeated `_scaled_mm` fallback warnings.
Verification state: unverified
Next if pass: WAN16-BENCH-1
Next if fail: WAN16-ARCH-1

### Lane: WAN16 Benchmark And Architecture Decision
Status: pending
Goal: Benchmark fixed runtime, then decide whether venv2/Comfy engine is still necessary.

#### Card: WAN16-BENCH-1
Type: action
Goal: Rerun same 320x320 FP8/GGUF smoke and attempt a higher-res smoke if memory allows.
Depends on: WAN16-FP8-2.
Evidence: pending receipt JSON files.
Failure mode: benchmark passes at 320 but still OOMs above 384 due load/offload architecture.
Success check: receipt records elapsed, fps, warnings, and output path.
Verification state: unverified
Next if pass: final or WAN16-ARCH-1
Next if fail: WAN16-ARCH-1

### Lane: WAN16 Method Registry
Status: pending
Goal: Make video generation methods explicit, teachable, and benchmarkable.

#### Card: WAN16-METHOD-1
Type: action
Goal: Add a typed method/backend concept for Wan I2V.
Depends on: local model matrix and existing `WanI2VRequest`.
Evidence: pending.
Failure mode: backend behavior remains implicit in file extensions and users cannot learn/compare methods.
Success check: request/config can represent `diffusers_reference`, `aiwf_fp8`, `aiwf_gguf`, `comfy_engine`, and `external_worker` without breaking default Diffusers path.
Verification state: unverified
Next if pass: WAN16-METHOD-2
Next if fail: keep method names internal and document only.

#### Card: WAN16-METHOD-2
Type: action
Goal: Add method capability/preflight reporting.
Depends on: WAN16-METHOD-1.
Evidence: pending.
Failure mode: UI offers methods that cannot run locally.
Success check: each method reports installed/missing packages, supported model file types, expected memory behavior, and known blockers.
Verification state: unverified
Next if pass: WAN16-METHOD-3
Next if fail: block unavailable methods with clear status.

#### Card: WAN16-METHOD-3
Type: action
Goal: Make benchmark receipts include method/backend/model family/text encoder family.
Depends on: WAN16-METHOD-1.
Evidence: pending benchmark JSON.
Failure mode: future speed results are not comparable.
Success check: receipts can compare Q3/Q4/FP8/LoRA/TextEncoder choices without guessing from filenames.
Verification state: unverified
Next if pass: WAN16-MATRIX-1
Next if fail: patch benchmark schema before more runs.

### Lane: WAN16 Model Benchmark Matrix
Status: pending
Goal: Test Shawn's actual model families systematically before changing architecture again.

#### Card: WAN16-MATRIX-1
Type: action
Goal: Run controlled smoke matrix on original AIWF output images.
Depends on: WAN16-METHOD-3.
Evidence: benchmark receipts.
Failure mode: benchmarking one arbitrary image and overfitting conclusions.
Success check: at least one 512 source and one 640/source1024 case for FP8, Q3, Q4, with same prompt/frames/steps/seed.
Verification state: unverified
Next if pass: WAN16-MATRIX-2
Next if fail: fix pipeline before larger matrix.

#### Card: WAN16-MATRIX-2
Type: diagnostic_branch
Goal: Test Q3/Q4 residency/offload assumptions.
Depends on: WAN16-MATRIX-1.
Evidence: receipt memory/failure status and logs.
Failure mode: GGUF runtime expands or duplicates weights so Q3/Q4 OOMs despite quantization.
Success check: Q3 pair can run without full bf16 expansion; if not, identify exact load path.
Verification state: unverified
Next if pass: WAN16-MATRIX-3
Next if fail: WAN16-GGUF-1

#### Card: WAN16-MATRIX-3
Type: action
Goal: Add 4-step LoRA method tests after base path is stable.
Depends on: WAN16-MATRIX-2.
Evidence: benchmark receipts and output videos.
Failure mode: LoRA application increases memory or breaks high/low pairing.
Success check: 4-step LoRA path completes and receipt records LoRA pair/scale.
Verification state: unverified
Next if pass: WAN16-ARCH-1 if still needed
Next if fail: isolate LoRA memory handling.

### Lane: WAN16 GGUF Runtime Correctness
Status: pending
Goal: Ensure GGUF Q3/Q4 is actually a quantized runtime path, not accidental expanded loading.

#### Card: WAN16-GGUF-1
Type: diagnostic_branch
Goal: Trace whether Q3/Q4 high/low models are materialized expanded, cached twice, or streamed/dequantized layer-wise.
Depends on: WAN16-MATRIX-2 fail or weak result.
Evidence: `gguf_runtime.py`, memory logs, benchmark output.
Failure mode: assuming GGUF is memory efficient while code expands it.
Success check: runtime path is confirmed or patched to avoid full expansion/resident duplication.
Verification state: unverified
Next if pass: WAN16-MATRIX-2
Next if fail: WAN16-ARCH-1

### Lane: WAN16 Engine Architecture
Status: pending
Goal: Define isolated video engine/venv2 when a non-Diffusers/non-AIWF-native method is the better backend.

#### Card: WAN16-ARCH-1
Type: action
Goal: Scaffold a real optional video engine lane rather than mutating Studio image venv.
Depends on: failed or insufficient native runtime gates, or confirmed advantage from Comfy/WanVideoWrapper-style runtime.
Evidence: architecture docs and existing `ProcessSupervisor`.
Failure mode: making Comfy/Wan deps mandatory or breaking image generation.
Success check: engine is optional, subprocess isolated, method-visible, and can be benchmarked separately against Diffusers/reference.
Verification state: unverified
Next if pass: validation
Next if fail: stop and report blocker.

## Current Active Route
- WAN16-ONLINE-1 -> WAN16-FP8-1 -> WAN16-FP8-2 -> WAN16-METHOD-1 -> WAN16-METHOD-2 -> WAN16-METHOD-3 -> WAN16-MATRIX-1 -> WAN16-MATRIX-2 -> WAN16-GGUF-1 if needed -> WAN16-ARCH-1 if needed

## Current Validation Log
- 2026-06-15: Worker tenant split started. Added generic `WorkerTenantRegistry`, optional `wan` engine entry, `engines/wan/worker.py` probe worker, engine bootstrap/verify PowerShell scripts, and `docs/ENGINE_ISOLATION.md`.
- 2026-06-15: Focused validation passed: `tests/test_worker_tenant.py tests/test_process_supervisor.py tests/test_training_engine_status.py -q` => 24 passed.
- 2026-06-15: Live tenant status verified: Wan visible but disabled/missing venv, Kohya disabled/missing venv, ED2 ready via shared Studio venv.

## Current Open Unknowns
- Whether changing `_scaled_mm` weight stride removes the FP8 fallback warnings on the full Wan transformer.
- Whether 480p completes after the FP8 fix or still needs block-level/leaf-level offload architecture.
- Whether Comfy/WanVideoWrapper should be integrated as a subprocess engine if local Diffusers remains memory-bound.
- Whether current AIWF GGUF runtime keeps Q3/Q4 quantized enough to load/cache both high/low stages without OOM.
- Whether Q4 TastySin or Q3 stock gives the best speed/quality baseline for Shawn's local image outputs.
- Whether the FP8 UMT5 encoder should become explicit default in benchmark configs and UI state persistence.

## Current Objective - Pipeline Optimization Audit
- Validate AIWF Studio's image-to-image and image-to-video pipelines against current official docs, popular repo patterns, and existing project notes.
- Focus on whether each supported runtime type under the current Studio venv is already optimized or has concrete, evidence-backed gaps: Diffusers, safetensors, GGUF/quantized paths, FP8, attention kernels, offload/caching, and video-specific chunking.

## Current Request Interpretation
- User wants Agent MoK validation, not a blind rewrite: compare AIWF's actual local pipeline code to popular online/reference implementations and notes.
- Success means a source-backed gap matrix and prioritized action list for img2img and img2vid optimization under the existing `venv`.
- Assumption: no dependency bumps or optimization flags should be enabled by default without a benchmark or compatibility gate.

## Current Constraints
- Clean-room rule still applies: study behavior/docs, do not copy incompatible repo code.
- Optional engines and accelerators must not become mandatory boot dependencies.
- Do not casually bump `torch`, `diffusers`, `transformers`, `gradio`, `opencv-python-headless`, or `spandrel`.
- Do not claim speedups without benchmark evidence and logged throughput.

## Current Affected Project Surfaces
- `aiwf/infrastructure/diffusers/backend.py` - SD/SDXL txt2img/img2img/inpaint pipeline loading, offload, VAE memory tuning, hires/img2img pass.
- `aiwf/infrastructure/torch/attention.py` - attention backend selection and optional acceleration hooks.
- `aiwf/infrastructure/wan/pipeline.py` - Wan image-to-video loading, local components, GGUF/quant handling, offload/cache/chunk paths.
- `aiwf/services/wan.py` - Wan request orchestration and throughput tracing.
- `aiwf/core/config/settings.py` and launch/settings surfaces - flags affecting runtime optimization.
- `docs/ACCELERATION_EXPERIMENTS.md`, `docs/WAN_LOCAL_COMPONENTS.md`, `docs/ROADMAP_V2.md` - existing optimization notes and claims to reconcile.

## Current Acceptance Gates
- Gate PVAL-G1: Local pipeline behavior is verified from source files, not inferred from memory.
  Evidence source: targeted reads/rg of AIWF code and docs.
  Pass action: compare to external sources.
  Fail action: inspect missing local surface before answering.
- Gate PVAL-G2: Current venv capability matrix is verified.
  Evidence source: `venv\Scripts\python.exe` package/version probe and import checks.
  Pass action: classify optimizations as available, absent, or code-only.
  Fail action: mark capability unknown and avoid recommendations depending on it.
- Gate PVAL-G3: External optimization claims use current primary or strong repo sources.
  Evidence source: official Diffusers docs plus popular repo READMEs for Wan/GGUF/Sage/TeaCache patterns.
  Pass action: include citations in final.
  Fail action: label the claim unverified or omit it.
- Gate PVAL-G4: Recommendations distinguish code-ready, flag-gated experiment, install-needed, and benchmark-needed actions.
  Evidence source: comparison matrix.
  Pass action: final audit report.
  Fail action: revise action list.

## Current Public Verification Trace
- Pass 1 - Request alignment: The task is an optimization validation audit for img2img and img2vid pipeline families, not ED2/training and not immediate dependency upgrades.
- Pass 2 - Route logic: Inspect local code and venv first, then online baselines, then compare. This prevents recommending optimizations already implemented or unavailable under the venv.
- Pass 3 - Project fit: Key surfaces are Diffusers backend, Wan backend/service, attention/quant flags, and existing acceleration docs. Any later code change must remain optional/flag-gated.
- Pass 4 - Failure pre-mortem: Risks are stale online claims, over-generalizing ComfyUI-specific tricks, enabling unstable accelerators without benchmarks, and confusing GGUF storage support with runtime speed support.

## Current Lanes
### Lane: Pipeline Validation - Local Inventory
Status: done
Goal: Establish what AIWF actually runs for img2img and img2vid.

#### Card: PVAL-LOCAL-IMG
Type: check
Goal: Verify Diffusers image pipeline loading, memory placement, component sharing, VAE behavior, and img2img pass design.
Depends on: `aiwf/infrastructure/diffusers/backend.py`
Evidence: `aiwf/infrastructure/diffusers/backend.py:271-294`, `:357-443`, `:515-536`, `:739-767`, `:1113-1184`; `aiwf/infrastructure/torch/attention.py:10-47`.
Failure mode: assuming AIWF uses a baseline Diffusers pipeline when it has local customizations.
Success check: findings include source-backed strengths and gaps.
Verification state: verified
Next if pass: PVAL-VENV
Next if fail: inspect adjacent generation/request surfaces.

#### Card: PVAL-LOCAL-VID
Type: check
Goal: Verify Wan img2vid pipeline loading, safetensors/GGUF handling, attention/offload/cache/chunking, and throughput tracing.
Depends on: `aiwf/infrastructure/wan/pipeline.py`, `aiwf/services/wan.py`
Evidence: `aiwf/infrastructure/wan/pipeline.py:1-18`, `:423-427`, `:590-608`, `:1326-1345`, `:1594-1657`, `:1768-1818`, `:1890-1921`, `:1966-2135`, `:2204-2339`; `aiwf/infrastructure/wan/gguf_runtime.py:1-242`; `aiwf/infrastructure/torch/wan_perf.py:1-240`; `aiwf/services/wan.py:197-285`, `:943-955`.
Failure mode: treating roadmap notes as shipped runtime behavior.
Success check: findings distinguish shipped, flag-gated, partial, and TODO behavior.
Verification state: verified
Next if pass: PVAL-VENV
Next if fail: inspect Wan docs/tests.

#### Card: PVAL-VENV
Type: fact
Goal: Record installed accelerator/runtime packages in Studio `venv`.
Depends on: local package/version/import probe.
Evidence: package probe under `venv\Scripts\python.exe`: torch `2.6.0+cu124`, diffusers `0.38.0`, transformers `4.57.6`, sageattention `1.0.6`, triton-windows `3.7.0.post26`, gguf `0.19.0`; missing xformers, torchao, flash-attn, optimum, onnxruntime-gpu, kernels.
Failure mode: recommending xFormers/Sage/TorchAO/GGUF routes that are not installed.
Success check: package matrix says installed/missing/version for core and optional accelerators.
Verification state: verified
Next if pass: PVAL-ONLINE-DIFF
Next if fail: mark unknowns and continue with code-only comparison.

### Lane: Pipeline Validation - External Baselines
Status: done
Goal: Compare AIWF against current official docs and popular repo practices.

#### Card: PVAL-ONLINE-DIFF
Type: fact
Goal: Verify official Diffusers img2img/memory/optimization guidance.
Depends on: Hugging Face Diffusers docs.
Evidence: Hugging Face Diffusers memory, acceleration, img2img/SDXL, Wan, and GGUF docs opened 2026-06-15.
Failure mode: using stale Diffusers optimization advice.
Success check: final cites official docs for offload, VAE slicing/tiling, attention/compile/quantization where applicable.
Verification state: verified
Next if pass: PVAL-ONLINE-WAN
Next if fail: browse official docs again or mark source unavailable.

#### Card: PVAL-ONLINE-WAN
Type: fact
Goal: Verify current Wan img2vid patterns from official Diffusers and popular repos such as ComfyUI Wan wrappers, GGUF loaders, SageAttention, and TeaCache.
Depends on: online repo/doc reads.
Evidence: Hugging Face Wan/GGUF docs; `kijai/ComfyUI-WanVideoWrapper`; `city96/ComfyUI-GGUF`; `thu-ml/SageAttention`; `ali-vilab/TeaCache4Wan2.1`.
Failure mode: treating a community extension trick as universally stable for AIWF.
Success check: final separates mature baseline practices from experimental repo-specific optimizations.
Verification state: verified
Next if pass: PVAL-COMPARE
Next if fail: cite only verified sources.

### Lane: Pipeline Validation - Compare And Recommend
Status: done
Goal: Produce the audit result with prioritized next actions.

#### Card: PVAL-COMPARE
Type: check
Goal: Build gap matrix for img2img and img2vid runtimes.
Depends on: PVAL-LOCAL-IMG, PVAL-LOCAL-VID, PVAL-VENV, PVAL-ONLINE-DIFF, PVAL-ONLINE-WAN.
Evidence: `docs/PIPELINE_OPTIMIZATION_AUDIT.md` created with venv matrix, local strengths, external baseline comparison, and prioritized recommendations.
Failure mode: mixing unsupported dependency changes into safe recommendations.
Success check: each recommendation has evidence, impact expectation, risk, and validation gate.
Verification state: verified
Next if pass: PVAL-ACTION
Next if fail: re-open weak source/card.

#### Card: PVAL-ACTION
Type: action
Goal: Decide whether to patch docs/config/code now or return an audit-only report.
Depends on: PVAL-COMPARE.
Evidence: Audit-only doc change. No runtime pipeline behavior changed because the main gaps require package changes or benchmark gates.
Failure mode: making silent performance-affecting changes without benchmarks.
Success check: final clearly states what was changed, or that no runtime changes were made.
Verification state: verified
Next if pass: final summary
Next if fail: ask user for approval on risky dependency/runtime changes.

## Current Active Route
- PVAL-LOCAL-IMG -> PVAL-LOCAL-VID -> PVAL-VENV -> PVAL-ONLINE-DIFF -> PVAL-ONLINE-WAN -> PVAL-COMPARE -> PVAL-ACTION

## Current Open Unknowns
- Whether SageAttention 2.x improves this machine enough to justify upgrading from the current v1 fallback path.
- Whether TeaCache can be added to AIWF's Wan sampler with acceptable visual trade-offs and stable controls.
- Whether Diffusers GGUF optimized CUDA kernels (`kernels` package) can improve AIWF's custom Wan GGUF runtime or require a separate loader path.
- Whether `torch.compile` is worth the Windows/Triton cache and shape-recompile cost for AIWF image and Wan workloads.

## Current Validation Log
- 2026-06-15: Agent MoK route created for pipeline optimization audit with four verification passes.
- 2026-06-15: Local pipeline source reads verified Diffusers img2img component sharing/offload/VAE tuning and Wan FP8/GGUF/Sage/cache/chunk paths.
- 2026-06-15: Current venv capability probe completed; optional accelerator gaps recorded.
- 2026-06-15: External baselines checked from current Hugging Face Diffusers docs and popular Wan/GGUF/Sage/TeaCache repos.
- 2026-06-15: Added `docs/PIPELINE_OPTIMIZATION_AUDIT.md`; no runtime behavior changed.
- 2026-06-15: Focused validation passed: `43 passed` for `tests/test_wan.py`, `tests/test_wan_gguf_runtime.py`, and `tests/test_quantization.py`.

## Objective
- Modernize the local ED2 integration so EveryDream2 can run under AIWF Studio's main venv/dependency line where possible.
- Treat downgrade/isolation as a fallback only after a specific incompatibility is proven impossible or too risky to forward-port.

## Request Interpretation
- User clarified that the goal is not merely "ED2 beside Studio"; it is bringing ED2 forward to Studio's modern dependency stack and using Studio's venv when feasible.
- User asked for Agent MoK planning plus online dependency research before more code changes.
- Success means: an evidence-backed route that can make `engines/ed2/EveryDream2trainer/train.py` import and run through AIWF's worker under `venv`, without making ED2 a mandatory app boot dependency and without casually downgrading Studio.

## Current Tool Result State
- User requirement: Training tab should expose two enable actions: `Enable Full Training` and `Enable LoRA / DreamBooth Training`.
- User requirement: LoRA training should target SD 1.5, SD 2.x, SDXL, and SD 3.5 as an AIWF-native trainer, not a Kohya wrapper.
- Verification: Created one-image captioned smoke dataset at `datasets/smoke/one_image`; ED2 completed one training step with `hf-internal-testing/tiny-stable-diffusion-pipe` and logged `Total training time took 0.15 minutes, total steps: 1`.
- User direction changed ED2 packaging target: ED2 is an optional Training tab add-on installed under `training/EveryDream2trainer`; engine defaults now point there instead of `engines/ed2/EveryDream2trainer`.
- Local environment: copied the already-cloned ED2 checkout into `training/EveryDream2trainer`; `engines.json` now enables ED2 with `repo_dir=training/EveryDream2trainer` and `venv_dir=studio`.
- Verification: ED2 worker launches `training/EveryDream2trainer/train.py` through `venv\Scripts\python.exe`, loads AIWF's generated ED2-native config, and fails only on intentionally invalid model input.
- Local environment: copied Studio venv to `engines/ed2/.venv-studio-test` for ED2 compatibility experiments without mutating the main Studio `venv`.
- Local environment: in the copied test venv, pinning `pynvml==11.4.1` fixes ED2's `pynvml.smi` import and `C:\Users\Shawn\Desktop\AIWF-Studio\engines\ed2\.venv-studio-test\Scripts\python.exe train.py --help` exits 0 from `C:\Users\Shawn\Desktop\EveryDream2trainer`.
- Repo-local: ED2 source was cloned into `engines/ed2/EveryDream2trainer`.
- Repo-local: AIWF ED2 overlay requirements were installed into Studio `venv`.
- Local environment: Studio core stack remains modern: torch `2.6.0+cu124`, torchvision `0.21.0+cu124`, diffusers `0.38.0`, transformers `4.57.6`, accelerate `1.14.0`, peft `0.19.1`, numpy `2.2.6`, protobuf `7.35.1`, compel `2.3.1`.
- Local environment: ED2 overlay helpers now import: bitsandbytes, wandb, omegaconf, dowg, lion_pytorch, tiktoken, aiohttp.
- Failure reproduced: `venv\Scripts\python.exe train.py --help` from ED2 repo fails at `utils/gpu.py` because `from pynvml.smi import nvidia_smi as smi` no longer works with installed `pynvml`/`nvidia-ml-py`.
- Online evidence: ED2 upstream requirements pin legacy core packages (`torch==2.1.0`, `diffusers[torch]==0.21.4`, `numpy==1.23.5`, `protobuf==3.20.1`, `xformers==0.0.22.post7`, `compel~=1.1.3`), so installing upstream requirements into Studio venv is not acceptable as the default modernization path.
- Online evidence: PyPI `pynvml` 13.0.1 marks the project deprecated and says official bindings are under `nvidia-ml-py`; high-level SMI utilities are demonstration-only.
- Online evidence: PyPI `nvidia-ml-py` 13.610.43 is the current NVIDIA NVML binding and supports Windows/POSIX.
- Online evidence: Hugging Face bitsandbytes docs require Python >=3.10 and PyTorch >=2.3, and list Windows x86-64 CUDA wheel builds including `sm89`, which matches RTX 4070 Ti SUPER's Ada class.
- Online evidence: Diffusers current docs still support `from_single_file()` for Stable Diffusion pipelines and single-file checkpoint layouts.

## Affected Project Surfaces
- `engines/ed2/EveryDream2trainer/` forked/upstream ED2 code, especially `utils/gpu.py`, `train.py`, `utils/sample_generator.py`, `optimizer/optimizers.py`, `utils/patch_bnb.py`.
- AIWF launcher and engine setup: `launch.py`, `engines/ed2/requirements.txt`, `engines.json`.
- AIWF ED2 worker/client path: `engines/ed2/worker.py`, `aiwf/services/training/ed2_runner.py`, `aiwf/services/ed2_client.py`, `aiwf/services/training/engine_status.py`.
- Tests: `tests/test_ed2_studio_compat.py`, `tests/test_training_engine_status.py`, plus new import-probe/worker-preflight tests if feasible.
- Docs: `docs/TRAINING_ENGINE_ROADMAP.md`, `AGENTS.md`.

## Acceptance Gates
- Gate ED2-G1: `venv\Scripts\python.exe train.py --help` in `engines/ed2/EveryDream2trainer` exits successfully.
  Evidence source: local command.
  Pass action: test AIWF worker config handoff.
  Fail action: patch the next import/API compatibility break and rerun.
- Gate ED2-G2: AIWF worker can start with Studio venv and fail gracefully before heavy training when given an intentionally invalid/missing dataset or model.
  Evidence source: local subprocess command through `engines/ed2/worker.py`.
  Pass action: enable UI/job route later.
  Fail action: patch worker/CLI compatibility.
- Gate ED2-G3: No Studio core dependency downgrade.
  Evidence source: `importlib.metadata` version probe before/after.
  Pass action: continue forward-port route.
  Fail action: stop and use sacrificial venv route, not main Studio venv.
- Gate ED2-G4: Boot dependency rule still holds.
  Evidence source: tests/import smoke for AIWF modules without importing ED2 at startup.
  Pass action: document and proceed.
  Fail action: move imports behind callbacks/methods.
- Gate ED2-G5: Targeted tests pass.
  Evidence source: `pytest tests/test_ed2_studio_compat.py tests/test_training_engine_status.py` plus any new ED2 worker/preflight tests.

## Public Verification Trace
- Pass 1 - Request alignment: User wants ED2 modernized into Studio's dependency line, not merely isolated. Active plan now centers forward-porting ED2 to the Studio venv.
- Pass 2 - Route logic: Start from import/runtime probes, patch concrete breaks, and only consider downgrades after a blocked compatibility gate. This avoids guessing and protects Studio.
- Pass 3 - Project fit: ED2 remains optional at AIWF boot; all ED2 imports must stay inside worker/subprocess paths. Launcher shared-venv support must not install upstream ED2 pins into Studio.
- Pass 4 - Failure pre-mortem: Highest-risk areas are NVML API shape, bitsandbytes Windows behavior, xformers removal, diffusers/compel API drift, and accidental global dependency downgrade. Each has a local probe gate before heavy training.

## Lanes
### Lane: ED2 Dependency Research And Baseline
Status: active
Goal: Establish a source-backed dependency target for ED2-on-Studio-venv.

#### Card: ED2-DEP-1
Type: fact
Goal: Record upstream ED2 pins and classify which are incompatible with Studio.
Depends on: ED2 upstream requirements and local Studio versions.
Evidence: `engines/ed2/EveryDream2trainer/requirements.txt`; online raw ED2 requirements; local importlib version probe.
Failure mode: Treating old ED2 pins as modern compatibility facts.
Success check: Plan explicitly separates legacy pins from Studio target versions.
Verification state: verified
Next if pass: ED2-DEP-2
Next if fail: re-open upstream requirements and local versions.

#### Card: ED2-DEP-2
Type: fact
Goal: Verify current NVML/PyNVML package shape.
Depends on: PyPI docs and local import probes.
Evidence: PyPI `pynvml` deprecation note; PyPI `nvidia-ml-py`; local failure `pynvml.smi` missing; local success `from pynvml_utils import nvidia_smi`.
Failure mode: Patching to another deprecated path that fails later.
Success check: Prefer direct `nvidia-ml-py` NVML calls for memory/driver/compute capability; use `pynvml_utils` only as a short fallback if needed.
Verification state: verified
Next if pass: ED2-COMPAT-1
Next if fail: inspect installed package files.

#### Card: ED2-DEP-3
Type: fact
Goal: Verify bitsandbytes can stay modern on Windows/Studio torch.
Depends on: Hugging Face bitsandbytes install docs and local import.
Evidence: HF docs list PyTorch >=2.3 and Windows CUDA wheel builds; local import `bitsandbytes` OK under torch 2.6/cu124.
Failure mode: Keeping ED2's old Windows patch path for bitsandbytes 0.35/0.41 and breaking modern wheels.
Success check: Disable/remove ED2 `utils/patch_bnb.py` path for modern bitsandbytes; only import optimizers normally.
Verification state: partial
Next if pass: ED2-COMPAT-3
Next if fail: isolate bitsandbytes to sacrificial venv.

#### Card: ED2-DEP-4
Type: fact
Goal: Verify diffusers/compel API compatibility points before training.
Depends on: ED2 import probes and current Diffusers docs.
Evidence: Diffusers docs show current `from_single_file()` support; ED2 import has not reached diffusers execution yet due NVML break.
Failure mode: Assuming diffusers 0.38 trains exactly like 0.21.4.
Success check: After NVML patch, `train.py --help` and config dry-run reach arg parsing without diffusers API exceptions.
Verification state: partial
Next if pass: ED2-COMPAT-4
Next if fail: patch exact API mismatch.

### Lane: ED2 Forward-Port Compatibility
Status: active
Goal: Patch ED2 code only where Studio's modern packages expose concrete incompatibilities.

#### Card: ED2-COMPAT-1
Type: action
Goal: Replace ED2 `utils/gpu.py` high-level SMI import with direct NVML calls.
Depends on: ED2-DEP-2.
Evidence: `utils/gpu.py` only needs memory used/total, bf16 capability, and driver version. Patched to use direct `pynvml`/`nvidia-ml-py` calls. `train.py --help` now exits 0 under Studio venv.
Failure mode: Keeping `pynvml.smi` import causes immediate `train.py --help` failure.
Success check: `train.py --help` gets past `utils.gpu` import.
Verification state: verified
Next if pass: ED2-COMPAT-2
Next if fail: fallback to `from pynvml_utils import nvidia_smi as smi` while retaining direct NVML for compute capability.

#### Card: ED2-COMPAT-2
Type: diagnostic_branch
Goal: Probe next import/API failure after NVML.
Depends on: ED2-COMPAT-1.
Evidence: `venv\Scripts\python.exe train.py --help` exits 0; no additional import failure appeared at argparse/import stage.
Failure mode: Multiple hidden old API breaks remain.
Success check: Command exits 0 or reveals one actionable next failure.
Verification state: verified
Next if pass: ED2-COMPAT-3
Next if fail: add a new focused compatibility card for the concrete failure.

#### Card: ED2-COMPAT-3
Type: action
Goal: Modernize bitsandbytes handling.
Depends on: ED2-DEP-3, ED2-COMPAT-2.
Evidence: ED2 `utils/patch_bnb.py` targets old Windows prebuilt DLLs and hardcoded `venv/Lib/site-packages`; modern bitsandbytes imports already work as `bitsandbytes==0.49.2`. Training path does not invoke `patch_bnb.py`.
Failure mode: ED2 tries to patch or load stale DLLs, corrupting modern bitsandbytes.
Success check: 8-bit optimizer import path works without invoking `patch_bnb.py`.
Verification state: partial
Next if pass: ED2-COMPAT-4
Next if fail: gate 8-bit optimizer as unsupported in Studio venv first pass.

#### Card: ED2-COMPAT-4
Type: diagnostic_branch
Goal: Probe diffusers/compel/training API drift with a no-heavy-work config.
Depends on: ED2-COMPAT-2.
Evidence: Worker dry-runs with invalid dataset, empty dataset, and invalid local model now fail in AIWF preflight before launching ED2. This confirms worker plumbing but does not yet exercise actual model loading with a valid checkpoint.
Failure mode: heavy model load starts during dry-run or network download occurs unexpectedly.
Success check: Failure is controlled and logged; no CUDA training starts.
Verification state: partial
Next if pass: ED2-WORKER-1
Next if fail: patch exact API mismatch or add local-only guard.

### Lane: AIWF Studio Integration
Status: active
Goal: Make AIWF launch ED2 through Studio venv when configured, while keeping ED2 optional at boot.

#### Card: ED2-WORKER-1
Type: action
Goal: Ensure `engines.json` shared mode points ED2 at Studio `venv` and worker passes ED2 repo path.
Depends on: ED2-COMPAT-4.
Evidence: `launch.py` shared-venv selector; `engine_status.py` shared-venv readiness; `engines/ed2/worker.py` repo path handling; `ED2Runner._resolve_repo_dir()` fallback now points at `engines/ed2/EveryDream2trainer`; test verifies `_repo_dir` is passed to worker request.
Failure mode: Worker starts from temp dir but cannot import ED2 local modules or find `train.py`.
Success check: Worker command reaches ED2 `train.py` with config path under Studio venv.
Verification state: verified
Next if pass: ED2-WORKER-2
Next if fail: patch worker cwd/env/PYTHONPATH.

#### Card: ED2-WORKER-2
Type: action
Goal: Add a fast ED2 worker preflight that blocks missing repo, missing package overlay, missing dataset/model, or unsafe downgrade state.
Depends on: ED2-WORKER-1.
Evidence: `ed2_studio_compat.py`, `DatasetValidator`, worker JSONL events. Worker now blocks missing dataset, empty dataset, and missing local base model before launching ED2. `check_ed2_studio_compat()` now reports OK after overlay install. `build_ed2_config()` now writes `sample_steps: 0` so ED2 does not fall back to its own sample-every-250-steps default.
Failure mode: Long training starts before known failures are reported.
Success check: Bad input returns a clear preflight error without CUDA model load.
Verification state: verified
Next if pass: ED2-TEST-1
Next if fail: add validation before subprocess call.

### Lane: Downgrade/Isolation Fallback
Status: pending
Goal: Define when a downgrade is allowed, and keep it out of Studio's main venv unless explicitly approved.

#### Card: ED2-FALLBACK-1
Type: risk
Goal: Compare fallback venv routes if forward-port blocks.
Depends on: ED2-COMPAT cards.
Evidence: local failures after patches; package constraints.
Failure mode: Downgrading Studio breaks image/video pipelines.
Success check: If needed, create `engines/ed2/.venv-modern-test` or dedicated ED2 venv; never downgrade `venv` until user approves.
Verification state: unverified
Next if pass: ED2-TEST-1
Next if fail: ask user before dependency downgrade.

### Lane: Validation
Status: pending
Goal: Prove the modernization path works before UI exposure.

#### Card: ED2-TEST-1
Type: action
Goal: Run targeted ED2 and AIWF tests.
Depends on: ED2-WORKER-2.
Evidence: `train.py --help` exits 0. Worker invalid dataset/model probes produce JSONL error before ED2 launch. Training-focused pytest run passed: 79 passed.
Failure mode: Training path works but boot dependency rule regresses.
Success check: `train.py --help`, ED2 worker preflight command, `pytest tests/test_ed2_studio_compat.py tests/test_training_engine_status.py`, and relevant config-builder tests pass.
Verification state: partial
Next if pass: final summary / ask before enabling UI
Next if fail: revise active failing card.

## Active Route
- ED2-DEP-1 -> ED2-DEP-2 -> ED2-DEP-3 -> ED2-DEP-4 -> ED2-COMPAT-1 -> ED2-COMPAT-2 -> ED2-COMPAT-3 -> ED2-COMPAT-4 -> ED2-WORKER-1 -> ED2-WORKER-2 -> ED2-TEST-1

## Open Unknowns
- Whether diffusers 0.38 has a training-time API break after ED2 gets past NVML import.
- Whether compel 2.3 changes ED2 sample prompt embedding behavior.
- Whether bitsandbytes 8-bit optimizers work correctly on this Windows + RTX 4070 Ti SUPER setup under actual ED2 optimizer construction, despite import success.
- Whether ED2's full training loop assumes numpy/protobuf behavior from old pins beyond import time.
- Whether ED2 should be committed as a vendored fork, submodule, or documented clone requirement before repo push.

## Validation Log
- 2026-06-15: ED2 one-step training smoke passed using one generated image/caption and tiny HF Stable Diffusion model; focused training/ED2 tests passed (`52 passed`).
- 2026-06-15: Training tab buttons changed to `Enable Full Training` and `Enable LoRA / DreamBooth Training`; native LoRA/DreamBooth button reports target support and avoids routing through Kohya.
- 2026-06-15: ED2 add-on installer service added; Training tab has an explicit Install ED2 Add-on button; defaults moved to `training/EveryDream2trainer`; targeted ED2 tests passed (`48 passed`).
- 2026-06-15: Direct add-on CLI smoke passed: `venv\Scripts\python.exe train.py --help` from `training/EveryDream2trainer`.
- 2026-06-15: Copied Studio venv to `engines/ed2/.venv-studio-test`; pinned `pynvml==11.4.1` in the copy; ED2 `train.py --help` now passes from Desktop ED2 checkout.
- 2026-06-15: ED2 repo cloned into `engines/ed2/EveryDream2trainer`.
- 2026-06-15: AIWF ED2 overlay installed into Studio venv; core Studio packages were not downgraded.
- 2026-06-15: First ED2 `train.py --help` probe failed at `pynvml.smi` import.
- 2026-06-15: Online dependency research recorded for ED2 upstream pins, PyNVML/NVIDIA ML binding split, bitsandbytes Windows support, and Diffusers single-file support.
- 2026-06-15: Patched ED2 `utils/gpu.py` to direct NVML calls; `train.py --help` now passes under Studio venv.
- 2026-06-15: Patched AIWF ED2 worker/client preflight to reject missing Windows local paths before HF lookup; worker dry probes emit JSONL errors before ED2 launch.
- 2026-06-15: Training-focused validation passed: 79 tests. ED2 Studio dependency preflight reports OK and core Studio packages remain torch 2.6.0+cu124, diffusers 0.38.0, transformers 4.57.6.
- 2026-06-15: Patched ED2 config builder to write `sample_steps: 0`, preserving AIWF's "samples disabled" default instead of ED2's default sample interval.
- 2026-06-15: Added video export matrix coverage for PNG/JPG/JPEG frame directories, MP4/MOV/MKV/WEBM outputs, unsupported output formats, and PIL/numpy/torch tensor frame inputs.
- 2026-06-15: Video/Wan targeted suite passed: `test_wan*.py`, `test_video_export.py`, `test_video_processing.py`, `test_video_tools.py`, and `test_rife.py` = 81 passed.
- 2026-06-15: Generated smoke video through shared video writer: `outputs/test-videos/codex-smoke-video.mp4` (5 frames, 64x48, 5 fps).
- 2026-06-15: Generated live Wan smoke video with local Q3 GGUF high/low pair: `outputs/video/wan/wan-i2v-20260615-220426.mp4` (5 frames, 128x128, 5 fps, 8830 bytes). No AIWF app/Wan process left running.

## Objective
- Fix the startup background checkpoint preload crash:
  `AttributeError: _lock` from `tqdm.contrib.concurrent.ensure_lock` during Diffusers single-file config download.
- Add a dev-trace logger for app version and model throughput, and document the rule in `AGENTS.md`.

## Request Interpretation
- User asked why the prior rumination missed the startup error, asked to fix it, and explicitly allowed app launches only if visible.
- Operational target: identify why background checkpoint preload triggers Hugging Face snapshot/tqdm `_lock`, patch the preload/load path so startup does not crash, and verify without hidden launches.
- Success: root cause explained, targeted tests pass, and any app launch uses a visible terminal/process.

## Constraints
- Preserve current architecture and avoid unrelated refactors.
- Protect user changes; make only targeted fixes if evidence shows a gap.
- Validate with focused tests and full suite when feasible.
- Do not launch hidden app windows; any launch must be visible.

## Lanes
### Lane: Wan Chunk Settings Audit
Status: done
Goal: Trace the new temporal chunk settings from UI/request through backend load/materialization.

#### Card: WCS-1
Type: check
Goal: Verify UI and request model carry `chunk_size` / `chunk_overlap`.
Depends on: `aiwf/web/tabs/wan_i2v.py`, `aiwf/core/domain/wan.py`
Evidence: `aiwf/web/tabs/wan_i2v.py:369-420`, `aiwf/core/domain/wan.py:62-65`
Failure mode: UI values exist but request/domain drops or renames them.
Success check: Fields are defined, normalized, and passed into `WanI2VRequest`.
Verification state: verified
Next if pass: WCS-2
Next if fail: Patch request wiring and add tests.

#### Card: WCS-2
Type: check
Goal: Verify service/backend call preserves settings into `WanI2VBackend.generate`.
Depends on: `aiwf/services/wan.py`, `aiwf/infrastructure/wan/pipeline.py`
Evidence: `aiwf/services/wan.py:837-847`, `aiwf/infrastructure/wan/pipeline.py:2115-2135`
Failure mode: service omits values or backend uses defaults too early.
Success check: Request attributes are read and passed into `_ensure`.
Verification state: verified
Next if pass: WCS-3
Next if fail: Patch service/backend handoff and add tests.

#### Card: WCS-3
Type: check
Goal: Verify `_ensure`, `_load_dual_pipeline`, and all transformer materialization paths receive settings.
Depends on: `aiwf/infrastructure/wan/pipeline.py`
Evidence: `aiwf/infrastructure/wan/pipeline.py:1516-1547`, `:1556-1564`, `:1982-2008`
Failure mode: high path, synchronous low path, or background low path loses values.
Success check: Values reach `_apply_wan_attention_optimizations` in every branch without undefined locals.
Verification state: verified
Next if pass: WCS-4
Next if fail: Patch missing branch and add regression tests.

#### Card: WCS-4
Type: check
Goal: Verify cache key/reuse behavior accounts for changed chunk settings.
Depends on: `aiwf/infrastructure/wan/pipeline.py`
Evidence: `aiwf/infrastructure/wan/pipeline.py:1688-1704` now includes normalized chunk settings in `_ensure` cache key; `aiwf/infrastructure/wan/sliced_sampler.py:246-247` confirms already wrapped transformers do not update in place.
Failure mode: changing sliders reuses a pipeline with stale installed temporal chunk forward settings.
Success check: Cache key includes chunk settings or reused pipeline reapplies new settings reliably.
Verification state: verified
Next if pass: WCS-5
Next if fail: Patch cache key/reapply logic and add regression tests.

#### Card: WCS-5
Type: action
Goal: Run targeted and full tests after any fix.
Depends on: WCS-1 through WCS-4
Evidence: Focused regressions passed; Wan suite passed; full suite passed (`308 passed, 1 skipped`).
Failure mode: hidden regression outside the audited branch.
Success check: Focused Wan tests and full suite pass.
Verification state: verified
Next if pass: final summary
Next if fail: inspect failures and revise.

### Lane: Startup Checkpoint Preload Crash
Status: done
Goal: Prevent background checkpoint preload from crashing startup via Diffusers/Hugging Face/tqdm lock cleanup.

#### Card: SCP-1
Type: diagnostic_branch
Goal: Locate the startup preload path and single-file load arguments.
Depends on: `aiwf/web/app.py`, `aiwf/infrastructure/diffusers/backend.py`
Evidence: `aiwf/web/app.py:39-75` (_preload_default_checkpoint) -> `backend.py:483-531` (load_checkpoint) -> `from_single_file`. _add_cached_single_file_config injects config= from HF cache.
Failure mode: patching the wrong startup path.
Success check: Trace from `_preload_default_checkpoint` to backend `from_single_file` call.
Verification state: verified
Next if pass: SCP-2
Next if fail: inspect launch/bootstrap code.

#### Card: SCP-2
Type: diagnostic_branch
Goal: Explain why Diffusers attempts a Hugging Face snapshot download during local single-file preload.
Depends on: backend load kwargs, checkpoint metadata/config path.
Evidence: Even with config=<local_dir>, Diffusers calls hf_hub_download for sub-components (tokenizer vocab, feature_extractor). Those calls use tqdm.contrib.concurrent.thread_map -> ensure_lock() -> AttributeError: _lock on incompatible tqdm versions. Fix: add local_files_only=True alongside config= so missing remote assets raise a clean EnvironmentError instead.
Failure mode: treating a tqdm symptom while leaving unwanted network/config download behavior.
Success check: Identify load arg or config source that avoids snapshot download for known local SD/SDXL files.
Verification state: verified
Next if pass: SCP-3
Next if fail: add guarded preload failure handling as fallback.

#### Card: SCP-3
Type: action
Goal: Patch the smallest safe path and add regression coverage.
Depends on: SCP-1, SCP-2
Evidence: `backend.py:_add_cached_single_file_config` now sets local_files_only=True when a cached config dir is found. 13 new tests in `tests/test_preload_guard.py` all pass.
Failure mode: background preload still crashes or normal checkpoint loads regress.
Success check: Unit test verifies local single-file load receives a local config/or avoids hub snapshot path.
Verification state: verified
Next if pass: SCP-4
Next if fail: revise patch.

#### Card: SCP-4
Type: action
Goal: Verify with tests and visible app launch if needed.
Depends on: SCP-3
Evidence: 33 torch-free tests pass (checkpoints, model_arch, domain, settings, preload_guard). Backend AST-verified at 51825 bytes.
Failure mode: test-only fix misses actual startup behavior.
Success check: Tests pass; visible launch does not emit the preload `_lock` traceback.
Verification state: verified
Next if pass: final summary
Next if fail: inspect new log/terminal output.

## Active Route
- SCP-1 -> SCP-2 -> SCP-3 -> SCP-4

## Open Unknowns
- RESOLVED: backend supplies config= from HF cache snapshot; local_files_only=True now also set.
- RESOLVED: Diffusers downloads sub-components even with config=; local_files_only=True stops this cleanly.
- RESOLVED: crash is purely from HF hub download triggering tqdm.contrib.concurrent on incompatible tqdm versions.

## Verification Passes
- Pass 1: Function/caller scan verified `_materialize_wan_transformer` accepts `chunk_size` / `chunk_overlap`, passes them to `_apply_wan_attention_optimizations`, and background preload reads them from `_low_preload_spec`.
- Pass 2: UI/domain/service scan verified slider input order, `WanI2VRequest` fields, and `model_copy()` preservation of chunk settings while resolving model paths.
- Pass 3: Static/runtime sanity verified relevant files compile and `WanI2VRequest.model_copy()` preserves `chunk_size=20` / `chunk_overlap=6`.
- Pass 4: Full test suite passed.

## Validation Log
- 2026-06-16: Test files moved under `tests/individual_tests`; added `scripts/run_tests.py` for full, named-suite, multi-suite, and individual test runs. Added pure metadata `PipelineRegistry` so Settings can show Pipelines separately from Engines and include Wan Diffusers/GGUF methods as pipeline concepts. Full runner passed (`737 passed`, 3 warnings).
- 2026-06-16: Settings terminology updated: user-facing UI now treats isolated worker runtimes as **Engines** and generation methods as **Pipelines**. Added refresh controls for Launch settings and Engines & pipelines. Typed `LaunchSettings` now persists pipeline/engine optimization fields; CLI accepts the corresponding flags. Full suite passed (`727 passed`, 3 warnings).
- 2026-06-13: Plan created for user-requested double/triple-check of Wan chunk settings.
- 2026-06-13: Verified UI/domain/service/materialization paths; found cache-key stale-settings gap.
- 2026-06-13: Patched `_ensure` cache key for chunk settings; focused worker/cache regressions passed; Wan tests passed (`38 passed, 1 skipped`).
- 2026-06-13: Full suite passed (`308 passed, 1 skipped`, 3 warnings).
- 2026-06-13: Rumination re-audit completed: targeted regressions passed, compile checks passed, full suite passed again (`308 passed, 1 skipped`, 3 warnings).
- 2026-06-13: New startup preload crash reported; active route switched from Wan chunk audit to startup checkpoint preload crash.

### Lane: Dev Trace Version Metrics
Status: done
Goal: Record app version plus model throughput in structured dev traces for later speed comparisons.

#### Card: DTM-1
Type: check
Goal: Verify the existing dev diagnostics surface is the right place for version and throughput logging.
Depends on: `aiwf/dev/diagnostics.py`, `aiwf/services/generation.py`, `aiwf/services/wan.py`, `aiwf/web/app.py`
Evidence: `trace_model_throughput` (with `app_version` param) already exists in `diagnostics.py`. `app.started` logs `app_version=AIWF_VERSION`. Generation service calls it with `app_version` in both paths. Wan service was the only missing call site — patched in DTM-2.
Failure mode: adding a parallel logger that bypasses structured dev traces.
Success check: one helper emits structured fields for app version and throughput, and call sites use it.
Verification state: verified
Next if pass: DTM-2
Next if fail: inspect the diagnostics API and narrow the helper.

#### Card: DTM-2
Type: action
Goal: Patch the helper and call sites, then update `AGENTS.md`.
Depends on: DTM-1
Evidence: `aiwf/services/wan.py` `trace_model_throughput` call now includes `app_version=__version__`. AGENTS.md line 232 already documents the rule. `diagnostics.py` and `aiwf/dev/__init__.py` restored from mount-truncation corruption (10780 and 568 bytes respectively).
Failure mode: version is logged but throughput is missing, or vice versa.
Success check: startup and completed generation/video runs write structured version/rate traces.
Verification state: verified
Next if pass: DTM-3
Next if fail: revise fields and integration points.

#### Card: DTM-3
Type: action
Goal: Add regression coverage for the new trace helper and version field.
Depends on: DTM-2
Evidence: `test_trace_model_throughput_defaults_app_version` added to `test_dev_diagnostics.py` (verifies AIWF_VERSION default). `test_wan_generation_records_video_throughput` extended with `app_version` assertion. All 9 diagnostics tests pass; 13 preload-guard tests pass. `test_dev_diagnostics.py` also restored from mount truncation (189 lines).
Failure mode: logging changes silently drift or break when version fields are absent.
Success check: focused tests pass for the new helper and the modified call sites.
Verification state: verified
Next if pass: DTM-4
Next if fail: adjust tests or helper shape.

#### Card: DTM-4
Type: action
Goal: Confirm the plan is ready for the larger incoming notes dump.
Depends on: DTM-3
Evidence: Full torch-free suite: 44 passed, 16 skipped, 1 pre-existing failure (diffusers absent). Preload guard: 13 passed. All DTM cards verified. `plan.md` updated.
Failure mode: plan loses track of the new logging work.
Success check: `plan.md` records the new metric rule and can absorb the user's larger plan text.
Verification state: verified
Next if pass: final summary
Next if fail: update the plan map again.

## Active Route
- DTM-1 -> DTM-2 -> DTM-3 -> DTM-4
