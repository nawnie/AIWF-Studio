# AIWF Studio research notes

**Prepared:** 2026-06-16
**Purpose:** identify research gaps, experiment candidates, and evidence still needed before promoting optimization features.

This file is not a chat log. It is a working research map for deciding where the next deep-dive pass should go.

---

## Current thesis

AIWF Studio should first implement an optimization substrate: profile objects, capability detection, receipts, fallback behavior, and benchmark suites. Once that exists, individual accelerators can be evaluated safely. Without that substrate, every new backend increases instability faster than it increases generation speed.

**Evidence base:** Diffusers documents multiple performance and memory features, but many are model-dependent, stateful, or experimental; PyTorch/NVIDIA compile and engine paths have meaningful first-run/build costs and static-profile constraints [S01][S02][S07][S08][S12][S13].

---

## Research gap map

| Gap ID | Question | Why it matters | Recommended next action | Sources |
|---|---|---|---|---|
| G-001 | Which Diffusers/Torch/CUDA/cu124 combination is the best stable AIWF baseline on Windows? | All backend choices depend on this. | Build a small Windows matrix around current AIWF constraints; test boot, SD1.5, SDXL, inpaint, ControlNet, LoRA. | S01; S02; S09; S11 |
| G-002 | Which Compel version is compatible with `transformers>=4.44,<5` and AIWFâ€™s SDXL prompt embeddings? | Compel mainline appears Transformers-5 oriented. | Pin a known-compatible release and write prompt embedding tests. | S51; S52; S53 |
| G-003 | Does channels-last provide measurable speedup on RTX 4070 Ti SUPER for SD1.5/SDXL? | Low-risk candidate for Balanced default. | Run benchmark suite A/B with and without channels-last. | S02 |
| G-004 | Does xFormers beat PyTorch SDPA on current Torch/cu124 for AIWF target workflows? | xFormers used to be a default recommendation, but SDPA is now strong. | Benchmark xFormers vs SDPA across SD1.5, SDXL, ControlNet, inpaint. | S02; S03; S09; S10 |
| G-005 | Is regional `torch.compile` worth it for fixed SDXL 1024 txt2img on Windows? | Could be major speedup, but Windows and compile latency are risks. | Test Linux first, then Windows; record compile time, recompile count, and steady-state. | S02; S08; S11 |
| G-006 | Can compiled profiles coexist with AIWF LoRA UX? | LoRA is central; compile can recompile/fail. | Test static LoRA, hotswap same-rank, hotswap different-rank, text-encoder LoRA. | S67; S68; S08 |
| G-007 | Which SDXL scheduler default avoids DPM++ artifacts while keeping speed? | User-facing image quality. | Compare Euler, DPM++ Karras, UniPC, solver settings at 20/30/40 steps. | S36; S37; S38; S42 |
| G-008 | Is model CPU offload acceptable on 16 GB GPUs for SDXL + ControlNet + hires? | Determines Low VRAM ladder. | Measure latency and VRAM against no offload, VAE tiling, sequential offload. | S01 |
| G-009 | How often does VAE tiling create visible tone shifts in hires fix? | Tiling may be needed for high-res. | Create before/after grids and artifact labels for SD1.5/SDXL high-res. | S01; S65 |
| G-010 | Which SDXL fp16 VAE path is best for AIWF defaults? | Original VAE upcast can cost memory; fp16 fix changes output slightly. | Compare original/upcast vs fp16-fix on prompts and img2img/inpaint. | S65; S66 |
| G-011 | Which ControlNet preprocessors can be packaged lazily with minimal Windows pain? | ControlNet UX depends on preprocessor availability. | Start canny; test depth/pose/DWPose optional extras. | S59; S60; S61; S63 |
| G-012 | Does SDXL ControlNet quality justify default exposure in AIWF? | SDXL ControlNet ecosystem remains mixed. | Test selected SDXL ControlNet models with receipts and user grids. | S63 |
| G-013 | Which Fast Mode should ship first? | Fast generation is a user-visible differentiator. | Compare LCM-LoRA, SDXL Lightning 4/8, SDXL Turbo. Avoid Hyper-SD/TCD/PCM until recipe registry is solid. | S39; S44; S46; S48; S49; S50 |
| G-014 | Can torchao FP8 help SDXL on Ada in practice? | Hardware supports FP8, but model/module support and quality are uncertain. | Test only after receipt system; start with supported Linear-heavy components. | S23; S27; S30 |
| G-015 | Is TensorRT worth a consumer Engine Lab? | Potential speedup, but high UX/support cost. | Prototype fixed SDXL 1024 no-LoRA engine; then add LoRA conversion/refit research. | S12; S13; S14; S17 |
| G-016 | Can Torch-TensorRT Mutable Module solve LoRA dynamism better than classic TensorRT? | Would be valuable if reliable. | Research branch after compile/receipt substrate. | S16; S17; S18 |
| G-017 | Should AIWF support GGUF model loading? | Could help memory/future community models. | Test Diffusers GGUF loader for specific models; document config requirements. | S33 |
| G-018 | What metadata should AIWF embed in generated images? | Reproducibility and support. | Use benchmark receipt subset as image metadata sidecar. | S08 |

---

## Immediate research experiments

### Experiment R1 â€” SDPA vs xFormers

**Hypothesis:** On modern PyTorch/cu124, native SDPA will be close enough to xFormers that xFormers should remain optional.

**Protocol**
- SD1.5 txt2img 512, SDXL txt2img 1024, SDXL inpaint, SD1.5 ControlNet canny.
- Same seed/prompt/scheduler.
- Compare median time, p90, peak VRAM, visual hashes.
- Run on Windows target.

**Promotion decision**
- If xFormers gives >=10% speed or >=20% VRAM improvement with no install instability, keep experimental or beta.
- If marginal, leave off by default.

**Sources:** S02; S03; S09; S10

---

### Experiment R2 â€” Regional compile fixed SDXL 1024

**Hypothesis:** Regional compile may give useful steady-state speed with lower compile latency than full compile, but Windows support may block default exposure.

**Protocol**
- SDXL 1024, batch 1, no LoRA, no ControlNet, fixed scheduler.
- Compare eager Balanced, regional compile, full compile.
- Record compile time, first generation, steady-state median, recompile count.
- Repeat with one static LoRA after LoRA manager exists.

**Promotion decision**
- Beta only if compile time is acceptable, recompile count stays zero for fixed profile, and fallback works.

**Sources:** S02; S08; S11; S67

---

### Experiment R3 â€” Scheduler defaults

**Hypothesis:** SD1.5 can default to DPM++/DPMSolverMultistep or Euler depending model; SDXL should avoid low-step DPM++ without mitigation.

**Protocol**
- SD1.5: Euler 25, DPM++ 20/25, UniPC 15/20.
- SDXL: Euler 30, DPM++ 30 with Karras, DPM++ 30 without Karras, UniPC 20/30.
- Human grid review plus artifact labels.

**Promotion decision**
- Set defaults by family and record scheduler config in profile registry.

**Sources:** S35; S36; S37; S38; S42

---

### Experiment R4 â€” Fast Mode first ship

**Hypothesis:** LCM-LoRA and SDXL Lightning 4/8-step are the best first Fast Mode candidates; SDXL Turbo should be separate-model beta; Hyper-SD/TCD/PCM should wait.

**Protocol**
- Compare normal SDXL 30-step, LCM-LoRA 4/8, Lightning 4/8, Turbo 1/2/4.
- Use method-card-correct CFG/scheduler/steps.
- Review speed, prompt adherence, face/hands/detail artifacts.

**Promotion decision**
- Ship the method with clearest UX and least dependency burden first.

**Sources:** S39; S44; S46; S49

---

### Experiment R5 â€” Low VRAM ladder

**Hypothesis:** For 16 GB target GPU, VAE tiling and model CPU offload solve more real OOM cases than quantization for SDXL/ControlNet/hires.

**Protocol**
- SDXL + ControlNet + hires.
- Compare no offload, VAE slicing, VAE tiling, model offload, sequential offload.
- Track latency and VRAM.

**Promotion decision**
- Set Low VRAM fallback order and warnings.

**Sources:** S01; S65

---

## Evidence watchlist

Track these upstream areas monthly or per release:

| Area | Watch for | Sources |
|---|---|---|
| Diffusers attention backend dispatcher | API stabilization and backend support changes | S03 |
| Diffusers memory/offload hooks | group offload compatibility and hook behavior | S01 |
| PyTorch compile | Windows GPU support, compile cache, Diffusers graph-break improvements | S05; S06; S08; S11 |
| xFormers | Wheel compatibility with AIWF's Torch/CUDA pin | S09; S10 |
| Compel | Transformers 4-compatible release or migration path | S52 |
| torchao | Diffusers model coverage beyond Linear-heavy components | S30 |
| TensorRT/Torch-TensorRT | LoRA refit and Windows-friendly local workflows | S13; S16; S17 |
| ModelOpt | Cache Diffusion practicality outside expert TensorRT workflows | S19; S34 |
| Fast methods | New model cards/recipes for LCM, Lightning, Hyper-SD, TCD, PCM | S39; S40; S46; S48; S50 |
| FP4/NVFP4 | Consumer Blackwell hardware and Diffusers support | S24; S25; S26 |

---

## Research route options

### Route A â€” Stability-first
Focus on profile substrate, receipts, baseline SDPA/channels-last, schedulers, LoRA manager, and inpaint/ControlNet quality. This should be the default continuation path.

### Route B â€” Speed lab
After Route A foundation, test xFormers, regional compile, and Fast Mode. This produces visible performance wins without committing to heavy engine dependencies.

### Route C â€” Engine lab
Prototype TensorRT/Torch-TensorRT only after benchmark receipts and fixed-profile cache design exist.

### Route D â€” Quant lab
Explore torchao FP8 and bitsandbytes selected components after baseline benchmarks exist. Avoid broad quantization claims until component support is proven.

### Route E â€” Quality lab
Focus on FreeU, PAG, SDXL refiner, hires fix, VAE choices, inpaint edge handling, and ControlNet preprocessor recommendations.

---

## Recommended next research command

Run **Route A** first. It prepares the app to safely run Routes Bâ€“E without destabilizing normal generation.
