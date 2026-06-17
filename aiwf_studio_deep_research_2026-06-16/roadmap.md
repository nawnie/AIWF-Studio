# Prioritized implementation roadmap

**Prepared:** 2026-06-16

This roadmap converts the research report into AIWF Studio implementation stages. It prioritizes maintainability, Windows NVIDIA support, and local-first consumer usability.

---

## Roadmap overview

| Phase | Theme | Outcome |
|---|---|---|
| Phase 0 | Stabilize dependency and baseline behavior | Known-good Safe/Balanced profiles. |
| Phase 1 | Optimization substrate | Profiles, capability detection, receipts, fallback. |
| Phase 2 | Workflow quality | Prompt, inpaint, ControlNet, hires, SDXL refiner. |
| Phase 3 | Safe speed experiments | xFormers, compile, FreeU/PAG, LCM/Lightning. |
| Phase 4 | Advanced acceleration lab | TensorRT, Torch-TensorRT, torchao, ModelOpt. |
| Phase 5 | Future model formats/precision | GGUF, FP8 refinement, Blackwell FP4/NVFP4 later. |

---

## Do now

### 1. Implement profile and receipt domain models

**Tasks**
- Add `OptimizationProfile`.
- Add `CapabilityReport`.
- Add `OptimizationPlan`.
- Add `BenchmarkReceipt`.
- Add profile versioning and schema migration.

**Why now**
- It prevents global-state optimization bugs.
- It makes later experiments safe.
- It gives support/debugging a shared language.

**Readiness:** production architecture
**Sources:** S01; S02; S08

---

### 2. Build capability detection and lazy optional dependency probes

**Tasks**
- Core probe: OS, Python, GPU, VRAM, compute capability, driver, CUDA runtime.
- Core packages: torch, diffusers, transformers, accelerate, peft, safetensors, compel.
- Optional packages lazy only: xFormers, flash-attn, sageattention, bitsandbytes, torchao, optimum-quanto, TensorRT, Torch-TensorRT, ONNX Runtime, ModelOpt.

**Why now**
- Avoids mandatory heavy dependencies at boot.
- Prevents unsupported Windows options from appearing.

**Readiness:** production architecture
**Sources:** S09; S10; S11; S16; S28; S30; S32; S34

---

### 3. Lock baseline profiles

**Tasks**
- Safe profile.
- Balanced profile.
- Quality profile shell.
- Low VRAM profile.
- Fast Mode shell with no default fast method.

**Why now**
- Users need stable defaults before advanced flags.
- Bug reports need a reproducible Safe path.

**Readiness:** production
**Sources:** S01; S02; S35; S36; S37; S42

---

### 4. Scheduler recipe registry

**Tasks**
- SD1.5: DPM++/DPMSolverMultistep, Euler, UniPC presets.
- SDXL: DPM++ with Karras/mitigation, Euler fallback, 1024 target warnings.
- Fast methods: separate recipe class, not ordinary scheduler option.

**Why now**
- Scheduler choice gives immediate speed/quality gains without new dependencies.
- Prevents SDXL low-step artifact defaults.

**Readiness:** production
**Sources:** S35; S36; S37; S38; S39; S42

---

### 5. VAE and memory policy

**Tasks**
- VAE slicing for batch >1.
- VAE tiling as visible high-res/Low VRAM option.
- Model CPU offload Low VRAM.
- Sequential offload emergency only.
- VAE selector including SDXL fp16-fix as optional asset.

**Why now**
- Directly addresses OOM/high-res pain.
- Minimal optional dependency footprint.

**Readiness:** production fallback
**Sources:** S01; S65; S66

---

### 6. Prompt embedding service

**Tasks**
- Wrap Compel behind `PromptEmbeddingService`.
- Support SD1.5 and SDXL pooled embeddings.
- Add negative prompt embeddings.
- Add version probe and Transformers 4.x compatibility tests.
- Pin known-compatible Compel.

**Why now**
- Weighted prompts are user-facing quality basics.
- Compel/Transformers drift is a real compatibility risk.

**Readiness:** production with pinning
**Sources:** S51; S52; S53

---

### 7. LoRA manager service

**Tasks**
- Own load/unload/enable/disable/fuse/unfuse state.
- Track LoRA rank, target modules, text encoder involvement.
- Prepare compile-aware hotswap only in compile profile.
- Store LoRA state in receipts.

**Why now**
- LoRA is central to consumer workflows.
- Compile/TensorRT plans depend on correct LoRA state.

**Readiness:** production
**Sources:** S67; S68

---

### 8. Inpaint and ControlNet workflow correctness

**Tasks**
- Inpaint-specific checkpoint preference.
- Mask blur, padding crop, mask hash receipts.
- SDXL mask blur regression test.
- ControlNet model-family matching.
- Lazy preprocessor dependencies; canny core first.

**Why now**
- Quality instability often comes from workflow mismatch, not GPU speed.
- ControlNet dependencies can bloat startup if not isolated.

**Readiness:** production/beta depending SDXL ControlNet model
**Sources:** S56; S57; S58; S59; S60; S61; S63

---

## Experiment behind flag

### 1. xFormers attention backend

**Entry criteria**
- xFormers installed and compatible.
- Kernel availability logged.
- Benchmark against SDPA shows improvement.

**Exit criteria**
- >=10% speed or >=20% VRAM improvement on target profile.
- No dependency conflict with pinned Torch/CUDA.
- Windows verified.

**Sources:** S09; S10

---

### 2. Regional `torch.compile`

**Entry criteria**
- compile probe passes.
- Fixed 1024 SDXL txt2img or fixed 512 SD1.5 txt2img profile.
- LoRA state static or prepared.

**Exit criteria**
- Compile time recorded.
- Steady-state improvement meaningful.
- No recompile churn.
- Windows behavior clear.

**Sources:** S02; S05; S06; S08; S11; S67

---

### 3. Full UNet compile fixed profiles

**Entry criteria**
- Regional compile tested first.
- No dynamic resolution or dynamic ControlNet.
- Receipt system in place.

**Exit criteria**
- Clear speed gain after amortizing compile.
- No graph-break failures.
- Fallback eager path works.

**Sources:** S02; S08

---

### 4. FreeU and PAG

**Entry criteria**
- Quality profile UI can show visible output-changing toggles.
- Receipts record parameters.

**Exit criteria**
- User A/B grids show value.
- Defaults are per-model family.

**Sources:** S54; S55

---

### 5. Fast Mode: LCM-LoRA and SDXL Lightning

**Entry criteria**
- Separate Fast Mode UI exists.
- Method recipe registry supports scheduler/CFG/step locking.

**Exit criteria**
- Users cannot accidentally apply normal CFG/negative prompt assumptions.
- 4/8 step Lightning and LCM recipes pass visual review.

**Sources:** S39; S46; S49

---

### 6. torchao FP8 / bitsandbytes selected components

**Entry criteria**
- Capability detector reports cc >=8.9 for FP8.
- Target module is Linear-heavy or upstream example supports it.
- Quality regression protocol active.

**Exit criteria**
- Memory/speed gain outweighs complexity.
- No unacceptable visual changes.

**Sources:** S23; S27; S28; S30

---

## Research later

### 1. Torch-TensorRT Mutable Module

**Research question**
Can Torch-TensorRT support AIWF's dynamic local workflows better than classic TensorRT engines, especially LoRA refit?

**Why later**
Profile/receipt architecture must exist first.

**Sources:** S16; S17; S18

---

### 2. TensorRT Engine Lab

**Research question**
Can AIWF provide a consumer-friendly engine builder for fixed SDXL/SD1.5 profiles without making normal workflows brittle?

**Why later**
Requires engine cache UI, model/LoRA hash invalidation, and support docs.

**Sources:** S12; S13; S14

---

### 3. NVIDIA ModelOpt and Cache Diffusion

**Research question**
Can ModelOpt cache/quantization paths produce useful local gains without heavy workflow burden?

**Why later**
It is powerful but expert-oriented and TensorRT-facing.

**Sources:** S19; S34

---

### 4. GGUF support

**Research question**
Is GGUF useful for AIWF users as an import path for low-bit community models?

**Why later**
Diffusers support exists but pipeline loading and config flow are not as simple as normal model folders.

**Sources:** S33

---

### 5. ORT/Olive

**Research question**
Does Olive provide an actively maintained Windows-local Diffusers acceleration path worth supporting?

**Why later**
The older ORT Stable Diffusion optimization tree is deprecated.

**Sources:** S15

---

### 6. FP4/NVFP4

**Research question**
When consumer Blackwell-class GPUs are common, can FP4/NVFP4 become a useful local image-generation path?

**Why later**
RTX 4070 Ti SUPER/Ada is not the target hardware for NVFP4/FP4 Tensor Core inference.

**Sources:** S24; S25; S26

---

## Avoid for now

| Avoid | Reason | Sources |
|---|---|---|
| Mandatory xFormers | Dependency churn and pinned Torch/CUDA risk. | S09; S10 |
| Silent global `torch.compile` | Compile latency, shape recompiles, LoRA/Windows risks. | S02; S08; S11; S67 |
| Mandatory TensorRT/ONNX/ModelOpt at boot | Heavy dependencies and nontrivial support burden. | S13; S15; S19 |
| Raw custom CUDA graph capture | Static-shape/control-flow constraints; use compile modes first. | S07 |
| Silent quantization | Can change quality and component support is uneven. | S27; S28; S30 |
| FP4/NVFP4 on RTX 4070 Ti SUPER | Blackwell-facing, not Ada. | S24; S25; S26 |
| Transformers >=5 migration in mainline | Project constraint and compatibility churn. | S52; S71; S72 |
| Treating Turbo/Lightning/LCM as schedulers | They require different checkpoints/CFG/scheduler semantics. | S39; S44; S46 |
| Sequential CPU offload default | Diffusers warns it can be extremely slow/impractical. | S01 |
| Copying incompatible project code | Clean-room maintainability constraint. | Project constraint |

---

## Suggested release milestones

### Alpha 1: Stable baseline
- Safe/Balanced profiles.
- Capability detector.
- Receipt writer.
- Scheduler recipes.
- VAE slicing/tiling and model offload.

### Alpha 2: Workflow quality
- Prompt service.
- LoRA manager.
- Inpaint correctness.
- ControlNet core with canny.
- Hires fix workflow.

### Alpha 3: Experimental Lab
- xFormers.
- regional compile.
- FreeU/PAG.
- LCM/Lightning Fast Mode.

### Beta 1: Benchmarked defaults
- Promote only features with receipts.
- Windows target verified.
- Public troubleshooting diagnostics.

### Beta 2: Advanced acceleration preview
- Engine Lab prototype.
- torchao selected components.
- TensorRT/Torch-TensorRT research branch results.

---

## Decision checkpoint

Before any feature moves into default Balanced profile, require:
- benchmark receipts,
- Windows verification,
- quality review,
- fallback path,
- no hidden output semantics change,
- no mandatory optional dependency.
