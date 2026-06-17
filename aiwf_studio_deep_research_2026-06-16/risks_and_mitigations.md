# Risks and mitigations

**Prepared:** 2026-06-16

This register is organized around failure modes AIWF Studio is likely to encounter when optimizing local Diffusers workflows.

---

## Risk severity scale

| Severity | Meaning |
|---|---|
| Low | Minor inconvenience; clear fallback. |
| Medium | User-visible issue; support burden; recoverable. |
| High | Crashes, OOM, wrong output semantics, dependency breakage, or data/security concern. |
| Critical | Prevents app launch or creates unsafe model-loading behavior. |

---

## Risk register

| ID | Risk | Severity | Trigger | Mitigation | Sources |
|---|---|---:|---|---|---|
| R-001 | Optional optimizer dependency breaks boot | High | xFormers/TensorRT/torchao/ORT import failure at startup | Lazy imports only after user selects feature; optional extras; capability probes | S09; S10; S16; S30 |
| R-002 | xFormers upgrades or mismatches pinned Torch/CUDA | High | `pip install xformers` pulls different Torch or wheel ABI mismatch | Never auto-install; document matching wheels; detect version and disable if mismatch | S09; S10 |
| R-003 | `torch.compile` first-run latency feels like hang | Medium/high | Compile profile selected without warning | Record compile time; UI progress; keep off by default | S02; S08 |
| R-004 | `torch.compile` recompiles on resolution change | High | User changes image size, batch, ControlNet shape, inpaint crop | Cache key includes shapes; reject dynamic changes unless dynamic profile tested | S02; S08 |
| R-005 | Windows GPU compile unsupported or unstable | High | Torch Inductor/Triton path fails on Windows | Probe before exposing; default to eager SDPA on Windows | S11 |
| R-006 | Compiled LoRA workflow recompiles or fails | High | LoRA hotswap without proper preparation or different rank/layers | LoRA manager enforces `enable_lora_hotswap` ordering; compile only after first LoRA; fallback eager | S67; S68; S08 |
| R-007 | Text-encoder LoRA hotswap assumption is wrong | Medium/high | User swaps LoRA targeting text encoder in compiled profile | Block compiled hotswap for text-encoder LoRA; rebuild eager/compiled pipeline | S67 |
| R-008 | TensorRT engine mismatch | High | User changes resolution, batch, model, LoRA | Engine metadata and hash; offer rebuild; fallback PyTorch | S13 |
| R-009 | TensorRT build time surprises user | Medium | Engine generation takes minutes | Separate Engine Lab; display expected build/cache behavior | S13 |
| R-010 | TensorRT LoRA incompatibility | High | Unconverted LoRA used with engine | Require LoRA conversion/refit workflow or fallback | S13; S17 |
| R-011 | ONNX Runtime path becomes maintenance burden | Medium/high | Deprecated scripts drift from Diffusers | Avoid primary implementation; track Olive only | S15 |
| R-012 | Silent quantization degrades image quality | High | bnb/torchao/ModelOpt quant applied without user knowledge | Quantization only experimental; visual receipts; never hidden | S27; S28; S30; S34 |
| R-013 | Quantization targets wrong modules | Medium/high | VAE/CLIP or conv-heavy UNet quantized with Linear-only backend | Component capability registry; default no VAE quantization | S28; S30; S31 |
| R-014 | optimum-quanto becomes stale dependency | Medium | Added as production backend despite maintenance mode | Avoid unless required by specific model | S32 |
| R-015 | GGUF loading confused with normal Diffusers pipeline loading | Medium | User loads GGUF as if full pipeline folder | Separate import flow; require config path; warning on dynamic dequant | S33 |
| R-016 | FP4/NVFP4 falsely advertised on Ada | High | UI enables Blackwell-only formats on RTX 40-series | GPU cc/hardware gate; disable for 4070 Ti SUPER | S24; S25; S26 |
| R-017 | VAE tiling changes output silently | Medium | High-res/Low VRAM auto-enables tiling | Make visible; receipt flag; compare baseline separately | S01; S65 |
| R-018 | SDXL VAE fp16 NaNs or upcast memory surprise | Medium | Original SDXL VAE fp16 issues | Expose VAE choice; allow fp16-fix VAE as asset; no silent replacement | S65; S66 |
| R-019 | Sequential offload makes app appear frozen | Medium/high | Emergency offload enabled by default | Use only with explicit warning; progress messages | S01 |
| R-020 | Offload hooks conflict with manual `.to()` calls | High | Pipeline/component moved after hooks installed | Pipeline lifecycle service owns device moves; compatibility checks | S01 |
| R-021 | Fast Mode breaks negative prompt expectations | Medium/high | LCM/Turbo/Lightning with normal CFG UI | Separate Fast Mode UI; method-specific controls | S39; S44; S46 |
| R-022 | Lightning wrong step/checkpoint pairing | Medium | User selects 8 steps with 4-step LoRA or wrong scheduler | Recipe registry enforces step/scheduler/CFG | S46 |
| R-023 | LCM/TCD/PCM scheduler mismatch | Medium | Wrong scheduler/timesteps/eta | Method-specific recipes; block unsupported combinations | S39; S40; S48; S50 |
| R-024 | SDXL DPM++ low-step artifacts | Medium | SDXL DPM++ under 50 steps without mitigation | Use Karras/lu/euler_at_final mitigations or Euler fallback | S42 |
| R-025 | SDXL generated below recommended size | Low/medium | User selects <512 or very small SDXL size | Warn; recommend 1024 target | S42 |
| R-026 | Inpaint poor quality from base checkpoint | Medium | Base txt2img checkpoint used for inpaint | Prefer inpaint checkpoint; label fallback lower quality | S57 |
| R-027 | SDXL mask blur regression | Medium | Diffusers version changes mask processing | Add regression tests around blurred masks | S56; S58 |
| R-028 | ControlNet preprocessor dependency bloat | Medium/high | Heavy pose/depth stacks installed at boot | Lazy preprocessor packages; canny core; optional DWPose route | S61 |
| R-029 | SDXL ControlNet model quality inconsistent | Medium | Experimental SDXL ControlNet checkpoints | Model-family warnings; per-model receipts | S63 |
| R-030 | Prompt weighting breaks with Transformers/Compel version | High | Compel main requires Transformers >=5 or behavior changes | Pin compatible Compel; prompt embedding tests; migration branch | S52; S53; S71; S72 |
| R-031 | Unsafe checkpoint loading | Critical | Pickle `.ckpt` loaded directly from untrusted source | Prefer safetensors; restrict pickle loading; explicit unsafe mode if ever supported | S69; S70 |
| R-032 | Benchmark numbers compare apples to asteroids | Medium | Different model/scheduler/steps/resolution | Receipt schema enforces variables; benchmark compare validates | S01; S02; S08 |
| R-033 | Quality regression missed by speed benchmark | Medium/high | Only latency/VRAM measured | Save images, pHash, artifact labels, human grid review | S27 |
| R-034 | Profile state leaks through global mutable pipeline | High | Shared globals mutate optimization hooks | AppContext composition root owns services; pipeline instances are profile-bound | Architecture inference from AIWF constraints |
| R-035 | User cannot reproduce result | Medium | Missing version/profile metadata | Store receipts and metadata in image info sidecar | S08 |

---

## Mitigation patterns

### 1. Feature detection before feature exposure

Do not show a toggle as available until:
- package import succeeds,
- version is compatible,
- minimal smoke test passes,
- planner has no known conflict.

### 2. Fallback ladder

Every experimental backend must declare:
- fallback profile,
- whether output semantics change,
- whether automatic fallback is allowed,
- user-facing error text.

### 3. Receipts and image metadata

Every generation in developer/beta mode should save enough metadata to reproduce:
- model and VAE hashes,
- profile ID,
- scheduler config,
- prompt hashes,
- seed,
- active LoRAs,
- optimizer flags,
- package versions.

### 4. CI and local test matrix

Minimum local matrix:
- Windows + RTX 4070 Ti SUPER target.
- Windows + lower VRAM NVIDIA if possible.
- Linux + NVIDIA if contributors have it.
- CPU fallback smoke test for app boot only.
- No optional optimizer installed.
- xFormers installed.
- ControlNet optional preprocessors absent/present.
- Transformers 4.x lane.

### 5. Release gates

A feature cannot be default unless:
- the benchmark protocol has receipts,
- Windows path is verified,
- fallback path works,
- support documentation exists,
- no output-changing behavior is hidden.

---

## Highest-priority mitigations to implement first

1. Optional dependency/lazy import system.
2. Profile-bound pipeline lifecycle service.
3. Receipt writer.
4. Capability detector.
5. Scheduler/model-family recipe registry.
6. LoRA manager with compiled-profile awareness.
7. Inpaint mask regression test.
8. Fast Mode UI separation.
