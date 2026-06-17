# Recommended default generation profiles

**Prepared:** 2026-06-16
**Purpose:** provide AIWF Studio with stable user-facing profiles that map to internal `OptimizationProfile` objects.

These profiles are intentionally conservative. They are designed for local consumer NVIDIA GPUs, with RTX 4070 Ti SUPER-class 16 GB VRAM as the default target. Hidden output-changing optimizations are avoided.

---

## Profile naming policy

| UI name | Internal profile ID | User promise |
|---|---|---|
| Safe | `safe_eager_cuda` | Maximum compatibility and debuggability. |
| Balanced | `balanced_sdpa_fp16` | Recommended default for most RTX users. |
| Quality | `quality_visible_modifiers` | Better output through visible quality controls, not invisible magic. |
| Low VRAM | `low_vram_model_offload` | Fit larger workflows with explicit speed tradeoff. |
| Fast Mode | `fast_method_recipe` | Use distilled/low-step models with method-specific controls. |
| Experimental Lab | `experimental_feature_flags` | Power-user features that require receipts. |

---

## 1. Safe profile

**Readiness:** production-ready
**Default audience:** first launch, unknown GPU, bug reproduction, diagnostics
**Source basis:** Diffusers SDPA/dtype/memory docs and scheduler docs [S01][S02][S35][S36][S37]

### Settings

| Setting | Value |
|---|---|
| dtype | fp16 on CUDA if supported; fp32 for CPU/debug fallback |
| attention | PyTorch SDPA / native PyTorch |
| memory format | default contiguous |
| compile | off |
| xFormers / Flash / Sage | off |
| TensorRT / ORT / Torch-TensorRT | off |
| quantization | off |
| VAE slicing | off unless batch >1 and memory pressure detected |
| VAE tiling | off |
| CPU offload | off unless needed to avoid OOM |
| scheduler SD1.5 | Euler 25â€“30 or DPM++ 2M/DPMSolverMultistep 20â€“25 |
| scheduler SDXL | Euler 30 or DPM++ with Karras/mitigation, 30 |
| quality modifiers | off |
| fast/distilled methods | off |

### When to force Safe

- An optimization backend fails.
- A user reports a bug.
- The pipeline has untested custom components.
- Windows capability probe for a feature fails.
- The app detects a dependency mismatch.

### Receipt requirements

Always record:
- model path/hash,
- VAE path/hash,
- scheduler class/config,
- dtype,
- attention backend,
- GPU name/VRAM,
- package versions,
- seed,
- prompt hash,
- output image hash.

---

## 2. Balanced profile

**Readiness:** recommended default
**Default audience:** RTX 4070 Ti SUPER and similar consumer NVIDIA GPUs
**Source basis:** Diffusers SDPA, channels-last, dtype, and scheduler guidance [S02][S35][S36][S37][S42]

### Settings

| Setting | Value |
|---|---|
| dtype | fp16 |
| attention | PyTorch SDPA |
| memory format | channels-last for UNet and VAE after smoke test |
| compile | off |
| VAE slicing | on for batch >1 |
| VAE tiling | off by default; suggest at high-res/OOM |
| CPU offload | off |
| scheduler SD1.5 | DPMSolverMultistep / DPM++ 2M-style, 20â€“25 steps; Euler fallback |
| scheduler SDXL | DPM++ with Karras/compatible mitigation or Euler, 30 steps |
| SDXL size | 1024Ã—1024 or equivalent area; avoid below 512 |
| ControlNet | one ControlNet default; explicit scale/start/end controls |
| LoRA | standard PEFT Diffusers path; no compile by default |
| negative prompt | enabled for standard CFG modes |

### Why this is the default

Balanced uses optimizations that are:
- available without optional heavy dependencies,
- easy to fall back from,
- compatible with most SD1.5/SDXL community workflows,
- relatively simple to diagnose.

It intentionally excludes hidden `torch.compile`, xFormers, TensorRT, and quantization because those features are environment-sensitive or can complicate LoRA/resolution workflows [S08][S09][S10][S12][S13][S67].

---

## 3. Quality profile

**Readiness:** production-ready for standard settings; beta for FreeU/PAG
**Default audience:** users prioritizing visual quality over speed
**Source basis:** SDXL refiner docs/model card, FreeU docs, PAG docs, VAE caveats [S41][S43][S54][S55][S65][S66]

### Settings

| Setting | Value |
|---|---|
| Base | Balanced profile |
| SD1.5 steps | 28â€“35 |
| SDXL steps | 35â€“45 |
| SDXL refiner | optional, visible; high-noise split around 0.8 as preset |
| hires fix | enabled as workflow, not backend trick |
| FreeU | off by default, visible presets |
| PAG | off by default, visible supported-pipeline presets |
| VAE tiling | visible high-res memory option |
| VAE choice | user-selectable; no silent replacement |
| ControlNet | scales/start/end exposed; multi-control warnings |
| inpaint | prefer inpaint checkpoints and mask blur/padding crop controls |

### Quality modifiers that must be visible

| Modifier | Why visible? | Source |
|---|---|---|
| FreeU | Changes skip/backbone balance and output style/detail. | S54 |
| PAG | Changes attention guidance behavior. | S55 |
| SDXL refiner | Adds second denoising model/stage and latency. | S41; S43 |
| hires fix | Adds second denoise/upscale pass. | S41; S42 |
| VAE tiling | Can change tonal consistency/detail. | S01; S65 |
| Alternate VAE | VAE directly changes decode appearance. | S65; S66 |
| clip skip | Changes text encoder layer used for prompt semantics. | S57 |

---

## 4. Low VRAM profile

**Readiness:** production fallback
**Default audience:** users below target VRAM, large SDXL/ControlNet/hires workflows, OOM recovery
**Source basis:** Diffusers memory/offload docs [S01]

### Settings

| Setting | Value |
|---|---|
| Base | Safe or Balanced depending on capability |
| CPU offload | model CPU offload first |
| Sequential offload | emergency only, with â€œvery slowâ€ warning |
| Group offload | experimental only |
| VAE slicing | on for batch >1 |
| VAE tiling | suggest for high-res decode/OOM |
| batch size | force or suggest batch=1 |
| preview | generate lower-res preview before full job |
| compile | off by default |
| TensorRT | off |
| quantization | off unless experimental profile explicitly selected |

### UX language

Use plain warnings:
- â€œLow VRAM mode may be slower because model parts move between CPU and GPU.â€
- â€œEmergency offload can be much slower but may prevent out-of-memory errors.â€
- â€œVAE tiling can slightly alter tones or seams at high resolution.â€

### Fallback ladder

1. Clear CUDA cache and retry same profile once if safe.
2. Enable VAE slicing if batch >1.
3. Enable VAE tiling if high-res decode fails.
4. Switch to model CPU offload.
5. Reduce batch/resolution if user allows.
6. Sequential offload only after explicit consent.

---

## 5. Fast Mode profiles

**Readiness:** beta/experimental by method
**Default audience:** preview, rapid iteration, casual generation, low-step workflows
**Source basis:** LCM, SDXL Turbo, Lightning, Hyper-SD, TCD, PCM docs/model cards [S39][S40][S44][S46][S48][S49][S50]

Fast Mode is a **mode family**, not one scheduler. Each method owns:
- compatible model/checkpoint,
- scheduler,
- timestep spacing,
- step count,
- CFG behavior,
- negative prompt behavior,
- LoRA requirement,
- resolution recommendation.

### LCM / LCM-LoRA profile

| Setting | Value |
|---|---|
| scheduler | LCMScheduler |
| steps | 4â€“8 |
| guidance | method-card range; often low |
| negative prompt | annotate as non-standard |
| status | beta |
| sources | S39; S49 |

### SDXL Turbo profile

| Setting | Value |
|---|---|
| model | SDXL Turbo checkpoint |
| steps | 1â€“4 |
| guidance | 0.0 |
| negative prompt | disabled/annotated |
| recommended resolution | model-card recipe; often 512-oriented |
| status | beta as separate model family |
| sources | S44; S45 |

### SDXL Lightning profile

| Setting | Value |
|---|---|
| model/LoRA | step-specific 2/4/8 first; 1-step experimental |
| scheduler | Euler |
| timestep spacing | trailing |
| guidance | 0 |
| status | beta for 4/8; experimental for 1 |
| sources | S46; S47 |

### Hyper-SD / TCD / PCM profiles

| Setting | Value |
|---|---|
| status | Experimental Lab |
| reason | multiple recipes, schedulers, eta/CFG/timestep requirements |
| sources | S40; S48; S50 |

---

## 6. Experimental Lab profile

Experimental Lab should never auto-enable features. It should offer explicit tests and receipts.

| Feature | Enable only if | Receipt fields |
|---|---|---|
| xFormers | package import + kernel info + benchmark improvement | xformers version, torch version, backend, latency/VRAM |
| Flash/Sage attention | package import + backend support + visual receipt | backend, GPU cc, output pHash/notes |
| regional compile | compile probe succeeds and fixed profile selected | compile time, recompile count, graph breaks if available |
| full compile | repeated same-size workload and Windows probe passes | compile mode, dynamic/fullgraph, first-run vs steady-state |
| torchao FP8 | cc>=8.9 + torchao installed + component support | quant config, calibration/quality notes |
| bitsandbytes | target module is Linear-heavy | quantized modules, dtype, memory delta |
| TensorRT | user builds engine profile | engine hash, min/opt/max shapes, build time |
| ModelOpt | expert mode only | quant config, calibration set, TensorRT version |

---

## 7. Suggested initial app defaults

### SD1.5 txt2img

| Field | Default |
|---|---|
| Resolution | 512Ã—512 |
| Steps | 25 |
| Scheduler | DPM++/DPMSolverMultistep or Euler fallback |
| CFG | 7 |
| Dtype | fp16 |
| Attention | SDPA |
| Hires fix | off; suggest for >768 |
| Clip skip | model default; advanced setting |

### SDXL txt2img

| Field | Default |
|---|---|
| Resolution | 1024Ã—1024 |
| Steps | 30 |
| Scheduler | DPM++ with Karras/mitigation or Euler fallback |
| CFG | 5â€“7 |
| Dtype | fp16 |
| Attention | SDPA |
| Refiner | off in Balanced; optional in Quality |
| VAE | model default; optional fp16-fix VAE user choice |

### Inpaint

| Field | Default |
|---|---|
| Model | inpaint-specific checkpoint when available |
| Mask semantics | white = inpaint, black = preserve |
| Mask blur | visible default, e.g. 4â€“8 px |
| Padding/crop | enabled as visible control |
| Strength | 0.6â€“0.85 depending workflow |
| Scheduler | same family as base model |
| Source tracking | save image hash, mask hash, mask blur |

### ControlNet

| Field | Default |
|---|---|
| Preprocessor | canny built-in first |
| Optional preprocessors | depth/pose/lineart/segmentation lazy installed |
| Scale | 0.8â€“1.0 initial |
| Start/end | 0.0 / 1.0 visible |
| Multi-control | warn after 1 active control |
| Model match | SD1.5 ControlNet with SD1.5; SDXL ControlNet with SDXL |

---

## 8. Implementation checklist

- [ ] Encode these profiles as data, not scattered conditionals.
- [ ] Add profile versioning.
- [ ] Record profile ID in image metadata/receipt.
- [ ] Add fallback to Safe.
- [ ] Hide unsupported experimental flags based on capability probe.
- [ ] Do not silently substitute Fast Mode or quality modifiers.
