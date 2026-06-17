# Optimization compatibility matrix

**Prepared:** 2026-06-16

This matrix treats every optimization as **profile-specific**. A green result for SDXL txt2img does not imply the same result for SDXL inpaint + ControlNet + LoRA + hires fix. Use this matrix to drive AIWF Studio's `OptimizationPlanner` and compatibility registry.

| Feature | Category | SD1.5 | SDXL | Img2img | Inpaint | ControlNet | Windows | Status | Main risk | Sources |
|---|---|---|---|---|---|---|---|---|---|---|
| PyTorch native SDPA | Attention | Good | Good | Good | Good | Good | Good | Default / production-ready | Low. SDPA API itself is marked beta in PyTorch, but Diffusers uses it by default on PyTorch >=2.0. | S02; S03; S04 |
| xFormers memory-efficient attention | Attention | Good if wheel matches | Good if wheel matches | Good if ops supported | Good if ops supported | Good if ops supported | Version-gated | Experimental optional backend | Medium. Pip wheels can require newest PyTorch; build/source friction; dependency churn. | S09; S10 |
| FlashAttention backend | Attention | Model/backend dependent | Model/backend dependent | Model/backend dependent | Model/backend dependent | Model/backend dependent | Package-gated | Experimental Lab | Medium/high. Backend package and kernel support vary; FlashAttention-3 is Hopper-focused. | S03 |
| SageAttention backend | Attention | Experimental | Experimental | Experimental | Experimental | Experimental | Package-gated | Experimental Lab | High until benchmarked with visual receipts. | S03 |
| channels_last memory format | Layout | Good | Good | Good | Good | Good | Good with fallback | Default after smoke test | Low/medium. Some custom modules may not support it cleanly. | S02 |
| torch.compile UNet fullgraph | Compile | Fixed-profile good | Fixed-profile good | Riskier due variable inputs | Riskier due masks/crops | Riskier with extra modules | Probe required | Benchmark-gated experimental/beta | High. First compile latency; graph breaks; recompilation on shape/LoRA changes; Windows GPU risk. | S02; S05; S06; S08; S11; S67 |
| regional torch.compile | Compile | Good candidate | Good candidate | Candidate | Candidate | Candidate after tests | Probe required | Preferred compile experiment | Medium/high. Still compile infrastructure; shape and dependency sensitive. | S02; S08 |
| CUDA graphs through torch.compile modes | Compile/runtime | Fixed-profile only | Fixed-profile only | Usually poor fit | Usually poor fit | Usually poor fit | Probe required | Use only via compile experiments | High if done manually; static shape/control-flow restrictions. | S06; S07 |
| VAE slicing | Memory | Useful for batch >1 | Useful for batch >1 | Useful if multi-image | Useful if multi-image | Useful if multi-image | Good | Conditional production | Low. Little benefit for batch=1. | S01 |
| VAE tiling | Memory | High-res useful | High-res useful | High-res useful | High-res useful | High-res useful | Good | User-visible production option | Medium because it changes output subtly. | S01; S65 |
| SDXL fp16-fix VAE | VAE/model asset | Not relevant | Useful option | Useful option | Useful option | Useful option | Good | User-selectable model asset | Medium if silently substituted. | S65; S66 |
| model CPU offload | Memory/offload | Low VRAM fallback | Low VRAM fallback | Low VRAM fallback | Low VRAM fallback | Low VRAM fallback | Good but slower | Low VRAM production profile | Medium. Stateful hooks; slower than full-GPU. | S01 |
| sequential CPU offload | Memory/offload | Emergency | Emergency | Emergency | Emergency | Emergency | Very slow | Emergency only | High usability cost; Diffusers warns it can be extremely slow and impractical. | S01 |
| group offload | Memory/offload | Not first target | Not first target | Experimental | Experimental | Experimental | Probe required | Experimental | Medium/high. Model implementation/device casting caveats; CPU RAM can increase. | S01 |
| TensorRT engine | NVIDIA acceleration | Expert fixed-profile | Expert fixed-profile | Harder | Harder | Harder | Install/runtime gated | Engine Lab / expert experimental | High. Engine build time; resolution/batch specificity; LoRA conversion; heavy dependency. | S12; S13; S14 |
| Torch-TensorRT | NVIDIA acceleration | Research | Research | Research | Research | Research | Probe/install gated | Research later | Medium/high for local consumer app until tested. | S16; S17; S18 |
| ONNX Runtime Stable Diffusion path | NVIDIA/ORT acceleration | Deprecated path | Deprecated path | Deprecated path | Deprecated path | Deprecated path | Possible but not current path | Avoid now / research Olive later | High maintenance burden because referenced path is deprecated. | S15 |
| bitsandbytes 8/4-bit | Quantization | Limited for UNet | Limited/mixed | Limited/mixed | Limited/mixed | Risky | Supported for CUDA but package-gated | Experimental for selected components/future models | Medium/high if marketed as whole-pipeline SD quantization. | S27; S28; S29 |
| torchao FP8/INT8 | Quantization | Experimental | Experimental | Experimental | Experimental | Experimental | Package/version gated | Experimental | Medium/high. PyTorch version and component support constraints. | S23; S27; S30 |
| optimum-quanto | Quantization | Poor default | Poor default | Poor default | Poor default | Poor default | Package-gated | Avoid for now | High for production because repository states maintenance mode. | S31; S32 |
| GGUF loading | Quantization/model format | Experimental import path | Experimental import path | Experimental | Experimental | Experimental | Likely good if deps available | Experimental | Medium. Pipeline loading limitations and config requirements. | S33 |
| LCM / LCM-LoRA | Fast generation | Beta | Beta | Method-specific | Method-specific | Supported in some recipes | Good | Fast Mode beta | Medium if hidden as normal scheduler. | S39; S49 |
| SDXL Turbo | Fast generation | Not relevant | Separate model | Supported with recipe caveat | Not standard | Not standard | Good | Fast Mode beta as separate model family | Medium if confused with SDXL base. | S44; S45 |
| SDXL Lightning | Fast generation | Not primary | Beta | Recipe-specific | Recipe-specific | Recipe-specific | Good | Fast Mode beta; 1-step experimental | Medium/high for wrong checkpoint/step combos. | S46; S47 |
| Hyper-SD / TCD / PCM | Fast generation | Experimental | Experimental | Experimental | Experimental | Experimental | Good if simple deps | Experimental Lab | Medium/high due many recipes and moving model cards. | S40; S48; S50 |
| FreeU | Quality | Beta | Beta | Beta | Not universal | Test first | Good | Quality Lab beta | Medium; parameter-sensitive. | S54 |
| PAG | Quality | Selected pipelines | Selected pipelines | Selected pipelines | Selected pipelines | Selected pipelines | Good if supported | Experimental/beta | Medium; pipeline/layer support limited. | S55 |
| SDXL refiner | Quality/workflow | Not relevant | Good optional | Post-hoc option | Optional | Complex | Good | Quality profile optional | Low/medium; more VRAM/time. | S41; S43 |
| Inpaint-specific checkpoints | Quality/workflow | Strongly recommended | Recommended if available | Not applicable | Core recommendation | Can combine after tests | Good | Production-ready | Low; asset availability/model-family match. | S56; S57 |
| ControlNet preprocessors | Control/workflow | Good | Mixed/beta | Good | Can combine with tests | Core | Dependency-gated | SD1.5 production; SDXL mixed | Medium due optional dependencies and preprocessor quality. | S59; S60; S61; S62; S63 |


## Denylist / conflict rules for the planner

| Rule ID | Reject or warn when | Reason | Fallback |
|---|---|---|---|
| C-001 | `sequential_cpu_offload` after pipeline has been moved to CUDA | Diffusers offload hooks are stateful and must be installed before device movement. | Rebuild pipeline and apply hooks first, or use Safe profile. |
| C-002 | `torch.compile` with resolution not matching compiled cache and no dynamic profile | Shape changes can trigger recompilation. | Eager pipeline or compile a new fixed profile. |
| C-003 | `torch.compile` with arbitrary LoRA hotswap | Diffusers requires correct hotswap preparation and rank/layer compatibility. | Eager pipeline or prepared hotswap profile. |
| C-004 | TensorRT engine with resolution/batch outside engine range | TensorRT engines are resolution/batch-profile specific. | Offer engine rebuild or fallback to PyTorch. |
| C-005 | TensorRT engine with unconverted/unrefitted LoRA | TensorRT extension paths can require LoRA conversion/export. | PyTorch LoRA path or explicit engine refit workflow. |
| C-006 | SDXL Turbo with normal CFG/negative prompt controls | Turbo recipe expects CFG=0/no normal negative prompt. | Switch UI to Turbo recipe controls. |
| C-007 | SDXL Lightning with wrong step count or scheduler spacing | Lightning checkpoints/LoRAs are step-specific and require trailing timesteps. | Select matching recipe or fallback. |
| C-008 | VAE tiling in a strict baseline quality comparison | Tiling can change tone/detail. | Disable tiling or mark comparison as non-baseline. |
| C-009 | FP4/NVFP4 on Ada RTX 40-series | NVFP4/FP4 Tensor Core path is Blackwell-facing, not 4070 Ti SUPER. | Disable flag; show hardware requirement. |
| C-010 | Compel version requiring Transformers >=5 under AIWF's Transformers 4.x lane | Project constraint is `transformers>=4.44,<5`. | Pin compatible Compel release or use adapter. |

## Implementation note

Treat the CSV as the machine-readable seed for an internal registry. The planner can convert each row into:
- `requires_capability`
- `blocks_feature`
- `warns_user`
- `quality_changes_output`
- `fallback_profile`
- `receipt_fields_required`
