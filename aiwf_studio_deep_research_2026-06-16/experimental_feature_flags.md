# Experimental feature flags for AIWF Studio

**Prepared:** 2026-06-16

This file defines proposed feature flags, capability predicates, conflicts, and logging requirements. The flags are designed to keep optional heavy dependencies out of normal boot and prevent unstable optimization combinations.

---

## Flag design principles

1. **Flags are profile inputs, not global state.** A flag modifies an `OptimizationProfile`.
2. **Flags require capability probes.** Import checks and smoke tests happen lazily.
3. **Flags must be logged in receipts.** If a feature changes output, the receipt must prove it was enabled.
4. **Flags need denylist rules.** Some combinations are not allowed.
5. **Flags should fail closed.** If uncertain, fall back to Safe or Balanced.

---

## Recommended flag namespace

```text
attention.*
compile.*
memory.*
quant.*
engine.*
fast.*
quality.*
debug.*
```

---

## 1. Attention flags

### `attention.xformers`

| Field | Value |
|---|---|
| Default | false |
| Readiness | experimental optional backend |
| Capability predicate | `import xformers`, compatible torch/CUDA wheel, `python -m xformers.info` style kernel availability where practical |
| Benefits | attention speed/memory |
| Risks | dependency churn; PyTorch/CUDA wheel mismatch |
| Conflicts | avoid automatic install; do not combine with other explicit attention backend flags |
| Receipt fields | xformers version, torch version, CUDA wheel, selected backend, latency/VRAM delta |
| Sources | S09; S10 |

### `attention.flash`

| Field | Value |
|---|---|
| Default | false |
| Readiness | experimental |
| Capability predicate | backend package installed; Diffusers backend available; GPU capability suitable |
| Benefits | possible attention speed/memory |
| Risks | package/platform support; FlashAttention-3 is Hopper-focused, not Ada default |
| Conflicts | no simultaneous xFormers/Sage |
| Receipt fields | backend name/version, GPU cc, output pHash, latency/VRAM |
| Sources | S03 |

### `attention.sage`

| Field | Value |
|---|---|
| Default | false |
| Readiness | experimental |
| Capability predicate | backend package installed and Diffusers backend selectable |
| Benefits | quantized attention speed potential |
| Risks | quality differences; less mature consumer coverage |
| Conflicts | do not combine with denoiser quantization until tested |
| Receipt fields | backend version, quant attention config, visual review result |
| Sources | S03 |

---

## 2. Compile flags

### `compile.unet`

| Field | Value |
|---|---|
| Default | false |
| Readiness | benchmark-gated beta/experimental |
| Capability predicate | torch.compile smoke test, CUDA, stable shape profile, Windows GPU probe passes |
| Benefits | steady-state denoiser speed |
| Risks | first-run latency, recompilation, graph breaks, Windows/Triton, subtle numerical differences |
| Conflicts | dynamic resolution; uncontrolled LoRA hotswap; unsupported ControlNet graph; TensorRT engine |
| Receipt fields | compile mode, fullgraph/dynamic, compile time, first-run latency, steady-state latency, recompile count if observable |
| Sources | S02; S05; S06; S08; S11; S67 |

### `compile.regional`

| Field | Value |
|---|---|
| Default | false |
| Readiness | preferred compile experiment |
| Capability predicate | repeated block support or Accelerate compile_regions available |
| Benefits | lower cold-start compile time; possible similar steady-state speed |
| Risks | still shape/backend sensitive |
| Conflicts | same as `compile.unet`, but usually lower blast radius |
| Receipt fields | regional method, compiled block classes, compile time, steady-state latency |
| Sources | S02; S08 |

### `compile.vae_decode`

| Field | Value |
|---|---|
| Default | false |
| Readiness | experimental |
| Capability predicate | fixed decode shapes; torch.compile VAE decode smoke test |
| Benefits | possible decode speed |
| Risks | compile latency may outweigh benefit; VAE tiling/slicing interaction |
| Conflicts | VAE tiling profiles unless specifically tested |
| Receipt fields | decode compile time, decode latency, VAE tiling/slicing flags |
| Sources | S02; S08 |

---

## 3. Memory flags

### `memory.vae_slicing`

| Field | Value |
|---|---|
| Default | auto when batch >1 or Low VRAM |
| Readiness | production-ready conditional |
| Capability predicate | pipeline exposes enable_vae_slicing |
| Benefits | lowers multi-image decode memory |
| Risks | little benefit for batch=1 |
| Conflicts | unsupported autoencoder variants |
| Receipt fields | enabled, batch size, decode memory delta |
| Sources | S01 |

### `memory.vae_tiling`

| Field | Value |
|---|---|
| Default | false |
| Readiness | production-ready with quality caveat |
| Capability predicate | pipeline exposes enable_vae_tiling; resolution above threshold |
| Benefits | lowers high-res decode memory |
| Risks | tile-to-tile tone variation / visual differences |
| Conflicts | strict baseline comparisons; untested VAE variants |
| Receipt fields | tile size/overlap if configurable, resolution, output pHash |
| Sources | S01; S65 |

### `memory.model_cpu_offload`

| Field | Value |
|---|---|
| Default | false except Low VRAM |
| Readiness | production fallback |
| Capability predicate | Accelerate available; single CUDA device |
| Benefits | reduces VRAM with moderate speed penalty |
| Risks | stateful hooks; slower generation |
| Conflicts | manual component `.to()` after hooks; device map not reset |
| Receipt fields | offload mode, peak VRAM, total latency |
| Sources | S01 |

### `memory.sequential_cpu_offload`

| Field | Value |
|---|---|
| Default | false |
| Readiness | emergency |
| Capability predicate | Accelerate available; pipeline not already moved to CUDA |
| Benefits | maximum VRAM reduction |
| Risks | extremely slow/impractical |
| Conflicts | pipeline moved to CUDA before hook install |
| Receipt fields | warning acknowledged, peak VRAM, total latency |
| Sources | S01 |

### `memory.group_offload`

| Field | Value |
|---|---|
| Default | false |
| Readiness | experimental |
| Capability predicate | Diffusers hook support; model supports group offload; CPU RAM available |
| Benefits | memory reduction with better transfer behavior than sequential offload |
| Risks | model implementation caveats; CPU RAM increase with streams |
| Conflicts | weight-dependent device casting; untested VAE tiling combinations |
| Receipt fields | offload type, group size, stream/record_stream, CPU RAM peak |
| Sources | S01 |

---

## 4. Quantization flags

### `quant.bitsandbytes`

| Field | Value |
|---|---|
| Default | false |
| Readiness | experimental for selected components |
| Capability predicate | bitsandbytes import, CUDA support, target module mostly Linear |
| Benefits | 8/4-bit memory reduction for Linear-heavy modules |
| Risks | quality loss, outlier sensitivity, limited value for classic conv-heavy UNet |
| Conflicts | VAE/CLIP default quantization; uncontrolled LoRA/compile combination |
| Receipt fields | quantized modules, quant type, compute dtype, memory delta, visual notes |
| Sources | S27; S28; S29 |

### `quant.torchao_fp8`

| Field | Value |
|---|---|
| Default | false |
| Readiness | experimental |
| Capability predicate | torchao import, PyTorch >=2.5, GPU cc >=8.9, target module supported |
| Benefits | FP8 speed/memory on Ada/Hopper for supported modules |
| Risks | quality differences; component support; compile/version sensitivity |
| Conflicts | unsupported conv-heavy components; TensorRT engine path unless coordinated |
| Receipt fields | quant config, target modules, calibration inputs if any, visual review |
| Sources | S23; S27; S30 |

### `quant.optimum_quanto`

| Field | Value |
|---|---|
| Default | false |
| Readiness | avoid for now |
| Capability predicate | only if a specific model requires it |
| Benefits | flexible quantization experiments |
| Risks | maintenance mode upstream |
| Conflicts | active production support preference for bnb/torchao |
| Receipt fields | backend version, quant config, why selected |
| Sources | S31; S32 |

### `quant.gguf`

| Field | Value |
|---|---|
| Default | false |
| Readiness | experimental import/model-format support |
| Capability predicate | GGUF loader support, model config path available |
| Benefits | load low-bit community weights |
| Risks | dynamic dequant/cast cost, pipeline loading limitations, visual differences |
| Conflicts | normal Diffusers folder assumptions |
| Receipt fields | GGUF quant type, config source, CUDA kernel flag, visual notes |
| Sources | S33 |

### `quant.modelopt`

| Field | Value |
|---|---|
| Default | false |
| Readiness | expert experimental |
| Capability predicate | nvidia-modelopt installed; TensorRT deployment target defined |
| Benefits | PTQ/QAT/QAD/Cache Diffusion paths |
| Risks | heavy workflow, calibration, deployment complexity |
| Conflicts | standard local dynamic LoRA/resolution workflow |
| Receipt fields | ModelOpt version, quant config, calibration set hash, TensorRT version |
| Sources | S19; S34 |

---

## 5. Engine flags

### `engine.tensorrt`

| Field | Value |
|---|---|
| Default | false |
| Readiness | expert experimental |
| Capability predicate | TensorRT runtime available, engine profile built for model/resolution/batch |
| Benefits | strong fixed-profile speedups |
| Risks | engine build minutes, resolution/batch specificity, LoRA conversion/export |
| Conflicts | dynamic LoRA; dynamic resolution; incompatible pipelines |
| Receipt fields | engine hash, model hash, min/opt/max shape, build time, TensorRT version |
| Sources | S12; S13; S14 |

### `engine.torch_tensorrt`

| Field | Value |
|---|---|
| Default | false |
| Readiness | research later |
| Capability predicate | torch_tensorrt installed, model graph compiles, fallback available |
| Benefits | TensorRT performance with PyTorch integration; possible LoRA refit |
| Risks | local consumer/Windows coverage not yet proven for AIWF use cases |
| Conflicts | other compile/engine paths |
| Receipt fields | backend mode, graph partitions, refit status, speed/memory |
| Sources | S16; S17; S18 |

### `engine.onnx_runtime`

| Field | Value |
|---|---|
| Default | false |
| Readiness | avoid now |
| Capability predicate | only in research branch |
| Benefits | possible fusions/ORT acceleration |
| Risks | referenced stable diffusion optimization path deprecated |
| Conflicts | conversion/export debug burden |
| Receipt fields | ORT version, model export hash, optimization config |
| Sources | S15 |

---

## 6. Fast Mode flags

Fast Mode flags should be mutually exclusive unless a method explicitly supports composition.

### `fast.lcm`

| Field | Value |
|---|---|
| Default | false |
| Readiness | beta |
| Capability predicate | LCM checkpoint/LoRA loaded; LCMScheduler available |
| Benefits | 2â€“8 step generation |
| Risks | negative prompt/CFG semantics differ |
| Conflicts | normal scheduler/CFG controls |
| Receipt fields | LCM model/LoRA hash, steps, guidance, scheduler |
| Sources | S39; S49 |

### `fast.sdxl_turbo`

| Field | Value |
|---|---|
| Default | false |
| Readiness | beta as separate model family |
| Capability predicate | SDXL Turbo checkpoint |
| Benefits | 1â€“4 step generation |
| Risks | CFG=0/no normal negative prompt; different resolution expectations |
| Conflicts | standard SDXL base/refiner assumptions |
| Receipt fields | steps, CFG forced value, resolution |
| Sources | S44; S45 |

### `fast.sdxl_lightning`

| Field | Value |
|---|---|
| Default | false |
| Readiness | beta for 4/8-step, experimental for 1-step |
| Capability predicate | matching Lightning checkpoint/LoRA for requested step count |
| Benefits | few-step SDXL |
| Risks | wrong step/scheduler/CFG recipe degrades output |
| Conflicts | non-trailing timesteps, normal CFG, mismatched checkpoint |
| Receipt fields | step checkpoint, scheduler, timestep spacing, CFG |
| Sources | S46; S47 |

### `fast.hyper_sd`, `fast.tcd`, `fast.pcm`

| Field | Value |
|---|---|
| Default | false |
| Readiness | experimental |
| Capability predicate | method-specific checkpoint/LoRA and scheduler recipe present |
| Benefits | low-step generation options |
| Risks | many moving recipes; quality tradeoffs |
| Conflicts | wrong scheduler/CFG/eta/timesteps |
| Receipt fields | method, checkpoint/LoRA, scheduler, steps, CFG, eta/timesteps |
| Sources | S40; S48; S50 |

---

## 7. Quality flags

### `quality.freeu`

| Field | Value |
|---|---|
| Default | false |
| Readiness | beta |
| Capability predicate | pipeline supports FreeU |
| Benefits | detail/quality changes without training |
| Risks | visible output change; parameter sensitivity |
| Conflicts | should not be hidden in Balanced |
| Receipt fields | b1, b2, s1, s2 |
| Sources | S54 |

### `quality.pag`

| Field | Value |
|---|---|
| Default | false |
| Readiness | experimental/beta |
| Capability predicate | selected pipeline supports PAG |
| Benefits | quality gains without external modules/training |
| Risks | layer selection complexity; output changes |
| Conflicts | unsupported pipelines; regex/layer misconfiguration |
| Receipt fields | PAG scale, adaptive scale, selected layers |
| Sources | S55 |

### `quality.refiner_sdxl`

| Field | Value |
|---|---|
| Default | false in Balanced, optional in Quality |
| Readiness | production optional |
| Capability predicate | refiner model available and memory sufficient |
| Benefits | visual fidelity/detail |
| Risks | latency/VRAM increase |
| Conflicts | insufficient VRAM; incompatible schedulers/components |
| Receipt fields | high_noise_frac, base/refiner hashes, stage timings |
| Sources | S41; S43 |

---

## 8. Debug flags

### `debug.receipts_always_on`

Records full receipts for every generation. Recommended for alpha builds.

### `debug.save_intermediate_latents`

Only for developer builds. Latents can be large and may create storage/privacy issues.

### `debug.disable_all_optimizations`

For reproducible bug reports. Forces Safe profile.

### `debug.backend_trace`

Records selected backend decisions and capability predicate results.

---

## 9. Suggested capability probe flow

1. Detect GPU, VRAM, compute capability, driver/CUDA runtime.
2. Load core package versions.
3. Do not import optional heavy packages.
4. When a user enables a flag:
   - import package lazily,
   - check version,
   - run minimal smoke test,
   - update capability cache,
   - allow profile if compatible.
5. On failure:
   - mark flag unavailable,
   - show actionable reason,
   - keep app running.

---

## 10. Receipt fields required for any experimental flag

```json
{
  "profile_id": "balanced_sdpa_fp16",
  "experimental_flags": {
    "attention.xformers": false,
    "compile.unet": false,
    "quant.torchao_fp8": false
  },
  "capability_report_id": "sha256...",
  "decision_log": [
    {
      "flag": "compile.unet",
      "decision": "disabled",
      "reason": "windows_compile_probe_failed"
    }
  ]
}
```
