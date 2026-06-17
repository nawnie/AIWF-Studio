# AIWF Studio Deep Technical Research Report

**Topic:** Practical speed, memory, stability, and quality improvements for a local-first Diffusers-based Stable Diffusion application
**Prepared:** 2026-06-16
**Target application:** AIWF Studio
**Target platform assumption:** consumer NVIDIA GPUs, especially RTX 4070 Ti SUPER-class Ada Lovelace cards with 16 GB VRAM
**Core dependency constraint:** keep `transformers>=4.44,<5` unless a separately tested migration plan proves otherwise
**Primary source list:** see [`sources.md`](sources.md)

---

## 1. Executive summary

AIWF Studio should treat performance optimization as a **profiled capability system**, not a pile of global toggles. The safest path for a consumer local-first app is to establish a strong, boring, debuggable baseline: PyTorch native SDPA attention, fp16 CUDA execution for SD1.5/SDXL, optional channels-last conversion, scheduler defaults tuned by model family, VAE memory tools exposed only when they matter, and CPU/model offload as explicit low-VRAM fallback rather than a default. Diffusersâ€™ current guidance makes PyTorch SDPA the default for PyTorch 2.x, documents channels-last and `torch.compile` as important acceleration tools, and clearly warns that offload strategies are stateful and can be slow or model-dependent [S01][S02][S03][S04].

The next production-quality implementation should therefore be **an optimization planner and receipt system**. Every pipeline instance should carry a declared `OptimizationProfile`; every non-baseline optimization should be selected by capability detection and recorded in a benchmark receipt. This keeps AIWFâ€™s clean layered architecture intact: domain models define policy, services plan/validate profiles, infrastructure backends apply backend-specific operations, and the Gradio UI exposes safe presets instead of raw foot-guns. This is more important than chasing one dramatic acceleration library, because repeated instability usually comes from hidden interactions: LoRA plus compile, offload hooks plus manual `.to("cuda")`, TensorRT engines plus dynamic resolutions, or fast distillation schedulers plus normal CFG/negative prompt expectations [S01][S08][S13][S39][S44][S46][S67].

**Recommended default stance:**

| Area | Default recommendation | Readiness | Rationale |
|---|---|---:|---|
| Attention | PyTorch native SDPA. Let PyTorch select optimized kernels. | Production-ready | SDPA is enabled by default in Diffusers with PyTorch >=2.0 and supports optimized CUDA backends without extra dependency boot cost [S02][S04]. |
| xFormers | Optional backend, not mandatory. | Experimental / compatibility-gated | xFormers can improve speed/memory, but wheels are tied to current PyTorch/CUDA combinations and can push dependency churn [S09][S10]. |
| FlashAttention / SageAttention | Optional lab backend only. | Experimental | Diffusers exposes backend selection, but backend availability and model support vary; FlashAttention-3 is Hopper-focused, not RTX 4070 Ti SUPER [S03]. |
| `torch.compile` | Benchmark-gated profile for repeated same-size workflows. | Experimental-to-beta | Can improve steady-state speed, but first-run compile latency, recompilation on shape changes, Windows GPU/Triton risk, and LoRA interactions make it unsuitable as a hidden default [S02][S05][S06][S08][S11][S67]. |
| CUDA graphs | Use only through `torch.compile` modes initially. | Experimental | Manual CUDA graph capture has static-shape and forbidden-operation constraints; NVIDIA recommends quantifying benefit and validating correctness [S07]. |
| VAE slicing/tiling | Conditional low-VRAM/high-res setting, user-visible. | Production-ready with caveat | Slicing helps multi-image decode; tiling saves memory at high resolution but may cause tone variation [S01][S65]. |
| CPU offload | Model CPU offload as Low VRAM profile; sequential CPU offload only emergency. | Production-ready fallback | Sequential offload drastically reduces VRAM but can be extremely slow; model offload is faster but saves less memory [S01]. |
| TensorRT / Torch-TensorRT | Separate advanced acceleration lane. | Experimental / expert | TensorRT can be fast, but engines are profile-specific, take minutes to build, and complicate LoRA/resolution workflows; Torch-TensorRT is promising but still an optional heavy path [S12][S13][S16][S17][S18]. |
| Quantization | Do not broadly quantize SD1.5/SDXL by default. Experiment with torchao/bnb for large Linear-heavy components and future DiT models. | Experimental | Current Diffusers quantization backends mostly target Linear layers or TensorRT deployment; VAE/CLIP are not obvious wins, and quality/compatibility must be measured [S27][S28][S30][S31][S32][S33][S34]. |
| Fast generation | Separate â€œFast Modeâ€ model family; never hidden optimizer. | Beta for LCM/Turbo/Lightning, experimental for Hyper-SD/TCD/PCM | These methods change scheduler, CFG, negative prompt behavior, and sometimes required checkpoints/LoRAs [S39][S44][S46][S48][S49][S50]. |
| Quality features | Expose as quality controls: Compel weights, clip skip, FreeU, PAG, SDXL refiner, hires fix, inpaint-specific checkpoint rules, ControlNet preprocessors. | Mixed | Some are stable workflow controls; FreeU/PAG are inference-time quality options but should be user-visible because they change output [S41][S42][S51][S52][S54][S55][S56][S57][S59]. |

**Practical conclusion:** the best near-term AIWF improvement is not â€œadd TensorRTâ€ or â€œturn on compile.â€ It is a **safe optimization substrate**: profile objects, capability detection, compatibility rules, benchmark receipts, fallback behavior, and a small set of well-tested default profiles. Once that exists, AIWF can test xFormers, compile, TensorRT, torchao, LCM, Lightning, and future FP8/FP4 paths without turning the app into a haunted toaster with opinions.

---

## 2. Scope, assumptions, and evidence standards

This report is scoped to **local consumer image generation** on Windows/Linux NVIDIA systems, with RTX 4070 Ti SUPER as the concrete design point. The RTX 4070 Ti SUPER is an Ada Lovelace RTX 40-series GPU with 16 GB GDDR6X; NVIDIA documents the RTX 4070 Ti SUPER family specs and Ada fourth-generation Tensor Cores, and NVIDIA forum guidance identifies RTX 40-series cards as compute capability 8.9 [S20][S21][S22][S23]. This matters because Ada supports FP8 Tensor Core capability, but **does not support Blackwell NVFP4/FP4 Tensor Core inference**; NVFP4/FP4 should be treated as Blackwell/future-facing, not as a 4070 Ti SUPER optimization [S24][S25][S26].

Evidence categories used here:

1. **Production-ready:** documented in current official Diffusers/PyTorch APIs, low dependency burden, stable failure behavior, and straightforward fallback.
2. **Beta / benchmark-gated:** credible and useful, but sensitive to model family, resolution, OS, driver, or dependency versions.
3. **Experimental:** promising, but heavy dependencies, nontrivial build/compile steps, changed quality semantics, or incomplete Windows/local-consumer coverage.
4. **Avoid for now:** misleading for the target GPU, too brittle for consumer support, or incompatible with the projectâ€™s version constraints.

---

## 3. Findings by research area

### 3.1 Diffusers performance optimization

#### 3.1.1 Baseline dtype and model loading

For SD1.5 and SDXL on NVIDIA CUDA, AIWFâ€™s baseline should be **fp16 model weights and fp16 inference** unless a specific model component requires fp32/bf16. Diffusers notes that model dtype strongly affects memory and speed, and its examples load SDXL with lower-precision weights for CUDA inference [S02]. fp16 remains the practical compatibility default for the SD1.5/SDXL ecosystem because community UNets, VAEs, LoRAs, and ControlNets are usually tested with fp16. BF16 is attractive on modern GPUs because it is more robust to numerical error, but many community Stable Diffusion assets are not tested as thoroughly with bf16 as with fp16 [S02][S28].

**Recommendation:** implement a `DTypePolicy` with `fp16_default`, `bf16_experimental`, and `fp32_safe_debug`. Use fp16 by default for SD1.5/SDXL; expose bf16 only in advanced settings or for model families where Diffusers docs or receipts show it is beneficial. **Status:** production-ready baseline; bf16 experimental for SD1.5/SDXL consumer workflows. **Sources:** [S02][S28].

#### 3.1.2 Attention processors and backends

PyTorch SDPA should be AIWFâ€™s default attention backend. Diffusers states SDPA is enabled by default with PyTorch >=2.0, and PyTorch SDPA can dispatch among FlashAttention-2, memory-efficient attention, and a math kernel on CUDA [S02][S04]. Diffusersâ€™ newer attention backend dispatcher also supports native PyTorch SDPA, xFormers, FlashAttention, SageAttention, cuDNN attention, and other backends, but labels dispatcher behavior as an optimization interface that should be selected deliberately [S03].

xFormers remains useful but should not be mandatory at boot. Hugging Faceâ€™s xFormers page reports speed and memory benefits, but also warns that the pip package can require the latest PyTorch; the xFormers repository currently offers Linux/Windows wheels for specific current CUDA/PyTorch combinations and notes that the project contains research-first, bleeding-edge components [S09][S10]. That combination is acceptable for a power-user flag, but risky as a universal consumer default.

FlashAttention and SageAttention should be treated as optional backend experiments. Diffusers lists them as available attention backend options, but FlashAttention-3 is Hopper-specific, and SageAttention involves quantized attention internals that should be benchmarked for visual impact and package compatibility [S03].

**Recommendation:** make `attention_backend="auto_sdpa"` the default. Add optional backend flags for `xformers`, `flash_attention`, and `sage_attention` only after import checks, GPU capability checks, and a short local benchmark. **Status:** SDPA production-ready; alternatives experimental. **Sources:** [S02][S03][S04][S09][S10].

#### 3.1.3 `torch.compile`

`torch.compile` can help Diffusers, but it is not a drop-in default for AIWF. Diffusers recommends compiling compute-heavy parts such as UNet, transformer, or VAE, and documents channels-last plus compile examples [S02]. PyTorch documents compile modes including `default`, `reduce-overhead`, and `max-autotune`; reduce-overhead can use CUDA graphs and may increase memory, while max-autotune does more tuning and graphing [S05][S06]. Diffusers also notes first compile is slow and shape changes can trigger recompilation [S02]. The PyTorch Diffusers compile guide emphasizes the same operational issues: compile latency, graph breaks, recompilation, dynamic-shape tradeoffs, and the benefit of regional compilation for repeated blocks [S08].

Windows is an extra risk. A PyTorch forum explanation states that GPU `torch.compile` with the default Inductor path relies on Triton, and Tritonâ€™s Windows support has historically been a blocker for Windows GPU compile workflows [S11]. This does not mean compile is impossible forever on Windows, but it does mean AIWF should detect capability rather than assume support.

LoRA also complicates compiled pipelines. Diffusers documents LoRA hotswapping and notes compiled models should use `enable_lora_hotswap()` before loading the first LoRA and compile after loading the first LoRA; text-encoder LoRA hotswapping is unsupported [S67][S68]. PyTorchâ€™s Diffusers compile guide also highlights LoRA hotswap caveats around rank and layer compatibility [S08].

**Recommendation:** add `torch_compile_unet` as a benchmark-gated experimental profile, not a default toggle. Support only pinned resolution/batch/scheduler profiles at first. Do not compile if active LoRA state is dynamic unless AIWF has already called the correct LoRA hotswap preparation and has a receipt proving no recompilation churn. **Status:** experimental-to-beta for repeated same-size workflows; risky on Windows until detected. **Sources:** [S02][S05][S06][S08][S11][S67][S68].

#### 3.1.4 CUDA graphs

CUDA graphs reduce CPU launch overhead, but they are only useful when CPU launch overhead is meaningful and shapes/control flow are sufficiently static. NVIDIA recommends quantifying benefit first, checking utilization, starting with small captures, and validating correctness against eager execution [S07]. For AIWF, the practical first step is **not manual graph capture**. Use `torch.compile(mode="reduce-overhead")` or `max-autotune` experiments where PyTorch manages graph capture, then inspect receipts [S06][S07].

**Recommendation:** no custom CUDA graph subsystem yet. Use CUDA graph behavior only through `torch.compile` modes inside experimental profiles. **Status:** experimental. **Sources:** [S06][S07].

#### 3.1.5 VAE slicing and tiling

VAE slicing and tiling are credible low-VRAM tools. Diffusers states VAE slicing helps multi-image batches by decoding one image at a time; VAE tiling divides high-resolution images into overlapping tiles and may introduce tile-to-tile tone variation [S01]. Older AutoencoderKL docs also warn tiled encode/decode can change output because tiles are blended [S65]. SDXL VAE behavior deserves special handling: Diffusersâ€™ AutoencoderKL historically uses `force_upcast=True` for SDXL-class VAEs, and the community `sdxl-vae-fp16-fix` model card says it was built to avoid fp16 NaNs while acknowledging slight discrepancies [S65][S66].

**Recommendation:** enable VAE slicing automatically only for batch size >1 or low-VRAM profile. Expose VAE tiling as a visible high-resolution/low-VRAM setting with a quality caveat. Allow an SDXL fp16-fix VAE selector, but treat it as a model asset choice, not a silent optimization. **Status:** slicing production-ready; tiling production-ready with visible quality caveat; alternate fp16 VAE user-selectable. **Sources:** [S01][S65][S66].

#### 3.1.6 CPU, sequential, model, and group offload

Diffusers documents several offload strategies. Sequential CPU offload can drastically reduce VRAM, but Diffusers warns it can be extremely slow and often impractical; it is also stateful and must be called before moving the pipeline to CUDA [S01]. Model CPU offload is faster because it moves whole components rather than individual submodules, but saves less memory [S01]. Group offload is promising but has model-specific caveats and can increase CPU memory when streams are used [S01].

**Recommendation:** AIWF should implement offload as explicit memory profiles:
- `normal_cuda`: no offload.
- `model_cpu_offload`: Low VRAM profile.
- `sequential_cpu_offload`: Emergency mode with warning.
- `group_offload`: experimental for large future models/video, not default for SD1.5/SDXL.

**Status:** model offload production fallback; sequential emergency; group experimental. **Sources:** [S01].

#### 3.1.7 Channels-last memory format

Diffusers recommends channels-last as a memory/layout optimization and demonstrates applying it to UNet and VAE before compile [S02]. Channels-last is comparatively low risk, but it can still interact with custom modules and should be applied per component with fallback.

**Recommendation:** enable channels-last in the Balanced profile only after one smoke test per pipeline instance. Log component-level success/failure; fallback to contiguous memory format if any exception occurs. **Status:** production-ready with fallback. **Sources:** [S02].

---

### 3.2 NVIDIA-specific acceleration

#### 3.2.1 TensorRT

TensorRT can accelerate diffusion pipelines, but it should be a separate expert lane for AIWF. NVIDIA reports up to near-2x SDXL acceleration with INT8/FP8 post-training quantization in a TensorRT workflow involving calibration, ONNX export, and engine build [S12]. NVIDIAâ€™s TensorRT extension for Stable Diffusion WebUI supports SD1.5, SD2.1, SDXL, SDXL Turbo, and LCM variants, but also documents that engine generation can take minutes, engines are resolution/batch-specific, dynamic ranges cost performance/VRAM, and LoRA/LyCORIS require conversion/export into TensorRT format [S13]. That is the opposite of frictionless local consumer UX.

**Recommendation:** do not make TensorRT part of default AIWF startup. Add an optional â€œEngine Labâ€ later, where users build named engine profiles for fixed model + resolution + batch + scheduler classes. Store engines under an AIWF cache with metadata and invalidate on model/LoRA/profile hash changes. **Status:** experimental / expert. **Sources:** [S12][S13][S14].

#### 3.2.2 Torch-TensorRT

Torch-TensorRT is strategically interesting because it integrates with PyTorch workflows and supports `torch.compile` and export routes [S16]. NVIDIAâ€™s 2025 writeup presents Mutable Torch-TensorRT Module behavior for Diffusers and LoRA refit, but the example is FLUX-heavy and vendor-optimized, not a guaranteed consumer Windows SD1.5/SDXL path [S17][S18].

**Recommendation:** research Torch-TensorRT after AIWF has the profile/receipt architecture. Treat it as a future alternative to raw TensorRT export, especially if it handles LoRA/refit better in practice. **Status:** research later / expert experimental. **Sources:** [S16][S17][S18].

#### 3.2.3 ONNX Runtime

Microsoftâ€™s ONNX Runtime Stable Diffusion GPU optimization tree documents useful ideasâ€”Flash Attention, memory-efficient attention, NHWC/channels-last, operator fusionsâ€”but the specific README is marked deprecated and points to Olive recipes [S15]. ONNX export also adds another model format, cache, and debug surface.

**Recommendation:** do not implement ORT as a primary acceleration path now. Track Olive/ORT only if a clear Windows consumer pipeline appears with active Diffusers-compatible maintenance. **Status:** avoid for now / research later. **Sources:** [S15].

#### 3.2.4 RTX 40-series precision: FP8, FP4, NVFP4

Ada Lovelace supports FP8 Tensor Core operations, so FP8 is not fantasy on RTX 4070 Ti SUPER [S21][S23][S30]. The practical issue is not hardware existence; it is safe end-to-end diffusion support. NVIDIAâ€™s TensorRT SDXL PTQ writeup shows that diffusion quantization requires calibration methods because naive PTQ does not work out of the box [S12]. Diffusers torchao docs say FP8 post-training quantization is effective on compute capability >=8.9 and can be combined with compile, but this is still a quantization path requiring local validation [S30]. FP4/NVFP4, by contrast, should be treated as Blackwell-only for this target because NVIDIAâ€™s NVFP4 material and Blackwell architecture pages tie those features to Blackwell Tensor Cores [S24][S25][S26].

**Recommendation:** mark FP8 as experimental for Ada; mark FP4/NVFP4 as unavailable on RTX 4070 Ti SUPER and â€œresearch later for Blackwell.â€ **Status:** FP8 experimental; FP4/NVFP4 avoid for current target. **Sources:** [S12][S21][S23][S24][S25][S26][S30].

---

### 3.3 Quantization and memory reduction

Quantization is not one feature. It is a family of techniques with different hardware assumptions, module support, quality risks, and LoRA implications. Diffusersâ€™ quantization blog explicitly frames backend choice by objective and hardware: bitsandbytes for easy NVIDIA memory savings, torchao/GGUF/bnb+compile for speed experiments, Quanto for some device coverage, FP8 layerwise casting on Hopper/Ada, and GGUF when models are already provided in that format [S27].

#### 3.3.1 bitsandbytes

Diffusersâ€™ bitsandbytes docs describe 8-bit and 4-bit quantization for models with Linear layers, especially transformer/T5-style components; examples avoid quantizing CLIP and AutoencoderKL because CLIP is small and VAE has few Linear layers [S28]. Transformersâ€™ bitsandbytes docs also warn about outlier thresholds and possible quality loss when dequantizing [S29].

For SD1.5/SDXL, the denoiser is UNet-heavy with many convolutions; bitsandbytes does not magically compress the whole pipeline. It may help text encoders or newer transformer-based diffusion models, but the memory win for classic SDXL image generation may be modest unless targeting a huge text encoder or future model family.

**Recommendation:** do not present bitsandbytes as â€œquantize Stable Diffusionâ€ for SD1.5/SDXL. Add it as an experimental backend for Linear-heavy components and future DiT/Flux/SD3-style models; never quantize VAE by default. **Status:** experimental for AIWFâ€™s current SD1.5/SDXL target; more useful for future large transformer diffusion. **Sources:** [S27][S28][S29].

#### 3.3.2 torchao

Diffusers torchao docs support quantization with PyTorch 2.5+, Linear layers, and `torch.compile`; they call FP8 effective on compute capability >=8.9 and show benchmark-oriented examples [S30]. For RTX 4070 Ti SUPER, compute capability 8.9 makes FP8 plausible. The catch is still component support and quality validation: classic SD UNet convolution layers are not the same as transformer blocks.

**Recommendation:** implement a torchao experiment after the receipt system. Start with SDXL text encoder/transformer-like future components, then evaluate UNet support only if upstream Diffusers examples explicitly cover it for the target model. **Status:** experimental. **Sources:** [S23][S27][S30].

#### 3.3.3 optimum-quanto

Diffusers has Quanto docs, but the optimum-quanto repository says the project is in maintenance mode and recommends bitsandbytes or torchao for active production work [S31][S32]. That makes it a poor default dependency for a maintainable local consumer app.

**Recommendation:** do not add optimum-quanto as a first-class AIWF feature unless a specific model requires it and no better backend exists. **Status:** avoid for now. **Sources:** [S31][S32].

#### 3.3.4 GGUF-style approaches

Diffusers supports GGUF loading, but docs say pipeline loading is not supported in the same way, weights are dynamically dequantized/cast during forward, and optimized kernels may introduce minor numerical/visual differences [S33]. GGUF is valuable for community model distribution and memory-constrained future paths, but it is not a clean universal optimization for SD1.5/SDXL Diffusers pipelines.

**Recommendation:** add GGUF support only as an import/model-loading compatibility experiment, not as a default optimization. Require model config resolution and clear warnings about visual differences. **Status:** experimental. **Sources:** [S33].

#### 3.3.5 NVIDIA ModelOpt

NVIDIA ModelOpt provides diffusion quantization, cache diffusion, PTQ/QAT/QAD, and TensorRT deployment support [S19][S34]. It is powerful but heavy. It also blends quantization with a deployment toolchain and, for NVFP4, points at Blackwell requirements [S19][S24][S25][S26][S34].

**Recommendation:** keep ModelOpt in the TensorRT/Engine Lab lane, not baseline AIWF. **Status:** expert experimental. **Sources:** [S19][S34].

---

### 3.4 Samplers and schedulers

Schedulers are one of the highest-value, lowest-dependency speed/quality controls. Diffusers exposes many schedulers and maps common k-diffusion/A1111-style names to Diffusers scheduler classes [S35].

#### 3.4.1 SD1.5 scheduler defaults

For SD1.5, AIWF should default to a balanced scheduler that gives strong quality around 20â€“30 steps. Euler is simple, robust, and documented as producing good outputs around 20â€“30 steps [S37]. DPM-Solver++ / DPMSolverMultistep is a strong quality default; Diffusers says it can produce high-quality samples around 20 steps and quite good samples around 10 steps, with solver_order=2 recommended for guided sampling [S36]. UniPC is also attractive for lower step counts, with official docs framing it as a predictor-corrector method that improves quality in few steps [S38].

**Recommendation:** SD1.5 default profile should offer:
- Balanced: `DPMSolverMultistepScheduler` / DPM++ 2M-style, Karras sigmas if model receipt supports it, 20â€“25 steps.
- Compatibility: Euler, 20â€“30 steps.
- Fast preview: UniPC or DPM++ around 10â€“15 steps, visibly labeled lower-step preview.

**Status:** production-ready. **Sources:** [S35][S36][S37][S38].

#### 3.4.2 SDXL scheduler defaults

SDXL needs more care. Diffusersâ€™ SDXL pipeline docs warn that DPM++ at fewer than 50 steps can produce artifacts due to numerical instability, and recommend using Karras sigmas, `lu_lambdas=True`, or `euler_at_final=True` to mitigate [S42]. SDXL also works best at 1024Ã—1024; lower sizes can work, but quality drops, and below 512 is not recommended [S42].

**Recommendation:** SDXL default should be:
- Balanced: DPM++/DPMSolverMultistep with Karras sigmas or Euler fallback, 30 steps, 1024 square or equivalent area.
- Quality: 35â€“45 steps, optional refiner at 0.8 high-noise split.
- Compatibility: Euler 30 steps.
Do not ship â€œDPM++ 15 steps SDXL qualityâ€ as default unless receipts show artifact-free behavior for a given checkpoint.

**Status:** production-ready with SDXL-specific safeguards. **Sources:** [S36][S37][S41][S42].

#### 3.4.3 Low-step and distilled schedulers

LCMScheduler, TCDScheduler, and distilled-model schedulers are not replacements for normal schedulers. LCM docs state LCMs are distilled models/adapters that can generate in about four steps but negative prompts do not work in the usual way because guidance embeddings are used and batch doubling is avoided [S39]. TCD is an explicit consistency distillation method for low-step generation [S40]. Lightning, Turbo, Hyper-SD, and PCM similarly require specific checkpoint/LoRA/scheduler/CFG combinations [S44][S46][S48][S49][S50].

**Recommendation:** low-step schedulers belong in Fast Mode, not hidden behind the ordinary scheduler dropdown. **Status:** beta/experimental by method. **Sources:** [S39][S40][S44][S46][S48][S49][S50].

---

### 3.5 Fast generation methods

Fast generation methods are exciting, but they change the generation contract. AIWF should not disguise them as normal optimizations. A user selecting â€œSDXL qualityâ€ and getting CFG=0, no negative prompt, a step-specific LoRA, and a different denoising target will feel like the app has opened the airlock without asking.

#### 3.5.1 LCM and LCM-LoRA

Diffusers documents LCMs and LCM-LoRAs as distilled adapters/checkpoints that can generate in about 4 steps and support SD1.5, SDXL, SSD-1B, ControlNet, and T2I-Adapter in some forms [S39]. LCM-LoRA SDXL model cards describe 2â€“8 step usage and require LCMScheduler [S49]. The important UX caveat is negative prompt and guidance behavior: Diffusers says negative prompts do not work with LCM in the normal way and LCM guidance values differ from standard CFG [S39].

**Recommendation:** add LCM-LoRA as **Fast Mode / Beta**. Use separate defaults: 4â€“8 steps, LCMScheduler, guidance around 1â€“2 if the model card recommends it, and hide or annotate normal negative prompt behavior. **Status:** beta. **Sources:** [S39][S49].

#### 3.5.2 SDXL Turbo

SDXL Turbo is a distilled SDXL model using Adversarial Diffusion Distillation. Its model card says it can synthesize images in one network evaluation, supports 1â€“4 steps, uses guidance_scale=0.0, should not use normal negative prompts, and generally prefers 512Ã—512 in its examples [S44][S45].

**Recommendation:** implement SDXL Turbo as a separate model profile, not as an SDXL base acceleration toggle. **Status:** beta as separate model family. **Sources:** [S44][S45].

#### 3.5.3 SDXL Lightning

SDXL-Lightning provides step-specific 1/2/4/8-step checkpoints and LoRAs. The model card states the full UNet has best quality, LoRA is intended for other SDXL base models/custom checkpoints, Euler scheduler should use `timestep_spacing="trailing"`, CFG should be 0, and the 1-step model is experimental/less stable [S46]. The associated paper presents progressive adversarial distillation for one/few-step 1024px generation [S47].

**Recommendation:** implement Lightning as Fast Mode / Beta:
- 4-step and 8-step LoRA/full-UNet variants first.
- 1-step hidden under experimental.
- Scheduler and CFG locked to model-card-compatible settings unless advanced override is enabled.

**Status:** beta for 4/8 steps; experimental for 1 step. **Sources:** [S46][S47].

#### 3.5.4 Hyper-SD, TCD, PCM

Hyper-SD supports multiple base families and both LoRA/UNet variants, with scheduler-specific instructions including DDIM/TCD/LCM depending on checkpoint [S48]. TCD is documented by Diffusers as a consistency distillation path with scheduler-specific behavior [S40]. PCM LoRAs have step/CFG/scheduler caveats and note that LoRA may not be sufficient for true 1-step quality [S50].

**Recommendation:** keep Hyper-SD/TCD/PCM in Experimental Lab until AIWF has per-method recipes, compatibility checks, and quality receipts. **Status:** experimental. **Sources:** [S40][S48][S50].

---

### 3.6 Quality improvements

#### 3.6.1 Prompt processing and Compel

Diffusersâ€™ prompting guide supports weighted prompt embeddings through `prompt_embeds` and `pooled_prompt_embeds` [S51]. Compel remains the practical library for weighted prompts and blends in Diffusers-style pipelines [S52][S53]. However, current Compel main documentation should be checked against AIWFâ€™s Transformers 4.x constraint; the repository currently indicates a Transformers 5-oriented mainline, so AIWF should pin a known-compatible Compel release or maintain a compatibility adapter [S52].

**Recommendation:** keep Compel, but make prompt embedding generation a service with version checks:
- Stable weighted syntax: `(...)`, `++`, explicit weights, blends if supported by pinned version.
- SDXL path must generate both prompt embeddings and pooled prompt embeddings.
- Textual inversion and LoRA text-encoder edge cases should have tests.

**Status:** production-ready with pinning; dependency risk if unpinned. **Sources:** [S51][S52][S53].

#### 3.6.2 Negative prompts

Negative prompts remain useful for standard SD1.5/SDXL CFG generation, but fast/distilled methods often alter or remove normal CFG semantics. LCM docs explicitly warn about negative prompt behavior, and SDXL Turbo/Lightning model cards require CFG=0/no normal negative prompt guidance in many recipes [S39][S44][S46].

**Recommendation:** negative prompt UI should be context-sensitive. In standard modes, keep it prominent. In Fast Mode, annotate or disable it when the selected method does not use standard CFG. **Status:** production-ready UX safeguard. **Sources:** [S39][S44][S46].

#### 3.6.3 Clip skip

Diffusers pipeline APIs include `clip_skip` parameters for relevant Stable Diffusion pipelines, and it remains important for compatibility with many anime/community SD1.5 checkpoints [S57]. Clip skip changes prompt interpretation and should be model-profile-specific, not global.

**Recommendation:** expose clip skip in advanced model settings and allow checkpoint metadata/default override. **Status:** production-ready. **Sources:** [S57].

#### 3.6.4 FreeU

FreeU rebalances U-Net skip/backbone features and is an inference-only quality tweak. Diffusers documents it for text-to-image, image-to-image, and text-to-video pipelines and provides parameter examples for SDXL [S54]. It changes output and should therefore be visible.

**Recommendation:** add FreeU as a Quality Lab toggle with named presets per model family. Store parameters in receipts. **Status:** beta. **Sources:** [S54].

#### 3.6.5 PAG / Perturbed Attention Guidance

Diffusers documents PAG as a quality improvement that does not require external modules/training and is implemented for selected pipelines [S55]. Because it changes denoising behavior and supported pipelines are limited, it should be opt-in.

**Recommendation:** add PAG after standard profiles are stable. Detect pipeline support and expose per-layer presets only in advanced mode. **Status:** experimental-to-beta for supported pipelines. **Sources:** [S55].

#### 3.6.6 Hi-res fix

Hi-res fix is best treated as a two-stage generation workflow:
1. Generate a lower-resolution latent/image.
2. Upscale.
3. Re-denoise with controlled strength.

AIWF should support both latent upscale and external upscaler integration, but should record which path was used. For SDXL, native 1024 generation often reduces the need for classic SD1.5-style hires fix; for SD1.5, hires fix remains important for >512 outputs. VAE tiling may be required at high resolutions, but because tiling can cause tonal variation, it must be recorded and visible [S01][S41][S42][S65].

**Recommendation:** implement hires fix as a first-class workflow service, not a UI hack. Include `first_pass_size`, `upscale_method`, `denoise_strength`, `vae_tiling`, `scheduler_second_pass`, and `seed_policy` in receipts. **Status:** production-ready once benchmarked. **Sources:** [S01][S41][S42][S65].

#### 3.6.7 SDXL refiner

Diffusers documents SDXL base/refiner ensemble workflows using `denoising_end` and `denoising_start`, commonly around a high-noise split such as 0.8, and notes a post-hoc img2img refiner path is slower because it requires more function evaluations [S41]. Stability AIâ€™s SDXL model card also frames base+refiner as the highest-quality two-stage workflow [S43].

**Recommendation:** keep SDXL refiner optional. Default standard SDXL base-only for speed; Quality profile can enable base+refiner with a visible latency estimate. **Status:** production-ready optional. **Sources:** [S41][S43].

#### 3.6.8 VAE choices

VAE selection changes output. The SDXL fp16-fix VAE can reduce fp16 numerical issues but is not identical to the original VAE [S66]. AutoencoderKL docs show why SDXL VAEs historically used upcast behavior [S65].

**Recommendation:** model profiles should include VAE metadata and allow user selection. Do not silently replace VAEs. **Status:** production-ready. **Sources:** [S65][S66].

#### 3.6.9 Inpaint quality

Diffusers recommends inpainting-specific checkpoints for inpainting; base checkpoints can be used but are less performant for inpaint behavior [S57]. Diffusers inpainting docs define white mask = inpaint and black = preserve, and show mask blur to soften transitions [S56]. For quality, AIWF should support `padding_mask_crop`, mask blur, strength, inpaint area, and correct VAE/component reuse. There is also a historical Diffusers issue about SDXL mask blur behavior, so AIWF should add a regression test around mask blur softness [S58].

**Recommendation:** implement inpaint as a dedicated workflow with model-family-aware defaults:
- Use inpaint checkpoint when available.
- Mask blur visible.
- `padding_mask_crop`/masked-area crop supported.
- Store mask hash and blur radius in receipts.
- Regression-test SDXL mask blur.

**Status:** production-ready with tests. **Sources:** [S56][S57][S58].

#### 3.6.10 ControlNet quality and preprocessors

ControlNet quality depends heavily on preprocessor quality, conditioning scale, image resize behavior, and model family match. Diffusers documents conditioning scales and MultiControlNet tips such as masking conditionings to avoid overlap and experimenting with scales [S59]. The `controlnet_aux` repository provides many preprocessors and notes DWPose can be used via `easy-dwpose` without heavyweight MMDetection/MMCV/MMPose dependencies [S61]. SDXL ControlNet support remains more fragmented; Diffusers docs note many SDXL ControlNet checkpoints are experimental and have room for improvement [S63].

**Recommendation:** AIWF should ship a minimal ControlNet core and lazy-load optional preprocessors:
- Built-in: canny, simple image resize/normalization.
- Optional: depth, lineart, openpose/dwpose, segmentation, normal maps.
- Match SD1.5 ControlNet with SD1.5 base and SDXL ControlNet with SDXL base.
- Expose control scale, start/end, preprocessor resolution.
- Add per-preprocessor dependency probes.

**Status:** SD1.5 ControlNet production-ready; SDXL ControlNet mixed/beta depending on model. **Sources:** [S59][S60][S61][S62][S63].

---

### 3.7 Architecture recommendations for AIWF Studio

AIWFâ€™s clean layered architecture is a major advantage. Optimization should be modeled as data and capability decisions, not as mutable global state. The goal is to make every generated image explainable: â€œwhich model, which profile, which backend, which scheduler, which memory tricks, which quality modifiers, which source image/mask/control image, and which dependency versions produced this?â€

#### 3.7.1 Proposed core domain models

```python
@dataclass(frozen=True)
class OptimizationProfile:
    profile_id: str
    pipeline_kind: Literal["txt2img", "img2img", "inpaint", "controlnet", "hires", "fast"]
    model_family: Literal["sd15", "sdxl", "sdxl_turbo", "flux", "sd3", "unknown"]
    dtype_policy: str
    attention_backend: str
    memory_policy: str
    vae_policy: str
    compile_policy: str
    quant_policy: str
    engine_policy: str
    scheduler_policy: str
    quality_modifiers: tuple[str, ...]
    fast_method: str | None
```

This is a domain object, not an infrastructure object. It expresses intent. Infrastructure resolves it into actual Diffusers/PyTorch calls.

#### 3.7.2 Capability detection service

Create `CapabilityDetector` in infrastructure, composed through `AppContext`, returning a serializable capability report:

- OS, Python, GPU name, VRAM, compute capability.
- Torch, CUDA runtime, driver, Diffusers, Transformers, Accelerate, PEFT, Safetensors versions.
- Optional package presence: xFormers, flash-attn, sageattention, bitsandbytes, torchao, optimum-quanto, TensorRT, Torch-TensorRT, ONNX Runtime, ModelOpt.
- Runtime probes: SDPA available, channels-last smoke test, compile smoke test, xFormers backend available, CUDA graph compatibility signal if compile mode selected.

**Recommendation:** no optional heavy dependency should import during normal boot unless a feature path requests it. **Status:** production-ready architecture. **Sources:** [S01][S02][S03][S09][S10][S16][S28][S30][S32][S34].

#### 3.7.3 Optimization planner

Create `OptimizationPlanner` as a service that maps:
- model family,
- pipeline kind,
- resolution,
- batch size,
- active LoRAs,
- ControlNet count,
- VAE choice,
- target preset,
- capability report,

into an `OptimizationPlan`.

The planner should use a compatibility registry. Examples:

| Combination | Rule |
|---|---|
| `sequential_cpu_offload` + prior `.to("cuda")` | Reject; offload hooks must be installed before moving pipeline to CUDA [S01]. |
| `torch_compile` + dynamic resolution | Reject unless `dynamic=True` profile or cached profile matches dimensions [S02][S08]. |
| `torch_compile` + LoRA hotswap | Require `enable_lora_hotswap()` ordering and LoRA rank/layer compatibility [S67][S68]. |
| TensorRT + resolution outside engine profile | Reject and offer engine rebuild [S13]. |
| TensorRT + unconverted LoRA | Reject or fallback to PyTorch pipeline [S13]. |
| SDXL Turbo + normal CFG/negative prompt | Replace with Turbo recipe and annotate [S44]. |
| Lightning + wrong step count | Reject or switch checkpoint/LoRA recipe [S46]. |
| VAE tiling + quality-sensitive comparison | Mark output as non-baseline because tiling can alter tones [S01][S65]. |

#### 3.7.4 Fallback behavior

Fallback should be deterministic and logged:

1. Try requested profile.
2. On known compatibility failure, do not attempt generation; return actionable UI message.
3. On runtime backend failure, unload failed pipeline variant, clear CUDA cache, fall back to Safe profile if user allowed fallback.
4. Attach failure to a diagnostic receipt.
5. Never silently switch quality-changing features. If fallback changes output semantics, UI must say so.

#### 3.7.5 User-facing settings

Recommended UI layers:

- **Simple mode:** Safe, Balanced, Quality, Low VRAM, Fast Mode.
- **Advanced mode:** scheduler, steps, precision, VAE slicing/tiling, attention backend, compile, offload, ControlNet settings.
- **Experimental Lab:** TensorRT, Torch-TensorRT, torchao, bitsandbytes, GGUF, SageAttention, FlashAttention, Hyper-SD/TCD/PCM.

This keeps normal users away from the â€œpress all accelerators and discover smokeâ€ workflow.

---

### 3.8 Testing and benchmarking

Benchmarks should answer: â€œIs this faster, does it use less VRAM, did it change output, and is it stable on Windows NVIDIA consumer hardware?â€ A single wall-clock screenshot is not enough.

Core metrics:

- Total latency per image.
- Stage timings: model load, prompt encode, preprocess, denoise, VAE decode, postprocess.
- Steady-state images/sec and iterations/sec.
- First-run latency separately from steady-state latency.
- Compile/build time separately for compile/TensorRT.
- Peak `torch.cuda.max_memory_allocated()`.
- Peak `torch.cuda.max_memory_reserved()`.
- Optional NVML total VRAM peak.
- CPU RAM peak.
- Output hash and perceptual hash.
- Quality labels and visual artifact notes.

Warmup rules:

- Baseline/xFormers/channels-last: one untimed warmup, five timed runs minimum.
- `torch.compile`: separate compile warmup; record compile time and first post-compile run; then five timed steady-state runs.
- TensorRT: record engine build time, first load/warmup, and steady-state.
- Offload: include cold and warm runs because hooks and transfer behavior matter.

Graduation criteria:

- At least 10% median latency improvement or 20% peak VRAM reduction on the target profile.
- No critical crash across repeated runs.
- No unacceptable quality regression in image grid review.
- No unsupported dependency/version conflict.
- Windows behavior verified before defaulting for Windows users.
- Fallback path tested.
- Receipts stored.

**Recommendation:** no optimization graduates to Balanced/Default without benchmark receipts on at least RTX 4070 Ti SUPER-class hardware and one lower-VRAM NVIDIA card if possible. **Status:** production process requirement. **Sources:** [S01][S02][S07][S08][S12][S13][S27].

---

### 3.9 Compatibility risks

#### Diffusers versions

Diffusers is moving quickly. Current docs include experimental APIs such as device placement and attention backend dispatchers [S01][S03]. AIWF should avoid writing to unstable internal APIs and should pin a tested Diffusers version range for releases.

**Mitigation:** dependency lock file, compatibility CI matrix, startup diagnostics, and release-specific migration notes. **Sources:** [S01][S03].

#### Transformers 4.x vs 5.x

The project constraint `transformers>=4.44,<5` should remain. Current ecosystem churn around Transformers/Diffusers compatibility is real; recent GitHub issues show breakage across latest Diffusers/Transformers combinations, and Compel mainline appears to be moving toward Transformers 5 [S52][S71][S72]. AIWF should not casually upgrade Transformers to 5.

**Mitigation:** maintain a Transformers 4 compatibility lane; pin Compel; build an isolated migration branch for Transformers 5 only after checkpoint loading and prompt embedding tests pass. **Sources:** [S52][S71][S72].

#### Torch and CUDA versions

Torch, CUDA wheel, xFormers, flash-attn, torchao, and TensorRT compatibility form a dependency lattice. xFormers wheels are tied to current PyTorch/CUDA combinations; TensorRT requires its own installed runtime; torchao needs modern PyTorch; TensorRT/ModelOpt paths add separate requirements [S09][S10][S16][S19][S30][S34].

**Mitigation:** optional dependency groups, lazy imports, package probe diagnostics, and no mandatory optimizer dependency at boot. **Sources:** [S09][S10][S16][S19][S30][S34].

#### Windows

Windows users are central to AIWF. xFormers now publishes Windows wheels for current combinations, but source builds and long paths remain possible friction [S09]. `torch.compile` on Windows GPU remains riskier because of historical Triton/Inductor support limitations [S11]. TensorRT and ONNX paths add extra runtime installation complexity.

**Mitigation:** Windows-first smoke tests, feature detection, and hide unsupported flags unless probes pass. **Sources:** [S09][S11][S13][S16].

#### PEFT LoRA loading

Diffusers supports LoRA loading/hotswapping and loader mixins, but compiled-model hotswapping requires correct ordering and text encoder LoRA hotswap is unsupported [S67][S68]. TensorRT extension paths may require LoRA conversion [S13].

**Mitigation:** LoRA manager service owns adapter state; compile/engine paths declare LoRA compatibility; no ad hoc LoRA injection from UI. **Sources:** [S13][S67][S68].

#### Safetensors and checkpoint loading

Safetensors is the safest default format because it avoids pickle-style arbitrary code execution and supports fast tensor loading [S69][S70]. However, Diffusers-format folders and single-file checkpoints differ; checkpoint conversion/loading should remain its own tested infrastructure boundary.

**Mitigation:** prefer safetensors; use Diffusers-supported loaders; never use arbitrary pickle checkpoint load in normal user flows. **Sources:** [S69][S70].

---

## 4. Optimization compatibility matrix

The full matrix is in [`compatibility_matrix.md`](compatibility_matrix.md) and [`compatibility_matrix.csv`](compatibility_matrix.csv). The governing rule is: **an optimization is compatible only with a specific pipeline/model/resolution/dependency profile**, not with â€œStable Diffusionâ€ in general.

High-level summary:

| Feature | SD1.5 | SDXL | Inpaint | Img2img | ControlNet | Windows | Default? |
|---|---:|---:|---:|---:|---:|---:|---:|
| PyTorch SDPA | Good | Good | Good | Good | Good | Good | Yes |
| Channels-last | Good | Good | Good | Good | Good | Good with fallback | Yes, after smoke test |
| xFormers | Good if wheel matches | Good if wheel matches | Good if backend supports ops | Good | Good | Wheel/version gated | No |
| `torch.compile` | Profile-specific | Profile-specific | Riskier | Riskier | Riskier | Probe required | No |
| VAE slicing | Useful for batch | Useful for batch | Useful | Useful | Useful | Good | Conditional |
| VAE tiling | High-res useful | High-res useful | Useful | Useful | Useful | Good | User-visible |
| Model CPU offload | Low VRAM | Low VRAM | Low VRAM | Low VRAM | Low VRAM | Good but slower | Low VRAM only |
| Sequential offload | Emergency | Emergency | Emergency | Emergency | Emergency | Very slow | No |
| TensorRT | Expert | Expert | Harder | Harder | Harder | Install gated | No |
| bnb/torchao quant | Limited for classic UNet | Limited/mixed | Mixed | Mixed | Risky | Package gated | No |
| LCM/Turbo/Lightning | Fast family | Fast family | Method-specific | Method-specific | Method-specific | Good if deps simple | Separate Fast Mode |

---

## 5. Recommended default generation profiles

The detailed profile definitions are in [`default_generation_profiles.md`](default_generation_profiles.md).

### 5.1 Safe profile

Use Safe when diagnostics are uncertain.

- PyTorch SDPA.
- fp16 CUDA if available; fp32 CPU/debug fallback.
- No xFormers, no compile, no quantization, no TensorRT.
- Euler or DPM++ stable scheduler.
- No VAE tiling unless user requests.
- No offload unless needed to fit.

**Status:** production-ready. **Sources:** [S01][S02][S35][S36][S37].

### 5.2 Balanced profile

Default for RTX 4070 Ti SUPER.

- fp16.
- PyTorch SDPA.
- Channels-last after smoke test.
- SD1.5: DPM++/DPMSolverMultistep 20â€“25 steps or Euler 25.
- SDXL: DPM++ with Karras/compatible mitigation or Euler 30; 1024 target.
- VAE slicing for batch >1.
- No hidden output-changing quality features.

**Status:** recommended default. **Sources:** [S02][S35][S36][S37][S42].

### 5.3 Quality profile

- SDXL optional base+refiner.
- Higher steps.
- Optional FreeU/PAG as visible toggles.
- Hires fix workflow for SD1.5/high-res.
- VAE tiling only if high-res memory requires it.

**Status:** production-ready with visible quality modifiers. **Sources:** [S41][S42][S43][S54][S55].

### 5.4 Low VRAM profile

- Model CPU offload first.
- VAE slicing/tiling as needed.
- Smaller batch.
- Sequential offload only emergency.

**Status:** production fallback. **Sources:** [S01].

### 5.5 Fast Mode

- Separate method recipes for LCM, SDXL Turbo, Lightning, Hyper-SD/TCD/PCM.
- No normal CFG/negative-prompt assumptions unless the selected method supports them.
- Model-card-compatible scheduler and step count.

**Status:** beta/experimental by method. **Sources:** [S39][S40][S44][S46][S48][S49][S50].

---

## 6. Experimental feature flags to add

The full flag plan is in [`experimental_feature_flags.md`](experimental_feature_flags.md). Recommended first flags:

| Flag | Default | Status | Notes |
|---|---:|---|---|
| `attention.xformers` | off | experimental | Requires import/version probe and benchmark [S09][S10]. |
| `attention.flash` | off | experimental | Backend-specific, not FlashAttention-3 on Ada [S03]. |
| `attention.sage` | off | experimental | Quantized attention; quality receipt required [S03]. |
| `compile.unet` | off | beta/experimental | Same-shape workflows; record compile time and recompiles [S02][S08]. |
| `compile.regional` | off | beta | Prefer over full compile when available [S02][S08]. |
| `memory.group_offload` | off | experimental | Model-dependent; CPU RAM risk [S01]. |
| `quant.bnb` | off | experimental | Linear-heavy components only [S28][S29]. |
| `quant.torchao_fp8` | off | experimental | Ada-capable but quality/backend testing required [S23][S30]. |
| `engine.tensorrt` | off | expert | Engine build/cache lifecycle required [S12][S13]. |
| `engine.torch_tensorrt` | off | research | Promising, but do after receipt substrate [S16][S17][S18]. |
| `fast.lcm` | off | beta | Separate Fast Mode recipe [S39][S49]. |
| `fast.lightning` | off | beta | Step-specific LoRA/UNet; CFG=0 [S46]. |
| `quality.freeu` | off | beta | Visible quality change [S54]. |
| `quality.pag` | off | experimental/beta | Supported pipelines only [S55]. |

---

## 7. Benchmark protocol

See [`benchmark_protocol.md`](benchmark_protocol.md), [`benchmark_receipt_schema.json`](benchmark_receipt_schema.json), and [`benchmark_prompts.json`](benchmark_prompts.json).

The benchmark protocol is intentionally practical:

1. Fix model, VAE, scheduler, prompt, seed, resolution, batch size, LoRA state, ControlNet state.
2. Record dependency and GPU capability information.
3. Warm up according to backend type.
4. Measure timed runs.
5. Record memory peaks.
6. Save images and perceptual hashes.
7. Compare quality using grids and optional metrics.
8. Decide promotion based on predeclared thresholds.

Suggested promotion thresholds:
- Speed optimization: median latency improves by at least 10% with no quality regression.
- Memory optimization: peak VRAM falls by at least 20% or prevents OOM with acceptable latency.
- Quality optimization: user-visible quality improvement in A/B review and no unacceptable artifact class.
- Stability: no crash across repeated runs, fallback works, Windows path verified if feature visible on Windows.

---

## 8. Risks and mitigations

The detailed risk register is in [`risks_and_mitigations.md`](risks_and_mitigations.md).

Top risks:

| Risk | Severity | Mitigation |
|---|---:|---|
| Hidden optimization changes output | High | Every quality-changing feature must be visible and recorded. |
| Dependency churn from xFormers/torchao/TensorRT | High | Optional extras, lazy imports, version probes, no boot requirement. |
| Compile recompiles on resolution/LoRA changes | High | Profile cache keys include shape and LoRA rank/layers; fallback to eager. |
| Windows GPU compile fails | High | Probe before exposing; default to SDPA/eager. |
| TensorRT engines mismatch model/resolution/LoRA | High | Engine metadata, cache invalidation, explicit rebuild path. |
| Fast Mode breaks user expectations | Medium/high | Separate UI mode, method-specific labels, disabled incompatible controls. |
| Offload hooks create state bugs | Medium/high | Pipeline lifecycle service owns hooks; no shared global pipelines. |
| SDXL inpaint mask behavior regresses | Medium | Add mask blur regression image tests. |
| Prompt embedding dependency break | Medium | Pin Compel-compatible release; test SD1.5/SDXL prompt embeds. |

---

## 9. Prioritized implementation roadmap

### 9.1 Do now

1. Implement `OptimizationProfile`, `CapabilityReport`, `OptimizationPlan`, and `BenchmarkReceipt` domain models.
2. Add `CapabilityDetector` with lazy optional dependency probes.
3. Build Safe, Balanced, Quality, Low VRAM, and Fast Mode profile presets.
4. Standardize schedulers and SDXL DPM++ safeguards.
5. Implement VAE slicing/tiling policy with visible quality caveats.
6. Implement model CPU offload Low VRAM profile and sequential emergency profile.
7. Add prompt embedding service around Compel with version pinning and SDXL pooled embeddings tests.
8. Add LoRA manager service that owns load/unload/hotswap state.
9. Add benchmark receipt writing for every generation in developer mode and every benchmark run.
10. Add regression tests for SD1.5, SDXL, inpaint, ControlNet, hires fix.

**Sources:** [S01][S02][S35][S36][S37][S41][S42][S51][S52][S56][S57][S59][S67][S68].

### 9.2 Experiment behind flag

1. xFormers backend.
2. Regional `torch.compile` for repeated fixed-resolution txt2img.
3. Full UNet compile for fixed profiles.
4. FreeU and PAG quality modifiers.
5. LCM-LoRA and SDXL Lightning Fast Mode.
6. torchao FP8/INT8 component experiments.
7. Group offload for larger future models/video.
8. TensorRT Engine Lab prototype.

**Sources:** [S03][S08][S09][S10][S12][S13][S30][S39][S46][S54][S55].

### 9.3 Research later

1. Torch-TensorRT Mutable Module for Diffusers and LoRA refit.
2. NVIDIA ModelOpt Cache Diffusion / PTQ / QAT for stable local workflows.
3. GGUF community model loading.
4. ORT/Olive if active Windows-local recipes mature.
5. Blackwell FP4/NVFP4 when consumer GPUs support it.
6. Hyper-SD/TCD/PCM after LCM/Lightning receipts are stable.

**Sources:** [S15][S16][S17][S18][S19][S24][S25][S33][S40][S48][S50].

### 9.4 Avoid for now

1. Mandatory xFormers at boot.
2. Silent global `torch.compile`.
3. Mandatory TensorRT/ONNX/ModelOpt dependencies.
4. Raw custom CUDA graph capture.
5. Silent quantization of UNet/VAE/ControlNet.
6. Treating FP4/NVFP4 as available on RTX 4070 Ti SUPER.
7. Upgrading to Transformers >=5 without a compatibility branch and checkpoint-loading test suite.
8. Treating SDXL Turbo/Lightning/LCM as ordinary scheduler presets.
9. Sequential CPU offload as default.
10. Copying architecture/code from incompatible projects.

**Sources:** [S01][S02][S07][S10][S11][S13][S15][S24][S25][S26][S39][S44][S46][S52][S71][S72].

---

## 10. Final decision thesis

For AIWF Studio, **maintainability is an optimization**. The fastest local image app is not the one with the longest list of toggles; it is the one that knows which toggles can safely coexist for a specific model, GPU, resolution, and workflow, and can prove it with receipts. Build the optimization substrate first. Then let experimental acceleration modules compete in a controlled arena. The winner earns a profile. The losers go back into the cupboard with a label, not into the default path with a grin.
