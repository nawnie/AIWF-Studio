# AIWF Studio benchmark protocol

**Prepared:** 2026-06-16
**Purpose:** compare image-generation speed, VRAM, stability, and quality fairly across Diffusers optimization profiles.

---

## 1. Benchmark philosophy

Benchmarks must separate:

1. **Cold cost** â€” model load, TensorRT engine build, first `torch.compile`.
2. **First usable generation** â€” what a consumer experiences after clicking generate.
3. **Steady-state generation** â€” repeated same-profile runs.
4. **Memory fit** â€” whether the workflow avoids OOM.
5. **Quality impact** â€” whether the image visibly changes or degrades.

An optimization that saves 800 ms but adds a 90-second compile and breaks LoRA hotswap is not a universal default. It may still be excellent for a fixed batch workflow. That distinction is the whole point of receipts.

---

## 2. Metrics to collect

### Timing metrics

| Metric | Description |
|---|---|
| `load_time_s` | Time to construct/load pipeline and move/apply profile. |
| `compile_or_build_time_s` | `torch.compile`, TensorRT build, ModelOpt calibration/build, etc. |
| `first_generation_time_s` | First completed image after profile is applied. |
| `steady_state_times_s` | List of timed generation runs after warmup. |
| `median_time_s` | Median steady-state generation time. |
| `p90_time_s` | p90 steady-state time if enough runs. |
| `denoise_time_s` | Denoising loop only, if instrumentation available. |
| `prompt_encode_time_s` | Text encoder/Compel time. |
| `preprocess_time_s` | ControlNet/inpaint preprocessing time. |
| `vae_decode_time_s` | VAE decode time. |
| `postprocess_time_s` | PIL/metadata/upscale postprocessing. |

### Memory metrics

| Metric | Description |
|---|---|
| `torch_max_memory_allocated_bytes` | `torch.cuda.max_memory_allocated()` after reset. |
| `torch_max_memory_reserved_bytes` | `torch.cuda.max_memory_reserved()` after reset. |
| `nvml_peak_used_bytes` | Optional NVML peak memory including non-PyTorch allocations. |
| `cpu_rss_peak_bytes` | Optional process RSS peak. |
| `oom` | Boolean. |
| `oom_stage` | load, prompt_encode, preprocess, denoise, vae_decode, postprocess. |

### Quality metrics

| Metric | Description |
|---|---|
| `sha256_image` | Exact PNG/image bytes hash. |
| `phash` | Perceptual hash for difference detection. |
| `clip_score` | Optional prompt-image alignment metric. |
| `lpips_vs_baseline` | Optional perceptual distance if baseline image exists. |
| `artifact_labels` | Human labels: seam, banding, face damage, bad hands, composition drift, prompt miss, control drift. |
| `human_verdict` | better / same / worse / invalid. |

### Stability metrics

| Metric | Description |
|---|---|
| `exception_type` | Exception class if failure. |
| `trace_digest` | Hash/summary of traceback, not full private paths by default. |
| `fallback_used` | Whether fallback profile ran. |
| `recompile_count` | If observable. |
| `engine_cache_hit` | For TensorRT/Torch-TensorRT. |
| `lora_reload_or_hotswap` | LoRA state changes during run. |

---

## 3. Warmup rules

| Feature class | Warmup |
|---|---|
| Safe/Balanced eager | 1 untimed warmup, then 5 timed runs. |
| xFormers/attention backend | 1 backend warmup, then 5 timed runs. |
| channels-last | 1 smoke generation, then 5 timed runs. |
| `torch.compile` | Record compile time separately; run 1 post-compile warmup; then 5 timed runs. |
| regional compile | Record regional compile time; 1 warmup; 5 timed runs. |
| TensorRT | Record engine build time; 1 engine load/warmup; 5 timed runs. |
| CPU/model offload | 1 warmup; 5 timed runs; include CPU RAM peak. |
| Fast Mode | 1 warmup; 5 timed runs for each method recipe. |

Minimum timed runs: **5**. Preferred: **10** for promotion decisions.

---

## 4. Fair comparison controls

Hold these constant when comparing profiles:

- model checkpoint path and hash,
- VAE path and hash,
- LoRA set, ranks, weights, and order,
- ControlNet models and control images,
- input image and mask,
- prompt and negative prompt,
- Compel settings,
- seed,
- resolution,
- batch size,
- scheduler and scheduler config unless scheduler is the variable,
- step count unless step count is the variable,
- CFG/guidance unless guidance method is the variable,
- dtype unless dtype is the variable,
- output format.

For fast/distilled methods, do **not** compare â€œ4-step Lightningâ€ to â€œ30-step SDXLâ€ as if it is a hidden optimizer. Compare it as a separate mode with quality labels.

---

## 5. Benchmark suites

### Suite A â€” SD1.5 txt2img baseline

| Field | Value |
|---|---|
| Model | SD1.5-compatible local checkpoint |
| Resolution | 512Ã—512 |
| Batch | 1 |
| Steps | 25 |
| Scheduler | DPM++/DPMSolverMultistep and Euler separate runs |
| CFG | 7 |
| Seed | fixed |
| Prompt set | `sd15_txt2img` from benchmark_prompts.json |

Purpose:
- baseline speed,
- attention backend comparison,
- channels-last comparison,
- scheduler comparison.

### Suite B â€” SDXL txt2img baseline

| Field | Value |
|---|---|
| Model | SDXL base local checkpoint |
| Resolution | 1024Ã—1024 |
| Batch | 1 |
| Steps | 30 |
| Scheduler | DPM++ with Karras/mitigation and Euler separate runs |
| CFG | 5â€“7 |
| Seed | fixed |
| Prompt set | `sdxl_txt2img` |

Purpose:
- SDXL default validation,
- VRAM profile,
- refiner comparison,
- compile fixed-profile test.

### Suite C â€” SDXL refiner

| Field | Value |
|---|---|
| Base model | SDXL base |
| Refiner | SDXL refiner |
| Split | 0.8 high_noise_frac preset |
| Steps | Base/refiner combined recipe |
| Prompt set | 3 SDXL prompts |

Purpose:
- latency/VRAM impact of refiner,
- quality review.

### Suite D â€” img2img

| Field | Value |
|---|---|
| Input images | portrait, product, landscape |
| Strength | 0.35 and 0.65 |
| Resolution | source-normalized |
| Steps | 25 SD1.5, 30 SDXL |
| Scheduler | default family |

Purpose:
- variable input path stability,
- prompt encode vs denoise timing,
- compile suitability.

### Suite E â€” inpaint

| Field | Value |
|---|---|
| Input images | portrait face region, object removal, background extension |
| Mask | binary + blurred variants |
| Mask blur | 0, 4, 8 |
| Padding/crop | off/on |
| Strength | 0.75 default |
| Model | inpaint checkpoint when available |

Purpose:
- inpaint quality,
- mask blur regression,
- `padding_mask_crop`,
- VAE tiling/slicing at high resolution.

### Suite F â€” ControlNet

| Field | Value |
|---|---|
| Control types | canny, depth, pose |
| Model family | SD1.5 first; SDXL separately |
| Scale | 0.7, 1.0 |
| Start/end | 0.0/1.0 and 0.0/0.8 |
| Preprocessor resolution | fixed |

Purpose:
- preprocessor latency,
- ControlNet memory impact,
- multi-control warning behavior.

### Suite G â€” hires fix

| Field | Value |
|---|---|
| First pass | 512 or 768 |
| Upscale | 1.5Ã— and 2Ã— |
| Denoise | 0.25, 0.35, 0.45 |
| VAE tiling | off/on when needed |
| Upscalers | latent and external enhancer path if available |

Purpose:
- second-pass quality,
- high-res memory,
- VAE tiling impact.

### Suite H â€” Fast Mode

| Method | Required checks |
|---|---|
| LCM-LoRA | LCMScheduler, 4/8 steps, low guidance, negative prompt annotation |
| SDXL Turbo | CFG=0, 1/2/4 steps, resolution recipe |
| SDXL Lightning | matching 4/8-step LoRA/UNet, Euler trailing, CFG=0 |
| Hyper-SD/TCD/PCM | method-specific scheduler/CFG/timestep recipe |

Purpose:
- user-facing Fast Mode defaults,
- quality tradeoff documentation,
- no hidden scheduler substitution.

---

## 6. Suggested benchmark prompt sets

The full JSON is in [`benchmark_prompts.json`](benchmark_prompts.json). Keep prompt sets small but diagnostic:

- photoreal portrait,
- product/object,
- landscape/environment,
- stylized/anime,
- complex composition,
- hands/face stress,
- high-frequency detail,
- negative prompt stress,
- ControlNet structure,
- inpaint boundary.

Prompts should be stable over time. Do not tune prompts to flatter one backend.

---

## 7. VRAM tracking implementation notes

Recommended PyTorch memory capture:

```python
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()
image = pipeline(**kwargs).images[0]
torch.cuda.synchronize()
elapsed = time.perf_counter() - start
allocated = torch.cuda.max_memory_allocated()
reserved = torch.cuda.max_memory_reserved()
```

Optional NVML capture catches non-PyTorch allocations and TensorRT engines:

```python
# Use pynvml if installed; keep optional.
```

Record both allocated and reserved. Reserved memory can look scary but explains why repeated runs may fit after warmup.

---

## 8. Quality regression detection

Use three layers:

1. **Hard invalid:** error, black image, NaNs, OOM, obvious corruption.
2. **Image difference:** pHash/LPIPS detects major shifts.
3. **Human review:** compare baseline grid vs candidate grid.

Quality-changing features must not be judged against exact-pixel equality. `torch.compile`, alternate attention kernels, quantization, VAE tiling, and TensorRT can produce numerical differences. The decision is whether differences are acceptable for the profile.

### Artifact labels

Use a controlled vocabulary:

```text
seam
tone_shift
banding
washed_out
over_sharp
face_damage
hand_damage
text_artifact
composition_drift
prompt_miss
control_drift
mask_edge
inpaint_texture_mismatch
color_cast
checkerboard
nan_black
```

---

## 9. Promotion criteria

### To graduate from Experimental to Beta

- Runs on target GPU without critical crash.
- Fallback works.
- Receipts complete.
- Quality review does not show unacceptable artifacts.
- Documentation explains changed semantics.
- No mandatory boot dependency.

### To graduate from Beta to Balanced/Default

- Median speed improvement >=10% **or** peak VRAM reduction >=20%.
- Windows path verified if exposed to Windows users.
- LoRA path tested if feature touches denoiser/text encoder.
- Inpaint/ControlNet unaffected or correctly blocked.
- No hidden output-changing behavior.
- At least one lower-VRAM or non-target NVIDIA card tested if memory feature.

### To remain Experimental

- Works only for fixed resolutions.
- Requires minutes of build/compile.
- Needs heavyweight dependency.
- Changes quality or CFG semantics.
- Has incomplete Windows coverage.

### To avoid

- Cannot be detected reliably.
- Frequently breaks dependency constraints.
- Silently changes output.
- Requires Transformers >=5 without migration.
- Claims FP4/NVFP4 benefit on Ada RTX 40-series.

---

## 10. Benchmark receipt storage

Recommended path:

```text
aiwf_data/
  benchmarks/
    receipts/
      2026-06-16/
        <receipt_id>.json
    images/
      <receipt_id>/
        baseline.png
        candidate.png
        grid.png
```

Receipt ID should be a hash of:
- timestamp,
- git commit,
- model hash,
- profile ID,
- prompt ID,
- seed,
- resolution,
- dependency versions.

---

## 11. Minimal benchmark command design

```text
aiwf benchmark run --suite sdxl_txt2img --profile balanced
aiwf benchmark run --suite sdxl_txt2img --profile compile_unet_fixed_1024
aiwf benchmark compare --baseline <receipt_id> --candidate <receipt_id>
aiwf benchmark summarize --gpu "RTX 4070 Ti SUPER"
```

The CLI can be internal at first. The UI can call the same service later.

---

## 12. Common benchmark traps

| Trap | Fix |
|---|---|
| Counting TensorRT engine build as normal generation | Record build separately and report first-use cost. |
| Comparing different schedulers and calling it an optimization | Label scheduler as the variable. |
| Ignoring VAE decode memory | Stage timing/memory around decode. |
| Testing only txt2img | Include img2img, inpaint, ControlNet, hires. |
| Changing resolution between compile runs | Use fixed compile profile or record recompile. |
| Using random seeds | Fixed seed set. |
| Judging fast methods by normal CFG assumptions | Use method-specific UX and quality labels. |
| Hiding VAE tiling | Record it; it can change output. |
