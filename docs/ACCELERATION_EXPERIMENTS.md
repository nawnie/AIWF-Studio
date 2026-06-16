# AIWF Studio — Acceleration Experiments

This document tracks every performance experiment that is **flagged but not yet verified** on local hardware. Nothing listed here is enabled by default. Each experiment ships only after a local benchmark confirms measurable improvement with no regressions.

---

## Ground rules

1. **Flag-gated only.** Every experiment lives behind an `AIWF_*=1` env var. The flag is never read in UI or domain code — only in the relevant infrastructure module.
2. **Benchmark before claiming.** A claim about speed, VRAM, or quality requires a logged benchmark entry (see [Protocol](#benchmark-protocol) below).
3. **No partial enabling.** Do not enable an experiment partially (e.g. only for certain resolutions) without documenting the condition in this file.
4. **Revert cleanly.** Removing the env var must restore the original code path with zero behaviour change.
5. **No `shell=True`.** Even in benchmark scripts.

---

## Experiments

### CUDA Graphs — SDXL denoising loop

| Property | Value |
|----------|-------|
| Flag | `AIWF_CUDA_GRAPHS=1` |
| Status | **Scaffolded, not benchmarked** |
| Expected gain | 5–15% step throughput on RTX 30/40 series |
| Risk | Graph capture requires static tensor shapes; dynamic batch size or resolution breaks capture |
| File | `aiwf/infrastructure/diffusers/cuda_graphs.py` |

**What to do:** Wrap the UNet/transformer forward pass in `torch.cuda.graph`. Capture once on the first run with a fixed shape, replay on subsequent steps. Fall back to eager if shapes change.

**Benchmark signal:** `steps/second` on SDXL 1024×1024, 20 steps, batch 1, same seed. Report first-run (capture) vs. steady-state (replay).

---

### `torch.compile` — UNet / transformer

| Property | Value |
|----------|-------|
| Flag | `AIWF_TORCH_COMPILE=1` |
| Status | **Not started** |
| Expected gain | 10–30% on Ampere+ with `mode="reduce-overhead"` |
| Risk | Long first-run compile (~60 s); breaks with some custom attention kernels |
| File | `aiwf/infrastructure/diffusers/backend.py` |

**What to do:** Call `torch.compile(unet, mode="reduce-overhead", fullgraph=False)` after model load. Guard with `hasattr(torch, "compile")` (requires PyTorch ≥ 2.0).

**Benchmark signal:** Same as CUDA Graphs. Also measure cold-start (first image) vs. warm (second image).

---

### Channels-last memory format — image UNet

| Property | Value |
|----------|-------|
| Flag | `AIWF_CHANNELS_LAST=1` |
| Status | **Not started** |
| Expected gain | 2–8% on NHWC-optimised cuDNN kernels |
| Risk | Some custom layers break with non-contiguous tensors |
| File | `aiwf/infrastructure/diffusers/backend.py` |

**What to do:** Call `unet.to(memory_format=torch.channels_last)` after loading. Only apply to 2D-conv heavy models (SD 1.5, SDXL) — not transformers.

---

### SageAttention 2 / Triton-Windows

| Property | Value |
|----------|-------|
| Flag | `AIWF_SAGE_ATTN=1` |
| Status | **Verify install first** |
| Expected gain | 20–40% attention speedup (claimed by SageAttention authors) |
| Risk | Requires Triton; Windows Triton support is unstable; not available via pip on all platforms |
| File | `aiwf/infrastructure/torch/attention.py` |

**What to do:** Check `import sageattention` succeeds; if so, call `sageattention.sageattn` as the attention function. Gate the entire code path on a successful import — never hard-require.

**Prerequisite:** Confirm `triton` wheels for Windows/CUDA are available in the user's venv before advertising this flag.

---

### torchao native quantization

| Property | Value |
|----------|-------|
| Flag | `AIWF_TORCHAO=1` |
| Status | **Scaffolded, not benchmarked** |
| Expected gain | Possible VRAM/file-size reduction; no speed claim without benchmark |
| Risk | torchao API changed between 0.3 and 0.5; verify `quantize_` signature before using |
| File | `aiwf/infrastructure/quantization/torchao_quant.py` |

**What to do:** Keep runtime quantization optional and flag-gated. The Models tab now has a preflight/job lane for quantization planning; destructive quantized export stays blocked until quality validation and benchmark receipts exist.

---

### NVFP4 storage/compression

| Property | Value |
|----------|-------|
| Flag | None for runtime speed; preflight only in Models tab |
| Status | **Preflight guidance only** |
| Expected gain | Smaller storage where a supported exporter exists |
| Risk | RTX 4070 Ti SUPER is Ada Lovelace, not Blackwell; do not present NVFP4 as a native speedup |
| File | `aiwf/services/model_ops.py` |

**What to do:** Treat NVFP4 as a compression/storage choice unless future Blackwell-class hardware and a verified runtime path are detected. VAE quantization is allowed as a research target but remains blocked until decode-quality validation lands.

---

### FP8 Ada Lovelace paths

| Property | Value |
|----------|-------|
| Flag | `AIWF_FP8=1` |
| Status | **Partially done (Wan pipeline only)** |
| Expected gain | 40–60% VRAM reduction on RTX 40 series |
| Risk | Only RTX 40 series has native FP8 tensor cores; falls back to emulation on earlier cards |
| File | `aiwf/infrastructure/wan/pipeline.py` (already gated) |

**Remaining work:** Extend to SDXL denoising UNet. Requires `torch.float8_e4m3fn` dtype support (PyTorch ≥ 2.1).

---

### NVENC video export

| Property | Value |
|----------|-------|
| Flag | `AIWF_NVENC=1` |
| Status | **Scaffolded, not benchmarked** |
| Expected gain | GPU-accelerated H.264/H.265 encoding; 3–5× faster export |
| Risk | Requires NVIDIA GPU + NVENC support in ffmpeg build |
| File | `aiwf/infrastructure/video/export.py` |

**What to do:** When `AIWF_NVENC=1`, pass `-c:v h264_nvenc` to ffmpeg subprocess instead of `libx264`. Detect support at startup with a probe call; fall back silently if NVENC not available.

---

### RTX VSR (Video Super Resolution)

| Property | Value |
|----------|-------|
| Flag | `AIWF_VSR=1` |
| Status | **Do not claim without measurement** |
| Expected gain | Perceptual quality improvement; no pure speed gain |
| Risk | RTX VSR is a driver-level feature, not a PyTorch API — integration path unclear |
| File | TBD |

**What to do:** Research the integration path (TensorRT plugin? driver API?). Do not implement until a verified path exists on Windows + RTX.

---

## Benchmark protocol

Every benchmark entry must record:

| Field | Example |
|-------|---------|
| Date | 2026-06-14 |
| Flag enabled | `AIWF_TORCH_COMPILE=1` |
| GPU | RTX 4090 24 GB |
| Driver | 551.86 |
| CUDA | 12.3 |
| PyTorch | 2.3.0+cu121 |
| Model | SDXL 1.0 (stabilityai/stable-diffusion-xl-base-1.0) |
| Resolution | 1024×1024 |
| Steps | 20 |
| Sampler | Euler a |
| Batch | 1 |
| Seed | 42 |
| VRAM peak | 10.2 GB |
| First-run time | 38.4 s |
| Second-run time | 9.1 s |
| Baseline time | 11.3 s |
| Gain | −19% (second run vs. baseline) |
| Output hash | SHA-256 of PNG bytes (must match baseline for lossless paths) |

Log entries go in `docs/benchmark_log.jsonl` (one JSON object per line).

---

## Adding a new experiment

1. Pick a flag name: `AIWF_<UPPERCASE>=1`
2. Add a row to the table at the top of this file
3. Add a section with Flag / Status / Expected gain / Risk / File
4. Implement behind the flag
5. Run benchmark and append entry to `docs/benchmark_log.jsonl`
6. Update Status from **Not started** → **Verified** (with date and GPU)
