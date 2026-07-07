# AIWF Studio stable-diffusion.cpp alternate engine

This branch keeps the AIWF Studio production UI and swaps the image-generation backend to a `stable-diffusion.cpp` CLI bridge.

The first pass is intentionally boring in the good way: UI, queue, settings, model scan, output saving, history, and logs stay in AIWF. The actual inference call is delegated to `sd-cli` from `stable-diffusion.cpp`.

Tiny robot note: this is not the final C++ engine marriage ceremony. It is the first clean wire through the wall.

## Current scope

Implemented:

- `AIWF_INFERENCE_BACKEND=sdcpp` route in `aiwf/bootstrap.py`
- `aiwf.infrastructure.sdcpp.StableDiffusionCppBackend`
- production UI compatibility through the existing `InferenceBackend` protocol
- model scanning through the existing AIWF model inventory
- txt2img command generation
- basic img2img and inpaint input handoff
- preview polling through `sd-cli --preview`
- cancellation by terminating the `sd-cli` subprocess
- output import back into AIWF's normal save/metadata flow
- Windows PowerShell launch helper at `scripts/launch_sdcpp.ps1`

Not done yet:

- embedded C API binding
- stable-diffusion.cpp install/download automation
- advanced LoRA argument translation
- full Flux/Qwen/Wan split-asset path mapping
- deep sampler parity matrix
- Pro UI settings surface for every `AIWF_SDCPP_*` option

## How to launch

Install or build `stable-diffusion.cpp` separately, then point AIWF at `sd-cli.exe`.

```powershell
# From the AIWF-Studio repo root
.\scripts\launch_sdcpp.ps1 -SdCli "F:\tools\stable-diffusion.cpp\bin\sd-cli.exe" -Backend cuda0 -MaxVram 14
```

Useful low-VRAM lane:

```powershell
.\scripts\launch_sdcpp.ps1 `
  -SdCli "F:\tools\stable-diffusion.cpp\bin\sd-cli.exe" `
  -Backend cuda0 `
  -MaxVram 14 `
  -OffloadToCpu `
  -StreamLayers `
  -VaeTiling
```

Manual environment version:

```powershell
$env:AIWF_INFERENCE_BACKEND = "sdcpp"
$env:AIWF_SDCPP_BINARY = "F:\tools\stable-diffusion.cpp\bin\sd-cli.exe"
$env:AIWF_SDCPP_BACKEND = "cuda0"
$env:AIWF_SDCPP_MAX_VRAM = "14"
$env:AIWF_SDCPP_DIFFUSION_FA = "1"
python launch.py --skip-install
```

## Environment knobs

| Variable | Default | Purpose |
|---|---:|---|
| `AIWF_INFERENCE_BACKEND` | `diffusers` | Set to `sdcpp` to activate this backend. |
| `AIWF_SDCPP_BINARY` | auto-detect | Full path to `sd-cli.exe` or `sd-cli`. |
| `AIWF_SDCPP_BACKEND` | `cuda0` | Passed to `sd-cli --backend`. Use `cpu`, `cuda0`, `vulkan0`, etc. |
| `AIWF_SDCPP_PARAMS_BACKEND` | blank | Passed to `sd-cli --params-backend`. |
| `AIWF_SDCPP_MAX_VRAM` | `0` | Passed to `sd-cli --max-vram`. Use a GiB value like `14`. |
| `AIWF_SDCPP_OFFLOAD_TO_CPU` | `0` | Adds `--offload-to-cpu`. |
| `AIWF_SDCPP_STREAM_LAYERS` | `0` | Adds `--stream-layers`. Useful with max VRAM. |
| `AIWF_SDCPP_DIFFUSION_FA` | `1` | Adds `--diffusion-fa` unless set to `0`. |
| `AIWF_SDCPP_VAE_TILING` | `0` | Adds `--vae-tiling`. |
| `AIWF_SDCPP_MMAP` | `1` | Adds `--mmap` unless set to `0`. |
| `AIWF_SDCPP_PREVIEW` | `vae` | Preview decoder mode sent to `--preview`. |
| `AIWF_SDCPP_EXTRA_ARGS` | blank | Extra raw arguments appended to `sd-cli`. |

## Architecture

AIWF already has a backend boundary:

```text
Production UI -> Pro API -> GenerationService -> InferenceBackend
```

This branch adds:

```text
InferenceBackend -> StableDiffusionCppBackend -> sd-cli subprocess
```

That gives us a safe test lane before deeper C++ integration. If it works well, the second pass can replace the subprocess bridge with a Python binding or direct C API bridge.

## Test checklist

Start with boring SD1.5 or SDXL single-file models first.

1. Launch with `scripts/launch_sdcpp.ps1`.
2. Confirm the model dropdown still scans existing model folders.
3. Generate 512x512 SD1.5 txt2img.
4. Generate 1024x1024 SDXL txt2img.
5. Test cancel mid-run.
6. Test preview updates.
7. Test img2img handoff.
8. Test inpaint handoff.
9. Compare output save/history metadata against Diffusers backend.
10. Only then move to Flux/Qwen/Z-Image split-asset routes.

Logs first. The gremlins can have snacks later.
