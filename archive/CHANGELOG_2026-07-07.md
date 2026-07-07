# AIWF Studio Changelog - 2026-07-07

## stable-diffusion.cpp QA branch pass

### Added

- Shared backend profile launch system:
  - `diffusers`
  - `sdcpp`
  - `onnx`
- `AIWF Studio Pro.bat` now launches through `launch_backend_profile.py`.
- `launch_backend_profile.py` now checks and patches the Pro frontend before launch.
- `scripts/ensure_pro_frontend.py`
  - Adds `stable-diffusion.cpp` into the native React Settings backend dropdown.
  - Rebuilds the frontend when source files are newer than `frontend/dist/index.html`.
- `scripts/install_sdcpp.ps1`
  - Clones/builds `stable-diffusion.cpp`.
  - Supports CPU and CUDA build paths.
  - Can save the sd.cpp profile after build.
- `plugins/sdcpp_pipeline/`
  - sd.cpp profile UI.
  - Split-asset argument mapping for Flux/Qwen-style routes.
  - LoRA directory field.
  - ControlNet field.
  - Inpaint/img2img profile support.
  - Image and video smoke-test endpoints.
- Smoke-test scripts:
  - `scripts/sdcpp_smoke_test.py`
  - `scripts/sdcpp_video_smoke_test.py`
- QA docs:
  - `docs/SDCPP_PIPELINE_REQUIREMENTS_QA.md`
  - `docs/SDCPP_ALT_ENGINE.md`

### UI notes

New sd.cpp QA surfaces use a graphite/amber/cream palette. No new green-blue AI defaults were added.

### Current limitations

- sd.cpp video is wired as an experimental smoke-test lane first.
- Full split-asset parity depends on real local models and sd-cli behavior.
- Main AIWF Diffusers pipelines remain the fallback when sd.cpp lacks parity.

### Local QA order

1. Launch Diffusers and confirm existing behavior.
2. Launch sd.cpp profile and confirm backend class.
3. SD1.5 smoke.
4. SDXL smoke.
5. Inpaint smoke.
6. LoRA directory smoke.
7. ControlNet smoke.
8. Split-asset Flux/Qwen smoke.
9. Video smoke.
10. Compare output history and logs.
