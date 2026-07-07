# AIWF Studio Changelog - 2026-07-07

## Backend profile lane

Added a shared-venv backend profile system for AIWF Studio Pro so the production UI can launch against different image backends without maintaining separate Python environments.

### Added

- `launch_backend_profile.py`
  - Launches Pro with a selected backend profile.
  - Supports `diffusers`, `sdcpp`, and `onnx`.
  - Saves `_local/backend_profile.json` when launched with `--set-default`.
- `webui_backend_profile.py`
  - Wrapper that strips profile-only args before the normal Pro parser runs.
  - Forces the selected backend into `RuntimeFlags` before `build_context()` creates the engine.
- `scripts/launch_backend_profile.ps1`
  - Windows helper for profile launches.
  - Supports stable-diffusion.cpp knobs like sd-cli path, CUDA backend, max VRAM, CPU offload, streamed layers, VAE tiling, and diffusion flash attention.
- `scripts/launch_sdcpp.ps1`
  - Dedicated stable-diffusion.cpp launch helper routed through the profile launcher.
- `plugins/backend_switch/`
  - Small Pro extension with `/api/ext/backend-switch/status` and `/api/ext/backend-switch/ui`.
  - Lets a running Pro session save the preferred backend profile default.
- `aiwf/infrastructure/sdcpp/`
  - First stable-diffusion.cpp backend adapter.
  - Calls `sd-cli` as a subprocess while preserving the AIWF UI, queue, model scan, output handling, metadata, and history.

### Changed

- `AIWF Studio Pro.bat`
  - Now launches through `launch_backend_profile.py` instead of directly launching `launch_pro.py`.
  - Default behavior remains Diffusers unless `_local/backend_profile.json` selects another backend.
  - This means shortcuts created by the initial installer are now backend-profile-aware without requiring a new desktop shortcut target.
- `aiwf/bootstrap.py`
  - Added backend selection for `sdcpp` / `stable-diffusion.cpp` / `stable_diffusion_cpp`.
- `docs/SDCPP_ALT_ENGINE.md`
  - Expanded with profile launcher usage, sd.cpp launch examples, environment knobs, architecture notes, and test checklist.

### Current limitations

- The React settings dropdown still only exposes the legacy backend options in the main settings panel.
- The backend-switch extension page saves the profile default but does not hot-swap the running engine. Restart is required because the backend object is created at Pro startup.
- stable-diffusion.cpp support is first-pass image generation through `sd-cli`; split-asset Flux/Qwen/Wan/LTX mapping still needs deeper wiring.
- No actual sd-cli runtime smoke test was possible in this environment.

### Test priorities for local machine

1. Launch Diffusers profile and confirm current behavior is unchanged.
2. Launch stable-diffusion.cpp profile with a known SD1.5 `.safetensors` file.
3. Test SDXL single-file model.
4. Test cancel mid-generation.
5. Test preview updates.
6. Confirm recent output history receives sd.cpp outputs.
7. Use `/api/ext/backend-switch/ui` to save backend profile defaults.
8. Relaunch with `AIWF Studio Pro.bat` and confirm it uses the saved profile.

Logs first. Panic later.
