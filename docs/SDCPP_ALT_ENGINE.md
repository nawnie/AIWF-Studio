# AIWF Studio stable-diffusion.cpp alternate engine

This branch keeps the AIWF Studio production UI and swaps the image-generation backend to a `stable-diffusion.cpp` CLI bridge.

## Launch profiles

The profile launcher keeps one AIWF venv and chooses the backend at Pro boot time:

```text
AIWF Pro UI -> Pro API -> GenerationService -> selected backend
                                     |-> Diffusers
                                     |-> stable-diffusion.cpp sd-cli
                                     |-> ONNX
```

```powershell
# Launch with saved profile default.
.\AIWF Studio Pro.bat --terminal

# Launch a specific profile.
.\scripts\launch_backend_profile.ps1 -Profile diffusers -Terminal
.\scripts\launch_backend_profile.ps1 -Profile sdcpp -SdCli "F:\tools\stable-diffusion.cpp\bin\sd-cli.exe" -SdcppBackend cuda0 -MaxVram 14 -Terminal
.\scripts\launch_backend_profile.ps1 -Profile onnx -Terminal
```

The launcher runs `scripts/ensure_pro_frontend.py` before Pro starts. If `frontend/src/App.tsx` does not expose `stable-diffusion.cpp` in the Settings backend dropdown, the patch is applied and the Pro frontend rebuilds.

## Build/install sd-cli

```powershell
.\scripts\install_sdcpp.ps1 -Mode build -Backend cuda -SetProfile
```

That clones `stable-diffusion.cpp` into `tools/stable-diffusion.cpp`, builds `sd-cli`, and saves the `sdcpp` backend profile.

## QA UI

After Pro starts:

```text
http://127.0.0.1:7860/api/ext/sdcpp-pipeline/ui
```

Use it to set sd.cpp component paths, LoRA directory, ControlNet path, max VRAM, offload mode, and smoke tests.

## Helper endpoints

```text
GET  /api/ext/sdcpp-pipeline/status
GET  /api/ext/sdcpp-pipeline/profile
POST /api/ext/sdcpp-pipeline/profile
GET  /api/ext/sdcpp-pipeline/requirements
POST /api/ext/sdcpp-pipeline/smoke/image
POST /api/ext/sdcpp-pipeline/smoke/video
```

## CLI smoke tests

```powershell
python .\scripts\sdcpp_smoke_test.py --model "F:\models\sd15.safetensors" --name sd15-512

python .\scripts\sdcpp_smoke_test.py --model "F:\models\sdxl.safetensors" --name sdxl-1024 --width 1024 --height 1024 --steps 6

python .\scripts\sdcpp_video_smoke_test.py --model "F:\models\video-model.safetensors" --frames 25 --fps 16
```

## Current limitation

The sd.cpp route is a QA-ready CLI bridge. The main AIWF pipelines remain the fallback when a model family, split-asset graph, or video route needs features that the sd.cpp CLI bridge has not proven yet.
