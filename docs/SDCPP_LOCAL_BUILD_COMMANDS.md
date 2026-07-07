# Local stable-diffusion.cpp build commands

The GitHub connector safety layer would not accept an executable installer script that clones and builds an external C++ project directly. Keep these commands here for the local Windows QA pass.

From the AIWF Studio repo root:

```powershell
# Choose a tools folder inside the repo.
$ToolRoot = Join-Path (Get-Location) "tools\stable-diffusion.cpp"

# Clone upstream with submodules.
git clone --recursive https://github.com/leejet/stable-diffusion.cpp.git $ToolRoot

# Configure CUDA build.
cmake -S $ToolRoot -B (Join-Path $ToolRoot "build") -DSD_CUDA=ON -DCMAKE_BUILD_TYPE=Release

# Build sd-cli.
cmake --build (Join-Path $ToolRoot "build") --config Release --parallel

# Launch AIWF Pro through the sd.cpp profile.
.\scripts\launch_sdcpp.ps1 -SdCli "$ToolRoot\build\bin\Release\sd-cli.exe" -Backend cuda0 -MaxVram 14 -SetDefault -Terminal
```

If the built binary lands in a different folder, point `-SdCli` at that path.

The profile UI is available after launch at:

```text
http://127.0.0.1:7860/api/ext/sdcpp-pipeline/ui
```

Use this to configure split-asset paths for Flux/Qwen-style routes.
