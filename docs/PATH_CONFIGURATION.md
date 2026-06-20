# Path Configuration

AIWF Studio should not require hard links, junctions, symlinks, or machine-local
source edits. Point the app at real folders through Settings, `launch.json`,
CLI flags, or environment variables.

## Recommended Setup

Use the UI first:

1. Open **Settings -> Model paths**.
2. Set **Models folder** only if AIWF should own a primary model tree.
3. Add existing A1111, ComfyUI, or shared model roots under **Extra model
   library folders**.
4. Add checkpoint-only folders under **Extra checkpoint folders**.
5. Save the launch profile and restart AIWF.

Use **Settings -> Engines & pipelines -> External tool paths** for optional SDK
and executable paths such as NVIDIA VideoFX sample apps.

## Launch Profile Fields

`launch.json` is written beside the normal local config and is read on app
startup. Important path fields:

- `models_dir`: primary model root.
- `ckpt_dir`: primary checkpoint root.
- `output_dir`: generated output root.
- `extra_model_dirs`: newline-delimited shared model roots.
- `extra_ckpt_dirs`: newline-delimited checkpoint-only roots.
- `nvidia_vfx_sdk_root`: optional NVIDIA Video Effects / VFX SDK root.
- `vsr_video_effects_app`: optional `VideoEffectsApp.exe`.
- `vsr_upscale_app`: optional `UpscalePipelineApp.exe`.
- `videofx_denoise_app`: optional `DenoiseEffectApp.exe`.
- `videofx_aigs_app`: optional `AigsEffectApp.exe`.
- `videofx_relight_app`: optional `RelightingEffectApp.exe`.
- `vsr_model_dir`: optional NVIDIA VideoFX model package directory.

Blank fields use repo-local defaults and auto-detection.

## CLI Equivalents

Every path can be supplied at launch:

```powershell
python launch.py --models-dir D:\AI\Models --extra-model-dir D:\Shared\ComfyUI\models
python launch.py --nvidia-vfx-sdk-root "C:\Program Files\NVIDIA Corporation\NVIDIA VFX SDK"
python launch.py --vsr-video-effects-app D:\SDKs\NVIDIA\VideoEffectsApp.exe
```

## Environment Fallbacks

External SDK/app paths can also come from environment variables:

- `AIWF_NVIDIA_VFX_SDK_ROOT`
- `AIWF_VSR_VIDEO_EFFECTS_APP`
- `AIWF_VSR_UPSCALE_APP`
- `AIWF_VIDEOFX_DENOISE_APP`
- `AIWF_VIDEOFX_AIGS_APP`
- `AIWF_VIDEOFX_RELIGHT_APP`
- `AIWF_VSR_MODEL_DIR`

The saved launch profile is easier for normal users; env vars are best for
portable scripts and developer machines.

## Link Policy

Do not require users to create hard links or junctions for normal operation.
They are fragile on Windows, hard to explain, and easy to break when drives move.

Acceptable alternatives:

- Add extra scan roots in Settings.
- Use CLI flags for portable launch scripts.
- Use environment variables for SDK/app paths.
- Keep optional engines under `engines/` or configure them through
  `engines.json`.

If a future integration appears to require a link, treat that as a bug or an
installer-design problem first.
