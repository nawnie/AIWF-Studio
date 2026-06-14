# Changelog

## 2026-06-12

### Wan I2V

- Made Wan 2.2 image-to-video require both a high-noise and a low-noise model (the two-stage transformer pair Wan 2.2 always needs, even with LoRAs). Removed the single-model load path entirely from the UI, service, and backend so it no longer falls through to downloading the default HF repo when models are already selected.
- Removed the "Model (single / fallback)" dropdown and the "Download base model" button; High noise + Low noise are now the required model selectors. The local `Wan2.2-TI2V-5B-Diffusers` folder is used only as the text-encoder / VAE / scheduler component provider, not as a generation model.
- Selecting fewer than both models now raises a clear "select both" error at every layer instead of silently triggering a download.

### UI Updates

- Added Studio size preset controls alongside the existing width and height sliders, with one-click preset sizes and aspect ratios tuned for SD 1.5 and SDXL workflows.
- Restyled the new size and ratio controls to better match the app's dark professional theme, including the later hotfix pass that tightened the outer tray spacing around each button.
- Reworked the Settings page structure so generation defaults, model paths, launch profile, access and security, and remote access each have clearer separation.
- Moved remote-session tooling into its own Settings tab and kept live session/network details grouped there instead of crowding the main workspace settings surface.

### Hotfixes

- Added launch-profile support for extra model-library roots and extra checkpoint roots so shared installs can be scanned without manually repointing every built-in folder.
- Added import buttons in Settings for existing AUTOMATIC1111 and ComfyUI installs. These imports merge shared model paths into AIWF's next-start launch profile instead of replacing the user's current entries.
- Expanded model discovery so imported libraries now cover more than checkpoints: LoRAs, VAEs, embeddings, ControlNet models, upscalers, and face-restoration models can all be discovered from shared external folders.
- Tightened tests around launch-path parsing, imported path merging, and shared-library scanning to keep the new settings and UI behavior from regressing.
