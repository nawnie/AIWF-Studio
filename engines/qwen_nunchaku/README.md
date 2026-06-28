# Qwen Nunchaku sidecar

Isolated Qwen Image Lightning runtime for testing Nunchaku without changing
AIWF Studio's main `venv`.

Installed runtime:
- Python: `engines/qwen_nunchaku/.venv`
- Torch: `2.11.0+cu130`
- Nunchaku: `1.3.0.dev20260213+cu13.0torch2.11`
- Diffusers: `0.36.0`
- Transformers: `4.55.2`

Local model files:
- Base components: `models/qwen-image/Diffusers/Qwen-Image` when installed, falling back to `downloads/qwen_nunchaku/base`
- Transformer checkpoint: `models/qwen-image/Nunchaku/svdq-int4_r32-qwen-image-lightningv1.0-4steps.safetensors`
- Downloads mirror: `downloads/qwen_nunchaku/transformer/svdq-int4_r32-qwen-image-lightningv1.0-4steps.safetensors`

Storage policy:
- Keep the selectable Qwen checkpoint as one safetensors transformer file.
- Keep base components as a support folder because Qwen still needs tokenizer, text encoder, scheduler, and VAE assets.
- Do not keep both full base folders unless you need a download cache copy.

Smoke commands:

```powershell
engines\qwen_nunchaku\.venv\Scripts\python.exe engines\qwen_nunchaku\run_qwen_lightning.py --load-only
engines\qwen_nunchaku\.venv\Scripts\python.exe engines\qwen_nunchaku\run_qwen_lightning.py
```

The 16 GB RTX 4070 Ti SUPER path validated with `--blocks-on-gpu 4`, now the
runner default. Lower values may fail or exit natively in the Nunchaku extension.
