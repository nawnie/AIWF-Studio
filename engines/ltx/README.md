# LTX 2.3 Video Engine

This is AIWF's optional Lightricks LTX 2.3 worker. It lives in its own venv
because LTX 2.3 uses a newer torch/CUDA stack and a separate upstream repo.

Install or repair the engine from Settings, or run:

```powershell
.\scripts\bootstrap_ltx.ps1 -Enable
```

Model files stay outside this repo under `models/ltx/`:

- `models/ltx/checkpoints/ltx-2.3-22b-distilled-1.1.safetensors`
- `models/ltx/upscalers/ltx-2.3-spatial-upscaler-x2-1.1.safetensors`
- `models/ltx/text_encoder/gemma-3-12b-it-qat-q4_0-unquantized/`

The worker reads one JSON request, calls the upstream `ltx_pipelines` CLI, and
emits JSONL status/artifact events back to AIWF.
