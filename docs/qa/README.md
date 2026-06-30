# QA Pipeline Matrix

`pipeline_feature_testing_matrix.csv` is an Excel-compatible CSV for route and model QA.

Current release bucket report:

- `RELEASE_READINESS_STEP3.md`

Source:

- `_local/logs/pipeline_readiness_with_downloads_latest.json`
- Known smoke receipts from the current LTX/Wan/readiness pass

Important columns:

- `current_status` / `current_result`: current readiness result.
- `blocking_path` / `blocking_reason`: populated when a route or asset is blocked, broken, or unsupported.
- `smoke_*`: bounded smoke-test evidence when available; `blocked-cleanly` means the probe ran and found a known unsupported route.
- LTX script smokes can run without an interactive pause by setting `AIWF_NO_PAUSE=1`.
- `full_sane_*`: QA-script fields for future full runs; blank unless a run already has evidence.
- `full_sane_video_seconds`: target `5` for video routes.
- `full_sane_image_passes`: target `2` for image routes; record warmed speed from the second image.
- `supports_*`: coarse feature flags inferred from family, route, path, and asset type.

Regenerate:

```powershell
venv\Scripts\python.exe scripts\export_pipeline_qa_matrix.py --readiness _local\logs\pipeline_readiness_with_downloads_latest.json --output docs\qa\pipeline_feature_testing_matrix.csv
```

LTX smoke example:

```powershell
cmd /c "set AIWF_NO_PAUSE=1&& scripts\run_ltx_smoketest.bat"
```

Smallest Heretic Q3 conversion:

```powershell
venv\Scripts\python.exe scripts\convert_gemma_gguf_to_hf.py --dry-run
venv\Scripts\python.exe scripts\convert_gemma_gguf_to_hf.py --overwrite
```

The working converted Gemma root is `models\ltx\text_encoder\gemma-3-12b-heretic-q3km-converted`.
It is not memory-saving at runtime: the Q3 GGUF is dequantized into BF16 safetensors, with an official Gemma vision/projector sidecar for LTX loader compatibility.

Heretic GGUF contract probe:

```powershell
venv\Scripts\python.exe scripts\probe_ltx_runtime.py --gguf --json --allow-blocked
```

Gemma GGUF metadata inventory, no tensor dequant:

```powershell
venv\Scripts\python.exe scripts\probe_ltx_runtime.py --gguf-inventory --json
```

Model Manager follow-up:

- Add tags for GGUF LLM assets that separate llama.cpp serving, TensorRT-LLM export/serving, bitsandbytes safetensors, and metadata-only GGUF inspection.
- TensorRT-LLM and llama.cpp do not satisfy the current LTX hidden-state contract by themselves. Keep them outside the LTX HF-safetensors worker until a dedicated adapter exists.
