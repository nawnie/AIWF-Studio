# V10 model family matrix merge validation

## Merge scope

- Base archive: `AIWF-Studio-main (1).zip`
- Applied prior visual/product patch: v9 paid visual polish overlay
- Added model family support matrix, precision/quant detector, live API endpoint, Families rail UI, docs, and unit coverage.

## Source audit

- Text/source files scanned: 546
- Text/source lines scanned: 142,835
- Exclusions: `node_modules`, `frontend/dist`, `__pycache__`, `.pytest_cache`, and binary/model artifacts.
- The matrix is code-indexed; it is not derived from README marketing copy.

## Validation commands run

```powershell
python -m py_compile aiwf/services/model_family_support.py aiwf/services/pipeline_readiness.py aiwf/web/paid_ext_api.py
python -m compileall -q aiwf tests/individual_tests/test_model_family_support.py
pytest -q tests/individual_tests/test_model_family_support.py
cd frontend; npm ci --ignore-scripts; npm run build
```

## Results

- Python targeted compile: pass
- Python package compileall: pass
- Model family unit tests: 4 passed
- Frontend dependency install: pass, 202 packages, 0 vulnerabilities reported by npm
- Frontend TypeScript/Vite build: pass
- Model family endpoint smoke: returned schema `aiwf.model-family-support.v1`, 13 families, and 25 precision vocabulary entries.

## Broader test caveat

A broader focused pytest attempt collected and ran most targeted tests in this Linux sandbox but did not count as a release gate here because optional runtime dependencies and Windows-specific executable behavior are absent in the container:

- `tests/individual_tests/test_web_registry.py` collection requires `diffusers`, which is not installed in this sandbox.
- `test_qwen_nunchaku_preflight_checks_engine_and_assets` expects a Windows-style `python.exe` path to behave as executable.
- `test_sana_video_preflight_reports_runtime_and_default_model_path` expects the optional Sana Video runtime/Diffusers components to be importable.

The new model family patch itself is covered by the passing unit tests, py_compile, compileall, endpoint smoke, and full frontend build.

## Packaging checks

Packaging checks were run after cache cleanup and before delivery:

- `MERGED_MANIFEST_SHA256.json` created for packaged payload files.
- Manifest self-check before ZIP: pass.
- `zip -T`: pass.
- `unzip -t`: pass.
- Re-extracted manifest verification: pass.
- Confirmed `node_modules`, `__pycache__`, and `.pyc` files are not packaged.

Archive SHA-256 is recorded in the final response.
