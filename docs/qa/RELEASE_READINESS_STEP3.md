# Step 3 Release Readiness Buckets

Date: 2026-06-29

Scope: no generation, no training, no server startup. This pass uses the QA matrix, live no-download readiness, test collection, and list/plan commands.

## Evidence Used

- `docs/qa/pipeline_feature_testing_matrix.csv`: 401 rows.
- `venv\Scripts\python.exe scripts\pipeline_readiness.py --json --no-downloads`: live preflight/readiness view.
- `venv\Scripts\python.exe -m pytest --collect-only -q tests`: 1220 tests collected, with the existing FastAPI `TestClient` warning.
- `venv\Scripts\python.exe scripts\smoke_backend.py --video --list`: video registry and VAE loader list check.
- `venv\Scripts\python.exe scripts\smoke_image_routes.py --plan-json`: bounded image route smoke plan.

## Current Counts

Saved QA matrix, including manual smoke rows:

| Status | Count | Release bucket |
| --- | ---: | --- |
| `working` | 22 | Ready, receipt-backed or manually verified |
| `metadata-only` | 282 | Wired or discovered, needs smoke receipt |
| `unsupported-no-route` | 91 | Scaffold-only or deferred |
| `broken-runtime` | 3 | Must stay blocked from normal selection |
| `blocked-cleanly` | 3 | Correctly blocked with a known reason |

Live no-download readiness, without manual matrix rows:

| Status | Count |
| --- | ---: |
| `working` | 13 |
| `metadata-only` | 286 |
| `unsupported-no-route` | 83 |
| `broken-runtime` | 3 |
| `blocked-cleanly` | 3 |

The matrix is the release ledger because it includes smoke receipts and manual verification rows. The live readiness command is the stricter current preflight view.

## Ready Bucket

These are the release-safe lanes right now:

- Core A1111-style image contracts: txt2img, img2img, inpaint, hires/refiner, ControlNet, XYZ plot, PNG/API replay, extras/enhance receipts, and segment-to-inpaint workflow receipts.
- Diffusers image route: `preflight:diffusers` and `registry:diffusers`, with receipt at `outputs\txt2img-images\20260625-213032.png`.
- Sana Sprint image: `registry:sana`, `asset:image:sana-sprint-0:models`, and `manual:sana-sprint-06b-image-smoke`.
- Sana Video 480p: `preflight:sana-video`, `registry:sana-video`, and `manual:sana-video-2b-480p-smoke`.
- LTX 2B Diffusers: `preflight:ltx-2b`, `registry:ltx-2b-diffusers`, and `route:ltx-0.9.5-diffusers-local-t5xxl`.
- LTX 2.3 one-stage HF Gemma: `registry:ltx-2.3`, `route:ltx-one-stage-hf-gemma`, and the converted Heretic Q3 Gemma manual rows.
- Wan I2V release surface: `preflight:wan-fast-5b`, `registry:wan-diffusers`, and `registry:wan-gguf`.

Keep this bucket small. A row only belongs here when it has a receipt, a passing bounded smoke, or a focused test proving the non-runtime contract.

## Wired, Needs Smoke

The largest release risk is not missing code. It is discovered assets that look selectable but have no bounded receipt.

| Family | Metadata-only count | Practical next check |
| --- | ---: | --- |
| Image | 241 | Run targeted image smoke routes one at a time; keep large Qwen Image opt-in. |
| Wan | 22 | Smoke the fast 5B route and any Q4/Q5 high-low pair before naming individual assets working. |
| LTX | 13 | Keep LTX 2.3/2B as ready; smoke other assets only after route-specific preflight passes. |
| LLM/VL | 6 | Keep metadata-only until the GGUF worker/API route exists. |

Default image smoke plan currently has 10 bounded routes:

- Flux Kontext FP4
- Flux NF4
- Flux GGUF dev
- Flux Fusion GGUF Q4
- Z-Image GGUF
- SDXL inpaint
- SD 1.5 inpaint
- Qwen Nunchaku 4-step
- Sana Sprint
- Flux.2 Klein 4B

Do not run the full image chain without approval. Run one route at a time and record the output path in the matrix.

## Scaffold-Only Or Deferred

The `unsupported-no-route` bucket is not trash. It means "do not sell this as working."

Largest groups:

- Wan adapters, Animate, T2V, Fun-Control, and route-mismatched files: 57 rows.
- LTX native GGUF or unsupported LTX routes: 26 rows.
- LLM/VL GGUF assets without a native worker route: 7 rows.
- Image Flux route mismatch: 1 row.

For release, these should stay visible only as future work, blocked assets, or advanced research notes.

## Blocked Cleanly

- ONNX image route: blocked because `models\onnx` does not exist.
- ONNX registry row: blocked for the same missing folder.
- Native LTX Heretic GGUF generation: blocked because the current GGUF path cannot return every Gemma hidden-state layer and attention mask needed by LTX.

These are acceptable blockers. Keep the reasons visible.

## Broken Runtime

These should not be normal selectable release assets:

- `4xBHI_dat2_multiblurjpg.safetensors`: auxiliary/upscale-looking asset misclassified as an image checkpoint; missing expected CLIP text model.
- `fluxedUpFluxNSFW_110FP8.safetensors`: Flux FP8 key schema mismatch.
- `fluxFusionV24StepsGGUFNF4_V2GGUFQ4KM.gguf`: Flux GGUF/NF4 mismatch against the current image route.

Next coding cleanup should make sure these are filtered, blocked, or labeled before the user can waste time on them.

## Test State

`pytest --collect-only -q tests` collected 1220 tests. The current test suite is broad enough to guard most wiring, but the release distinction is:

- Tests prove contracts, filters, preflight behavior, API compatibility, and receipt writing.
- Smoke receipts prove specific local model routes.
- Metadata-only rows are not release-ready, even when tests cover the surrounding code.

Verification run in this pass:

- `tests\individual_tests\test_pro_api.py`, `test_pipeline_readiness.py`, `test_pipeline_preflight.py`, `test_web_registry.py`: 55 passed, 1 existing FastAPI warning.
- `tests\individual_tests\test_api_parity.py`, `test_image_lab.py`, `test_image_workflow_service.py`, `test_smoke_image_routes.py`: 22 passed, 1 existing FastAPI warning.
- `tests\individual_tests\test_wan.py`, `test_wan_models.py`, `test_ltx.py`, `test_sana_video.py`: 119 passed.

Recommended focused gates before a release tag:

```powershell
venv\Scripts\python.exe -m pytest tests\individual_tests\test_pro_api.py tests\individual_tests\test_pipeline_readiness.py tests\individual_tests\test_pipeline_preflight.py tests\individual_tests\test_web_registry.py -q
venv\Scripts\python.exe -m pytest tests\individual_tests\test_api_parity.py tests\individual_tests\test_image_lab.py tests\individual_tests\test_image_workflow_service.py tests\individual_tests\test_smoke_image_routes.py -q
venv\Scripts\python.exe -m pytest tests\individual_tests\test_wan.py tests\individual_tests\test_wan_models.py tests\individual_tests\test_ltx.py tests\individual_tests\test_sana_video.py -q
```

## Next Step

Close the easy release blocker next: prevent the three `broken-runtime` assets from appearing as normal selectable working models, then add a regression test for that filter. After that, run targeted image smoke receipts only for the routes Shawn approves.
