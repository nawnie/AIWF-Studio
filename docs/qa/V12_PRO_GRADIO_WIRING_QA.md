# V12 Pro + Gradio Wiring QA Pass

Date: 2026-07-04

Scope: wiring only. This pass intentionally does not certify model execution, adapter quality, or family precision runtime behavior. It makes the Pro shell, Gradio Studio, Gradio Wan, and Pipeline Atlas share one settings capture contract so later model-family testing has less friction.

## Shared Contract

New shared packet builder:

- `aiwf/services/studio_generation_packet.py`
- Packet schema: `aiwf.studio-generation-packet.v1`
- Workflow document schema: `aiwf.workflow-code-blocks.v1`

The packet captures:

- mode, route, model family, precision guess
- model selection gate
- prompt, seed, dimensions, sampler, scheduler, steps, CFG
- image and inpaint settings
- video settings
- Wan model-pack sidecars
- Wan LoRA stack sidecars
- Wan offload plan sidecars

## Pro Fixes

- `ProGeneratePayload` now accepts full Wan routing fields:
  - runtime mode
  - high/low model IDs
  - high/low steps
  - boundary ratio
  - high/low LoRA IDs and scales
  - VAE
  - text encoder
  - offload
  - sampler / sigma / flow shift
- `_wan_video_request_from_payload()` forwards those fields into `WanI2VRequest`.
- Legacy `high_low` runtime values are accepted and canonicalized to `native_high_low` where generation needs the current Wan constants.
- Pro generation now blocks normal generation for blocked/broken/no-route model statuses before calling the backend.
- Paid surfaces now expose `Send to workflow` actions for QA capture instead of pretending every page has a runnable pipeline.

## Gradio Fixes

- Studio tab now has `Send to workflow` beside `Generate`.
- Wan I2V tab now has `Send to workflow` beside `Generate video`.
- Both output movable JSON workflow documents without running generation.
- Gradio Wan capture preserves high/low model pair, VAE, UMT5/text encoder, stage LoRAs, offload, sampler, sigma, flow shift, temporal chunk settings, RIFE, VSR, and audio post-chain intent.

## Validation Commands Used

```powershell
python -m py_compile aiwf/services/studio_generation_packet.py aiwf/web/pro_api.py aiwf/web/paid_ext_api.py aiwf/web/studio/tab.py aiwf/web/tabs/wan_i2v.py tests/individual_tests/test_studio_generation_packet.py
python -m pytest tests/individual_tests/test_studio_generation_packet.py tests/individual_tests/test_model_family_support.py tests/individual_tests/test_paid_workflow_code_blocks.py tests/individual_tests/test_pro_api.py -q
cd frontend
npm ci --ignore-scripts
npm run build
```

## Result

- Targeted Python tests: 62 passed.
- Frontend TypeScript + Vite build: passed.
- No model execution certification is implied by this pass.
