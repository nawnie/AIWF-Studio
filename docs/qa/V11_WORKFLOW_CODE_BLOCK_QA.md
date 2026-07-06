# V11 Workflow Code Block QA Pass

Date: 2026-07-04

## Intent

This pass intentionally rolls Pipeline Atlas back from a visual graph/DAG editor into a linear workflow queue. The Studio shell still captures complex model-family settings, but the workflow view now treats each capture as a self-contained JSON code block that can be dragged and reordered.

## Product Decision

- Linear workflow order is acceptable.
- Visual wires are not required for this view.
- The primary Create panel now has a **Send to workflow** button beside **Generate**.
- Sending to workflow does not run a model.
- Each captured block embeds the current prompt, model, family/precision guess, dimensions, sampler, video settings, inpaint settings, runtime status, and Wan sidecar hook points when applicable.
- Adapter support remains a testing lane. The UI preserves adapter hook points without claiming every adapter path is proven.

## Files Changed

- `frontend/src/App.tsx`
- `frontend/src/styles.css`
- `frontend/src/layouts/paid/PaidLayoutTypes.ts`
- `frontend/src/layouts/paid/PipelineAtlasPaidLayout.tsx`
- `frontend/src/layouts/paid/paidApiClient.ts`
- `frontend/src/layouts/paid/paidLayouts.css`
- `frontend/src/layouts/paid/workflowBlocks.ts`
- `aiwf/web/paid_ext_api.py`
- `aiwf/web/paid_worker.py`
- `tests/individual_tests/test_paid_workflow_code_blocks.py`

## Backend Contract

Workflow payloads may now use:

```json
{
  "schema": "aiwf.workflow-code-blocks.v1",
  "blocks": [
    {
      "id": "workflow-block-...",
      "label": "Image · Flux",
      "nodeId": "generation-request",
      "order": 1,
      "payload": {},
      "code": "{...}"
    }
  ]
}
```

The validator checks block shape and JSON validity. It does not require graph edges for code-block workflows.

## Queue Behavior

The frontend queues these plans as `workflow-plan`, a validation-only job kind. The worker records a receipt and does not render media.

## Validation

- Python compile passed for changed backend files.
- `tests/individual_tests/test_paid_workflow_code_blocks.py`: 5 passed.
- `tests/individual_tests/test_model_family_support.py` + new workflow tests: 9 passed.
- Frontend `npm ci --ignore-scripts` passed.
- Frontend `npm run build` passed.

## Remaining QA Work

- Model adapter compatibility testing still needs real smoke runs per family/precision.
- The family matrix should continue receiving receipts as local models are tested.
- Old unused graph/wire CSS can be removed later after one more UI review if desired.
