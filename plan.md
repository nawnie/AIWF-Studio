# Plan

## Objective
- Fix the startup background checkpoint preload crash:
  `AttributeError: _lock` from `tqdm.contrib.concurrent.ensure_lock` during Diffusers single-file config download.
- Add a dev-trace logger for app version and model throughput, and document the rule in `AGENTS.md`.

## Request Interpretation
- User asked why the prior rumination missed the startup error, asked to fix it, and explicitly allowed app launches only if visible.
- Operational target: identify why background checkpoint preload triggers Hugging Face snapshot/tqdm `_lock`, patch the preload/load path so startup does not crash, and verify without hidden launches.
- Success: root cause explained, targeted tests pass, and any app launch uses a visible terminal/process.

## Constraints
- Preserve current architecture and avoid unrelated refactors.
- Protect user changes; make only targeted fixes if evidence shows a gap.
- Validate with focused tests and full suite when feasible.
- Do not launch hidden app windows; any launch must be visible.

## Lanes
### Lane: Wan Chunk Settings Audit
Status: done
Goal: Trace the new temporal chunk settings from UI/request through backend load/materialization.

#### Card: WCS-1
Type: check
Goal: Verify UI and request model carry `chunk_size` / `chunk_overlap`.
Depends on: `aiwf/web/tabs/wan_i2v.py`, `aiwf/core/domain/wan.py`
Evidence: `aiwf/web/tabs/wan_i2v.py:369-420`, `aiwf/core/domain/wan.py:62-65`
Failure mode: UI values exist but request/domain drops or renames them.
Success check: Fields are defined, normalized, and passed into `WanI2VRequest`.
Verification state: verified
Next if pass: WCS-2
Next if fail: Patch request wiring and add tests.

#### Card: WCS-2
Type: check
Goal: Verify service/backend call preserves settings into `WanI2VBackend.generate`.
Depends on: `aiwf/services/wan.py`, `aiwf/infrastructure/wan/pipeline.py`
Evidence: `aiwf/services/wan.py:837-847`, `aiwf/infrastructure/wan/pipeline.py:2115-2135`
Failure mode: service omits values or backend uses defaults too early.
Success check: Request attributes are read and passed into `_ensure`.
Verification state: verified
Next if pass: WCS-3
Next if fail: Patch service/backend handoff and add tests.

#### Card: WCS-3
Type: check
Goal: Verify `_ensure`, `_load_dual_pipeline`, and all transformer materialization paths receive settings.
Depends on: `aiwf/infrastructure/wan/pipeline.py`
Evidence: `aiwf/infrastructure/wan/pipeline.py:1516-1547`, `:1556-1564`, `:1982-2008`
Failure mode: high path, synchronous low path, or background low path loses values.
Success check: Values reach `_apply_wan_attention_optimizations` in every branch without undefined locals.
Verification state: verified
Next if pass: WCS-4
Next if fail: Patch missing branch and add regression tests.

#### Card: WCS-4
Type: check
Goal: Verify cache key/reuse behavior accounts for changed chunk settings.
Depends on: `aiwf/infrastructure/wan/pipeline.py`
Evidence: `aiwf/infrastructure/wan/pipeline.py:1688-1704` now includes normalized chunk settings in `_ensure` cache key; `aiwf/infrastructure/wan/sliced_sampler.py:246-247` confirms already wrapped transformers do not update in place.
Failure mode: changing sliders reuses a pipeline with stale installed temporal chunk forward settings.
Success check: Cache key includes chunk settings or reused pipeline reapplies new settings reliably.
Verification state: verified
Next if pass: WCS-5
Next if fail: Patch cache key/reapply logic and add regression tests.

#### Card: WCS-5
Type: action
Goal: Run targeted and full tests after any fix.
Depends on: WCS-1 through WCS-4
Evidence: Focused regressions passed; Wan suite passed; full suite passed (`308 passed, 1 skipped`).
Failure mode: hidden regression outside the audited branch.
Success check: Focused Wan tests and full suite pass.
Verification state: verified
Next if pass: final summary
Next if fail: inspect failures and revise.

### Lane: Startup Checkpoint Preload Crash
Status: active
Goal: Prevent background checkpoint preload from crashing startup via Diffusers/Hugging Face/tqdm lock cleanup.

#### Card: SCP-1
Type: diagnostic_branch
Goal: Locate the startup preload path and single-file load arguments.
Depends on: `aiwf/web/app.py`, `aiwf/infrastructure/diffusers/backend.py`
Evidence: pending
Failure mode: patching the wrong startup path.
Success check: Trace from `_preload_default_checkpoint` to backend `from_single_file` call.
Verification state: unverified
Next if pass: SCP-2
Next if fail: inspect launch/bootstrap code.

#### Card: SCP-2
Type: diagnostic_branch
Goal: Explain why Diffusers attempts a Hugging Face snapshot download during local single-file preload.
Depends on: backend load kwargs, checkpoint metadata/config path.
Evidence: pending
Failure mode: treating a tqdm symptom while leaving unwanted network/config download behavior.
Success check: Identify load arg or config source that avoids snapshot download for known local SD/SDXL files.
Verification state: unverified
Next if pass: SCP-3
Next if fail: add guarded preload failure handling as fallback.

#### Card: SCP-3
Type: action
Goal: Patch the smallest safe path and add regression coverage.
Depends on: SCP-1, SCP-2
Evidence: pending
Failure mode: background preload still crashes or normal checkpoint loads regress.
Success check: Unit test verifies local single-file load receives a local config/or avoids hub snapshot path.
Verification state: unverified
Next if pass: SCP-4
Next if fail: revise patch.

#### Card: SCP-4
Type: action
Goal: Verify with tests and visible app launch if needed.
Depends on: SCP-3
Evidence: pending
Failure mode: test-only fix misses actual startup behavior.
Success check: Tests pass; visible launch does not emit the preload `_lock` traceback.
Verification state: unverified
Next if pass: final summary
Next if fail: inspect new log/terminal output.

## Active Route
- SCP-1 -> SCP-2 -> SCP-3 -> SCP-4

## Open Unknowns
- Whether backend currently supplies `config=` to `from_single_file`.
- Whether Diffusers can use a local config path from the checkpoint catalog/profile for the preloaded checkpoint.
- Whether the tqdm `_lock` crash is only from unwanted hub download or also a package compatibility issue.

## Verification Passes
- Pass 1: Function/caller scan verified `_materialize_wan_transformer` accepts `chunk_size` / `chunk_overlap`, passes them to `_apply_wan_attention_optimizations`, and background preload reads them from `_low_preload_spec`.
- Pass 2: UI/domain/service scan verified slider input order, `WanI2VRequest` fields, and `model_copy()` preservation of chunk settings while resolving model paths.
- Pass 3: Static/runtime sanity verified relevant files compile and `WanI2VRequest.model_copy()` preserves `chunk_size=20` / `chunk_overlap=6`.
- Pass 4: Full test suite passed.

## Validation Log
- 2026-06-13: Plan created for user-requested double/triple-check of Wan chunk settings.
- 2026-06-13: Verified UI/domain/service/materialization paths; found cache-key stale-settings gap.
- 2026-06-13: Patched `_ensure` cache key for chunk settings; focused worker/cache regressions passed; Wan tests passed (`38 passed, 1 skipped`).
- 2026-06-13: Full suite passed (`308 passed, 1 skipped`, 3 warnings).
- 2026-06-13: Rumination re-audit completed: targeted regressions passed, compile checks passed, full suite passed again (`308 passed, 1 skipped`, 3 warnings).
- 2026-06-13: New startup preload crash reported; active route switched from Wan chunk audit to startup checkpoint preload crash.

### Lane: Dev Trace Version Metrics
Status: active
Goal: Record app version plus model throughput in structured dev traces for later speed comparisons.

#### Card: DTM-1
Type: check
Goal: Verify the existing dev diagnostics surface is the right place for version and throughput logging.
Depends on: `aiwf/dev/diagnostics.py`, `aiwf/services/generation.py`, `aiwf/services/wan.py`, `aiwf/web/app.py`
Evidence: current trace helpers and event subscriptions in `aiwf/dev/diagnostics.py`; generation and Wan services already compute elapsed time.
Failure mode: adding a parallel logger that bypasses structured dev traces.
Success check: one helper emits structured fields for app version and throughput, and call sites use it.
Verification state: unverified
Next if pass: DTM-2
Next if fail: inspect the diagnostics API and narrow the helper.

#### Card: DTM-2
Type: action
Goal: Patch the helper and call sites, then update `AGENTS.md`.
Depends on: DTM-1
Evidence: pending
Failure mode: version is logged but throughput is missing, or vice versa.
Success check: startup and completed generation/video runs write structured version/rate traces.
Verification state: unverified
Next if pass: DTM-3
Next if fail: revise fields and integration points.

#### Card: DTM-3
Type: action
Goal: Add regression coverage for the new trace helper and version field.
Depends on: DTM-2
Evidence: pending
Failure mode: logging changes silently drift or break when version fields are absent.
Success check: focused tests pass for the new helper and the modified call sites.
Verification state: unverified
Next if pass: DTM-4
Next if fail: adjust tests or helper shape.

#### Card: DTM-4
Type: action
Goal: Confirm the plan is ready for the larger incoming notes dump.
Depends on: DTM-3
Evidence: pending
Failure mode: plan loses track of the new logging work.
Success check: `plan.md` records the new metric rule and can absorb the user's larger plan text.
Verification state: unverified
Next if pass: final summary
Next if fail: update the plan map again.

## Active Route
- DTM-1 -> DTM-2 -> DTM-3 -> DTM-4
