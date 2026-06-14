# Sprint Plan: AIWF Studio — "Settings Parity & Browser Foundation"

**Dates:** Jun 11 — Jun 25, 2026 (2 weeks) | **Team:** 1 maintainer (Nawnie) + AI agent sessions
**Sprint Goal:** A1111 users can configure saving/metadata like home, and the model browser foundation exists with safety rails — without breaking the working Studio.

## Capacity

| Person | Available | Allocation | Notes |
|--------|-----------|------------|-------|
| Nawnie | ~6 agent sessions | direction, live testing, commits | Session limits gate throughput; git runs locally only |
| AI agent | per session | implementation + verification | Sandbox: no torch/GPU, no live CivitAI; verified-write protocol for the flaky mount |
| **Total** | **~6 sessions** | planned to ~75% (4.5) | leave buffer for bug sessions like LoRA/Compel day |

## Sprint Backlog

| Priority | Item | Estimate | Source | Dependencies / Notes |
|----------|------|----------|--------|----------------------|
| P0 | Gallery hover actions (carryover from Phase 1) | 0.5 session | P1.8 | Needs live browser testing (Claude in Chrome + running app) |
| P0 | Saving & Output settings group (format/quality already exist; add grids, sidecar .txt, filename pattern, save-before-hires/restore, interrupted saves) | 1 session | P2.1 | Storage layer touch — regression-test saves |
| P0 | Metadata & PNG Info settings group (include model/VAE/LoRA hashes, app version, paste-behavior controls) | 0.5 session | P2.2 | Builds on existing infotext module |
| P1 | Live preview advanced (TAESD fast preview, refresh interval, title progress) | 0.5 session | P2.4 | Only expose what backend supports |
| P1 | API security settings (auth-with-listen, CORS origins, rate limit, URL blocking) | 0.5 session | P2.5 | Ship before advertising API compat |
| P1 | Model browser scaffolding: `civitai_browser.py` service + `domain/civitai.py` + Browse tab skeleton (Installed + HF first) | 1 session | P3.1/3.4 | CivitAI API unreachable from sandbox — service built with mocked tests, live-verified locally |
| P2 | Download safety: safetensors preference, .ckpt/.pt warnings, download receipts | 0.5 session | P5 | Prereq before browser goes wide |
| P2 | Gallery & viewer settings (lightbox, gallery height, send seed/size between tabs) | 0.5 session | P2.3 | |
| Stretch | CivitAI browser UI with safe-mode defaults + mature opt-in policy | 1 session | P3.2/3.3 | Only after P5 safety lands |
| Stretch | Popular-download presets (live queries, not hard-coded lists) | 0.5 session | P4 | Needs live API verification |

**Planned: ~4.5 sessions of ~6 (75%)** — stretch items absorb any surplus.

## Carryover (honest accounting)

Gallery hover actions deferred from Phase 1 deliberately: it requires DOM-overlay work inside gradio's gallery that couldn't be verified without a browser. Re-committed as P0 *with* the mitigation (live test via Claude in Chrome against the running app).

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Cloud-mount write corruption (recurred 4× this sprint) | Source files nulled on disk | Verified-write protocol (byte read-back); commit locally after every session; never trust unverified writes |
| CivitAI/HF API behavior unverifiable from sandbox | Browser features ship broken | Service layer with mocked tests; live verification step on Nawnie's machine before commit |
| Session limit mid-feature | Half-wired UI | Land work in compile-clean, test-passing increments (this sprint's pattern); commit per phase |
| gradio 6 quirks (dropdowns, tabs) | UI regressions | Keep AST wiring guard tests; hard-refresh discipline; stub UI build every change |
| Torch-dependent tests can't run in sandbox | Hidden regressions | Nawnie runs full `pytest tests -q` locally before each commit |

## Definition of Done (per item)

- [ ] `python -m compileall -q aiwf launch.py webui.py` clean
- [ ] Sandbox-runnable tests pass (133+); full suite run locally by Nawnie
- [ ] Stub UI build + settings handlers verified against gradio 6.17.3
- [ ] Synced to `AIWF_Studio` byte-identical; committed locally
- [ ] No model/runtime data folders touched

## Key Dates

| Date | Event |
|------|-------|
| Jun 11 | Sprint start (Phase 1 shipped) |
| ~Jun 18 | Mid-sprint: Phase 2 settings groups done, browser scaffolding started |
| Jun 25 | Sprint end: demo settings parity + Installed/HF browse tab |
| Jun 25 | Retro: did the mount hold? did safety land before browsing? |
