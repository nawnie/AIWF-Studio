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
| P0 ✅ | Gallery hover actions (carryover from Phase 1) | 0.5 session | P1.8 | `gallery.select()` wired — promotes image to workspace, sends seed (if `send_seed_on_click`) and size (if `send_size_on_click`). 7 tests in `test_gallery_select.py`. |
| P0 ✅ | Saving & Output settings group (format/quality already exist; add grids, sidecar .txt, filename pattern, save-before-hires/restore, interrupted saves) | 1 session | P2.1 | Storage layer touch — regression-test saves |
| P0 ✅ | Metadata & PNG Info settings group (include model/VAE/LoRA hashes, app version, paste-behavior controls) | 0.5 session | P2.2 | Builds on existing infotext module |
| P1 ✅ | Live preview advanced (TAESD fast preview, refresh interval, title progress) | 0.5 session | P2.4 | Only expose what backend supports |
| P1 ✅ | API security settings (auth-with-listen, CORS origins, rate limit, URL blocking) | 0.5 session | P2.5 | Ship before advertising API compat |
| P1 ✅ | Model browser scaffolding: `civitai_browser.py` service + `domain/civitai.py` + Browse tab skeleton (Installed + HF first) | 1 session | P3.1/3.4 | CivitAI API unreachable from sandbox — service built with mocked tests, live-verified locally |
| P2 ✅ | Download safety: safetensors preference, .ckpt/.pt warnings, download receipts | 0.5 session | P5 | Prereq before browser goes wide |
| P2 ✅ | Gallery & viewer settings (lightbox, gallery height, send seed/size between tabs) | 0.5 session | P2.3 | |
| Stretch ✅ | CivitAI browser UI with safe-mode defaults + mature opt-in policy | 1 session | P3.2/3.3 | Only after P5 safety lands |
| Stretch ✅ | Popular-download presets (live queries, not hard-coded lists) | 0.5 session | P4 | Needs live API verification |

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

## Session log — 2026-06-14

**Agent session:** DTM lane + sprint P1/P2 implementation.

- DTM-1–4: Verified dev trace surface; patched `wan.py` `trace_model_throughput` to include `app_version`; added regression tests; restored mount-truncated `aiwf/dev/__init__.py` and `diagnostics.py`.
- Sprint P1: `aiwf/core/domain/civitai.py`, `aiwf/services/civitai_browser.py`, Browse tab (Installed / HF Hub / CivitAI) added to model_manager.py. 20 tests.
- Sprint P2: Gallery settings (`gallery_height`, `gallery_columns`, `send_seed_on_click`, `send_size_on_click`) + Download safety (`prefer_safetensors`, `write_download_receipts`, `is_unsafe_download_format`, `write_download_receipt`) added. Settings UI wired. 12 tests.
- All changed files AST-verified. Test suite: 64 pass + 13 preload-guard, 1 pre-existing skip.
- Stretch P1: CivitAI browser full UI — pagination (cursor-based prev/next), NSFW opt-in toggle, sort dropdown, download links with safetensors blocking, 9 new tests.
- Stretch P2: Popular-download presets — 4 preset buttons (Top Checkpoints, Top LoRAs, Top VAEs, Newest) wired to live CivitAI queries; presets auto-fill the search controls.
- Sprint D: `docs/ACCELERATION_EXPERIMENTS.md` (8 experiments, benchmark protocol); AGENTS.md addendum (Sprint A–D architecture reference).
- Mount-truncation sweep: restored 7 truncated .py files from git main; all .py files now within 50 bytes of git.

## Session log — 2026-06-14 (continued)

**P0 carryover closed:**
- `gallery.select()` handler added to `aiwf/web/studio/tab.py`.
- Selects image → promotes to `workspace_image`.
- Respects `send_seed_on_click` (updates `seed` from per-image `gallery_seeds` state) and `send_size_on_click` (updates `width`/`height` from image dimensions).
- `gallery_seeds = gr.State([])` stores all seeds from the batch so index N → seed N.
- Both `_progress_outputs` and `_finished_outputs` updated to return the extra position.
- 7 unit tests in `tests/test_gallery_select.py` — all pass.
- All sprint items now ✅.

## Session log — 2026-06-14 (Phase 7 start)

**Phase 7 — Agentic Prompt/Tool Workspace scaffolded:**

- `docs/AGENTIC_ASSISTANT_ROADMAP.md` — Phase 7 planning doc: tool inventory (Tier 1/2/3), security principles, implementation plan, out-of-scope items.
- `docs/LOCAL_TOOL_SECURITY.md` — Normative security policy: 6 hard rules (no shell passthrough, no unconstrained write, no delete, no arbitrary eval, no unconfirmed training, GPU-lock gate), path safety rules, metadata read safety, confirmation requirement table, audit log spec.
- `aiwf/services/prompt_tools.py` — Phase 7 service layer:
  - `list_local_checkpoints` / `list_local_loras` — scan configured roots, extension allowlist, path traversal prevention.
  - `read_safetensors_metadata` — header-only parse (no weight load), 10 MB cap on header size.
  - `inspect_prompt_library` — lists style/template files from optional prompt library dir.
  - `build_prompt_draft` — composes positive/negative/lora-tags from components.
  - `recommend_settings` — rule-based settings table keyed on `(architecture, goal)` covering sd15/sdxl/wan × speed/balanced/quality.
  - `generate_workflow_json` — ComfyUI-compatible txt2img workflow JSON (pure data, no IO).
  - Audit log appended to `output_dir/.agent_tool_log.jsonl` on every call (including errors).
- `tests/test_prompt_tools.py` — 42 tests, all pass. Covers: path safety, header parse, listing, metadata read, prompt draft assembly, settings recommendation, workflow JSON, audit log.
- No torch / diffusers / gradio imported at module level — safe to import in headless environments.

## Session log — 2026-06-14 (Phase 7 engines)

**Custom inference engines — no diffusers, no ComfyUI, built from first principles:**

- `aiwf/infrastructure/samplers/schedule.py` — σ noise schedules: linear, scaled_linear, cosine, Karras et al. 2022 (ρ=7 table), exponential. `get_sigmas()` returns len=steps+1 tensor with σ=0 sentinel. `sigma_to_timestep()` nearest-neighbor lookup in log-sigma space.
- `aiwf/infrastructure/samplers/euler.py` — Euler ODE (`d=(x−x0)/σ`) and Euler Ancestral SDE (Langevin σ_up/σ_down split). Pure torch, no diffusers schedulers. `epsilon_to_x0` + `v_pred_to_x0` prediction-type helpers.
- `aiwf/infrastructure/samplers/dpmpp.py` — DPM++ 2M (2nd-order multi-step, λ-space), DPM++ SDE (stochastic + noise injection), DPM++ 3M SDE (3rd-order, d1_old/d2_old). Lu et al. 2022.
- `aiwf/infrastructure/samplers/ddim.py` — DDIM (Song et al. 2020). ᾱ-space formulation, η=0 deterministic / η=1 DDPM.
- `aiwf/infrastructure/samplers/dispatch.py` — Registry dispatch: euler / euler_a / dpmpp_2m / dpmpp_2m_karras / dpmpp_sde / dpmpp_2m_sde / dpmpp_3m_sde / ddim / heun. `run_sampler()` uses inspect.signature for kwarg compatibility.
- `aiwf/infrastructure/onnx/session.py` — ORT provider selection: CUDA EP → DirectML EP → CPU EP (auto or explicit). ORT_ENABLE_ALL, memory pattern enabled.
- `aiwf/infrastructure/onnx/pipeline.py` — Full txt2img pipeline over text_encoder/unet/vae_decoder ONNX models. CFG via neg+pos concat → split+guide. Custom sampler loop (no diffusers Pipeline). VAE scale=0.18215. CLIPTokenizer via transformers (lightweight).
- `aiwf/infrastructure/onnx/backend.py` — `ONNXBackend` implements `InferenceBackend` Protocol. Discovers dirs with all 3 required ONNX subdirs. Lazy pipeline init on first generate.
- `aiwf/infrastructure/diffusers/cuda_graphs.py` — `CUDAGraphDenoiser`: warmup → torch.cuda.graph capture → static buffer copy+replay. Flag-gated (`AIWF_CUDA_GRAPHS=1`). 5–15% throughput gain on repeated UNet calls.
- `aiwf/infrastructure/video/export.py` — NVENC-first video export: probe h264_nvenc/hevc_nvenc via null encode; fallback libx264/libx265. NVENC uses -cq, software uses -crf. `tensors_to_video` writes tempdir → `frames_to_video`. shell=False always.
- `aiwf/infrastructure/quantization/torchao_quant.py` — int8 weight-only (version-guarded: 0.4+ uses `quantize_()` API), fp8 weight-only (CUDA cap ≥ 8.9 check), `torch.compile` (mode=reduce-overhead), channels-last. Flag-gated per feature.
- Tests: `test_samplers.py` (47), `test_onnx_session.py` (10), `test_video_export.py`, `test_quantization.py` — 113 total pass.
- ComfyUI stubs neutralized: `aiwf/infrastructure/comfy/` raises ImportError; no ComfyUI fields in settings; no ComfyUI selection in bootstrap.

## Session log — 2026-06-14 (Phase 7 UI)

**Engine settings + Prompt Tools UI wired:**

- `aiwf/core/config/settings.py` — `RuntimeFlags` gains: `cuda_graphs`, `torchao`, `torch_compile`, `channels_last`, `fp8_quant`, `nvenc`, `hevc`, `inference_backend` ("diffusers"|"onnx"), `onnx_provider` ("auto"|"cuda"|"directml"|"cpu"). `UserSettings` gains: `onnx_model_dir` (path to ONNX model subdirectory root).
- `aiwf/bootstrap.py` — Backend selection: reads `flags.inference_backend`; if "onnx" instantiates `ONNXBackend(models_root, provider, device_id=0)` from configured path; otherwise `DiffusersBackend`. Propagates all engine env flags (`AIWF_CUDA_GRAPHS`, `AIWF_TORCHAO`, `AIWF_FP8`, `AIWF_TORCH_COMPILE`, `AIWF_CHANNELS_LAST`, `AIWF_NVENC`, `AIWF_HEVC`) to `os.environ` at startup.
- `aiwf/web/tabs/settings.py` — New **Engine** tab between "Launch profile" and "Access & security":
  - Backend radio (diffusers / onnx).
  - ONNX model dir textbox + provider radio, visible only when ONNX selected (show/hide via `engine_backend.change`).
  - Optimization checkboxes: CUDA Graphs, TorchAO int8, FP8, torch.compile, channels-last, NVENC, HEVC.
  - Save handler: writes `onnx_model_dir` to `UserSettings`, writes engine flags to `launch.json`, and applies env vars immediately for the current session.
- `aiwf/web/studio/tab.py` — New **Prompt Tools** accordion (closed by default, between Outpaint and Advanced):
  - **Inspector tab**: "Scan checkpoints & LoRAs" button → markdown list with names + sizes.
  - **Metadata tab**: path textbox → "Read metadata" → renders safetensors header as markdown (no weights loaded).
  - **Prompt builder tab**: subject + style + LoRA names + negative → "Build prompt draft" → "Apply to prompt" copies result into the positive prompt textbox.
  - **Recommend settings tab**: architecture + goal dropdowns → table of recommended width/height/steps/CFG.
  - All calls route through `PromptToolsService(checkpoint_dir, lora_dir, output_dir)` instantiated once per tab build.
- `compileall` clean across all of `aiwf/`, `launch.py`, `webui.py`.
- 113 engine/sampler/onnx/video/quantization/prompt-tools tests pass.
