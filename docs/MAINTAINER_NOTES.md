# Maintainer Notes

AIWF Studio is a local-first consumer AI app with optional heavy engines around a
stable core UI. Keep maintenance work practical: preserve app boot, isolate
volatile ML stacks, and record enough proof that another maintainer can tell
what was tested without replaying the whole investigation.

## Daily Workflow

- Start from the current checkout, not an older Desktop path. The repo can live
  anywhere; path resolution should come from `AppContext`, `launch.json`, CLI
  flags, environment variables, or `engines.json`.
- Check `git status --short` before editing. Multiple agents often work in this
  tree, so do not revert unrelated dirty files.
- Keep generated media, local datasets, engine checkouts, venvs, logs, and
  private config out of source control. The tracked project story should be
  docs, tests, helper scripts, and source.
- Prefer focused tests first, then broader suites when the change touches shared
  contracts. Full-suite receipts matter after dependency, engine, or launch
  changes.
- Do not claim speedups without timing evidence. For Wan/video work, record
  method, model pair, resolution, frame count, steps, elapsed time, peak VRAM/RAM
  when available, and output path.

## Engine Boundaries

The core app should launch even when optional engines are absent. Engine-specific
imports belong inside workers, services, or lazily executed methods rather than
module import paths that run during UI startup.

## License Posture

For AIWF / local AI repos, default to a free non-commercial attribution license:
people can play, study, modify, and share with credit, while commercial use
requires contacting the project owner. Keep third-party model and SDK licenses
separate and explicit.

Use `engines/<name>/` for local engine ownership:

- `.venv/` is ignored and may be rebuilt.
- upstream runtime checkouts such as `engines/ed2/EveryDream2trainer`,
  `engines/audio/MMAudio`, and NVIDIA sample repos are local dependencies, not
  source files to edit casually.
- `requirements.txt`, `README.md`, and worker scripts are the durable tracked
  contract for each engine.
- `docs/ENGINE_ISOLATION.md` and `docs/DEPENDENCY_POLICY.md` are the source of
  truth for why these boundaries exist.

## Bootstrap Constraints

Windows bootstrap is intentionally conservative. Python 3.11/3.12 are the main
targets; Python 3.13 is still experimental for CUDA-heavy optional engines.

When installing or repairing an engine:

1. Use the relevant `scripts/bootstrap_*.ps1` helper when one exists.
2. Keep upstream legacy requirements out of the shared Studio venv unless the
   dependency policy explicitly allows it.
3. Verify the worker entry point with `scripts/verify_engine.ps1` when the engine
   has a worker contract.
4. Leave local machine paths in config or ignored files, not in import-time
   defaults.
5. Do not require hard links, junctions, or symlinks for normal operation. Add a
   Settings field, CLI flag, env var, or `engines.json` option instead.

## Tests And Receipts

Tests here are mostly contract tests: they prove bootability, request shaping,
catalog safety, worker isolation, and model/path discovery behavior without
requiring every heavy model to run. A passing unit suite is not the same as a
validated GPU smoke.

Use these receipt levels:

- contract: `python scripts/run_tests.py <suite>` or focused pytest node.
- startup: create the Gradio UI from `build_context()` without launching a long
  server.
- worker: probe an engine with its isolated Python and JSON request.
- runtime: produce or process a tiny media artifact and report the path.
- benchmark: compare timings with fixed method, model, settings, and hardware
  notes.

## Works But Not Optimized

These areas are usable or partially wired, but should not be treated as finished
performance work:

- Wan acceleration: runtime safety and backend selection are improving, but every
  speed claim still needs a benchmark receipt.
- Video post-processing: RIFE, VideoFX, face swap, and audio paths may soft-fail
  and preserve the last good video instead of aborting the whole job.
- ED2 and training: engine-local layout is the maintainable direction, while the
  UI visibility and full training lifecycle can stay separate from video-focused
  work.
- Audio generation: MMAudio is an optional after-video post-process with a
  separate venv and non-commercial checkpoint licensing constraints.

## Come Back To This

These are not blockers for normal development, but they are the next places a
maintainer should look when the surrounding feature becomes the active lane:

- Wan speed: add stage-level timing for high-load, high-denoise, low-load,
  low-denoise, VAE decode, video write, and post-processing. Use those numbers
  before changing defaults.
- Wan temporal/reference oddities: add a small debug export for source image,
  encoded reference latents, noise seed, frame count, and chunking plan. That is
  the fastest way to distinguish bad conditioning from bad model weights.
- Video audio post-processing: after MMAudio runs locally, parse the generated
  audio file for duration/sample rate and report the actual values in infotext
  instead of relying on model-variant assumptions.
- Engine install UX: Settings should eventually show per-engine readiness
  checks: Python exists, repo exists, package imports, checkpoint cache present,
  and last smoke result.
- AMD/DirectML: keep NVIDIA as the proven path for now, but leave the code
  boundaries clean enough that a future AMD route can be added without changing
  the user-facing workflow.

## Documentation Hygiene

Keep broad notes here. Keep user-facing video tool details in
`docs/VIDEO_TOOLS.md` and Wan local runtime details in
`docs/WAN_LOCAL_COMPONENTS.md`; edit those only when the specific surface changes.
Avoid adding long logs to docs. Summarize the command, result, and residual risk.
