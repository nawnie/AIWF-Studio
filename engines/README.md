# Engines

`engines/` contains optional runtimes that should not destabilize the core AIWF
Studio app. The app should boot when every engine venv and upstream checkout is
missing.

## Maintainer Rules

- Track engine contracts: README files, requirements, worker scripts, and small
  adapters.
- Do not track venvs, model weights, generated media, or upstream runtime
  checkouts. These are ignored local assets.
- Keep imports lazy. Engine-only packages should not be required just to import
  `aiwf` or open the Gradio UI.
- Prefer configuration and explicit roots over hard-coded machine paths.
- Use worker JSONL events for long-running jobs instead of direct UI imports.

## Current Engine Roles

- `wan`: optional video generation runtime and experimental acceleration work.
- `ltx`: optional LTX 2.3 video generation runtime under `engines/ltx/LTX-2`.
- `generation`: image-generation dependency boundary for heavier backends.
- `ed2`: EveryDream2 full fine-tuning integration; local fork checkout lives
  under `engines/ed2/EveryDream2trainer` when installed.
- `kohya`: LoRA training stack boundary.
- `audio`: optional video-conditioned audio post-processing; MMAudio lives under
  `engines/audio/MMAudio` when installed.
- `pipeline_accel`: acceleration experiments; treat benchmark claims as invalid
  until reproduced with receipts.
- NVIDIA sample SDK folders may exist locally for VideoFX work, but they are
  dependency checkouts, not AIWF source.

## Bootstrap And Verify

Use the helper scripts from the repo root:

```powershell
.\scripts\bootstrap_engine.ps1 -Name wan
.\scripts\verify_engine.ps1 -Name wan
```

Specialized engines may have their own bootstrap script, such as
`scripts/bootstrap_mmaudio.ps1` or `scripts/bootstrap_ltx.ps1`.

## Adding An Engine

1. Create `engines/<name>/README.md` and `requirements.txt`.
2. Add a worker only if the engine needs subprocess isolation.
3. Register the engine through the existing tenant/supervisor path.
4. Add focused tests for missing-engine startup, command construction, and any
   request translation.
5. Document what is contract-tested versus what still requires a real GPU smoke.
