# Studio v5 QA checklist

## Before launch

1. Confirm the repository is at commit `bbb1cae` or is an already-applied Studio v4/v5 overlay.
2. Extract the archive into the repository root.
3. Run `STUDIO-V5-VERIFY.bat`.
4. Run `STUDIO-V5-QA.bat`.
5. Launch with `webui.bat --genlog`.

## Image Lab

- Open Image Lab -> Workflow.
- Toggle each process and confirm only its settings appear.
- Confirm Export returns automatically if unchecked.
- Apply each preset and verify stage selection remains editable.
- Test Tone + Resize + Export without loading a model.
- Test uploaded-mask inpaint.
- Test Auto mask + inpaint with a SAM model.
- Verify mask presets update threshold, candidate, dilation, blur, and feather.
- Test restoration and upscaling separately before combining them.
- Confirm output and `job.json` exist.

## Video Lab

- Toggle every process and verify dynamic settings.
- Apply each preset, then modify at least one parameter in every selected stage.
- Verify custom resize rejects an empty width/height and accepts width-only, height-only, and bounded-box modes.
- Inspect a progressive and an interlaced source if available.
- Build a plan before running and inspect the concrete FFmpeg filter graph.
- Test H.264 software and Auto encoder.
- Test cancellation and verify partial output is removed.
- Test a clip with audio and one without.
- Verify output, FFmpeg log, and `job.json`.

## Audio Lab

- Open Engine and install the isolated environment.
- Refresh status and confirm self-test succeeds.
- Inspect WAV and FLAC inputs.
- Run a short clip through Gate -> EQ -> Normalize -> Limiter.
- Test mono pan output, fades, and a gain envelope.
- Test a bounded pitch region.
- Inspect a MIDI file and verify tracks, velocity statistics, tempo changes, and duration.
- Preview both documented DAW-command examples.
- Verify output, request JSON, and final `job.json`.

## GPU and model validation

The archive's automated tests do not certify production CUDA inference. On the target machine, test one cold and one warm run for SD/SDXL, Flux, Flux.2 Klein, Z-Image, and Wan. Record `--genlog` receipts and confirm no active job can be unloaded by another tenant.
