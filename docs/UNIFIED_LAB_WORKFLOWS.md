# Unified Studio Lab workflow contract

Image, Video, and Audio use the same Gradio interaction model while retaining different processing engines.

## UI contract

- A preset populates stage selection and defaults.
- The stage selector remains fully editable.
- A stage's controls are hidden until the stage is selected.
- Export is mandatory and is reinserted if removed.
- The application, not checkbox click order, determines the execution order.
- The resolved order is visible before work starts.
- A JSON plan and final job manifest preserve the actual settings.

## Canonical orders

### Image

`Auto mask -> Inpaint -> Denoise -> Restore -> Tone/color -> AI upscale -> Final resize -> Export`

Mask generation may be omitted when the user uploads a mask. Restore and upscale are skipped with an explicit log entry when no model is selected.

### Video

`Inspect -> Trim -> Deinterlace -> Stabilize -> Deflicker -> Denoise -> Sharpen -> Resize -> Frame-rate conversion -> Audio cleanup -> Loudness normalize -> Export`

FFmpeg capability checks can remove unsupported filters and record warnings. The execution plan remains authoritative.

### Audio

`Trim -> Noise gate -> Filters -> EQ -> Compressor -> Pitch region -> Gain -> Pan -> Automation/fades -> Loudness normalize -> Limiter -> Export`

The current engine processes a single audio file. Future multitrack nodes will use the same stage-plan structure but add track, clip, bus, tempo-map, and automation-lane identities.

## GUI track boundaries

Studio is the target. Modern and Pro should consume shared backend contracts later; they should not fork the order rules or model/job services. No Modern, Pro, or React source is included in this archive.
