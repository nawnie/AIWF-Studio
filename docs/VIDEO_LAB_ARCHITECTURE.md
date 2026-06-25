# AIWF Studio Video Lab — Gradio Architecture

## Scope of Studio v5

Studio v5 includes a deterministic, local-first **Video Lab** to the original Gradio application. It is deliberately implemented on the existing Studio backend rather than in Modern or Pro.

The user now chooses stages, while Studio resolves this canonical graph:

```text
Inspect → Trim → Deinterlace → Stabilize → Deflicker → Denoise → Sharpen → Resize → FPS → Audio cleanup → Loudness normalize → Encode → Atomic publish
```

Implemented stages:

- ffprobe metadata inspection with an OpenCV fallback.
- Accurate trim range planning.
- Deinterlace controls for cadence, parity, and marked-frame scope.
- Stabilization controls for X/Y search radius, edge fill, block size, and contrast threshold.
- Deflicker window and averaging mode.
- Direct `hqdn3d` spatial/temporal coefficients and `unsharp` kernel/amount controls.
- 720p, 1080p, 2×, or custom Lanczos resizing with optional aspect preservation.
- FPS conversion or FFmpeg motion interpolation.
- Audio high-pass, low-pass, FFT noise reduction/floor/type/tracking, and editable LUFS, true-peak, and LRA targets.
- H.264/H.265 software encoding or NVIDIA NVENC when available.
- MP4 and MKV output, metadata/chapter preservation, and MKV subtitle copying.
- One active Video Lab encode per local Gradio process.
- Cancellation, atomic partial-file handling, FFmpeg logs, and `job.json` manifests.

## Memory model

Long media jobs must not retain every frame in Python. Studio v4 introduced `StreamingVideoWriter` and converts RIFE to an overlapping chunk pipeline:

```text
read N frames → interpolate N frames → write output → retain 1 boundary frame → repeat
```

The RIFE model remains loaded across chunks. The one-frame overlap prevents temporal gaps and the duplicate boundary output is removed before encoding.

## Existing AIWF engines to join next

The following engines remain separate services in Studio v5. They should join the Video Lab graph only after cancellation, tenancy, and resume semantics are shared:

1. RIFE optical-flow interpolation as a graph node.
2. NVIDIA RTX VSR / Video Effects SDK adapters.
3. Face restoration and tracked face replacement.
4. Existing enhancement/upscale services.
5. Audio generation and MMAudio-style conditioning.
6. Wan/VACE video-to-video and masked editing.

## Job graph contract for the next slice

Each future graph node should declare:

- input and output media contracts;
- CPU, CUDA, NVENC, and scratch-disk requirements;
- whether it is resumable;
- checkpoint boundary and cleanup behavior;
- compatible neighboring nodes for decode/encode fusion;
- estimated frame count and duration;
- cancellation granularity;
- GPU tenant and owner job ID.

The engine supervisor remains the authority for CUDA ownership. CPU-only FFmpeg work should release the GPU before encoding whenever no later GPU node needs immediate residency.

## Media fidelity rules

- Use ffprobe/FFmpeg as the source of truth for streams, timing, chapters, rotation, codecs, and filters.
- Never build FFmpeg commands through a shell.
- Write to a partial file and publish with an atomic rename.
- Keep source files unchanged.
- Preserve metadata and chapters unless the user disables them.
- Make audio/subtitle stream selection explicit in the next UI slice.
- Record the resolved command and settings in the job manifest.

## Video Lab release phases

### Phase A — included through Studio v5

Deterministic FFmpeg graph, editable presets, stage-by-stage settings visibility, manifests, cancellation, and chunked RIFE backend.

### Phase B — next

Unified queue, resumable chunks, RIFE node integration, stream selection, before/after preview, and history reopen.

### Phase C

VSR/upscale, face restoration, optical-flow masks, scene detection, and per-scene settings.

### Phase D

Wan/VACE video editing, reference conditioning, tracked masks, generated audio, and multi-output timeline jobs.
