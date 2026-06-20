# Audio engine

This folder is for optional audio runtimes that should not be installed into the
shared Studio venv.

## MMAudio video-conditioned audio

Internal notes may call this MVP `VAP` for video-audio post-processing. In UI
and user-facing copy, call it video-conditioned audio: AIWF generates the video
first, then asks an audio model to create matching sound for that finished clip.
This is intentionally after-video post-processing, not an audio-driving-video
pipeline.

AIWF's first video-audio MVP expects:

```text
engines/audio/.venv/
engines/audio/MMAudio/demo.py
```

Use:

```powershell
.\scripts\bootstrap_mmaudio.ps1
```

The Wan Video tab can then run a generated MP4 through MMAudio, save a `.flac`,
and mux it into the final MP4 with ffmpeg.

The service calls MMAudio through its `demo.py` CLI using the dedicated
`engines/audio/.venv` Python. Keep this engine local-only and isolated from the
shared Studio environment so its dependencies cannot destabilize the video stack.

MMAudio code is MIT licensed. The released checkpoints are CC-BY-NC 4.0, so keep
commercial use disabled or separately licensed.
