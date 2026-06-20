# Video tools handoff

AIWF Studio now has shared frame-by-frame video infrastructure for image tools
that already operate on `PIL.Image` frames.

## Shared infrastructure

Start with `aiwf/infrastructure/video/frames.py`.

- `process_frame_sequence(...)` applies a frame processor to an in-memory frame
  list. Tests use this to avoid codec-specific failures.
- `process_video_file(...)` reads frames with OpenCV, converts each frame to PIL
  RGB, calls the supplied processor, and writes a new video.
- `VideoProcessResult` in `aiwf/core/domain/video.py` reports output path,
  frame count, FPS, dimensions, infotext, and message.

The shared frame processor can preserve an existing source audio stream when
callers pass `keep_audio=True`. Generated audio is handled separately by
`AudioGenerationService`, which writes WAV output and muxes it into a video with
ffmpeg.

## Face Swap video

`FaceSwapService.swap_video(...)` applies the existing image face swap to each
target video frame:

```python
ctx.faceswap.swap_video(
    "input.mp4",
    source_face_image,
    FaceSwapOptions(model_id="inswapper_128"),
    output_path="output.mp4",
)
```

It reuses `FaceSwapService.swap(...)`, so model loading, gender filters, face
mask correction, and optional restore callbacks stay in one place.

## Enhance / Upscale video

`EnhanceService.run_video_pipeline(...)` applies the existing Enhance pipeline to
each frame:

```python
ctx.enhance.run_video_pipeline(
    "input.mp4",
    upscale=UpscaleOptions(model_id="realesrgan-x2plus", scale=2),
)
```

`EnhanceService.upscale_video(...)` is a convenience wrapper for upscale-only
jobs. Restore and upscale use the same options as still images.

## Testing

Targeted tests live in `tests/test_video_tools.py`.

They verify:

- frame order and progress callbacks in `process_frame_sequence`
- `FaceSwapService.swap_video(...)` calls image swap once per frame
- `EnhanceService.run_video_pipeline(...)` calls image pipeline once per frame

## Generated audio

`AudioGenerationService.generate(...)` creates music or sound effects when the
optional AudioCraft stack is installed, with a Transformers MusicGen fallback
for music. `generate_and_mux(...)` generates audio at the target video duration
and writes a new MP4 with that audio track.

In code and planning notes, `VAP` means video audio post-processing: take an
already generated video, create or repair the audio layer, and mux the result
without changing the visual frames.

The first video-audio MVP route is video-conditioned audio through MMAudio. It
is isolated under `engines/audio/` instead of the shared Studio venv:

```text
engines/audio/.venv/
engines/audio/MMAudio/demo.py
```

When installed, the Wan Video tab can run a generated MP4 through MMAudio with
the selected prompt, write a `.flac`, then mux it back into the final MP4. If
MMAudio is not installed, the post-process soft-fails with an explicit setup
message and preserves the visual video output.

MMAudio code is MIT licensed, while its released checkpoints are CC-BY-NC 4.0.
Keep that visible in UI/help text before treating generated soundtrack output
as commercial-safe.

Do not blur this MVP with Wan S2V. MMAudio after Wan is the near-term VAP path
for completed clips; Wan S2V is a separate future route where sound participates
in the generation workflow itself.

Follow-up after the first local MMAudio smoke: record actual generated audio
duration/sample rate, preserve the audio prompt in the final video infotext, and
add an install probe that tells the user whether MMAudio is missing, importable
but missing checkpoints, or ready to run.

The tests patch `process_video_file(...)` at the service boundary so they do not
depend on local OpenCV codec availability.

## Next steps

- Add UI upload/output controls in Face Swap and Enhance tabs.
- Add cancellation/progress streaming if long video jobs need interrupt support.
- Consider chunked temp-frame output for very long clips if OpenCV writer errors
  need easier recovery.
