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

Current limitation: the video writer is frame-only. Audio is not copied or muxed
back into the result. Add an ffmpeg muxing step later if UI workflows need audio
preservation.

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

The tests patch `process_video_file(...)` at the service boundary so they do not
depend on local OpenCV codec availability.

## Next steps

- Add UI upload/output controls in Face Swap and Enhance tabs.
- Add cancellation/progress streaming if long video jobs need interrupt support.
- Preserve audio by muxing the original audio stream into the processed video.
- Consider chunked temp-frame output for very long clips if OpenCV writer errors
  need easier recovery.
