AIWF STUDIO bbb1cae — STUDIO v5 ROOT UPDATE
============================================

TARGET
  Commit: bbb1cae619ba592e3d7c94ae7bb933206dd0f2ae
  Primary UI: original Gradio Studio
  Excluded from source overlay: Modern Gradio, Pro/React, frontend/

INSTALL — EXTRACT AND LAUNCH
  1. Close AIWF Studio.
  2. Extract ALL ZIP contents directly into the AIWF-Studio root.
  3. Allow matching files to be overwritten.
  4. Double-click STUDIO-V5-VERIFY.bat.
  5. Start normally with webui.bat.

No PowerShell command is required. The first normal launch runs the inherited
transactional bbb1cae shared-backend hotfix once. Studio v5 already contains
all Studio v4 files, so do not apply v4 afterward.

WHAT CHANGED
  IMAGE LAB
    - New Workflow sub-tab before XYZ, Batch, and Loopback.
    - User-selectable stages with settings shown only for selected stages.
    - Canonical order: mask -> inpaint -> denoise -> restore -> tone ->
      upscale -> resize -> export.
    - Auto-mask, uploaded-mask, inpaint, deterministic cleanup, AI restore,
      AI upscale, final resize, output, and job.json manifest support.

  SEGMENT / AUTO MASK
    - Presets now include threshold, candidate, dilation, blur, feather, and
      a reliability note instead of only a prompt word.
    - Separate edge feather control is available in Segment and Image Lab.

  VIDEO LAB
    - Same stage-selector interaction as Image Lab.
    - Presets are starting points; every stage can be enabled or disabled.
    - Settings appear only when their stage is selected.
    - Editable controls include deinterlace cadence/parity, stabilization
      search and edge fill, deflicker window/mode, denoise and sharpen
      coefficients, custom resize, FPS conversion, audio noise profiling,
      LUFS/true-peak/LRA targets, codec quality, and audio bitrate.
    - Studio resolves filter order and still preserves FFmpeg planning,
      cancellation, atomic output, logs, and manifests.

  AUDIO LAB
    - Replaces the simple Audio tab with Mix & Sweeten, Generate,
      Project/MIDI, and Engine sub-tabs.
    - User-selectable gate, filters, three-band parametric EQ, compressor,
      pitch region, gain, pan, fades/gain envelope, loudness normalization,
      limiter, sample-rate conversion, and export.
    - Optional isolated environment at engines/audio_lab/.venv.
    - Install from the Engine sub-tab or install_audio_lab.bat.
    - MIDI metadata inspection and a structured DAW-command preview grammar.
    - Existing MusicGen, AudioGen, and MMAudio generation remains available.

AUDIO SCOPE NOTE
  Studio v5 implements deterministic single-file mixing and MIDI/project
  inspection. Natural-language multitrack arrangement commands are parsed into
  structured plans but are not executed yet. Full tempo-map, region-marker,
  MIDI-note editing, plugin automation, stems, and multitrack playback are the
  next Audio Lab engine milestone.

TOOLS
  STUDIO-V5-VERIFY.bat    Hash, scope, syntax, and backend invariant checks.
  STUDIO-V5-QA.bat        Targeted regression tests that do not need CUDA.
  STUDIO-V5-ROLLBACK.bat  Restores the bbb1cae direct files and backend backup.
  install_audio_lab.bat   Optional isolated Audio Lab dependency install.

QA NOTE
  Real CUDA image generation, Segment models, enhancement models, Wan, and
  target-GPU load timing still require the target machine. Follow
  docs\STUDIO_V5_QA.md and use webui.bat --genlog.
