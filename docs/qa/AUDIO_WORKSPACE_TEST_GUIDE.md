# Audio workspace tester guide

Use this guide for the first practical review of AIWF's local audio tools. The goal is to learn which workflows are useful and where the current controls or output fall short. It is not a quality benchmark.

## Current routes

- Pro Audio Studio generates music with `facebook/musicgen-small` and sound effects with MMAudio Small 16 kHz.
- MMAudio Small 16 kHz generates text sound effects and video-conditioned audio in `engines/audio/.venv`.
- Audio Lab cleans, mixes, normalizes, and exports existing audio in `engines/audio_lab/.venv`.
- Video Lab attaches MMAudio output to an existing clip with FFmpeg.

MusicGen Small and the released MMAudio checkpoints use CC-BY-NC 4.0 model-weight licenses. Treat their output as non-commercial research material unless separate rights are available. See the [MusicGen model card](https://huggingface.co/facebook/musicgen-small) and the [MMAudio repository](https://github.com/hkchengrex/MMAudio).

## First setup

1. Open Audio in Pro or Audio Lab in Gradio.
2. Click **Download minimum models & dependencies**.
3. Leave Studio open while setup runs. A clean machine may need about 10 GB; existing files are reused.
4. Confirm MusicGen Small, MMAudio Small 16 kHz, Audio Lab DSP, and FFmpeg all report ready.

The minimum setup does not install AudioGen Medium or larger MMAudio checkpoints.

## Test passes

### Music

Generate 8 seconds from this prompt:

> restrained ambient synth pulse, soft analog bass, sparse percussion, no vocals

Check that the player opens, the saved file plays outside Studio, the duration is close to 8 seconds, and a second render with the same seed is reasonably repeatable.

### Sound effects

In the Gradio Generate tab, choose Sound effects and MMAudio Small 16 kHz. Try:

> boots walking on wet concrete in a quiet tunnel, cloth movement, distant ventilation

Listen for prompt coverage, timing, obvious digital artifacts, speech-like noise, and unwanted music.

### Video-conditioned audio

Use a short clip with one obvious event. Describe only sounds that match the visible action. Check synchronization, the final mux duration, and whether the original video remains visually unchanged.

### Mix and sweeten

Process one clean music clip and one noisy spoken clip. Test the Music sweeten and Podcast cleanup presets. Confirm the output does not clip, the loudness target is sensible, and the export length matches the intended trim.

## What to record

For each run, note:

- GPU and system RAM
- model, prompt, duration, steps, guidance, and seed
- first-load time and repeat-run time
- peak VRAM if available
- output path and whether it plays in another app
- one thing that worked and one thing that needs improvement

## Questions for an experienced audio user

- Which three jobs should Audio Studio make fastest: scoring video, sound design, cleanup/mastering, voice work, or loop creation?
- Which controls belong in the main view, and which should stay in an advanced panel?
- Are waveform editing, multitrack arrangement, automation, or stem export necessary for the first useful release?
- Which meters are essential: peak, true peak, LUFS, spectrum, phase, or none?
- What failure is most damaging: wrong timing, noise, clipping, weak prompt match, repetition, or poor export handling?

Current limits are intentional: Pro generates MusicGen clips up to 30 seconds, project save/load is disabled, the DAW command parser is planning-only, and larger or commercial-friendly model options still need evaluation.
