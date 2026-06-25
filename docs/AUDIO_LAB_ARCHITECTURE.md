# Audio Lab architecture

## Current v5 engine

The optional engine lives in `engines/audio_lab/.venv` and is installed by `scripts/bootstrap_audio_lab.py`. It is intentionally separate from the image/video Python environment.

Core processing uses:

- Pedalboard for gate, filters, EQ, compressor, pitch shift, gain, and limiter;
- SoundFile for PCM/FLAC I/O;
- pyloudnorm for integrated-loudness normalization;
- librosa for optional sample-rate conversion;
- pretty_midi and mido for MIDI metadata;
- music21 as an optional foundation for later harmonic analysis.

The core environment does not install a second CUDA or PyTorch stack.

## Signal-chain capabilities

Implemented:

- trim;
- noise gate;
- high-pass and low-pass filters;
- low shelf, parametric mid, and high shelf EQ;
- compressor;
- whole-file or time-region pitch shift;
- gain and pan;
- fade in/out and gain-envelope interpolation;
- LUFS target normalization;
- limiter;
- WAV 24-bit and FLAC 24-bit output;
- sample-rate conversion;
- input and MIDI metadata inspection;
- per-job request and final manifest.

## DAW project model — next engine milestone

The full workstation needs stable IDs and metadata for:

- project tempo and time-signature maps;
- measures, beats, markers, named regions, and chorus/verse labels;
- audio and MIDI tracks;
- clips, source media, offsets, stretch state, and take lanes;
- instruments, programs, note events, velocities, articulations, and expression;
- buses, sends, inserts, plugin parameters, and automation lanes;
- non-destructive edit history and render snapshots.

A request such as "transpose the second chorus at measure 64 up three semitones" must resolve a project region and tempo map before execution. A request to add cello in unison with trumpet track 2 needs note-event metadata or a transcription stage; a mixed waveform alone is not sufficient.

## Planned slices

1. Multitrack project schema, waveform overview, transport, and timeline.
2. MIDI event editor, tempo map, markers, and command execution.
3. Stems, buses, sends, plugin racks, and automation lanes.
4. Harmonic/key/mode analysis and chord-structure suggestions.
5. Optional AI stem separation, transcription, source modification, and generation adapters.
