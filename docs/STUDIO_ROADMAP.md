# Studio-first roadmap after v5

## Priority 1 — Image Lab hardening

- Add mask painting/refinement and mask history.
- Add per-stage before/after preview and stage bypass.
- Add face-region-only restore and compositing.
- Add batch Workflow execution and resumable job folders.
- Add architecture-aware model residency badges and load receipts.

## Priority 2 — Video Lab AI graph

- Join chunked RIFE, Enhance/VSR, face restore, ReActor, and audio nodes under one resumable graph.
- Stream decode/process/encode in bounded chunks.
- Preserve all user-selected streams, timestamps, rotation, color metadata, chapters, and subtitles where the container permits.
- Add tracked masks and temporal consistency.
- Add Wan/VACE editing only after deterministic restoration is stable.

## Priority 3 — Audio Lab workstation

- Implement the multitrack project schema and timeline.
- Add waveform peaks, transport, markers, tempo/time signatures, and measure/beat rulers.
- Add MIDI note/velocity editing and command execution.
- Add plugin racks, buses, sends, envelopes, and render snapshots.
- Add optional stem separation, transcription, harmonic analysis, and AI assistance.

## Other GUI readiness

Modern Gradio remains a useful layout prototype but is not parity-complete. Pro React/FastAPI remains a separate rebuild track with a frontend build dependency. Shared backend/job contracts from Studio should flow outward only after Studio QA; UI source should not be copied between tracks.
