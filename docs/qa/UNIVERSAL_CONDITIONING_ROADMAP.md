# Universal Conditioning Roadmap

This note tracks the practical roadmap for AIWF Studio's universal conditioning work.

## Goal

Build a shared prompt-conditioning system that can reuse a smaller backbone with model-specific adapter heads, instead of depending on a separate large text encoder for every route.

AIWF already has a working smaller-adapter proof on **Flux.1**. That path should be treated as the proven reference implementation.

## Roadmap

1. **Lock Flux.1 as the proven reference**  
   Freeze and document the working Flux.1 small-encoder + adapter path as the baseline control route.

2. **Build Gemma-small backbone**  
   Start the next backbone work from a Gemma-based small model so the project lines up better with LTX 2.3 and other newer decoder-style routes.

3. **Train Gemma → Flux.1 adapter as comparison**  
   Use the proven Flux.1 path as the first comparison target so Gemma-based conditioning can be measured against the already-working Flux.1 replacement.

4. **Train Gemma → Flux2 Klein / Qwen adapter**  
   Target the Qwen-style prompt conditioning used by Flux2 Klein.

5. **Train Gemma → Wan UMT5 adapter**  
   Build a Wan-specific adapter that matches the UMT5-style conditioning contract instead of treating Wan like Flux.

6. **Train Gemma → LTX 2.3 hidden-state adapter**  
   Final boss room: match the Gemma hidden-state contract used by LTX 2.3.

## Why this matters

The biggest practical win is reducing prompt encoding overhead. Large text encoders add latency, CPU/GPU pressure, model swaps, and VRAM overhead before generation even starts.

A shared conditioning layer could improve:

- prompt encode time
- VRAM use
- route consistency
- prompt embedding cache reuse
- maintainability across Flux, Flux2, Wan, Qwen, and LTX families

## Immediate next step

Research and document what each supported model family expects from its text encoder or conditioning input:

- Flux.1
- Flux2 Klein / Qwen
- Qwen Image
- Wan 2.2
- LTX 2.3

That research should become the contract map for future adapter work.
