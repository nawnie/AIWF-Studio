# AI Grant

Official site: https://aigrant.org/

## Fit

Strong fit. AI Grant is the highest-priority target because AIWF Studio is visibly built by an individual technical builder, has a real repo, and is focused on practical AI infrastructure rather than a generic app wrapper.

Best angle:

- Local-first AI infrastructure.
- Consumer GPU enablement.
- Open-source developer tooling.
- Real code and tests instead of a pitch deck.
- Reducing dependence on centralized cloud AI platforms.

## What You Need

- Public GitHub repo updated to show the local v2 reality.
- Short application pitch, 1 to 3 paragraphs.
- Clear funding ask: start with `$25,000` or `$50,000` depending on confidence.
- Demo link or screenshots.
- Roadmap with 3 concrete milestones.
- Evidence that the work already exists: tests, modules, docs, screenshots.

## Suggested Ask

Ask for `$50,000` if the local v2 work is pushed and demoable. Ask for `$25,000` if applying before the public repo is fully updated.

## Milestones

1. Publish and document the local v2 architecture: workers, GPU tenants, chat, training, Wan/GGUF, model ops.
2. Harden worker isolation and optional-engine boot safety.
3. Produce consumer-GPU benchmarks for image/video/training flows with reproducible logs.

## Application Draft

AIWF Studio is an open-source, local-first AI workspace that helps users run image generation, video generation, chat, model operations, and training on their own hardware instead of depending on centralized cloud services.

The project started as a clean-room Stable Diffusion web UI, but the local version has grown into an organized local AI platform: typed domain models, a service layer, backend adapters, Gradio/FastAPI UI/API, supervised workers, GPU tenant coordination, Wan/GGUF video generation work, local chat, Kohya-style LoRA training, EveryDream2 full fine-tuning, model operations, and 715 collected tests.

Funding would let me spend focused time hardening the engine/worker boundary, publishing the local v2 work, documenting the architecture, and improving consumer-hardware workflows for people who want capable local AI without renting cloud GPUs or giving up control of their data.

## Before Applying

- Push local v2 work or create a visible branch.
- Add README section: "Current Local v2 Capabilities".
- Add screenshots.
- Add `fundings/shared/project-evidence.md` numbers to the README or a funding page.
- Prepare a 2 minute demo video.

## Caveats

The AI Grant site and intake process can change. Verify the current form and submission instructions before sending.

