# AIWF Studio GUI Readiness — June 24, 2026

## Decision

The **original Studio Gradio app remains the release surface**. Modern and Pro are useful design and API research tracks, but neither should receive the primary update at the expense of Studio parity or backend stability.

## Readiness matrix

| Track | Current role | Readiness | Studio v5 action |
|---|---|---:|---|
| Studio Gradio | Default and broadest workspace | Production-first | Receives all UI and backend work in this archive |
| Modern Gradio | Restyled navigation shell | Preview / incomplete parity | Port runtime visibility and calmer layout patterns; do not replace Studio files |
| Pro React + FastAPI | Active rebuild/API experiment | Promising, build-dependent, incomplete parity | Keep API concepts; shared backend fixes benefit it later; no frontend build in this archive |
| `second-gui-preview` | Earlier alternate-GUI exploration | Historical preview | Treat as design reference, not merge source |
| `dev` | Experimental integration lane | Not a release target | Review changes selectively; promote only tested shared code |
| A1111 compatibility branch | Separate compatibility work | Feature-specific | Keep out of Studio v5 to avoid scope collision |

## Reusable work ported into Studio

Studio v5 adopts the useful cross-GUI ideas without importing a second UI runtime:

- visible runtime/job status instead of opaque spinners;
- a resolved execution plan before starting expensive work;
- compact workflow navigation and grouped controls;
- diagnostics and manifest links beside output media;
- local-first messaging near file inputs;
- explicit busy/cancelled/failed/completed job states;
- backend-first interfaces that can later serve Gradio or React.

Image Lab, Video Lab, and Audio Lab now use the same concrete Studio pattern: selectable stages, resolved order, focused settings, visible status, and reproducible manifests.

## What is intentionally not merged

- React/Vite build output.
- Pro-only route assumptions.
- Modern placeholder panels.
- duplicated model loaders, queues, or settings stores.
- alternate launchers that could change the default narrative.

## Promotion gates for Modern

Modern should not be called parity-ready until it passes the Studio checklist for image generation, img2img/inpaint, ControlNet, model management, Wan, history/library, enhancement, faces, audio, settings, cancellation, errors, and mobile layout.

## Promotion gates for Pro

Pro additionally needs reproducible frontend builds, packaged static assets, API parity, streaming job events, authentication/network behavior, file upload/download parity, settings persistence, and a no-Node-at-runtime release path.
