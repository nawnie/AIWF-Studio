# AIWF Studio

AIWF Studio is a clean-room rebuild of the AUTOMATIC1111-style Stable Diffusion web UI. The goal is not to port old internals forward. The goal is to ship a local-first creative tool with explicit wiring, typed models, predictable behavior, and room to grow.

## What makes this different

- No global `shared` state.
- No mystery callbacks or monkey-patched extension hooks.
- UI actions call services, not torch code directly.
- Requests flow through typed domain models instead of ad-hoc dicts.
- The app is designed to run from its own repo-local folders instead of silently leaning on a neighboring legacy install.

## Current feature set

### Studio

- txt2img, img2img, and inpaint
- live preview, continuous generation, interrupt
- hires fix, CFG, steps, sampler, clip skip, VAE selection
- tags, PNG metadata, seed reuse, before/after compare
- dynamic prompts, wildcards, prompt files, Compel support
- style presets with editable templates
- single-unit ControlNet in Studio Advanced
- SAM-assisted masking for inpaint
- ReActor-style face swap on results

### Extra tabs

- Models
- Segment
- Enhance
- Chat
- Video
- RIFE
- Training
- Workflows
- Face Swap
- Library
- PNG Info
- History
- Settings

### API

- native `/api/v1`
- A1111-style `/sdapi/v1` adapter

## Quick start

```bat
webui-user.bat
```

Or:

```powershell
python launch.py
```

By default the app uses dedicated local folders inside this repo:

```text
models/
outputs/
prompts/
wildcards/
workflows/
```

Put checkpoints in:

```text
models/Stable-diffusion/
```

Put LoRAs in:

```text
models/Loras/
```

Put VAEs in:

```text
models/VAE/
```

## Remote access and security

AIWF Studio includes Tailscale-aware connection info in Settings, which is the preferred remote path when you want phone or tablet access without exposing the app broadly.

Security guidance:

- `--listen` makes the UI reachable from other devices on your network.
- Add `username:password` auth before using remote access outside a trusted desk setup.
- Treat Gradio public share links as convenience tools, not private tunnels.
- Tailscale is the safest built-in option for routine remote use.

## Workflows status

The Workflows tab is still a work in progress. It is useful, but it should be treated as experimental until it gets a deeper validation pass.

## Active roadmap

- extension management
- broader theme and workspace customization
- more mature workflow authoring
- deeper training-engine setup flows

## Credits and thanks

This project is clean-room code, but it is absolutely standing in conversation with the wider local-image community.

- [AUTOMATIC1111 / stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
- [ControlNet](https://github.com/lllyasviel/ControlNet)
- [sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet)
- [ReActor](https://github.com/Gourieff/sd-webui-reactor)
- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [Diffusers](https://github.com/huggingface/diffusers)
- [GFPGAN](https://github.com/TencentARC/GFPGAN)
- [CodeFormer](https://github.com/sczhou/CodeFormer)
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)

See [docs/ATTRIBUTION.md](docs/ATTRIBUTION.md) for the fuller attribution trail.

## Clean-room rule

Allowed:

- studying behavior
- reading public docs
- reimplementing compatible ideas

Not allowed:

- copying incompatible source
- importing abandoned plugin code wholesale
- recreating legacy global-state architecture

## Development

```powershell
python -m pytest tests/ -q
python -m aiwf.app
```

## Repo notes

- `venv/` is local only and should not ship in the public repo.
- `models/` and `outputs/` are user-local runtime data.
- `AGENTS.md` is for local build sessions, not end-user repo content.
