# AIWF Studio Funding Folder

This folder tracks non-dilutive funding targets for AIWF Studio and related local-agent work.

Source check date: 2026-06-16.

## Positioning

AIWF Studio is a clean-room, local-first creative AI workspace. The funding angle is not "another hosted AI app." The strongest angle is local infrastructure that helps users run image generation, video generation, chat, model ops, and training on consumer hardware with explicit architecture instead of cloud lock-in.

Core claims to support with evidence:

- Public repo: https://github.com/nawnie/AIWF-Studio
- Local maturity: 715 collected tests across 87 test files in the current venv.
- GitHub main maturity: 289 collected tests across 56 test files.
- Architecture: typed domain models, service layer, backend adapters, AppContext wiring, no global `shared` state.
- Local-only progress: worker protocol, GPU tenant supervision, Ollama chat, Kohya/EveryDream2 training services, Wan/GGUF/video work, ONNX/Comfy scaffolding, quantization/model ops.

## Funding Targets

| Priority | Target | Best Use | File |
| --- | --- | --- | --- |
| 1 | AI Grant | Fast non-dilutive solo-builder grant | `targets/ai-grant.md` |
| 2 | a16z Open Source AI Grants | Network-backed open-source AI infrastructure support | `targets/a16z-open-source-ai-grants.md` |
| 3 | Deep Funding / ASI ecosystem | Milestone grant for decentralized local AI infrastructure | `targets/deep-funding.md` |
| 4 | Mozilla Technology Fund | Privacy, user-control, trustworthy local AI framing | `targets/mozilla-technology-fund.md` |
| 5 | Polar.sh + GitHub Sponsors | Immediate issue/roadmap funding | `targets/polar-and-github-sponsors.md` |
| 6 | Gitcoin Grants Stack | Community matching for public-goods AI infra | `targets/gitcoin-grants-stack.md` |

## Immediate Checklist

1. Update GitHub README to match local reality: chat, training, workers, GPU tenant supervision, Wan/GGUF, tests.
2. Add screenshots or a short demo GIF/video for Studio, Video, Training, Chat, and Model Manager.
3. Publish or push the local v2 work so grant reviewers can inspect the code.
4. Fill `shared/project-evidence.md` with exact repo/test/demo links.
5. Use `shared/core-pitch.md` as the base for each application.
6. Keep `tracker.md` updated with status, deadlines, and submitted material.

## Do Not Lead With

- A vague startup pitch.
- "AI art app" as the main category.
- A promise to beat cloud platforms on raw scale.
- Unsupported performance claims.

Lead with local control, consumer hardware, clean architecture, open-source maintainability, and concrete shipped code.

