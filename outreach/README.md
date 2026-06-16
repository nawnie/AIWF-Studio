# Outreach Package

This folder holds public launch material for getting infrastructure-minded developers to inspect and contribute to AIWF Studio.

Use the project angle carefully:

- Do not pitch AIWF Studio as another image generator.
- Lead with local runtime architecture, engine isolation, VRAM tenant control, typed worker protocols, and test coverage.
- Link directly to setup and architecture material before asking for help.

## Recommended order

1. Finish and verify root `CONTRIBUTING.md`.
2. Finish and verify root `ARCHITECTURE.md`.
3. Open 3-5 GitHub issues from `github-issues.md`.
4. Post Show HN only when the public repo has a working clone/setup path.
5. Post Reddit variants only after checking each subreddit rule set that day.

## Files

- `research.md` - source-backed outreach notes and caveats.
- `show-hn.md` - Hacker News launch draft.
- `reddit-local-llama.md` - local hardware and runtime isolation draft.
- `reddit-python.md` - Python process supervision and multi-venv draft.
- `github-issues.md` - issue drafts for inbound contributors.

## Current proof points

- 715+ collected tests; 722 collected locally in this workspace.
- Core app uses `AppContext` instead of global `shared` state.
- Heavy workers are subprocess-controlled through `ProcessSupervisor`.
- Optional engines must not become boot dependencies.
- Worker events are JSONL and documented in `docs/WORKER_PROTOCOL.md`.
