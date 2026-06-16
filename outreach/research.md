# Outreach Research Notes

## Verified repo-local facts

- `CONTRIBUTING.md` should live at repo root for GitHub contributor discovery.
- `launch.py` creates `venv/`, installs CUDA Torch, installs `requirements.txt`, and starts the app.
- The first test collection with system Python failed because dependencies such as `pydantic` and real `torch` were not available.
- Test collection through `.\venv\Scripts\python.exe` succeeded with 722 tests collected.
- Optional engines are configured through `engines.json`.
- `scripts/bootstrap_engine.ps1 <name>` creates isolated engine venvs for engine folders such as `wan` and `kohya`.
- The current worktree is dirty; documentation work should not revert unrelated changes.

## External source notes

GitHub contributor guidelines:

- Source: https://docs.github.com/en/communities/setting-up-your-project-for-healthy-contributions/setting-guidelines-for-repository-contributors
- GitHub supports contributing guidelines in repo root, `docs`, or `.github`.
- GitHub surfaces a `CONTRIBUTING.md` link for issues, pull requests, repository overview, and the repository contribute page.

Hacker News Show HN:

- Source: https://news.ycombinator.com/showhn.html
- A Show HN must be something people can try.
- The title should begin with `Show HN`.
- The project should be non-trivial and personally worked on.
- Do not ask friends to upvote or comment.
- Do not post a landing page or fundraiser as Show HN.

Reddit:

- r/LocalLLaMA rule snippets emphasize limited self-promotion and affiliation disclosure.
- r/MachineLearning has strict self-promotion rules and prefers posts that offer technical value or invite concrete discussion.
- Treat subreddit rules as time-sensitive. Re-check rules immediately before posting.

## Positioning

Strong pitch:

```text
AIWF Studio is a local-first multi-engine AI runtime with explicit VRAM tenant isolation, subprocess workers, and a 715+ test suite.
```

Weak pitch:

```text
I made an AI image generator.
```

## Claims to avoid until separately verified

- Do not claim all 722 tests pass unless the full suite has been run successfully after the final docs changes.
- Do not claim performance wins without benchmark receipts in `docs/benchmark_log.jsonl` or `outputs/benchmarks/`.
- Do not imply optional engines are required for first-time contributors.
- Do not imply HN or Reddit endorsement.

## Current recommended public numbers

- Use `715+ collected tests` in launch copy.
- Use `722 collected locally as of this workspace` only in contributor or research notes where the exact verification context is clear.
