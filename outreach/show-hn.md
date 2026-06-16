# Show HN Draft

## Title

Show HN: AIWF Studio - local-first multi-engine AI runtime with VRAM isolation

## URL

https://github.com/nawnie/AIWF-Studio

## First comment / text draft

Hey HN,

I am Shawn (`nawnie` on GitHub). I have been building AIWF Studio as a clean-room alternative to monolithic, global-state-heavy local AI web UIs.

The project started as a Diffusers-based image UI, but the real work has become systems plumbing: keeping local image generation, GGUF chat, Wan video, Kohya-style LoRA training, and EveryDream2 full fine-tuning from fighting over the same Python environment and the same 16 GB consumer GPU.

The architecture is built around:

1. Engine tenant models for GPU ownership. Heavy runtimes switch through a supervisor instead of assuming they can all stay resident.
2. A `ProcessSupervisor` and typed JSONL worker protocol. Volatile engines can run in separate venvs and crash without taking down the Gradio UI.
3. Explicit `AppContext` dependency wiring. There is no legacy-style global `shared` object.
4. A broad regression suite. The current local collection is 715+ tests, with 722 collected in this workspace.

The contributor setup path is intentionally boring:

```powershell
git clone https://github.com/nawnie/AIWF-Studio.git
cd AIWF-Studio
python launch.py --skip-sageattention --skip-wan --skip-kohya --skip-ed2
.\venv\Scripts\python.exe -m pytest --collect-only -q tests
.\venv\Scripts\python.exe -m pytest tests -q
```

I am looking for contributors who like Python infrastructure, process supervision, local GPU memory behavior, test design, and clean engine boundaries.

Useful places to start:

- `CONTRIBUTING.md`
- `ARCHITECTURE.md`
- `docs/ENGINE_ISOLATION.md`
- `docs/WORKER_PROTOCOL.md`

I would especially value feedback on the worker protocol, engine readiness flow, and whether the multi-venv boundary is clean enough for new runtimes to be added without turning the core app into a dependency trap.

## Posting notes

- Post only when the public repo clone/setup path is ready.
- Use the URL field for the GitHub repo.
- Do not ask for upvotes or comments.
- Stay available for technical questions after posting.
