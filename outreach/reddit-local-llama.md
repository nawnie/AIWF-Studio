# r/LocalLLaMA Draft

## Title

Orchestrating Diffusers, GGUF chat, Wan video, and training workers under one UI without turning 16 GB VRAM into a crash loop

## Draft

Disclosure: this is my open-source project, AIWF Studio.

I am building AIWF Studio as a local-first AI runtime rather than another single-purpose image UI. The hard part has been making several local AI paths coexist on consumer hardware:

- Diffusers image generation
- GGUF/Ollama chat
- Wan image-to-video experiments
- Kohya-style LoRA training
- EveryDream2 full fine-tuning

The approach is to treat VRAM as a tenant-owned resource. Heavy runtimes switch through an engine supervisor, and optional engines can run as subprocess workers in their own venvs. The core Gradio app stays up even when an engine venv is missing or a worker crashes.

The pieces that may interest this sub:

- `EngineTenant` domain model for image/video/chat/training/enhance ownership
- `ProcessSupervisor` for named worker slots, tree-kill, and stdout streaming
- JSONL worker protocol for progress/status/artifact/error events
- Core `AppContext` dependency injection instead of global shared state
- 715+ collected tests; 722 collected in the current workspace

First-time contributor setup avoids optional engines:

```powershell
git clone https://github.com/nawnie/AIWF-Studio.git
cd AIWF-Studio
python launch.py --skip-sageattention --skip-wan --skip-kohya --skip-ed2
.\venv\Scripts\python.exe -m pytest --collect-only -q tests
```

Repo:

https://github.com/nawnie/AIWF-Studio

I am specifically looking for feedback on whether the engine boundary is sane for local hardware users: what should be inside the stable core venv, what should be isolated, and what runtime state should be visible in receipts/benchmarks so people can compare methods honestly.

## Posting notes

- Re-check r/LocalLLaMA rules before posting.
- Keep the affiliation disclosure at the top.
- Do not frame as "I found this."
- Prefer a discussion tag if the subreddit requires it.
