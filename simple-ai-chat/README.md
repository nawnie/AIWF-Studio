# Simple AI Chat

Simple AI Chat is a local, agentic chat harness that routes every task to the
best Qwen-family model that fits on one 16 GB GPU, and swaps models in and out
of VRAM as needed.

- **Target machine:** RTX 4070 Ti SUPER 16 GB · 40 GB DDR4 · Intel i5 13th gen · Windows.
- **Brain:** [Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf). This is Qwen3.8-27B at 1.72 bits/weight: a 5.5 GiB file that keeps 98.2% of FP16 quality. It stays resident and leaves ~5 GiB free for specialists.
- **Specialists, loaded on demand:**

| Need | Model |
|---|---|
| Speech-to-text | Qwen3-ASR |
| Text-to-speech | Qwen3-TTS |
| Image generation and editing | Qwen-Image-2.1 + its official prompt enhancers |
| Environment simulation | Qwen-AgentWorld |
| Memory / RAG (on CPU) | Qwen3-Embedding / Reranker |
| Fast routing | Qwen3.5-0.8B |

## Status: phase 0 (research + scheduler core)

| Done | Next |
|---|---|
| [Research report](docs/RESEARCH.md): Qwen 3.8, Bonsai 2, every useful Qwen model, VRAM budgets, toggles, LoRAs, sources | Phase 1: process supervisor on the real GPU, OpenAI-compatible API, streaming chat UI, benchmark receipts |
| [Architecture](docs/ARCHITECTURE.md) | Phase 2: tool loop + MCP client toggles, web search, RAG |
| Model registry (`config/models.yaml`, `config/hardware.yaml`) | Phase 3: voice (ASR → brain → streaming TTS) |
| Byte-budget VRAM planner, affinity job queue, model manager with pinned restore and KV save/restore hooks | Phase 4: image gen/edit via sd.cpp or the AIWF Studio bridge |
| Capability router, llama-server command builder and process backend | Phase 5: simulation + game profile |
| 50 unit tests (no GPU needed) | Phase 6: A/B vs mainline Qwen3.8 + MTP, LoRA hot-swap |

## Try it (no GPU required)

```powershell
cd simple-ai-chat
python -m pip install -e .[dev]
python -m pytest -q

# list models and VRAM estimates against the 14.5 GiB planning budget
python -m simple_ai_chat models

# dry-run a session: voice models, then a prompt-enhanced image chain (a+b), then a simulation
python -m simple_ai_chat simulate qwen3-asr-0.6b qwen3-tts-1.7b-customvoice qwen-image-2.1-pe-t2i+qwen-image-2.1 qwen-agentworld-35b-a3b

# print the exact llama-server command for a model
python -m simple_ai_chat command bonsai2-27b
```

## Layout

```
simple-ai-chat/
  config/        models.yaml (registry, toggles, profiles), hardware.yaml (budget, binaries)
  docs/          RESEARCH.md, ARCHITECTURE.md
  simple_ai_chat/
    registry.py  model specs + VRAM estimates
    vram.py      GiB-budget eviction planner
    jobs.py      priority / affinity / aging queue
    manager.py   executes plans, pinned restore, KV save/restore, NVML calibration
    router.py    request -> capability steps, tool -> model
    gpu.py       optional NVML probe
    backends/    llama_server.py (Prism fork + mainline), fake.py (dry runs/tests)
  tests/
```

Runtimes, weights, logs and KV slots live in `bin/`, `models/`, `logs/` and
`slots/`. All of these are git-ignored.

## Before the first GPU run

1. Set **NVIDIA Control Panel → CUDA – Sysmem Fallback Policy → Prefer No Sysmem Fallback**, so VRAM over-commit fails loudly instead of silently spilling into system RAM.
2. Install the **PrismML llama.cpp fork** (release `prism-b10709` or newer, Windows CUDA 12.4 build) into `bin/llama-prism/`, and mainline llama.cpp into `bin/llama/`.
3. Download `Ternary-Bonsai-2-27B-PTQ1_0.gguf` and the `mmproj-Q8_0` file into `models/bonsai2-27b/`.

Stock llama.cpp, Ollama and LM Studio cannot load Bonsai 2.
