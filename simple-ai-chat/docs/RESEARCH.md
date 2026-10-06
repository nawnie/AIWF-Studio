# Simple AI Chat: Research Pass (Qwen 3.8 era)

Research date: 2026-10-06. Target machine: **RTX 4070 Ti SUPER 16 GB · 40 GB DDR4 · Intel i5 13th gen · Windows**.

This document is the pre-build research for Simple AI Chat, a local agentic
harness that routes each task to the best Qwen-family model that fits on one
16 GB GPU. It covers what exists, what fits, what to skip, and how to swap
models under VRAM pressure. Sizes come from the actual Hugging Face file
listings unless marked *estimate*.

---

## 0. TL;DR: decisions this research supports

1. **Ternary Bonsai 2 27B is the brain.** It is Qwen3.8-27B compressed to 1.72 bits per weight (5.54 GiB file). It keeps 98.2% of the FP16 benchmark average, against 98.7% for a 4-bit build three times its size. It is the only way to run Qwen3.8-27B-class reasoning and still leave **~5 GiB of VRAM free** for other models.
2. **The bottleneck is VRAM, not model choice.** Qwen3.8-27B is already the best open Qwen for coding, agents, vision and reasoning. Most of the "integrate every model" work is adding *abilities* the brain lacks: image gen/edit, ASR, TTS, audio understanding, environment simulation and RAG. Each is a separate model that must be scheduled.
3. **Use two llama.cpp builds:** the PrismML fork (required for Bonsai's PTQ1_0/PQ2_0 formats) plus mainline llama.cpp (MTP, router mode, Qwen3-ASR, Qwen3-TTS, embeddings, AgentWorld). Use stable-diffusion.cpp for Qwen-Image-2.1. Ollama and LM Studio cannot load Bonsai 2.
4. **Build our own byte-budget VRAM scheduler.** llama-server router mode and llama-swap both budget by *model count*, not gigabytes, and the router has a documented concurrency race. Our manager plans evictions in GiB, pins the brain, groups queued jobs by model, and saves and restores the brain's KV cache around exclusive jobs such as image generation.
5. **Many "official enhancers" become toggles:** thinking on/off, `reasoning_effort`, `preserve_thinking`, reasoning budget, MTP speculative decoding (mainline Qwen3.8 only), DFlash2 (third-party), the Qwen-Image-2.1 prompt enhancers, Qwen3Guard, KV-cache quantization, lazy vision tower and YaRN long context.
6. **I found no dedicated Qwen gaming model and no first-party Qwen MCP server.** Gaming uses the brain plus AgentWorld as a game-master/simulator. MCP comes from the harness acting as an MCP client, the same way Qwen-Agent and Qwen Code do.

---

## 1. Hardware envelope

| Resource | Value | Implication |
|---|---|---|
| GPU | RTX 4070 Ti SUPER, 16 GB GDDR6X, ~672 GB/s, Ada (sm_89) | Ada favors Bonsai's **PTQ1_0** packing for decode (PrismML's own table) |
| Usable VRAM | ~14.5 GiB planning budget | 16 GiB minus ~1 GiB for Windows/WDDM, desktop and browser, minus a 0.5 GiB safety margin |
| System RAM | 40 GB DDR4 | Enough to keep ~20 GiB of GGUFs warm in the OS page cache, or to hold MoE experts (`--n-cpu-moe`) |
| CPU | i5 13th gen (P+E cores, AVX2, no AVX-512) | Fine for embeddings, rerankers and the guard model on CPU. The Bonsai AVX-512 load crash does **not** apply |

**Critical Windows setting:** set *NVIDIA Control Panel → Manage 3D settings → CUDA – Sysmem Fallback Policy → Prefer No Sysmem Fallback*.
Otherwise the driver silently spills an over-committed model into system RAM
and decode speed collapses with no error. The scheduler needs a hard OOM it
can react to, not a silent slowdown.

### KV-cache cost (Qwen3.8-27B / Bonsai 2 27B)

Only 16 of the 64 layers use full attention (4 KV heads × 256 dim). The other
48 are Gated DeltaNet with a small fixed recurrent state, so KV is cheap:
**64 KiB per token at f16** (this matches PrismML's KV-CACHE.md).

| Context | f16 | q8_0 | q4_0 |
|---|---|---|---|
| 8K | 0.50 GiB | 0.27 GiB | 0.14 GiB |
| 32K | 2.00 GiB | 1.06 GiB | 0.56 GiB |
| 64K | 4.00 GiB | 2.12 GiB | 1.12 GiB |
| 128K | 8.00 GiB | 4.25 GiB | 2.25 GiB |
| 262K (native max) | 16.0 GiB | 8.50 GiB | 4.50 GiB |

On the PrismML fork, avoid `q5_0` KV: it is several times slower. Use f16,
q8_0 or q4_0. Quantized KV with flash attention may need
`-DGGML_CUDA_FA_ALL_QUANTS=ON` at build time.

---

## 2. Qwen 3.8: what was released

Qwen3.8 builds on the Qwen3.5 architecture: a hybrid of Gated DeltaNet linear
attention and gated full attention, with native vision and trained MTP heads.

| Model | Type | Open weights? | Fits this PC? |
|---|---|---|---|
| **Qwen3.8-27B** (Aug 14 2026) | Dense 27B, text+image+video → text, 262K ctx (1M with YaRN), Apache 2.0 | Yes (`Qwen/Qwen3.8-27B`, `-FP8`) | **Yes, via Bonsai 2 (ternary) or a ≤IQ4 GGUF** |
| Qwen3.8-Flash-Next (Aug 26) | MoE, 125B + 51B n-gram embeddings + MTP, ~6B active, multimodal, previews the Qwen4 architecture, Qwen Community License | Yes | **No.** Even ~2-bit exceeds 40 GB RAM plus 16 GB VRAM |
| Qwen3.8-2.4T-A95B (Aug 12) | MoE flagship, text-only, custom license | Yes | No |
| Qwen3.8-Max / Max-0902 / Max-Prime | API-only (Qwen Cloud) | No | Optional cloud fallback only |
| Qwen3.8-Omni-Flash / Omni-Flash-Realtime | API-only omni models with full-duplex voice | No | Optional cloud fallback only |
| Qwen3.8-LiveTranslate | API-only simultaneous interpretation | No | Optional cloud fallback only |

### Qwen3.8-27B facts that drive the harness design

- **Layout:** 64 layers = 16 × (3 × Gated DeltaNet + 1 × Gated Attention). Hidden size 5120, vocabulary 248,320. MTP is trained with multiple steps.
- **Thinking control:** thinking is on by default and can be disabled per request with `chat_template_kwargs.enable_thinking=false`. Depth is set with `reasoning_effort` ∈ {`xhigh` (default), `medium`, `low`}. `preserve_thinking` (default on) keeps reasoning from earlier turns.
- **Sampling:** thinking mode uses `temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence 0`. Instruct mode uses `temperature 0.7, top_p 0.8, top_k 20, presence 1.5`.
- **Output budget:** the model card recommends generous output limits. Small `max_tokens` values cause empty answers because thinking consumes the budget.
- **Tool calling:** OpenAI-style tools; vLLM and SGLang use the `qwen3_coder` parser.
- **Selected benchmarks (official card):**

  | Area | Benchmark | Score |
  |---|---|---|
  | Coding | SWE-bench Pro | 61.7 |
  | Coding | Terminal Bench 2.1 | 73.0 |
  | Coding | LiveCodeBench v6 | 90.3 |
  | Reasoning | GPQA-Diamond | 89.2 |
  | Agents | OSWorld-Verified (computer use) | 84.3 |
  | Agents | WebArena-Verified | 64.8 |
  | Agents | AndroidWorld | 81.9 |
  | Instruction following | IFBench | 79.5 |

  The card claims it beats Qwen3.7-Plus on most coding and agent rows.

**Conclusion:** Qwen3.8-27B is the strongest Qwen you can run locally for
coding, logic, agents and vision. No separate Qwen "coder" or "VL" model beats
it at a size that fits.

---

## 3. Ternary Bonsai 2 27B (PrismML)

Bonsai 2 is Qwen3.8-27B with an unchanged architecture and ternary weights
{−1, 0, +1}. It uses FP16 scales per group of 128 weights, stored in a
Hadamard-rotated basis. Released 2026-09-17 under Apache 2.0.

| Item | Value |
|---|---|
| Files (`prism-ml/Ternary-Bonsai-2-27B-gguf`) | `PTQ1_0` 5.54 GiB (1.75 bpw) · `PQ2_0` 6.71 GiB (2.13 bpw) · vision `mmproj-Q8_0` 0.59 GiB (optional) |
| Quality (14 thinking-mode benchmarks) | FP16 86.32 · **Bonsai 2 84.78 (98.2%)** · UD-Q4_K_XL 85.18 at 16.4 GiB · IQ2_XXS 72.59 |
| Retention by area | Strong: math (96.6 vs 97.1), coding (89.4 vs 89.1), instruction following (above baseline). Weaker: vision (66.2 vs 71.4), knowledge (79.9 vs 85.6), BFCL tool calling (74.9 vs 76.7) |
| Throughput (llama-bench, tg128) | RTX 4090: PTQ1_0 **91 tok/s**, PQ2_0 81 tok/s. On Ada, PTQ1_0 decodes faster and PQ2_0 prefills faster |
| Expected on 4070 Ti SUPER | **~55–65 tok/s decode** (*estimate*: scaled by memory bandwidth, 672/1008 × 91). Verify with `llama-bench` |
| Context | 262K native |
| Runtime | **PrismML llama.cpp fork only** (`prism-b10709`+; Windows CUDA 12.4 build recommended). Stock llama.cpp, Ollama and LM Studio cannot load PTQ1_0/PQ2_0, and the F16 file produces garbage on stock llama.cpp |
| Speculative decoding | **None for Bonsai 2 yet.** No drafter has been released; the older Bonsai 27B had DSpark drafters |

### Known issues that affect the harness (KNOWN_ISSUES.md, 2026-09-23)

- **Output limits:** use `-n 16384+` and `-c 65536`, or send `reasoning_effort: "medium"`, to avoid empty or truncated answers.
- **Effort values:** `"high"` returns HTTP 500. Only `low`, `medium` and `xhigh` are valid, and `low` ≈ `xhigh` in practice. The harness should map UI effort to {`medium`, `xhigh`} for Bonsai.
- **Thinking budget:** send it as a **top-level request field** or start the server with `--reasoning-budget N`. Putting it in `chat_template_kwargs` silently does nothing. To turn reasoning off, send `reasoning_effort: "none"` with the server left at `--reasoning auto`.
- **System messages:** the template requires exactly one, and it must come first. The harness must merge system prompts.
- **Tool calls:** tool calls are occasionally malformed or loop. Calls with empty arguments must send `"{}"`. Don't echo reasoning back into tool loops, because it breaks the prompt cache. The harness needs a tool-call validator and repair/retry loop.
- **Single-user server:** start with `-np 1 --cache-ram 24576`, or the prompt is reprocessed every turn.
- **Sampling defaults:** `presence_penalty` and `repetition_penalty` are not in the GGUF metadata. Send them per request.
- **Windows builds:** CUDA 13.3 builds crash on some systems. Use the CUDA 12.4 build.
- **LoRAs:** using LoRAs with Bonsai is *not documented*. Weights live in a rotated ternary basis, so assume standard LoRAs don't apply. LoRA work targets mainline models instead (§6).

**Why Bonsai over a normal 4-bit Qwen3.8-27B:** UD-Q4_K_M is 15.33 GiB and
does not fit 16 GB with any useful context. IQ3/Q3 builds (10.2–12.2 GiB) fit
alone but leave no room for companions. Bonsai fits with ~5 GiB to spare.

---

## 4. Capability map: which model does what

"Resident" means it can stay loaded next to the brain. "Exclusive" means the
brain must be evicted, or its KV saved first.

| Ability | Pick | Runtime | VRAM (est.) | Residency | Notes |
|---|---|---|---|---|---|
| **Chat · reasoning · logic · coding · agents** | Ternary Bonsai 2 27B (PTQ1_0) | PrismML llama-server | 5.54 GiB weights + KV (2.0 GiB @32K f16) + ~1.2 GiB compute | **Pinned** | One model covers all of these. "Coding mode" is a profile (system prompt, tools, `xhigh`), not a separate model |
| **Vision (images, documents, screenshots, video)** | Bonsai 2 mmproj (Q8_0) | same server | +0.59 GiB | Lazy-load on first image | Video input runs through llama.cpp's mtmd. Vision is the weakest retained area for Bonsai (MMMU-Pro 75.5 vs 81.7) |
| **Fast router / triage / titles / summaries** | Qwen3.5-0.8B (Q8_0, 0.76 GiB) or Qwen3.5-2B (Q4_K_M, 1.19 GiB) | mainline llama-server | ~1.1–1.8 GiB, or CPU | Warm, or CPU | Answers "which tool / which model" when the brain is evicted (e.g., during image gen). Natively multimodal |
| **Web search / research** | Brain + search tool (SearXNG MCP / fetch) + Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B | brain + CPU llama-server (`--embedding`, `--reranking`) | ~0 GPU (CPU) | CPU | Tongyi DeepResearch-30B-A3B (Alibaba, Qwen3-based, Sep 2025) is older than Qwen3.8. Skip unless an A/B test proves otherwise |
| **Memory / RAG over chats and files** | Qwen3-Embedding-0.6B (0.60 GiB Q8) + Qwen3-Reranker-0.6B; multimodal memory: Qwen3-VL-Embedding-2B / VL-Reranker-2B | CPU llama-server or transformers | CPU | CPU | The VL embedder lets you search your own image outputs by text |
| **Image generation + editing (+RGBA, up to 10 references, masks)** | **Qwen-Image-2.1** (7B DiT + Qwen3-VL-8B text encoder) | stable-diffusion.cpp (`leejet/Qwen-Image-2.1-GGUF`), or ComfyUI / AIWF Studio | DiT Q8_0 7.16 GiB / Q6_K 5.59 / Q4_K 3.91 + text encoder (Q4 ≈ 4.7 GiB *est.*, offloaded after conditioning) + VAE 1.26 GiB + activations | **Exclusive** (default) | Research-only license (fine for personal use; flag before commercial use). sd.cpp offload modes `cond_only` / `layer_streaming` help on 16 GB. AIWF Studio already has an sd.cpp lane and a Qwen image editor tab, so we can call it as a tool |
| **Image prompt enhancer** (official) | Qwen-Image-2.1-PE-T2I / PE-I2I (Qwen3.5-VL 9B fine-tunes, emit `{rewritten_prompt, wh_ratio}`) | mainline llama-server (`prithivMLmods/...-GGUF`) | Q4_K_M 5.24 GiB (+0.86 GiB mmproj for I2I) + KV | Transient | **Toggle.** Cheaper alternative: run the brain with PE's shipped `system_prompt.txt` (no swap) |
| **Speech-to-text** | Qwen3-ASR-0.6B (0.75 + 0.20 GiB Q8) / 1.7B (2.02 + 0.33 GiB) + Qwen3-ForcedAligner-0.6B | mainline llama-server (`ggml-org/Qwen3-ASR-*-GGUF`, audio via mtmd) | ~1.2 / ~2.6 GiB | Warm in voice mode | 52 languages/dialects, streaming. The aligner gives word timestamps for subtitles and lip-sync |
| **Text-to-speech** | Qwen3-TTS-12Hz 0.6B / 1.7B: CustomVoice (9 premium timbres + style), VoiceDesign (voice from a description), Base (3-second voice clone) | `qwentts.cpp` GGUF (CUDA), mainline `llama-tts` (Base clone), or the `qwen-tts` Python package | 0.6B Q8 0.92 + tokenizer 0.27 GiB; 1.7B Q8 1.94 + 0.27 GiB | Warm in voice mode | ~97 ms first-audio latency; #1 TTFA on the Open TTS Leaderboard (GGUF port) |
| **Audio understanding** (music, sounds, tone, not just words) | Qwen3-Omni-30B-A3B-Instruct / Captioner (Sep 2025) | mainline llama.cpp (audio input landed 2026-04; marked experimental) | MoE: experts on CPU via `--n-cpu-moe` | Transient | Optional. Its speech *output* path needs transformers/vLLM. Use the ASR → brain → TTS cascade for voice chat instead |
| **Environment simulation** ("run a simulation") | **Qwen-AgentWorld-35B-A3B**: a language world model for 7 domains (MCP tool calls, Search, Terminal, SWE, Android, Web, OS) | mainline llama-server, `unsloth/...-GGUF` UD-Q4_K_M (20.6 GiB) with `--n-cpu-moe` | ~3–4 GiB GPU + KV (20 KiB/token → 2.5 GiB @128K f16); ~17 GiB of experts in RAM | Transient | Dry-run agent actions before executing them, "what-if" sandboxes, fictional worlds, game-mastering. The card advises ≥128K context |
| **Web-page simulation** | WebWorld-8B / 14B (Qwen3-based) | mainline llama-server | 8B Q4 ≈ 5 GiB *est.* | Transient | Predicts the next page state (A11y/HTML/Markdown) for 30+ steps. Useful for browser-agent lookahead. Optional |
| **Gaming** | *No dedicated Qwen gaming model exists.* Brain (logic, NPCs, puzzles), AgentWorld (game-master/world state), brain vision (read game screens; strong OSWorld score) | n/a | n/a | n/a | Implement as a "Game" profile: system prompt + tools + optional AgentWorld simulator |
| **Safety / moderation** (toggle) | Qwen3Guard-Gen-0.6B, or Qwen3Guard-Stream-0.6B (token-level streaming, updated 2026-09) | CPU / transformers | CPU | CPU | Off by default. Useful for screening fetched web content or kid-safe mode |
| **Autonomous-driving scenes** | Qwen-Drive-1.0-4B | n/a | n/a | n/a | **Skip:** it needs BEV sensor inputs and planner heads, which don't fit a chat app |
| **Interpretability** | Qwen-Scope SAEs | n/a | n/a | n/a | **Skip:** research tooling |

### Mainline alternatives for the brain (off by default; for A/B tests)

| Profile | File | Weights | Fits? | Why you'd use it |
|---|---|---|---|---|
| Qwen3.8-27B UD-IQ3_XXS + MTP | `unsloth/Qwen3.8-27B-GGUF` + `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` | 10.18 + 1.28 GiB | Exclusive only (~13.2 GiB with 32K q4 KV) | MTP speculative decoding (+33–127% decode reported on 16 GB cards); mainline tool-call parser; runtime LoRA |
| Qwen3.8-27B + DFlash2 drafter | `z-lab/Qwen3.8-27B-DFlash2-GGUF` Q4_K_M | +1.1 GiB | Exclusive only | Block-diffusion drafter (Inco AI, third-party); acceptance length ~5.3 |
| Qwen3.6-35B-A3B (MoE, 3B active) | community GGUF + `--n-cpu-moe` | ~20 GiB, mostly in RAM | Yes | Very fast, but weaker than Qwen3.8-27B. Fallback "speed" brain |

Out of reach on this hardware: Qwen3.8-Flash-Next, Qwen3.8-2.4T,
Qwen3.5-122B/397B, and Qwen3-Coder-Next-80B. Coder-Next is a stretch at Q3
with experts in RAM, and it is older and weaker than Qwen3.8-27B at coding.

---

## 5. Enhancers and toggles (official ones marked ★)

| Toggle | Applies to | Default | Mechanism |
|---|---|---|---|
| ★ Thinking on/off | brain | On | `chat_template_kwargs.enable_thinking`, or `reasoning_effort:"none"` on Bonsai |
| ★ Reasoning effort | brain | `medium` for chat, `xhigh` for code/logic | `reasoning_effort` (Bonsai: only medium/xhigh are meaningful) |
| ★ Preserve thinking | brain | On (off inside tool loops on Bonsai) | `chat_template_kwargs.preserve_thinking` |
| Reasoning budget | brain | Off | Top-level `thinking_budget_tokens`, or `--reasoning-budget N` |
| ★ MTP speculative decoding | mainline Qwen3.8 / Qwen3.5 only | On when that profile is active | `--spec-type draft-mtp --spec-draft-n-max 2-4 --spec-draft-model mtp-*.gguf -np 1` |
| DFlash2 drafter (third-party) | mainline Qwen3.8-27B | Off | `--spec-type draft-dflash --spec-draft-n-max 7` |
| ★ Image prompt enhancer | image gen/edit | On | PE-T2I / PE-I2I model, or brain + PE system prompt |
| Lightning / turbo LoRAs | image | Off | Community 4–8-step LoRAs (lightx2v for Qwen-Image/Edit-2511; community turbo LoRAs for 2.1) |
| ★ Qwen3Guard | inputs, tool outputs | Off | CPU classifier |
| KV-cache quantization | any llama.cpp model | q8_0 above 32K context | `-ctk/-ctv q8_0` or `q4_0` (avoid q5_0 on the Prism fork) |
| Lazy vision tower | brain | On | Load the mmproj only when an image arrives (`BONSAI_MMPROJ_CPU=1` keeps it in RAM) |
| ★ YaRN long context (up to 1M) | mainline Qwen3.8 | Off | Static YaRN hurts short prompts; enable per session only |
| Cloud fallback | Qwen3.8-Max, Omni-Flash(-Realtime), LiveTranslate | Off | Qwen Cloud OpenAI-compatible API; explicit opt-in, never automatic |

### MCP and agent tooling

- **No first-party Qwen MCP server was found.** Qwen-Agent (`qwen-agent[mcp]`) and the Qwen Code CLI are MCP *clients*. Third-party wrappers such as `mcp-qwen-cli` expose Qwen Code as an MCP server.
- **Plan:** the harness is an MCP client with a per-server toggle. Starter servers: filesystem (sandboxed folder), web search (SearXNG), fetch, a browser/Playwright, a Python code sandbox, and Qwen Code CLI (delegated coding).
- **Built-in tools are also MCP-shaped.** `generate_image`, `edit_image`, `speak`, `transcribe`, `simulate` and `memory_search` follow the same pattern, so the brain sees one uniform tool list.

---

## 6. LoRAs

- **LLM side:** Qwen publishes no official chat LoRAs.
  - **Runtime hot-swap:** mainline llama-server supports runtime LoRA scaling (`GET/POST /lora-adapters`, plus a per-request `lora` field), so persona/style LoRAs can be toggled per request on **mainline** models such as Qwen3.5-9B or Qwen3.8-27B IQ3.
  - **Bonsai:** LoRA is not supported (rotated ternary basis; see §3).
  - **Training:** AIWF Studio's `engines/llm` QLoRA trainer can produce adapters for ≤9B Qwen models on 16 GB. QLoRA on 27B is not practical on this card.
- **Image side:**
  - **Acceleration:** Lightning 4/8-step LoRAs (lightx2v) exist for Qwen-Image, Qwen-Image-2512 and Qwen-Image-Edit-2511; community turbo LoRAs exist for 2.1.
  - **Pipeline:** AIWF Studio's LoRA pipeline and Nunchaku Qwen Lightning sidecar (validated on this exact 16 GB card) are the reference implementation. The harness should call AIWF rather than duplicate it.

---

## 7. Runtimes

| Runtime | Used for | Why |
|---|---|---|
| **PrismML llama.cpp fork** (`llama-server`) | Bonsai 2 27B | The only runtime with PTQ1_0/PQ2_0 + Hadamard kernels. Pin a release; prefer the CUDA 12.4 Windows build |
| **Mainline llama.cpp** (`llama-server`, `llama-tts`) | Qwen3.5 small models, Qwen3-ASR, embeddings/rerankers, AgentWorld, WebWorld, mainline Qwen3.8 + MTP, Qwen3-Omni (audio input) | Has MTP/DFlash speculative decoding, router mode, runtime LoRA, mtmd audio, and slot save/restore |
| **stable-diffusion.cpp** (`sd-server`/`sd-cli`) | Qwen-Image-2.1 | GGUF DiT, built-in offload modes; already integrated in AIWF Studio |
| **qwentts.cpp**, or the `qwen-tts` Python package | Qwen3-TTS (all three modes) | GGUF on CUDA with the lowest TTFA. The Python package is the reference |
| transformers (isolated venv) | PE models, Qwen3Guard-Stream, Qwen-Drive (skip) | Fallback when no GGUF path exists |
| **Not used** | Ollama, LM Studio (cannot load Bonsai 2); vLLM/SGLang (Linux-first, oversized KV pools for a 16 GB single-user box) | |

---

## 8. Managing and queueing models under 16 GB VRAM

### What already exists, and why it isn't enough

- **llama-server router mode** (`--models-preset`, `--models-max`, `POST /models/load|unload`, `--sleep-idle-seconds`) runs each model as a child process.
  - It counts **models, not GiB**: the default `--models-max 4` is unsafe on one GPU.
  - It has a documented race that can load more models than allowed under concurrent requests.
  - It only manages llama.cpp models, and only one build of llama.cpp. Bonsai needs the Prism fork.
- **llama-swap** proxies one front door to many upstream commands, with TTLs and groups (`swap`, `exclusive`, `persistent`). It is also model-count based and knows nothing about sd.cpp/TTS VRAM.

### Our design (implemented in `simple_ai_chat/`; see ARCHITECTURE.md)

1. **Registry.** Every model has a VRAM estimate computed as weights + mmproj + KV(ctx, kv_type) + compute overhead. After the first load, NVML replaces the estimate with a *measured* value, so planning self-calibrates.
2. **Residency classes:**
   - `pinned`: the brain. Evicted only for an exclusive job, then restored.
   - `warm`: kept while room allows; LRU eviction.
   - `transient`: unloaded right after its job.
   - `cpu`: never touches the VRAM budget.
3. **Byte-budget planner.** To load model X it evicts non-busy, non-pinned models in order of (priority, last used) until X fits. Exclusive jobs may also evict pinned models.
4. **Model-affinity queue.** Jobs carry priority and target model. The picker prefers jobs whose model is already resident, and ages waiting jobs so nothing starves. Example: five queued TTS chunks run back to back instead of ping-ponging with an image job.
5. **KV slot save/restore.** Before evicting the brain for an exclusive job, `POST /slots/0?action=save`. After restoring, `action=restore`. The conversation resumes without re-prefilling a long context. Must be verified on the Prism fork with Bonsai's recurrent DeltaNet state.
6. **RAM tier.** With 40 GB RAM, GGUFs stay hot in the OS page cache, so reloads cost ~1–3 s instead of an NVMe read. A background "prefetch" reads the next likely model's file while the GPU is busy.
7. **Hard OOM over silent spill.** Sysmem fallback is disabled (§1). Loads are verified with `/health` plus NVML free memory. On failure, the planner retries with the next fallback (smaller context, quantized KV, lower quant).
8. **Cross-app lock.** AIWF Studio has an in-process `GpuTenantLock`. Simple AI Chat should take a cross-process lease (lock file or localhost endpoint) when it runs exclusive jobs, so the two apps don't fight over the card.

### Example "scenes" (planning numbers; calibrate with NVML)

| Scene | Resident set | Est. VRAM |
|---|---|---|
| **Chat** | Bonsai PTQ1_0 (5.54) + 32K f16 KV (2.0) + compute (1.2) | ~8.7 GiB |
| **Chat + vision** | + mmproj Q8 (0.59) | ~9.3 GiB |
| **Voice** | Chat + vision + ASR-0.6B (~1.25) + TTS-1.7B Q8 (~2.6) | ~13.2 GiB ✓ |
| **Voice + router** | + Qwen3.5-0.8B (~1.1) | ~14.3 GiB (tight; put the router on CPU) |
| **Image (exclusive)** | Qwen-Image-2.1 DiT Q8 (7.16) + VAE (1.26) + activations (~2–3); text encoder offloaded after conditioning | ~11 GiB; brain KV saved, then evicted |
| **Image co-resident** (experimental) | Brain with 8K q8 KV (~6.8) + DiT Q4 (3.91) + VAE + activations | ~14.3 GiB, tight |
| **Simulation** | Brain with 32K q8 KV (~7.8) + AgentWorld GPU share (~3.5) + 128K q8 KV (1.3) + compute (1.0) | ~13.6 GiB ✓, experts in RAM |

---

## 9. Risks and open questions

1. **Bonsai 2 tool-call reliability:** malformed calls are a known open issue. Mitigations: JSON-schema validation, automatic repair, a retry at `medium` effort, and an optional A/B against mainline Qwen3.8-27B IQ3 + MTP.
2. **Prism fork drift:** the fork lags mainline (no MTP for Bonsai, separate binary). Keep both builds pinned by version in config.
3. **Slot save/restore with hybrid attention:** unverified on Bonsai. Fallback is to re-prefill: an estimated ~800–1,500 tok/s on this card (scaled from the RTX 4090's 1,645 / 3,124 tok/s pp512), which is costly at 100K+ tokens.
4. **Qwen-Image-2.1 license:** research-only. Fine for personal use, not for commercial products without an agreement.
5. **Speed numbers for the 4070 Ti SUPER** are extrapolated. The first milestone is a `llama-bench` receipt for each model on the real card, stored under `docs/benchmarks/`.
6. **Windows-specific runtime bugs:** the CUDA 13.3 crash and some CPUs failing to start the CUDA build. The supervisor must surface stderr and offer the CPU or Vulkan build as a diagnostic.
7. **Release-tracker noise:** some aggregator sites list Qwen3.7-Plus, Qwen-Image-3.0, Qwen-Robot and Qwen3.5-Omni-Light as open-weight. None of these appear under the `Qwen/` org on Hugging Face as of this date, so treat them as unverified.

---

## 10. Phased roadmap

| Phase | Deliverable |
|---|---|
| 0 (this PR) | Research, architecture, model registry, VRAM planner, queue, router, llama-server command builder, tests |
| 1 | Process supervisor (spawn, `/health`, NVML measure), Bonsai chat over an OpenAI-compatible API, streaming UI, thinking toggles, benchmark receipts |
| 2 | Tool loop with validator/repair; MCP client + toggles; web search + RAG (CPU embeddings/reranker) |
| 3 | Voice: ASR → brain → streaming TTS; voice-design and clone presets |
| 4 | Image gen/edit via sd.cpp or the AIWF bridge, with prompt-enhancer toggle and exclusive scheduling with KV save/restore |
| 5 | Simulation (AgentWorld) and Game profile; WebWorld lookahead for browser tools |
| 6 | A/B harness: Bonsai vs mainline Qwen3.8 + MTP; LoRA persona hot-swap on mainline models |

---

## Sources

- Qwen3.8-27B model card: <https://huggingface.co/Qwen/Qwen3.8-27B>
- Qwen3.8 GitHub: <https://github.com/QwenLM/Qwen3.8>
- Qwen3.8 blog: <https://qwen.ai/blog?id=qwen3.8>
- Qwen3.8 open weights announcement (The Decoder): <https://the-decoder.com/alibabas-qwen-team-releases-qwen-3-8-models-with-open-weights-under-the-apache-2-0-license/>
- Qwen3.8 lineup overview: <https://codersera.com/blog/qwen-3-8-model-lineup-2026/>
- Ternary Bonsai 2 27B GGUF card: <https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf>
- Known issues: <https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/blob/main/KNOWN_ISSUES.md>
- PrismML docs: <https://docs.prismml.com/bonsai-2-27b>
- PrismML announcement: <https://prismml.com/news/bonsai-2-27b>
- Bonsai-demo (run scripts, SPECULATIVE.md, KV-CACHE.md): <https://github.com/PrismML-Eng/Bonsai-demo>
- MarkTechPost coverage: <https://www.marktechpost.com/2026/09/18/prismml-releases-ternary-bonsai-2-27b-a-5-9-gb-apache-2-0-model-retaining-98-2-of-qwen3-8-27b-performance/>
- Unsloth Qwen3.8-27B GGUF (incl. MTP sidecar): <https://huggingface.co/unsloth/Qwen3.8-27B-GGUF>
- MTP on consumer GPUs: <https://github.com/sudoingX/qwen38-mtp>
- DFlash2 drafter: <https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF>
- 16 GB Qwen3.8 discussion: <https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/discussions/70>
- Qwen-Image-2.1: <https://huggingface.co/Qwen/Qwen-Image-2.1>
- Qwen-Image-2.1 prompt enhancer: <https://huggingface.co/Qwen/Qwen-Image-2.1-PE-T2I>
- Qwen-Image-2.1 sd.cpp GGUF: <https://huggingface.co/leejet/Qwen-Image-2.1-GGUF>
- Qwen-Image-2.1 local guide: <https://openclawdc.com/blog/can-i-run-qwen-image-2-1-locally/>
- Qwen-AgentWorld-35B-A3B: <https://huggingface.co/Qwen/Qwen-AgentWorld-35B-A3B>
- AgentWorld GGUF: <https://huggingface.co/unsloth/Qwen-AgentWorld-35B-A3B-GGUF>
- WebWorld: <https://huggingface.co/Qwen/WebWorld-8B>
- Qwen-Drive-1.0: <https://huggingface.co/Qwen/Qwen-Drive-1.0-4B>
- Qwen3-ASR: <https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf>
- Qwen3-ASR GGUF: <https://huggingface.co/ggml-org/Qwen3-ASR-0.6B-GGUF>
- Qwen3-TTS: <https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice>
- Qwen3-TTS GGUF: <https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF>
- llama.cpp Qwen3-TTS PR: <https://github.com/ggml-org/llama.cpp/pull/26254>
- Qwen3-Omni: <https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct>
- llama.cpp audio support: <https://www.creativeainews.com/blog/llama-cpp-audio-qwen3-omni-multimodal/>
- Embeddings: <https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF>
- VL embeddings: <https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B>
- Rerankers: <https://huggingface.co/Qwen/Qwen3-Reranker-0.6B>
- Qwen3Guard: <https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B>
- llama-server README (router, LoRA, spec types, slots): <https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md>
- Router-mode VRAM traps: <https://runaihome.com/blog/llama-server-router-mode-multi-model-setup-2026/>
- llama-swap: <https://betterstack.com/community/guides/ai/llama-swap/>
- Qwen-Agent MCP: <https://mcpservers.org/ja/servers/QwenLM/Qwen-Agent>
- Qwen Code: <https://howaiworks.ai/ai-tools/qwen-code>
- Qwen3.5-Omni coverage: <https://rits.shanghai.nyu.edu/ai/qwen3-5-omni-alibabas-omnimodal-ai-speaks-36-languages-and-codes-from-voice>
- Release tracker (partly unverified, see §9): <https://releasebot.io/updates/qwen>
- Tongyi DeepResearch: <https://openrouter.ai/alibaba/tongyi-deepresearch-30b-a3b:free>
