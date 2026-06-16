# Full-Learning Research Agent — Side Project Plan

A standalone autonomous research agent that implements the full-learning algorithm. Given a topic, it returns a comprehensive knowledge document built through multi-pass research, source verification, claim challenging, and recursive back-propagation.

---

## What it is

A Python agent loop that takes a research prompt and runs the 12-step full-learning protocol autonomously. Unlike a chatbot session, it persists state across steps, runs searches in parallel where possible, and produces a structured output document — not a conversation.

---

## Core Architecture

```
User prompt
     │
     ▼
┌─────────────────────────────────────────┐
│              Orchestrator               │
│  (manages state, runs phase loop)       │
└────────────┬────────────────────────────┘
             │
     ┌───────┼────────────┐
     ▼       ▼            ▼
  Planner  Researcher  Challenger
  (map +   (curated +  (claim check,
  threads)  web search) contradiction)
             │
     ┌───────┼───────┐
     ▼       ▼       ▼
  HF API  GH API  Web Search
  arXiv   CivitAI  (Brave/SERP)
```

---

## State Schema

All state lives in a single `ResearchState` object that gets checkpointed to disk after each phase. This lets the agent resume if it crashes mid-run.

```python
@dataclass
class ResearchState:
    prompt: str
    subject_map: SubjectMap | None = None
    threads: list[LearningThread] = field(default_factory=list)
    phase: int = 0                    # current phase (0–5)
    reread_pass: int = 0              # which reread pass we're on
    created_at: str = ""
    checkpoint_path: str = ""

@dataclass
class LearningThread:
    question: str
    draft: str | None = None
    claims: list[str] = field(default_factory=list)
    curated_evidence: list[Evidence] = field(default_factory=list)
    web_evidence: list[Evidence] = field(default_factory=list)
    challenged: str | None = None
    rebuilt: str | None = None
    verified: str | None = None       # final answer for this thread
    changed_in_reread: bool = False

@dataclass
class Evidence:
    url: str
    source_type: str                  # maps to SOURCE_TRUST
    trust: float
    excerpt: str
    retrieved_at: str
```

---

## Module Breakdown

### `agent/orchestrator.py`
- Entry point: `run(prompt) -> ResearchOutput`
- Loads or creates `ResearchState`
- Calls phases in order, checkpointing after each
- Handles resume from checkpoint

### `agent/planner.py`
- `map_subject(prompt) -> SubjectMap`
- `decompose_threads(subject_map) -> list[LearningThread]`
- Uses LLM call to produce structured JSON output

### `agent/researcher.py`
- `draft_answer(thread) -> str`
- `extract_claims(draft) -> list[str]`
- `search_curated(thread, claims) -> list[Evidence]`
- `search_web(thread, rebuilt) -> list[Evidence]`
- Searches run in parallel via `asyncio.gather`

### `agent/challenger.py`
- `challenge_claims(draft, evidence) -> ChallengeReport`
- `rebuild_answer(thread, challenge, evidence) -> str`
- `patch_answer(rebuilt, web_evidence) -> str`
- `is_improved(old, new) -> bool`   ← used in reread loop

### `agent/rereader.py`
- `reread_pass(threads, full_context) -> list[LearningThread]`
- Runs up to 3 passes; short-circuits if nothing changed
- Produces `changed_in_reread` flag per thread

### `agent/compiler.py`
- `compile(state) -> ResearchOutput`
- `final_qc(output, state) -> ResearchOutput`

### `agent/sources.py`
- Search client wrappers per source
- Each returns `list[Evidence]`
- Trust scores applied here

### `agent/llm.py`
- Thin wrapper around LLM API (Anthropic by default)
- Structured output helpers (JSON mode)
- Token budget tracking

---

## Source Clients to Implement (Priority Order)

| Client | API | Notes |
|--------|-----|-------|
| Web search | Brave Search API or SerpAPI | General + recent |
| HuggingFace | `huggingface_hub` | Model cards, datasets |
| GitHub | GitHub REST API | READMEs, issues, code |
| arXiv | `arxiv` Python lib | Papers |
| PapersWithCode | Unofficial API / scrape | Benchmarks |
| CivitAI | `/api/v1/models` | Already in AIWF |
| Replicate | `replicate` Python lib | Model pages |

Start with web search + HuggingFace. That covers ~80% of AI research topics.

---

## Output Format

```python
@dataclass
class ResearchOutput:
    prompt: str
    direct_answer: str
    subject_map: str              # prose description of the territory
    key_findings: list[str]
    corrected_mistakes: list[str] # what the naive understanding gets wrong
    unknowns: list[str]
    contradictions: list[str]
    open_questions: list[str]
    practical_next_steps: list[str]
    threads: list[LearningThread] # full detail if needed
    sources: list[Evidence]
    generated_at: str
    reread_passes_run: int
    total_threads: int
```

Saved as both `output.json` (machine-readable) and `output.md` (human-readable primer).

---

## Tech Stack

| Layer | Choice | Reason |
|-------|--------|--------|
| LLM | Anthropic Claude API | Structured output, long context |
| Async | `asyncio` + `httpx` | Parallel searches |
| State | JSON checkpoints | Simple, no DB dependency |
| CLI | `click` | Quick to wire up |
| Config | `.env` + `python-dotenv` | API keys |
| Packaging | `pyproject.toml` | Modern Python |

Optional later: streaming output, web UI (Gradio or FastAPI), vector store for evidence retrieval across sessions.

---

## Phased Build Plan

### Phase 1 — Core loop, no real searches (Week 1)
- `orchestrator.py` with full 12-step loop
- `planner.py` (LLM-backed)
- `researcher.py` with mock search (returns static evidence)
- `challenger.py` + `rereader.py` + `compiler.py`
- End-to-end test: given a prompt, produces a `ResearchOutput`

### Phase 2 — Real source clients (Week 2)
- Brave/SerpAPI web search client
- HuggingFace client
- arXiv client
- Replace mocks with real calls
- Parallel search with `asyncio.gather`

### Phase 3 — Checkpointing + resume (Week 3)
- Checkpoint to `~/.research-agent/runs/<id>/state.json` after each phase
- `agent resume <id>` CLI command
- Token usage tracking + budget guard

### Phase 4 — Quality (Week 4)
- GitHub + PapersWithCode clients
- Better `is_improved()` — use LLM judge, not string diff
- Source deduplication
- Output quality eval (run on 5 known topics, compare to ground truth)

---

## Repository Layout

```
research-agent/
├── agent/
│   ├── __init__.py
│   ├── orchestrator.py
│   ├── planner.py
│   ├── researcher.py
│   ├── challenger.py
│   ├── rereader.py
│   ├── compiler.py
│   ├── sources/
│   │   ├── __init__.py
│   │   ├── web.py
│   │   ├── huggingface.py
│   │   ├── arxiv.py
│   │   ├── github.py
│   │   └── civitai.py
│   ├── llm.py
│   └── models.py        # dataclasses
├── cli.py               # click entry points
├── tests/
├── pyproject.toml
├── .env.example
└── README.md
```

---

## CLI Interface

```bash
# Run a research session
research-agent run "How does ControlNet condition diffusion models?"

# Resume a previous run
research-agent resume <run-id>

# List runs
research-agent list

# Export a run to markdown
research-agent export <run-id> --format md
```

---

## Key Design Decisions

**Why checkpointing?** A full run touches 5–12 threads × 7 steps each. At ~15s per LLM call that's 5–15 minutes. Crashes happen.

**Why asyncio for searches?** Step 5 (curated search) hits 7+ sources per thread. Sequential would be 2–3× slower. Most of the time is network I/O.

**Why JSON output + MD output?** Machine-readable output lets you pipe runs into downstream tools (AIWF Studio, a vector store, a diff tool for re-runs on the same topic). The MD is for humans.

**Why LLM judge for `is_improved()`?** String diff catches changes but not whether they're improvements. A judge prompt comparing old vs. new answer on dimensions like accuracy, completeness, and clarity is more reliable.

---

## First PR Target

Get a full end-to-end run working on this prompt:

> "How does LoRA fine-tuning work and when should I use it vs. full fine-tuning?"

Expected output: ~2000-word primer covering the math, the implementation, the tradeoffs, the failure modes, and the community best practices — with corrected myths (e.g. "LoRA is not always faster than full fine-tuning").

This is a good calibration target because the answer is verifiable, the community has strong opinions that differ from papers, and there are active debates (rank selection, alpha scaling, layer targeting).
