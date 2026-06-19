# Agentic Prompt / Tool Workspace â€” Phase 7 Roadmap

**Status:** In Progress
**Phase:** 7 (depends on stable engine boundaries from Phases 0â€“6)

---

## Goal

Add a structured local-tool layer that lets an embedded assistant (Claude or compatible model) inspect the user's local checkpoint/LoRA library, build prompts, recommend generation settings, and submit requests â€” all without unsafe shell access, file mutations, or unconfirmed GPU-heavy operations.

This is an *additive* layer. It does not change the existing engine, service, or UI architecture.

---

## Principles

1. **Read-mostly.** The tooling surface is almost entirely read operations: listing files, reading metadata, querying active config. Write operations require explicit confirmation.
2. **No shell passthrough.** No tool accepts a raw shell command string.
3. **Paths from context only.** All path resolution goes through `AppContext`. No hardcoded user paths.
4. **GPU lock respected.** Tools that would trigger generation check the GPU tenant lock and refuse (not queue silently) if another tenant is active.
5. **Training never auto-triggered.** Starting a LoRA or full training run requires a human-in-the-loop confirmation step even if the assistant composes the full command.
6. **Security doc is normative.** `docs/LOCAL_TOOL_SECURITY.md` is the authoritative reference for what any tool in this workspace may and may not do.

---

## Tool Inventory

### Tier 1 â€” Pure Read (safe to call at any time)

| Tool | Description |
|------|-------------|
| `list_local_checkpoints` | Scan checkpoint root, return name + path + size + mtime for each `.safetensors` / `.ckpt` file |
| `list_local_loras` | Same scan for the LoRA directory |
| `read_safetensors_metadata` | Parse the JSON metadata header from a `.safetensors` file (no weight loading) |
| `inspect_prompt_library` | List available styles and prompt templates from `aiwf/data/` or user-configured paths |
| `get_current_settings_snapshot` | Return the current resolved `UserSettings` as a dict (read-only) |

### Tier 2 â€” Compute / Compose (CPU-only, no GPU)

| Tool | Description |
|------|-------------|
| `build_prompt_draft` | Assemble a generation prompt from style, subject, negative, and LoRA activation tags |
| `recommend_settings` | Given a checkpoint + goal (speed/quality/video), return suggested width/height/steps/cfg |
| `generate_workflow_json` | Produce a ComfyUI-compatible workflow JSON blob from a generation request struct |

### Tier 3 â€” Side-Effect (require confirmation or GPU-lock check)

| Tool | Description |
|------|-------------|
| `submit_generation_request` | POST a generation request to the Studio API; blocked if GPU tenant lock is held by non-idle tenant |
| `report_output_path` | Given a job ID, return the output directory path (read-only filesystem lookup) |

Training launch is **not exposed** as an agent tool. The UI remains the only trigger for training.

---

## Implementation Plan

### Step 1 â€” Service layer (`aiwf/services/prompt_tools.py`)

Pure-Python service, no torch imports at module level. All file I/O through `pathlib`. See module docstring for contract details.

### Step 2 â€” Safetensors metadata reader

Parse the 8-byte length prefix + UTF-8 JSON header only. Do **not** `import safetensors` or map weights into memory for metadata-only reads.

```python
def _read_safetensors_header(path: Path) -> dict:
    with path.open("rb") as f:
        length = int.from_bytes(f.read(8), "little")
        raw = f.read(length)
    return json.loads(raw)
```

### Step 3 â€” Prompt builder

`build_prompt_draft(subject, style_name, lora_names, negative)` â€” concatenates with correct weight syntax and LoRA activation tags. Pure string operations.

### Step 4 â€” Settings recommender

Rule-based lookup table keyed on `(base_model_class, goal)`. No ML inference. Returns a `RecommendedSettings` dataclass.

### Step 5 â€” Tests

`tests/test_prompt_tools.py` â€” covers all Tier 1 and Tier 2 tools. Uses `tmp_path` fixtures to mock checkpoint directories; no GPU required.

### Step 6 â€” UI integration (future sprint)

A collapsible "Prompt Tools" accordion in the Studio tab or a dedicated lightweight tab. Out of scope for this sprint â€” service layer ships first, UI wired after tests pass.

---

## Dependencies

| Dependency | Notes |
|------------|-------|
| `pathlib` | stdlib â€” always present |
| `json` | stdlib â€” always present |
| `struct` | stdlib â€” for safetensors header parse |
| `safetensors` (optional) | Only for full metadata; fallback to manual header parse if absent |
| `aiwf.core.context.AppContext` | Required â€” provides all root paths |
| `aiwf.core.config.settings.UserSettings` | Read-only access for snapshot tool |

No torch, diffusers, gradio, or heavy ML packages imported at module level.

---

## Out of Scope (Phase 7)

- Agent-to-agent orchestration
- Remote model APIs (only local endpoints)
- Automatic prompt optimization / fine-tuning loop
- Video generation via agent (GPU lock complexity deferred)
- Filesystem writes beyond `output_dir` (blocked by security rules)
