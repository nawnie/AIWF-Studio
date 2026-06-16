# Local Tool Security Policy

**Scope:** All tools exposed to an embedded assistant (Agent MoK or compatible) operating within AIWF Studio.  
**Status:** Normative — implementation must conform to this document.

---

## Threat Model

An embedded assistant receives user-typed natural language and translates it into structured tool calls. The risk surface is:

1. **Prompt injection** — malicious content in filenames, model metadata, or prompt library entries causes the assistant to take destructive actions.
2. **Privilege escalation** — a tool intended for reading metadata is abused to execute arbitrary code.
3. **Silent side effects** — GPU-heavy or irreversible operations triggered without user awareness.
4. **Path traversal** — a crafted filename escapes the expected root directory.

The policy below addresses each of these.

---

## Hard Rules (never violate, no exceptions)

### R1 — No raw shell command tool

No tool accepts a string that is executed as a shell command (`subprocess.run(..., shell=True)` or equivalent). All subprocess calls use `shell=False` with an explicit `list[str]` argument.

**Why:** A single unconstrained shell tool gives an injected payload complete system access.

### R2 — No unconstrained file write

Tools may not write to arbitrary filesystem paths. The only permitted write target is the configured `output_dir` (from `AppContext`). Creating or overwriting files outside `output_dir` is forbidden.

**Why:** Prevents overwriting model checkpoints, config files, or system files.

### R3 — No file deletion

No tool exposed to the assistant may delete files. Deletion operations remain UI-only.

**Why:** Irreversible. LoRA and checkpoint files are expensive to recreate.

### R4 — No arbitrary Python execution

No tool accepts a Python expression or code string for `eval`/`exec`. Dynamic code paths are not permitted even for "convenience" features.

**Why:** Direct code execution bypass for all other rules.

### R5 — No unconfirmed training launch

Starting a LoRA training run, full-model training run, or dataset processing job requires explicit user confirmation through the UI. An assistant may compose and preview a training command but may not submit it autonomously.

**Why:** Training runs occupy the GPU for hours and may corrupt datasets if misconfigured.

### R6 — GPU lock check before generation

Any tool that would submit a generation request (image or video) must check the GPU tenant lock. If the lock is held by a non-idle tenant, the tool returns an error — it does not queue silently or wait.

**Why:** Silent queuing hides resource contention from the user.

---

## Path Safety Rules

### P1 — Resolve through AppContext only

All filesystem paths must be constructed from roots provided by `AppContext` (e.g., `ctx.checkpoint_dir`, `ctx.lora_dir`, `ctx.output_dir`). No hardcoded user-home paths.

### P2 — Resolve and validate before use

Before opening any file passed as a tool argument:

```python
resolved = (root / user_input).resolve()
if not resolved.is_relative_to(root.resolve()):
    raise PermissionError(f"Path escapes allowed root: {resolved}")
```

This prevents `../../../etc/passwd`-style traversal.

### P3 — Extension allowlist for read tools

File-reading tools only open files with permitted extensions:

| Tool | Allowed extensions |
|------|--------------------|
| `list_local_checkpoints` | `.safetensors`, `.ckpt` |
| `list_local_loras` | `.safetensors`, `.pt` |
| `read_safetensors_metadata` | `.safetensors` |
| `inspect_prompt_library` | `.json`, `.yaml`, `.txt` |

Files with other extensions are silently skipped during listing and rejected during direct reads.

---

## Metadata Read Safety

### M1 — Header-only safetensors reads

Reading `.safetensors` metadata must parse **only** the JSON header (first 8-byte length prefix + N bytes). Weight tensors must never be mapped into memory for a metadata-only operation.

### M2 — Header size cap

Reject any safetensors file whose declared header length exceeds 10 MB. Legitimate metadata headers are measured in kilobytes.

### M3 — Sanitize returned strings

Metadata values returned from tool calls must be treated as untrusted strings by the caller. The tool layer must not interpret metadata values as commands, paths, or code.

---

## Confirmation Requirements

| Action | Requires confirmation |
|--------|-----------------------|
| List checkpoints / LoRAs | No |
| Read safetensors metadata | No |
| Build prompt draft | No |
| Recommend settings | No |
| Submit generation request (GPU idle) | Optional — configurable per user |
| Submit generation request (GPU busy) | Blocked — no confirmation path |
| Start training run | Yes — always |
| Delete any file | Blocked — no tool exists |
| Write outside output_dir | Blocked — no tool exists |

---

## Audit Logging

Every tool call made by the embedded assistant is appended to a rotating log at `output_dir/.agent_tool_log.jsonl`. Each entry records:

```json
{
  "ts": "<ISO-8601>",
  "tool": "<tool_name>",
  "args": { "<sanitized args>" },
  "result_summary": "<ok | error: message>",
  "gpu_tenant_at_call": "<idle | image | video | ...>"
}
```

Log entries are written even if the call is rejected (rule violation). This provides a post-incident audit trail without exposing full response payloads.

---

## Change Process

Changes to this document require a code review that includes:
1. Updated tests demonstrating the rule is enforced.
2. Review of every tool in `aiwf/services/prompt_tools.py` for conformance.

No tool may be merged that contradicts a Hard Rule, even if marked experimental.
