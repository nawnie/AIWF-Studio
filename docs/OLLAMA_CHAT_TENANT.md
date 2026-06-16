# Ollama Chat Tenant

This document describes how the Chat tab integrates with the GPU tenant system.

---

## Architecture

```
User clicks Chat tab
        │
        ▼
chat_workspace.py  ──► OllamaClient.healthcheck()
        │                OllamaClient.list_models()
        │
User presses Send
        │
        ▼
on_send() ──► EngineSupervisor.request_switch(CHAT)
        │           │
        │           ├─ If current == CHAT → no-op
        │           └─ If current == IMAGE/VIDEO/TRAINING
        │                 → refuse (GPU busy)
        │
        ▼
OllamaClient.stream_chat(model, messages)
        │
        ▼
Tokens streamed to gr.Chatbot
```

---

## GPU tenant rules

| Situation | Behaviour |
|-----------|-----------|
| GPU is IDLE | `request_switch(CHAT)` succeeds immediately |
| GPU is held by IMAGE or VIDEO | `request_switch(CHAT)` returns `ok=False`; chat shows "GPU busy" message |
| GPU is already in CHAT | `request_switch(CHAT)` is a no-op (returns ok=True, "already active") |
| User clicks Unload | `client.unload(model)` + `request_switch(IDLE)` |
| User switches to Video tab | Video service calls `request_switch(VIDEO)`; supervisor calls `client.unload(active_model)` first |

Chat is classified as **not** GPU-heavy (`EngineTenant.CHAT.is_gpu_heavy() == False`) because Ollama manages its own VRAM independently of the diffusers pipeline.  However, the tenant lock still prevents accidental conflicts.

---

## OllamaClient

`aiwf/services/ollama_client.py` — thin `httpx` wrapper.

```python
client = OllamaClient(base_url="http://127.0.0.1:11434")

# Health check (fast, <1s)
alive = client.healthcheck()       # bool

# Available models
models = client.list_models()      # list[str]

# Stream a chat response
for token in client.stream_chat(model, messages, options):
    print(token, end="", flush=True)

# Evict from VRAM before switching to image/video
client.unload("llama3:8b")         # keep_alive: 0

# Show model metadata
info = client.model_info("llama3:8b")
```

`httpx` must be installed (`pip install httpx`).  If missing, `OllamaClient` raises `ImportError` with an install hint on first use.

---

## EngineSupervisor wiring

`EngineSupervisor` gains three new members (Sprint A1):

| Member | Description |
|--------|-------------|
| `active_tenant` property | Returns current `EngineTenant` |
| `request_switch(EngineSwitchRequest)` | Switch tenant; unloads Ollama on CHAT→* transitions |
| `set_chat_model(name)` | Track which model is loaded so we can unload it later |

The supervisor is injected via `AppContext` — the chat tab accesses it as `ctx.supervisor`.

---

## Ollama not installed

If Ollama is not reachable:
- `healthcheck()` returns `False`
- The status pill shows 🔴 Ollama not detected
- An install guide appears in the tab body
- No crash, no exception propagated to the UI

---

## VRAM behaviour

Ollama keeps models in VRAM until they are explicitly unloaded or `keep_alive` expires (default 5 minutes).

To free VRAM before a video/image job:
1. The supervisor calls `client.unload(active_model)` in `request_switch()` when switching away from CHAT.
2. Ollama receives `POST /api/generate {"model": "...", "keep_alive": 0}`.
3. The model is evicted; VRAM is returned to the OS/CUDA allocator.

This happens automatically — the user does not need to click Unload before starting a video job.

---

## Security note

`OllamaClient` only connects to `127.0.0.1` (loopback) by default.  Do not expose the Ollama port to an external network without authentication.

---

## Future work

- `allow_wait=True` in `EngineSwitchRequest` — queue chat requests while GPU is busy
- System prompt persistence (per-model)
- Chat history export
- Model pull UI (progress bar, cancel)
