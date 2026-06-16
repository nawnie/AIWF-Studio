# Ollama Chat Tenant

This document describes how the Chat tab integrates with the GPU tenant system.

---

## Architecture

```text
User clicks Chat tab
  -> chat_workspace.py
  -> OllamaClient.healthcheck()
  -> OllamaClient.list_models()

User presses Send
  -> on_send()
  -> EngineSupervisor.request_switch(CHAT, job_id="chat")
      - if GPU is idle: grant CHAT
      - if CHAT already owns the lock for the same job: no-op
      - if image/video/enhance/training owns the GPU: deny unless the caller opted into waiting
  -> OllamaClient.stream_chat(model, messages)
  -> stream tokens to gr.Chatbot
```

---

## GPU Tenant Rules

| Situation | Behavior |
|-----------|----------|
| GPU is IDLE | `request_switch(CHAT, job_id="chat")` succeeds immediately. |
| GPU is held by IMAGE, VIDEO, ENHANCE, or TRAINING | Chat is denied by default and the UI shows the busy message. |
| GPU is already in CHAT for the same job | The request is idempotent and succeeds. |
| User clicks Unload | The tab calls `client.unload(model)` and releases with `request_switch(IDLE, job_id="chat")`. |
| A GPU-heavy tenant starts while CHAT is active | The supervisor asks Ollama to unload the active chat model before granting the heavy tenant. |

Chat is classified as not GPU-heavy (`EngineTenant.CHAT.is_gpu_heavy() == False`) because Ollama manages its own VRAM independently of the diffusers pipeline. The tenant lock still tracks CHAT so GPU-heavy work has a single place to preempt and unload it.

---

## OllamaClient

`aiwf/services/ollama_client.py` is a thin `httpx` wrapper.

```python
client = OllamaClient(base_url="http://127.0.0.1:11434")

alive = client.healthcheck()
models = client.list_models()

for token in client.stream_chat(model, messages, options):
    print(token, end="", flush=True)

client.unload("llama3:8b")  # keep_alive: 0
info = client.model_info("llama3:8b")
```

`httpx` is imported on first use. If it is missing, `OllamaClient` raises `ImportError` with an install hint instead of breaking app import.

---

## EngineSupervisor Wiring

| Member | Description |
|--------|-------------|
| `active_tenant` | Returns the current `EngineTenant`. |
| `request_switch(EngineSwitchRequest)` | Acquires, waits for, denies, or releases GPU ownership. |
| `tenant_session(target, job_id=...)` | Context manager for service-level GPU ownership. |
| `borrow_active_tenant(target, job_id=...)` | Lets same-thread nested post-processing reuse an already-held tenant without releasing it. |
| `set_ollama_client(client)` | Installs the client used to evict chat models. |
| `set_chat_model(name)` | Tracks which model is loaded so it can be unloaded before heavy work. |

The supervisor is injected via `AppContext`; the Chat tab accesses it as `ctx.supervisor`.

---

## Ollama Not Installed

If Ollama is not reachable:

- `healthcheck()` returns `False`
- the status pill shows Ollama is not detected
- the tab renders guidance instead of crashing
- no exception is propagated to app startup

---

## VRAM Behavior

Ollama keeps models in VRAM until they are explicitly unloaded or `keep_alive` expires.

To free VRAM before a video/image/enhance/training job:

1. The supervisor calls `client.unload(active_model)` when a higher-priority tenant preempts CHAT.
2. Ollama receives `POST /api/generate {"model": "...", "keep_alive": 0}`.
3. The model is evicted before the GPU-heavy tenant starts.

This happens automatically; the user does not need to click Unload before starting a video or image job.

---

## Security Note

`OllamaClient` connects to `127.0.0.1` by default. Do not expose the Ollama port to an external network without authentication.

---

## Future Work

- System prompt persistence per model
- Chat history export
- Model pull UI with progress and cancel support
