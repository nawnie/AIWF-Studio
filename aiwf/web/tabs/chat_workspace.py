"""
aiwf/web/tabs/chat_workspace.py

Chat tab — powered by a locally-running Ollama instance.

Architecture
------------
* Tab select → health-check Ollama, refresh model list.
* Send → acquire GPU lock (CHAT tenant), stream tokens into the chatbot.
* Unload → call client.unload(model) and release tenant.
* Ollama not running → friendly guidance, no crash.

Rules
-----
* No torch imports here.
* Services only — no subprocess calls in this file.
* GPU lock acquired through EngineSupervisor.request_switch().
"""
from __future__ import annotations

import logging
import threading
from typing import Generator

import gradio as gr

from aiwf.bootstrap import AppContext
from aiwf.core.domain.engine import EngineSwitchRequest, EngineTenant
from aiwf.services.ollama_client import OllamaClient
from aiwf.web.registry import WebRegistry

logger = logging.getLogger(__name__)

# Module-level client — one per process, shared across calls.
# AppContext may inject its own; this is the default.
_client: OllamaClient | None = None
_client_lock = threading.Lock()


def _get_client(base_url: str = "http://127.0.0.1:11434") -> OllamaClient:
    global _client
    if _client is None or _client._base_url != base_url.rstrip("/"):
        with _client_lock:
            if _client is None or _client._base_url != base_url.rstrip("/"):
                _client = OllamaClient(base_url=base_url)
    return _client


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

_STATUS_OK  = "🟢 Ollama connected"
_STATUS_ERR = "🔴 Ollama not detected"

_INSTALL_NOTE = (
    "**Ollama is not running.**\n\n"
    "To use the chat tab:\n"
    "1. Download Ollama from [ollama.com](https://ollama.com)\n"
    "2. Install and run it (`ollama serve`)\n"
    "3. Pull a model: `ollama pull llama3:8b`\n"
    "4. Click **Refresh** above\n\n"
    "Chat and image/video share the GPU — switching tabs "
    "automatically unloads the current model."
)


def _check_ollama(client: OllamaClient) -> tuple[bool, list[str]]:
    """Return (is_alive, model_names)."""
    if not client.healthcheck():
        return False, []
    return True, client.list_models()


# ---------------------------------------------------------------------------
# Tab registration
# ---------------------------------------------------------------------------

def register_chat_workspace(registry: WebRegistry) -> None:

    @registry.tab("Chat", order=15)
    def build(ctx: AppContext, tab: gr.Tab | None = None) -> None:
        client = _get_client()

        # ---- initial probe ----
        alive, models = _check_ollama(client)
        status_text = _STATUS_OK if alive else _STATUS_ERR

        with gr.Row():
            status_pill = gr.Markdown(value=status_text, elem_id="ollama-status")
            refresh_btn = gr.Button("⟳ Refresh", variant="secondary", scale=0)

        model_dd = gr.Dropdown(
            choices=models,
            value=models[0] if models else None,
            label="Model",
            interactive=True,
        )

        install_note = gr.Markdown(
            value=_INSTALL_NOTE if not alive else "",
            visible=not alive,
        )

        chatbot = gr.Chatbot(
            label="Chat",
            height=480,
            show_copy_button=True,
            type="messages",
        )

        with gr.Row():
            msg_box = gr.Textbox(
                placeholder="Type a message and press Enter or Send…",
                show_label=False,
                scale=8,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)

        with gr.Row():
            clear_btn  = gr.Button("Clear", variant="secondary")
            unload_btn = gr.Button("Unload model", variant="stop")

        # System-prompt accordion (collapsed by default)
        with gr.Accordion("System prompt", open=False):
            system_prompt = gr.Textbox(
                placeholder="Optional system prompt…",
                lines=3,
                show_label=False,
            )

        # ----------------------------------------------------------------
        # Callbacks
        # ----------------------------------------------------------------

        def on_refresh():
            nonlocal alive, models
            alive, models = _check_ollama(client)
            return (
                gr.update(value=_STATUS_OK if alive else _STATUS_ERR),
                gr.update(choices=models, value=models[0] if models else None),
                gr.update(value=_INSTALL_NOTE if not alive else "", visible=not alive),
            )

        refresh_btn.click(
            fn=on_refresh,
            outputs=[status_pill, model_dd, install_note],
        )

        # Trigger refresh when the tab is selected (if tab is wired up)
        if tab is not None:
            tab.select(fn=on_refresh, outputs=[status_pill, model_dd, install_note])

        def on_send(user_msg: str, history: list, model: str, sys_prompt: str):
            """Stream a chat response token by token."""
            if not user_msg.strip():
                yield history, gr.update(value="")
                return

            if not model:
                history = history + [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": "⚠️ No model selected. Choose a model from the dropdown."},
                ]
                yield history, gr.update(value="")
                return

            # Acquire CHAT tenant via supervisor
            supervisor = getattr(ctx, "supervisor", None)
            if supervisor is not None:
                result = supervisor.request_switch(
                    EngineSwitchRequest(target=EngineTenant.CHAT, reason="Chat send")
                )
                if not result.ok:
                    history = history + [
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": f"⚠️ GPU busy: {result.message}"},
                    ]
                    yield history, gr.update(value="")
                    return
                supervisor.set_chat_model(model)

            # Build message list
            messages: list[dict] = []
            if sys_prompt.strip():
                messages.append({"role": "system", "content": sys_prompt.strip()})
            for entry in history:
                messages.append(entry)
            messages.append({"role": "user", "content": user_msg})

            # Add user message to history immediately
            history = history + [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": ""},
            ]
            yield history, gr.update(value="")

            # Stream tokens
            assistant_text = ""
            try:
                for token in client.stream_chat(model, messages):
                    assistant_text += token
                    history[-1] = {"role": "assistant", "content": assistant_text}
                    yield history, gr.update(value="")
            except Exception as exc:
                logger.exception("[Chat] stream_chat error")
                history[-1] = {
                    "role": "assistant",
                    "content": f"⚠️ Error: {exc}",
                }
                yield history, gr.update(value="")

        send_btn.click(
            fn=on_send,
            inputs=[msg_box, chatbot, model_dd, system_prompt],
            outputs=[chatbot, msg_box],
        )
        msg_box.submit(
            fn=on_send,
            inputs=[msg_box, chatbot, model_dd, system_prompt],
            outputs=[chatbot, msg_box],
        )

        def on_clear():
            return []

        clear_btn.click(fn=on_clear, outputs=[chatbot])

        def on_unload(model: str):
            if not model:
                return gr.update()
            ok = client.unload(model)
            supervisor = getattr(ctx, "supervisor", None)
            if supervisor is not None and ok:
                supervisor.request_switch(
                    EngineSwitchRequest(target=EngineTenant.IDLE, reason="Chat model unloaded")
                )
            return gr.update(value=f"{'✅ Unloaded' if ok else '⚠️ Unload failed'}: {model}")

        unload_btn.click(
            fn=on_unload,
            inputs=[model_dd],
            outputs=[status_pill],
        )
