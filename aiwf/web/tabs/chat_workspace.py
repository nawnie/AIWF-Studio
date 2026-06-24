"""Chat tab with optional Ollama backend and Atlas-style RAG activation."""
from __future__ import annotations

import json
import logging
import threading
import uuid
from pathlib import Path

import gradio as gr

from aiwf.bootstrap import AppContext
from aiwf.core.domain.engine import EngineSwitchRequest, EngineTenant
from aiwf.services.chat_agent_tools import ChatAgentToolService, run_agentic_turn
from aiwf.services.chat_atlas_rag import format_cards_for_prompt, retrieve_atlas_cards
from aiwf.services.ollama_client import OllamaClient
from aiwf.services.training.dataset_validator import DatasetValidator
from aiwf.web.registry import WebRegistry

logger = logging.getLogger(__name__)

_DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
_client: OllamaClient | None = None
_client_lock = threading.Lock()
_validator = DatasetValidator()

_STATUS_OK = "Ollama connected"
_STATUS_ERR = "Ollama not detected"
_INSTALL_NOTE = (
    "**Chat backend required.** Install and start Ollama, then pull at least one model. "
    "AIWF connects to a local Ollama server and does not install it automatically."
)
_BACKEND_SETUP = (
    "**Backend setup**\n\n"
    "1. Install Ollama from `https://ollama.com/download`.\n"
    "2. Start Ollama.\n"
    "3. Pull a model, for example `ollama pull qwen2.5-coder:3b`.\n"
    "4. Return here and click Refresh."
)


def _format_chat_signals(*, alive: bool, models: list[str], trainer_ready: bool) -> str:
    backend = "online" if alive else "offline"
    trainer = "ready" if trainer_ready else "not ready"
    model_count = len(models)
    return f"**Backend:** {backend}  \n**Models:** {model_count}  \n**Trainer:** {trainer}"


def _get_client(base_url: str = _DEFAULT_OLLAMA_URL) -> OllamaClient:
    global _client
    normalized = (base_url or _DEFAULT_OLLAMA_URL).strip().rstrip("/")
    if _client is None or _client._base_url != normalized:
        with _client_lock:
            if _client is None or _client._base_url != normalized:
                _client = OllamaClient(base_url=normalized)
    return _client


def _check_ollama(client: OllamaClient) -> tuple[bool, list[str]]:
    if not client.healthcheck():
        return False, []
    return True, client.list_models()


def _llm_trainer_ready() -> bool:
    try:
        from launch import _build_engine_registry, _engine_enabled, _load_engines_config  # type: ignore[import]

        cfg = _load_engines_config()
        specs = {spec.name: spec for spec in _build_engine_registry()}
        spec = specs.get("llm")
        return _engine_enabled("llm", cfg, default=False) and spec is not None and spec.is_ready()
    except Exception:
        return False


def _atlas_training_path(packet_or_file: str) -> str:
    raw = (packet_or_file or "").strip()
    if not raw:
        return ""
    path = Path(raw)
    if path.is_dir():
        candidate = path / "training_data.jsonl"
        return str(candidate) if candidate.exists() else ""
    return str(path)


def _build_atlas_qlora_request(
    *,
    base_model_path: str,
    packet_path: str,
    adapter_name: str,
    output_dir: str,
    max_steps: int | float,
    batch_size: int | float,
    grad_accum: int | float,
    learning_rate: float,
    max_seq_length: int | float,
    lora_rank: int | float,
    lora_alpha: int | float,
) -> dict:
    dataset_path = _atlas_training_path(packet_path)
    return {
        "job_name": (adapter_name or "atlas_rag_adapter").strip(),
        "base_model_path": (base_model_path or "").strip(),
        "dataset_path": dataset_path,
        "dataset_format": "messages",
        "method": "qlora",
        "output_dir": (output_dir or "outputs/training/atlas_adapters").strip(),
        "max_steps": int(max_steps or 100),
        "num_train_epochs": 1.0,
        "batch_size": int(batch_size or 1),
        "gradient_accumulation_steps": int(grad_accum or 8),
        "learning_rate": float(learning_rate or 2e-5),
        "max_seq_length": int(max_seq_length or 1024),
        "mixed_precision": "bf16",
        "optimizer": "paged_adamw_8bit",
        "packing": False,
        "gradient_checkpointing": True,
        "local_files_only": True,
        "trust_remote_code": True,
        "lora_rank": int(lora_rank or 16),
        "lora_alpha": float(lora_alpha or 32.0),
        "lora_dropout": 0.05,
    }


def _atlas_context(packet_path: str, query: str, top_k: int | float) -> str:
    if not packet_path.strip():
        return ""
    cards = retrieve_atlas_cards(packet_path.strip(), query, top_k=max(1, int(top_k or 4)))
    return format_cards_for_prompt(cards)


def _default_agent_roots(ctx: AppContext) -> list[Path]:
    roots = [ctx.flags.data_dir, ctx.flags.resolved_output_dir()]
    return _dedupe_paths(path for path in roots if path)


def _default_skill_roots(ctx: AppContext) -> list[Path]:
    home = Path.home()
    repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        repo_root / ".agents" / "skills",
        ctx.flags.data_dir / "plugins",
        home / ".codex" / "skills",
        home / ".orchestra" / "skills",
        home / ".claude" / "skills",
    ]
    return _dedupe_paths(candidates)


def _dedupe_paths(paths) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for raw in paths:
        try:
            path = Path(raw).expanduser().resolve(strict=False)
        except OSError:
            continue
        key = str(path).lower()
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def _paths_from_text(value: str, fallback: list[Path]) -> list[Path]:
    rows = [line.strip().strip('"') for line in (value or "").splitlines()]
    paths = [Path(row) for row in rows if row and not row.startswith("#")]
    return _dedupe_paths(paths) or fallback


def _build_agent_tools(
    ctx: AppContext,
    *,
    allowed_roots_text: str,
    skill_roots_text: str,
    allow_file_edits: bool,
) -> ChatAgentToolService:
    return ChatAgentToolService(
        allowed_roots=_paths_from_text(allowed_roots_text, _default_agent_roots(ctx)),
        skill_roots=_paths_from_text(skill_roots_text, _default_skill_roots(ctx)),
        output_dir=ctx.flags.resolved_output_dir(),
        allow_file_edits=allow_file_edits,
    )


def register_chat_workspace(registry: WebRegistry) -> None:
    @registry.tab("Chat", order=3)
    def build(ctx: AppContext, tab: gr.Tab | None = None) -> None:
        client = _get_client()
        alive, models = _check_ollama(client)
        trainer_ready = _llm_trainer_ready()

        with gr.Column(elem_classes=["aiwf-page-header"]):
            gr.Markdown("Chat", elem_classes=["aiwf-section-label"])
            gr.Markdown("Local LLM workbench", elem_classes=["aiwf-page-intro"])

        with gr.Row(elem_classes=["aiwf-chat-workbench"]):
            with gr.Column(scale=8, min_width=520, elem_classes=["aiwf-chat-main"]):
                chatbot = gr.Chatbot(label="Conversation", height=620, buttons=["copy"], elem_classes=["aiwf-chatbot"])
                with gr.Row(elem_classes=["aiwf-chat-composer"]):
                    msg_box = gr.Textbox(placeholder="Type a message...", show_label=False, scale=9)
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                with gr.Row(elem_classes=["aiwf-chat-actions"]):
                    clear_btn = gr.Button("Clear", variant="secondary")
                    unload_btn = gr.Button("Unload model", variant="stop")

            with gr.Column(scale=4, min_width=360, elem_classes=["aiwf-chat-control-panel"]):
                gr.Markdown("Controls", elem_classes=["aiwf-section-label"])
                status_pill = gr.Markdown(
                    value=_STATUS_OK if alive else _STATUS_ERR,
                    elem_id="ollama-status",
                    elem_classes=["aiwf-chat-status"],
                )
                chat_signals = gr.Markdown(
                    value=_format_chat_signals(alive=alive, models=models, trainer_ready=trainer_ready),
                    elem_classes=["aiwf-chat-signals"],
                )
                model_dd = gr.Dropdown(
                    choices=models,
                    value=models[0] if models else None,
                    label="Model",
                    interactive=True,
                )

                with gr.Tabs(elem_classes=["aiwf-chat-tool-tabs"]):
                    with gr.Tab("Backend"):
                        with gr.Group(elem_classes=["aiwf-chat-tab-panel"]):
                            with gr.Row():
                                backend = gr.Dropdown(label="Backend", choices=["Ollama"], value="Ollama", interactive=False)
                                refresh_btn = gr.Button("Refresh", variant="secondary")
                            ollama_url = gr.Textbox(label="Ollama URL", value=_DEFAULT_OLLAMA_URL)
                            install_note = gr.Markdown(value=_INSTALL_NOTE if not alive else "", visible=not alive)
                            backend_setup = gr.Markdown(value=_BACKEND_SETUP if not alive else "", visible=not alive)

                    with gr.Tab("Agent"):
                        with gr.Group(elem_classes=["aiwf-chat-tab-panel"]):
                            with gr.Row():
                                agent_enabled = gr.Checkbox(label="Agent mode", value=False)
                                agent_allow_edits = gr.Checkbox(label="Allow file edits", value=False)
                            with gr.Row():
                                agent_max_steps = gr.Number(label="Max tool steps", value=4, precision=0, minimum=1, maximum=12)
                                agent_preview_btn = gr.Button("Refresh tools", variant="secondary")
                            agent_allowed_roots = gr.Textbox(
                                label="Allowed file roots",
                                value="\n".join(str(path) for path in _default_agent_roots(ctx)),
                                lines=3,
                            )
                            agent_skill_roots = gr.Textbox(
                                label="Skill/plugin roots",
                                value="\n".join(str(path) for path in _default_skill_roots(ctx)),
                                lines=3,
                            )
                            agent_status = gr.Markdown("")

                    with gr.Tab("Atlas"):
                        with gr.Group(elem_classes=["aiwf-chat-tab-panel"]):
                            with gr.Row():
                                atlas_enabled = gr.Checkbox(label="Use Atlas cards", value=False)
                                atlas_top_k = gr.Number(label="Cards", value=4, precision=0, minimum=1, maximum=20)
                            atlas_packet_path = gr.Textbox(
                                label="Atlas packet folder or atlas_cards.jsonl",
                                placeholder=r"datasets\atlas_rag\...\atlas_cards.jsonl",
                            )
                            atlas_query = gr.Textbox(label="Retrieval query preview", placeholder="What should Atlas retrieve?")
                            with gr.Accordion("Cartographer", open=False):
                                atlas_source_paths = gr.Textbox(
                                    label="Sources to cartograph",
                                    value=str(Path(__file__).resolve().parents[3]),
                                    lines=4,
                                )
                                with gr.Row():
                                    atlas_packet_name = gr.Textbox(label="Packet name", value="atlas_chat_rag")
                                    atlas_output_root = gr.Textbox(label="Output root", value="datasets/atlas_rag")
                                with gr.Row():
                                    atlas_max_files = gr.Number(label="Max files", value=200, precision=0, minimum=1, maximum=10_000)
                                    atlas_max_chars = gr.Number(label="Max chars/source", value=6000, precision=0, minimum=500, maximum=50_000)
                                with gr.Row():
                                    atlas_build_btn = gr.Button("Build Atlas packet", variant="secondary")
                                    atlas_preview_btn = gr.Button("Preview retrieval", variant="secondary")
                            atlas_status = gr.Markdown("")
                            atlas_context_preview = gr.Textbox(label="Atlas preview / QLoRA data", lines=7, interactive=False)

                            with gr.Accordion("Adapter activation", open=False):
                                atlas_base_model = gr.Textbox(
                                    label="Trainable base model path or HuggingFace ID",
                                    placeholder=r"F:\Ai_Models\hf\posttrain_candidates\Qwen--Qwen3.5-2B",
                                )
                                with gr.Row():
                                    atlas_adapter_name = gr.Textbox(label="Adapter name", value="atlas_rag_adapter")
                                    atlas_adapter_output = gr.Textbox(label="Adapter output", value="outputs/training/atlas_adapters")
                                with gr.Row():
                                    atlas_steps = gr.Number(label="Max steps", value=100, precision=0, minimum=1, maximum=1_000_000)
                                    atlas_batch = gr.Number(label="Batch size", value=1, precision=0, minimum=1, maximum=64)
                                with gr.Row():
                                    atlas_grad_accum = gr.Number(label="Gradient accumulation", value=8, precision=0, minimum=1, maximum=1024)
                                    atlas_lr = gr.Number(label="Learning rate", value=2e-5, step=1e-6)
                                with gr.Row():
                                    atlas_seq_len = gr.Number(label="Max sequence length", value=1024, precision=0, minimum=128, maximum=32768)
                                    atlas_rank = gr.Number(label="LoRA rank", value=16, precision=0, minimum=1, maximum=512)
                                    atlas_alpha = gr.Number(label="LoRA alpha", value=32, minimum=0.1, maximum=1024)
                                with gr.Row():
                                    atlas_config_btn = gr.Button("Preview QLoRA config", variant="secondary")
                                    atlas_activate_btn = gr.Button("Activate RAG adapter", variant="primary", interactive=trainer_ready)
                                    atlas_stop_btn = gr.Button("Stop adapter training", variant="stop", interactive=False)

                    with gr.Tab("System"):
                        with gr.Group(elem_classes=["aiwf-chat-tab-panel"]):
                            system_prompt = gr.Textbox(placeholder="Optional system prompt", lines=6, show_label=False, value="")

                    with gr.Tab("Trace"):
                        with gr.Group(elem_classes=["aiwf-chat-tab-panel"]):
                            agent_trace = gr.Textbox(label="Agent trace", lines=8, interactive=False)
                            atlas_train_log = gr.Textbox(
                                label="Atlas adapter training log",
                                lines=10,
                                interactive=False,
                                autoscroll=True,
                            )

        def on_refresh(base_url: str):
            active_client = _get_client(base_url)
            is_alive, names = _check_ollama(active_client)
            llm_trainer_ready = _llm_trainer_ready()
            return (
                gr.update(value=_STATUS_OK if is_alive else _STATUS_ERR),
                gr.update(choices=names, value=names[0] if names else None),
                gr.update(value=_INSTALL_NOTE if not is_alive else "", visible=not is_alive),
                gr.update(value=_BACKEND_SETUP if not is_alive else "", visible=not is_alive),
                gr.update(value=_format_chat_signals(alive=is_alive, models=names, trainer_ready=llm_trainer_ready)),
            )

        refresh_btn.click(
            fn=on_refresh,
            inputs=[ollama_url],
            outputs=[status_pill, model_dd, install_note, backend_setup, chat_signals],
        )
        if tab is not None:
            tab.select(
                fn=on_refresh,
                inputs=[ollama_url],
                outputs=[status_pill, model_dd, install_note, backend_setup, chat_signals],
            )

        def on_agent_preview(roots_text: str, skill_roots_text: str, allow_edits: bool):
            try:
                tools = _build_agent_tools(
                    ctx,
                    allowed_roots_text=roots_text,
                    skill_roots_text=skill_roots_text,
                    allow_file_edits=allow_edits,
                )
                root_lines = "\n".join(f"- `{root}`" for root in tools.allowed_roots)
                packs = tools.discover_instruction_packs(max_packs=12)
                pack_lines = "\n".join(
                    f"- `{pack.kind}` `{pack.name}` from `{pack.entry_file}`" for pack in packs
                ) or "- No skills/plugins found in configured roots."
                edit_state = "enabled" if tools.allow_file_edits else "disabled"
                return f"**Agent tools ready.** File edits {edit_state}.\n\n**Allowed roots**\n{root_lines}\n\n**Skills/plugins**\n{pack_lines}"
            except Exception as exc:
                return f"ERROR: Agent tools are not ready: {exc}"

        agent_preview_btn.click(
            fn=on_agent_preview,
            inputs=[agent_allowed_roots, agent_skill_roots, agent_allow_edits],
            outputs=[agent_status],
        )

        def on_build_atlas(paths: str, name: str, out_root: str, max_files: int, max_chars: int):
            try:
                from aiwf.services.chat_atlas_rag import build_atlas_rag_packet

                result = build_atlas_rag_packet(
                    paths,
                    packet_name=name or "atlas_chat_rag",
                    output_root=out_root or "datasets/atlas_rag",
                    max_files=int(max_files or 200),
                    max_chars_per_source=int(max_chars or 6000),
                )
                return result.markdown(), str(result.output_dir), str(result.training_data_path)
            except Exception as exc:
                logger.exception("[Chat] Atlas packet build failed")
                return f"ERROR: Atlas packet build failed: {exc}", gr.update(), gr.update()

        atlas_build_btn.click(
            fn=on_build_atlas,
            inputs=[atlas_source_paths, atlas_packet_name, atlas_output_root, atlas_max_files, atlas_max_chars],
            outputs=[atlas_status, atlas_packet_path, atlas_context_preview],
        )

        def on_preview_atlas(packet_path: str, query: str, top_k: int):
            try:
                context = _atlas_context(packet_path, query, top_k)
                return context or "No Atlas cards matched."
            except Exception as exc:
                return f"ERROR: Atlas retrieval failed: {exc}"

        atlas_preview_btn.click(
            fn=on_preview_atlas,
            inputs=[atlas_packet_path, atlas_query, atlas_top_k],
            outputs=[atlas_context_preview],
        )

        atlas_training_inputs = [
            atlas_base_model,
            atlas_packet_path,
            atlas_adapter_name,
            atlas_adapter_output,
            atlas_steps,
            atlas_batch,
            atlas_grad_accum,
            atlas_lr,
            atlas_seq_len,
            atlas_rank,
            atlas_alpha,
        ]

        def _atlas_req_from_values(values: tuple) -> dict:
            return _build_atlas_qlora_request(
                base_model_path=values[0],
                packet_path=values[1],
                adapter_name=values[2],
                output_dir=values[3],
                max_steps=values[4],
                batch_size=values[5],
                grad_accum=values[6],
                learning_rate=values[7],
                max_seq_length=values[8],
                lora_rank=values[9],
                lora_alpha=values[10],
            )

        def on_preview_atlas_config(*values):
            try:
                from aiwf.services.training.llm_config import build_llm_training_config

                req = _atlas_req_from_values(values)
                return "```json\n" + json.dumps(build_llm_training_config(req), indent=2) + "\n```"
            except Exception as exc:
                return f"ERROR: Could not preview Atlas QLoRA config: {exc}"

        atlas_config_btn.click(fn=on_preview_atlas_config, inputs=atlas_training_inputs, outputs=[atlas_status])

        _active_atlas_runner: list = [None]
        _active_atlas_job: list[str] = [""]

        def on_activate_atlas(*values):
            yield gr.update(interactive=False), gr.update(interactive=True), "Starting Atlas adapter activation...\n", "Starting Atlas QLoRA adapter..."
            if not _llm_trainer_ready():
                yield (
                    gr.update(interactive=True),
                    gr.update(interactive=False),
                    "ERROR: LLM trainer engine is not ready. Enable AI bot trainer in the Training tab, restart through launch.py, then try again.\n",
                    "ERROR: LLM trainer not ready",
                )
                return

            req = _atlas_req_from_values(values)
            validation = _validator.validate_llm(req)
            if not validation.ok:
                errors = "\n\n".join(f"ERROR: {error}" for error in validation.errors)
                yield gr.update(interactive=True), gr.update(interactive=False), errors, "ERROR: Atlas adapter preflight failed"
                return

            tenant_job_id = f"{EngineTenant.LORA_TRAINING.value}_{uuid.uuid4().hex[:8]}"
            tenant_acquired = False
            supervisor = getattr(ctx, "supervisor", None)
            if supervisor is not None:
                result = supervisor.request_switch(
                    EngineSwitchRequest(
                        target=EngineTenant.LORA_TRAINING,
                        reason=f"Atlas RAG QLoRA adapter: {req['job_name']}",
                        job_id=tenant_job_id,
                    )
                )
                if not result.ok:
                    yield gr.update(interactive=True), gr.update(interactive=False), f"ERROR: GPU busy: {result.message}\n", "ERROR: GPU busy"
                    return
                tenant_acquired = True
                _active_atlas_job[0] = tenant_job_id

            log_lines: list[str] = []
            try:
                from aiwf.services.training.llm_runner import LLMBotTrainerRunner

                runner = LLMBotTrainerRunner()
                _active_atlas_runner[0] = runner
                for line in runner.start(req, job_id=tenant_job_id):
                    log_lines.append(line)
                    if len(log_lines) % 5 == 0:
                        yield gr.update(interactive=False), gr.update(interactive=True), "\n".join(log_lines[-200:]), f"Atlas adapter training... ({len(log_lines)} lines)"
            except Exception as exc:
                logger.exception("[Chat] Atlas adapter training error")
                log_lines.append(f"ERROR: {exc}")
                if tenant_acquired:
                    _release_chat_tenant(ctx, "Atlas adapter training failed", tenant_job_id)
                _active_atlas_runner[0] = None
                _active_atlas_job[0] = ""
                yield gr.update(interactive=True), gr.update(interactive=False), "\n".join(log_lines), f"ERROR: {exc}"
                return

            if tenant_acquired:
                _release_chat_tenant(ctx, "Atlas adapter training complete", tenant_job_id)
            _active_atlas_runner[0] = None
            _active_atlas_job[0] = ""
            yield gr.update(interactive=True), gr.update(interactive=False), "\n".join(log_lines), "Atlas adapter training complete."

        atlas_activate_btn.click(
            fn=on_activate_atlas,
            inputs=atlas_training_inputs,
            outputs=[atlas_activate_btn, atlas_stop_btn, atlas_train_log, atlas_status],
        )

        def on_stop_atlas():
            runner = _active_atlas_runner[0]
            if runner is None:
                return gr.update(interactive=True), gr.update(interactive=False), "WARNING: No active Atlas adapter job."
            try:
                msg = runner.stop()
            except Exception as exc:
                msg = f"Stop error: {exc}"
            _active_atlas_runner[0] = None
            tenant_job_id = _active_atlas_job[0]
            _active_atlas_job[0] = ""
            _release_chat_tenant(ctx, "Atlas adapter training stopped by user", tenant_job_id)
            return gr.update(interactive=True), gr.update(interactive=False), f"Stopped: {msg}"

        atlas_stop_btn.click(fn=on_stop_atlas, outputs=[atlas_activate_btn, atlas_stop_btn, atlas_status])

        def on_send(
            user_msg: str,
            history: list,
            model: str,
            sys_prompt: str,
            base_url: str,
            agent_mode: bool,
            allow_file_edits: bool,
            allowed_roots_text: str,
            skill_roots_text: str,
            max_agent_steps: int,
            use_atlas: bool,
            packet_path: str,
            top_k: int,
        ):
            if not user_msg.strip():
                yield history, gr.update(value=""), gr.update()
                return
            if not model:
                history = history + [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": "No model selected. Choose a model from the dropdown."},
                ]
                yield history, gr.update(value=""), gr.update()
                return

            active_client = _get_client(base_url)
            supervisor = getattr(ctx, "supervisor", None)
            if supervisor is not None:
                supervisor.set_ollama_client(active_client)
                supervisor.set_chat_model(model)
                result = supervisor.request_switch(
                    EngineSwitchRequest(target=EngineTenant.CHAT, reason="Chat send", job_id="chat")
                )
                if not result.ok:
                    history = history + [
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": f"GPU busy: {result.message}"},
                    ]
                    yield history, gr.update(value=""), gr.update()
                    return

            system_parts: list[str] = []
            if sys_prompt and sys_prompt.strip():
                system_parts.append(sys_prompt.strip())
            agent_tools: ChatAgentToolService | None = None
            if agent_mode:
                try:
                    agent_tools = _build_agent_tools(
                        ctx,
                        allowed_roots_text=allowed_roots_text,
                        skill_roots_text=skill_roots_text,
                        allow_file_edits=allow_file_edits,
                    )
                    system_parts.append(agent_tools.agent_system_prompt())
                except Exception as exc:
                    history = history + [
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": f"Agent tools error: {exc}"},
                    ]
                    yield history, gr.update(value=""), f"ERROR: {exc}"
                    return
            if use_atlas:
                try:
                    atlas_text = _atlas_context(packet_path, user_msg, top_k)
                except Exception as exc:
                    history = history + [
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": f"Atlas RAG error: {exc}"},
                    ]
                    yield history, gr.update(value=""), gr.update()
                    return
                if atlas_text:
                    system_parts.append(atlas_text)

            messages: list[dict] = []
            if system_parts:
                messages.append({"role": "system", "content": "\n\n".join(system_parts)})
            for entry in history:
                messages.append(entry)
            messages.append({"role": "user", "content": user_msg})

            history = history + [{"role": "user", "content": user_msg}, {"role": "assistant", "content": ""}]
            yield history, gr.update(value=""), "Agent mode active." if agent_tools is not None else ""

            if agent_tools is not None:
                try:
                    def chat_once(turn_messages: list[dict[str, str]]) -> str:
                        return "".join(active_client.stream_chat(model, turn_messages))

                    for update in run_agentic_turn(
                        chat_once,
                        messages,
                        agent_tools,
                        max_steps=int(max_agent_steps or 4),
                    ):
                        history[-1] = {"role": "assistant", "content": update.content}
                        yield history, gr.update(value=""), update.trace
                except Exception as exc:
                    logger.exception("[Chat] agentic stream error")
                    history[-1] = {"role": "assistant", "content": f"Agent error: {exc}"}
                    yield history, gr.update(value=""), f"ERROR: {exc}"
                return

            assistant_text = ""
            try:
                for token in active_client.stream_chat(model, messages):
                    assistant_text += token
                    history[-1] = {"role": "assistant", "content": assistant_text}
                    yield history, gr.update(value=""), ""
            except Exception as exc:
                logger.exception("[Chat] stream_chat error")
                history[-1] = {"role": "assistant", "content": f"Error: {exc}"}
                yield history, gr.update(value=""), ""

        send_inputs = [
            msg_box,
            chatbot,
            model_dd,
            system_prompt,
            ollama_url,
            agent_enabled,
            agent_allow_edits,
            agent_allowed_roots,
            agent_skill_roots,
            agent_max_steps,
            atlas_enabled,
            atlas_packet_path,
            atlas_top_k,
        ]
        send_btn.click(fn=on_send, inputs=send_inputs, outputs=[chatbot, msg_box, agent_trace])
        msg_box.submit(fn=on_send, inputs=send_inputs, outputs=[chatbot, msg_box, agent_trace])

        clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot, agent_trace])

        def on_unload(model: str, base_url: str):
            if not model:
                return gr.update(value="No model selected.")
            active_client = _get_client(base_url)
            ok = active_client.unload(model)
            supervisor = getattr(ctx, "supervisor", None)
            if supervisor is not None and ok:
                supervisor.request_switch(
                    EngineSwitchRequest(target=EngineTenant.IDLE, reason="Chat model unloaded", job_id="chat")
                )
            return gr.update(value=f"{'Unloaded' if ok else 'Unload failed'}: {model}")

        unload_btn.click(fn=on_unload, inputs=[model_dd, ollama_url], outputs=[status_pill])


def _release_chat_tenant(ctx: AppContext, reason: str, job_id: str = "") -> None:
    supervisor = getattr(ctx, "supervisor", None)
    if supervisor is not None:
        supervisor.request_switch(EngineSwitchRequest(target=EngineTenant.IDLE, reason=reason, job_id=job_id))
