"""Training tab for image trainers and AI bot text-model post-training.

Rules enforced here:
  - No torch imports.
  - Heavy engine imports happen only inside callbacks.
  - Training engines are opt-in; dependencies are not installed at first app
    install unless the user enables the engine from this tab.
  - GPU tenant lock is acquired before starting a job.
"""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

import gradio as gr

from aiwf.bootstrap import AppContext
from aiwf.core.domain.engine import EngineSwitchRequest, EngineTenant
from aiwf.services.training.dataset_validator import DatasetValidator
from aiwf.web.registry import WebRegistry

logger = logging.getLogger(__name__)

_validator = DatasetValidator()


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "launch.py").is_file():
            return parent
    return here.parents[3]


def _probe_engines() -> dict[str, bool]:
    """Return availability of each training engine. Never raises."""
    result = {"kohya": False, "ed2": False, "llm": False}
    try:
        from launch import _build_engine_registry, _engine_enabled, _load_engines_config  # type: ignore[import]

        cfg = _load_engines_config()
        specs = {spec.name: spec for spec in _build_engine_registry()}
        for name in result:
            spec = specs.get(name)
            result[name] = (
                _engine_enabled(name, cfg, default=False)
                and spec is not None
                and spec.is_ready()
            )
    except Exception:
        pass
    return result


_NOT_CONFIGURED = (
    "**No training engine is ready.** Enable the engine you need from this tab, "
    "then restart AIWF Studio through `launch.py` so its isolated venv can be prepared."
)
_KOHYA_NOT_CONFIGURED = (
    "**Kohya is not ready.** Configure `kohya` in `engines.json`, install its engine, then restart."
)


def _format_validation_result(result) -> str:
    lines: list[str] = []
    if result.ok:
        lines.append("OK: Dataset looks good.")
    else:
        lines.extend(f"ERROR: {err}" for err in result.errors)
    lines.extend(f"WARNING: {warning}" for warning in result.warnings)
    return "\n\n".join(lines)


def _is_lora_engine(engine: str) -> bool:
    return "Kohya" in engine or "LoRA" in engine or "DreamBooth" in engine


def _is_llm_engine(engine: str) -> bool:
    return "AI Bot" in engine or engine.lower() in {"llm", "ai bot trainer"}


def _validate_training_request(engine: str, request: dict):
    if _is_llm_engine(engine):
        return _validator.validate_llm(request)
    if _is_lora_engine(engine):
        return _validator.validate_kohya(request)
    return _validator.validate_ed2(request)


def _runner_for_engine(engine: str):
    if _is_llm_engine(engine):
        from aiwf.services.training.llm_runner import LLMBotTrainerRunner

        return LLMBotTrainerRunner()
    if _is_lora_engine(engine):
        from aiwf.services.training.kohya_runner import KohyaRunner

        return KohyaRunner()
    if "ED2" in engine:
        from aiwf.services.training.ed2_runner import ED2Runner

        return ED2Runner()
    raise RuntimeError(f"Unknown training engine: {engine}")


def _release_tenant(ctx: AppContext, reason: str, job_id: str = "") -> None:
    supervisor = getattr(ctx, "supervisor", None)
    if supervisor is not None:
        supervisor.request_switch(EngineSwitchRequest(target=EngineTenant.IDLE, reason=reason, job_id=job_id))


def _llm_request(
    *,
    method: str,
    model_path: str,
    dataset_path: str,
    dataset_format: str,
    job_name: str,
    output_dir: str,
    max_steps: int | float,
    epochs: int | float,
    batch_size: int | float,
    grad_accum: int | float,
    learning_rate: float,
    max_seq_length: int | float,
    mixed_precision: str,
    optimizer: str,
    packing: bool,
    gradient_checkpointing: bool,
    local_files_only: bool,
    trust_remote_code: bool,
    lora_rank: int | float,
    lora_alpha: int | float,
    lora_dropout: float,
) -> dict:
    return {
        "job_name": (job_name or "ai_bot_job").strip(),
        "base_model_path": (model_path or "").strip(),
        "dataset_path": (dataset_path or "").strip(),
        "dataset_format": dataset_format or "auto",
        "method": method or "qlora",
        "output_dir": (output_dir or "outputs/training/llm").strip(),
        "max_steps": int(max_steps or 100),
        "num_train_epochs": float(epochs or 1.0),
        "batch_size": int(batch_size or 1),
        "gradient_accumulation_steps": int(grad_accum or 8),
        "learning_rate": float(learning_rate or 2e-5),
        "max_seq_length": int(max_seq_length or 1024),
        "mixed_precision": mixed_precision or "bf16",
        "optimizer": optimizer or "",
        "packing": bool(packing),
        "gradient_checkpointing": bool(gradient_checkpointing),
        "local_files_only": bool(local_files_only),
        "trust_remote_code": bool(trust_remote_code),
        "lora_rank": int(lora_rank or 16),
        "lora_alpha": float(lora_alpha or 32.0),
        "lora_dropout": float(lora_dropout or 0.05),
    }


def register_training(registry: WebRegistry) -> None:
    @registry.tab("Training", order=28)
    def build(ctx: AppContext, tab: gr.Tab | None = None) -> None:
        engines = _probe_engines()
        any_ready = any(engines.values())
        root = _repo_root()

        with gr.Column(elem_classes=["aiwf-page-header"]):
            gr.Markdown("Training", elem_classes=["aiwf-section-label"])
            gr.Markdown(
                "Image trainers and AI bot post-training run as optional isolated workers.",
                elem_classes=["aiwf-page-intro"],
            )

        gr.Markdown(value=_NOT_CONFIGURED if not any_ready else "", visible=not any_ready)

        with gr.Row():
            enable_llm_btn = gr.Button(
                "Enable AI bot trainer",
                variant="primary" if not engines["llm"] else "secondary",
                interactive=not engines["llm"],
            )
            enable_full_btn = gr.Button(
                "Enable ED2 full image training",
                variant="secondary",
                interactive=not engines["ed2"],
            )
            enable_lora_btn = gr.Button("Enable Kohya image LoRA", variant="secondary")
        training_enable_status = gr.Markdown(
            value="AI bot trainer is ready." if engines["llm"] else "",
            elem_classes=["aiwf-status-bar"],
        )

        with gr.Group():
            gr.Markdown("Dataset Builder", elem_classes=["aiwf-section-label"])
            with gr.Row():
                dataset_source_paths = gr.Textbox(
                    label="Source files or folders",
                    value=str(root),
                    lines=4,
                    scale=7,
                    placeholder="One path per line. Supports .py, .md, .txt, .json, .jsonl, .toml, .yaml.",
                )
                with gr.Column(scale=3):
                    dataset_name = gr.Textbox(label="Dataset name", value="aiwf_ai_bot_dataset")
                    dataset_output_root = gr.Textbox(label="Output root", value="datasets/ai_model_training")
            with gr.Row():
                dataset_max_files = gr.Number(label="Max files", value=200, precision=0, minimum=1, maximum=10_000)
                dataset_max_chars = gr.Number(label="Max chars per source excerpt", value=6000, precision=0, minimum=500, maximum=50_000)
                build_dataset_btn = gr.Button("Build dataset", variant="secondary")
            dataset_builder_status = gr.Markdown("")

        with gr.Group():
            gr.Markdown("AI Bot Trainer", elem_classes=["aiwf-section-label"])
            llm_ready_note = (
                "Ready. Choose LoRA, QLoRA, or full fine-tune."
                if engines["llm"]
                else "Not ready. Enable AI bot trainer, restart through launch.py, then return here."
            )
            gr.Markdown(llm_ready_note, elem_classes=["aiwf-settings-hint"])
            with gr.Row():
                llm_method = gr.Dropdown(
                    label="Method",
                    choices=[
                        ("QLoRA 4-bit adapter", "qlora"),
                        ("LoRA adapter", "lora"),
                        ("Full fine-tune", "full"),
                    ],
                    value="qlora",
                )
                llm_dataset_format = gr.Dropdown(
                    label="Dataset format",
                    choices=[
                        ("Auto detect", "auto"),
                        ("Chat messages", "messages"),
                        ("Prompt/completion", "prompt_completion"),
                        ("Plain text", "text"),
                    ],
                    value="auto",
                )
                llm_mixed_precision = gr.Dropdown(
                    label="Mixed precision",
                    choices=["bf16", "fp16", "no"],
                    value="bf16",
                )
            with gr.Row():
                llm_model_path = gr.Textbox(
                    label="Base model path or HuggingFace ID",
                    placeholder=r"F:\Ai_Models\hf\posttrain_candidates\Qwen--Qwen3.5-2B",
                    scale=6,
                )
                llm_dataset_path = gr.Textbox(
                    label="Training JSONL/JSON path",
                    placeholder=r"datasets\ai_model_training\...\splits\train.jsonl",
                    scale=6,
                )
            with gr.Row():
                llm_job_name = gr.Textbox(label="Job name", value="ai_bot_coder")
                llm_output_dir = gr.Textbox(label="Output directory", value="outputs/training/llm")
            with gr.Accordion("AI bot training parameters", open=True):
                with gr.Row():
                    llm_max_steps = gr.Number(label="Max steps", value=100, precision=0, minimum=1, maximum=1_000_000)
                    llm_epochs = gr.Number(label="Epochs", value=1.0, minimum=0.0, maximum=1000.0)
                    llm_batch_size = gr.Number(label="Batch size", value=1, precision=0, minimum=1, maximum=64)
                    llm_grad_accum = gr.Number(label="Gradient accumulation", value=8, precision=0, minimum=1, maximum=1024)
                with gr.Row():
                    llm_learning_rate = gr.Number(label="Learning rate", value=2e-5, step=1e-6)
                    llm_max_seq = gr.Number(label="Max sequence length", value=1024, precision=0, minimum=128, maximum=32768)
                    llm_optimizer = gr.Dropdown(
                        label="Optimizer",
                        choices=[
                            ("Auto recommended", ""),
                            ("adamw_torch", "adamw_torch"),
                            ("adamw_bnb_8bit", "adamw_bnb_8bit"),
                            ("paged_adamw_8bit", "paged_adamw_8bit"),
                            ("paged_adamw_32bit", "paged_adamw_32bit"),
                            ("adafactor", "adafactor"),
                        ],
                        value="",
                    )
                with gr.Row():
                    llm_lora_rank = gr.Number(label="LoRA rank", value=16, precision=0, minimum=1, maximum=512)
                    llm_lora_alpha = gr.Number(label="LoRA alpha", value=32, minimum=0.1, maximum=1024)
                    llm_lora_dropout = gr.Number(label="LoRA dropout", value=0.05, minimum=0.0, maximum=1.0)
                with gr.Row():
                    llm_packing = gr.Checkbox(label="Pack short examples", value=False)
                    llm_gradient_checkpointing = gr.Checkbox(label="Gradient checkpointing", value=True)
                    llm_local_files_only = gr.Checkbox(label="Local files only", value=True)
                    llm_trust_remote_code = gr.Checkbox(label="Trust remote code", value=True)
            with gr.Row():
                llm_validate_btn = gr.Button("Validate bot dataset", variant="secondary")
                llm_preview_btn = gr.Button("Preview config", variant="secondary")
                llm_start_btn = gr.Button("Start bot training", variant="primary", interactive=engines["llm"])
                llm_stop_btn = gr.Button("Stop bot training", variant="stop", interactive=False)
            llm_status = gr.Markdown("")
            llm_log_box = gr.Textbox(label="AI bot training log", lines=16, max_lines=20, interactive=False, autoscroll=True)

        with gr.Group():
            gr.Markdown("Image Training", elem_classes=["aiwf-section-label"])
            image_engine_choices = []
            if engines["kohya"]:
                image_engine_choices.append("Kohya LoRA")
            if engines["ed2"]:
                image_engine_choices.append("ED2 Full Fine-tune")
            if not image_engine_choices:
                image_engine_choices = ["Kohya LoRA", "ED2 Full Fine-tune"]

            image_engine_radio = gr.Radio(
                choices=image_engine_choices,
                value=image_engine_choices[0],
                label="Image training engine",
                interactive=engines["kohya"] or engines["ed2"],
            )
            with gr.Row():
                dataset_dir = gr.Textbox(label="Dataset directory", placeholder=r"C:\training\my_concept", scale=8)
                validate_btn = gr.Button("Validate", variant="secondary", scale=1)
            validation_result = gr.Markdown(value="")
            base_model = gr.Textbox(
                label="Base model path or HuggingFace ID",
                placeholder=r"C:\models\sdxl_base.safetensors  or  stabilityai/stable-diffusion-xl-base-1.0",
            )
            with gr.Row():
                job_name = gr.Textbox(label="Job name", value="my_lora", scale=4)
                output_dir = gr.Textbox(label="Output directory", value="outputs/training", scale=6)
            with gr.Accordion("Image training parameters", open=False):
                with gr.Row():
                    max_steps_input = gr.Number(label="Max steps (LoRA / DreamBooth)", value=1500, precision=0, minimum=100, maximum=100_000)
                    max_epochs_input = gr.Number(label="Max epochs (ED2)", value=20, precision=0, minimum=1, maximum=1000)
                with gr.Row():
                    batch_size_input = gr.Number(label="Batch size", value=1, precision=0, minimum=1, maximum=64)
                    lr_input = gr.Number(label="Learning rate", value=1e-4, step=1e-5)
                with gr.Row():
                    resolution_input = gr.Dropdown(choices=[512, 768, 1024], value=1024, label="Resolution")
                    mixed_prec_input = gr.Dropdown(choices=["bf16", "fp16", "no"], value="bf16", label="Mixed precision")
            with gr.Row():
                image_start_btn = gr.Button("Start image training", variant="primary", interactive=engines["kohya"] or engines["ed2"])
                image_stop_btn = gr.Button("Stop image training", variant="stop", interactive=False)
            image_log_box = gr.Textbox(label="Image training log", lines=16, max_lines=20, interactive=False, autoscroll=True)
            image_status_bar = gr.Markdown(value="")

        def on_enable_llm_training():
            try:
                from aiwf.services.training.llm_installer import install_llm_trainer_addon

                lines = install_llm_trainer_addon()
                return "\n\n".join(lines)
            except Exception as exc:
                logger.exception("[Training tab] LLM trainer enable failed")
                return f"ERROR: AI bot trainer enable failed: {exc}"

        def on_enable_full_training():
            try:
                from aiwf.services.training.ed2_installer import install_ed2_addon

                lines = install_ed2_addon()
                return "\n\n".join([*lines, "Restart AIWF Studio to refresh training engine availability."])
            except Exception as exc:
                logger.exception("[Training tab] ED2 add-on install failed")
                return f"ERROR: Full training enable failed: {exc}"

        def on_enable_lora_training():
            return _KOHYA_NOT_CONFIGURED

        enable_llm_btn.click(fn=on_enable_llm_training, outputs=[training_enable_status])
        enable_full_btn.click(fn=on_enable_full_training, outputs=[training_enable_status])
        enable_lora_btn.click(fn=on_enable_lora_training, outputs=[training_enable_status])

        def on_build_dataset(paths: str, name: str, out_root: str, max_files: int, max_chars: int):
            try:
                from aiwf.services.training.llm_dataset_builder import build_ai_model_dataset

                result = build_ai_model_dataset(
                    paths,
                    dataset_name=name or "ai_bot_dataset",
                    output_root=out_root or "datasets/ai_model_training",
                    max_files=int(max_files or 200),
                    max_chars_per_file=int(max_chars or 6000),
                )
                return result.markdown(), str(result.train_path)
            except Exception as exc:
                logger.exception("[Training tab] Dataset build failed")
                return f"ERROR: Dataset build failed: {exc}", gr.update()

        build_dataset_btn.click(
            fn=on_build_dataset,
            inputs=[dataset_source_paths, dataset_name, dataset_output_root, dataset_max_files, dataset_max_chars],
            outputs=[dataset_builder_status, llm_dataset_path],
        )

        llm_inputs = [
            llm_method,
            llm_model_path,
            llm_dataset_path,
            llm_dataset_format,
            llm_job_name,
            llm_output_dir,
            llm_max_steps,
            llm_epochs,
            llm_batch_size,
            llm_grad_accum,
            llm_learning_rate,
            llm_max_seq,
            llm_mixed_precision,
            llm_optimizer,
            llm_packing,
            llm_gradient_checkpointing,
            llm_local_files_only,
            llm_trust_remote_code,
            llm_lora_rank,
            llm_lora_alpha,
            llm_lora_dropout,
        ]

        def _llm_req_from_values(values: tuple) -> dict:
            return _llm_request(
                method=values[0],
                model_path=values[1],
                dataset_path=values[2],
                dataset_format=values[3],
                job_name=values[4],
                output_dir=values[5],
                max_steps=values[6],
                epochs=values[7],
                batch_size=values[8],
                grad_accum=values[9],
                learning_rate=values[10],
                max_seq_length=values[11],
                mixed_precision=values[12],
                optimizer=values[13],
                packing=values[14],
                gradient_checkpointing=values[15],
                local_files_only=values[16],
                trust_remote_code=values[17],
                lora_rank=values[18],
                lora_alpha=values[19],
                lora_dropout=values[20],
            )

        def on_validate_llm(*values):
            req = _llm_req_from_values(values)
            return _format_validation_result(_validator.validate_llm(req))

        def on_preview_llm(*values):
            try:
                from aiwf.services.training.llm_config import build_llm_training_config

                req = _llm_req_from_values(values)
                cfg = build_llm_training_config(req)
                return "```json\n" + json.dumps(cfg, indent=2) + "\n```"
            except Exception as exc:
                return f"ERROR: Could not preview config: {exc}"

        llm_validate_btn.click(fn=on_validate_llm, inputs=llm_inputs, outputs=[llm_status])
        llm_preview_btn.click(fn=on_preview_llm, inputs=llm_inputs, outputs=[llm_status])

        _active_llm_runner: list = [None]
        _active_llm_tenant_job_id: list[str] = [""]
        _active_image_runner: list = [None]
        _active_image_tenant_job_id: list[str] = [""]

        def on_start_llm(*values):
            yield gr.update(interactive=False), gr.update(interactive=True), "Starting AI bot training...\n", "Starting..."
            req = _llm_req_from_values(values)
            validation = _validator.validate_llm(req)
            if not validation.ok:
                yield gr.update(interactive=True), gr.update(interactive=False), _format_validation_result(validation), "ERROR: Preflight failed"
                return

            method = req["method"]
            tenant = EngineTenant.FULL_TRAINING if method == "full" else EngineTenant.LORA_TRAINING
            tenant_job_id = f"{tenant.value}_{uuid.uuid4().hex[:8]}"
            tenant_acquired = False
            supervisor = getattr(ctx, "supervisor", None)
            if supervisor is not None:
                result = supervisor.request_switch(
                    EngineSwitchRequest(target=tenant, reason=f"AI bot training: {req['job_name']}", job_id=tenant_job_id)
                )
                if not result.ok:
                    yield gr.update(interactive=True), gr.update(interactive=False), f"ERROR: GPU busy: {result.message}\n", "ERROR: GPU busy"
                    return
                tenant_acquired = True
                _active_llm_tenant_job_id[0] = tenant_job_id

            log_lines: list[str] = []
            try:
                runner = _runner_for_engine("AI Bot Trainer")
                _active_llm_runner[0] = runner
                for line in runner.start(req, job_id=tenant_job_id):
                    log_lines.append(line)
                    if len(log_lines) % 5 == 0:
                        yield gr.update(interactive=False), gr.update(interactive=True), "\n".join(log_lines[-200:]), f"Training... ({len(log_lines)} lines)"
            except Exception as exc:
                logger.exception("[Training tab] LLM training error")
                log_lines.append(f"ERROR: {exc}")
                if tenant_acquired:
                    _release_tenant(ctx, "AI bot training failed", tenant_job_id)
                _active_llm_runner[0] = None
                _active_llm_tenant_job_id[0] = ""
                yield gr.update(interactive=True), gr.update(interactive=False), "\n".join(log_lines), f"ERROR: {exc}"
                return

            if tenant_acquired:
                _release_tenant(ctx, "AI bot training complete", tenant_job_id)
            _active_llm_runner[0] = None
            _active_llm_tenant_job_id[0] = ""
            yield gr.update(interactive=True), gr.update(interactive=False), "\n".join(log_lines), "AI bot training complete."

        llm_start_btn.click(
            fn=on_start_llm,
            inputs=llm_inputs,
            outputs=[llm_start_btn, llm_stop_btn, llm_log_box, llm_status],
        )

        def on_stop_llm():
            runner = _active_llm_runner[0]
            if runner is None:
                return gr.update(interactive=True), gr.update(interactive=False), "WARNING: No active AI bot job."
            try:
                msg = runner.stop()
            except Exception as exc:
                msg = f"Stop error: {exc}"
            _active_llm_runner[0] = None
            tenant_job_id = _active_llm_tenant_job_id[0]
            _active_llm_tenant_job_id[0] = ""
            _release_tenant(ctx, "AI bot training stopped by user", tenant_job_id)
            return gr.update(interactive=True), gr.update(interactive=False), f"Stopped: {msg}"

        llm_stop_btn.click(fn=on_stop_llm, outputs=[llm_start_btn, llm_stop_btn, llm_status])

        def on_validate_image(dataset_path: str):
            if not dataset_path.strip():
                return "WARNING: Enter a dataset directory path first."
            result = _validator.validate_dataset_dir(dataset_path.strip())
            return _format_validation_result(result)

        validate_btn.click(fn=on_validate_image, inputs=[dataset_dir], outputs=[validation_result])

        def on_start_image(
            engine: str,
            ds_dir: str,
            base_mdl: str,
            jname: str,
            out_dir: str,
            steps: int,
            epochs: int,
            bs: int,
            lr: float,
            res: int,
            mp: str,
        ):
            yield gr.update(interactive=False), gr.update(interactive=True), "Starting image training...\n", "Starting..."

            if not ds_dir.strip():
                yield gr.update(interactive=True), gr.update(interactive=False), "ERROR: Dataset directory is required.\n", "ERROR"
                return
            if not base_mdl.strip():
                yield gr.update(interactive=True), gr.update(interactive=False), "ERROR: Base model path is required.\n", "ERROR"
                return

            req: dict = {
                "job_name": jname or "image_training_job",
                "base_model_path": base_mdl.strip(),
                "dataset_dir": ds_dir.strip(),
                "output_dir": out_dir.strip() or "outputs/training",
                "resolution": int(res),
                "mixed_precision": mp,
                "batch_size": int(bs),
                "seed": 42,
            }
            if _is_lora_engine(engine):
                req["max_train_steps"] = int(steps)
                req["learning_rate"] = float(lr)
                req["base_arch"] = "sdxl"
            elif "ED2" in engine:
                req["max_epochs"] = int(epochs)
                req["lr"] = float(lr)

            validation = _validate_training_request(engine, req)
            if not validation.ok:
                yield gr.update(interactive=True), gr.update(interactive=False), _format_validation_result(validation), "ERROR: Preflight failed"
                return

            tenant = EngineTenant.LORA_TRAINING if _is_lora_engine(engine) else EngineTenant.FULL_TRAINING
            tenant_job_id = f"{tenant.value}_{uuid.uuid4().hex[:8]}"
            tenant_acquired = False
            supervisor = getattr(ctx, "supervisor", None)
            if supervisor is not None:
                result = supervisor.request_switch(
                    EngineSwitchRequest(target=tenant, reason=f"Image training: {jname}", job_id=tenant_job_id)
                )
                if not result.ok:
                    yield gr.update(interactive=True), gr.update(interactive=False), f"ERROR: GPU busy: {result.message}\n", "ERROR: GPU busy"
                    return
                tenant_acquired = True
                _active_image_tenant_job_id[0] = tenant_job_id

            log_lines: list[str] = []
            try:
                runner = _runner_for_engine(engine)
                _active_image_runner[0] = runner
                for line in runner.start(req, job_id=tenant_job_id):
                    log_lines.append(line)
                    if len(log_lines) % 5 == 0:
                        yield gr.update(interactive=False), gr.update(interactive=True), "\n".join(log_lines[-200:]), f"Training... ({len(log_lines)} lines)"
            except Exception as exc:
                logger.exception("[Training tab] Image training error")
                log_lines.append(f"ERROR: {exc}")
                if tenant_acquired:
                    _release_tenant(ctx, "Image training failed", tenant_job_id)
                _active_image_runner[0] = None
                _active_image_tenant_job_id[0] = ""
                yield gr.update(interactive=True), gr.update(interactive=False), "\n".join(log_lines), f"ERROR: {exc}"
                return

            if tenant_acquired:
                _release_tenant(ctx, "Image training complete", tenant_job_id)
            _active_image_runner[0] = None
            _active_image_tenant_job_id[0] = ""
            yield gr.update(interactive=True), gr.update(interactive=False), "\n".join(log_lines), "Image training complete."

        image_start_btn.click(
            fn=on_start_image,
            inputs=[
                image_engine_radio,
                dataset_dir,
                base_model,
                job_name,
                output_dir,
                max_steps_input,
                max_epochs_input,
                batch_size_input,
                lr_input,
                resolution_input,
                mixed_prec_input,
            ],
            outputs=[image_start_btn, image_stop_btn, image_log_box, image_status_bar],
        )

        def on_stop_image():
            runner = _active_image_runner[0]
            if runner is None:
                return gr.update(interactive=True), gr.update(interactive=False), "WARNING: No active image training job."
            try:
                msg = runner.stop()
            except Exception as exc:
                msg = f"Stop error: {exc}"
            _active_image_runner[0] = None
            tenant_job_id = _active_image_tenant_job_id[0]
            _active_image_tenant_job_id[0] = ""
            _release_tenant(ctx, "Image training stopped by user", tenant_job_id)
            return gr.update(interactive=True), gr.update(interactive=False), f"Stopped: {msg}"

        image_stop_btn.click(fn=on_stop_image, outputs=[image_start_btn, image_stop_btn, image_status_bar])
