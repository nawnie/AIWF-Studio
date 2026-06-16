"""
aiwf/web/tabs/training.py

Training tab â€” Kohya LoRA and EveryDream2 full fine-tuning.

Rules enforced here:
  - No torch imports.
  - No engine imports at module level (import inside callbacks only).
  - All engine errors surfaced as human-readable UI messages; full
    tracebacks go to the logger only.
  - GPU tenant lock acquired through EngineSupervisor.request_switch()
    before starting a training job.
  - Tab renders even if neither engine is installed â€” shows setup guidance.
"""
from __future__ import annotations

import logging

import gradio as gr

from aiwf.bootstrap import AppContext
from aiwf.core.domain.engine import EngineSwitchRequest, EngineTenant
from aiwf.services.training.dataset_validator import DatasetValidator
from aiwf.web.registry import WebRegistry

logger = logging.getLogger(__name__)

_validator = DatasetValidator()


# ---------------------------------------------------------------------------
# Engine availability probe â€” deferred, never at import time
# ---------------------------------------------------------------------------

def _probe_engines() -> dict[str, bool]:
    """Return availability of each training engine.  Never raises."""
    result = {"kohya": False, "ed2": False}
    try:
        from launch import _build_engine_registry, _load_engines_config, _engine_enabled  # type: ignore[import]
        cfg   = _load_engines_config()
        specs = {s.name: s for s in _build_engine_registry()}
        result["kohya"] = _engine_enabled("kohya", cfg, default=False) and specs.get("kohya", None) is not None and specs["kohya"].is_ready()
        result["ed2"]   = _engine_enabled("ed2", cfg, default=False)   and specs.get("ed2", None)   is not None and specs["ed2"].is_ready()
    except Exception:
        pass
    return result


_NOT_CONFIGURED = (
    "**Training engine not ready.**\n\n"
    "Install a training add-on below, then restart AIWF Studio.\n\n"
    "See `docs/TRAINING_ENGINE_ROADMAP.md` for details."
)

_NATIVE_LORA_NOT_READY = (
    "**Native LoRA / DreamBooth training is not installed yet.**\n\n"
    "Target coverage is SD 1.5, SD 2.x, SDXL, and SD 3.5. "
    "This path should be built as an AIWF-native trainer, not a Kohya wrapper."
)


def _format_validation_result(result) -> str:
    lines: list[str] = []
    if result.ok:
        lines.append("OK: Dataset looks good.")
    else:
        lines.extend(f"ERROR: {err}" for err in result.errors)
    lines.extend(f"WARNING: {warning}" for warning in result.warnings)
    return "\n\n".join(lines)


def _validate_training_request(engine: str, request: dict):
    if "Kohya" in engine or "LoRA" in engine:
        return _validator.validate_kohya(request)
    return _validator.validate_ed2(request)


def _release_tenant(ctx: AppContext, reason: str) -> None:
    supervisor = getattr(ctx, "supervisor", None)
    if supervisor is not None:
        supervisor.request_switch(EngineSwitchRequest(target=EngineTenant.IDLE, reason=reason))


# ---------------------------------------------------------------------------
# Tab registration
# ---------------------------------------------------------------------------

def register_training(registry: WebRegistry) -> None:

    @registry.tab("Training", order=20)
    def build(ctx: AppContext, tab: gr.Tab | None = None) -> None:

        engines = _probe_engines()
        any_ready = any(engines.values())

        not_configured_note = gr.Markdown(
            value=_NOT_CONFIGURED if not any_ready else "",
            visible=not any_ready,
        )
        with gr.Row():
            enable_full_btn = gr.Button(
                "Enable Full Training",
                variant="secondary",
                interactive=not engines["ed2"],
            )
            enable_lora_btn = gr.Button("Enable LoRA / DreamBooth Training", variant="secondary")
        training_enable_status = gr.Markdown(
            value="Full training is enabled." if engines["ed2"] else "",
        )

        # ---- Mode picker ----
        engine_choices = []
        if engines["kohya"]:
            engine_choices.append("AIWF LoRA / DreamBooth")
        if engines["ed2"]:
            engine_choices.append("ED2 Full Fine-tune")
        if not engine_choices:
            engine_choices = ["AIWF LoRA / DreamBooth", "ED2 Full Fine-tune"]   # show disabled UI

        engine_radio = gr.Radio(
            choices=engine_choices,
            value=engine_choices[0],
            label="Training engine",
            interactive=any_ready,
        )

        # ---- Dataset ----
        with gr.Group():
            gr.Markdown("### Dataset")
            with gr.Row():
                dataset_dir = gr.Textbox(
                    label="Dataset directory",
                    placeholder="C:\\training\\my_concept",
                    scale=8,
                )
                validate_btn = gr.Button("Validate", variant="secondary", scale=1)
            validation_result = gr.Markdown(value="")

        # ---- Base model ----
        with gr.Group():
            gr.Markdown("### Base model")
            base_model = gr.Textbox(
                label="Base model path or HuggingFace ID",
                placeholder="C:\\models\\sdxl_base.safetensors  or  stabilityai/stable-diffusion-xl-base-1.0",
            )

        # ---- Output ----
        with gr.Group():
            gr.Markdown("### Output")
            with gr.Row():
                job_name   = gr.Textbox(label="Job name", value="my_lora", scale=4)
                output_dir = gr.Textbox(
                    label="Output directory",
                    value="outputs/training",
                    scale=6,
                )

        # ---- Training parameters ----
        with gr.Accordion("Training parameters", open=True):
            with gr.Row():
                max_steps_input  = gr.Number(label="Max steps (LoRA / DreamBooth)", value=1500, precision=0, minimum=100, maximum=100_000)
                max_epochs_input = gr.Number(label="Max epochs (ED2)",  value=20,   precision=0, minimum=1,   maximum=1000)
            with gr.Row():
                batch_size_input = gr.Number(label="Batch size",        value=1, precision=0, minimum=1, maximum=64)
                lr_input         = gr.Number(label="Learning rate",     value=1e-4, step=1e-5)
            with gr.Row():
                resolution_input = gr.Dropdown(
                    choices=[512, 768, 1024],
                    value=1024,
                    label="Resolution",
                )
                mixed_prec_input = gr.Dropdown(
                    choices=["bf16", "fp16", "no"],
                    value="bf16",
                    label="Mixed precision",
                )

        # ---- Controls ----
        with gr.Row():
            start_btn = gr.Button(
                "â–¶ Start Training",
                variant="primary",
                interactive=any_ready,
            )
            stop_btn = gr.Button(
                "â¹ Stop",
                variant="stop",
                interactive=False,
            )

        # ---- Log output ----
        log_box = gr.Textbox(
            label="Training log",
            lines=20,
            max_lines=20,
            interactive=False,
            autoscroll=True,
        )

        status_bar = gr.Markdown(value="")

        # ----------------------------------------------------------------
        # Callbacks
        # ----------------------------------------------------------------

        def on_validate(dataset_path: str):
            if not dataset_path.strip():
                return "WARNING: Enter a dataset directory path first."
            result = _validator.validate_dataset_dir(dataset_path.strip())
            return _format_validation_result(result)

        validate_btn.click(fn=on_validate, inputs=[dataset_dir], outputs=[validation_result])

        # State: active runner reference (kept in a list so we can mutate from closure)
        _active_runner: list = [None]

        def on_enable_full_training():
            try:
                from aiwf.services.training.ed2_installer import install_ed2_addon

                lines = install_ed2_addon()
                return "\n\n".join([*lines, "Restart AIWF Studio to refresh training engine availability."])
            except Exception as exc:
                logger.exception("[Training tab] ED2 add-on install failed")
                return f"ERROR: Full training enable failed: {exc}"

        def on_enable_lora_training():
            return _NATIVE_LORA_NOT_READY

        enable_full_btn.click(fn=on_enable_full_training, outputs=[training_enable_status])
        enable_lora_btn.click(fn=on_enable_lora_training, outputs=[training_enable_status])

        def on_start(
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
            yield (
                gr.update(interactive=False),
                gr.update(interactive=True),
                "Starting training...\n",
                "Starting...",
            )

            if not ds_dir.strip():
                yield gr.update(interactive=True), gr.update(interactive=False), "ERROR: Dataset directory is required.\n", "ERROR"
                return
            if not base_mdl.strip():
                yield gr.update(interactive=True), gr.update(interactive=False), "ERROR: Base model path is required.\n", "ERROR"
                return

            req: dict = {
                "job_name": jname or "lora_job",
                "base_model_path": base_mdl.strip(),
                "dataset_dir": ds_dir.strip(),
                "output_dir": out_dir.strip() or "outputs/training",
                "resolution": int(res),
                "mixed_precision": mp,
                "batch_size": int(bs),
                "seed": 42,
            }

            if "LoRA" in engine or "DreamBooth" in engine:
                req["max_train_steps"] = int(steps)
                req["learning_rate"] = float(lr)
                req["base_arch"] = "sdxl"
            elif "ED2" in engine:
                req["max_epochs"] = int(epochs)
                req["lr"] = float(lr)

            validation = _validate_training_request(engine, req)
            if not validation.ok:
                yield (
                    gr.update(interactive=True),
                    gr.update(interactive=False),
                    _format_validation_result(validation),
                    "ERROR: Preflight failed",
                )
                return

            supervisor = getattr(ctx, "supervisor", None)
            tenant = EngineTenant.LORA_TRAINING if "LoRA" in engine else EngineTenant.FULL_TRAINING
            tenant_acquired = False
            if supervisor is not None:
                result = supervisor.request_switch(EngineSwitchRequest(target=tenant, reason=f"Training: {jname}"))
                if not result.ok:
                    yield gr.update(interactive=True), gr.update(interactive=False), f"ERROR: GPU busy: {result.message}\n", "ERROR: GPU busy"
                    return
                tenant_acquired = True

            log_lines: list[str] = []
            try:
                if "LoRA" in engine or "DreamBooth" in engine:
                    raise RuntimeError("Native LoRA / DreamBooth training is not implemented yet.")
                elif "ED2" in engine:
                    from aiwf.services.training.ed2_runner import ED2Runner  # type: ignore

                    runner = ED2Runner()
                else:
                    raise RuntimeError(f"Unknown training engine: {engine}")

                _active_runner[0] = runner
                for line in runner.start(req):
                    log_lines.append(line)
                    if len(log_lines) % 5 == 0:
                        yield (
                            gr.update(interactive=False),
                            gr.update(interactive=True),
                            "\n".join(log_lines[-200:]),
                            f"Training... ({len(log_lines)} lines)",
                        )
            except Exception as exc:
                logger.exception("[Training tab] Training error")
                log_lines.append(f"ERROR: {exc}")
                if tenant_acquired:
                    _release_tenant(ctx, "Training failed")
                _active_runner[0] = None
                yield gr.update(interactive=True), gr.update(interactive=False), "\n".join(log_lines), f"ERROR: {exc}"
                return

            if tenant_acquired:
                _release_tenant(ctx, "Training complete")

            _active_runner[0] = None
            yield (
                gr.update(interactive=True),
                gr.update(interactive=False),
                "\n".join(log_lines),
                "Training complete.",
            )
        start_btn.click(
            fn=on_start,
            inputs=[
                engine_radio,
                dataset_dir, base_model, job_name, output_dir,
                max_steps_input, max_epochs_input,
                batch_size_input, lr_input, resolution_input, mixed_prec_input,
            ],
            outputs=[start_btn, stop_btn, log_box, status_bar],
        )

        def on_stop():
            runner = _active_runner[0]
            if runner is None:
                return gr.update(interactive=True), gr.update(interactive=False), "WARNING: No active job."
            try:
                msg = runner.stop()
            except Exception as exc:
                msg = f"Stop error: {exc}"
            _active_runner[0] = None
            _release_tenant(ctx, "Training stopped by user")

            return gr.update(interactive=True), gr.update(interactive=False), f"Stopped: {msg}"
        stop_btn.click(
            fn=on_stop,
            outputs=[start_btn, stop_btn, status_bar],
        )
