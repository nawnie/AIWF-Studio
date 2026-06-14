from __future__ import annotations

import json
import logging
import os
import sys
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

from aiwf import __version__ as AIWF_VERSION
from aiwf.core.domain.generation import GenerationMode, GenerationRequest, JobState
from aiwf.core.events.types import (
    AfterGenerate,
    AppStarted,
    BeforeGenerate,
    JobCancelled,
    JobFailed,
    JobFinished,
    JobProgressed,
    JobQueued,
    JobStarted,
)

if TYPE_CHECKING:
    from aiwf.bootstrap import AppContext

logger = logging.getLogger("aiwf.dev")

_installed: DevDiagnostics | None = None


def _enabled_by_default() -> bool:
    """Dev repo: trace on unless explicitly disabled."""
    return os.environ.get("AIWF_DEV_TRACE", "1").strip().lower() not in {"0", "false", "no", "off"}


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + "Z"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, GenerationMode):
        return value.value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _summarize_request(request: GenerationRequest) -> dict[str, Any]:
    cn_units = request.controlnet_units or []
    return {
        "mode": request.mode.value,
        "checkpoint_id": request.checkpoint_id,
        "vae_id": request.vae_id,
        "width": request.width,
        "height": request.height,
        "steps": request.steps,
        "sampler": request.sampler,
        "scheduler": request.scheduler,
        "seed": request.seed,
        "batch_size": request.batch_size,
        "batch_count": request.batch_count,
        "enable_hr": request.enable_hr,
        "inpaint_only_masked": request.inpaint_only_masked,
        "controlnet_count": len(cn_units),
        "controlnet_enabled": any(unit.enabled for unit in cn_units),
        "prompt_len": len(request.prompt or ""),
        "negative_len": len(request.negative_prompt or ""),
    }


def trace_model_throughput(
    *,
    kind: str,
    elapsed_seconds: float,
    units: int,
    units_label: str,
    model_id: str | None = None,
    model_name: str | None = None,
    app_version: str | None = None,
    **fields: Any,
) -> None:
    """Record a model run speed snapshot in the structured dev trace log."""
    elapsed = float(elapsed_seconds or 0.0)
    rate = float(units) / elapsed if elapsed > 0 else None
    trace_safe(
        "model.rate",
        "Model throughput recorded",
        app_version=app_version or AIWF_VERSION,
        kind=kind,
        model_id=model_id,
        model_name=model_name,
        elapsed_seconds=round(elapsed, 3),
        units=int(units),
        units_label=units_label,
        units_per_second=round(rate, 3) if rate is not None else None,
        **fields,
    )


class DevDiagnostics:
    """Structured dev trace log — writes to outputs/dev-trace.log and aiwf.dev logger."""

    def __init__(self, output_dir: Path, *, enabled: bool | None = None) -> None:
        self.enabled = _enabled_by_default() if enabled is None else enabled
        self._output_dir = output_dir
        self._log_path = output_dir / "dev-trace.log"
        self._lock = threading.Lock()
        if self.enabled:
            output_dir.mkdir(parents=True, exist_ok=True)

    def trace(self, category: str, message: str, **fields: Any) -> None:
        if not self.enabled:
            return
        payload = {
            "time": _utc_stamp(),
            "category": category,
            "message": message,
            **{k: _json_safe(v) for k, v in fields.items() if v is not None},
        }
        line = json.dumps(payload, ensure_ascii=False)
        logger.info("%s | %s", category, message)
        with self._lock:
            with self._log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def trace_exception(
        self,
        category: str,
        exc: BaseException,
        *,
        message: str | None = None,
        **fields: Any,
    ) -> None:
        if not self.enabled:
            return
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self.trace(
            category,
            message or str(exc),
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=tb[-8000:],
            **fields,
        )


def get_diagnostics() -> DevDiagnostics | None:
    return _installed


def trace_safe(category: str, message: str = "", **fields: Any) -> None:
    diag = _installed
    if diag is None:
        return
    if message:
        diag.trace(category, message, **fields)
    elif fields:
        diag.trace(category, category, **fields)


def trace_exception_safe(
    category: str,
    exc: BaseException,
    *,
    message: str | None = None,
    **fields: Any,
) -> None:
    diag = _installed
    if diag is None:
        return
    diag.trace_exception(category, exc, message=message, **fields)


def _subscribe_job_events(diag: DevDiagnostics, ctx: AppContext) -> None:
    def on_queued(event: JobQueued) -> None:
        diag.trace(
            "job.queued",
            "Job queued",
            job_id=str(event.job_id),
            **_summarize_request(event.request),
        )

    def on_started(event: JobStarted) -> None:
        active = ctx.generation.active_job()
        diag.trace(
            "job.started",
            "Job started",
            job_id=str(event.job_id),
            queue_active=str(active.id) if active else None,
            **_summarize_request(event.request),
        )

    def on_progress(event: JobProgressed) -> None:
        if event.step == 1 or event.step == event.total_steps or event.step % 5 == 0:
            diag.trace(
                "job.progress",
                event.message,
                job_id=str(event.job_id),
                step=event.step,
                total_steps=event.total_steps,
            )

    def on_before(event: BeforeGenerate) -> None:
        diag.trace(
            "generation.before",
            "Before generate",
            job_id=str(event.job_id),
            **_summarize_request(event.request),
        )

    def on_after(event: AfterGenerate) -> None:
        diag.trace(
            "generation.after",
            "After generate",
            job_id=str(event.job_id),
            image_count=len(event.result.images),
            artifact_count=len(event.result.artifacts),
        )

    def on_finished(event: JobFinished) -> None:
        diag.trace(
            "job.finished",
            "Job completed",
            job_id=str(event.job_id),
            image_count=len(event.result.images),
        )

    def on_cancelled(event: JobCancelled) -> None:
        diag.trace("job.cancelled", "Job cancelled", job_id=str(event.job_id))

    def on_failed(event: JobFailed) -> None:
        diag.trace(
            "job.failed",
            "Job failed",
            job_id=str(event.job_id),
            error=event.error,
        )

    ctx.events.subscribe(JobQueued, on_queued)
    ctx.events.subscribe(JobStarted, on_started)
    ctx.events.subscribe(JobProgressed, on_progress)
    ctx.events.subscribe(BeforeGenerate, on_before)
    ctx.events.subscribe(AfterGenerate, on_after)
    ctx.events.subscribe(JobFinished, on_finished)
    ctx.events.subscribe(JobCancelled, on_cancelled)
    ctx.events.subscribe(JobFailed, on_failed)


def _install_global_hooks(diag: DevDiagnostics) -> None:
    def _thread_excepthook(args: threading.ExceptHookArgs) -> None:
        if args.exc_value is None:
            return
        diag.trace_exception(
            "thread.uncaught",
            args.exc_value,
            thread=args.thread.name if args.thread else "unknown",
        )

    if hasattr(threading, "excepthook"):
        previous = threading.excepthook

        def _chained(args: threading.ExceptHookArgs) -> None:
            _thread_excepthook(args)
            if previous is not None:
                previous(args)

        threading.excepthook = _chained

    previous_sys = sys.excepthook

    def _sys_excepthook(exc_type, exc_value, exc_tb) -> None:
        if exc_value is not None:
            diag.trace_exception("process.uncaught", exc_value)
        previous_sys(exc_type, exc_value, exc_tb)

    sys.excepthook = _sys_excepthook


def install_dev_diagnostics(ctx: AppContext) -> DevDiagnostics:
    global _installed
    output_dir = ctx.flags.resolved_output_dir()
    diag = DevDiagnostics(output_dir)
    _installed = diag
    _subscribe_job_events(diag, ctx)
    _install_global_hooks(diag)

    def on_app_started(_event: AppStarted) -> None:
        diag.trace(
            "app.started",
            "AIWF Studio started",
            app_version=AIWF_VERSION,
            data_dir=str(ctx.flags.data_dir),
            output_dir=str(output_dir),
            port=ctx.runtime_port,
            last_checkpoint=ctx.settings.last_checkpoint_id,
        )

    ctx.events.subscribe(AppStarted, on_app_started)
    return diag


def trace_studio_generate(
    *,
    run_number: int,
    mode_label: str,
    continuous: bool,
    editing_mask: bool,
    has_source: bool,
    has_editor_value: bool,
    cn_enabled: bool,
    input_count: int,
) -> None:
    trace_safe(
        "studio.generate",
        "Studio generate invoked",
        run_number=run_number,
        mode=mode_label,
        continuous=continuous,
        editing_mask=editing_mask,
        has_source=has_source,
        has_editor_value=has_editor_value,
        cn_enabled=cn_enabled,
        input_count=input_count,
    )


def trace_studio_request_built(
    *,
    mode: str,
    width: int | None = None,
    height: int | None = None,
    init_count: int = 0,
    mask_count: int = 0,
    control_count: int = 0,
    checkpoint_id: str | None = None,
) -> None:
    trace_safe(
        "studio.request",
        "Generation request built",
        mode=mode,
        width=width,
        height=height,
        init_count=init_count,
        mask_count=mask_count,
        control_count=control_count,
        checkpoint_id=checkpoint_id,
    )


def trace_job_record_state(job_id: UUID, state: JobState, error: str | None = None) -> None:
    trace_safe(
        "studio.job_state",
        f"Job {state.value}",
        job_id=str(job_id),
        state=state.value,
        error=error,
    )
