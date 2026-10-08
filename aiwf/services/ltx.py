from __future__ import annotations

import json
import logging
import os
import random
import struct
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.engine import EngineTenant
from aiwf.core.domain.errors import GenerationCancelledError
from aiwf.core.domain.ltx import (
    LTX_DISTILLED_CHECKPOINT,
    LTX_DIFFUSERS_2B_CHECKPOINT,
    LTX_FULL_CHECKPOINT,
    LTX_FULL_CHECKPOINT_FP8,
    LTX_GEMMA_BACKEND_GGUF,
    LTX_GEMMA_BACKEND_HF_SAFETENSORS,
    LTX_GEMMA_REPO,
    LTX_HERETIC_Q3_CONVERTED_FOLDER,
    LTX_HERETIC_Q3_GGUF,
    LTX_PIPELINE_DIFFUSERS_2B,
    LTX_PIPELINE_DISTILLED,
    LTX_PIPELINE_ONE_STAGE,
    LTX_SPATIAL_UPSCALER_X2,
    LTX_T5_TOKENIZER,
    LTX_T5XXL_FP16,
    LtxVideoRequest,
    LtxVideoResult,
)
from aiwf.services.engine_supervisor import EngineSupervisor
from aiwf.infrastructure.video.processing import VideoProcessor
from aiwf.services.model_files import indexed_safetensors_shards_ready, resolve_model_asset
from aiwf.services.process_supervisor import ProcessSupervisor, get_process_supervisor
from aiwf.services.worker_tenant import WorkerTenantRegistry

logger = logging.getLogger(__name__)


class LtxUnavailable(RuntimeError):
    pass


class LtxService:
    def __init__(
        self,
        flags: RuntimeFlags | None = None,
        settings: UserSettings | None = None,
        *,
        registry: WorkerTenantRegistry | None = None,
        supervisor: EngineSupervisor | None = None,
        process_supervisor: ProcessSupervisor | None = None,
    ) -> None:
        self.flags = flags or RuntimeFlags()
        self.settings = settings or UserSettings()
        self.registry = registry or WorkerTenantRegistry(self.flags.data_dir)
        self.supervisor = supervisor
        self.process_supervisor = process_supervisor or get_process_supervisor()
        self._generation_lock = threading.Lock()
        self._generation_cancel = threading.Event()
        self._active_worker_id: str | None = None
        self._generation_prepared = False

    def cancel_active_generation(self) -> bool:
        """Request cancellation of the current LTX job, stopping worker variants promptly."""
        self._generation_cancel.set()
        with self._generation_lock:
            worker_id = self._active_worker_id
        if worker_id:
            def _stop_worker() -> None:
                # The supervisor registers the subprocess just after start(); allow that
                # small race to settle before giving up on immediate process-tree stop.
                for _ in range(20):
                    if self.process_supervisor.is_running(worker_id):
                        self.process_supervisor.stop(worker_id)
                        return
                    time.sleep(0.05)
            threading.Thread(target=_stop_worker, name="aiwf-ltx-cancel", daemon=True).start()
        return True

    def begin_generation(self) -> None:
        """Clear the prior job's token before publishing a new job as running."""
        with self._generation_lock:
            self._generation_cancel.clear()
            self._generation_prepared = True

    def models_root(self) -> Path:
        return self.flags.resolved_models_dir() / "ltx"

    def default_checkpoint_path(self, pipeline: str = LTX_PIPELINE_DISTILLED) -> Path:
        root = self.models_root() / "checkpoints"
        if pipeline == LTX_PIPELINE_DIFFUSERS_2B:
            name = LTX_DIFFUSERS_2B_CHECKPOINT
        elif pipeline == LTX_PIPELINE_DISTILLED:
            name = LTX_DISTILLED_CHECKPOINT
        else:
            fp8 = self._resolve_ltx_asset((Path("checkpoints") / LTX_FULL_CHECKPOINT_FP8,))
            name = LTX_FULL_CHECKPOINT_FP8 if _nonempty_file(fp8) else LTX_FULL_CHECKPOINT
        relative = Path("checkpoints") / name
        return self._resolve_ltx_asset((relative,))

    def _resolve_ltx_asset(self, relative_candidates, *, predicate=None) -> Path:  # noqa: ANN001
        return resolve_model_asset(
            self.flags,
            (Path("ltx") / relative for relative in relative_candidates),
            predicate=predicate,
            fallback=self.models_root() / next(iter(relative_candidates)),
        )

    def default_t5_encoder_path(self) -> Path:
        return resolve_model_asset(
            self.flags,
            (
                Path("flux") / "Textencoder" / LTX_T5XXL_FP16,
                Path("Textencoder") / LTX_T5XXL_FP16,
                Path("text_encoders") / LTX_T5XXL_FP16,
            ),
            fallback=self.flags.resolved_models_dir() / "flux" / "Textencoder" / LTX_T5XXL_FP16,
        )

    def default_t5_tokenizer_path(self) -> Path:
        return self._resolve_ltx_asset(
            (Path("tokenizer") / "t5-v1_1-xxl",),
            predicate=ltx_t5_tokenizer_ready,
        )

    def default_launch_pipeline(self) -> str:
        if (
            _nonempty_file(self.default_checkpoint_path(LTX_PIPELINE_DIFFUSERS_2B))
            and _nonempty_file(self.default_t5_encoder_path())
            and ltx_t5_tokenizer_ready(self.default_t5_tokenizer_path())
        ):
            return LTX_PIPELINE_DIFFUSERS_2B
        one_stage = self.default_checkpoint_path(LTX_PIPELINE_ONE_STAGE)
        if (
            _nonempty_file(one_stage)
            and not ltx_checkpoint_openability_error(one_stage)
            and not ltx_native_checkpoint_runtime_blocker(one_stage)
        ):
            return LTX_PIPELINE_ONE_STAGE
        distilled = self.default_checkpoint_path(LTX_PIPELINE_DISTILLED)
        if (
            _nonempty_file(distilled)
            and not ltx_checkpoint_openability_error(distilled)
            and not ltx_native_checkpoint_runtime_blocker(distilled)
        ):
            return LTX_PIPELINE_DISTILLED
        return LTX_PIPELINE_DISTILLED

    def default_launch_request(self) -> LtxVideoRequest:
        return LtxVideoRequest(pipeline=self.default_launch_pipeline())

    def default_spatial_upsampler_path(self) -> Path:
        return self._resolve_ltx_asset((Path("upscalers") / LTX_SPATIAL_UPSCALER_X2,))

    def default_official_gemma_root(self) -> Path:
        return self._resolve_ltx_asset(
            (Path("text_encoder") / LTX_GEMMA_REPO.split("/", 1)[1],),
            predicate=ltx_gemma_hf_assets_ready,
        )

    def default_heretic_converted_gemma_root(self) -> Path:
        return self._resolve_ltx_asset(
            (Path("text_encoder") / LTX_HERETIC_Q3_CONVERTED_FOLDER,),
            predicate=_converted_heretic_gemma_ready,
        )

    def default_gemma_root(self) -> Path:
        heretic = self.default_heretic_converted_gemma_root()
        if _converted_heretic_gemma_ready(heretic):
            return heretic
        return self.default_official_gemma_root()

    def default_gemma_gguf_path(self) -> Path:
        return resolve_model_asset(
            self.flags,
            (
                Path("ltx") / "GGUF" / LTX_HERETIC_Q3_GGUF,
                Path("LLM") / "GGUF" / LTX_HERETIC_Q3_GGUF,
                Path("llm") / "gguf" / LTX_HERETIC_Q3_GGUF,
            ),
            fallback=self.models_root() / "GGUF" / LTX_HERETIC_Q3_GGUF,
        )

    def output_dir(self) -> Path:
        return self.flags.resolved_output_dir() / "ltx-videos"

    def status_markdown(self) -> str:
        try:
            status = self.registry.status("ltx")
            lines = [status.markdown_line()]
        except KeyError:
            lines = ["**LTX 2.3:** not registered."]

        launch_pipeline = self.default_launch_pipeline()
        lines.append(f"- Default launch pipeline: `{launch_pipeline}`")
        lines.append(f"- Default checkpoint: `{self.default_checkpoint_path(launch_pipeline)}`")
        lines.append(f"- Default upscaler: `{self.default_spatial_upsampler_path()}`")
        lines.append(f"- Default Gemma root: `{self.default_gemma_root()}`")
        lines.append(f"- Default Heretic GGUF: `{self.default_gemma_gguf_path()}`")
        lines.append(f"- Default LTX 2B T5XXL: `{self.default_t5_encoder_path()}`")
        return "\n".join(lines)

    def generate(self, request: LtxVideoRequest, *, on_progress=None) -> LtxVideoResult:
        with self._generation_lock:
            if not self._generation_prepared:
                self._generation_cancel.clear()
            self._generation_prepared = False
        normalized = self._resolve_request(request)
        if normalized.get("pipeline") == LTX_PIPELINE_DIFFUSERS_2B:
            return self._generate_diffusers_2b(normalized, on_progress=on_progress)

        status = self.registry.status("ltx")
        if not status.ready:
            details = "; ".join(status.messages) if status.messages else "not ready"
            raise LtxUnavailable(
                "LTX 2.3 engine is not ready. Run `scripts/bootstrap_ltx.ps1 -Enable`, "
                f"then refresh Settings. Details: {details}"
            )

        self._validate_request_paths(normalized)
        headroom_issue = self._ltx_worker_headroom_issue(normalized)
        if headroom_issue:
            raise LtxUnavailable(headroom_issue)

        job_id = f"ltx_{uuid4().hex[:8]}"
        request_path = self._write_worker_request(job_id, normalized)
        env = {
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get(
                "PYTORCH_CUDA_ALLOC_CONF",
                "expandable_segments:True",
            ),
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        }
        command = self.registry.build_command(
            "ltx",
            request_path,
            env=env,
            cwd=status.repo_dir or self.flags.data_dir,
        )

        events: list[dict] = []
        output_path = Path(normalized["output_path"])
        error_message = ""
        try:
            if self.supervisor is not None:
                with self.supervisor.tenant_session(
                    EngineTenant.VIDEO,
                    reason="LTX 2.3 video generation",
                    job_id=job_id,
                    allow_wait=False,
                ):
                    self._run_worker(job_id, command, events, on_progress=on_progress)
            else:
                self._run_worker(job_id, command, events, on_progress=on_progress)
        except Exception as exc:
            if isinstance(exc, GenerationCancelledError):
                raise
            error_message = str(exc)
            logger.exception("LTX 2.3 generation failed")

        terminal_error = _last_error(events)
        if error_message:
            status_tail = _interesting_status_tail(events)
            if status_tail and status_tail not in error_message:
                error_message = f"{error_message}\n\nRecent LTX output:\n{status_tail}"
            raise LtxUnavailable(_clip(error_message))
        if terminal_error:
            raise LtxUnavailable(terminal_error)
        if not output_path.is_file():
            raise LtxUnavailable(f"LTX worker finished but did not create output: {output_path}")
        has_audio = False
        try:
            has_audio = VideoProcessor().probe(output_path).has_audio
        except Exception:
            logger.debug("Could not probe LTX output audio stream", exc_info=True)

        return LtxVideoResult(
            output_path=str(output_path),
            message=f"LTX 2.3 video saved to {output_path.name}" + (" with native audio" if has_audio else ""),
            events=events,
            has_audio=has_audio,
            audio_mode="native",
        )

    def _generate_diffusers_2b(self, payload: dict, *, on_progress=None) -> LtxVideoResult:
        self._validate_request_paths(payload)
        source = payload.get("source_image_path")
        if source:
            raise LtxUnavailable("The local LTX 2B Diffusers route is text-to-video only; clear the source image.")

        from aiwf.services.ltx_diffusers import is_ltx2b_pipeline_cached, run_ltx2b_diffusers

        checkpoint = Path(str(payload["checkpoint_path"]))
        if not is_ltx2b_pipeline_cached(
            checkpoint=checkpoint,
            t5_weights=Path(str(payload["t5_encoder_path"])),
            tokenizer_id=str(payload.get("t5_tokenizer") or LTX_T5_TOKENIZER),
        ):
            headroom_issue = self._ltx2b_headroom_issue(checkpoint)
            if headroom_issue:
                raise LtxUnavailable(headroom_issue)

        job_id = f"ltx2b_{uuid4().hex[:8]}"
        output_path = Path(str(payload["output_path"]))

        def _run():
            return run_ltx2b_diffusers(
                checkpoint=Path(str(payload["checkpoint_path"])),
                t5_weights=Path(str(payload["t5_encoder_path"])),
                tokenizer_id=str(payload.get("t5_tokenizer") or LTX_T5_TOKENIZER),
                output=output_path,
                prompt=str(payload.get("prompt") or ""),
                negative_prompt=str(payload.get("negative_prompt") or ""),
                width=int(payload.get("width") or 128),
                height=int(payload.get("height") or 128),
                frames=int(payload.get("num_frames") or 9),
                fps=int(round(float(payload.get("fps") or 8))),
                steps=int(payload.get("steps") or 1),
                seed=int(payload.get("seed") or 0),
                on_progress=on_progress,
                should_cancel=self._generation_cancel.is_set,
            )

        try:
            if self.supervisor is not None:
                with self.supervisor.tenant_session(
                    EngineTenant.VIDEO,
                    reason="LTX 2B Diffusers video generation",
                    job_id=job_id,
                    allow_wait=False,
                ):
                    result = _run()
            else:
                result = _run()
        except Exception as exc:
            logger.exception("LTX 2B Diffusers generation failed")
            if isinstance(exc, GenerationCancelledError):
                raise
            raise LtxUnavailable(f"LTX 2B Diffusers generation failed: {exc}") from exc

        return LtxVideoResult(
            output_path=str(result.output_path),
            message=(
                f"LTX 2B Diffusers video saved to {result.output_path.name} "
                f"({result.width}x{result.height}, {result.frame_count} frames, {result.fps} fps"
                f", {'cache hit' if result.cache_hit else 'loaded pipeline'})"
            ),
            events=[
                {
                    "kind": "complete",
                    "message": "LTX 2B Diffusers generation complete",
                    "bytes": result.bytes,
                    "cache_hit": result.cache_hit,
                }
            ],
            has_audio=False,
            audio_mode="none",
        )

    def prepare(self, request: LtxVideoRequest) -> dict:
        """Load the selected in-process LTX 2B pipeline without starting generation.

        Other LTX variants are worker/subprocess routes and only receive a
        readiness check here; their worker loads weights when a job starts.
        """
        normalized = self._resolve_request(request)
        if normalized.get("pipeline") != LTX_PIPELINE_DIFFUSERS_2B:
            self._validate_request_paths(normalized)
            return {
                "loaded": False,
                "resident": None,
                "detail": "LTX worker route passed local-file checks; its worker loads weights when generation starts.",
            }

        self._validate_request_paths(normalized)
        if normalized.get("source_image_path"):
            raise LtxUnavailable("The local LTX 2B Diffusers route is text-to-video only; clear the source image.")
        from aiwf.services.ltx_diffusers import is_ltx2b_pipeline_cached, load_ltx2b_pipeline

        checkpoint = Path(str(normalized["checkpoint_path"]))
        t5_weights = Path(str(normalized["t5_encoder_path"]))
        tokenizer_id = str(normalized.get("t5_tokenizer") or LTX_T5_TOKENIZER)
        already_resident = is_ltx2b_pipeline_cached(
            checkpoint=checkpoint,
            t5_weights=t5_weights,
            tokenizer_id=tokenizer_id,
        )
        if not already_resident:
            headroom_issue = self._ltx2b_headroom_issue(checkpoint)
            if headroom_issue:
                raise LtxUnavailable(headroom_issue)
        job_id = f"ltx2b_prepare_{uuid4().hex[:8]}"
        try:
            if self.supervisor is not None:
                with self.supervisor.tenant_session(
                    EngineTenant.VIDEO,
                    reason="LTX 2B Diffusers model selection",
                    job_id=job_id,
                    allow_wait=False,
                ):
                    load_ltx2b_pipeline(
                        checkpoint=checkpoint,
                        t5_weights=t5_weights,
                        tokenizer_id=tokenizer_id,
                    )
            else:
                load_ltx2b_pipeline(
                    checkpoint=checkpoint,
                    t5_weights=t5_weights,
                    tokenizer_id=tokenizer_id,
                )
        except Exception as exc:
            logger.exception("LTX 2B Diffusers selection-time load failed")
            raise LtxUnavailable(f"LTX 2B Diffusers model load failed: {exc}") from exc

        resident = is_ltx2b_pipeline_cached(
            checkpoint=checkpoint,
            t5_weights=t5_weights,
            tokenizer_id=tokenizer_id,
        )
        if not resident:
            raise LtxUnavailable("LTX 2B Diffusers loader returned without confirming the selected pipeline is resident.")
        return {
            "loaded": True,
            "resident": True,
            "modelId": str(checkpoint),
            "detail": "LTX 2B Diffusers pipeline and T5XXL support assets are loaded and resident. No generation was run.",
        }

    def _ltx2b_headroom_issue(self, checkpoint: Path) -> str | None:
        """Apply the Studio model-load VRAM policy before an uncached LTX 2B load."""
        try:
            import torch

            if not torch.cuda.is_available():
                return "LTX 2B loading deferred because available GPU memory could not be checked."
            from aiwf.services.gpu_memory import measured_cuda_free_bytes

            free_bytes = measured_cuda_free_bytes(torch)
            if free_bytes is None:
                return "LTX 2B loading deferred because available GPU memory could not be matched to the active CUDA device."
            size_bytes = max(0, checkpoint.stat().st_size)
            estimated_gb = size_bytes / 1024**3
            low_memory_profile = bool(
                getattr(self.flags, "lowvram", False) or getattr(self.flags, "medvram", False)
            )
            required_gb = 3.0 if low_memory_profile else max(3.5, min(12.0, estimated_gb * 0.75 + 1.5))
            free_gb = float(free_bytes) / 1024**3
        except Exception:
            return "LTX 2B loading deferred because available GPU memory could not be checked."
        if free_gb < required_gb:
            return (
                f"LTX 2B loading deferred: {free_gb:.1f} GB VRAM is free; "
                f"this model needs an estimated {required_gb:.1f} GB headroom."
            )
        return None

    def _ltx_worker_headroom_issue(self, payload: dict) -> str | None:
        """Gate external LTX worker launches on device-wide free VRAM."""
        try:
            # The isolated worker has its own CUDA-enabled environment, so the
            # Studio process's Torch/CUDA visibility is not authoritative here.
            from aiwf.services.gpu_memory import nvidia_smi_free_bytes

            free_bytes = nvidia_smi_free_bytes()
        except Exception:
            free_bytes = None
        if free_bytes is None:
            return "LTX worker launch deferred because available GPU memory could not be verified."

        offload = str(payload.get("offload") or "disk").strip().lower()
        required_gb = 6.0
        if offload == "none":
            checkpoint = Path(str(payload.get("checkpoint_path") or ""))
            try:
                model_gb = max(0, checkpoint.stat().st_size) / 1024**3
            except OSError:
                model_gb = 0.0
            required_gb = max(6.0, min(14.0, model_gb * 0.75 + 2.0))
        free_gb = float(free_bytes) / 1024**3
        if free_gb < required_gb:
            return (
                f"LTX worker launch deferred: {free_gb:.1f} GB VRAM is free; "
                f"the selected {offload} route needs at least {required_gb:.1f} GB headroom."
            )
        return None

    def unload(self) -> bool:
        """Release the retained Diffusers 2B pipeline, if one is cached."""
        from aiwf.services.ltx_diffusers import unload_ltx2b_diffusers_cache
        if self.supervisor is None:
            return unload_ltx2b_diffusers_cache()
        with self.supervisor.tenant_session(
            EngineTenant.VIDEO,
            reason="Release cached LTX 2B Diffusers model",
            job_id=f"ltx2b_unload_{uuid4().hex[:8]}",
            allow_wait=False,
        ):
            return unload_ltx2b_diffusers_cache()

    def probe_gemma_gguf(self, request: LtxVideoRequest | None = None) -> list[dict]:
        """Run the isolated worker's native-GGUF text-encoder feasibility probe.

        This does not run video generation or dequantize weights. It verifies
        path wiring and reports whether the LTX worker has a backend that can
        return the full Gemma hidden-state tuple LTX requires.
        """

        status = self.registry.status("ltx")
        if not status.ready:
            details = "; ".join(status.messages) if status.messages else "not ready"
            raise LtxUnavailable(
                "LTX 2.3 engine is not ready. Run `scripts/bootstrap_ltx.ps1 -Enable`, "
                f"then refresh Settings. Details: {details}"
            )

        base_request = request or LtxVideoRequest()
        if base_request.gemma_backend != LTX_GEMMA_BACKEND_GGUF:
            base_request = base_request.model_copy(update={"gemma_backend": LTX_GEMMA_BACKEND_GGUF})
        normalized = self._resolve_request(base_request)
        self._validate_gemma_paths(normalized)

        job_id = f"ltx_probe_{uuid4().hex[:8]}"
        normalized["mode"] = "probe_gemma_gguf"
        request_path = self._write_worker_request(job_id, normalized)
        command = self.registry.build_command(
            "ltx",
            request_path,
            env={"HF_HUB_DISABLE_PROGRESS_BARS": "1"},
            cwd=status.repo_dir or self.flags.data_dir,
        )

        events: list[dict] = []
        error_message = ""
        try:
            self._run_worker(job_id, command, events)
        except Exception as exc:
            error_message = str(exc)
            logger.info("LTX Gemma GGUF probe failed: %s", exc)

        terminal_error = _last_error(events)
        if error_message:
            status_tail = _interesting_status_tail(events)
            if status_tail and status_tail not in error_message:
                error_message = f"{error_message}\n\nRecent LTX output:\n{status_tail}"
            raise LtxUnavailable(_clip(error_message))
        if terminal_error:
            raise LtxUnavailable(terminal_error)
        return events

    def _resolve_request(self, request: LtxVideoRequest) -> dict:
        pipeline = request.pipeline
        if not str(request.checkpoint_path or "").strip() and pipeline == LTX_PIPELINE_DISTILLED:
            fallback_pipeline = self.default_launch_pipeline()
            if fallback_pipeline != pipeline:
                pipeline = fallback_pipeline
        checkpoint = _resolve_path(
            request.checkpoint_path,
            self.default_checkpoint_path(pipeline),
            self.flags.data_dir,
        )
        t5_encoder_path = _resolve_path(
            request.t5_encoder_path,
            self.default_t5_encoder_path(),
            self.flags.data_dir,
        )
        spatial_upsampler = _resolve_path(
            request.spatial_upsampler_path,
            self.default_spatial_upsampler_path(),
            self.flags.data_dir,
        )
        gemma_root = _resolve_path(
            _gemma_root_text(request),
            self.default_gemma_root(),
            self.flags.data_dir,
        )
        gemma_backend = str(request.gemma_backend or LTX_GEMMA_BACKEND_HF_SAFETENSORS).strip().lower()
        gemma_gguf_raw = _gemma_gguf_text(request) if gemma_backend == LTX_GEMMA_BACKEND_GGUF else ""
        if str(request.gemma_root or "").strip().lower().endswith(".gguf"):
            gemma_backend = LTX_GEMMA_BACKEND_GGUF
            gemma_gguf_raw = _gemma_gguf_text(request)
        gemma_gguf_path = _resolve_optional_path(gemma_gguf_raw, self.flags.data_dir)
        if gemma_backend == LTX_GEMMA_BACKEND_GGUF and gemma_gguf_path is None:
            gemma_gguf_path = self.default_gemma_gguf_path().resolve()
        source_image = _resolve_optional_path(request.source_image_path, self.flags.data_dir)
        seed = int(request.seed)
        if seed < 0:
            seed = random.randint(0, 2**31 - 1)

        self.output_dir().mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        prefix = "ltx2b" if pipeline == LTX_PIPELINE_DIFFUSERS_2B else "ltx23"
        output_path = self.output_dir() / f"{prefix}-{stamp}-{uuid4().hex[:6]}.mp4"

        payload = request.model_dump()
        tokenizer_id = str(request.t5_tokenizer or LTX_T5_TOKENIZER)
        if tokenizer_id == LTX_T5_TOKENIZER:
            tokenizer_id = str(self.default_t5_tokenizer_path())
        payload.update(
            {
                "pipeline": pipeline,
                "checkpoint_path": str(checkpoint),
                "t5_encoder_path": str(t5_encoder_path),
                "t5_tokenizer": tokenizer_id,
                "spatial_upsampler_path": str(spatial_upsampler),
                "gemma_root": str(gemma_root),
                "gemma_backend": gemma_backend,
                "gemma_gguf_path": str(gemma_gguf_path) if gemma_gguf_path is not None else "",
                "source_image_path": str(source_image) if source_image is not None else None,
                "seed": seed,
                "output_path": str(output_path),
            }
        )
        _apply_native_runtime_profile(payload)
        return payload

    def _validate_request_paths(self, payload: dict) -> None:
        checkpoint = Path(str(payload.get("checkpoint_path") or ""))
        if payload.get("pipeline") == LTX_PIPELINE_DIFFUSERS_2B:
            if not _nonempty_file(checkpoint):
                raise LtxUnavailable(f"LTX 2B checkpoint missing or empty: {checkpoint}")
            t5_encoder = Path(str(payload.get("t5_encoder_path") or ""))
            if not _nonempty_file(t5_encoder):
                raise LtxUnavailable(f"LTX 2B T5XXL text encoder missing or empty: {t5_encoder}")
            tokenizer_path = Path(str(payload.get("t5_tokenizer") or ""))
            if not ltx_t5_tokenizer_ready(tokenizer_path):
                raise LtxUnavailable(f"LTX 2B T5 tokenizer is missing or incomplete: {tokenizer_path}")
            source = payload.get("source_image_path")
            if source and not Path(str(source)).is_file():
                raise LtxUnavailable(f"LTX source image missing: {source}")
            return
        if not checkpoint.is_file():
            raise LtxUnavailable(f"LTX checkpoint missing: {checkpoint}")
        openability_error = ltx_checkpoint_openability_error(checkpoint)
        if openability_error:
            raise LtxUnavailable(openability_error)
        self._validate_gemma_paths(payload)
        if payload.get("gemma_backend") == LTX_GEMMA_BACKEND_GGUF:
            raise LtxUnavailable(_native_gemma_gguf_blocker(payload))
        runtime_blocker = ltx_native_checkpoint_runtime_blocker(checkpoint)
        if runtime_blocker:
            raise LtxUnavailable(runtime_blocker)
        if payload.get("pipeline") == LTX_PIPELINE_DISTILLED:
            upsampler = Path(str(payload.get("spatial_upsampler_path") or ""))
            if not upsampler.is_file():
                raise LtxUnavailable(f"LTX spatial upscaler missing: {upsampler}")
        source = payload.get("source_image_path")
        if source and not Path(str(source)).is_file():
            raise LtxUnavailable(f"LTX source image missing: {source}")

    def _validate_gemma_paths(self, payload: dict) -> None:
        gemma_root = Path(str(payload.get("gemma_root") or ""))
        backend = str(payload.get("gemma_backend") or LTX_GEMMA_BACKEND_HF_SAFETENSORS)
        missing = ltx_gemma_missing_local_files(gemma_root, backend=backend)
        if missing:
            raise LtxUnavailable(
                "LTX Gemma assets are incomplete: " + ", ".join(str(path) for path in missing[:8])
            )
        if backend != LTX_GEMMA_BACKEND_GGUF:
            return
        gguf = Path(str(payload.get("gemma_gguf_path") or ""))
        if not gguf.is_file():
            raise LtxUnavailable(f"LTX Gemma GGUF file missing: {gguf}")
        if gguf.suffix.lower() != ".gguf":
            raise LtxUnavailable(f"LTX Gemma GGUF path must end in .gguf: {gguf}")

    def _write_worker_request(self, job_id: str, payload: dict) -> Path:
        root = self.output_dir() / "requests"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{job_id}.json"
        worker_payload = {
            "_job_id": job_id,
            "_engine": "ltx",
            "_created_at": datetime.now(timezone.utc).isoformat(),
            "mode": "generate",
            **payload,
        }
        path.write_text(json.dumps(worker_payload, indent=2, sort_keys=True), encoding="utf-8")
        return path

    def _run_worker(self, job_id: str, command, events: list[dict], *, on_progress=None) -> None:  # noqa: ANN001
        with self._generation_lock:
            self._active_worker_id = job_id
        try:
            for line in self.process_supervisor.start(job_id, command, check=True):
                if self._generation_cancel.is_set():
                    raise GenerationCancelledError("LTX video generation cancelled.")
                event = _parse_event(line)
                if event is not None:
                    events.append(event)
                    if callable(on_progress):
                        on_progress(event)
            if self._generation_cancel.is_set():
                raise GenerationCancelledError("LTX video generation cancelled.")
        finally:
            with self._generation_lock:
                if self._active_worker_id == job_id:
                    self._active_worker_id = None


def _resolve_path(raw: str | None, default: Path, root: Path) -> Path:
    text = str(raw or "").strip()
    path = Path(text) if text else default
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _resolve_optional_path(raw: str | None, root: Path) -> Path | None:
    text = str(raw or "").strip()
    if not text:
        return None
    path = Path(text)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _converted_heretic_gemma_ready(path: Path) -> bool:
    return all(
        _nonempty_confined_file(path, path / filename)
        for filename in (
            "model.safetensors",
            "vision_projector.safetensors",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "tokenizer.model",
        )
    )


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def ltx_t5_tokenizer_ready(path: Path) -> bool:
    """Validate only the small local T5 tokenizer files used by LTX 2B."""
    root = Path(path).expanduser()
    required = (
        root / "config.json",
        root / "special_tokens_map.json",
        root / "spiece.model",
        root / "tokenizer_config.json",
    )
    if not root.is_dir() or any(not _nonempty_file(item) for item in required):
        return False
    try:
        config = json.loads(required[0].read_text(encoding="utf-8"))
        tokenizer_config = json.loads(required[3].read_text(encoding="utf-8"))
        special_tokens = json.loads(required[1].read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return (
        isinstance(config, dict)
        and config.get("model_type") == "t5"
        and config.get("vocab_size") == 32128
        and isinstance(tokenizer_config, dict)
        and isinstance(special_tokens, dict)
    )


def ltx_gemma_missing_local_files(path: Path, *, backend: str = LTX_GEMMA_BACKEND_HF_SAFETENSORS) -> list[Path]:
    """Check route-specific Gemma tokenizer/processor and optional HF weights.

    HF generation requires a local Gemma model in either the official indexed
    safetensors layout or AIWF's recognized converted layout. The experimental
    GGUF route supplies weights separately, so it only needs tokenizer and
    processor sidecars here.
    """
    root = Path(path).expanduser().resolve()
    missing: list[Path] = []
    if not root.is_dir():
        return [root]

    alternatives = (
        (root / "tokenizer.json", root / "tokenizer.model"),
        (root / "processor_config.json", root / "preprocessor_config.json"),
    )
    for group in alternatives:
        if not any(_nonempty_confined_file(root, candidate) for candidate in group):
            missing.append(group[0])
    tokenizer_config = root / "tokenizer_config.json"
    if not _nonempty_confined_file(root, tokenizer_config):
        missing.append(tokenizer_config)

    if backend == LTX_GEMMA_BACKEND_GGUF:
        return missing

    if _converted_heretic_gemma_ready(root):
        return missing

    config = root / "config.json"
    if not _nonempty_confined_file(root, config):
        missing.append(config)
    index = root / "model.safetensors.index.json"
    monolithic = root / "model.safetensors"
    if _nonempty_confined_file(root, index):
        if not indexed_safetensors_shards_ready(root, index):
            missing.append(index)
    elif not _nonempty_confined_file(root, monolithic):
        missing.append(index)
    return list(dict.fromkeys(missing))


def ltx_gemma_hf_assets_ready(path: Path) -> bool:
    return not ltx_gemma_missing_local_files(path, backend=LTX_GEMMA_BACKEND_HF_SAFETENSORS)


def _nonempty_confined_file(root: Path, path: Path) -> bool:
    try:
        resolved_root = root.resolve()
        resolved = path.resolve(strict=True)
        resolved.relative_to(resolved_root)
        return resolved.is_file() and resolved.stat().st_size > 0
    except (OSError, ValueError, RuntimeError):
        return False


def ltx_checkpoint_openability_error(path: Path) -> str:
    """Return a user-facing error when a native LTX checkpoint cannot be opened.

    This stays intentionally shallow: it only parses the safetensors JSON
    header. It does not mmap tensor payloads or run inference. On Windows,
    very large LTX-2.3 checkpoints can fail when a normal safetensors open
    mmaps the whole file, so keep this as a header-only check.
    """

    if path.suffix.lower() != ".safetensors":
        return ""
    try:
        if path.stat().st_size < 1024 * 1024:
            return ""
    except OSError as exc:
        return f"LTX checkpoint could not be inspected: {path}. {exc}"

    try:
        with path.open("rb") as handle:
            raw_size = handle.read(8)
            if len(raw_size) != 8:
                return f"LTX checkpoint failed a shallow safetensors header check: {path}. Header is incomplete."
            header_size = struct.unpack("<Q", raw_size)[0]
            if header_size <= 0:
                return f"LTX checkpoint failed a shallow safetensors header check: {path}. Header is empty."
            json.loads(handle.read(header_size))
    except OSError as exc:
        return (
            f"LTX checkpoint is not safely loadable in this Windows runtime: {path}. "
            f"{exc}. Increase the Windows paging file or use the working LTX 2B Diffusers route."
        )
    except Exception as exc:
        return f"LTX checkpoint failed a shallow safetensors open check: {path}. {type(exc).__name__}: {exc}"
    return ""


def ltx_native_checkpoint_runtime_blocker(path: Path) -> str:
    """Return a blocker for native LTX-2.3 checkpoints known to crash this Windows worker."""

    if str(os.environ.get("AIWF_ALLOW_UNSTABLE_LTX23") or "").strip().lower() in {"1", "true", "yes", "on"}:
        return ""
    if os.name != "nt" or path.suffix.lower() != ".safetensors":
        return ""
    name = path.name.lower()
    if name == LTX_FULL_CHECKPOINT_FP8.lower():
        return ""
    if name != LTX_FULL_CHECKPOINT.lower():
        return ""
    try:
        if path.stat().st_size < 1024**3:
            return ""
    except OSError:
        return ""
    return (
        "Native LTX 2.3 22B worker is blocked on Windows after the bounded smoke exited with "
        "access violation 3221225477. Use the working LTX 2B Diffusers route, or set "
        "AIWF_ALLOW_UNSTABLE_LTX23=1 only for a manual retest after changing the LTX runtime/pagefile."
    )


def ltx_checkpoint_requires_no_offload(path: Path) -> bool:
    return path.name.lower() == LTX_FULL_CHECKPOINT_FP8.lower()


def _apply_native_runtime_profile(payload: dict) -> None:
    if payload.get("pipeline") != LTX_PIPELINE_ONE_STAGE:
        return
    checkpoint = Path(str(payload.get("checkpoint_path") or ""))
    if not ltx_checkpoint_requires_no_offload(checkpoint):
        return
    payload["offload"] = "none"
    if str(payload.get("quantization") or "").strip().lower() not in {"fp8-cast", "fp8-scaled-mm"}:
        payload["quantization"] = "fp8-cast"


def _gemma_root_text(request: LtxVideoRequest) -> str:
    text = str(request.gemma_root or "").strip()
    if text.lower().endswith(".gguf"):
        return ""
    return text


def _gemma_gguf_text(request: LtxVideoRequest) -> str:
    text = str(request.gemma_gguf_path or "").strip()
    if text:
        return text
    root_text = str(request.gemma_root or "").strip()
    if root_text.lower().endswith(".gguf"):
        return root_text
    return ""


def _parse_event(line: str) -> dict | None:
    text = line.strip()
    if not text.startswith("{"):
        return None
    try:
        event = json.loads(text)
    except json.JSONDecodeError:
        return None
    return event if isinstance(event, dict) and "kind" in event else None


def _last_error(events: list[dict]) -> str:
    status_tail = _interesting_status_tail(events)
    for event in reversed(events):
        if event.get("kind") == "error":
            detail = str(event.get("detail") or event.get("message") or "LTX worker failed")
            if status_tail and status_tail not in detail:
                detail = f"{detail}\n\nRecent LTX output:\n{status_tail}"
            return _clip(detail)
    return ""


def _interesting_status_tail(events: list[dict]) -> str:
    needles = (
        "traceback",
        "runtimeerror",
        "cuda",
        "out of memory",
        "invalid python storage",
        "error",
        "failed",
        "unsupported",
        "probing native gemma gguf",
        "gguf metadata",
        "ltx gemma contract",
        "ltx embeddings processor contract",
        "installed gguf capability",
        "generation is blocked",
        "aiwf ltx loader",
    )
    messages = [
        str(event.get("message") or "")
        for event in events
        if event.get("kind") == "status" and any(token in str(event.get("message") or "").lower() for token in needles)
    ]
    return "\n".join(messages[-8:])


def _clip(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:].lstrip()


def _native_gemma_gguf_blocker(payload: dict) -> str:
    return (
        "Native Gemma GGUF was selected for LTX, but this worker cannot generate with it yet. "
        "LTX needs Gemma hidden states from the embedding output plus every layer, plus an attention mask; "
        "the current upstream LTX CLI only accepts a repo-shaped safetensors Gemma text encoder. "
        f"Selected GGUF: {payload.get('gemma_gguf_path')}. "
        "Run `venv\\Scripts\\python.exe scripts\\probe_ltx_runtime.py --gguf --json --allow-blocked` to verify the "
        "native-GGUF blocker without dequantizing weights."
    )
