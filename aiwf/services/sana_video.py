from __future__ import annotations

import gc
import json
import logging
import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from PIL import Image

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.audio import AudioGenerationOptions
from aiwf.core.domain.engine import EngineTenant
from aiwf.core.domain.sana_video import (
    SANA_VIDEO_MODEL_REPO_480P,
    SANA_VIDEO_MODEL_REPO_720P,
    SANA_VIDEO_MODEL_VARIANT_480P,
    SANA_VIDEO_PIPELINE_I2V,
    SANA_VIDEO_QUANTIZATION_AUTO,
    SANA_VIDEO_QUANTIZATION_BF16,
    SANA_VIDEO_QUANTIZATION_FP8,
    SANA_VIDEO_QUANTIZATION_BNB_FP4,
    SANA_VIDEO_QUANTIZATION_BNB_INT8,
    SANA_VIDEO_QUANTIZATION_BNB_NF4,
    SANA_VIDEO_VAE_TILING_ALWAYS,
    SANA_VIDEO_VAE_TILING_AUTO,
    SanaVideoProgressEvent,
    SanaVideoRequest,
    SanaVideoResult,
    resolve_sana_video_path,
    sana_video_repo_for_variant,
    sana_video_model_folder_name,
)
from aiwf.infrastructure.video.processing import VideoProcessor
from aiwf.services.audio import AudioGenerationService, AudioUnavailable
from aiwf.infrastructure.diffusers.checkpoints import (
    sana_video_dir_has_required_local_files,
    sana_video_missing_local_files,
)
from aiwf.services.model_files import resolve_model_asset
from aiwf.services.route_lifecycle import support_revision

logger = logging.getLogger(__name__)

SanaProgressCallback = Callable[..., None]


class SanaVideoUnavailable(RuntimeError):
    pass


class _SanaStageTracker:
    def __init__(self, on_progress: SanaProgressCallback | None = None) -> None:
        self.on_progress = on_progress
        self.events: list[SanaVideoProgressEvent] = []
        self.started = time.perf_counter()

    def emit(
        self,
        stage: str,
        progress: float,
        message: str,
        *,
        step: int = 0,
        total: int = 0,
    ) -> None:
        progress = max(0.0, min(1.0, float(progress)))
        event = SanaVideoProgressEvent(
            stage=stage,
            progress=progress,
            message=message,
            step=max(0, int(step)),
            total=max(0, int(total)),
            seconds=round(time.perf_counter() - self.started, 3),
        )
        self.events.append(event)
        pct = int(round(event.progress * 100))
        step_text = f" {event.step}/{event.total}" if event.total else ""
        print(f"[AIWF] Sana Video: {event.stage}{step_text} {pct}% - {event.message}", flush=True)
        if self.on_progress is None:
            return
        for args in (
            (event.stage, event.progress, event.message, event.step, event.total, event.seconds),
            (event.stage, event.progress, event.message),
            (event.progress, event.message),
            (event.step, event.total, event.message),
        ):
            try:
                self.on_progress(*args)
                return
            except TypeError:
                continue


class SanaVideoService:
    def __init__(
        self,
        flags: RuntimeFlags | None = None,
        settings: UserSettings | None = None,
        devices=None,  # noqa: ANN001
        supervisor=None,  # noqa: ANN001
        unload_image_models: Callable[[], None] | None = None,
    ) -> None:
        self.flags = flags or RuntimeFlags()
        self.settings = settings or UserSettings()
        self.devices = devices
        self.supervisor = supervisor
        self._unload_image_models = unload_image_models
        self._operation_lock = threading.Lock()
        self._pipeline_cache_lock = threading.RLock()
        self._prepared_pipeline_key: tuple[Any, ...] | None = None
        self._prepared_pipeline: Any | None = None
        self._prepared_pipeline_info: dict[str, str] = {}

    def models_root(self) -> Path:
        return self.flags.resolved_models_dir() / "sana-video" / "Diffusers"

    def default_model_path(self, variant: str = SANA_VIDEO_MODEL_VARIANT_480P) -> Path:
        repo_id = sana_video_repo_for_variant(variant)
        folder = sana_video_model_folder_name(repo_id)
        relative_candidates = (
            Path("sana-video") / "Diffusers" / folder,
            Path("Diffusers") / folder,
            Path(folder),
        )
        return resolve_model_asset(
            self.flags,
            relative_candidates,
            predicate=sana_video_dir_has_required_local_files,
            fallback=self.models_root() / folder,
        )

    def output_dir(self) -> Path:
        root = self.flags.resolved_output_dir() / "sana-videos"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def log_dir(self) -> Path:
        root = self.flags.data_dir / "_local" / "logs"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def status_markdown(self) -> str:
        model = self.default_model_path()
        model_index = model / "model_index.json"
        runtime_ok = self.runtime_available()
        lines = [
            "**Sana video:** "
            + ("runtime ready" if runtime_ok else "Diffusers Sana video classes missing"),
            f"- Default model: `{model}`",
            f"- Attention: `{self.sage_status()}`",
            f"- Quantization: `{self.bitsandbytes_status()}`",
            "- Audio: no native Sana audio path detected; optional MMAudio post-process can attach audio after video.",
        ]
        missing = sana_video_missing_local_files(model)
        if missing:
            if not model_index.is_file():
                lines.append("- Model snapshot missing. Download the 480p SANA-Video Diffusers folder first.")
            else:
                missing_names = ", ".join(str(path.relative_to(model)) for path in missing[:8])
                lines.append(f"- Model snapshot incomplete; missing or invalid: {missing_names}")
        return "\n".join(lines)

    @staticmethod
    def runtime_available(image_to_video: bool = False) -> bool:
        try:
            import diffusers

            pipeline_class = "SanaImageToVideoPipeline" if image_to_video else "SanaVideoPipeline"
            return hasattr(diffusers, pipeline_class)
        except Exception:
            return False

    def _pipeline_key(self, model_path: Path, request: SanaVideoRequest) -> tuple[Any, ...]:
        return (
            str(model_path.resolve()),
            support_revision([str(model_path)]),
            bool(request.wants_image_to_video),
            self._effective_quantization(request),
            bool(request.use_sage_attention),
            str(request.vae_tiling),
        )

    def _get_or_load_pipeline(
        self,
        model_path: Path,
        request: SanaVideoRequest,
    ) -> tuple[Any, dict[str, str]]:
        key = self._pipeline_key(model_path, request)
        with self._pipeline_cache_lock:
            if self._prepared_pipeline is not None and self._prepared_pipeline_key == key:
                return self._prepared_pipeline, dict(self._prepared_pipeline_info)
            self._unload_pipeline_locked()
            headroom_issue = self._prepare_headroom_issue()
            if headroom_issue:
                raise SanaVideoUnavailable(headroom_issue)
            pipe, info = self._load_pipeline(
                model_path,
                request,
                image_to_video=request.wants_image_to_video,
            )
            self._prepared_pipeline_key = key
            self._prepared_pipeline = pipe
            self._prepared_pipeline_info = dict(info)
            return pipe, dict(info)

    def prepare(self, request: SanaVideoRequest) -> dict[str, Any]:
        """Load and retain the selected pipeline without overlapping generate/unload."""
        if not self._operation_lock.acquire(blocking=False):
            raise SanaVideoUnavailable("Sana Video is busy with another prepare, generation, or model switch operation.")
        try:
            return self._prepare_with_operation_lock(request)
        finally:
            self._operation_lock.release()

    def _prepare_with_operation_lock(self, request: SanaVideoRequest) -> dict[str, Any]:
        """Load and retain the exact selected Sana pipeline without running inference."""
        if not self.runtime_available(request.wants_image_to_video):
            pipeline_class = "SanaImageToVideoPipeline" if request.wants_image_to_video else "SanaVideoPipeline"
            raise SanaVideoUnavailable(f"Installed Diffusers does not expose {pipeline_class}.")
        model_path = resolve_sana_video_path(
            request.model_path,
            self.default_model_path(request.model_variant),
            self.flags.data_dir,
        )
        if not (model_path / "model_index.json").is_file():
            raise SanaVideoUnavailable(f"Sana video model folder missing model_index.json: {model_path}")
        missing_files = sana_video_missing_local_files(model_path)
        if missing_files:
            missing_text = ", ".join(str(path) for path in missing_files[:8]) or "required model files"
            raise SanaVideoUnavailable(
                f"Sana video snapshot is incomplete; missing or invalid local files: {missing_text}"
            )

        def load_selected_pipeline() -> tuple[Any, dict[str, str]]:
            if self._unload_image_models is not None:
                try:
                    self._unload_image_models()
                except Exception as exc:
                    raise SanaVideoUnavailable(
                        f"Could not unload the active image model before Sana Video loading: {exc}"
                    ) from exc
            return self._get_or_load_pipeline(model_path, request)

        if self.supervisor is not None:
            try:
                with self.supervisor.tenant_session(
                    EngineTenant.VIDEO,
                    reason="Sana Video model selection",
                    allow_wait=False,
                ):
                    pipe, info = load_selected_pipeline()
            except RuntimeError as exc:
                raise SanaVideoUnavailable(f"GPU busy: {exc}") from exc
        else:
            pipe, info = load_selected_pipeline()
        return {
            "loaded": self._prepared_pipeline is pipe,
            "modelPath": str(model_path),
            "quantization": str(info.get("quantization", "")),
            "attentionBackend": str(info.get("attention_backend", "")),
        }

    @staticmethod
    def _prepare_headroom_issue() -> str | None:
        """Defer a new Sana load when the GPU cannot safely hold its weights."""
        try:
            import torch

            if not torch.cuda.is_available():
                return None
            from aiwf.services.gpu_memory import measured_cuda_free_bytes

            free_bytes = measured_cuda_free_bytes(torch)
            if free_bytes is None:
                return "Sana Video preparation deferred because available GPU memory could not be matched to the active CUDA device."
        except Exception:
            return "Sana Video preparation deferred because available GPU memory could not be checked."
        free_gb = float(free_bytes) / 1024**3
        required_gb = 6.0
        if free_gb < required_gb:
            return (
                f"Sana Video preparation deferred: {free_gb:.1f} GB VRAM is free; "
                f"at least {required_gb:.0f} GB is required to load the selected pipeline."
            )
        return None

    def _unload_pipeline_locked(self) -> bool:
        had_pipeline = self._prepared_pipeline is not None
        self._prepared_pipeline = None
        self._prepared_pipeline_key = None
        self._prepared_pipeline_info = {}
        if had_pipeline:
            gc.collect()
            self._empty_cuda_cache()
        return had_pipeline

    def unload(self) -> bool:
        """Release the selected cached pipeline before another family claims VRAM."""
        with self._pipeline_cache_lock:
            return self._unload_pipeline_locked()

    def release_cached_model_for_modality_switch(self) -> bool:
        """Evict a resident Sana pipeline only while idle or owning VIDEO safely."""
        if not self._operation_lock.acquire(blocking=False):
            return False
        try:
            with self._pipeline_cache_lock:
                if self._prepared_pipeline is None:
                    return True
            supervisor = self.supervisor
            tenant_session = getattr(supervisor, "tenant_session", None)
            if supervisor is None or not callable(tenant_session):
                return False
            active_tenant = getattr(supervisor, "active_tenant", None)
            active_value = str(getattr(active_tenant, "value", active_tenant) or "").strip().lower()
            if active_value not in {"", "idle", "none", EngineTenant.VIDEO.value}:
                return False
            try:
                with tenant_session(
                    EngineTenant.VIDEO,
                    reason="Release cached Sana Video model before modality switch",
                    allow_wait=False,
                ):
                    with self._pipeline_cache_lock:
                        self._unload_pipeline_locked()
                        return self._prepared_pipeline is None and self._prepared_pipeline_key is None
            except Exception:
                logger.info("Could not release cached Sana Video model for modality switch", exc_info=True)
                return False
        finally:
            self._operation_lock.release()

    @staticmethod
    def sage_status() -> str:
        try:
            from diffusers.models import attention_dispatch as dispatch

            if bool(getattr(dispatch, "_CAN_USE_SAGE_ATTN", False)) and getattr(dispatch, "sageattn", None) is not None:
                return "diffusers_sage"
        except Exception:
            pass
        try:
            import sageattention  # noqa: F401

            return "sageattention_importable"
        except Exception:
            return "unavailable"

    @staticmethod
    def bitsandbytes_status() -> str:
        try:
            import bitsandbytes as bnb

            return str(getattr(bnb, "__version__", "available"))
        except Exception:
            return "unavailable"

    def generate(
        self,
        request: SanaVideoRequest,
        *,
        on_progress: SanaProgressCallback | None = None,
    ) -> SanaVideoResult:
        if not self._operation_lock.acquire(blocking=False):
            raise SanaVideoUnavailable("Sana Video is busy with another prepare, generation, or model switch operation.")
        try:
            return self._generate_with_operation_lock(request, on_progress=on_progress)
        finally:
            self._operation_lock.release()

    def _generate_with_operation_lock(
        self,
        request: SanaVideoRequest,
        *,
        on_progress: SanaProgressCallback | None = None,
    ) -> SanaVideoResult:
        if request.generate_audio:
            audio_model_id = str(request.audio_model_id or "")
            if not audio_model_id.startswith("mmaudio:"):
                raise SanaVideoUnavailable("Sana video audio requires an installed MMAudio variant.")
            audio_variant = audio_model_id.split(":", 1)[1]
            audio_service = AudioGenerationService(self.flags, self.settings, self.devices, self.supervisor)
            if not audio_service._mmaudio_variant_ready(audio_variant):
                raise SanaVideoUnavailable(
                    f"Sana video audio requires the complete MMAudio {audio_variant} model bundle."
                )
            runtime_error = audio_service._mmaudio_runtime_import_error()
            if runtime_error:
                raise SanaVideoUnavailable(f"Sana video audio runtime is not ready: {runtime_error}")
        if self.supervisor is None:
            return self._generate_for_request(request, on_progress=on_progress)

        from aiwf.services.engine_supervisor import EngineSwitchRequest, EngineTenant

        tenant_job_id = f"sana_{uuid4().hex[:8]}"
        switch = self.supervisor.request_switch(
            EngineSwitchRequest(
                target=EngineTenant.VIDEO,
                reason="Sana Video generation",
                job_id=tenant_job_id,
            )
        )
        if not switch.ok:
            raise SanaVideoUnavailable(f"GPU busy: {switch.message}")
        result: SanaVideoResult | None = None
        try:
            with self.supervisor.borrow_active_tenant(EngineTenant.VIDEO, job_id=tenant_job_id):
                if self._unload_image_models is not None:
                    try:
                        self._unload_image_models()
                    except Exception as exc:
                        raise SanaVideoUnavailable(
                            f"Could not unload the active image model before Sana Video loading: {exc}"
                        ) from exc
                video_request = (
                    request.model_copy(update={"generate_audio": False})
                    if request.generate_audio
                    else request
                )
                result = self._generate_for_request(video_request, on_progress=on_progress)
                if request.generate_audio:
                    # Audio post-processing runs after VIDEO ownership is
                    # released. Evict Sana while this call still owns VIDEO.
                    self.unload()
        finally:
            released = self.supervisor.request_switch(
                EngineSwitchRequest(
                    target=EngineTenant.IDLE,
                    reason="Sana Video generation complete",
                    job_id=tenant_job_id,
                )
            )
            if not released.ok:
                logger.error("Could not release Sana Video GPU ownership: %s", released.message)
                raise SanaVideoUnavailable(
                    f"Sana video finished, but GPU ownership could not be released safely: {released.message}"
                )

        if result is None:
            raise SanaVideoUnavailable("Sana video did not return a result.")
        if request.generate_audio:
            return self._attach_audio_postprocess(result, request, on_progress=on_progress)
        return result

    def _attach_audio_postprocess(
        self,
        result: SanaVideoResult,
        request: SanaVideoRequest,
        *,
        on_progress: SanaProgressCallback | None = None,
    ) -> SanaVideoResult:
        """Run optional audio only after the VIDEO tenant has been released."""
        tracker = _SanaStageTracker(on_progress)
        tracker.emit("audio", 0.97, "Running video-conditioned audio post-process")
        started = time.perf_counter()
        audio_prompt = (request.audio_prompt or request.prompt or "").strip()
        audio_service = AudioGenerationService(self.flags, self.settings, self.devices, self.supervisor)
        try:
            audio, muxed = audio_service.generate_and_mux(
                result.output_path,
                AudioGenerationOptions(
                    prompt=audio_prompt,
                    kind="video_audio",
                    model_id=request.audio_model_id,
                    duration_seconds=max(1.0, float(request.frames) / max(float(request.fps), 1.0)),
                    cfg_coef=float(request.audio_cfg),
                    steps=int(request.audio_steps),
                    seed=int(request.seed),
                ),
                duration_seconds=max(1.0, float(request.frames) / max(float(request.fps), 1.0)),
            )
        except Exception as exc:
            receipt_path = self._write_receipt_payload(
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "status": "error",
                    "request": request.model_dump(),
                    "output_path": result.output_path,
                    "result": result.model_dump(),
                    "partial_artifact": {
                        "path": result.output_path,
                        "kind": "video_without_requested_audio",
                    },
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                },
                run_id=uuid4().hex[:6],
                stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            )
            raise SanaVideoUnavailable(
                f"Sana video rendered, but audio generation failed: {exc}. Silent video preserved at {result.output_path}. Receipt: {receipt_path}"
            ) from exc
        audio_seconds = round(time.perf_counter() - started, 3)
        tracker.emit("done", 1.0, "Sana video and audio saved")
        timings = dict(result.timings)
        timings["audio"] = audio_seconds
        timings["total"] = round(sum(value for key, value in timings.items() if key != "total"), 3)
        final_result = result.model_copy(
            update={
                "output_path": str(muxed.output_path),
                "message": f"Sana video saved to {Path(muxed.output_path).name} with audio",
                "has_audio": True,
                "audio_path": audio.output_path,
                "video_only_path": result.output_path,
                "timings": timings,
                "progress": [*result.progress, *(event.model_dump() for event in tracker.events)],
            }
        )
        receipt_path = self._write_receipt(
            final_result,
            request,
            run_id=uuid4().hex[:6],
            stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        )
        return final_result.model_copy(update={"receipt_path": str(receipt_path)})

    def _generate_for_request(
        self,
        request: SanaVideoRequest,
        *,
        on_progress: SanaProgressCallback | None = None,
    ) -> SanaVideoResult:
        tracker = _SanaStageTracker(on_progress)
        timings: dict[str, float] = {}
        prompt = (request.prompt or "").strip()
        if not prompt:
            raise SanaVideoUnavailable("Enter a Sana video prompt first.")
        if not self.runtime_available(request.wants_image_to_video):
            pipeline_class = "SanaImageToVideoPipeline" if request.wants_image_to_video else "SanaVideoPipeline"
            raise SanaVideoUnavailable(f"Installed Diffusers does not expose {pipeline_class}.")

        model_path = resolve_sana_video_path(
            request.model_path,
            self.default_model_path(request.model_variant),
            self.flags.data_dir,
        )
        if not (model_path / "model_index.json").is_file():
            raise SanaVideoUnavailable(f"Sana video model folder missing model_index.json: {model_path}")
        from aiwf.infrastructure.diffusers.checkpoints import sana_video_missing_local_files

        missing_files = sana_video_missing_local_files(model_path)
        if missing_files:
            missing_text = ", ".join(str(path) for path in missing_files) or "required model files"
            raise SanaVideoUnavailable(
                f"Sana video snapshot is incomplete; missing or empty local files: {missing_text}"
            )

        source_image = resolve_sana_video_path(request.source_image_path, Path(), self.flags.data_dir) if request.source_image_path else None
        if request.wants_image_to_video and (source_image is None or not source_image.is_file()):
            raise SanaVideoUnavailable(f"Sana image-to-video source image missing: {source_image}")

        run_id = uuid4().hex[:6]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = self.output_dir() / f"sana-video-{stamp}-{run_id}.mp4"
        video_only_path = output_path

        pipe = None
        attention_backend = ""
        quantization = ""
        vae_tiling = request.vae_tiling
        try:
            tracker.emit("load", 0.01, "Loading Sana Video pipeline")
            started = time.perf_counter()
            pipe, load_info = self._get_or_load_pipeline(model_path, request)
            timings["load"] = round(time.perf_counter() - started, 3)
            attention_backend = str(load_info.get("attention_backend", ""))
            quantization = str(load_info.get("quantization", ""))
            tracker.emit(
                "load",
                0.10,
                f"Loaded pipeline ({quantization}, attention={attention_backend})",
            )

            tracker.emit("encode", 0.12, "Encoding prompt")
            started = time.perf_counter()
            prompt_inputs = self._encode_prompt(pipe, request)
            timings["encode"] = round(time.perf_counter() - started, 3)
            tracker.emit("encode", 0.20, f"Prompt encoded in {timings['encode']:.2f}s")
            if request.offload_text_encoder_after_encode:
                self._offload_text_encoder(pipe, tracker)

            tracker.emit("inference", 0.22, f"Running denoise loop for {int(request.steps)} steps")
            started = time.perf_counter()
            try:
                latents = self._run_pipeline_to_latents(
                    pipe,
                    request,
                    source_image=source_image,
                    prompt_inputs=prompt_inputs,
                    tracker=tracker,
                )
            except Exception as exc:
                if attention_backend == "diffusers.sage" and self._is_sage_attention_mask_error(exc):
                    tracker.emit(
                        "inference",
                        0.23,
                        "Sage attention cannot handle Sana attention masks; retrying with native attention",
                    )
                    timings["inference_sage_failed_after"] = round(time.perf_counter() - started, 3)
                    attention_backend = self._disable_sage_attention(pipe)
                    started = time.perf_counter()
                    latents = self._run_pipeline_to_latents(
                        pipe,
                        request,
                        source_image=source_image,
                        prompt_inputs=prompt_inputs,
                        tracker=tracker,
                    )
                else:
                    raise
            timings["inference"] = round(time.perf_counter() - started, 3)
            tracker.emit("inference", 0.82, f"Denoise complete in {timings['inference']:.2f}s")
            self._release_denoise_components(pipe, tracker)

            tracker.emit("decode", 0.84, "Decoding latents")
            started = time.perf_counter()
            frames, vae_tiling = self._decode_latents(pipe, latents, request, tracker)
            timings["decode"] = round(time.perf_counter() - started, 3)

            tracker.emit("export", 0.94, "Writing MP4")
            started = time.perf_counter()
            self._export_frames(frames, output_path, fps=float(request.fps))
            timings["export"] = round(time.perf_counter() - started, 3)
            tracker.emit("export", 0.97, f"Wrote {output_path.name}")
        except Exception as exc:
            timings["total"] = round(sum(value for key, value in timings.items() if key != "total"), 3)
            tracker.emit("error", 1.0, f"Sana video failed during {tracker.events[-1].stage if tracker.events else 'unknown'}: {exc}")
            receipt_path = self._write_failure_receipt(
                request,
                run_id=run_id,
                stamp=stamp,
                output_path=output_path,
                timings=timings,
                progress=tracker.events,
                attention_backend=attention_backend,
                quantization=quantization,
                vae_tiling=vae_tiling,
                error=exc,
            )
            logger.error("Sana Video failed; receipt written to %s", receipt_path, exc_info=True)
            raise
        finally:
            if pipe is not None:
                del pipe
            gc.collect()
            self._empty_cuda_cache()

        audio_path = ""
        has_audio = VideoProcessor().probe(output_path).has_audio
        if request.generate_audio:
            self.unload()
            tracker.emit("audio", 0.97, "Running video-conditioned audio post-process")
            started = time.perf_counter()
            audio_prompt = (request.audio_prompt or request.prompt or "").strip()
            audio_service = AudioGenerationService(self.flags, self.settings, self.devices, self.supervisor)
            try:
                audio, muxed = audio_service.generate_and_mux(
                    output_path,
                    AudioGenerationOptions(
                        prompt=audio_prompt,
                        kind="video_audio",
                        model_id=request.audio_model_id,
                        duration_seconds=max(1.0, float(request.frames) / max(float(request.fps), 1.0)),
                        cfg_coef=float(request.audio_cfg),
                        steps=int(request.audio_steps),
                        seed=int(request.seed),
                    ),
                    duration_seconds=max(1.0, float(request.frames) / max(float(request.fps), 1.0)),
                )
            except AudioUnavailable as exc:
                raise SanaVideoUnavailable(f"Sana video rendered, but audio generation failed: {exc}") from exc
            timings["audio"] = round(time.perf_counter() - started, 3)
            audio_path = audio.output_path
            video_only_path = output_path
            output_path = Path(muxed.output_path)
            has_audio = True

        timings["total"] = round(sum(value for key, value in timings.items() if key != "total"), 3)
        tracker.emit("done", 1.0, f"Sana video saved to {output_path.name}")
        result = SanaVideoResult(
            output_path=str(output_path),
            message=f"Sana video saved to {output_path.name}" + (" with audio" if has_audio else ""),
            frames=int(request.frames),
            fps=float(request.fps),
            width=int(request.width),
            height=int(request.height),
            has_audio=has_audio,
            audio_path=audio_path,
            video_only_path=str(video_only_path) if Path(video_only_path) != Path(output_path) else "",
            infotext=(
                f"Sana video {request.pipeline}: {request.width}x{request.height}, "
                f"{request.frames} frames, {request.steps} steps, CFG {request.cfg_scale:.2f}"
            ),
            timings=timings,
            progress=[event.model_dump() for event in tracker.events],
            attention_backend=attention_backend,
            quantization=quantization,
            vae_tiling=vae_tiling,
        )
        receipt_path = self._write_receipt(result, request, run_id=run_id, stamp=stamp)
        return result.model_copy(update={"receipt_path": str(receipt_path)})

    def _load_pipeline(
        self,
        model_path: Path,
        request: SanaVideoRequest,
        *,
        image_to_video: bool,
    ) -> tuple[Any, dict[str, str]]:
        import diffusers

        pipeline_name = "SanaImageToVideoPipeline" if image_to_video else "SanaVideoPipeline"
        pipeline_cls = getattr(diffusers, pipeline_name, None)
        if pipeline_cls is None:
            raise SanaVideoUnavailable(f"Installed Diffusers does not expose {pipeline_name}.")
        requested_quant = self._effective_quantization(request)
        attempts = [requested_quant]
        if requested_quant != SANA_VIDEO_QUANTIZATION_BF16:
            attempts.append(SANA_VIDEO_QUANTIZATION_BF16)

        last_error: Exception | None = None
        for quantization in attempts:
            kwargs = self._load_kwargs(quantization)
            try:
                pipe = pipeline_cls.from_pretrained(str(model_path), **kwargs)
            except TypeError:
                kwargs.pop("torch_dtype", None)
                kwargs["dtype"] = self._dtype()
                try:
                    pipe = pipeline_cls.from_pretrained(str(model_path), **kwargs)
                except Exception as exc:
                    last_error = exc
                    if quantization != SANA_VIDEO_QUANTIZATION_BF16:
                        logger.warning("Sana quantized load failed for %s: %s", quantization, exc)
                        continue
                    raise
            except Exception as exc:
                last_error = exc
                if quantization != SANA_VIDEO_QUANTIZATION_BF16:
                    logger.warning("Sana quantized load failed for %s: %s", quantization, exc)
                    continue
                raise
            try:
                self._prepare_pipeline_after_load(pipe, quantization)
                attention_backend = self._apply_sage_attention(pipe, request)
                if request.vae_tiling == SANA_VIDEO_VAE_TILING_ALWAYS:
                    self._enable_vae_tiling(pipe)
                return pipe, {"quantization": quantization, "attention_backend": attention_backend}
            except Exception as exc:
                last_error = exc
                if quantization != SANA_VIDEO_QUANTIZATION_BF16:
                    logger.warning("Sana post-load setup failed for %s: %s", quantization, exc)
                    del pipe
                    gc.collect()
                    self._empty_cuda_cache()
                    continue
                raise
        raise SanaVideoUnavailable(f"Sana Video failed to load: {last_error}")

    def _load_kwargs(self, quantization: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"torch_dtype": self._dtype(), "local_files_only": True}
        qconfig = self._quantization_config(quantization)
        if qconfig is not None:
            kwargs["quantization_config"] = qconfig
            kwargs["device_map"] = "balanced"
        return kwargs

    def _quantization_config(self, quantization: str):  # noqa: ANN201
        if quantization in {SANA_VIDEO_QUANTIZATION_BF16, SANA_VIDEO_QUANTIZATION_FP8}:
            return None
        try:
            import torch
            from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
            from diffusers import PipelineQuantizationConfig
            from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
        except Exception as exc:
            logger.warning("Sana bitsandbytes quantization unavailable: %s", exc)
            return None

        if quantization == SANA_VIDEO_QUANTIZATION_BNB_INT8:
            return PipelineQuantizationConfig(
                quant_mapping={
                    "transformer": DiffusersBitsAndBytesConfig(load_in_8bit=True),
                    "text_encoder": TransformersBitsAndBytesConfig(load_in_8bit=True),
                }
            )
        if quantization in {SANA_VIDEO_QUANTIZATION_BNB_NF4, SANA_VIDEO_QUANTIZATION_BNB_FP4}:
            quant_type = "nf4" if quantization == SANA_VIDEO_QUANTIZATION_BNB_NF4 else "fp4"
            compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            return PipelineQuantizationConfig(
                quant_mapping={
                    "transformer": DiffusersBitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type=quant_type,
                        bnb_4bit_compute_dtype=compute_dtype,
                        bnb_4bit_use_double_quant=True,
                    ),
                    "text_encoder": TransformersBitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type=quant_type,
                        bnb_4bit_compute_dtype=compute_dtype,
                        bnb_4bit_use_double_quant=True,
                    ),
                }
            )
        return None

    def _effective_quantization(self, request: SanaVideoRequest) -> str:
        if request.quantization != SANA_VIDEO_QUANTIZATION_AUTO:
            return request.quantization
        try:
            import torch

            if torch.cuda.is_available():
                total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if total_gb <= 18 and self.bitsandbytes_status() != "unavailable":
                    return SANA_VIDEO_QUANTIZATION_BNB_INT8
        except Exception:
            pass
        return SANA_VIDEO_QUANTIZATION_BF16

    def _prepare_pipeline_after_load(self, pipe, quantization: str) -> None:  # noqa: ANN001
        if quantization == SANA_VIDEO_QUANTIZATION_FP8:
            self._apply_fp8_layerwise_casting(pipe)
        if quantization not in {SANA_VIDEO_QUANTIZATION_BF16, SANA_VIDEO_QUANTIZATION_FP8}:
            return
        device = self._device()
        pipe.to(device)

    @staticmethod
    def _apply_fp8_layerwise_casting(pipe) -> None:  # noqa: ANN001
        import torch
        from diffusers.hooks.layerwise_casting import apply_layerwise_casting

        transformer = getattr(pipe, "transformer", None)
        if transformer is None:
            raise SanaVideoUnavailable("Sana FP8 mode requires a transformer component.")
        apply_layerwise_casting(
            transformer,
            storage_dtype=torch.float8_e4m3fn,
            compute_dtype=torch.bfloat16,
            skip_modules_pattern=("patch_embed", "norm", "proj_out", "pos_embed"),
            non_blocking=True,
        )

    def _apply_sage_attention(self, pipe, request: SanaVideoRequest) -> str:  # noqa: ANN001
        if not request.use_sage_attention:
            return "disabled"
        transformer = getattr(pipe, "transformer", None)
        if transformer is None or not hasattr(transformer, "set_attention_backend"):
            return "unavailable"
        try:
            from diffusers.models import attention_dispatch as dispatch
            from diffusers.models.attention_dispatch import AttentionBackendName

            if bool(getattr(dispatch, "_CAN_USE_SAGE_ATTN", False)) and getattr(dispatch, "sageattn", None) is not None:
                transformer.set_attention_backend(AttentionBackendName.SAGE)
                logger.info("Sana Video attention backend: diffusers.sage")
                return "diffusers.sage"
        except Exception as exc:
            logger.warning("Sana Video Sage attention setup failed: %s", exc)
        return "native"

    def _encode_prompt(self, pipe, request: SanaVideoRequest) -> dict[str, Any]:  # noqa: ANN001
        device = self._prompt_encode_device(pipe)
        guidance = float(request.cfg_scale) > 1.0
        kwargs = dict(
            negative_prompt=request.negative_prompt or "",
            num_videos_per_prompt=1,
            device=device,
            clean_caption=False,
            max_sequence_length=int(request.max_sequence_length),
            complex_human_instruction=self._complex_human_instruction(pipe),
        )
        prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = pipe.encode_prompt(
            request.prompt,
            guidance,
            **kwargs,
        )
        denoise_device = self._execution_device(pipe)
        return {
            "prompt_embeds": self._tensor_to_device(prompt_embeds, denoise_device),
            "prompt_attention_mask": self._tensor_to_device(prompt_attention_mask, denoise_device),
            "negative_prompt_embeds": self._tensor_to_device(negative_prompt_embeds, denoise_device),
            "negative_prompt_attention_mask": self._tensor_to_device(negative_prompt_attention_mask, denoise_device),
        }

    def _prompt_encode_device(self, pipe) -> Any:  # noqa: ANN001
        device = self._execution_device(pipe)
        if device.type != "cuda":
            return device
        text_encoder = getattr(pipe, "text_encoder", None)
        if text_encoder is not None and hasattr(text_encoder, "to"):
            try:
                text_encoder.to(device)
                logger.info("Sana Video text encoder moved to %s for prompt encode", device)
            except Exception:
                logger.warning("Sana Video text encoder could not move to GPU for prompt encode.", exc_info=True)
        return device

    @staticmethod
    def _tensor_to_device(value, device):  # noqa: ANN001, ANN202
        if value is not None and hasattr(value, "to"):
            return value.to(device=device)
        return value

    def _offload_text_encoder(self, pipe, tracker: _SanaStageTracker) -> None:  # noqa: ANN001
        if self._release_component(pipe, "text_encoder"):
            tracker.emit("encode", 0.21, "Text encoder released after prompt encode")

    def _release_denoise_components(self, pipe, tracker: _SanaStageTracker) -> None:  # noqa: ANN001
        released = []
        for component_name in ("transformer", "text_encoder"):
            if self._release_component(pipe, component_name):
                released.append(component_name)
        if released:
            tracker.emit("decode", 0.83, "Released denoise components before VAE decode: " + ", ".join(released))

    def _release_component(self, pipe, component_name: str) -> bool:  # noqa: ANN001
        component = getattr(pipe, component_name, None)
        if component is None:
            return False
        try:
            if not hasattr(component, "_hf_hook") and hasattr(component, "to"):
                component.to("cpu")
        except Exception as exc:
            logger.debug("Could not move Sana %s to CPU before release: %s", component_name, exc, exc_info=True)
        try:
            setattr(pipe, component_name, None)
            device_map = getattr(pipe, "hf_device_map", None)
            if isinstance(device_map, dict):
                device_map.pop(component_name, None)
        except Exception as exc:
            logger.debug("Could not detach Sana %s after use: %s", component_name, exc, exc_info=True)
        gc.collect()
        self._empty_cuda_cache()
        return True

    @staticmethod
    def _disable_sage_attention(pipe) -> str:  # noqa: ANN001
        transformer = getattr(pipe, "transformer", None)
        if transformer is None or not hasattr(transformer, "set_attention_backend"):
            return "native_after_sage_mask_retry"
        try:
            from diffusers.models.attention_dispatch import AttentionBackendName

            transformer.set_attention_backend(AttentionBackendName.NATIVE)
        except Exception as exc:
            logger.warning("Could not reset Sana Video attention backend after Sage mask failure: %s", exc)
        return "native_after_sage_mask_retry"

    def _run_pipeline_to_latents(
        self,
        pipe,  # noqa: ANN001
        request: SanaVideoRequest,
        *,
        source_image: Path | None,
        prompt_inputs: dict[str, Any],
        tracker: _SanaStageTracker,
    ):
        import torch

        generator = torch.Generator(device=self._execution_device(pipe) if self._execution_device(pipe).type == "cuda" else "cpu")
        seed = int(request.seed)
        if seed < 0:
            seed = random.randint(0, 2**31 - 1)
        generator.manual_seed(seed)

        def on_step_end(_pipe, step_index, _timestep, _callback_kwargs):  # noqa: ANN001
            step = int(step_index) + 1
            total = max(1, int(request.steps))
            stage_progress = 0.22 + 0.60 * (step / total)
            tracker.emit("inference", stage_progress, f"Denoising step {step}/{total}", step=step, total=total)
            return {}

        kwargs = dict(
            prompt=None,
            negative_prompt=None if prompt_inputs.get("negative_prompt_embeds") is not None else "",
            prompt_embeds=prompt_inputs["prompt_embeds"],
            prompt_attention_mask=prompt_inputs["prompt_attention_mask"],
            negative_prompt_embeds=prompt_inputs.get("negative_prompt_embeds"),
            negative_prompt_attention_mask=prompt_inputs.get("negative_prompt_attention_mask"),
            num_inference_steps=int(request.steps),
            guidance_scale=float(request.cfg_scale),
            height=int(request.height),
            width=int(request.width),
            frames=int(request.frames),
            generator=generator,
            output_type="latent",
            clean_caption=False,
            use_resolution_binning=bool(request.use_resolution_binning),
            max_sequence_length=int(request.max_sequence_length),
            callback_on_step_end=on_step_end,
            callback_on_step_end_tensor_inputs=[],
        )
        if source_image is not None:
            kwargs["image"] = Image.open(source_image).convert("RGB")
        output = pipe(**kwargs)
        return getattr(output, "frames", output[0] if isinstance(output, tuple) else output)

    def _decode_latents(
        self,
        pipe,  # noqa: ANN001
        latents,  # noqa: ANN001
        request: SanaVideoRequest,
        tracker: _SanaStageTracker,
    ) -> tuple[Any, str]:
        try:
            return self._decode_latents_once(pipe, latents, request, tracker=tracker), request.vae_tiling
        except Exception as exc:
            if request.vae_tiling != SANA_VIDEO_VAE_TILING_AUTO or not self._is_oom(exc):
                raise
            tracker.emit("decode", 0.86, "VAE decode OOM; retrying with tiling and slicing")
            self._clear_vae_cache(pipe)
            self._enable_vae_tiling(pipe)
            self._empty_cuda_cache()
            try:
                return (
                    self._decode_latents_once(pipe, latents, request, chunk_latent_frames=1, tracker=tracker),
                    "auto_retry_tiled_chunked",
                )
            except Exception as retry_exc:
                if not self._is_oom(retry_exc):
                    raise
                tracker.emit("decode", 0.88, "VAE tiled decode still OOM; retrying decode on CPU")
                self._clear_vae_cache(pipe)
                self._move_vae_to_cpu(pipe)
                self._empty_cuda_cache()
                return (
                    self._decode_latents_once(
                        pipe,
                        latents,
                        request,
                        chunk_latent_frames=1,
                        force_cpu=True,
                        tracker=tracker,
                    ),
                    "cpu_tiled_chunked",
                )

    def _decode_latents_once(
        self,
        pipe,
        latents,  # noqa: ANN001
        request: SanaVideoRequest,
        *,
        chunk_latent_frames: int = 0,
        force_cpu: bool = False,
        tracker: _SanaStageTracker | None = None,
    ):
        import torch

        vae = getattr(pipe, "vae", None)
        processor = getattr(pipe, "video_processor", None)
        if vae is None or processor is None:
            raise SanaVideoUnavailable("Sana Video pipeline is missing VAE or video processor.")
        if force_cpu:
            latents = latents.to(device="cpu", dtype=torch.float32)
        else:
            latents = latents.to(getattr(vae, "dtype", latents.dtype))
        latents = self._scale_latents_for_vae(vae, latents)
        with torch.inference_mode():
            video = self._decode_vae_latents(
                vae,
                latents,
                chunk_latent_frames=chunk_latent_frames,
                tracker=tracker,
                force_cpu=force_cpu,
            )
            video = video.detach()
            if request.use_resolution_binning:
                video = processor.resize_and_crop_tensor(video, int(request.width), int(request.height))
        return processor.postprocess_video(video, output_type="pil")

    def _decode_vae_latents(
        self,
        vae,
        latents,  # noqa: ANN001
        *,
        chunk_latent_frames: int = 0,
        tracker: _SanaStageTracker | None = None,
        force_cpu: bool = False,
    ):
        import torch

        if chunk_latent_frames <= 0 or int(latents.shape[2]) <= chunk_latent_frames:
            return vae.decode(latents, return_dict=False)[0]
        chunks = []
        starts = list(range(0, int(latents.shape[2]), int(chunk_latent_frames)))
        total = len(starts)
        device_label = "CPU" if force_cpu else "GPU"
        for chunk_index, start in enumerate(starts, start=1):
            end = min(start + int(chunk_latent_frames), int(latents.shape[2]))
            clear_cache = getattr(vae, "clear_cache", None)
            if callable(clear_cache):
                try:
                    clear_cache()
                except Exception:
                    logger.debug("Could not clear Sana VAE cache before chunk decode", exc_info=True)
            chunks.append(vae.decode(latents[:, :, start:end], return_dict=False)[0])
            if tracker is not None:
                tracker.emit(
                    "decode",
                    0.88 + 0.05 * (chunk_index / max(total, 1)),
                    f"Decoded VAE {device_label} chunk {chunk_index}/{total}",
                    step=chunk_index,
                    total=total,
                )
        clear_cache = getattr(vae, "clear_cache", None)
        if callable(clear_cache):
            try:
                clear_cache()
            except Exception:
                logger.debug("Could not clear Sana VAE cache after chunk decode", exc_info=True)
        return torch.cat(chunks, dim=2)

    @staticmethod
    def _scale_latents_for_vae(vae, latents):  # noqa: ANN001, ANN201
        import torch

        config = getattr(vae, "config", None)
        latents_mean = getattr(config, "latents_mean", None)
        latents_std = getattr(config, "latents_std", None)
        z_dim = getattr(config, "z_dim", getattr(config, "latent_channels", None))
        if latents_mean is None or latents_std is None:
            module_vars = vars(vae)
            latents_mean = module_vars.get("latents_mean")
            latents_std = module_vars.get("latents_std")
            z_dim = getattr(config, "latent_channels", z_dim)
        z_dim = z_dim or latents.shape[1]
        if latents_mean is None or latents_std is None:
            mean = torch.zeros(latents.shape[1], device=latents.device, dtype=latents.dtype)
            std = torch.ones(latents.shape[1], device=latents.device, dtype=latents.dtype)
        else:
            mean = torch.as_tensor(latents_mean, device=latents.device, dtype=latents.dtype)
            std = torch.as_tensor(latents_std, device=latents.device, dtype=latents.dtype)
        mean = mean.view(1, int(z_dim), 1, 1, 1)
        std = std.view(1, int(z_dim), 1, 1, 1)
        return latents * std + mean

    @staticmethod
    def _enable_vae_tiling(pipe) -> None:  # noqa: ANN001
        vae = getattr(pipe, "vae", None)
        if vae is None:
            return
        tiling = getattr(vae, "enable_tiling", None)
        if callable(tiling):
            try:
                tiling(
                    tile_sample_min_height=128,
                    tile_sample_min_width=128,
                    tile_sample_stride_height=96,
                    tile_sample_stride_width=96,
                )
            except TypeError:
                try:
                    tiling()
                except Exception:
                    logger.debug("Could not call Sana VAE enable_tiling", exc_info=True)
            except Exception:
                logger.debug("Could not call Sana VAE enable_tiling", exc_info=True)
        slicing = getattr(vae, "enable_slicing", None)
        if callable(slicing):
            try:
                slicing()
            except Exception:
                logger.debug("Could not call Sana VAE enable_slicing", exc_info=True)

    @staticmethod
    def _clear_vae_cache(pipe) -> None:  # noqa: ANN001
        vae = getattr(pipe, "vae", None)
        clear_cache = getattr(vae, "clear_cache", None)
        if callable(clear_cache):
            try:
                clear_cache()
            except Exception:
                logger.debug("Could not clear Sana VAE cache", exc_info=True)

    @staticmethod
    def _move_vae_to_cpu(pipe) -> None:  # noqa: ANN001
        vae = getattr(pipe, "vae", None)
        if vae is None:
            return
        try:
            from accelerate.hooks import remove_hook_from_module

            remove_hook_from_module(vae, recurse=True)
        except Exception:
            logger.debug("Could not remove Sana VAE accelerate hooks before CPU decode", exc_info=True)
        try:
            import torch

            vae.to(device="cpu", dtype=torch.float32)
            device_map = getattr(pipe, "hf_device_map", None)
            if isinstance(device_map, dict):
                device_map.pop("vae", None)
        except Exception:
            logger.debug("Could not move Sana VAE to CPU for fallback decode", exc_info=True)

    @staticmethod
    def _is_oom(exc: Exception) -> bool:
        text = str(exc).lower()
        if "out of memory" in text or "cuda error: out of memory" in text:
            return True
        try:
            import torch

            return isinstance(exc, torch.OutOfMemoryError)
        except Exception:
            return False

    @staticmethod
    def _is_sage_attention_mask_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return "sage" in text and "attn_mask" in text and "not supported" in text

    @staticmethod
    def _complex_human_instruction(pipe) -> Any:  # noqa: ANN001
        try:
            import inspect

            return inspect.signature(pipe.__call__).parameters["complex_human_instruction"].default
        except Exception:
            return None

    @staticmethod
    def _execution_device(pipe):  # noqa: ANN001, ANN205
        device = getattr(pipe, "_execution_device", None)
        if device is not None:
            return device
        try:
            return next(pipe.transformer.parameters()).device
        except Exception:
            import torch

            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _export_frames(frames, output_path: Path, *, fps: float) -> None:  # noqa: ANN001
        frames = SanaVideoService._normalize_frame_batch(frames)
        try:
            from diffusers.utils import export_to_video

            export_to_video(frames, str(output_path), fps=int(round(fps)))
        except Exception:
            from aiwf.infrastructure.video.processing import write_frames

            write_frames(frames, output_path, fps=fps)
        if not output_path.is_file() or output_path.stat().st_size <= 0:
            raise SanaVideoUnavailable(f"Sana video export did not create output: {output_path}")

    @staticmethod
    def _normalize_frame_batch(frames):  # noqa: ANN001, ANN201
        if isinstance(frames, tuple):
            frames = list(frames)
        if isinstance(frames, list) and len(frames) == 1 and isinstance(frames[0], (list, tuple)):
            return list(frames[0])
        return frames

    def _write_receipt(self, result: SanaVideoResult, request: SanaVideoRequest, *, run_id: str, stamp: str) -> Path:
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "ok",
            "request": request.model_dump(),
            "result": result.model_dump(),
        }
        return self._write_receipt_payload(payload, run_id=run_id, stamp=stamp)

    def _write_failure_receipt(
        self,
        request: SanaVideoRequest,
        *,
        run_id: str,
        stamp: str,
        output_path: Path,
        timings: dict[str, float],
        progress: list[SanaVideoProgressEvent],
        attention_backend: str,
        quantization: str,
        vae_tiling: str,
        error: Exception,
    ) -> Path:
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "error",
            "request": request.model_dump(),
            "output_path": str(output_path),
            "timings": timings,
            "progress": [event.model_dump() for event in progress],
            "attention_backend": attention_backend,
            "quantization": quantization,
            "vae_tiling": vae_tiling,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
        return self._write_receipt_payload(payload, run_id=run_id, stamp=stamp)

    def _write_receipt_payload(self, payload: dict[str, Any], *, run_id: str, stamp: str) -> Path:
        unique = self.log_dir() / f"sana_video_{stamp}_{run_id}.json"
        latest = self.log_dir() / "sana_video_latest.json"
        payload["receipt_path"] = str(unique)
        result = payload.get("result")
        if isinstance(result, dict):
            result["receipt_path"] = str(unique)
        text = json.dumps(payload, indent=2, default=str)
        unique.write_text(text, encoding="utf-8")
        latest.write_text(text, encoding="utf-8")
        return unique

    def _device(self):
        if self.devices is not None:
            try:
                return self.devices.device()
            except Exception:
                pass
        try:
            import torch

            return torch.device("cuda" if torch.cuda.is_available() and not self.flags.cpu else "cpu")
        except Exception:
            import torch

            return torch.device("cpu")

    def _dtype(self):
        try:
            import torch

            device = self._device()
            if device.type == "cuda" and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            if device.type == "cuda":
                return torch.float16
            return torch.float32
        except Exception:
            import torch

            return torch.float32

    @staticmethod
    def _empty_cuda_cache() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
