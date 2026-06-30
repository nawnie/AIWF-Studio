from __future__ import annotations

import gc
import json
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from PIL import Image

from aiwf.core.config.settings import RuntimeFlags, UserSettings
from aiwf.core.domain.audio import AudioGenerationOptions
from aiwf.core.domain.sana_video import (
    SANA_VIDEO_MODEL_REPO_480P,
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
    sana_video_model_folder_name,
)
from aiwf.infrastructure.video.processing import VideoProcessor
from aiwf.services.audio import AudioGenerationService, AudioUnavailable

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
    ) -> None:
        self.flags = flags or RuntimeFlags()
        self.settings = settings or UserSettings()
        self.devices = devices
        self.supervisor = supervisor

    def models_root(self) -> Path:
        return self.flags.resolved_models_dir() / "sana-video" / "Diffusers"

    def default_model_path(self) -> Path:
        return self.models_root() / sana_video_model_folder_name(SANA_VIDEO_MODEL_REPO_480P)

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
        if not model_index.is_file():
            lines.append("- Model snapshot missing. Download the 480p SANA-Video Diffusers folder first.")
        return "\n".join(lines)

    @staticmethod
    def runtime_available() -> bool:
        try:
            import diffusers

            return hasattr(diffusers, "SanaVideoPipeline") and hasattr(diffusers, "SanaImageToVideoPipeline")
        except Exception:
            return False

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
        tracker = _SanaStageTracker(on_progress)
        timings: dict[str, float] = {}
        prompt = (request.prompt or "").strip()
        if not prompt:
            raise SanaVideoUnavailable("Enter a Sana video prompt first.")
        if not self.runtime_available():
            raise SanaVideoUnavailable("Installed Diffusers does not expose SanaVideoPipeline.")

        model_path = resolve_sana_video_path(request.model_path, self.default_model_path(), self.flags.data_dir)
        if not (model_path / "model_index.json").is_file():
            raise SanaVideoUnavailable(f"Sana video model folder missing model_index.json: {model_path}")

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
            pipe, load_info = self._load_pipeline(model_path, request, image_to_video=request.wants_image_to_video)
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
        from diffusers import SanaImageToVideoPipeline, SanaVideoPipeline

        pipeline_cls = SanaImageToVideoPipeline if image_to_video else SanaVideoPipeline
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
