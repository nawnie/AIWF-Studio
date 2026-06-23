from __future__ import annotations

import gc
import logging

import gradio as gr

from aiwf.bootstrap import AppContext
from aiwf.core.domain.wan import WAN_RUNTIME_FAST_5B, WAN_TI2V_5B, WanI2VRequest
from aiwf.infrastructure.wan import WanUnavailable
from aiwf.services.wan import WanService

logger = logging.getLogger(__name__)


def uploaded_file_path(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return getattr(value, "name", None) or getattr(value, "path", None)


class WanVideoController:
    """Thin web controller for Wan generation and video-tab side effects."""

    def __init__(
        self,
        ctx: AppContext,
        service: WanService,
        *,
        step_summary_for_runtime,
        format_rate,
    ) -> None:
        self._ctx = ctx
        self._service = service
        self._step_summary_for_runtime = step_summary_for_runtime
        self._format_rate = format_rate

    @staticmethod
    def uploaded_file_path(value) -> str | None:
        return uploaded_file_path(value)

    def release_memory_before_postprocess(self, label: str) -> None:
        try:
            self._service.unload_models()
        except Exception:
            logger.exception("Failed to unload Wan models before %s post-processing.", label)
        try:
            self._ctx.generation.backend.unload()
        except Exception:
            logger.debug("Image backend unload before %s failed.", label, exc_info=True)
        gc.collect()
        try:
            self._ctx.generation.backend.devices.empty_cache()
        except Exception:
            logger.debug("Device cache cleanup before %s failed.", label, exc_info=True)

    def release_memory_before_reactor(self) -> None:
        self.release_memory_before_postprocess("ReActor")
        try:
            self._ctx.faceswap.unload()
        except Exception:
            logger.debug("Face swap unload before ReActor failed.", exc_info=True)

    def persist_last_used(
        self,
        *,
        high: str | None,
        low: str | None,
        vae: str | None,
        text_encoder: str | None,
        offload: str | None,
    ) -> None:
        settings = self._ctx.settings
        changed = False
        for attr, value in [
            ("last_wan_high", str(high or "")),
            ("last_wan_low", str(low or "")),
            ("last_wan_vae", str(vae or "")),
            ("last_wan_text_encoder", str(text_encoder or "")),
            ("last_wan_offload", str(offload or "balanced")),
        ]:
            if getattr(settings, attr, None) != value:
                setattr(settings, attr, value)
                changed = True
        if changed:
            try:
                self._ctx.save_settings()
            except Exception:
                logger.debug("Failed to persist last-used Wan settings.", exc_info=True)

    def build_request(
        self,
        *,
        prompt,
        negative,
        width,
        height,
        frames,
        fps,
        high_steps,
        low_steps,
        guidance,
        sampler,
        sigma_type,
        flow,
        seed,
        runtime_mode,
        high,
        low,
        vae,
        text_encoder,
        high_lora,
        high_lora_scale,
        low_lora,
        low_lora_scale,
        offload,
        vram_reserve_enabled,
        vram_reserve_mb,
        temporal_chunks,
        chunk_size,
        chunk_overlap,
        image_guidance_scale,
    ) -> WanI2VRequest:
        selected_runtime = str(runtime_mode or WAN_RUNTIME_FAST_5B)
        requires_dual_runtime = selected_runtime != WAN_RUNTIME_FAST_5B
        step_count, _step_ratio = self._step_summary_for_runtime(selected_runtime, high_steps, low_steps)
        return WanI2VRequest(
            prompt=prompt or "",
            negative_prompt=negative or "",
            width=int(width),
            height=int(height),
            num_frames=int(frames),
            fps=int(fps),
            steps=step_count,
            high_noise_steps=max(1, int(high_steps or 0)),
            low_noise_steps=max(1, int(low_steps or 0)),
            guidance_scale=float(guidance),
            sampler=str(sampler or "euler"),
            sigma_type=str(sigma_type or "simple"),
            flow_shift=float(flow),
            seed=int(seed),
            runtime_mode=selected_runtime,
            model_id=high if selected_runtime == WAN_RUNTIME_FAST_5B else WAN_TI2V_5B,
            offload=offload,
            vram_reserve_enabled=bool(vram_reserve_enabled),
            vram_reserve_mb=int(vram_reserve_mb or 0),
            high_noise_model_id=high if requires_dual_runtime else None,
            low_noise_model_id=low if requires_dual_runtime else None,
            high_noise_lora_id=high_lora if (requires_dual_runtime or selected_runtime == WAN_RUNTIME_FAST_5B) else None,
            high_noise_lora_scale=float(high_lora_scale),
            low_noise_lora_id=low_lora if requires_dual_runtime else None,
            low_noise_lora_scale=float(low_lora_scale),
            boundary_ratio=0.5,
            vae_id=vae or None,
            text_encoder_path=str(text_encoder or "").strip(),
            temporal_chunks=bool(temporal_chunks),
            chunk_size=int(chunk_size or 24),
            chunk_overlap=int(chunk_overlap or 0),
            image_guidance_scale=float(image_guidance_scale or 1.0),
        )

    def generate(self, request: WanI2VRequest, image, progress) -> object:
        def on_progress(step, total, steps_per_second=None, message=None):
            rate_text = self._format_rate(steps_per_second)
            message_text = str(message or "").strip()
            if message_text:
                desc = message_text
                if rate_text:
                    desc = f"{desc} - {rate_text}"
                lower = message_text.lower()
                if "writing" in lower:
                    ratio = 0.97
                elif "saved" in lower or "complete" in lower:
                    ratio = 0.99
                elif "decoding" in lower or "post-processing" in lower:
                    ratio = 0.93
                elif "denoise" in lower:
                    ratio = 0.08
                else:
                    ratio = 0.03
                progress(ratio, desc=desc)
                return
            desc = f"Video denoise {step}/{total}"
            if rate_text:
                desc = f"{desc} - {rate_text}"
            progress(min(0.90, step / max(1, total)), desc=desc)

        try:
            return self._service.generate(request, image, on_progress=on_progress)
        except WanUnavailable as exc:
            raise gr.Error(str(exc)) from exc
        except Exception as exc:
            logger.exception("Video generation failed")
            raise gr.Error(f"Video generation failed: {exc}") from exc

    def archive_bad_video(self, video_value) -> str:
        if not video_value:
            raise gr.Error("Generate a video first.")
        record = self._ctx.failure_archive.archive_bad_video(
            video_value,
            note="Marked from Video tab",
            extra={"source": "wan_i2v_tab"},
        )
        if not record.ok:
            return f"**Failure gallery** -- saved with archive warnings: {record.archive_dir}"
        return f"**Failure gallery** -- saved bad result: {record.archive_dir}"
