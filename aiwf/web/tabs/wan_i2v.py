from __future__ import annotations

import logging
import math
from pathlib import Path

import gradio as gr

from aiwf.core.domain.audio import AudioGenerationOptions
from aiwf.core.domain.enhance import RestoreOptions
from aiwf.core.domain.faceswap import FaceSwapOptions
from aiwf.core.domain.ltx import (
    LTX_GEMMA_BACKEND_GGUF,
    LTX_GEMMA_BACKEND_HF_SAFETENSORS,
    LTX_PIPELINE_DIFFUSERS_2B,
    LTX_PIPELINE_DISTILLED,
    LTX_PIPELINE_ONE_STAGE,
    LTX_T5_TOKENIZER,
    LtxVideoRequest,
    snap_ltx_num_frames,
)
from aiwf.core.domain.rife import RifeOptions
from aiwf.core.domain.vsr import VideoFxAigsOptions, VideoFxDenoiseOptions, VideoFxRelightOptions, VsrOptions
from aiwf.bootstrap import AppContext
from aiwf.core.domain.wan import (
    WAN_RUNTIME_FAST_5B,
    WAN_RUNTIME_HIGH_LOW,
    WAN_RUNTIME_HIGH_LOW_FP8,
    WanI2VRequest,
    duration_seconds_for_frames,
    frames_for_duration_seconds,
)
from aiwf.infrastructure.wan.sampler_policy import audit_wan_sampler_settings
from aiwf.infrastructure.faceswap import FaceSwapUnavailable
from aiwf.infrastructure.rife import RifeUnavailable
from aiwf.infrastructure.video import VideoError, extract_first_frame
from aiwf.services.rife import RifeService
from aiwf.services.audio import AudioGenerationService, AudioUnavailable
from aiwf.services.ltx import LtxService, LtxUnavailable
from aiwf.services.vsr import VsrService, VsrUnavailable
from aiwf.services.wan import (
    WanService,
)
from aiwf.services.wan_models import (
    wan_model_pair_compatibility,
    wan_model_quant_family,
    wan_model_header_info,
    wan_model_stage_role,
    wan_model_storage_family,
    wan_selectable_loras,
    wan_selectable_transformers,
)
from aiwf.web.registry import WebRegistry
from aiwf.web.studio.resolution import (
    ASPECT_RATIO_PRESETS,
    NON_SQUARE_ASPECT_RATIO_PRESETS,
    dimensions_from_generation_preset,
)
from aiwf.web.video.wan_controller import WanVideoController

_SERVICES: dict[int, WanService] = {}
_RIFE_SERVICES: dict[int, RifeService] = {}
_VSR_SERVICES: dict[int, VsrService] = {}
_AUDIO_SERVICES: dict[int, AudioGenerationService] = {}
_LTX_SERVICES: dict[int, LtxService] = {}
_wan_cancel_flag: list[bool] = [False]
VIDEO_SIZE_PRESETS: tuple[int, ...] = (480, 512, 568, 640, 768, 896, 1024)
_WAN_FAST_OFFLOAD_CHOICES = [
    ("Balanced: model offload", "balanced"),
    ("Low VRAM: model offload", "model"),
    ("Sequential: slow fallback", "sequential"),
]
_WAN_GGUF_OFFLOAD_CHOICES = [
    ("Low VRAM GGUF: active stage swaps", "model"),
    ("Balanced GGUF: active stage swaps, VAE resident", "balanced"),
    ("Sequential GGUF: slow fallback", "sequential"),
]
_WAN_FP8_OFFLOAD_CHOICES = [
    ("Tested 14B FP8: streamed group offload", "streamed"),
]
VSR_UPSCALE_MODE_CHOICES = [
    ("Low", 1),
    ("Medium", 2),
    ("High", 3),
    ("Ultra", 4),
]
VSR_CLEANUP_MODE_CHOICES = [
    ("Denoise Low", 8),
    ("Denoise Medium", 9),
    ("Denoise High", 10),
    ("Denoise Ultra", 11),
    ("Deblur Low", 12),
    ("Deblur Medium", 13),
    ("Deblur High", 14),
    ("Deblur Ultra", 15),
    ("High bitrate Low", 16),
    ("High bitrate Medium", 17),
    ("High bitrate High", 18),
    ("High bitrate Ultra", 19),
]
AIGS_COMP_CHOICES = [
    ("Background blur", 6),
    ("Matte mask", 0),
    ("Mask overlay", 1),
    ("Green background", 2),
    ("White background", 3),
    ("Original frame", 4),
    ("Background image", 5),
]
RELIGHT_BG_MODE_CHOICES = [
    ("Original background", 0),
    ("Blur original background", 1),
    ("HDR background", 2),
    ("Background image", 3),
    ("Blur background image", 4),
]
logger = logging.getLogger(__name__)


def unload_wan_for_context(ctx: AppContext) -> bool:
    svc = _SERVICES.get(id(ctx))
    if svc is None:
        return False
    try:
        svc.unload_models()
        return True
    except Exception:
        logger.exception("Failed to unload Wan video pipeline")
        return False


def _service(ctx: AppContext) -> WanService:
    svc = _SERVICES.get(id(ctx))
    if svc is None:
        svc = WanService(
            ctx.flags,
            ctx.settings,
            unload_image_models=ctx.generation.backend.unload,
            supervisor=ctx.supervisor,
            failure_archive=ctx.failure_archive,
            genlog=ctx.genlog,
        )
        _SERVICES[id(ctx)] = svc
    return svc


def _rife_service(ctx: AppContext) -> RifeService:
    svc = _RIFE_SERVICES.get(id(ctx))
    if svc is None:
        svc = RifeService(ctx.flags, ctx.settings, ctx.generation.backend.devices, supervisor=ctx.supervisor)
        _RIFE_SERVICES[id(ctx)] = svc
    return svc


def _vsr_service(ctx: AppContext) -> VsrService:
    svc = _VSR_SERVICES.get(id(ctx))
    if svc is None:
        svc = VsrService(ctx.flags, ctx.settings, supervisor=ctx.supervisor)
        _VSR_SERVICES[id(ctx)] = svc
    return svc


def _audio_service(ctx: AppContext) -> AudioGenerationService:
    svc = _AUDIO_SERVICES.get(id(ctx))
    if svc is None:
        svc = AudioGenerationService(
            ctx.flags,
            ctx.settings,
            ctx.generation.backend.devices,
            supervisor=ctx.supervisor,
        )
        _AUDIO_SERVICES[id(ctx)] = svc
    return svc


def _ltx_service(ctx: AppContext) -> LtxService:
    svc = _LTX_SERVICES.get(id(ctx))
    if svc is None:
        svc = LtxService(ctx.flags, ctx.settings, supervisor=ctx.supervisor)
        _LTX_SERVICES[id(ctx)] = svc
    return svc


def _format_it_s(steps_per_second) -> str:
    try:
        rate = float(steps_per_second)
    except (TypeError, ValueError):
        return ""
    if rate <= 0 or rate != rate:
        return ""
    return f"{rate:.3f} it/s ({1.0 / rate:.2f} s/it)"


def _offload_choices_for_runtime(runtime_value: str | None) -> list[tuple[str, str]]:
    # Team note: route-specific UI choices are intentional guardrails. Keep 5B,
    # 14B FP8 safetensors, and GGUF high/low families filtered apart so users
    # do not accidentally mix incompatible runtime/model/settings paths.
    selected_runtime = str(runtime_value or WAN_RUNTIME_FAST_5B)
    if selected_runtime == WAN_RUNTIME_HIGH_LOW_FP8:
        return list(_WAN_FP8_OFFLOAD_CHOICES)
    if selected_runtime == WAN_RUNTIME_HIGH_LOW:
        return list(_WAN_GGUF_OFFLOAD_CHOICES)
    return list(_WAN_FAST_OFFLOAD_CHOICES)


def _default_offload_for_runtime(runtime_value: str | None, current_value: str | None = None) -> str:
    choices = _offload_choices_for_runtime(runtime_value)
    values = [value for _label, value in choices]
    if current_value and current_value in values:
        return current_value
    return values[0] if values else "balanced"


def _model_allowed_for_runtime(model_id: str | None, runtime_value: str | None) -> bool:
    selected_runtime = str(runtime_value or WAN_RUNTIME_FAST_5B)
    storage = wan_model_storage_family(model_id)
    size_class = _model_size_class_for_filter(model_id)
    if selected_runtime == WAN_RUNTIME_FAST_5B:
        return storage == "safetensors" and size_class in {"", "5b"}
    if selected_runtime == WAN_RUNTIME_HIGH_LOW_FP8:
        return storage == "safetensors" and size_class in {"", "14b"}
    if selected_runtime == WAN_RUNTIME_HIGH_LOW:
        return storage == "gguf"
    return True


def _model_size_class_for_filter(model_id: str | None) -> str:
    text = str(model_id or "").strip()
    if not text:
        return ""
    try:
        info = wan_model_header_info(text)
        if info.ok and info.size_class:
            return info.size_class
    except Exception:
        pass
    name = Path(text).name.lower()
    if "5b" in name or "ti2v" in name:
        return "5b"
    if "14b" in name or "a14b" in name:
        return "14b"
    return ""


def _existing_video_output_path(output_path: str | Path | None, stage: str) -> str:
    path = Path(str(output_path or "")).expanduser()
    if not path.is_file():
        raise VideoError(f"{stage} did not create a video file: {path}")
    return str(path.resolve())


def _step_summary_for_runtime(runtime_value: str | None, high_value, low_value) -> tuple[int, float]:
    high = max(1, int(high_value or 0))
    if str(runtime_value or WAN_RUNTIME_FAST_5B) == WAN_RUNTIME_FAST_5B:
        return high, 1.0
    low = max(1, int(low_value or 0))
    total = high + low
    return total, round(high / total, 3)


def _dual_step_split_from_total(total_value) -> tuple[int, int]:
    total = max(2, int(total_value or 0))
    high = max(1, (total + 1) // 2)
    low = max(1, total - high)
    return high, low


def _rife_multiplier_for_target(input_fps: int | float, target_fps: int | float) -> int:
    safe_input = max(1.0, float(input_fps or 1))
    safe_target = max(1.0, float(target_fps or safe_input))
    return max(2, min(8, int(math.ceil(safe_target / safe_input))))


def register_wan_i2v(registry: WebRegistry) -> None:
    @registry.tab("Video", order=2)
    def build(ctx: AppContext, tab: gr.Tab | None = None) -> None:
        service = _service(ctx)
        rife_service = _rife_service(ctx)
        vsr_service = _vsr_service(ctx)
        audio_service = _audio_service(ctx)
        ltx_service = _ltx_service(ctx)
        wan_controller = WanVideoController(
            ctx,
            service,
            step_summary_for_runtime=_step_summary_for_runtime,
            format_rate=_format_it_s,
        )
        rife_ckpts = rife_service.list_checkpoints()
        default_rife_ckpt = rife_service.default_checkpoint()
        vsr_help = vsr_service.folder_help()
        relight_hdr_choices = vsr_service.relighting_hdr_choices()
        audio_music_models = audio_service.music_model_choices()
        audio_sfx_models = audio_service.sfx_model_choices()
        audio_video_models = audio_service.video_audio_model_choices()

        def _faceswap_model_choices() -> list[tuple[str, str]]:
            return [(m.title, m.id) for m in ctx.faceswap.list_models()]

        def _reactor_face_model_choices() -> list[tuple[str, str]]:
            return [(m.title, m.id) for m in ctx.faceswap.list_face_models()]

        def _restorer_choices() -> list[tuple[str, str]]:
            return [(m.title, m.id) for m in ctx.enhance.list_restorers()]

        reactor_swapper_choices = _faceswap_model_choices()
        reactor_face_model_choices = _reactor_face_model_choices()
        reactor_restorer_choices = _restorer_choices()

        # Labeled (display_name, identifier) choices - read from model file headers.
        all_labeled = service.list_local_models_labeled() if hasattr(service, "list_local_models_labeled") else []
        if not all_labeled:
            all_labeled = [(m, m) for m in service.list_local_models()]

        all_lora_choices = service.list_local_loras() if hasattr(service, "list_local_loras") else []

        # Sort high/low noise models to the top of each dropdown; unknown-role in both.
        high_labeled = [c for c in all_labeled if "high" in c[0].lower() or "high" in c[1].lower()]
        low_labeled  = [c for c in all_labeled if "low"  in c[0].lower() or "low"  in c[1].lower()]
        other_labeled = [c for c in all_labeled if c not in high_labeled and c not in low_labeled]
        high_labeled = list(dict.fromkeys(high_labeled + other_labeled + all_labeled))
        low_labeled  = list(dict.fromkeys(low_labeled  + other_labeled + all_labeled))

        def _filter_stage_choices(
            labeled: list[tuple[str, str]],
            *,
            runtime_value: str,
            stage: str | None,
            peer_value: str | None,
        ) -> list[tuple[str, str]]:
            ids = [value for _label, value in labeled]
            selectable_runtime = "" if str(runtime_value or "") == WAN_RUNTIME_HIGH_LOW else runtime_value
            allowed = set(
                wan_selectable_transformers(
                    ids,
                    runtime_mode=str(selectable_runtime or ""),
                    want_role=stage,
                    peer_id=peer_value,
                )
            )
            return [
                (label, value)
                for label, value in labeled
                if value in allowed and _model_allowed_for_runtime(value, runtime_value)
            ]

        def _filter_lora_choices(runtime_value: str) -> list[str]:
            return wan_selectable_loras(
                all_lora_choices,
                runtime_mode=str(runtime_value or WAN_RUNTIME_FAST_5B),
            )

        def _valid_or_first(value: str | None, choices: list[tuple[str, str]]) -> str | None:
            ids = [v for _, v in choices]
            if value and value in ids:
                return value
            return ids[0] if ids else None

        def _pair_status(high_value: str | None, low_value: str | None) -> str:
            high_text = str(high_value or "").strip()
            low_text = str(low_value or "").strip()
            if not (high_text and low_text):
                return ""
            high_storage = wan_model_storage_family(high_text)
            low_storage = wan_model_storage_family(low_text)
            high_quant = wan_model_quant_family(high_text)
            low_quant = wan_model_quant_family(low_text)
            pair_check = wan_model_pair_compatibility(high_text, low_text)
            if pair_check.errors:
                return "**Model pair blocked:** " + " ".join(pair_check.errors)
            if high_storage != "unknown" and low_storage != "unknown" and high_storage != low_storage:
                return f"**Model pair blocked:** {high_storage} high + {low_storage} low."
            if high_quant != "unknown" and low_quant != "unknown" and high_quant != low_quant:
                return f"**Model pair blocked:** {high_quant.upper()} high + {low_quant.upper()} low."
            parts = [p for p in (high_storage, high_quant) if p != "unknown"]
            status = "**Model pair:** " + (" / ".join(parts) if parts else "stage roles only")
            if pair_check.warnings:
                status += "\n\n" + "\n\n".join(f"**Pair warning:** {warning}" for warning in pair_check.warnings)
            return status

        def _vae_looks_21(value: str | None) -> bool:
            text = str(value or "").lower()
            return "2.1" in text or "wan21" in text or "_21" in text

        def _vae_looks_22(value: str | None) -> bool:
            text = str(value or "").lower()
            return "2.2" in text or "wan22" in text or "_22" in text

        def _runtime_trace_status(
            runtime_value: str | None,
            high_value: str | None,
            low_value: str | None,
            vae_value: str | None,
            text_encoder_value: str | None,
            offload_value: str | None,
            width_value: int | float | str | None = None,
            height_value: int | float | str | None = None,
            frames_value: int | float | str | None = None,
            high_steps_value: int | float | str | None = None,
            low_steps_value: int | float | str | None = None,
            temporal_chunks_value: bool | None = None,
            chunk_size_value: int | float | str | None = None,
            chunk_overlap_value: int | float | str | None = None,
            low_guidance_value: int | float | str | None = None,
            sampler_value: str | None = None,
            flow_shift_value: int | float | str | None = None,
        ) -> str:
            selected_runtime = str(runtime_value or WAN_RUNTIME_FAST_5B)
            selected_vae = str(vae_value or "").strip()
            selected_te = str(text_encoder_value or "").strip() or "Default/local component"
            selected_offload = str(offload_value or "").strip() or "balanced"
            try:
                selected_width = int(width_value or 0)
                selected_height = int(height_value or 0)
                selected_frames = int(frames_value or 0)
            except (TypeError, ValueError):
                selected_width = selected_height = selected_frames = 0
            try:
                selected_chunk_size = int(chunk_size_value or 24)
                selected_chunk_overlap = int(chunk_overlap_value or 0)
            except (TypeError, ValueError):
                selected_chunk_size = 24
                selected_chunk_overlap = 0
            try:
                selected_low_guidance = float(low_guidance_value or 1.0)
            except (TypeError, ValueError):
                selected_low_guidance = 1.0
            selected_temporal_chunks = bool(temporal_chunks_value)
            step_high = high_steps_value if high_steps_value is not None else (20 if selected_runtime == WAN_RUNTIME_FAST_5B else 4)
            step_low = low_steps_value if low_steps_value is not None else 4
            selected_total_steps, selected_step_ratio = _step_summary_for_runtime(selected_runtime, step_high, step_low)
            lines: list[str] = []
            warnings: list[str] = []
            if selected_runtime == WAN_RUNTIME_FAST_5B:
                lines.append("**Route:** Fast 5B TI2V - single transformer")
                lines.append(f"**Model:** `{high_value or 'not selected'}`")
                lines.append("**Low-noise control:** ignored and locked for this route")
                lines.append("**Expected VAE:** Wan 2.2 / 48-channel")
                if selected_vae and _vae_looks_21(selected_vae) and not _vae_looks_22(selected_vae):
                    warnings.append("Selected VAE looks Wan 2.1; 5B needs Wan 2.2.")
            else:
                if selected_runtime == WAN_RUNTIME_HIGH_LOW_FP8:
                    lines.append("**Route:** Full 14B FP8 safetensors - dual transformer")
                    if selected_offload != "streamed":
                        warnings.append("Full 14B FP8 is locked to streamed group offload in the UI.")
                else:
                    lines.append("**Route:** GGUF high/low pair - dual transformer")
                lines.append(f"**High:** `{high_value or 'not selected'}`")
                lines.append(f"**Low:** `{low_value or 'not selected'}`")
                lines.append("**Expected VAE:** Wan 2.1 / 16-channel")
                if selected_vae and _vae_looks_22(selected_vae) and not _vae_looks_21(selected_vae):
                    warnings.append("Selected VAE looks Wan 2.2; 14B high/low needs Wan 2.1.")
                pair_text = _pair_status(high_value, low_value)
                if pair_text:
                    lines.append(pair_text)
                if (
                    selected_offload == "balanced"
                    and selected_width >= 768
                    and selected_height >= 768
                    and selected_frames >= 81
                ):
                    warnings.append("14B 768x768 / 81 frames OOM'd on balanced; use Low VRAM/model offload.")
            lines.append(f"**Text encoder:** `{selected_te}`")
            lines.append(f"**Offload:** `{selected_offload}`")
            if selected_runtime == WAN_RUNTIME_FAST_5B:
                lines.append(f"**Steps:** `{selected_total_steps}`")
            else:
                lines.append(
                    f"**Steps:** `{selected_total_steps}` "
                    f"(high {max(1, int(step_high or 0))} / low {max(1, int(step_low or 0))}, "
                    f"split {selected_step_ratio:g})"
                )
            lines.append(
                f"**Temporal chunks:** `{'on' if selected_temporal_chunks else 'off'}` "
                f"(latent chunk {selected_chunk_size}, overlap {selected_chunk_overlap})"
            )
            if selected_runtime != WAN_RUNTIME_FAST_5B:
                lines.append(f"**Low-noise guidance:** `{selected_low_guidance:g}`")
            try:
                selected_flow_shift = float(flow_shift_value if flow_shift_value is not None else 5.0)
            except (TypeError, ValueError):
                selected_flow_shift = 5.0
            selected_sampler = str(sampler_value or "unipc").strip().lower() or "unipc"
            lines.append(f"**Sampler:** `{selected_sampler}` | **Flow shift:** `{selected_flow_shift:g}`")
            sampler_audit = audit_wan_sampler_settings(
                WanI2VRequest(
                    runtime_mode=selected_runtime,
                    sampler=selected_sampler,
                    flow_shift=selected_flow_shift,
                ),
                enforce_5b_calibration=False,
            )
            if sampler_audit.errors:
                warnings.extend(sampler_audit.errors)
            if sampler_audit.warnings:
                warnings.extend(sampler_audit.warnings)
            if warnings:
                lines.extend(f"**Blocked:** {warning}" for warning in warnings)
            return "\n\n".join(lines)

        # Load persisted defaults
        _s = ctx.settings
        _last_high = getattr(_s, "last_wan_high", "")
        _last_low  = getattr(_s, "last_wan_low", "")
        _last_vae  = getattr(_s, "last_wan_vae", "")
        _last_te   = getattr(_s, "last_wan_text_encoder", "")
        _last_offload = getattr(_s, "last_wan_offload", "balanced")
        _last_sampler = str(getattr(_s, "last_wan_sampler", "") or "unipc").strip().lower() or "unipc"
        if _last_sampler not in {"unipc", "euler", "heun"}:
            _last_sampler = "unipc"
        try:
            _last_flow_shift = float(getattr(_s, "last_wan_flow_shift", 5.0) or 5.0)
        except (TypeError, ValueError):
            _last_flow_shift = 5.0
        _last_runtime = str(getattr(_s, "last_wan_runtime_mode", "") or WAN_RUNTIME_FAST_5B)
        if _last_runtime not in {WAN_RUNTIME_FAST_5B, WAN_RUNTIME_HIGH_LOW, WAN_RUNTIME_HIGH_LOW_FP8}:
            _last_runtime = WAN_RUNTIME_FAST_5B
        _offload_default = _default_offload_for_runtime(_last_runtime, _last_offload)

        def _best_default(labeled: list[tuple[str, str]], persisted: str) -> str | None:
            ids = [v for _, v in labeled]
            if persisted and persisted in ids:
                return persisted
            return ids[0] if ids else None

        vae_labeled = service.list_local_vaes_labeled() if hasattr(service, "list_local_vaes_labeled") else []
        if not vae_labeled:
            vae_labeled = [("Default VAE", "")]
        def _preferred_vae_for_runtime(runtime_value: str, current_value: str | None = None) -> str | None:
            ids = [v for _, v in vae_labeled if v]
            if current_value and current_value in ids:
                current_lower = current_value.lower()
                if runtime_value == WAN_RUNTIME_FAST_5B and "wan2.2" in current_lower:
                    return current_value
                if runtime_value != WAN_RUNTIME_FAST_5B and any(
                    token in current_lower for token in ("wan2.1_vae", "wan_2.1_vae", "wan21_vae")
                ):
                    return current_value
            if runtime_value == WAN_RUNTIME_FAST_5B:
                return next((v for v in ids if "wan2.2" in v.lower() and "vae" in v.lower()), None) or (
                    ids[0] if ids else None
                )
            return next(
                (
                    v
                    for v in ids
                    if any(token in v.lower() for token in ("wan2.1_vae", "wan_2.1_vae", "wan21_vae"))
                ),
                None,
            ) or next((v for v in ids if "wan" in v.lower() and "vae" in v.lower()), None) or (ids[0] if ids else None)

        preferred_vae_id = _preferred_vae_for_runtime(_last_runtime, _last_vae)

        te_labeled = service.list_local_text_encoders_labeled() if hasattr(service, "list_local_text_encoders_labeled") else []
        default_te_labeled = [("Default (full precision bundled encoder)", "")] + te_labeled
        default_te = _last_te if _last_te else (service.default_text_encoder() if hasattr(service, "default_text_encoder") else "")

        initial_lora_choices = _filter_lora_choices(_last_runtime)
        if _last_runtime == WAN_RUNTIME_FAST_5B:
            initial_high_choices = _filter_stage_choices(
                all_labeled,
                runtime_value=_last_runtime,
                stage=None,
                peer_value=None,
            )
            initial_low_choices: list[tuple[str, str]] = []
            initial_high = _valid_or_first(_last_high, initial_high_choices)
            initial_low = None
        else:
            initial_high_choices = _filter_stage_choices(
                high_labeled,
                runtime_value=_last_runtime,
                stage="high",
                peer_value=None,
            )
            initial_high = _valid_or_first(_last_high, initial_high_choices)
            initial_low_choices = _filter_stage_choices(
                low_labeled,
                runtime_value=_last_runtime,
                stage="low",
                peer_value=initial_high,
            )
            initial_low = _valid_or_first(_last_low, initial_low_choices)

        _initial_high_label = (
            "5B transformer" if _last_runtime == WAN_RUNTIME_FAST_5B else "High noise transformer"
        )
        _initial_low_interactive = _last_runtime != WAN_RUNTIME_FAST_5B
        _initial_pair_status = (
            "" if _last_runtime == WAN_RUNTIME_FAST_5B else _pair_status(initial_high, initial_low)
        )

        default_video_size = 512
        default_video_ratio = "1:1"

        with gr.Column(elem_classes=["aiwf-wan", "aiwf-video", "aiwf-mode-video"]):
            with gr.Column(elem_classes=["aiwf-page-header"]):
                gr.Markdown("Video", elem_classes=["aiwf-section-label"])
                video_mode = gr.Radio(
                    show_label=False,
                    container=False,
                    choices=[("Image2Video", "i2v")],
                    value="i2v",
                    elem_classes=["aiwf-mode-toggle"],
                )
                gr.Markdown(
                    "Wan image-to-video. Fast 5B, full 14B FP8, or matched GGUF high/low pairs.",
                    elem_classes=["aiwf-page-intro"],
                )
                gr.Markdown(service.folder_help(), elem_classes=["aiwf-page-path"])

            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=340, elem_classes=["aiwf-panel"]):
                    source = gr.Image(label="Source image", type="pil", sources=["upload", "clipboard"])
                    prompt = gr.Textbox(label="Prompt", lines=3, placeholder="Describe the motion / scene")
                    negative = gr.Textbox(label="Negative prompt", lines=2, value="")

                    runtime_mode = gr.Radio(
                        label="Wan route",
                        choices=[
                            ("Fast: 5B TI2V", WAN_RUNTIME_FAST_5B),
                            ("Full: 14B FP8 high/low", WAN_RUNTIME_HIGH_LOW_FP8),
                            ("GGUF: high/low pair", WAN_RUNTIME_HIGH_LOW),
                        ],
                        value=_last_runtime,
                        info="Routes are separated so FP8 safetensors and GGUF pairs cannot be mixed.",
                    )
                    runtime_previous = gr.State(_last_runtime)

                    gr.Markdown("Models", elem_classes=["aiwf-section-label"])
                    high_noise = gr.Dropdown(
                        label=_initial_high_label,
                        choices=initial_high_choices,
                        value=initial_high,
                        allow_custom_value=True,
                        interactive=True,
                        info="Selected route controls which files appear here.",
                    )
                    low_noise = gr.Dropdown(
                        label="Low noise transformer",
                        choices=initial_low_choices,
                        value=initial_low,
                        allow_custom_value=True,
                        interactive=_initial_low_interactive,
                        info="Late denoising stage. Must match the selected high-noise file.",
                    )
                    model_pair_status = gr.Markdown(
                        _initial_pair_status,
                        elem_classes=["aiwf-settings-paths"],
                    )
                    text_encoder = gr.Dropdown(
                        label="Text encoder (UMT5-XXL)",
                        choices=default_te_labeled,
                        value=default_te if default_te else "",
                        allow_custom_value=True,
                        info="UMT5-XXL only. Use GGUF/FP8 text encoders only if already tested locally.",
                    )
                    vae_id = gr.Dropdown(
                        label="VAE",
                        choices=vae_labeled,
                        value=preferred_vae_id,
                        allow_custom_value=True,
                        info="5B TI2V uses Wan 2.2 VAE. 14B high/low usually uses Wan 2.1 VAE.",
                    )
                    route_status = gr.Markdown(
                        _runtime_trace_status(
                            _last_runtime,
                            initial_high,
                            None,
                            preferred_vae_id,
                            default_te,
                            _offload_default,
                            sampler_value=_last_sampler,
                            flow_shift_value=_last_flow_shift,
                        ),
                        elem_classes=["aiwf-settings-paths"],
                    )

                    gr.Markdown("Stage LoRAs", elem_classes=["aiwf-section-label"])
                    high_lora = gr.Dropdown(
                        label="High noise LoRA",
                        choices=initial_lora_choices,
                        value=None,
                        allow_custom_value=False,
                        interactive=False,
                        info="Optional high-stage LoRA.",
                    )
                    with gr.Row():
                        high_lora_scale = gr.Slider(
                            0.0,
                            2.0,
                            value=1.0,
                            step=0.05,
                            label="High LoRA strength",
                            interactive=False,
                        )
                        low_lora_scale = gr.Slider(
                            0.0,
                            2.0,
                            value=1.0,
                            step=0.05,
                            label="Low LoRA strength",
                            interactive=False,
                        )
                    low_lora = gr.Dropdown(
                        label="Low noise LoRA",
                        choices=initial_lora_choices,
                        value=None,
                        allow_custom_value=False,
                        interactive=False,
                        info="Optional low-stage LoRA.",
                    )

                    gr.Markdown("Runtime", elem_classes=["aiwf-section-label"])
                    offload = gr.Dropdown(
                        label="VRAM / offload",
                        choices=_offload_choices_for_runtime(WAN_RUNTIME_FAST_5B),
                        value=_offload_default,
                        info="Choices are route-specific; full 14B FP8 uses the tested streamed path.",
                    )
                    vram_reserve_enabled = gr.Checkbox(
                        value=False,
                        label="Keep some VRAM free",
                        info="Smaller reserve lets AIWF use more VRAM.",
                    )
                    vram_reserve_mb = gr.Slider(
                        0,
                        8192,
                        value=1024,
                        step=128,
                        label="Keep free (MB)",
                        info="0 = no reserve. 1024 = keep about 1 GB free.",
                    )

                with gr.Column(scale=1, min_width=340, elem_classes=["aiwf-panel"]):
                    gr.Markdown("Resolution", elem_classes=["aiwf-section-label"])
                    with gr.Column(elem_classes=["aiwf-resolution-presets"]):
                        with gr.Row(elem_classes=["aiwf-resolution-row"]):
                            gr.HTML('<div class="aiwf-resolution-heading">Size</div>')
                            resolution_size = gr.Radio(
                                show_label=False,
                                container=False,
                                choices=[(str(size), size) for size in VIDEO_SIZE_PRESETS],
                                value=default_video_size,
                                elem_classes=["aiwf-resolution-toggle", "aiwf-resolution-size"],
                            )
                        with gr.Row(elem_classes=["aiwf-resolution-row"]):
                            gr.HTML('<div class="aiwf-resolution-heading">Ratio</div>')
                            with gr.Column(elem_classes=["aiwf-resolution-ratio-stack"]):
                                resolution_ratio = gr.Radio(
                                    show_label=False,
                                    container=False,
                                    choices=list(NON_SQUARE_ASPECT_RATIO_PRESETS),
                                    value=None,
                                    elem_classes=["aiwf-resolution-toggle", "aiwf-resolution-ratio"],
                                )
                                resolution_ratio_square = gr.Radio(
                                    show_label=False,
                                    container=False,
                                    choices=[("1:1", "1:1")],
                                    value=default_video_ratio,
                                    elem_classes=[
                                        "aiwf-resolution-toggle",
                                        "aiwf-resolution-ratio",
                                        "aiwf-resolution-ratio-square",
                                    ],
                                )
                    with gr.Row():
                        width = gr.Slider(128, 1280, value=512, step=8, label="Width")
                        height = gr.Slider(128, 1280, value=512, step=8, label="Height")

                    gr.Markdown("Motion", elem_classes=["aiwf-section-label"])
                    with gr.Row():
                        fps = gr.Slider(1, 24, value=16, step=1, label="FPS")
                        duration_seconds = gr.Slider(1, 10, value=5, step=1, label="Duration (seconds)")
                    with gr.Row():
                        num_frames = gr.Number(value=81, precision=0, label="Frames", interactive=False)
                        guidance = gr.Slider(1.0, 12.0, value=5.0, step=0.5, label="Guidance (CFG)")
                    frame_summary = gr.Markdown(
                        "**Frames:** 81 - **Duration:** 5.0s snapped for Wan",
                        elem_classes=["aiwf-settings-paths"],
                    )

                    gr.Markdown("Denoising steps", elem_classes=["aiwf-section-label"])
                    with gr.Row():
                        high_steps = gr.Slider(1, 30, value=20, step=1, label="Steps")
                        low_steps = gr.Slider(1, 30, value=1, step=1, label="Low noise steps", interactive=False)
                    with gr.Row():
                        total_steps = gr.Number(value=20, precision=0, label="Total steps", interactive=False)
                        boundary_ratio = gr.Number(value=1.0, precision=3, label="Stage split", interactive=False)

                    gr.Markdown("Sampler", elem_classes=["aiwf-section-label"])
                    sampler = gr.Dropdown(
                        label="Sampler",
                        choices=[
                            ("UniPC (recommended - the model's native, calibrated solver)", "unipc"),
                            ("FlowMatch Euler (fast, 1 NFE/step)", "euler"),
                            ("FlowMatch Heun (2nd-order, higher quality, ~2x slower)", "heun"),
                        ],
                        value=_last_sampler,
                        info="UniPC matches Wan2.2-TI2V-5B's shipped scheduler config and is the most "
                             "stable choice. Euler/Heun swap in a different solver family the checkpoint "
                             "wasn't tuned against; if motion looks warped, switch back to UniPC.",
                    )
                    sigma_type = gr.Dropdown(
                        label="Scheduler",
                        choices=[
                            ("Simple - linear uniform spacing (tested 5B default)", "simple"),
                            ("Beta - smooth motion, best quality at low steps", "beta"),
                            ("Exponential - more detail at high noise", "exponential"),
                            ("Karras - SD-style detail preservation", "karras"),
                        ],
                        value="simple",
                        info="Simple is the tested 5B baseline; Beta is the quality check.",
                    )
                    with gr.Row():
                        flow_shift = gr.Slider(
                            0.5, 25.0, value=_last_flow_shift, step=0.5, label="Flow shift",
                            info="5.0 matches the 5B checkpoint's own scheduler config (paired with "
                                 "UniPC). Higher shifts more work to high-noise; only raise this if "
                                 "you've also switched the sampler away from UniPC.",
                        )
                        seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")

                    gr.Markdown("Reference & chunks", elem_classes=["aiwf-section-label"])
                    gr.Markdown(
                        "Leave chunking off unless a long or high-resolution run OOMs. Values are latent frames.",
                        elem_classes=["aiwf-settings-paths"],
                    )
                    temporal_chunks = gr.Checkbox(
                        value=False,
                        label="Enable temporal chunking",
                        info="Each chunk reruns the transformer.",
                    )
                    with gr.Row():
                        chunk_size = gr.Slider(
                            4, 64, value=24, step=4, label="Latent chunk size",
                            info="24 avoids chunking an 81-frame run.",
                        )
                        chunk_overlap = gr.Slider(
                            0, 32, value=0, step=1, label="Latent overlap",
                            info="Higher overlap is smoother but slower.",
                        )
                    image_guidance_scale = gr.Slider(
                        1.0, 5.0, value=1.0, step=0.1, label="Low-noise guidance (CFG)",
                        info="Dual high/low models only: separate CFG for the low-noise stage. "
                             "1.0 = reuse the main guidance scale; raise to sharpen detail late in denoise.",
                        interactive=False,
                    )

                    with gr.Accordion("Post-processing", open=False, elem_classes=["aiwf-prompt-tools"]):
                        rife_enabled = gr.Checkbox(
                            value=False,
                            label="Run RIFE after generation",
                            info="Wan VRAM is unloaded before interpolation starts.",
                        )
                        rife_target_fps = gr.Radio(
                            label="RIFE output FPS",
                            choices=[("30 FPS", 30), ("60 FPS", 60)],
                            value=30,
                            info="AIWF preserves duration and writes the final video at this FPS.",
                        )
                        rife_ckpt = gr.Dropdown(
                            label="RIFE model",
                            choices=rife_ckpts,
                            value=default_rife_ckpt if default_rife_ckpt in rife_ckpts else (rife_ckpts[0] if rife_ckpts else None),
                        )
                        with gr.Row():
                            rife_scale_factor = gr.Dropdown(
                                label="Scale",
                                choices=[("Full resolution", 1.0), ("Half resolution", 0.5)],
                                value=1.0,
                            )
                            rife_clear_cache = gr.Slider(
                                1,
                                100,
                                value=50,
                                step=1,
                                label="Cache clear interval",
                                info="Higher favors throughput when VRAM is free.",
                            )
                        with gr.Row():
                            rife_fast_mode = gr.Checkbox(label="Fast mode", value=False)
                            rife_ensemble = gr.Checkbox(label="Ensemble", value=True)

                        with gr.Accordion("ReActor face swap", open=False, elem_classes=["aiwf-prompt-tools"]):
                            reactor_enabled = gr.Checkbox(
                                value=False,
                                label="Run ReActor after generation",
                                info="Wan/RIFE VRAM is cleared before face swap starts.",
                            )
                            reactor_source_mode = gr.Radio(
                                label="Source face",
                                choices=[
                                    ("First key frame", "first_frame"),
                                    ("Uploaded image", "image"),
                                    ("Saved face model", "face_model"),
                                ],
                                value="first_frame",
                            )
                            reactor_source_image = gr.Image(
                                label="Source face image",
                                type="pil",
                                sources=["upload", "clipboard"],
                                visible=False,
                            )
                            reactor_face_model = gr.Dropdown(
                                label="Saved face model",
                                choices=reactor_face_model_choices,
                                value=reactor_face_model_choices[0][1] if reactor_face_model_choices else None,
                                allow_custom_value=True,
                                visible=False,
                                info="Looks in models/reactor/faces.",
                            )
                            with gr.Row():
                                reactor_source_index = gr.Number(
                                    value=0,
                                    precision=0,
                                    label="Source face #",
                                    info="Used for first-frame or image sources.",
                                )
                                reactor_target_index = gr.Number(
                                    value=-1,
                                    precision=0,
                                    label="Target face #",
                                    info="-1 swaps every detected face.",
                                )
                            with gr.Row():
                                reactor_model = gr.Dropdown(
                                    label="Swapper model",
                                    choices=reactor_swapper_choices,
                                    value=reactor_swapper_choices[0][1] if reactor_swapper_choices else "inswapper_128",
                                    allow_custom_value=True,
                                    info="Install inswapper_128 on the Face Swap tab.",
                                )
                                reactor_mask_face = gr.Checkbox(label="Feather face mask", value=False)
                            with gr.Row():
                                reactor_restore_face = gr.Checkbox(label="Restore face after swap", value=True)
                                reactor_restorer = gr.Dropdown(
                                    label="Restorer",
                                    choices=reactor_restorer_choices,
                                    value=reactor_restorer_choices[0][1] if reactor_restorer_choices else None,
                                )
                            with gr.Row():
                                reactor_restore_visibility = gr.Slider(
                                    0,
                                    1,
                                    value=1.0,
                                    step=0.05,
                                    label="Restore visibility",
                                )
                                reactor_codeformer_weight = gr.Slider(
                                    0,
                                    1,
                                    value=0.5,
                                    step=0.05,
                                    label="CodeFormer weight",
                                )

                        with gr.Accordion("NVIDIA VideoFX", open=False, elem_classes=["aiwf-prompt-tools"]):
                            vsr_enabled = gr.Checkbox(
                                value=False,
                                label="Run upscale or cleanup",
                                info="Uses NVIDIA Video Effects SDK sample runners when installed.",
                            )
                            gr.Markdown(vsr_help, elem_classes=["aiwf-settings-paths"])
                            vsr_effect = gr.Radio(
                                label="Mode family",
                                choices=[
                                    ("RTX VSR upscale", "SuperRes"),
                                    ("Fast SDK upscale", "Upscale"),
                                    ("Same-size cleanup", "Cleanup"),
                                ],
                                value="SuperRes",
                                info="Cleanup forces same-size output for denoise/deblur/high-bitrate modes.",
                            )
                            with gr.Row():
                                vsr_scale = gr.Dropdown(
                                    label="Scale",
                                    choices=[
                                        ("Same size", 1.0),
                                        ("1.5x", 1.5),
                                        ("2x", 2.0),
                                        ("3x", 3.0),
                                        ("4x", 4.0),
                                    ],
                                    value=2.0,
                                )
                                vsr_mode = gr.Radio(
                                    label="Quality / cleanup mode",
                                    choices=VSR_UPSCALE_MODE_CHOICES,
                                    value=3,
                                )
                            vsr_strength = gr.Slider(
                                0,
                                1,
                                value=0.6,
                                step=0.05,
                                label="Fast upscale sharpness",
                                info="Used only by Fast SDK upscale.",
                                interactive=False,
                            )

                            gr.Markdown("Denoise", elem_classes=["aiwf-section-label"])
                            with gr.Row():
                                videofx_denoise_enabled = gr.Checkbox(
                                    value=False,
                                    label="Run dedicated denoise",
                                    info="Runs before AIGS and upscale.",
                                )
                                videofx_denoise_strength = gr.Slider(
                                    0,
                                    1,
                                    value=0.8,
                                    step=0.05,
                                    label="Denoise strength",
                                )

                            gr.Markdown("Background", elem_classes=["aiwf-section-label"])
                            videofx_aigs_enabled = gr.Checkbox(
                                value=False,
                                label="Run AI Green Screen",
                                info="Background blur, matte, green/white background, or background replacement.",
                            )
                            with gr.Row():
                                videofx_aigs_comp = gr.Dropdown(
                                    label="Output",
                                    choices=AIGS_COMP_CHOICES,
                                    value=6,
                                )
                                videofx_aigs_blur = gr.Slider(
                                    0,
                                    1,
                                    value=0.45,
                                    step=0.05,
                                    label="Blur strength",
                                )
                            with gr.Row():
                                videofx_aigs_bg = gr.Image(
                                    label="Background image",
                                    sources=["upload"],
                                    type="filepath",
                                )
                                videofx_aigs_cuda_graph = gr.Checkbox(
                                    value=False,
                                    label="CUDA graph",
                                    info="Experimental SDK optimization for AIGS.",
                                )

                            gr.Markdown("Relight", elem_classes=["aiwf-section-label"])
                            videofx_relight_enabled = gr.Checkbox(
                                value=False,
                                label="Run relighting",
                                info="Uses NVIDIA Video Relighting with an HDR illumination preset.",
                            )
                            with gr.Row():
                                videofx_relight_hdr = gr.Dropdown(
                                    label="HDR preset",
                                    choices=relight_hdr_choices,
                                    value=next((v for label, v in relight_hdr_choices if label.lower() == "default"), None)
                                    or (relight_hdr_choices[0][1] if relight_hdr_choices else None),
                                    allow_custom_value=True,
                                )
                                videofx_relight_bg_mode = gr.Dropdown(
                                    label="Background",
                                    choices=RELIGHT_BG_MODE_CHOICES,
                                    value=0,
                                )
                            with gr.Row():
                                videofx_relight_pan = gr.Slider(
                                    -180,
                                    180,
                                    value=-90,
                                    step=1,
                                    label="Pan",
                                )
                                videofx_relight_vfov = gr.Slider(
                                    10,
                                    140,
                                    value=60,
                                    step=1,
                                    label="Vertical FOV",
                                )
                            with gr.Row():
                                videofx_relight_bg = gr.Image(
                                    label="Relight background image",
                                    sources=["upload"],
                                    type="filepath",
                                )
                                videofx_relight_bg_text = gr.Textbox(
                                    label="Background color",
                                    placeholder="gray or 0x202020",
                                )
                                videofx_relight_autorotate = gr.Checkbox(
                                    value=False,
                                    label="Autorotate HDR",
                                )

                        with gr.Accordion("Audio", open=False, elem_classes=["aiwf-prompt-tools"]):
                            # Team note: this is audio post-processing after all visual effects
                            # have finished. It is not Wan S2V or part of Wan video generation.
                            audio_enabled = gr.Checkbox(
                                value=False,
                                label="Add audio after video",
                                info="Runs after all visual post-processing and writes a muxed MP4.",
                            )
                            gr.Markdown(audio_service.video_audio_status(), elem_classes=["aiwf-settings-paths"])
                            audio_prompt = gr.Textbox(
                                label="Audio prompt",
                                lines=3,
                                placeholder="footsteps, cloth movement, room tone, light cinematic ambience",
                            )
                            with gr.Row():
                                audio_kind = gr.Radio(
                                    label="Type",
                                    choices=[
                                        ("Video-conditioned audio", "video_audio"),
                                        ("Music", "music"),
                                        ("Sound effects", "sfx"),
                                    ],
                                    value="video_audio",
                                )
                                audio_model = gr.Dropdown(
                                    label="Audio model",
                                    choices=audio_video_models,
                                    value=audio_video_models[0][1] if audio_video_models else "mmaudio:large_44k_v2",
                                    allow_custom_value=True,
                                )
                            with gr.Row():
                                audio_duration = gr.Slider(
                                    0,
                                    120,
                                    value=0,
                                    step=1,
                                    label="Duration (seconds)",
                                    info="0 = match final video.",
                                )
                                audio_seed = gr.Number(value=-1, precision=0, label="Audio seed")
                            with gr.Row():
                                audio_temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.05, label="Temperature")
                                audio_cfg = gr.Slider(0.1, 10.0, value=4.5, step=0.1, label="Guidance")

                    with gr.Row():
                        run = gr.Button("Generate video", variant="primary", elem_classes=["aiwf-generate-btn"])
                        stop_btn = gr.Button("Stop", variant="stop", elem_classes=["aiwf-btn-stop"])
                    video_out = gr.Video(label="Result", interactive=False)
                    save_bad_video = gr.Button("Save bad result", elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"])
                    status = gr.Markdown("**Ready** - upload an image and generate.", elem_classes=["aiwf-status-bar"])

                    default_ltx_pipeline = ltx_service.default_launch_pipeline()
                    with gr.Accordion("LTX optional engine", open=False, elem_classes=["aiwf-prompt-tools"]):
                        gr.Markdown(ltx_service.status_markdown(), elem_classes=["aiwf-settings-paths"])
                        ltx_source = gr.Image(
                            label="LTX source image (optional)",
                            type="filepath",
                            sources=["upload", "clipboard"],
                        )
                        ltx_prompt = gr.Textbox(
                            label="LTX prompt",
                            lines=3,
                            placeholder="A simple dance performance, natural movement, stable camera",
                        )
                        ltx_negative = gr.Textbox(label="LTX negative prompt", lines=2, value="")
                        ltx_pipeline = gr.Radio(
                            label="LTX route",
                            choices=[
                                ("Local 2B Diffusers", LTX_PIPELINE_DIFFUSERS_2B),
                                ("Distilled two-stage", LTX_PIPELINE_DISTILLED),
                                ("Full one-stage checkpoint", LTX_PIPELINE_ONE_STAGE),
                            ],
                            value=default_ltx_pipeline,
                        )
                        with gr.Row():
                            ltx_width = gr.Slider(128, 1280, value=128, step=32, label="Width")
                            ltx_height = gr.Slider(128, 1280, value=128, step=32, label="Height")
                            ltx_frames = gr.Slider(
                                9,
                                257,
                                value=9,
                                step=8,
                                label="Frames",
                                info="LTX uses 8*k+1 frame counts.",
                            )
                        with gr.Row():
                            ltx_fps = gr.Slider(1, 60, value=8, step=1, label="FPS")
                            ltx_steps = gr.Slider(
                                1,
                                80,
                                value=1,
                                step=1,
                                label="Steps",
                            )
                            ltx_seed = gr.Number(value=-1, precision=0, label="Seed")
                        with gr.Row():
                            ltx_offload = gr.Radio(
                                label="Offload",
                                choices=[("Disk fallback", "disk"), ("CPU offload", "cpu"), ("None", "none")],
                                value="disk",
                            )
                            ltx_quantization = gr.Radio(
                                label="Quantization",
                                choices=[("FP8 cast", "fp8-cast"), ("None", ""), ("FP8 scaled MM", "fp8-scaled-mm")],
                                value="fp8-cast",
                            )
                            ltx_enhance_prompt = gr.Checkbox(label="Enhance prompt", value=False)
                        ltx_checkpoint = gr.Textbox(
                            label="LTX checkpoint",
                            value=str(ltx_service.default_checkpoint_path(default_ltx_pipeline)),
                            info=(
                                "Distilled route expects ltx-2.3-22b-distilled-1.1.safetensors. "
                                "Full one-stage route expects ltx-2.3-22b-dev-bf16.safetensors "
                                "(dequantized from the nvfp4 release; clear this box to use the "
                                "per-pipeline default)."
                            ),
                        )
                        ltx_upsampler = gr.Textbox(
                            label="LTX spatial upscaler",
                            value=str(ltx_service.default_spatial_upsampler_path()),
                            info="Required for the distilled two-stage route.",
                        )
                        ltx_gemma = gr.Textbox(
                            label="LTX Gemma tokenizer/processor folder",
                            value=str(ltx_service.default_gemma_root()),
                        )
                        ltx_t5_encoder = gr.Textbox(
                            label="LTX 2B T5XXL text encoder",
                            value=str(ltx_service.default_t5_encoder_path()),
                        )
                        ltx_t5_tokenizer = gr.Textbox(
                            label="LTX 2B tokenizer",
                            value=LTX_T5_TOKENIZER,
                        )
                        with gr.Row():
                            ltx_gemma_backend = gr.Radio(
                                label="Gemma backend",
                                choices=[
                                    ("HF safetensors", LTX_GEMMA_BACKEND_HF_SAFETENSORS),
                                    ("Heretic GGUF Q3", LTX_GEMMA_BACKEND_GGUF),
                                ],
                                value=LTX_GEMMA_BACKEND_HF_SAFETENSORS,
                            )
                            ltx_gemma_gguf = gr.Textbox(
                                label="Heretic GGUF",
                                value=str(ltx_service.default_gemma_gguf_path()),
                            )
                        ltx_run = gr.Button("Generate LTX video", variant="primary")
                        ltx_video_out = gr.Video(label="LTX result", interactive=False)
                        ltx_save_bad_video = gr.Button(
                            "Save bad LTX result",
                            elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"],
                        )
                        ltx_status = gr.Markdown(
                            "**LTX ready** - Local 2B Diffusers uses the app venv; LTX-2.3 uses the optional engine.",
                            elem_classes=["aiwf-status-bar"],
                        )

        def _active_resolution_ratio(ratio_value, square_ratio_value):
            return square_ratio_value or ratio_value or "1:1"

        def _apply_resolution_preset(size_value, ratio_value, square_ratio_value):
            next_width, next_height = dimensions_from_generation_preset(
                size_value,
                _active_resolution_ratio(ratio_value, square_ratio_value),
            )
            return gr.update(value=next_width), gr.update(value=next_height)

        def _apply_main_resolution_ratio(size_value, ratio_value):
            next_width, next_height = dimensions_from_generation_preset(size_value, ratio_value or "1:1")
            return gr.update(value=next_width), gr.update(value=next_height), gr.update(value=None)

        def _apply_square_resolution_ratio(size_value, square_ratio_value):
            next_width, next_height = dimensions_from_generation_preset(size_value, square_ratio_value or "1:1")
            return gr.update(value=next_width), gr.update(value=next_height), gr.update(value=None)

        def _ltx_route_defaults(route_value):
            route = str(route_value or default_ltx_pipeline)
            checkpoint = ltx_service.default_checkpoint_path(route)
            quantization = "" if route == LTX_PIPELINE_DIFFUSERS_2B else "fp8-cast"
            return (
                gr.update(value=str(checkpoint)),
                gr.update(value=str(ltx_service.default_spatial_upsampler_path())),
                gr.update(value=str(ltx_service.default_t5_encoder_path())),
                gr.update(value=LTX_T5_TOKENIZER),
                gr.update(value=LTX_GEMMA_BACKEND_HF_SAFETENSORS),
                gr.update(value=128),
                gr.update(value=128),
                gr.update(value=1),
                gr.update(value="disk"),
                gr.update(value=quantization),
            )

        resolution_size.change(
            _apply_resolution_preset,
            inputs=[resolution_size, resolution_ratio, resolution_ratio_square],
            outputs=[width, height],
            show_progress=False,
        )
        ltx_pipeline.change(
            _ltx_route_defaults,
            inputs=[ltx_pipeline],
            outputs=[
                ltx_checkpoint,
                ltx_upsampler,
                ltx_t5_encoder,
                ltx_t5_tokenizer,
                ltx_gemma_backend,
                ltx_width,
                ltx_height,
                ltx_steps,
                ltx_offload,
                ltx_quantization,
            ],
            show_progress=False,
        )
        resolution_ratio.change(
            _apply_main_resolution_ratio,
            inputs=[resolution_size, resolution_ratio],
            outputs=[width, height, resolution_ratio_square],
            show_progress=False,
        )
        resolution_ratio_square.change(
            _apply_square_resolution_ratio,
            inputs=[resolution_size, resolution_ratio_square],
            outputs=[width, height, resolution_ratio],
            show_progress=False,
        )

        def _sync_duration(fps_value, duration_value):
            frames = frames_for_duration_seconds(int(fps_value or 16), float(duration_value or 3))
            snapped = duration_seconds_for_frames(frames, int(fps_value or 16))
            return (
                gr.update(value=frames),
                gr.update(value=f"**Frames:** {frames} - **Duration:** {snapped:.1f}s snapped for Wan"),
            )

        fps.change(
            _sync_duration,
            inputs=[fps, duration_seconds],
            outputs=[num_frames, frame_summary],
            show_progress=False,
        )
        duration_seconds.change(
            _sync_duration,
            inputs=[fps, duration_seconds],
            outputs=[num_frames, frame_summary],
            show_progress=False,
        )

        def _sync_step_split(high_value, low_value, runtime_value):
            total, ratio = _step_summary_for_runtime(runtime_value, high_value, low_value)
            return gr.update(value=total), gr.update(value=ratio)

        high_steps.change(
            _sync_step_split,
            inputs=[high_steps, low_steps, runtime_mode],
            outputs=[total_steps, boundary_ratio],
            show_progress=False,
        )
        low_steps.change(
            _sync_step_split,
            inputs=[high_steps, low_steps, runtime_mode],
            outputs=[total_steps, boundary_ratio],
            show_progress=False,
        )

        def _route_status_from_values(
            runtime_value,
            high_value,
            low_value,
            vae_value,
            text_encoder_value,
            offload_value,
            width_value,
            height_value,
            frames_value,
            high_steps_value,
            low_steps_value,
            temporal_chunks_value,
            chunk_size_value,
            chunk_overlap_value,
            low_guidance_value,
            sampler_value="unipc",
            flow_shift_value=5.0,
        ):
            return _runtime_trace_status(
                runtime_value,
                high_value,
                low_value,
                vae_value,
                text_encoder_value,
                offload_value,
                width_value,
                height_value,
                frames_value,
                high_steps_value,
                low_steps_value,
                temporal_chunks_value,
                chunk_size_value,
                chunk_overlap_value,
                low_guidance_value,
                sampler_value,
                flow_shift_value,
            )

        def _sync_runtime_choices(
            runtime_value,
            previous_runtime_value,
            high_value,
            low_value,
            vae_value,
            text_encoder_value,
            offload_value,
            high_steps_value,
            low_steps_value,
            width_value,
            height_value,
            frames_value,
            temporal_chunks_value,
            chunk_size_value,
            chunk_overlap_value,
            low_guidance_value,
            sampler_value,
            flow_shift_value,
        ):
            selected_runtime = str(runtime_value or WAN_RUNTIME_FAST_5B)
            previous_runtime = str(previous_runtime_value or WAN_RUNTIME_FAST_5B)
            lora_choices = _filter_lora_choices(selected_runtime)
            offload_choices = _offload_choices_for_runtime(selected_runtime)
            next_offload = _default_offload_for_runtime(selected_runtime, offload_value)
            if selected_runtime == WAN_RUNTIME_FAST_5B:
                if previous_runtime == WAN_RUNTIME_FAST_5B:
                    single_steps = max(1, int(high_steps_value or 0))
                else:
                    single_steps = max(1, int(high_steps_value or 0)) + max(1, int(low_steps_value or 0))
                total, ratio = single_steps, 1.0
                model_choices = _filter_stage_choices(
                    all_labeled,
                    runtime_value=selected_runtime,
                    stage=None,
                    peer_value=None,
                )
                next_model = _valid_or_first(high_value, model_choices)
                next_vae = _preferred_vae_for_runtime(selected_runtime, vae_value)
                trace_status = _route_status_from_values(
                    selected_runtime,
                    next_model,
                    None,
                    next_vae,
                    text_encoder_value,
                    next_offload,
                    width_value,
                    height_value,
                    frames_value,
                    single_steps,
                    low_steps_value,
                    temporal_chunks_value,
                    chunk_size_value,
                    chunk_overlap_value,
                    1.0,
                    sampler_value,
                    flow_shift_value,
                )
                return (
                    gr.update(label="5B transformer", choices=model_choices, value=next_model, interactive=True),
                    gr.update(label="Low noise transformer", choices=[], value=None, interactive=False),
                    "",
                    gr.update(value=next_vae),
                    gr.update(choices=[], value=None, interactive=False),
                    gr.update(choices=[], value=None, interactive=False),
                    gr.update(value=1.0, interactive=False),
                    gr.update(value=1.0, interactive=False),
                    gr.update(choices=offload_choices, value=next_offload),
                    gr.update(label="Steps", value=single_steps, interactive=True),
                    gr.update(label="Low noise steps", interactive=False),
                    gr.update(value=total),
                    gr.update(value=ratio),
                    gr.update(value=1.0, interactive=False),
                    trace_status,
                    selected_runtime,
                )
            if previous_runtime == WAN_RUNTIME_FAST_5B:
                next_high_steps, next_low_steps = _dual_step_split_from_total(high_steps_value)
            else:
                next_high_steps = max(1, int(high_steps_value or 0))
                next_low_steps = max(1, int(low_steps_value or 0))
            high_choices = _filter_stage_choices(
                high_labeled,
                runtime_value=selected_runtime,
                stage="high",
                peer_value=None,
            )
            next_high = _valid_or_first(high_value, high_choices)
            low_choices = _filter_stage_choices(
                low_labeled,
                runtime_value=selected_runtime,
                stage="low",
                peer_value=next_high,
            )
            next_low = _valid_or_first(low_value, low_choices)
            next_vae = _preferred_vae_for_runtime(selected_runtime, vae_value)
            total, ratio = _step_summary_for_runtime(selected_runtime, next_high_steps, next_low_steps)
            trace_status = _route_status_from_values(
                selected_runtime,
                next_high,
                next_low,
                next_vae,
                text_encoder_value,
                next_offload,
                width_value,
                height_value,
                frames_value,
                next_high_steps,
                next_low_steps,
                temporal_chunks_value,
                chunk_size_value,
                chunk_overlap_value,
                low_guidance_value,
                sampler_value,
                flow_shift_value,
            )
            return (
                gr.update(label="High noise transformer", choices=high_choices, value=next_high, interactive=True),
                gr.update(label="Low noise transformer", choices=low_choices, value=next_low, interactive=True),
                _pair_status(next_high, next_low),
                gr.update(value=next_vae),
                gr.update(choices=lora_choices, interactive=True),
                gr.update(choices=lora_choices, interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(choices=offload_choices, value=next_offload),
                gr.update(label="High noise steps", value=next_high_steps, interactive=True),
                gr.update(label="Low noise steps", value=next_low_steps, interactive=True),
                gr.update(value=total),
                gr.update(value=ratio),
                gr.update(interactive=True),
                trace_status,
                selected_runtime,
            )

        def _sync_low_choices(
            high_value,
            low_value,
            runtime_value,
            vae_value,
            text_encoder_value,
            offload_value,
            width_value,
            height_value,
            frames_value,
            high_steps_value,
            low_steps_value,
            temporal_chunks_value,
            chunk_size_value,
            chunk_overlap_value,
            low_guidance_value,
            sampler_value,
            flow_shift_value,
        ):
            selected_runtime = str(runtime_value or WAN_RUNTIME_FAST_5B)
            if selected_runtime == WAN_RUNTIME_FAST_5B:
                status_text = _route_status_from_values(
                    selected_runtime,
                    high_value,
                    None,
                    vae_value,
                    text_encoder_value,
                    offload_value,
                    width_value,
                    height_value,
                    frames_value,
                    high_steps_value,
                    low_steps_value,
                    temporal_chunks_value,
                    chunk_size_value,
                    chunk_overlap_value,
                    low_guidance_value,
                    sampler_value,
                    flow_shift_value,
                )
                return gr.update(choices=[], value=None, interactive=False), "", status_text
            choices = _filter_stage_choices(
                low_labeled,
                runtime_value=selected_runtime,
                stage="low",
                peer_value=high_value,
            )
            next_low = _valid_or_first(low_value, choices)
            status_text = _route_status_from_values(
                selected_runtime,
                high_value,
                next_low,
                vae_value,
                text_encoder_value,
                offload_value,
                width_value,
                height_value,
                frames_value,
                high_steps_value,
                low_steps_value,
                temporal_chunks_value,
                chunk_size_value,
                chunk_overlap_value,
                low_guidance_value,
                sampler_value,
                flow_shift_value,
            )
            return gr.update(choices=choices, value=next_low, interactive=True), _pair_status(high_value, next_low), status_text

        def _sync_high_choices(
            low_value,
            high_value,
            runtime_value,
            vae_value,
            text_encoder_value,
            offload_value,
            width_value,
            height_value,
            frames_value,
            high_steps_value,
            low_steps_value,
            temporal_chunks_value,
            chunk_size_value,
            chunk_overlap_value,
            low_guidance_value,
            sampler_value,
            flow_shift_value,
        ):
            selected_runtime = str(runtime_value or WAN_RUNTIME_FAST_5B)
            if selected_runtime == WAN_RUNTIME_FAST_5B:
                status_text = _route_status_from_values(
                    selected_runtime,
                    high_value,
                    None,
                    vae_value,
                    text_encoder_value,
                    offload_value,
                    width_value,
                    height_value,
                    frames_value,
                    high_steps_value,
                    low_steps_value,
                    temporal_chunks_value,
                    chunk_size_value,
                    chunk_overlap_value,
                    low_guidance_value,
                    sampler_value,
                    flow_shift_value,
                )
                return gr.update(choices=[], value=None, interactive=False), "", status_text
            choices = _filter_stage_choices(
                high_labeled,
                runtime_value=selected_runtime,
                stage="high",
                peer_value=low_value,
            )
            next_high = _valid_or_first(high_value, choices)
            status_text = _route_status_from_values(
                selected_runtime,
                next_high,
                low_value,
                vae_value,
                text_encoder_value,
                offload_value,
                width_value,
                height_value,
                frames_value,
                high_steps_value,
                low_steps_value,
                temporal_chunks_value,
                chunk_size_value,
                chunk_overlap_value,
                low_guidance_value,
                sampler_value,
                flow_shift_value,
            )
            return gr.update(choices=choices, value=next_high, interactive=True), _pair_status(next_high, low_value), status_text

        runtime_mode.change(
            _sync_runtime_choices,
            inputs=[
                runtime_mode,
                runtime_previous,
                high_noise,
                low_noise,
                vae_id,
                text_encoder,
                offload,
                high_steps,
                low_steps,
                width,
                height,
                num_frames,
                temporal_chunks,
                chunk_size,
                chunk_overlap,
                image_guidance_scale,
                sampler,
                flow_shift,
            ],
            outputs=[
                high_noise,
                low_noise,
                model_pair_status,
                vae_id,
                high_lora,
                low_lora,
                high_lora_scale,
                low_lora_scale,
                offload,
                high_steps,
                low_steps,
                total_steps,
                boundary_ratio,
                image_guidance_scale,
                route_status,
                runtime_previous,
            ],
            show_progress=False,
        )
        high_noise.change(
            _sync_low_choices,
            inputs=[
                high_noise,
                low_noise,
                runtime_mode,
                vae_id,
                text_encoder,
                offload,
                width,
                height,
                num_frames,
                high_steps,
                low_steps,
                temporal_chunks,
                chunk_size,
                chunk_overlap,
                image_guidance_scale,
                sampler,
                flow_shift,
            ],
            outputs=[low_noise, model_pair_status, route_status],
            show_progress=False,
        )
        low_noise.change(
            _sync_high_choices,
            inputs=[
                low_noise,
                high_noise,
                runtime_mode,
                vae_id,
                text_encoder,
                offload,
                width,
                height,
                num_frames,
                high_steps,
                low_steps,
                temporal_chunks,
                chunk_size,
                chunk_overlap,
                image_guidance_scale,
                sampler,
                flow_shift,
            ],
            outputs=[high_noise, model_pair_status, route_status],
            show_progress=False,
        )

        def _sync_route_status(
            runtime_value,
            high_value,
            low_value,
            vae_value,
            text_encoder_value,
            offload_value,
            width_value,
            height_value,
            frames_value,
            high_steps_value,
            low_steps_value,
            temporal_chunks_value,
            chunk_size_value,
            chunk_overlap_value,
            low_guidance_value,
            sampler_value,
            flow_shift_value,
        ):
            return _route_status_from_values(
                runtime_value,
                high_value,
                low_value,
                vae_value,
                text_encoder_value,
                offload_value,
                width_value,
                height_value,
                frames_value,
                high_steps_value,
                low_steps_value,
                temporal_chunks_value,
                chunk_size_value,
                chunk_overlap_value,
                low_guidance_value,
                sampler_value,
                flow_shift_value,
            )

        route_trace_inputs = [
            runtime_mode,
            high_noise,
            low_noise,
            vae_id,
            text_encoder,
            offload,
            width,
            height,
            num_frames,
            high_steps,
            low_steps,
            temporal_chunks,
            chunk_size,
            chunk_overlap,
            image_guidance_scale,
            sampler,
            flow_shift,
        ]
        for route_input in [
            vae_id,
            text_encoder,
            offload,
            width,
            height,
            num_frames,
            high_steps,
            low_steps,
            temporal_chunks,
            chunk_size,
            chunk_overlap,
            image_guidance_scale,
            sampler,
            flow_shift,
        ]:
            route_input.change(
                _sync_route_status,
                inputs=route_trace_inputs,
                outputs=[route_status],
                show_progress=False,
            )

        def _sync_reactor_source_mode(mode_value):
            mode = str(mode_value or "first_frame")
            return (
                gr.update(visible=mode == "image"),
                gr.update(visible=mode == "face_model"),
                gr.update(visible=mode != "face_model"),
            )

        reactor_source_mode.change(
            _sync_reactor_source_mode,
            inputs=[reactor_source_mode],
            outputs=[reactor_source_image, reactor_face_model, reactor_source_index],
            show_progress=False,
        )

        def _sync_audio_kind(kind_value):
            selected = str(kind_value or "video_audio")
            if selected == "sfx":
                choices = audio_sfx_models
                fallback = "facebook/audiogen-medium"
                cfg_value = 3.0
            elif selected == "music":
                choices = audio_music_models
                fallback = "facebook/musicgen-small"
                cfg_value = 3.0
            else:
                choices = audio_video_models
                fallback = "mmaudio:large_44k_v2"
                cfg_value = 4.5
            return (
                gr.update(choices=choices, value=choices[0][1] if choices else fallback),
                gr.update(value=cfg_value),
            )

        audio_kind.change(
            _sync_audio_kind,
            inputs=[audio_kind],
            outputs=[audio_model, audio_cfg],
            show_progress=False,
        )

        def _sync_videofx_effect(effect_value):
            effect = str(effect_value or "SuperRes")
            if effect == "Cleanup":
                return (
                    gr.update(choices=VSR_CLEANUP_MODE_CHOICES, value=10, interactive=True),
                    gr.update(value=1.0, interactive=False),
                    gr.update(interactive=False),
                )
            if effect == "Upscale":
                return (
                    gr.update(choices=VSR_UPSCALE_MODE_CHOICES, value=3, interactive=False),
                    gr.update(value=2.0, interactive=True),
                    gr.update(interactive=True),
                )
            return (
                gr.update(choices=VSR_UPSCALE_MODE_CHOICES, value=3, interactive=True),
                gr.update(value=2.0, interactive=True),
                gr.update(interactive=False),
            )

        vsr_effect.change(
            _sync_videofx_effect,
            inputs=[vsr_effect],
            outputs=[vsr_mode, vsr_scale, vsr_strength],
            show_progress=False,
        )

        def _run(
            image,
            prompt_v,
            negative_v,
            offload_v,
            vram_reserve_enabled_v,
            vram_reserve_mb_v,
            width_v,
            height_v,
            frames_v,
            fps_v,
            high_steps_v,
            low_steps_v,
            guidance_v,
            sampler_v,
            sigma_type_v,
            flow_v,
            seed_v,
            runtime_mode_v,
            high_v,
            low_v,
            vae_v,
            text_encoder_v,
            high_lora_v,
            high_lora_scale_v,
            low_lora_v,
            low_lora_scale_v,
            chunk_size_v,
            chunk_overlap_v,
            temporal_chunks_v,
            image_guidance_scale_v,
            rife_enabled_v,
            rife_target_fps_v,
            rife_ckpt_v,
            rife_scale_factor_v,
            rife_clear_cache_v,
            rife_fast_mode_v,
            rife_ensemble_v,
            reactor_enabled_v,
            reactor_source_mode_v,
            reactor_source_image_v,
            reactor_face_model_v,
            reactor_source_index_v,
            reactor_target_index_v,
            reactor_model_v,
            reactor_mask_face_v,
            reactor_restore_face_v,
            reactor_restorer_v,
            reactor_restore_visibility_v,
            reactor_codeformer_weight_v,
            vsr_enabled_v,
            vsr_scale_v,
            vsr_mode_v,
            vsr_effect_v,
            vsr_strength_v,
            videofx_denoise_enabled_v,
            videofx_denoise_strength_v,
            videofx_aigs_enabled_v,
            videofx_aigs_comp_v,
            videofx_aigs_blur_v,
            videofx_aigs_bg_v,
            videofx_aigs_cuda_graph_v,
            videofx_relight_enabled_v,
            videofx_relight_hdr_v,
            videofx_relight_bg_mode_v,
            videofx_relight_pan_v,
            videofx_relight_vfov_v,
            videofx_relight_bg_v,
            videofx_relight_bg_text_v,
            videofx_relight_autorotate_v,
            audio_enabled_v,
            audio_prompt_v,
            audio_kind_v,
            audio_model_v,
            audio_duration_v,
            audio_seed_v,
            audio_temperature_v,
            audio_cfg_v,
            progress=gr.Progress(),
        ):
            _wan_cancel_flag[0] = False
            if image is None:
                raise gr.Error("Upload a source image first.")
            if not service.available():
                raise gr.Error(
                    "Wan video is unavailable - update diffusers (>=0.35) and install ftfy, then restart."
                )
            selected_runtime = str(runtime_mode_v or WAN_RUNTIME_FAST_5B)
            requires_dual_runtime = selected_runtime != WAN_RUNTIME_FAST_5B
            if selected_runtime == WAN_RUNTIME_FAST_5B and not high_v:
                raise gr.Error("Select a 5B safetensors transformer.")
            if selected_runtime != WAN_RUNTIME_FAST_5B and not (high_v and low_v):
                raise gr.Error(
                    "Select BOTH a High noise model and a Low noise model. Wan 2.2 image-to-video "
                    "high/low modes run a two-stage transformer pair."
                )
            if selected_runtime == WAN_RUNTIME_FAST_5B and wan_model_storage_family(high_v) != "safetensors":
                raise gr.Error("The 5B route only accepts a `.safetensors` transformer file.")
            if selected_runtime == WAN_RUNTIME_HIGH_LOW:
                if wan_model_storage_family(high_v) != "gguf" or wan_model_storage_family(low_v) != "gguf":
                    raise gr.Error("The GGUF route only accepts matched `.gguf` high/low files.")
            if selected_runtime == WAN_RUNTIME_HIGH_LOW_FP8:
                if wan_model_storage_family(high_v) != "safetensors" or wan_model_storage_family(low_v) != "safetensors":
                    raise gr.Error("The full 14B FP8 route only accepts matched `.safetensors` high/low files.")
                if str(offload_v or "").strip() != "streamed":
                    raise gr.Error("The full 14B FP8 route is locked to streamed group offload.")
            if selected_runtime == WAN_RUNTIME_FAST_5B and _vae_looks_21(vae_v) and not _vae_looks_22(vae_v):
                raise gr.Error("Fast 5B uses the Wan 2.2 VAE. Select `wan2.2_vae.safetensors`.")
            if selected_runtime != WAN_RUNTIME_FAST_5B and _vae_looks_22(vae_v) and not _vae_looks_21(vae_v):
                raise gr.Error("High/low Wan routes use the Wan 2.1 VAE. Select `wan2.1_vae.safetensors`.")
            if (
                selected_runtime != WAN_RUNTIME_FAST_5B
                and str(offload_v or "").strip() == "balanced"
                and int(width_v) >= 768
                and int(height_v) >= 768
                and int(frames_v) >= 81
            ):
                raise gr.Error(
                    "High/low 768x768 / 81 frames OOM'd on Balanced in local testing. "
                    "Use Low VRAM/model offload, or reduce size/frames."
                )
            # Guard stale settings or non-UI values; service preflight repeats this check.
            _te_path = str(text_encoder_v or "").strip()
            if _te_path and ("t5xxl" in _te_path.lower()) and not any(k in _te_path.lower() for k in ("umt5", "nsfw_wan")):
                raise gr.Error(
                    f"Warning: '{_te_path}' looks like a T5-XXL file (Flux/SD3). "
                    "T5-XXL is NOT compatible with Wan - it will produce garbage output. "
                    "Select 'Default' or a UMT5-XXL file (umt5-xxl-*.gguf or umt5/nsfw_wan_*.safetensors)."
                )

            request = wan_controller.build_request(
                prompt=prompt_v,
                negative=negative_v,
                width=width_v,
                height=height_v,
                frames=frames_v,
                fps=fps_v,
                high_steps=high_steps_v,
                low_steps=low_steps_v,
                guidance=guidance_v,
                sampler=sampler_v,
                sigma_type=sigma_type_v,
                flow=flow_v,
                seed=seed_v,
                runtime_mode=selected_runtime,
                high=high_v,
                low=low_v,
                vae=vae_v,
                text_encoder=_te_path,
                high_lora=high_lora_v,
                high_lora_scale=high_lora_scale_v,
                low_lora=low_lora_v,
                low_lora_scale=low_lora_scale_v,
                offload=offload_v,
                vram_reserve_enabled=vram_reserve_enabled_v,
                vram_reserve_mb=vram_reserve_mb_v,
                temporal_chunks=temporal_chunks_v,
                chunk_size=chunk_size_v,
                chunk_overlap=chunk_overlap_v,
                image_guidance_scale=image_guidance_scale_v,
            )

            sampler_audit = audit_wan_sampler_settings(request)
            if sampler_audit.errors:
                raise gr.Error("\n".join(sampler_audit.errors))
            if sampler_audit.corrections:
                request = sampler_audit.request

            if selected_runtime == WAN_RUNTIME_FAST_5B:
                runtime_label = "Wan 5B safetensors"
            elif selected_runtime == WAN_RUNTIME_HIGH_LOW_FP8:
                runtime_label = "Wan 14B FP8 safetensors"
            else:
                runtime_label = "Wan GGUF high/low"
            progress(0.02, desc=f"Preparing {runtime_label}: loading models and encoding inputs")

            def _should_cancel() -> bool:
                return _wan_cancel_flag[0]

            result = wan_controller.generate(request, image, progress, should_cancel=_should_cancel)
            wan_controller.persist_last_used(
                high=high_v,
                low=low_v,
                vae=vae_v,
                text_encoder=text_encoder_v,
                offload=offload_v,
                sampler=str(request.sampler or "unipc"),
                flow_shift=float(request.flow_shift),
                runtime_mode=selected_runtime,
            )

            final_video_path = _existing_video_output_path(result.output_path, "Wan")
            status_parts = [f"**Done** -- {result.message}"]
            if bool(rife_enabled_v):
                target_fps = int(rife_target_fps_v or 30)
                input_fps = int(getattr(result, "fps", None) or fps_v or 16)
                multiplier = _rife_multiplier_for_target(input_fps, target_fps)
                progress(0.0, desc="Unloading Wan VRAM before RIFE")
                wan_controller.release_memory_before_postprocess("RIFE")
                rife_options = RifeOptions(
                    ckpt_name=str(rife_ckpt_v or rife_service.default_checkpoint()),
                    multiplier=multiplier,
                    scale_factor=float(rife_scale_factor_v or 1.0),
                    fast_mode=bool(rife_fast_mode_v),
                    ensemble=bool(rife_ensemble_v),
                    clear_cache_every_n_frames=int(rife_clear_cache_v or 50),
                    target_fps=float(target_fps),
                )

                def on_rife_progress(step, total):
                    progress(
                        min(1.0, step / max(1, total)),
                        desc=f"RIFE {target_fps} FPS {step}/{total}",
                    )

                try:
                    progress(0.0, desc=f"Running RIFE x{multiplier} -> {target_fps} FPS")
                    rife_result = rife_service.interpolate(
                        final_video_path,
                        rife_options,
                        on_progress=on_rife_progress,
                    )
                    final_video_path = _existing_video_output_path(rife_result.output_path, "RIFE")
                    status_parts.append(f"**RIFE** -- {rife_result.message}")
                except RifeUnavailable as exc:
                    logger.warning("RIFE post-processing unavailable: %s", exc)
                    status_parts.append(f"**RIFE skipped** -- {exc}")
                except Exception as exc:
                    logger.exception("RIFE post-processing failed")
                    status_parts.append(f"**RIFE failed** -- {exc}")

            if bool(reactor_enabled_v):
                reactor_options = FaceSwapOptions(
                    source_face_index=max(0, int(reactor_source_index_v or 0)),
                    target_face_index=int(reactor_target_index_v if reactor_target_index_v is not None else -1),
                    source_faces_index=[max(0, int(reactor_source_index_v or 0))],
                    target_faces_index=[],
                    model_id=str(reactor_model_v or "inswapper_128"),
                    restore_face=bool(reactor_restore_face_v),
                    restorer_id=reactor_restorer_v or None,
                    restore_visibility=float(reactor_restore_visibility_v or 1.0),
                    codeformer_weight=float(reactor_codeformer_weight_v or 0.5),
                    mask_face=bool(reactor_mask_face_v),
                )

                restore_fn = None
                if bool(reactor_restore_face_v) and reactor_restorer_v:

                    def restore_fn(frame):
                        return ctx.enhance.restore(
                            frame,
                            RestoreOptions(
                                model_id=str(reactor_restorer_v),
                                visibility=float(reactor_restore_visibility_v or 1.0),
                                codeformer_weight=float(reactor_codeformer_weight_v or 0.5),
                            ),
                        )

                def on_reactor_progress(step, total):
                    progress(
                        min(1.0, step / max(1, total)),
                        desc=f"ReActor {step}/{total}",
                    )

                try:
                    progress(0.0, desc="Unloading VRAM before ReActor")
                    wan_controller.release_memory_before_reactor()
                    mode = str(reactor_source_mode_v or "first_frame")
                    if mode == "face_model":
                        if not reactor_face_model_v:
                            raise FaceSwapUnavailable("Select a saved ReActor face model.")
                        progress(0.0, desc="Running ReActor from saved face model")
                        reactor_result = ctx.faceswap.swap_video_with_face_model(
                            final_video_path,
                            str(reactor_face_model_v),
                            reactor_options,
                            restore_fn=restore_fn,
                            on_progress=on_reactor_progress,
                        )
                    else:
                        if mode == "image":
                            source_face_image = reactor_source_image_v
                            if source_face_image is None:
                                raise FaceSwapUnavailable("Upload a source face image for ReActor.")
                        else:
                            progress(0.0, desc="Extracting first key frame for ReActor")
                            source_face_image = extract_first_frame(final_video_path)
                        progress(0.0, desc="Running ReActor face swap")
                        reactor_result = ctx.faceswap.swap_video(
                            final_video_path,
                            source_face_image,
                            reactor_options,
                            restore_fn=restore_fn,
                            on_progress=on_reactor_progress,
                        )
                    final_video_path = _existing_video_output_path(reactor_result.output_path, "ReActor")
                    status_parts.append(f"**ReActor** -- {reactor_result.message}")
                except (FaceSwapUnavailable, VideoError) as exc:
                    logger.warning("ReActor post-processing unavailable: %s", exc)
                    status_parts.append(f"**ReActor skipped** -- {exc}")
                except Exception as exc:
                    logger.exception("ReActor post-processing failed")
                    status_parts.append(f"**ReActor failed** -- {exc}")

            if bool(videofx_denoise_enabled_v):
                try:
                    progress(0.0, desc="Unloading VRAM before NVIDIA VideoFX Denoise")
                    wan_controller.release_memory_before_postprocess("NVIDIA VideoFX Denoise")
                    denoise_options = VideoFxDenoiseOptions(
                        strength=float(videofx_denoise_strength_v or 0.8),
                    )
                    progress(0.0, desc="Running NVIDIA VideoFX Denoise")
                    denoise_result = vsr_service.denoise(final_video_path, denoise_options)
                    final_video_path = _existing_video_output_path(denoise_result.output_path, "NVIDIA Denoise")
                    status_parts.append(f"**NVIDIA Denoise** -- {denoise_result.message}")
                except VsrUnavailable as exc:
                    logger.warning("NVIDIA VideoFX Denoise unavailable: %s", exc)
                    status_parts.append(f"**NVIDIA Denoise skipped** -- {exc}")
                except Exception as exc:
                    logger.exception("NVIDIA VideoFX Denoise failed")
                    status_parts.append(f"**NVIDIA Denoise failed** -- {exc}")

            if bool(videofx_aigs_enabled_v):
                try:
                    progress(0.0, desc="Unloading VRAM before NVIDIA AI Green Screen")
                    wan_controller.release_memory_before_postprocess("NVIDIA AI Green Screen")
                    comp_mode = int(videofx_aigs_comp_v or 6)
                    bg_path = wan_controller.uploaded_file_path(videofx_aigs_bg_v)
                    if comp_mode == 5 and not bg_path:
                        raise VsrUnavailable("Upload a background image or choose a different AI Green Screen output.")
                    aigs_options = VideoFxAigsOptions(
                        comp_mode=comp_mode,
                        blur_strength=float(videofx_aigs_blur_v or 0.45),
                        background_file=bg_path,
                        cuda_graph=bool(videofx_aigs_cuda_graph_v),
                    )
                    progress(0.0, desc="Running NVIDIA AI Green Screen")
                    aigs_result = vsr_service.aigs(final_video_path, aigs_options)
                    final_video_path = _existing_video_output_path(aigs_result.output_path, "NVIDIA AI Green Screen")
                    status_parts.append(f"**NVIDIA AI Green Screen** -- {aigs_result.message}")
                except VsrUnavailable as exc:
                    logger.warning("NVIDIA AI Green Screen unavailable: %s", exc)
                    status_parts.append(f"**NVIDIA AI Green Screen skipped** -- {exc}")
                except Exception as exc:
                    logger.exception("NVIDIA AI Green Screen failed")
                    status_parts.append(f"**NVIDIA AI Green Screen failed** -- {exc}")

            if bool(videofx_relight_enabled_v):
                try:
                    progress(0.0, desc="Unloading VRAM before NVIDIA Relighting")
                    wan_controller.release_memory_before_postprocess("NVIDIA Relighting")
                    bg_mode = int(videofx_relight_bg_mode_v or 0)
                    bg_path = wan_controller.uploaded_file_path(videofx_relight_bg_v)
                    bg_value = bg_path or str(videofx_relight_bg_text_v or "").strip() or None
                    if bg_mode in {3, 4} and not bg_value:
                        raise VsrUnavailable("Upload a background image or enter a background color for relighting.")
                    relight_options = VideoFxRelightOptions(
                        hdr_file=str(videofx_relight_hdr_v or ""),
                        background_mode=bg_mode,
                        background=bg_value,
                        pan_degrees=float(videofx_relight_pan_v or -90),
                        vfov_degrees=float(videofx_relight_vfov_v or 60),
                        autorotate=bool(videofx_relight_autorotate_v),
                    )
                    progress(0.0, desc="Running NVIDIA Relighting")
                    relight_result = vsr_service.relight(final_video_path, relight_options)
                    final_video_path = _existing_video_output_path(relight_result.output_path, "NVIDIA Relighting")
                    status_parts.append(f"**NVIDIA Relighting** -- {relight_result.message}")
                except VsrUnavailable as exc:
                    logger.warning("NVIDIA Relighting unavailable: %s", exc)
                    status_parts.append(f"**NVIDIA Relighting skipped** -- {exc}")
                except Exception as exc:
                    logger.exception("NVIDIA Relighting failed")
                    status_parts.append(f"**NVIDIA Relighting failed** -- {exc}")

            if bool(vsr_enabled_v):
                try:
                    progress(0.0, desc="Unloading VRAM before NVIDIA VideoFX upscale/cleanup")
                    wan_controller.release_memory_before_postprocess("NVIDIA VideoFX")
                    vsr_options = VsrOptions(
                        effect=str(vsr_effect_v or "SuperRes"),
                        scale=float(vsr_scale_v or 2.0),
                        mode=int(vsr_mode_v if vsr_mode_v is not None else 1),
                        strength=float(vsr_strength_v or 0.6),
                    )
                    progress(0.0, desc="Running NVIDIA VideoFX upscale/cleanup")
                    vsr_result = vsr_service.upscale(final_video_path, vsr_options)
                    final_video_path = _existing_video_output_path(vsr_result.output_path, "NVIDIA VideoFX")
                    status_parts.append(f"**NVIDIA VideoFX** -- {vsr_result.message}")
                except VsrUnavailable as exc:
                    logger.warning("NVIDIA VideoFX post-processing unavailable: %s", exc)
                    status_parts.append(f"**NVIDIA VideoFX skipped** -- {exc}")
                except Exception as exc:
                    logger.exception("NVIDIA VideoFX post-processing failed")
                    status_parts.append(f"**NVIDIA VideoFX failed** -- {exc}")

            if bool(audio_enabled_v):
                try:
                    progress(0.0, desc="Unloading VRAM before audio generation")
                    wan_controller.release_memory_before_postprocess("Audio")
                    audio_text = str(audio_prompt_v or prompt_v or "").strip()
                    if not audio_text:
                        audio_text = "cinematic ambient soundtrack matching the video"
                    duration_value = float(audio_duration_v or 0)
                    audio_options = AudioGenerationOptions(
                        prompt=audio_text,
                        kind=str(audio_kind_v or "video_audio"),
                        model_id=str(
                            audio_model_v
                            or (
                                "facebook/audiogen-medium"
                                if audio_kind_v == "sfx"
                                else "facebook/musicgen-small"
                                if audio_kind_v == "music"
                                else "mmaudio:large_44k_v2"
                            )
                        ),
                        duration_seconds=max(1.0, duration_value) if duration_value > 0 else 8.0,
                        temperature=float(audio_temperature_v or 1.0),
                        cfg_coef=float(audio_cfg_v or (4.5 if audio_kind_v == "video_audio" else 3.0)),
                        seed=int(audio_seed_v if audio_seed_v is not None else -1),
                    )
                    progress(0.0, desc="Generating audio")
                    audio_result, mux_result = audio_service.generate_and_mux(
                        final_video_path,
                        audio_options,
                        duration_seconds=None if duration_value <= 0 else duration_value,
                    )
                    final_video_path = _existing_video_output_path(mux_result.output_path, "Audio mux")
                    status_parts.append(f"**Audio** -- {audio_result.message}")
                except AudioUnavailable as exc:
                    logger.warning("Audio post-processing unavailable: %s", exc)
                    status_parts.append(f"**Audio skipped** -- {exc}")
                except Exception as exc:
                    logger.exception("Audio post-processing failed")
                    status_parts.append(f"**Audio failed** -- {exc}")

            _wan_cancel_flag[0] = False
            return final_video_path, "\n\n".join(status_parts)

        def _clear_previous_video():
            return gr.update(value=None), "**Generating** -- preparing Wan video..."

        def _clear_previous_ltx_video():
            return gr.update(value=None), "**Generating** -- preparing LTX video..."

        def _stop_video():
            _wan_cancel_flag[0] = True
            try:
                ctx.generation.interrupt()
            except Exception:
                pass
            return "**Stopping** — interrupt requested for video"

        def _run_ltx(
            source_path,
            prompt_v,
            negative_v,
            pipeline_v,
            width_v,
            height_v,
            frames_v,
            fps_v,
            steps_v,
            seed_v,
            offload_v,
            quantization_v,
            enhance_prompt_v,
            checkpoint_v,
            upsampler_v,
            gemma_v,
            t5_encoder_v,
            t5_tokenizer_v,
            gemma_backend_v,
            gemma_gguf_v,
            progress=gr.Progress(),
        ):
            prompt_text = str(prompt_v or "").strip()
            if not prompt_text:
                raise gr.Error("Enter an LTX prompt first.")
            progress(0.02, desc="Validating LTX route and model paths")
            try:
                request = LtxVideoRequest(
                    prompt=prompt_text,
                    negative_prompt=str(negative_v or ""),
                    source_image_path=str(source_path or "") or None,
                    pipeline=str(pipeline_v or default_ltx_pipeline),
                    width=int(width_v or 128),
                    height=int(height_v or 128),
                    num_frames=snap_ltx_num_frames(int(frames_v or 9)),
                    fps=float(fps_v or 8),
                    steps=int(steps_v or 1),
                    seed=int(seed_v if seed_v is not None else -1),
                    offload=str(offload_v or "disk"),
                    quantization=str(quantization_v or ""),
                    enhance_prompt=bool(enhance_prompt_v),
                    checkpoint_path=str(checkpoint_v or ""),
                    spatial_upsampler_path=str(upsampler_v or ""),
                    gemma_root=str(gemma_v or ""),
                    t5_encoder_path=str(t5_encoder_v or ""),
                    t5_tokenizer=str(t5_tokenizer_v or LTX_T5_TOKENIZER),
                    gemma_backend=str(gemma_backend_v or LTX_GEMMA_BACKEND_HF_SAFETENSORS),
                    gemma_gguf_path=str(gemma_gguf_v or ""),
                )
                progress(0.05, desc="Launching LTX route")
                result = ltx_service.generate(request)
            except LtxUnavailable as exc:
                raise gr.Error(str(exc))
            except ValueError as exc:
                raise gr.Error(str(exc))
            except Exception as exc:
                logger.exception("LTX generation failed")
                raise gr.Error(f"LTX generation failed: {exc}") from exc
            return result.output_path, f"**Done** -- {result.message}"

        run_event = run.click(
            _clear_previous_video,
            outputs=[video_out, status],
            show_progress="hidden",
            queue=False,
        )
        run_event.then(
            _run,
            inputs=[
                source,
                prompt,
                negative,
                offload,
                vram_reserve_enabled,
                vram_reserve_mb,
                width,
                height,
                num_frames,
                fps,
                high_steps,
                low_steps,
                guidance,
                sampler,
                sigma_type,
                flow_shift,
                seed,
                runtime_mode,
                high_noise,
                low_noise,
                vae_id,
                text_encoder,
                high_lora,
                high_lora_scale,
                low_lora,
                low_lora_scale,
                chunk_size,
                chunk_overlap,
                temporal_chunks,
                image_guidance_scale,
                rife_enabled,
                rife_target_fps,
                rife_ckpt,
                rife_scale_factor,
                rife_clear_cache,
                rife_fast_mode,
                rife_ensemble,
                reactor_enabled,
                reactor_source_mode,
                reactor_source_image,
                reactor_face_model,
                reactor_source_index,
                reactor_target_index,
                reactor_model,
                reactor_mask_face,
                reactor_restore_face,
                reactor_restorer,
                reactor_restore_visibility,
                reactor_codeformer_weight,
                vsr_enabled,
                vsr_scale,
                vsr_mode,
                vsr_effect,
                vsr_strength,
                videofx_denoise_enabled,
                videofx_denoise_strength,
                videofx_aigs_enabled,
                videofx_aigs_comp,
                videofx_aigs_blur,
                videofx_aigs_bg,
                videofx_aigs_cuda_graph,
                videofx_relight_enabled,
                videofx_relight_hdr,
                videofx_relight_bg_mode,
                videofx_relight_pan,
                videofx_relight_vfov,
                videofx_relight_bg,
                videofx_relight_bg_text,
                videofx_relight_autorotate,
                audio_enabled,
                audio_prompt,
                audio_kind,
                audio_model,
                audio_duration,
                audio_seed,
                audio_temperature,
                audio_cfg,
            ],
            outputs=[video_out, status],
            show_progress="minimal",
            show_progress_on=[status],
        )
        save_bad_video.click(
            wan_controller.archive_bad_video,
            inputs=[video_out],
            outputs=[status],
            show_progress=False,
        )
        stop_btn.click(
            _stop_video,
            outputs=[status],
            show_progress=False,
        )
        ltx_event = ltx_run.click(
            _clear_previous_ltx_video,
            outputs=[ltx_video_out, ltx_status],
            show_progress="hidden",
            queue=False,
        )
        ltx_event.then(
            _run_ltx,
            inputs=[
                ltx_source,
                ltx_prompt,
                ltx_negative,
                ltx_pipeline,
                ltx_width,
                ltx_height,
                ltx_frames,
                ltx_fps,
                ltx_steps,
                ltx_seed,
                ltx_offload,
                ltx_quantization,
                ltx_enhance_prompt,
                ltx_checkpoint,
                ltx_upsampler,
                ltx_gemma,
                ltx_t5_encoder,
                ltx_t5_tokenizer,
                ltx_gemma_backend,
                ltx_gemma_gguf,
            ],
            outputs=[ltx_video_out, ltx_status],
            show_progress="minimal",
            show_progress_on=[ltx_status],
        )
        ltx_save_bad_video.click(
            wan_controller.archive_bad_video,
            inputs=[ltx_video_out],
            outputs=[ltx_status],
            show_progress=False,
        )

        if tab is not None:

            def _load_pending():
                img = ctx.infotext_bridge.consume_image()
                face_models = _reactor_face_model_choices()
                swapper_models = _faceswap_model_choices()
                source_update = gr.update(value=img) if img is not None else gr.update()
                face_model_update = gr.update(
                    choices=face_models,
                    value=face_models[0][1] if face_models else None,
                )
                swapper_update = gr.update(
                    choices=swapper_models,
                    value=swapper_models[0][1] if swapper_models else "inswapper_128",
                )
                return source_update, face_model_update, swapper_update

            tab.select(_load_pending, outputs=[source, reactor_face_model, reactor_model], show_progress=False)
