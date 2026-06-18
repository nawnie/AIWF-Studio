from __future__ import annotations

import gc
import logging
import math
import os

import gradio as gr

from aiwf.core.domain.enhance import RestoreOptions
from aiwf.core.domain.faceswap import FaceSwapOptions
from aiwf.core.domain.rife import RifeOptions
from aiwf.bootstrap import AppContext
from aiwf.core.domain.wan import (
    SIGMA_TYPES,
    SAMPLER_TYPES,
    WAN_RUNTIME_FAST_5B,
    WAN_RUNTIME_HIGH_LOW,
    WanI2VRequest,
    duration_seconds_for_frames,
    frames_for_duration_seconds,
)
from aiwf.infrastructure.faceswap import FaceSwapUnavailable
from aiwf.infrastructure.rife import RifeUnavailable
from aiwf.infrastructure.video import VideoError, extract_first_frame
from aiwf.infrastructure.wan import WanUnavailable
from aiwf.services.rife import RifeService
from aiwf.services.wan import (
    WanService,
    wan_model_quant_family,
    wan_model_stage_role,
    wan_model_storage_family,
)
from aiwf.web.registry import WebRegistry
from aiwf.web.studio.resolution import (
    ASPECT_RATIO_PRESETS,
    NON_SQUARE_ASPECT_RATIO_PRESETS,
    dimensions_from_generation_preset,
)

_SERVICES: dict[int, WanService] = {}
_RIFE_SERVICES: dict[int, RifeService] = {}
VIDEO_SIZE_PRESETS: tuple[int, ...] = (480, 568, 640, 768, 896, 1024)
logger = logging.getLogger(__name__)


def _service(ctx: AppContext) -> WanService:
    svc = _SERVICES.get(id(ctx))
    if svc is None:
        svc = WanService(
            ctx.flags,
            ctx.settings,
            unload_image_models=ctx.generation.backend.unload,
            supervisor=ctx.supervisor,
        )
        _SERVICES[id(ctx)] = svc
    return svc


def _rife_service(ctx: AppContext) -> RifeService:
    svc = _RIFE_SERVICES.get(id(ctx))
    if svc is None:
        svc = RifeService(ctx.flags, ctx.settings, ctx.generation.backend.devices, supervisor=ctx.supervisor)
        _RIFE_SERVICES[id(ctx)] = svc
    return svc


def _format_it_s(steps_per_second) -> str:
    try:
        rate = float(steps_per_second)
    except (TypeError, ValueError):
        return ""
    if rate <= 0 or rate != rate:
        return ""
    return f"{rate:.3f} it/s ({1.0 / rate:.2f} s/it)"


def _experimental_wan_formats_enabled() -> bool:
    return os.environ.get("AIWF_WAN_ENABLE_EXPERIMENTAL_FORMATS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _rife_multiplier_for_target(input_fps: int | float, target_fps: int | float) -> int:
    safe_input = max(1.0, float(input_fps or 1))
    safe_target = max(1.0, float(target_fps or safe_input))
    return max(2, min(8, int(math.ceil(safe_target / safe_input))))


def register_wan_i2v(registry: WebRegistry) -> None:
    @registry.tab("Video", order=2)
    def build(ctx: AppContext, tab: gr.Tab | None = None) -> None:
        service = _service(ctx)
        rife_service = _rife_service(ctx)
        rife_ckpts = rife_service.list_checkpoints()
        default_rife_ckpt = rife_service.default_checkpoint()

        def _faceswap_model_choices() -> list[tuple[str, str]]:
            return [(m.title, m.id) for m in ctx.faceswap.list_models()]

        def _reactor_face_model_choices() -> list[tuple[str, str]]:
            return [(m.title, m.id) for m in ctx.faceswap.list_face_models()]

        def _restorer_choices() -> list[tuple[str, str]]:
            return [(m.title, m.id) for m in ctx.enhance.list_restorers()]

        reactor_swapper_choices = _faceswap_model_choices()
        reactor_face_model_choices = _reactor_face_model_choices()
        reactor_restorer_choices = _restorer_choices()

        # Labeled (display_name, identifier) choices — read from model file headers.
        all_labeled = service.list_local_models_labeled() if hasattr(service, "list_local_models_labeled") else []
        if not all_labeled:
            all_labeled = [(m, m) for m in service.list_local_models()]

        high_lora_choices = service.list_local_loras("high") if hasattr(service, "list_local_loras") else []
        low_lora_choices = service.list_local_loras("low") if hasattr(service, "list_local_loras") else []

        # Sort high/low noise models to the top of each dropdown; unknown-role in both.
        high_labeled = [c for c in all_labeled if "high" in c[0].lower() or "high" in c[1].lower()]
        low_labeled  = [c for c in all_labeled if "low"  in c[0].lower() or "low"  in c[1].lower()]
        other_labeled = [c for c in all_labeled if c not in high_labeled and c not in low_labeled]
        high_labeled = list(dict.fromkeys(high_labeled + other_labeled + all_labeled))
        low_labeled  = list(dict.fromkeys(low_labeled  + other_labeled + all_labeled))
        if not _experimental_wan_formats_enabled():
            high_labeled = [c for c in high_labeled if wan_model_storage_family(c[1]) == "gguf"]
            low_labeled = [c for c in low_labeled if wan_model_storage_family(c[1]) == "gguf"]

        def _filter_stage_choices(
            labeled: list[tuple[str, str]],
            *,
            stage: str,
            peer_value: str | None,
        ) -> list[tuple[str, str]]:
            peer_storage = wan_model_storage_family(peer_value)
            peer_quant = wan_model_quant_family(peer_value)
            filtered: list[tuple[str, str]] = []
            for label, value in labeled:
                role = wan_model_stage_role(value)
                if role not in {stage, "unknown"}:
                    continue
                storage = wan_model_storage_family(value)
                quant = wan_model_quant_family(value)
                if peer_storage != "unknown" and storage != "unknown" and storage != peer_storage:
                    continue
                if peer_quant != "unknown" and quant != "unknown" and quant != peer_quant:
                    continue
                filtered.append((label, value))
            return filtered or labeled

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
            if not _experimental_wan_formats_enabled() and (high_storage != "gguf" or low_storage != "gguf"):
                return "**Model pair blocked:** stable Video currently accepts GGUF high/low pairs only."
            if high_storage != "unknown" and low_storage != "unknown" and high_storage != low_storage:
                return f"**Model pair blocked:** {high_storage} high + {low_storage} low."
            if high_quant != "unknown" and low_quant != "unknown" and high_quant != low_quant:
                return f"**Model pair blocked:** {high_quant.upper()} high + {low_quant.upper()} low."
            parts = [p for p in (high_storage, high_quant) if p != "unknown"]
            return "**Model pair:** " + (" / ".join(parts) if parts else "stage roles only")

        # Load persisted defaults
        _s = ctx.settings
        _last_high = getattr(_s, "last_wan_high", "")
        _last_low  = getattr(_s, "last_wan_low", "")
        _last_vae  = getattr(_s, "last_wan_vae", "")
        _last_te   = getattr(_s, "last_wan_text_encoder", "")
        _last_offload = getattr(_s, "last_wan_offload", "balanced")
        _working_offload_defaults = {"balanced", "model", "sequential"}
        _offload_default = _last_offload if _last_offload in _working_offload_defaults else "balanced"

        def _best_default(labeled: list[tuple[str, str]], persisted: str) -> str | None:
            ids = [v for _, v in labeled]
            if persisted and persisted in ids:
                return persisted
            return ids[0] if ids else None

        vae_labeled = service.list_local_vaes_labeled() if hasattr(service, "list_local_vaes_labeled") else []
        if not vae_labeled:
            vae_labeled = [("Default VAE", "")]
        preferred_vae_id = _last_vae or next(
            (v for _, v in vae_labeled if v and "wan" in v.lower() and "vae" in v.lower()), None
        ) or (vae_labeled[0][1] if vae_labeled else None)

        te_labeled = service.list_local_text_encoders_labeled() if hasattr(service, "list_local_text_encoders_labeled") else []
        default_te_labeled = [("Default (full precision bundled encoder)", "")] + te_labeled
        default_te = _last_te if _last_te else (service.default_text_encoder() if hasattr(service, "default_text_encoder") else "")

        initial_high = _best_default(high_labeled, _last_high)
        initial_low_choices = _filter_stage_choices(low_labeled, stage="low", peer_value=initial_high)
        initial_low = _valid_or_first(_best_default(low_labeled, _last_low), initial_low_choices)
        initial_high_choices = _filter_stage_choices(high_labeled, stage="high", peer_value=initial_low)
        initial_high = _valid_or_first(initial_high, initial_high_choices)

        default_video_size = 480
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
                    "Wan image-to-video. Stable sharing build: matched GGUF High Noise + Low Noise pairs only.",
                    elem_classes=["aiwf-page-intro"],
                )
                gr.Markdown(service.folder_help(), elem_classes=["aiwf-page-path"])

            with gr.Row(equal_height=False):
                with gr.Column(scale=1, min_width=340, elem_classes=["aiwf-panel"]):
                    source = gr.Image(label="Source image", type="pil", sources=["upload", "clipboard"])
                    prompt = gr.Textbox(label="Prompt", lines=3, placeholder="Describe the motion / scene")
                    negative = gr.Textbox(label="Negative prompt", lines=2, value="")

                    runtime_mode = gr.Radio(
                        label="Runtime",
                        choices=[
                            ("Stable: high/low GGUF", WAN_RUNTIME_HIGH_LOW),
                        ],
                        value=WAN_RUNTIME_HIGH_LOW,
                        info="Only the GGUF path is exposed on main; FP8 and resident tests live on dev.",
                    )

                    gr.Markdown("Models", elem_classes=["aiwf-section-label"])
                    high_noise = gr.Dropdown(
                        label="High noise transformer",
                        choices=initial_high_choices,
                        value=initial_high,
                        allow_custom_value=True,
                        info="Early denoising stage. GGUF only on stable main.",
                    )
                    low_noise = gr.Dropdown(
                        label="Low noise transformer",
                        choices=initial_low_choices,
                        value=initial_low,
                        allow_custom_value=True,
                        info="Late denoising stage. Must match the high GGUF.",
                    )
                    model_pair_status = gr.Markdown(
                        _pair_status(initial_high, initial_low),
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
                        info="Wan 2.1 VAE is recommended.",
                    )

                    gr.Markdown("Stage LoRAs", elem_classes=["aiwf-section-label"])
                    high_lora = gr.Dropdown(
                        label="High noise LoRA",
                        choices=high_lora_choices,
                        value=None,
                        allow_custom_value=True,
                        info="Optional high-stage LoRA.",
                    )
                    with gr.Row():
                        high_lora_scale = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="High LoRA strength")
                        low_lora_scale = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Low LoRA strength")
                    low_lora = gr.Dropdown(
                        label="Low noise LoRA",
                        choices=low_lora_choices,
                        value=None,
                        allow_custom_value=True,
                        info="Optional low-stage LoRA.",
                    )

                    gr.Markdown("Runtime", elem_classes=["aiwf-section-label"])
                    offload = gr.Dropdown(
                        label="VRAM / offload",
                        choices=[
                            ("Balanced 16 GB: active GGUF stage swaps, VAE stays hot", "balanced"),
                            ("Low VRAM: active GGUF stage swaps, VAE/text offload", "model"),
                            ("Sequential: slow fallback", "sequential"),
                        ],
                        value=_offload_default,
                        info="Use Balanced first. Use Low VRAM if it OOMs.",
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
                        width = gr.Slider(128, 1280, value=480, step=8, label="Width")
                        height = gr.Slider(128, 1280, value=480, step=8, label="Height")

                    gr.Markdown("Motion", elem_classes=["aiwf-section-label"])
                    with gr.Row():
                        fps = gr.Slider(1, 24, value=16, step=1, label="FPS")
                        duration_seconds = gr.Slider(1, 10, value=3, step=1, label="Duration (seconds)")
                    with gr.Row():
                        num_frames = gr.Number(value=49, precision=0, label="Frames", interactive=False)
                        guidance = gr.Slider(1.0, 12.0, value=1.0, step=0.5, label="Guidance (CFG)")
                    frame_summary = gr.Markdown(
                        "**Frames:** 49 · **Duration:** 3.0s snapped for Wan",
                        elem_classes=["aiwf-settings-paths"],
                    )

                    gr.Markdown("Denoising split", elem_classes=["aiwf-section-label"])
                    with gr.Row():
                        high_steps = gr.Slider(1, 30, value=4, step=1, label="High noise steps")
                        low_steps = gr.Slider(1, 30, value=4, step=1, label="Low noise steps")
                    with gr.Row():
                        total_steps = gr.Number(value=8, precision=0, label="Total steps", interactive=False)
                        boundary_ratio = gr.Number(value=0.5, precision=3, label="Stage split", interactive=False)

                    gr.Markdown("Sampler", elem_classes=["aiwf-section-label"])
                    sampler = gr.Dropdown(
                        label="Sampler",
                        choices=[
                            ("FlowMatch Euler (recommended — fast, 1 NFE/step)", "euler"),
                            ("FlowMatch Heun (2nd-order, higher quality, ~2× slower)", "heun"),
                        ],
                        value="euler",
                        info="Heun can improve motion but roughly doubles step time.",
                    )
                    sigma_type = gr.Dropdown(
                        label="Scheduler",
                        choices=[
                            ("Beta — smooth motion, best quality at low steps (recommended)", "beta"),
                            ("Simple — linear uniform spacing (fastest)", "simple"),
                            ("Exponential — more detail at high noise", "exponential"),
                            ("Karras — SD-style detail preservation", "karras"),
                        ],
                        value="simple",
                        info="Simple is fastest; Beta is the quality check.",
                    )
                    with gr.Row():
                        flow_shift = gr.Slider(
                            0.5, 25.0, value=5.0, step=0.5, label="Flow shift",
                            info="Default 5.0. Higher shifts more work to high-noise.",
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
                        1.0, 5.0, value=1.0, step=0.1, label="Image guidance scale",
                        info="Raise to reduce drift on longer clips.",
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

                    run = gr.Button("Generate video", variant="primary", elem_classes=["aiwf-generate-btn"])
                    video_out = gr.Video(label="Result", interactive=False)
                    status = gr.Markdown("**Ready** — upload an image and generate.", elem_classes=["aiwf-status-bar"])

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

        resolution_size.change(
            _apply_resolution_preset,
            inputs=[resolution_size, resolution_ratio, resolution_ratio_square],
            outputs=[width, height],
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
                gr.update(value=f"**Frames:** {frames} · **Duration:** {snapped:.1f}s snapped for Wan"),
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

        def _sync_step_split(high_value, low_value):
            high = max(1, int(high_value or 0))
            low = max(1, int(low_value or 0))
            total = high + low
            return gr.update(value=total), gr.update(value=round(high / total, 3))

        high_steps.change(
            _sync_step_split,
            inputs=[high_steps, low_steps],
            outputs=[total_steps, boundary_ratio],
            show_progress=False,
        )
        low_steps.change(
            _sync_step_split,
            inputs=[high_steps, low_steps],
            outputs=[total_steps, boundary_ratio],
            show_progress=False,
        )

        def _sync_runtime_choices(runtime_value, high_value, low_value):
            selected_runtime = str(runtime_value or WAN_RUNTIME_HIGH_LOW)
            if selected_runtime == WAN_RUNTIME_FAST_5B:
                return (
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    "",
                )
            high_choices = _filter_stage_choices(high_labeled, stage="high", peer_value=low_value)
            low_choices = _filter_stage_choices(low_labeled, stage="low", peer_value=high_value)
            next_high = _valid_or_first(high_value, high_choices)
            next_low = _valid_or_first(low_value, low_choices)
            return (
                gr.update(choices=high_choices, value=next_high, interactive=True),
                gr.update(choices=low_choices, value=next_low, interactive=True),
                _pair_status(next_high, next_low),
            )

        def _sync_low_choices(high_value, low_value, runtime_value):
            if str(runtime_value or WAN_RUNTIME_HIGH_LOW) == WAN_RUNTIME_FAST_5B:
                return gr.update(interactive=False), ""
            choices = _filter_stage_choices(low_labeled, stage="low", peer_value=high_value)
            next_low = _valid_or_first(low_value, choices)
            return gr.update(choices=choices, value=next_low, interactive=True), _pair_status(high_value, next_low)

        def _sync_high_choices(low_value, high_value, runtime_value):
            if str(runtime_value or WAN_RUNTIME_HIGH_LOW) == WAN_RUNTIME_FAST_5B:
                return gr.update(interactive=False), ""
            choices = _filter_stage_choices(high_labeled, stage="high", peer_value=low_value)
            next_high = _valid_or_first(high_value, choices)
            return gr.update(choices=choices, value=next_high, interactive=True), _pair_status(next_high, low_value)

        runtime_mode.change(
            _sync_runtime_choices,
            inputs=[runtime_mode, high_noise, low_noise],
            outputs=[high_noise, low_noise, model_pair_status],
            show_progress=False,
        )
        high_noise.change(
            _sync_low_choices,
            inputs=[high_noise, low_noise, runtime_mode],
            outputs=[low_noise, model_pair_status],
            show_progress=False,
        )
        low_noise.change(
            _sync_high_choices,
            inputs=[low_noise, high_noise, runtime_mode],
            outputs=[high_noise, model_pair_status],
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

        def _release_memory_before_postprocess(label: str):
            try:
                service.unload_models()
            except Exception:
                logger.exception("Failed to unload Wan models before %s post-processing.", label)
            try:
                ctx.generation.backend.unload()
            except Exception:
                logger.debug("Image backend unload before %s failed.", label, exc_info=True)
            gc.collect()
            try:
                ctx.generation.backend.devices.empty_cache()
            except Exception:
                logger.debug("Device cache cleanup before %s failed.", label, exc_info=True)

        def _release_memory_before_reactor():
            _release_memory_before_postprocess("ReActor")
            try:
                ctx.faceswap.unload()
            except Exception:
                logger.debug("Face swap unload before ReActor failed.", exc_info=True)

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
            progress=gr.Progress(),
        ):
            if image is None:
                raise gr.Error("Upload a source image first.")
            if not service.available():
                raise gr.Error(
                    "Wan video is unavailable — update diffusers (>=0.35) and install ftfy, then restart."
                )
            selected_runtime = str(runtime_mode_v or WAN_RUNTIME_FAST_5B)
            if selected_runtime != WAN_RUNTIME_FAST_5B and not (high_v and low_v):
                raise gr.Error(
                    "Select BOTH a High noise model and a Low noise model. Wan 2.2 image-to-video "
                    "high/low modes run a two-stage transformer pair."
                )
            if selected_runtime != WAN_RUNTIME_FAST_5B and not _experimental_wan_formats_enabled():
                if wan_model_storage_family(high_v) != "gguf" or wan_model_storage_family(low_v) != "gguf":
                    raise gr.Error(
                        "Stable Video currently supports GGUF high/low transformer pairs only. "
                        "FP8/safetensors video experiments are kept on the dev branch."
                    )
            # Warn if user somehow selected a t5xxl file (shouldn't happen via dropdown but allow_custom_value=True)
            _te_path = str(text_encoder_v or "").strip()
            if _te_path and ("t5xxl" in _te_path.lower()) and not any(k in _te_path.lower() for k in ("umt5", "nsfw_wan")):
                raise gr.Error(
                    f"⚠ '{_te_path}' looks like a T5-XXL file (Flux/SD3). "
                    "T5-XXL is NOT compatible with Wan — it will produce garbage output. "
                    "Select 'Default' or a UMT5-XXL file (umt5-xxl-*.gguf or umt5/nsfw_wan_*.safetensors)."
                )

            request = WanI2VRequest(
                prompt=prompt_v or "",
                negative_prompt=negative_v or "",
                width=int(width_v),
                height=int(height_v),
                num_frames=int(frames_v),
                fps=int(fps_v),
                steps=max(1, int(high_steps_v or 0) + int(low_steps_v or 0)),
                high_noise_steps=int(high_steps_v),
                low_noise_steps=int(low_steps_v),
                guidance_scale=float(guidance_v),
                sampler=str(sampler_v or "euler"),
                sigma_type=str(sigma_type_v or "beta"),
                flow_shift=float(flow_v),
                seed=int(seed_v),
                runtime_mode=selected_runtime,
                offload=offload_v,
                vram_reserve_enabled=bool(vram_reserve_enabled_v),
                vram_reserve_mb=int(vram_reserve_mb_v or 0),
                high_noise_model_id=high_v or None,
                low_noise_model_id=low_v or None,
                high_noise_lora_id=high_lora_v or None,
                high_noise_lora_scale=float(high_lora_scale_v),
                low_noise_lora_id=low_lora_v or None,
                low_noise_lora_scale=float(low_lora_scale_v),
                boundary_ratio=0.5,
                vae_id=vae_v or None,
                text_encoder_path=_te_path,
                temporal_chunks=bool(temporal_chunks_v),
                chunk_size=int(chunk_size_v or 24),
                chunk_overlap=int(chunk_overlap_v or 0),
                image_guidance_scale=float(image_guidance_scale_v or 1.0),
            )

            def on_progress(step, tot, steps_per_second=None):
                rate_text = _format_it_s(steps_per_second)
                desc = f"Video step {step}/{tot}"
                if rate_text:
                    desc = f"{desc} - {rate_text}"
                progress(min(1.0, step / max(1, tot)), desc=desc)

            runtime_label = "Wan 5B demo" if selected_runtime == WAN_RUNTIME_FAST_5B else "Wan 14B dual-stage I2V"
            progress(0.0, desc=f"Loading + encoding for {runtime_label} (watch terminal for [AIWF] Video: and step messages, then 'Video step X/Y' will appear)")
            try:
                result = service.generate(request, image, on_progress=on_progress)
            except WanUnavailable as exc:
                raise gr.Error(str(exc))
            except Exception as exc:
                logger.exception("Video generation failed")
                raise gr.Error(f"Video generation failed: {exc}") from exc

            # Persist last-used model/encoder selections so they restore on next launch
            s = ctx.settings
            changed = False
            for attr, val in [
                ("last_wan_high", str(high_v or "")),
                ("last_wan_low", str(low_v or "")),
                ("last_wan_vae", str(vae_v or "")),
                ("last_wan_text_encoder", str(text_encoder_v or "")),
                ("last_wan_offload", str(offload_v or "balanced")),
            ]:
                if getattr(s, attr, None) != val:
                    setattr(s, attr, val)
                    changed = True
            if changed:
                try:
                    ctx.save_settings()
                except Exception:
                    pass

            final_video_path = result.output_path
            status_parts = [f"**Done** -- {result.message}"]
            if bool(rife_enabled_v):
                target_fps = int(rife_target_fps_v or 30)
                input_fps = int(getattr(result, "fps", None) or fps_v or 16)
                multiplier = _rife_multiplier_for_target(input_fps, target_fps)
                progress(0.0, desc="Unloading Wan VRAM before RIFE")
                _release_memory_before_postprocess("RIFE")
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
                        result.output_path,
                        rife_options,
                        on_progress=on_rife_progress,
                    )
                    final_video_path = rife_result.output_path
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
                    _release_memory_before_reactor()
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
                    final_video_path = reactor_result.output_path
                    status_parts.append(f"**ReActor** -- {reactor_result.message}")
                except (FaceSwapUnavailable, VideoError) as exc:
                    logger.warning("ReActor post-processing unavailable: %s", exc)
                    status_parts.append(f"**ReActor skipped** -- {exc}")
                except Exception as exc:
                    logger.exception("ReActor post-processing failed")
                    status_parts.append(f"**ReActor failed** -- {exc}")

            return final_video_path, "\n\n".join(status_parts)

        run.click(
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
            ],
            outputs=[video_out, status],
            show_progress="minimal",
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
