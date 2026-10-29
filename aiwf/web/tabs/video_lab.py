from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

import gradio as gr

from aiwf.bootstrap import AppContext
from aiwf.core.domain.video_lab import MediaProbe, VideoLabSettings
from aiwf.services.video_lab import VideoLabBusy, VideoLabCancelled, VideoLabService, preset_settings
from aiwf.web.registry import WebRegistry

_SERVICES: dict[int, VideoLabService] = {}

VIDEO_STAGE_ORDER: tuple[str, ...] = (
    "timeline",
    "deinterlace",
    "stabilize",
    "deflicker",
    "denoise",
    "sharpen",
    "resize",
    "retime",
    "audio_cleanup",
    "audio_normalize",
    "export",
)
VIDEO_STAGE_LABELS: dict[str, str] = {
    "timeline": "Trim",
    "deinterlace": "Deinterlace",
    "stabilize": "Stabilize",
    "deflicker": "Deflicker",
    "denoise": "Denoise",
    "sharpen": "Sharpen",
    "resize": "Resize",
    "retime": "Frame-rate conversion",
    "audio_cleanup": "Audio cleanup",
    "audio_normalize": "Loudness normalize",
    "export": "Export",
}


def _service(ctx: AppContext) -> VideoLabService:
    service = _SERVICES.get(id(ctx))
    if service is None:
        service = VideoLabService(ctx.flags.resolved_output_dir())
        _SERVICES[id(ctx)] = service
    return service


def _video_path(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("path") or value.get("name") or "")
    return str(getattr(value, "path", None) or getattr(value, "name", None) or "")


def normalize_video_stages(values) -> list[str]:
    selected = {str(item) for item in (values or [])}
    result = [stage for stage in VIDEO_STAGE_ORDER if stage in selected]
    if "export" not in result:
        result.append("export")
    return result


def video_stages_from_settings(settings: VideoLabSettings) -> list[str]:
    stages: list[str] = []
    if settings.trim_start or settings.trim_end is not None:
        stages.append("timeline")
    if settings.deinterlace:
        stages.append("deinterlace")
    if settings.stabilize:
        stages.append("stabilize")
    if settings.deflicker:
        stages.append("deflicker")
    if settings.denoise != "off":
        stages.append("denoise")
    if settings.sharpen != "off":
        stages.append("sharpen")
    if settings.scale != "keep":
        stages.append("resize")
    if settings.target_fps is not None:
        stages.append("retime")
    if settings.audio_cleanup:
        stages.append("audio_cleanup")
    if settings.audio_normalize:
        stages.append("audio_normalize")
    stages.append("export")
    return normalize_video_stages(stages)


def video_order_markdown(stages, warnings: list[str] | None = None) -> str:
    normalized = normalize_video_stages(stages)
    flow = " → ".join(VIDEO_STAGE_LABELS[item] for item in normalized)
    extra = "" if not warnings else "  \n" + "  \n".join(f"⚠ {item}" for item in warnings)
    return f"**Resolved order:** Inspect → {flow}{extra}"


def _settings_from_values(
    selected_stages,
    preset,
    trim_start,
    trim_end,
    deinterlace_mode,
    deinterlace_parity,
    deinterlace_scope,
    stabilize_radius_x,
    stabilize_radius_y,
    stabilize_edge,
    stabilize_block_size,
    stabilize_contrast,
    deflicker_size,
    deflicker_mode,
    denoise_profile,
    denoise_luma_spatial,
    denoise_chroma_spatial,
    denoise_luma_temporal,
    denoise_chroma_temporal,
    sharpen_profile,
    sharpen_kernel,
    sharpen_amount,
    scale,
    custom_width,
    custom_height,
    keep_aspect,
    target_fps,
    motion_interpolation,
    audio_highpass,
    audio_lowpass,
    audio_noise_reduction,
    audio_noise_floor,
    audio_noise_type,
    audio_track_noise,
    audio_target_lufs,
    audio_true_peak,
    audio_lra,
    codec,
    container,
    quality,
    audio_bitrate,
) -> VideoLabSettings:
    selected = set(normalize_video_stages(selected_stages))
    end = float(trim_end or 0)
    fps = float(target_fps or 0)
    scale_value = str(scale or "keep") if "resize" in selected else "keep"
    return VideoLabSettings(
        preset=str(preset or "custom"),
        trim_start=float(trim_start or 0) if "timeline" in selected else 0.0,
        trim_end=(None if end <= 0 else end) if "timeline" in selected else None,
        deinterlace="deinterlace" in selected,
        deinterlace_mode=str(deinterlace_mode or "send_frame"),
        deinterlace_parity=str(deinterlace_parity or "auto"),
        deinterlace_scope=str(deinterlace_scope or "interlaced"),
        stabilize="stabilize" in selected,
        stabilize_radius_x=int(stabilize_radius_x),
        stabilize_radius_y=int(stabilize_radius_y),
        stabilize_edge=str(stabilize_edge or "mirror"),
        stabilize_block_size=int(stabilize_block_size),
        stabilize_contrast=int(stabilize_contrast),
        deflicker="deflicker" in selected,
        deflicker_size=int(deflicker_size),
        deflicker_mode=str(deflicker_mode or "pm"),
        denoise=str(denoise_profile or "custom") if "denoise" in selected else "off",
        denoise_luma_spatial=float(denoise_luma_spatial),
        denoise_chroma_spatial=float(denoise_chroma_spatial),
        denoise_luma_temporal=float(denoise_luma_temporal),
        denoise_chroma_temporal=float(denoise_chroma_temporal),
        sharpen=str(sharpen_profile or "custom") if "sharpen" in selected else "off",
        sharpen_kernel=int(sharpen_kernel),
        sharpen_amount=float(sharpen_amount),
        scale=scale_value,
        custom_width=int(custom_width or 0),
        custom_height=int(custom_height or 0),
        keep_aspect=bool(keep_aspect),
        target_fps=(None if fps <= 0 else fps) if "retime" in selected else None,
        motion_interpolation=bool(motion_interpolation) if "retime" in selected else False,
        audio_cleanup="audio_cleanup" in selected,
        audio_highpass_hz=float(audio_highpass),
        audio_lowpass_hz=float(audio_lowpass),
        audio_noise_reduction_db=float(audio_noise_reduction),
        audio_noise_floor_db=float(audio_noise_floor),
        audio_noise_type=str(audio_noise_type or "white"),
        audio_track_noise=bool(audio_track_noise),
        audio_normalize="audio_normalize" in selected,
        audio_target_lufs=float(audio_target_lufs),
        audio_true_peak_db=float(audio_true_peak),
        audio_lra=float(audio_lra),
        codec=str(codec or "auto"),
        container=str(container or "mp4"),
        quality=int(quality or 20),
        audio_bitrate_kbps=int(audio_bitrate or 192),
    )


def _probe_summary(probe: MediaProbe) -> str:
    duration = f"{probe.duration_seconds:.2f}s" if probe.duration_seconds else "unknown"
    audio = (
        f"{probe.audio_codec or 'audio'}, {probe.audio_channels or '?'} channel(s), "
        f"{probe.audio_sample_rate or '?'} Hz"
        if probe.has_audio
        else "none"
    )
    scan = "interlaced" if probe.is_interlaced else "progressive/unknown"
    return (
        f"**{Path(probe.path).name}**  \n"
        f"{probe.width}×{probe.height} · {probe.fps:.3f} FPS · {duration} · {probe.video_codec} / {probe.pixel_format}  \n"
        f"Scan: **{scan}** ({probe.field_order}) · Audio: **{audio}** · Subtitles: **{'yes' if probe.has_subtitles else 'no'}**  \n"
        f"Metadata source: `{probe.source}`"
    )


def _denoise_profile_values(name: str) -> tuple[float, float, float, float]:
    profiles = {
        "light": (1.5, 1.5, 6.0, 6.0),
        "strong": (3.0, 3.0, 9.0, 9.0),
        "custom": (1.5, 1.5, 6.0, 6.0),
    }
    return profiles.get(str(name or "custom"), profiles["custom"])


def _sharpen_profile_values(name: str) -> tuple[int, float]:
    profiles = {
        "light": (5, 0.45),
        "strong": (5, 0.80),
        "custom": (5, 0.45),
    }
    return profiles.get(str(name or "custom"), profiles["custom"])


def register_video_lab(registry: WebRegistry) -> None:
    @registry.tab("Video Lab", order=22)
    def build(ctx: AppContext, tab: gr.Tab | None = None) -> None:
        service = _service(ctx)
        initial = preset_settings("old_family_film")
        initial_stages = video_stages_from_settings(initial)

        with gr.Column(elem_classes=["aiwf-video-lab", "aiwf-video"]):
            with gr.Column(elem_classes=["aiwf-page-header"]):
                gr.Markdown("VIDEO LAB", elem_classes=["aiwf-section-label"])
                gr.Markdown(
                    "Choose the processes you want. Studio reveals their parameters and resolves the safest order.",
                    elem_classes=["aiwf-page-intro"],
                )

            with gr.Row(equal_height=False, elem_classes=["aiwf-lab-workspace"]):
                with gr.Column(scale=5, min_width=430, elem_classes=["aiwf-panel", "aiwf-lab-controls"]):
                    source = gr.Video(label="Input video", sources=["upload"], elem_classes=["aiwf-vlab-source"])
                    with gr.Row():
                        inspect_button = gr.Button("Inspect", elem_classes=["aiwf-btn-ghost"])
                        plan_button = gr.Button("Build plan", elem_classes=["aiwf-btn-ghost"])
                    probe_summary = gr.Markdown(
                        "Upload a local video, then inspect it. Nothing is uploaded outside this machine.",
                        elem_classes=["aiwf-status-bar"],
                    )
                    with gr.Row():
                        preset = gr.Dropdown(
                            label="Starting preset",
                            choices=[
                                ("Old family film", "old_family_film"),
                                ("Web video cleanup", "web_video_cleanup"),
                                ("Generated video polish", "generated_video_polish"),
                                ("Custom", "custom"),
                            ],
                            value="old_family_film",
                        )
                        apply_preset = gr.Button("Apply preset", elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"])

                    stages = gr.CheckboxGroup(
                        label="Processes to run",
                        choices=[(VIDEO_STAGE_LABELS[item], item) for item in VIDEO_STAGE_ORDER],
                        value=initial_stages,
                        info="Pick the work. Studio adds export and resolves the safe signal order.",
                        elem_classes=["aiwf-stage-picker"],
                    )
                    order_view = gr.Markdown(video_order_markdown(initial_stages), elem_classes=["aiwf-stage-order"])

                    with gr.Group(visible="timeline" in initial_stages, elem_classes=["aiwf-stage-card"]) as timeline_panel:
                        gr.Markdown("#### Trim")
                        with gr.Row():
                            trim_start = gr.Number(label="Start (seconds)", value=initial.trim_start, minimum=0)
                            trim_end = gr.Number(label="End (0 = source end)", value=initial.trim_end or 0, minimum=0)

                    with gr.Group(visible="deinterlace" in initial_stages, elem_classes=["aiwf-stage-card"]) as deinterlace_panel:
                        gr.Markdown("#### Deinterlace")
                        with gr.Row():
                            deinterlace_mode = gr.Dropdown(
                                label="Output cadence",
                                choices=[("Keep source frame rate", "send_frame"), ("One frame per field (2×)", "send_field")],
                                value=initial.deinterlace_mode,
                            )
                            deinterlace_parity = gr.Dropdown(
                                label="Field order",
                                choices=[("Auto", "auto"), ("Top field first", "tff"), ("Bottom field first", "bff")],
                                value=initial.deinterlace_parity,
                            )
                            deinterlace_scope = gr.Dropdown(
                                label="Process frames",
                                choices=[("All frames", "all"), ("Only marked interlaced", "interlaced")],
                                value=initial.deinterlace_scope,
                            )

                    with gr.Group(visible="stabilize" in initial_stages, elem_classes=["aiwf-stage-card"]) as stabilize_panel:
                        gr.Markdown("#### Stabilize")
                        with gr.Row():
                            stabilize_radius_x = gr.Slider(0, 64, value=initial.stabilize_radius_x, step=1, label="Horizontal search px")
                            stabilize_radius_y = gr.Slider(0, 64, value=initial.stabilize_radius_y, step=1, label="Vertical search px")
                            stabilize_edge = gr.Dropdown(
                                label="Edge fill",
                                choices=[("Mirror", "mirror"), ("Clamp", "clamp"), ("Original", "original"), ("Black", "blank")],
                                value=initial.stabilize_edge,
                            )
                        with gr.Row():
                            stabilize_block_size = gr.Slider(4, 128, value=initial.stabilize_block_size, step=4, label="Motion block size")
                            stabilize_contrast = gr.Slider(1, 255, value=initial.stabilize_contrast, step=1, label="Block contrast threshold")

                    with gr.Group(visible="deflicker" in initial_stages, elem_classes=["aiwf-stage-card"]) as deflicker_panel:
                        gr.Markdown("#### Deflicker")
                        with gr.Row():
                            deflicker_size = gr.Slider(2, 129, value=initial.deflicker_size, step=1, label="Temporal window (frames)")
                            deflicker_mode = gr.Dropdown(
                                label="Averaging mode",
                                choices=[
                                    ("Median", "median"), ("Power mean", "pm"), ("Arithmetic", "am"),
                                    ("Geometric", "gm"), ("Harmonic", "hm"), ("Quadratic", "qm"), ("Cubic", "cm"),
                                ],
                                value=initial.deflicker_mode,
                            )

                    with gr.Group(visible="denoise" in initial_stages, elem_classes=["aiwf-stage-card"]) as denoise_panel:
                        gr.Markdown("#### Denoise")
                        denoise_profile = gr.Dropdown(
                            label="Starting profile", choices=[("Light", "light"), ("Strong", "strong"), ("Custom", "custom")], value=initial.denoise
                        )
                        with gr.Row():
                            denoise_luma_spatial = gr.Slider(0, 20, value=initial.denoise_luma_spatial, step=0.1, label="Luma spatial")
                            denoise_chroma_spatial = gr.Slider(0, 20, value=initial.denoise_chroma_spatial, step=0.1, label="Chroma spatial")
                        with gr.Row():
                            denoise_luma_temporal = gr.Slider(0, 30, value=initial.denoise_luma_temporal, step=0.1, label="Luma temporal")
                            denoise_chroma_temporal = gr.Slider(0, 30, value=initial.denoise_chroma_temporal, step=0.1, label="Chroma temporal")

                    with gr.Group(visible="sharpen" in initial_stages, elem_classes=["aiwf-stage-card"]) as sharpen_panel:
                        gr.Markdown("#### Sharpen")
                        with gr.Row():
                            sharpen_profile = gr.Dropdown(
                                label="Starting profile", choices=[("Light", "light"), ("Strong", "strong"), ("Custom", "custom")], value=initial.sharpen
                            )
                            sharpen_kernel = gr.Slider(3, 23, value=initial.sharpen_kernel, step=2, label="Kernel size")
                            sharpen_amount = gr.Slider(-2, 5, value=initial.sharpen_amount, step=0.05, label="Luma amount")

                    with gr.Group(visible="resize" in initial_stages, elem_classes=["aiwf-stage-card"]) as resize_panel:
                        gr.Markdown("#### Resize")
                        scale = gr.Dropdown(
                            label="Output size",
                            choices=[
                                ("Keep source", "keep"), ("720p height", "720p"), ("1080p height", "1080p"),
                                ("2× Lanczos", "2x"), ("Custom", "custom"),
                            ],
                            value=initial.scale,
                        )
                        with gr.Group(visible=initial.scale == "custom") as custom_resize_panel:
                            with gr.Row():
                                custom_width = gr.Number(value=initial.custom_width, minimum=0, precision=0, label="Width (0 = auto)")
                                custom_height = gr.Number(value=initial.custom_height, minimum=0, precision=0, label="Height (0 = auto)")
                                keep_aspect = gr.Checkbox(value=initial.keep_aspect, label="Keep aspect")

                    with gr.Group(visible="retime" in initial_stages, elem_classes=["aiwf-stage-card"]) as retime_panel:
                        gr.Markdown("#### Frame-rate conversion")
                        with gr.Row():
                            target_fps = gr.Number(label="Target FPS", value=initial.target_fps or 30, minimum=1, maximum=120)
                            motion_interpolation = gr.Checkbox(
                                label="Motion interpolation",
                                value=initial.motion_interpolation,
                                info="FFmpeg MCI is CPU-heavy; chunked RIFE remains the higher-quality route.",
                            )

                    with gr.Group(visible="audio_cleanup" in initial_stages, elem_classes=["aiwf-stage-card"]) as audio_cleanup_panel:
                        gr.Markdown("#### Audio cleanup")
                        with gr.Row():
                            audio_highpass = gr.Slider(10, 2000, value=initial.audio_highpass_hz, step=1, label="High-pass Hz")
                            audio_lowpass = gr.Slider(1000, 24000, value=initial.audio_lowpass_hz, step=10, label="Low-pass Hz")
                        with gr.Row():
                            audio_noise_reduction = gr.Slider(0.01, 97, value=initial.audio_noise_reduction_db, step=0.1, label="Noise reduction dB")
                            audio_noise_floor = gr.Slider(-80, -20, value=initial.audio_noise_floor_db, step=1, label="Noise floor dB")
                        with gr.Row():
                            audio_noise_type = gr.Dropdown(
                                label="Noise profile", choices=[("White / broadband", "white"), ("Vinyl", "vinyl"), ("Shellac", "shellac")], value=initial.audio_noise_type
                            )
                            audio_track_noise = gr.Checkbox(value=initial.audio_track_noise, label="Track changing noise floor")

                    with gr.Group(visible="audio_normalize" in initial_stages, elem_classes=["aiwf-stage-card"]) as normalize_panel:
                        gr.Markdown("#### Loudness normalize")
                        with gr.Row():
                            audio_target_lufs = gr.Slider(-36, -5, value=initial.audio_target_lufs, step=0.1, label="Target LUFS")
                            audio_true_peak = gr.Slider(-9, 0, value=initial.audio_true_peak_db, step=0.1, label="True peak dBTP")
                            audio_lra = gr.Slider(1, 30, value=initial.audio_lra, step=0.5, label="Loudness range")

                    with gr.Group(visible=True, elem_classes=["aiwf-stage-card"]) as export_panel:
                        gr.Markdown("#### Export")
                        with gr.Row():
                            codec = gr.Dropdown(
                                label="Encoder",
                                choices=[
                                    ("Auto (NVENC when available)", "auto"), ("H.264 software", "h264"),
                                    ("H.265 software", "hevc"), ("H.264 NVENC", "h264_nvenc"), ("H.265 NVENC", "hevc_nvenc"),
                                ],
                                value=initial.codec,
                            )
                            container = gr.Dropdown(
                                label="Container", choices=[("MP4", "mp4"), ("MKV (keeps subtitles)", "mkv")], value=initial.container
                            )
                        with gr.Row():
                            quality = gr.Slider(14, 36, value=initial.quality, step=1, label="Video quality")
                            audio_bitrate = gr.Slider(64, 512, value=initial.audio_bitrate_kbps, step=16, label="Audio bitrate kbps")

                    with gr.Row():
                        process_button = gr.Button("Run Video Lab", variant="primary", elem_classes=["aiwf-generate-btn"])
                        cancel_button = gr.Button("Cancel", variant="stop", elem_classes=["aiwf-btn-stop"])

                with gr.Column(scale=6, min_width=420, elem_classes=["aiwf-panel", "aiwf-lab-output"]):
                    output = gr.Video(label="Processed output", interactive=False)
                    status = gr.Markdown("**Ready** — one Video Lab job can run at a time.", elem_classes=["aiwf-status-bar"])
                    plan_view = gr.Code(label="Resolved job plan", language="json", interactive=False, lines=18)
                    with gr.Accordion("Source metadata", open=False):
                        metadata_view = gr.Code(language="json", interactive=False, lines=12, label="ffprobe")
                    with gr.Accordion("Job files and diagnostics", open=False):
                        manifest_file = gr.File(label="Job manifest", interactive=False)
                        log_view = gr.Textbox(label="Last log / result", lines=8, interactive=False)

        panels = [
            timeline_panel, deinterlace_panel, stabilize_panel, deflicker_panel, denoise_panel,
            sharpen_panel, resize_panel, retime_panel, audio_cleanup_panel, normalize_panel, export_panel,
        ]

        def _stage_visibility(selected):
            normalized = normalize_video_stages(selected)
            visible = set(normalized)
            return (
                gr.update(value=normalized),
                video_order_markdown(normalized),
                *[gr.update(visible=stage in visible) for stage in VIDEO_STAGE_ORDER],
            )

        stages.change(_stage_visibility, inputs=[stages], outputs=[stages, order_view, *panels], show_progress=False)
        scale.change(
            lambda value: gr.update(visible=str(value or "") == "custom"),
            inputs=[scale], outputs=[custom_resize_panel], show_progress=False,
        )
        denoise_profile.change(
            lambda value: _denoise_profile_values(value),
            inputs=[denoise_profile],
            outputs=[denoise_luma_spatial, denoise_chroma_spatial, denoise_luma_temporal, denoise_chroma_temporal],
            show_progress=False,
        )
        sharpen_profile.change(
            lambda value: _sharpen_profile_values(value),
            inputs=[sharpen_profile], outputs=[sharpen_kernel, sharpen_amount], show_progress=False,
        )

        controls = [
            stages, preset, trim_start, trim_end,
            deinterlace_mode, deinterlace_parity, deinterlace_scope,
            stabilize_radius_x, stabilize_radius_y, stabilize_edge, stabilize_block_size, stabilize_contrast,
            deflicker_size, deflicker_mode,
            denoise_profile, denoise_luma_spatial, denoise_chroma_spatial, denoise_luma_temporal, denoise_chroma_temporal,
            sharpen_profile, sharpen_kernel, sharpen_amount,
            scale, custom_width, custom_height, keep_aspect,
            target_fps, motion_interpolation,
            audio_highpass, audio_lowpass, audio_noise_reduction, audio_noise_floor, audio_noise_type, audio_track_noise,
            audio_target_lufs, audio_true_peak, audio_lra,
            codec, container, quality, audio_bitrate,
        ]

        def _inspect(video):
            path = _video_path(video)
            if not path:
                raise gr.Error("Upload an input video first.")
            try:
                probe = service.inspect(path)
            except Exception as exc:
                raise gr.Error(str(exc)) from exc
            return _probe_summary(probe), json.dumps(probe.model_dump(mode="json"), indent=2)

        preset_controls = [
            stages, trim_start, trim_end,
            deinterlace_mode, deinterlace_parity, deinterlace_scope,
            stabilize_radius_x, stabilize_radius_y, stabilize_edge, stabilize_block_size, stabilize_contrast,
            deflicker_size, deflicker_mode,
            denoise_profile, denoise_luma_spatial, denoise_chroma_spatial, denoise_luma_temporal, denoise_chroma_temporal,
            sharpen_profile, sharpen_kernel, sharpen_amount,
            scale, custom_width, custom_height, keep_aspect, custom_resize_panel,
            target_fps, motion_interpolation,
            audio_highpass, audio_lowpass, audio_noise_reduction, audio_noise_floor, audio_noise_type, audio_track_noise,
            audio_target_lufs, audio_true_peak, audio_lra,
            codec, container, quality, audio_bitrate,
            order_view, *panels,
        ]

        def _apply_preset(name):
            settings = preset_settings(str(name or "custom"))
            selected = video_stages_from_settings(settings)
            selected_set = set(selected)
            return (
                selected, settings.trim_start, settings.trim_end or 0,
                settings.deinterlace_mode, settings.deinterlace_parity, settings.deinterlace_scope,
                settings.stabilize_radius_x, settings.stabilize_radius_y, settings.stabilize_edge,
                settings.stabilize_block_size, settings.stabilize_contrast,
                settings.deflicker_size, settings.deflicker_mode,
                settings.denoise if settings.denoise != "off" else "light",
                settings.denoise_luma_spatial, settings.denoise_chroma_spatial,
                settings.denoise_luma_temporal, settings.denoise_chroma_temporal,
                settings.sharpen if settings.sharpen != "off" else "light",
                settings.sharpen_kernel, settings.sharpen_amount,
                settings.scale, settings.custom_width, settings.custom_height, settings.keep_aspect,
                gr.update(visible=settings.scale == "custom"),
                settings.target_fps or 30, settings.motion_interpolation,
                settings.audio_highpass_hz, settings.audio_lowpass_hz,
                settings.audio_noise_reduction_db, settings.audio_noise_floor_db,
                settings.audio_noise_type, settings.audio_track_noise,
                settings.audio_target_lufs, settings.audio_true_peak_db, settings.audio_lra,
                settings.codec, settings.container, settings.quality, settings.audio_bitrate_kbps,
                video_order_markdown(selected),
                *[gr.update(visible=stage in selected_set) for stage in VIDEO_STAGE_ORDER],
            )

        apply_preset.click(_apply_preset, inputs=[preset], outputs=preset_controls, show_progress=False)

        def _plan(video, *values):
            path = _video_path(video)
            if not path:
                raise gr.Error("Upload an input video first.")
            try:
                settings = _settings_from_values(*values)
                plan = service.build_plan(path, settings)
            except Exception as exc:
                raise gr.Error(str(exc)) from exc
            warning = f" · {len(plan.warnings)} warning(s)" if plan.warnings else ""
            return (
                service.plan_text(plan),
                f"**Plan ready** — `{plan.selected_codec}`{warning}",
                video_order_markdown(video_stages_from_settings(settings), plan.warnings),
            )

        def _run(video, *values):
            path = _video_path(video)
            if not path:
                raise gr.Error("Upload an input video first.")
            try:
                settings = _settings_from_values(*values)
                plan = service.build_plan(path, settings)
            except Exception as exc:
                raise gr.Error(str(exc)) from exc

            events: queue.Queue[tuple[float, str]] = queue.Queue()
            result_box: dict[str, object] = {}

            def _worker() -> None:
                try:
                    result_box["result"] = service.execute(
                        plan, on_progress=lambda percent, message: events.put((percent, message))
                    )
                except Exception as exc:
                    result_box["error"] = exc

            thread = threading.Thread(target=_worker, name=f"aiwf-{plan.job_id}", daemon=True)
            thread.start()
            plan_text = service.plan_text(plan)
            yield None, f"**Running** — `{plan.job_id}`", plan_text, None, "FFmpeg process started."

            last_message = "Processing"
            while thread.is_alive():
                try:
                    percent, last_message = events.get(timeout=0.35)
                except queue.Empty:
                    continue
                yield gr.update(), f"**Running {percent * 100:.0f}%** — {last_message}", gr.update(), gr.update(), last_message
            thread.join()

            error = result_box.get("error")
            if error is not None:
                if isinstance(error, VideoLabCancelled):
                    yield None, "**Cancelled** — partial output was removed.", plan_text, str(Path(plan.output_path).parent / "job.json"), str(error)
                    return
                if isinstance(error, VideoLabBusy):
                    yield None, f"**Busy** — {error}", plan_text, None, str(error)
                    return
                message = str(error)
                yield None, f"**Failed** — {message.splitlines()[0]}", plan_text, str(Path(plan.output_path).parent / "job.json"), message
                return

            result = result_box["result"]
            warnings = "\n".join(f"Warning: {item}" for item in result.warnings)
            details = result.message + ("\n" + warnings if warnings else "")
            yield result.output_path, f"**Done** — {result.message}", plan_text, result.manifest_path, details

        def _cancel():
            return f"**Cancel** — {service.cancel_active()}"

        inspect_button.click(_inspect, inputs=[source], outputs=[probe_summary, metadata_view])
        plan_button.click(_plan, inputs=[source, *controls], outputs=[plan_view, status, order_view])
        process_button.click(
            _run,
            inputs=[source, *controls],
            outputs=[output, status, plan_view, manifest_file, log_view],
            concurrency_limit=1,
            concurrency_id="aiwf-video-lab",
        )
        cancel_button.click(_cancel, outputs=[status], queue=False)
