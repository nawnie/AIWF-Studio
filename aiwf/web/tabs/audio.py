from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from aiwf.bootstrap import AppContext
from aiwf.core.domain.audio import AudioGenerationOptions
from aiwf.core.domain.audio_lab import AUDIO_LAB_ORDER, AUDIO_STAGE_LABELS, AudioLabSettings
from aiwf.services.audio import AudioGenerationService, AudioUnavailable
from aiwf.services.audio_lab import AudioLabService, parse_audio_command, preset_audio_settings
from aiwf.web.registry import WebRegistry

_GENERATION_SERVICES: dict[int, AudioGenerationService] = {}
_LAB_SERVICES: dict[int, AudioLabService] = {}


def _generation_service(ctx: AppContext) -> AudioGenerationService:
    svc = _GENERATION_SERVICES.get(id(ctx))
    if svc is None:
        svc = AudioGenerationService(
            ctx.flags,
            ctx.settings,
            ctx.generation.backend.devices,
            supervisor=ctx.supervisor,
        )
        _GENERATION_SERVICES[id(ctx)] = svc
    return svc


def _lab_service(ctx: AppContext) -> AudioLabService:
    svc = _LAB_SERVICES.get(id(ctx))
    if svc is None:
        svc = AudioLabService(ctx.flags.resolved_output_dir())
        _LAB_SERVICES[id(ctx)] = svc
    return svc


def _file_path(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("path") or value.get("name") or "")
    return str(getattr(value, "path", None) or getattr(value, "name", None) or "")


def normalize_audio_stages(values) -> list[str]:
    selected = {str(item) for item in (values or [])}
    result = [stage for stage in AUDIO_LAB_ORDER if stage in selected]
    if "export" not in result:
        result.append("export")
    return result


def audio_order_markdown(stages, warnings: list[str] | None = None) -> str:
    normalized = normalize_audio_stages(stages)
    flow = " → ".join(AUDIO_STAGE_LABELS[item] for item in normalized)
    extra = "" if not warnings else "  \n" + "  \n".join(f"⚠ {item}" for item in warnings)
    return f"**Resolved order:** {flow}{extra}"


def _engine_markdown(status) -> str:
    icon = "✅" if status.installed else "⚙️"
    python = f"  \nPython: `{status.python_path}`" if status.python_path else ""
    return f"{icon} **{status.message}**{python}"


def _build_mix_panel(ctx: AppContext, service: AudioLabService) -> None:
    initial = preset_audio_settings("music_sweeten")
    initial_stages = normalize_audio_stages(initial.stages)

    with gr.Row(equal_height=False, elem_classes=["aiwf-lab-workspace"]):
        with gr.Column(scale=5, min_width=430, elem_classes=["aiwf-panel", "aiwf-lab-controls"]):
            source = gr.Audio(label="Input audio", type="filepath", sources=["upload"])
            with gr.Row():
                inspect_button = gr.Button("Inspect", elem_classes=["aiwf-btn-ghost"])
                plan_button = gr.Button("Build plan", elem_classes=["aiwf-btn-ghost"])
            with gr.Row():
                preset = gr.Dropdown(
                    label="Starting preset",
                    choices=[
                        ("Music sweeten", "music_sweeten"),
                        ("Podcast cleanup", "podcast_cleanup"),
                        ("Old recording", "old_recording"),
                        ("Custom", "custom"),
                    ],
                    value="music_sweeten",
                )
                apply_preset = gr.Button("Apply preset", elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"])

            stages = gr.CheckboxGroup(
                label="Processes to run",
                choices=[(AUDIO_STAGE_LABELS[item], item) for item in AUDIO_LAB_ORDER],
                value=initial_stages,
                info="Pick the processing. Audio Lab resolves a non-destructive signal-chain order.",
                elem_classes=["aiwf-stage-picker"],
            )
            order_view = gr.Markdown(audio_order_markdown(initial_stages), elem_classes=["aiwf-stage-order"])

            with gr.Group(visible="trim" in initial_stages, elem_classes=["aiwf-stage-card"]) as trim_panel:
                gr.Markdown("#### Trim")
                with gr.Row():
                    trim_start = gr.Number(value=0, minimum=0, label="Start (seconds)")
                    trim_end = gr.Number(value=0, minimum=0, label="End (0 = source end)")

            with gr.Group(visible="gate" in initial_stages, elem_classes=["aiwf-stage-card"]) as gate_panel:
                gr.Markdown("#### Noise gate")
                with gr.Row():
                    gate_threshold = gr.Slider(-100, 0, value=initial.gate_threshold_db, step=1, label="Threshold dB")
                    gate_ratio = gr.Slider(1, 20, value=initial.gate_ratio, step=0.1, label="Ratio")
                with gr.Row():
                    gate_attack = gr.Slider(0.1, 500, value=initial.gate_attack_ms, step=0.1, label="Attack ms")
                    gate_release = gr.Slider(1, 3000, value=initial.gate_release_ms, step=1, label="Release ms")

            with gr.Group(visible="filters" in initial_stages, elem_classes=["aiwf-stage-card"]) as filters_panel:
                gr.Markdown("#### High / low pass")
                with gr.Row():
                    highpass = gr.Slider(10, 2000, value=initial.highpass_hz, step=1, label="High-pass Hz")
                    lowpass = gr.Slider(1000, 24000, value=initial.lowpass_hz, step=10, label="Low-pass Hz")

            with gr.Group(visible="eq" in initial_stages, elem_classes=["aiwf-stage-card"]) as eq_panel:
                gr.Markdown("#### Three-band parametric EQ")
                with gr.Row():
                    low_hz = gr.Slider(20, 1000, value=initial.low_shelf_hz, step=1, label="Low shelf Hz")
                    low_gain = gr.Slider(-18, 18, value=initial.low_shelf_gain_db, step=0.1, label="Low gain dB")
                with gr.Row():
                    mid_hz = gr.Slider(100, 12000, value=initial.mid_hz, step=10, label="Mid center Hz")
                    mid_gain = gr.Slider(-18, 18, value=initial.mid_gain_db, step=0.1, label="Mid gain dB")
                    mid_q = gr.Slider(0.1, 10, value=initial.mid_q, step=0.1, label="Mid Q")
                with gr.Row():
                    high_hz = gr.Slider(1000, 20000, value=initial.high_shelf_hz, step=10, label="High shelf Hz")
                    high_gain = gr.Slider(-18, 18, value=initial.high_shelf_gain_db, step=0.1, label="High gain dB")

            with gr.Group(visible="compressor" in initial_stages, elem_classes=["aiwf-stage-card"]) as compressor_panel:
                gr.Markdown("#### Compressor")
                with gr.Row():
                    comp_threshold = gr.Slider(-60, 0, value=initial.compressor_threshold_db, step=1, label="Threshold dB")
                    comp_ratio = gr.Slider(1, 20, value=initial.compressor_ratio, step=0.1, label="Ratio")
                with gr.Row():
                    comp_attack = gr.Slider(0.1, 500, value=initial.compressor_attack_ms, step=0.1, label="Attack ms")
                    comp_release = gr.Slider(1, 3000, value=initial.compressor_release_ms, step=1, label="Release ms")

            with gr.Group(visible="pitch" in initial_stages, elem_classes=["aiwf-stage-card"]) as pitch_panel:
                gr.Markdown("#### Pitch shift")
                pitch_semitones = gr.Slider(-24, 24, value=0, step=0.1, label="Semitones")
                with gr.Row():
                    pitch_start = gr.Number(value=0, minimum=0, label="Region start seconds")
                    pitch_end = gr.Number(value=0, minimum=0, label="Region end (0 = clip end)")

            with gr.Group(visible="gain" in initial_stages, elem_classes=["aiwf-stage-card"]) as gain_panel:
                gr.Markdown("#### Gain")
                gain_db = gr.Slider(-36, 24, value=initial.gain_db, step=0.1, label="Gain dB")

            with gr.Group(visible="pan" in initial_stages, elem_classes=["aiwf-stage-card"]) as pan_panel:
                gr.Markdown("#### Pan")
                pan = gr.Slider(-1, 1, value=0, step=0.01, label="Left ← Pan → Right")

            with gr.Group(visible="envelope" in initial_stages, elem_classes=["aiwf-stage-card"]) as envelope_panel:
                gr.Markdown("#### Automation / fades")
                with gr.Row():
                    fade_in = gr.Number(value=0, minimum=0, label="Fade in seconds")
                    fade_out = gr.Number(value=0, minimum=0, label="Fade out seconds")
                gain_envelope = gr.Textbox(
                    label="Gain envelope",
                    placeholder="0:0, 12:-3, 18:0  (seconds:dB)",
                    info="Linear automation between points.",
                )

            with gr.Group(visible="normalize" in initial_stages, elem_classes=["aiwf-stage-card"]) as normalize_panel:
                gr.Markdown("#### Loudness normalize")
                target_lufs = gr.Slider(-36, -5, value=initial.target_lufs, step=0.1, label="Target LUFS")

            with gr.Group(visible="limiter" in initial_stages, elem_classes=["aiwf-stage-card"]) as limiter_panel:
                gr.Markdown("#### Limiter")
                with gr.Row():
                    limiter_threshold = gr.Slider(-20, 0, value=initial.limiter_threshold_db, step=0.1, label="Threshold dB")
                    limiter_release = gr.Slider(1, 3000, value=initial.limiter_release_ms, step=1, label="Release ms")

            with gr.Group(visible=True, elem_classes=["aiwf-stage-card"]) as export_panel:
                gr.Markdown("#### Export")
                with gr.Row():
                    export_format = gr.Dropdown(label="Format", choices=[("WAV 24-bit", "wav"), ("FLAC 24-bit", "flac")], value="wav")
                    sample_rate = gr.Dropdown(
                        label="Sample rate",
                        choices=[("Keep source", 0), ("44.1 kHz", 44100), ("48 kHz", 48000), ("96 kHz", 96000)],
                        value=0,
                    )

            run_button = gr.Button("Run Audio Lab", variant="primary", elem_classes=["aiwf-generate-btn"])

        with gr.Column(scale=6, min_width=420, elem_classes=["aiwf-panel", "aiwf-lab-output"]):
            output = gr.Audio(label="Processed audio", type="filepath", interactive=False)
            status = gr.Markdown("**Ready** — install the optional engine from the Engine tab before processing.", elem_classes=["aiwf-status-bar"])
            plan_json = gr.Code(label="Resolved job plan", language="json", lines=15, interactive=False)
            metadata_json = gr.Code(label="Input metadata", language="json", lines=12, interactive=False)
            manifest = gr.File(label="Job manifest", interactive=False)
            stage_log = gr.Textbox(label="Stage log", lines=8, interactive=False)

    panels = [
        trim_panel,
        gate_panel,
        filters_panel,
        eq_panel,
        compressor_panel,
        pitch_panel,
        gain_panel,
        pan_panel,
        envelope_panel,
        normalize_panel,
        limiter_panel,
        export_panel,
    ]

    def _stage_visibility(selected):
        normalized = normalize_audio_stages(selected)
        visible = set(normalized)
        return (
            gr.update(value=normalized),
            audio_order_markdown(normalized),
            *[gr.update(visible=stage in visible) for stage in AUDIO_LAB_ORDER],
        )

    stages.change(_stage_visibility, inputs=[stages], outputs=[stages, order_view, *panels], show_progress=False)

    preset_outputs = [
        stages,
        gate_threshold,
        gate_ratio,
        highpass,
        lowpass,
        low_gain,
        mid_hz,
        mid_gain,
        high_gain,
        comp_threshold,
        comp_ratio,
        target_lufs,
        limiter_threshold,
        order_view,
        *panels,
    ]

    def _apply_preset(name):
        settings = preset_audio_settings(str(name or "custom"))
        selected = normalize_audio_stages(settings.stages)
        visible = set(selected)
        return (
            selected,
            settings.gate_threshold_db,
            settings.gate_ratio,
            settings.highpass_hz,
            settings.lowpass_hz,
            settings.low_shelf_gain_db,
            settings.mid_hz,
            settings.mid_gain_db,
            settings.high_shelf_gain_db,
            settings.compressor_threshold_db,
            settings.compressor_ratio,
            settings.target_lufs,
            settings.limiter_threshold_db,
            audio_order_markdown(selected),
            *[gr.update(visible=stage in visible) for stage in AUDIO_LAB_ORDER],
        )

    apply_preset.click(_apply_preset, inputs=[preset], outputs=preset_outputs, show_progress=False)

    controls = [
        stages,
        preset,
        trim_start,
        trim_end,
        gate_threshold,
        gate_ratio,
        gate_attack,
        gate_release,
        highpass,
        lowpass,
        low_hz,
        low_gain,
        mid_hz,
        mid_gain,
        mid_q,
        high_hz,
        high_gain,
        comp_threshold,
        comp_ratio,
        comp_attack,
        comp_release,
        pitch_semitones,
        pitch_start,
        pitch_end,
        gain_db,
        pan,
        fade_in,
        fade_out,
        gain_envelope,
        target_lufs,
        limiter_threshold,
        limiter_release,
        export_format,
        sample_rate,
    ]

    def _settings(*values) -> AudioLabSettings:
        (
            selected,
            preset_value,
            trim_start_value,
            trim_end_value,
            gate_threshold_value,
            gate_ratio_value,
            gate_attack_value,
            gate_release_value,
            highpass_value,
            lowpass_value,
            low_hz_value,
            low_gain_value,
            mid_hz_value,
            mid_gain_value,
            mid_q_value,
            high_hz_value,
            high_gain_value,
            comp_threshold_value,
            comp_ratio_value,
            comp_attack_value,
            comp_release_value,
            pitch_semitones_value,
            pitch_start_value,
            pitch_end_value,
            gain_value,
            pan_value,
            fade_in_value,
            fade_out_value,
            envelope_value,
            target_lufs_value,
            limiter_threshold_value,
            limiter_release_value,
            export_format_value,
            sample_rate_value,
        ) = values
        trim_end_number = float(trim_end_value or 0)
        pitch_end_number = float(pitch_end_value or 0)
        return AudioLabSettings(
            stages=normalize_audio_stages(selected),
            preset=str(preset_value or "custom"),
            trim_start_seconds=float(trim_start_value or 0),
            trim_end_seconds=None if trim_end_number <= 0 else trim_end_number,
            gate_threshold_db=float(gate_threshold_value),
            gate_ratio=float(gate_ratio_value),
            gate_attack_ms=float(gate_attack_value),
            gate_release_ms=float(gate_release_value),
            highpass_hz=float(highpass_value),
            lowpass_hz=float(lowpass_value),
            low_shelf_hz=float(low_hz_value),
            low_shelf_gain_db=float(low_gain_value),
            mid_hz=float(mid_hz_value),
            mid_gain_db=float(mid_gain_value),
            mid_q=float(mid_q_value),
            high_shelf_hz=float(high_hz_value),
            high_shelf_gain_db=float(high_gain_value),
            compressor_threshold_db=float(comp_threshold_value),
            compressor_ratio=float(comp_ratio_value),
            compressor_attack_ms=float(comp_attack_value),
            compressor_release_ms=float(comp_release_value),
            pitch_semitones=float(pitch_semitones_value),
            pitch_start_seconds=float(pitch_start_value or 0),
            pitch_end_seconds=None if pitch_end_number <= 0 else pitch_end_number,
            gain_db=float(gain_value),
            pan=float(pan_value),
            fade_in_seconds=float(fade_in_value or 0),
            fade_out_seconds=float(fade_out_value or 0),
            gain_envelope=str(envelope_value or ""),
            target_lufs=float(target_lufs_value),
            limiter_threshold_db=float(limiter_threshold_value),
            limiter_release_ms=float(limiter_release_value),
            export_format=str(export_format_value or "wav"),
            sample_rate=int(sample_rate_value or 0),
        )

    def _inspect(source_value):
        path = _file_path(source_value)
        if not path:
            raise gr.Error("Upload an audio file first.")
        try:
            metadata = service.inspect_audio(path)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc
        return json.dumps(metadata, indent=2), f"**Inspected** — {metadata['duration_seconds']:.2f}s · {metadata['sample_rate']} Hz · {metadata['channels']} channel(s)"

    def _plan(*values):
        try:
            settings = _settings(*values)
            plan = service.build_plan(settings)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc
        payload = {
            "resolved_order": plan.stages,
            "labels": plan.labels,
            "warnings": plan.warnings,
            "settings": settings.model_dump(mode="json"),
        }
        return json.dumps(payload, indent=2), audio_order_markdown(plan.stages, plan.warnings)

    def _run(source_value, *values):
        path = _file_path(source_value)
        if not path:
            raise gr.Error("Upload an audio file first.")
        try:
            settings = _settings(*values)
            plan = service.build_plan(settings)
            result = service.process(path, settings)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc
        payload = {
            "resolved_order": plan.stages,
            "labels": plan.labels,
            "warnings": plan.warnings,
            "settings": settings.model_dump(mode="json"),
        }
        return (
            result["output_path"],
            f"**Done** — {result['duration_seconds']:.2f}s in {result['elapsed_seconds']:.2f}s.",
            json.dumps(payload, indent=2),
            result["manifest_path"],
            "\n".join(result.get("stage_log") or []),
        )

    inspect_button.click(_inspect, inputs=[source], outputs=[metadata_json, status])
    plan_button.click(_plan, inputs=controls, outputs=[plan_json, order_view])
    run_button.click(
        _run,
        inputs=[source, *controls],
        outputs=[output, status, plan_json, manifest, stage_log],
        concurrency_limit=1,
        concurrency_id="aiwf-audio-lab",
        show_progress="full",
    )


def _build_generate_panel(ctx: AppContext, service: AudioGenerationService) -> None:
    music_models = service.music_model_choices()
    sfx_models = service.sfx_model_choices()
    video_audio_models = service.video_audio_model_choices()

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=340, elem_classes=["aiwf-panel"]):
            prompt = gr.Textbox(
                label="Audio prompt",
                lines=4,
                placeholder="cinematic ambient score, soft percussion, warm synth pads",
            )
            source_video = gr.Video(label="Optional target video", sources=["upload"], interactive=True)
            with gr.Row():
                kind = gr.Radio(
                    label="Type",
                    choices=[
                        ("Video-conditioned audio", "video_audio"),
                        ("Music", "music"),
                        ("Sound effects", "sfx"),
                    ],
                    value="video_audio",
                )
                model = gr.Dropdown(
                    label="Model",
                    choices=video_audio_models,
                    value=video_audio_models[0][1] if video_audio_models else "mmaudio:large_44k_v2",
                    allow_custom_value=True,
                )
            with gr.Row():
                duration = gr.Slider(1, 120, value=8, step=1, label="Duration (seconds)")
                seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
            with gr.Row():
                temperature = gr.Slider(0.1, 2.0, value=1.0, step=0.05, label="Temperature")
                cfg = gr.Slider(0.1, 10.0, value=3.0, step=0.1, label="Guidance")
            generate = gr.Button("Generate audio", variant="primary", elem_classes=["aiwf-generate-btn"])

        with gr.Column(scale=1, min_width=340, elem_classes=["aiwf-panel"]):
            audio_out = gr.Audio(label="Generated audio", type="filepath", interactive=False)
            video_out = gr.Video(label="Video with audio", interactive=False)
            status = gr.Markdown("**Ready**", elem_classes=["aiwf-status-bar"])
            details = gr.Textbox(label="Details", lines=4, interactive=False, elem_classes=["aiwf-gen-info"])

    def _sync_kind(kind_value):
        selected = str(kind_value or "video_audio")
        if selected == "sfx":
            choices = sfx_models
            fallback = "facebook/audiogen-medium"
        elif selected == "music":
            choices = music_models
            fallback = "facebook/musicgen-small"
        else:
            choices = video_audio_models
            fallback = "mmaudio:large_44k_v2"
        return gr.update(choices=choices, value=choices[0][1] if choices else fallback)

    kind.change(_sync_kind, inputs=[kind], outputs=[model], show_progress=False)

    def _run(prompt_v, video_v, kind_v, model_v, duration_v, seed_v, temperature_v, cfg_v):
        video_path = _file_path(video_v) if video_v is not None else None
        options = AudioGenerationOptions(
            prompt=prompt_v or "",
            kind=str(kind_v or "music"),
            model_id=str(
                model_v
                or (
                    "facebook/audiogen-medium"
                    if kind_v == "sfx"
                    else "facebook/musicgen-small"
                    if kind_v == "music"
                    else "mmaudio:large_44k_v2"
                )
            ),
            duration_seconds=float(duration_v or 8),
            temperature=float(temperature_v or 1.0),
            cfg_coef=float(cfg_v or 3.0),
            seed=int(seed_v if seed_v is not None else -1),
        )
        try:
            if video_path:
                audio, muxed = service.generate_and_mux(video_path, options, duration_seconds=float(duration_v or 8))
                return audio.output_path, muxed.output_path, "**Audio mux complete.**", audio.infotext + "\n" + muxed.infotext
            audio = service.generate(options)
            return audio.output_path, None, "**Audio complete.**", audio.infotext
        except AudioUnavailable as exc:
            raise gr.Error(str(exc)) from exc

    generate.click(
        _run,
        inputs=[prompt, source_video, kind, model, duration, seed, temperature, cfg],
        outputs=[audio_out, video_out, status, details],
        show_progress="minimal",
    )


def _build_project_panel(service: AudioLabService) -> None:
    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=360, elem_classes=["aiwf-panel"]):
            midi_file = gr.File(label="MIDI file", file_types=[".mid", ".midi"])
            inspect_midi = gr.Button("Inspect MIDI metadata", elem_classes=["aiwf-btn-ghost"])
            command = gr.Textbox(
                label="DAW command planner",
                lines=5,
                placeholder=(
                    "Add a Cello track in unison of trumpet track 2 one octave below at 60% velocity, "
                    "fade out four beats before the end, pan right"
                ),
            )
            plan_command = gr.Button("Preview command structure", variant="primary")
            gr.Markdown(
                "This v5 slice parses intent and inspects MIDI metadata. It does **not** yet execute destructive "
                "multitrack arrangement commands. The engine needs a tempo map, track IDs, regions, and note events first.",
                elem_classes=["aiwf-stage-note"],
            )
        with gr.Column(scale=1, min_width=420, elem_classes=["aiwf-panel"]):
            midi_metadata = gr.Code(label="MIDI project metadata", language="json", lines=18, interactive=False)
            command_plan = gr.Code(label="Structured command plan", language="json", lines=16, interactive=False)
            status = gr.Markdown("**Project planner ready**", elem_classes=["aiwf-status-bar"])

    def _inspect(file_value):
        path = _file_path(file_value)
        if not path:
            raise gr.Error("Choose a MIDI file first.")
        try:
            metadata = service.inspect_midi(path)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc
        return json.dumps(metadata, indent=2), f"**Inspected** — {len(metadata.get('tracks') or [])} track(s)."

    def _plan(text):
        result = parse_audio_command(text)
        return json.dumps(result.model_dump(mode="json"), indent=2), (
            "**Command understood**" if result.understood else "**Command needs clarification or a future grammar rule**"
        )

    inspect_midi.click(_inspect, inputs=[midi_file], outputs=[midi_metadata, status])
    plan_command.click(_plan, inputs=[command], outputs=[command_plan, status], show_progress=False)


def _build_engine_panel(service: AudioLabService) -> None:
    current = service.status(deep=False)
    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=360, elem_classes=["aiwf-panel"]):
            gr.Markdown("### Optional isolated Audio Lab engine")
            gr.Markdown(
                "Mixing runs in `engines/audio_lab/.venv`, separate from the image/video environment. "
                "The core install adds deterministic audio and MIDI libraries, not a second CUDA stack."
            )
            install = gr.Button("Install / Repair Audio Lab Engine", variant="primary")
            refresh = gr.Button("Refresh status", elem_classes=["aiwf-btn-ghost"])
        with gr.Column(scale=1, min_width=420, elem_classes=["aiwf-panel"]):
            status = gr.Markdown(_engine_markdown(current), elem_classes=["aiwf-status-bar"])
            details = gr.Code(
                label="Engine details",
                language="json",
                value=json.dumps(current.model_dump(mode="json"), indent=2),
                lines=18,
                interactive=False,
            )

    def _refresh():
        result = service.status(deep=True)
        return _engine_markdown(result), json.dumps(result.model_dump(mode="json"), indent=2)

    def _install():
        try:
            log = service.install(upgrade=False)
            result = service.status(deep=True)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc
        return _engine_markdown(result), log

    refresh.click(_refresh, outputs=[status, details], show_progress=False)
    install.click(_install, outputs=[status, details], show_progress="full")


def register_audio(registry: WebRegistry) -> None:
    @registry.tab("Audio Lab", order=24)
    def build(ctx: AppContext, tab: gr.Tab | None = None) -> None:
        generation = _generation_service(ctx)
        lab = _lab_service(ctx)
        with gr.Column(elem_classes=["aiwf-audio", "aiwf-video"]):
            with gr.Column(elem_classes=["aiwf-page-header"]):
                gr.Markdown("AUDIO LAB", elem_classes=["aiwf-section-label"])
                gr.Markdown(
                    "Sweeten and mix existing work, inspect MIDI project metadata, or generate local audio. "
                    "The mixing engine is optional and isolated.",
                    elem_classes=["aiwf-page-intro"],
                )
                gr.Markdown(generation.folder_help(), elem_classes=["aiwf-page-path"])

            with gr.Tabs(elem_classes=["aiwf-enhance-tabs"]):
                with gr.Tab("Mix & Sweeten"):
                    _build_mix_panel(ctx, lab)
                with gr.Tab("Generate"):
                    _build_generate_panel(ctx, generation)
                with gr.Tab("Project / MIDI"):
                    _build_project_panel(lab)
                with gr.Tab("Engine"):
                    _build_engine_panel(lab)
