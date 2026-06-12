from __future__ import annotations

import platform
from pathlib import Path

import gradio as gr

from aiwf.bootstrap import AppContext
from aiwf.core.config.launch import LaunchSettings, format_launch_status
from aiwf.core.util.access import build_network_access_info, format_remote_access_markdown
from aiwf.web.registry import PINNED_TABS, WebRegistry
from aiwf.web.theme import accent_preset_names

TAB_VISIBILITY_CHOICES = [
    "Models",
    "Segment",
    "Enhance",
    "Workflows",
    "Face Swap",
    "Library",
    "PNG Info",
    "History",
]


def _remote_access_markdown(ctx: AppContext) -> str:
    port = ctx.runtime_port or ctx.flags.port
    info = build_network_access_info(listen=ctx.flags.listen, port=port)
    return format_remote_access_markdown(info)


def _launch_form_values(ctx: AppContext) -> LaunchSettings:
    saved = ctx.load_launch_settings()
    if saved is not None:
        return saved
    return LaunchSettings.from_runtime_flags(ctx.flags)


def _attention_summary(ctx: AppContext) -> str:
    if ctx.flags.xformers:
        return "xFormers"
    if ctx.flags.opt_sdp_attention or ctx.flags.opt_split_attention:
        return "SDP attention"
    return "Default attention"


def _format_gb(value: int | float | None) -> str:
    if not value:
        return "n/a"
    return f"{float(value) / (1024 ** 3):.1f} GB"


def _session_snapshot_markdown(ctx: AppContext) -> str:
    try:
        import psutil

        ram_total = _format_gb(psutil.virtual_memory().total)
    except Exception:
        ram_total = "n/a"

    try:
        import torch

        torch_version = torch.__version__
    except Exception:
        torch_version = "unavailable"

    device_summary = ctx.generation.backend.devices.describe()
    launch_theme = _launch_form_values(ctx).theme
    return (
        f"**Python** `{platform.python_version()}`  \n"
        f"**Torch** `{torch_version}`  \n"
        f"**Theme** `{launch_theme}`  \n"
        f"**Attention** `{_attention_summary(ctx)}`  \n"
        f"**System RAM** `{ram_total}`  \n"
        f"**Device** `{device_summary}`"
    )


def _connect_qr(ctx: AppContext):
    info = build_network_access_info(listen=ctx.flags.listen, port=ctx.runtime_port or ctx.flags.port)
    url = info.recommended_phone_url
    if not url:
        return None, (
            "**QR connect unavailable**  \n"
            "Enable `--listen` or connect Tailscale so AIWF has a phone-ready URL to encode."
        )

    try:
        import qrcode
    except Exception:
        return None, (
            f"**QR target ready**  \n`{url}`  \n"
            "Install the `qrcode` package to render a scannable code in Settings."
        )

    qr = qrcode.QRCode(border=2, box_size=6, error_correction=qrcode.constants.ERROR_CORRECT_M)
    qr.add_data(url)
    qr.make(fit=True)
    image = qr.make_image(fill_color="#0c0e14", back_color="white").convert("RGB")
    return image, f"**Scan to connect**  \n`{url}`"


def _security_markdown() -> str:
    return (
        "**Security notes**  \n"
        "- `--listen` exposes the UI to devices that can reach your machine.  \n"
        "- Use `username:password` auth before sharing outside your desk setup.  \n"
        "- Tailscale is the safest remote path here because it stays inside your tailnet.  \n"
        "- Public share links are convenient, but they trade convenience for privacy."
    )


def register_settings(registry: WebRegistry) -> None:
    @registry.tab("Settings", order=90)
    def build(ctx: AppContext, tab: gr.Tab | None = None) -> None:
        launch = _launch_form_values(ctx)
        saved_launch = ctx.load_launch_settings()
        from aiwf.core.domain.models import SCHEDULE_TYPES

        samplers = ctx.generation.list_samplers()
        schedule_label_to_id = {s.label: s.id for s in SCHEDULE_TYPES}
        schedule_id_to_label = {s.id: s.label for s in SCHEDULE_TYPES}
        default_schedule_label = schedule_id_to_label.get(ctx.settings.default_scheduler, "Automatic")
        sampler_label_to_id = {s.label: s.id for s in samplers}
        sampler_id_to_label = {s.id: s.label for s in samplers}
        default_sampler_label = sampler_id_to_label.get(
            ctx.settings.default_sampler, samplers[0].label if samplers else None
        )

        initial_qr, initial_qr_status = _connect_qr(ctx)

        with gr.Column(elem_classes=["aiwf-settings", "aiwf-settings-redesign"]):
            with gr.Column(elem_classes=["aiwf-page-header"]):
                gr.Markdown("Settings", elem_classes=["aiwf-section-label"])
                gr.Markdown(
                    "Run AIWF like a local creative tool, not a pile of launch flags. "
                    "This page separates live workspace preferences from the next-start launch profile.",
                    elem_classes=["aiwf-page-intro"],
                )

            with gr.Row(equal_height=False, elem_classes=["aiwf-settings-hero"]):
                with gr.Column(scale=2, min_width=340, elem_classes=["aiwf-panel", "aiwf-remote-panel"]):
                    gr.Markdown("Remote access", elem_classes=["aiwf-section-label"])
                    gr.Markdown(
                        "Use Tailscale or your local network for phone and tablet access. "
                        "Refresh after changing network state, VPNs, or launch settings.",
                        elem_classes=["aiwf-settings-paths"],
                    )
                    remote_access = gr.Markdown(
                        _remote_access_markdown(ctx),
                        elem_classes=["aiwf-remote-access"],
                    )
                    with gr.Row(elem_classes=["aiwf-settings-actions"]):
                        refresh_network = gr.Button(
                            "Refresh network info",
                            elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"],
                        )
                        refresh_ui_btn = gr.Button(
                            "Refresh UI",
                            elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"],
                        )
                    connect_qr = gr.Image(
                        label="Phone connect QR",
                        value=initial_qr,
                        type="pil",
                        interactive=False,
                        visible=initial_qr is not None,
                        elem_classes=["aiwf-settings-qr"],
                    )
                    qr_status = gr.Markdown(initial_qr_status, elem_classes=["aiwf-settings-hint"])

                with gr.Column(scale=1, min_width=280, elem_classes=["aiwf-panel", "aiwf-settings-snapshot"]):
                    gr.Markdown("Session snapshot", elem_classes=["aiwf-section-label"])
                    session_snapshot = gr.Markdown(
                        _session_snapshot_markdown(ctx),
                        elem_classes=["aiwf-settings-paths"],
                    )
                    launch_status = gr.Markdown(
                        format_launch_status(LaunchSettings.from_runtime_flags(ctx.flags), saved_launch),
                        elem_classes=["aiwf-settings-hint"],
                    )
                    gr.Markdown(
                        f"**Models** `{ctx.flags.resolved_models_dir()}`  \n"
                        f"**Checkpoints** `{ctx.flags.resolved_ckpt_dir()}`  \n"
                        f"**Output** `{ctx.flags.resolved_output_dir()}`",
                        elem_classes=["aiwf-page-path"],
                    )

            with gr.Tabs(elem_classes=["aiwf-settings-tabs"]):
                with gr.Tab("Workspace"):
                    with gr.Row(equal_height=False, elem_classes=["aiwf-settings-grid"]):
                        with gr.Column(scale=1, min_width=320, elem_classes=["aiwf-panel"]):
                            gr.Markdown("Live preview", elem_classes=["aiwf-section-label"])
                            gr.Markdown(
                                "Control how often the workspace decodes preview steps during generation.",
                                elem_classes=["aiwf-settings-paths"],
                            )
                            enable_live_preview = gr.Checkbox(
                                label="Show live preview while generating",
                                value=ctx.settings.enable_live_preview,
                            )
                            preview_every = gr.Slider(
                                label="Update preview every N steps",
                                minimum=1,
                                maximum=20,
                                step=1,
                                value=ctx.settings.show_progress_every_n_steps,
                                interactive=ctx.settings.enable_live_preview,
                            )
                            preview_hint = gr.Markdown(
                                ctx.settings.live_preview_summary(),
                                elem_classes=["aiwf-settings-hint"],
                            )

                        with gr.Column(scale=1, min_width=320, elem_classes=["aiwf-panel"]):
                            gr.Markdown("Appearance & navigation", elem_classes=["aiwf-section-label"])
                            gr.Markdown(
                                "Pick the accent mood and choose which secondary tabs stay visible. "
                                f"`{', '.join(sorted(PINNED_TABS))}` always stay on.",
                                elem_classes=["aiwf-settings-paths"],
                            )
                            accent_preset = gr.Radio(
                                label="Accent palette",
                                choices=accent_preset_names(),
                                value=ctx.settings.accent_preset,
                                info="Save, then use Refresh UI to apply the new accent immediately.",
                            )
                            visible_tabs = gr.CheckboxGroup(
                                label="Visible secondary tabs",
                                choices=TAB_VISIBILITY_CHOICES,
                                value=[tab_name for tab_name in TAB_VISIBILITY_CHOICES if tab_name not in ctx.settings.hidden_tabs],
                                info="Hide duplicate or rarely used tools without removing the feature from the project.",
                            )

                        with gr.Column(scale=1, min_width=320, elem_classes=["aiwf-panel"]):
                            gr.Markdown("Output behavior", elem_classes=["aiwf-section-label"])
                            gr.Markdown(
                                "Choose what gets saved and where the core generation modes write their files.",
                                elem_classes=["aiwf-settings-paths"],
                            )
                            save_images = gr.Checkbox(label="Save generated images", value=ctx.settings.save_images)
                            embed_metadata = gr.Checkbox(
                                label="Embed generation metadata in PNG",
                                value=ctx.settings.embed_metadata,
                            )
                            txt2img_dir = gr.Textbox(
                                label="txt2img output folder",
                                value=ctx.settings.txt2img_output_subdir,
                            )
                            img2img_dir = gr.Textbox(
                                label="img2img output folder",
                                value=ctx.settings.img2img_output_subdir,
                            )
                            inpaint_dir = gr.Textbox(
                                label="inpaint output folder",
                                value=ctx.settings.inpaint_output_subdir,
                            )
                            image_format = gr.Radio(
                                label="Image file format",
                                choices=["png", "jpg", "webp"],
                                value=ctx.settings.image_format,
                                info="PNG keeps generation metadata inside the file; jpg/webp save smaller files.",
                            )
                            image_quality = gr.Slider(
                                label="jpg/webp quality",
                                minimum=10,
                                maximum=100,
                                step=1,
                                value=ctx.settings.image_quality,
                            )
                            save_btn = gr.Button("Save workspace settings", variant="primary")
                            status = gr.Markdown("", elem_classes=["aiwf-status-bar"])

                    with gr.Row(equal_height=False, elem_classes=["aiwf-settings-grid"]):
                        with gr.Column(scale=1, min_width=320, elem_classes=["aiwf-panel"]):
                            gr.Markdown("Generation defaults", elem_classes=["aiwf-section-label"])
                            gr.Markdown(
                                "Starting values for the Studio tab. Save, then use Refresh UI (or restart) to apply.",
                                elem_classes=["aiwf-settings-paths"],
                            )
                            with gr.Row():
                                default_sampler = gr.Dropdown(
                                    label="Default sampler",
                                    choices=[s.label for s in samplers],
                                    value=default_sampler_label,
                                )
                                default_schedule = gr.Dropdown(
                                    label="Default schedule type",
                                    choices=[s.label for s in SCHEDULE_TYPES],
                                    value=default_schedule_label,
                                )
                            with gr.Row():
                                default_steps = gr.Slider(1, 150, value=ctx.settings.default_steps, step=1, label="Steps")
                                default_cfg = gr.Slider(1, 30, value=ctx.settings.default_cfg_scale, step=0.5, label="CFG scale")
                            with gr.Row():
                                default_width = gr.Slider(64, 2048, value=ctx.settings.default_width, step=8, label="Width")
                                default_height = gr.Slider(64, 2048, value=ctx.settings.default_height, step=8, label="Height")
                            default_clip_skip = gr.Slider(1, 12, value=ctx.settings.default_clip_skip, step=1, label="Clip skip")

                with gr.Tab("Launch profile"):
                    gr.Markdown(
                        "These options apply on the next app start. Saving writes `launch.json` (read automatically when you run `webui-user.bat` or `python launch.py`).",
                        elem_classes=["aiwf-page-intro"],
                    )

                    with gr.Row(equal_height=False, elem_classes=["aiwf-settings-grid"]):
                        with gr.Column(scale=1, min_width=320, elem_classes=["aiwf-panel"]):
                            gr.Markdown("Startup & appearance", elem_classes=["aiwf-section-label"])
                            launch_autolaunch = gr.Checkbox(
                                label="Open browser when the app starts",
                                value=launch.autolaunch,
                            )
                            launch_theme = gr.Radio(
                                label="Theme",
                                choices=["dark", "light"],
                                value=launch.theme,
                                info="Pick the default look for the next restart.",
                            )
                            launch_api = gr.Checkbox(
                                label="Enable REST API (--api)",
                                value=launch.api,
                                info="Mounts /api/v1 for tools, scripts, and integrations.",
                            )
                            launch_nowebui = gr.Checkbox(
                                label="API-only mode (--nowebui)",
                                value=launch.nowebui,
                                info="Starts FastAPI without the Gradio UI.",
                            )

                        with gr.Column(scale=1, min_width=320, elem_classes=["aiwf-panel"]):
                            gr.Markdown("Network & access", elem_classes=["aiwf-section-label"])
                            launch_listen = gr.Checkbox(
                                label="Allow access from other devices (--listen)",
                                value=launch.listen,
                                info="Required for phone, tablet, LAN, and Tailscale access.",
                            )
                            launch_port = gr.Number(
                                label="Port",
                                value=launch.port,
                                precision=0,
                            )
                            launch_auth = gr.Textbox(
                                label="Password protect UI (optional)",
                                value=launch.gradio_auth,
                                placeholder="username:password",
                                info="Recommended any time the UI is reachable from another device.",
                            )
                            launch_share = gr.Checkbox(
                                label="Gradio public share link",
                                value=launch.share,
                                info="Use only if you understand the privacy tradeoff.",
                            )

                    with gr.Row(equal_height=False, elem_classes=["aiwf-settings-grid"]):
                        with gr.Column(scale=1, min_width=320, elem_classes=["aiwf-panel"]):
                            gr.Markdown("Performance & memory", elem_classes=["aiwf-section-label"])
                            launch_cpu = gr.Checkbox(
                                label="Force CPU only (--cpu)",
                                value=launch.cpu,
                            )
                            launch_xformers = gr.Checkbox(
                                label="xFormers memory-efficient attention",
                                value=launch.xformers,
                            )
                            launch_sdp = gr.Checkbox(
                                label="PyTorch SDP attention",
                                value=launch.opt_sdp_attention,
                            )
                            launch_split = gr.Checkbox(
                                label="Split attention",
                                value=launch.opt_split_attention,
                            )
                            launch_medvram = gr.Checkbox(
                                label="Medium VRAM mode",
                                value=launch.medvram,
                            )
                            launch_lowvram = gr.Checkbox(
                                label="Low VRAM mode",
                                value=launch.lowvram,
                            )
                            launch_no_half = gr.Checkbox(
                                label="Disable half precision (FP32)",
                                value=launch.no_half,
                            )
                            launch_fp8 = gr.Checkbox(
                                label="FP8 UNet weights (experimental)",
                                value=launch.fp8,
                                info="Halves UNet VRAM — recommended for SDXL on 8GB cards. Tiny quality cost.",
                            )
                            launch_directml = gr.Checkbox(
                                label="DirectML — AMD/Intel GPU on Windows",
                                value=launch.directml,
                                info="Requires `pip install torch-directml`. NVIDIA cards should leave this off.",
                            )

                        with gr.Column(scale=1, min_width=320, elem_classes=["aiwf-panel"]):
                            gr.Markdown("Folders", elem_classes=["aiwf-section-label"])
                            gr.Markdown(
                                "Leave these blank to use the repo-local defaults. Point them somewhere else only on purpose.",
                                elem_classes=["aiwf-settings-paths"],
                            )
                            launch_models_dir = gr.Textbox(
                                label="Models folder",
                                value=launch.models_dir,
                                placeholder="Leave empty for default",
                            )
                            launch_ckpt_dir = gr.Textbox(
                                label="Checkpoints folder",
                                value=launch.ckpt_dir,
                                placeholder="Leave empty for default",
                            )
                            launch_output_dir = gr.Textbox(
                                label="Output folder",
                                value=launch.output_dir,
                                placeholder="Leave empty for default",
                            )

                    launch_preview = gr.Markdown(
                        f"**Next start command**  \n`{launch.command_preview()}`",
                        elem_classes=["aiwf-launch-preview"],
                    )
                    with gr.Row(elem_classes=["aiwf-settings-actions"]):
                        save_launch_btn = gr.Button("Save launch profile", variant="primary")
                    launch_save_status = gr.Markdown("", elem_classes=["aiwf-status-bar"])

                with gr.Tab("Access & security"):
                    with gr.Row(equal_height=False, elem_classes=["aiwf-settings-grid"]):
                        with gr.Column(scale=1, min_width=320, elem_classes=["aiwf-panel"]):
                            gr.Markdown("Security", elem_classes=["aiwf-section-label"])
                            gr.Markdown(_security_markdown(), elem_classes=["aiwf-settings-paths"])
                        with gr.Column(scale=1, min_width=320, elem_classes=["aiwf-panel"]):
                            gr.Markdown("Connection guidance", elem_classes=["aiwf-section-label"])
                            gr.Markdown(
                                "Tailscale support is built in to the access readout above. "
                                "For most remote use, the best order is: local network first, Tailscale second, public share last.",
                                elem_classes=["aiwf-settings-paths"],
                            )
                            gr.Markdown(
                                "Training tools, extension management, and deeper appearance controls are planned next. "
                                "They are not presented here as finished until the workflows around them are ready.",
                                elem_classes=["aiwf-settings-hint"],
                            )
                        with gr.Column(scale=1, min_width=320, elem_classes=["aiwf-panel"]):
                            gr.Markdown("API keys", elem_classes=["aiwf-section-label"])
                            gr.Markdown(
                                "Used by the Models tab for downloads. Stored locally in `config.json` — keep that file private.",
                                elem_classes=["aiwf-settings-paths"],
                            )
                            hf_token = gr.Textbox(
                                label="Hugging Face token",
                                type="password",
                                value=ctx.settings.huggingface_token,
                                placeholder="hf_...",
                                info="Needed for gated or private Hugging Face repos.",
                            )
                            civitai_token = gr.Textbox(
                                label="CivitAI API key",
                                type="password",
                                value=ctx.settings.civitai_token,
                                info="Needed for CivitAI models that require login to download.",
                            )
                            save_keys_btn = gr.Button("Save API keys", variant="primary")
                            keys_status = gr.Markdown("", elem_classes=["aiwf-status-bar"])

        launch_inputs = [
            launch_listen,
            launch_port,
            launch_auth,
            launch_share,
            launch_autolaunch,
            launch_theme,
            launch_cpu,
            launch_xformers,
            launch_sdp,
            launch_split,
            launch_medvram,
            launch_lowvram,
            launch_no_half,
            launch_fp8,
            launch_directml,
            launch_api,
            launch_nowebui,
            launch_models_dir,
            launch_ckpt_dir,
            launch_output_dir,
        ]

        def refresh_access():
            qr_image, qr_text = _connect_qr(ctx)
            return (
                _remote_access_markdown(ctx),
                _session_snapshot_markdown(ctx),
                gr.update(value=qr_image, visible=qr_image is not None),
                qr_text,
            )

        refresh_network.click(
            refresh_access,
            outputs=[remote_access, session_snapshot, connect_qr, qr_status],
            show_progress=False,
        )
        refresh_ui_btn.click(
            fn=None,
            js="() => window.location.reload()",
            inputs=None,
            outputs=None,
            show_progress=False,
        )

        def save_api_keys(hf_value, civitai_value):
            ctx.settings.huggingface_token = (hf_value or "").strip()
            ctx.settings.civitai_token = (civitai_value or "").strip()
            ctx.save_settings()
            ctx.settings.apply_token_env()
            return "**API keys saved.** They apply to new downloads immediately."

        save_keys_btn.click(
            save_api_keys,
            inputs=[hf_token, civitai_token],
            outputs=[keys_status],
            show_progress=False,
        )

        def toggle_preview_controls(enabled: bool, steps: float):
            preview_steps = max(1, min(20, int(steps or 1)))
            if enabled:
                summary = (
                    "Live preview every step"
                    if preview_steps == 1
                    else f"Live preview every {preview_steps} steps"
                )
            else:
                summary = "Live preview off"
            return gr.update(interactive=enabled), summary

        enable_live_preview.change(
            toggle_preview_controls,
            inputs=[enable_live_preview, preview_every],
            outputs=[preview_every, preview_hint],
            show_progress=False,
        )

        preview_every.change(
            toggle_preview_controls,
            inputs=[enable_live_preview, preview_every],
            outputs=[preview_every, preview_hint],
            show_progress=False,
        )

        def preview_launch_command(*values):
            try:
                settings = _launch_settings_from_inputs(values)
            except ValueError as exc:
                return gr.update(), f"**Error** — {exc}"
            return (
                f"**Next start command**  \n`{settings.command_preview()}`",
                "",
            )

        def _launch_settings_from_inputs(values) -> LaunchSettings:
            (
                listen,
                port,
                auth,
                share,
                autolaunch,
                theme,
                cpu,
                xformers,
                sdp,
                split,
                medvram,
                lowvram,
                no_half,
                fp8,
                directml,
                api,
                nowebui,
                models_dir,
                ckpt_dir,
                output_dir,
            ) = values
            return LaunchSettings(
                listen=bool(listen),
                port=int(port),
                gradio_auth=(auth or "").strip(),
                share=bool(share),
                autolaunch=bool(autolaunch),
                theme=theme,
                cpu=bool(cpu),
                xformers=bool(xformers),
                opt_sdp_attention=bool(sdp),
                opt_split_attention=bool(split),
                medvram=bool(medvram),
                lowvram=bool(lowvram),
                no_half=bool(no_half),
                fp8=bool(fp8),
                directml=bool(directml),
                api=bool(api),
                nowebui=bool(nowebui),
                models_dir=(models_dir or "").strip(),
                ckpt_dir=(ckpt_dir or "").strip(),
                output_dir=(output_dir or "").strip(),
            )

        for component in launch_inputs:
            component.change(
                preview_launch_command,
                inputs=launch_inputs,
                outputs=[launch_preview, launch_save_status],
                show_progress=False,
            )

        def save_launch_options(*values):
            try:
                settings = _launch_settings_from_inputs(values)
            except ValueError as exc:
                return f"**Error** — {exc}", gr.update(), gr.update()

            ctx.save_launch_settings(settings)
            status_text = (
                "**Launch profile saved.** Restart AIWF Studio to apply it. "
                f"Saved to `{ctx.launch_settings_path.name}`."
            )
            return (
                status_text,
                f"**Next start command**  \n`{settings.command_preview()}`",
                format_launch_status(LaunchSettings.from_runtime_flags(ctx.flags), settings),
            )

        save_launch_btn.click(
            save_launch_options,
            inputs=launch_inputs,
            outputs=[launch_save_status, launch_preview, launch_status],
            show_progress=False,
        )

        def persist(
            preview_enabled,
            preview_steps,
            save,
            embed,
            t2i,
            i2i,
            inpaint,
            fmt,
            quality,
            sampler_label,
            schedule_label,
            d_steps,
            d_cfg,
            d_width,
            d_height,
            d_clip,
            accent,
            selected_tabs,
        ):
            ctx.settings.enable_live_preview = bool(preview_enabled)
            ctx.settings.show_progress_every_n_steps = int(preview_steps)
            ctx.settings.save_images = save
            ctx.settings.embed_metadata = embed
            ctx.settings.txt2img_output_subdir = t2i
            ctx.settings.img2img_output_subdir = i2i
            ctx.settings.inpaint_output_subdir = inpaint
            ctx.settings.image_format = (fmt or "png").lower()
            ctx.settings.image_quality = int(quality or 95)
            ctx.settings.default_sampler = sampler_label_to_id.get(sampler_label, ctx.settings.default_sampler)
            ctx.settings.default_scheduler = schedule_label_to_id.get(schedule_label, "automatic")
            ctx.settings.default_steps = int(d_steps or 20)
            ctx.settings.default_cfg_scale = float(d_cfg or 7.0)
            ctx.settings.default_width = int(d_width or 512)
            ctx.settings.default_height = int(d_height or 512)
            ctx.settings.default_clip_skip = int(d_clip or 1)
            ctx.settings.accent_preset = accent or "mint"
            selected = set(selected_tabs or [])
            ctx.settings.hidden_tabs = [tab_name for tab_name in TAB_VISIBILITY_CHOICES if tab_name not in selected]
            ctx.save_settings()
            return (
                "Workspace settings saved. "
                f"{ctx.settings.live_preview_summary()}. "
                "Use Refresh UI to apply accent or tab visibility changes immediately.",
                ctx.settings.live_preview_summary(),
            )

        save_btn.click(
            persist,
            inputs=[
                enable_live_preview,
                preview_every,
                save_images,
                embed_metadata,
                txt2img_dir,
                img2img_dir,
                inpaint_dir,
                image_format,
                image_quality,
                default_sampler,
                default_schedule,
                default_steps,
                default_cfg,
                default_width,
                default_height,
                default_clip_skip,
                accent_preset,
                visible_tabs,
            ],
            outputs=[status, preview_hint],
            show_progress=False,
        )

        if tab is not None:
            tab.select(
                refresh_access,
                outputs=[remote_access, session_snapshot, connect_qr, qr_status],
                show_progress=False,
            )
