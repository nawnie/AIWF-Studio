from __future__ import annotations

import platform
from pathlib import Path

import gradio as gr

from aiwf.bootstrap import AppContext
from aiwf.core.config.launch import LaunchSettings, format_launch_status
from aiwf.core.util.access import build_network_access_info, format_remote_access_markdown
from aiwf.web.registry import WebRegistry


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
                            save_btn = gr.Button("Save workspace settings", variant="primary")
                            status = gr.Markdown("", elem_classes=["aiwf-status-bar"])

                with gr.Tab("Launch profile"):
                    gr.Markdown(
                        "These options apply on the next app start. Saving writes `launch.json` and syncs `webui.settings.bat`.",
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
            launch_api,
            launch_nowebui,
            launch_models_dir,
            launch_ckpt_dir,
            launch_output_dir,
        ]

        def refresh_access():
            return _remote_access_markdown(ctx), _session_snapshot_markdown(ctx)

        refresh_network.click(
            refresh_access,
            outputs=[remote_access, session_snapshot],
            show_progress=False,
        )
        refresh_ui_btn.click(
            fn=None,
            js="() => window.location.reload()",
            inputs=None,
            outputs=None,
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

            ctx.save_launch_settings(settings, project_root=Path(ctx.flags.data_dir))
            status_text = (
                "**Launch profile saved.** Restart AIWF Studio to apply it. "
                f"Saved to `{ctx.launch_settings_path.name}` and `webui.settings.bat`."
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
        ):
            ctx.settings.enable_live_preview = bool(preview_enabled)
            ctx.settings.show_progress_every_n_steps = int(preview_steps)
            ctx.settings.save_images = save
            ctx.settings.embed_metadata = embed
            ctx.settings.txt2img_output_subdir = t2i
            ctx.settings.img2img_output_subdir = i2i
            ctx.settings.inpaint_output_subdir = inpaint
            ctx.save_settings()
            return (
                f"Workspace settings saved. {ctx.settings.live_preview_summary()}.",
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
            ],
            outputs=[status, preview_hint],
            show_progress=False,
        )

        if tab is not None:
            tab.select(
                refresh_access,
                outputs=[remote_access, session_snapshot],
                show_progress=False,
            )
