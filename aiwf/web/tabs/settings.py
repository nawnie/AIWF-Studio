from __future__ import annotations

import platform
from pathlib import Path

import gradio as gr

from aiwf.bootstrap import AppContext
from aiwf.api.security import api_security_warnings
from aiwf.core.config.launch import LaunchSettings, format_launch_status
from aiwf.core.util.access import build_network_access_info, format_remote_access_markdown
from aiwf.services.model_path_imports import (
    import_automatic1111_paths,
    import_comfyui_paths,
    merge_imported_path_text,
)
from aiwf.services.pipeline_registry import PipelineRegistry
from aiwf.services.worker_probe import WorkerProbeService
from aiwf.services.worker_tenant import WorkerTenantRegistry
from aiwf.web.registry import PINNED_TABS, WebRegistry
from aiwf.web.theme import accent_preset_names

TAB_VISIBILITY_CHOICES = [
    "Models",
    "Segment",
    "Enhance",
    "Chat",
    "Video",
    "RIFE",
    "Training",
    "Workflows",
    "Face Swap",
    "Library",
    "PNG Info",
    "History",
]


def _paths_text(value: str) -> list[str]:
    return [line.strip() for line in (value or "").splitlines() if line.strip()]


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


def _model_paths_markdown(ctx: AppContext) -> str:
    launch = _launch_form_values(ctx)
    lines = [
        f"**Models root** `{ctx.flags.resolved_models_dir()}`",
        f"**Checkpoints root** `{ctx.flags.resolved_ckpt_dir()}`",
        f"**Output root** `{ctx.flags.resolved_output_dir()}`",
    ]
    extra_models = _paths_text(launch.extra_model_dirs)
    extra_ckpts = _paths_text(launch.extra_ckpt_dirs)
    if extra_models:
        lines.append(f"**Extra model libraries** `{len(extra_models)}` configured")
    if extra_ckpts:
        lines.append(f"**Extra checkpoint libraries** `{len(extra_ckpts)}` configured")
    return "  \n".join(lines)


def _pipeline_registry(ctx: AppContext) -> PipelineRegistry:
    return PipelineRegistry(ctx.flags, ctx.settings)


def _pipeline_choices(ctx: AppContext) -> list[tuple[str, str]]:
    return _pipeline_registry(ctx).launch_choices()


def _pipeline_status_markdown(ctx: AppContext) -> str:
    return _pipeline_registry(ctx).status_markdown()


def _optimization_profile_choices(ctx: AppContext) -> list[tuple[str, str]]:
    profiles = getattr(ctx.optimization_planner, "profiles", {})
    order = [
        "safe_eager_cuda",
        "balanced_sdpa_fp16",
        "quality_visible_modifiers",
        "low_vram_model_offload",
        "fast_method_recipe",
        "experimental_feature_flags",
    ]
    choices: list[tuple[str, str]] = []
    for profile_id in order:
        profile = profiles.get(profile_id)
        label = getattr(profile, "display_name", profile_id)
        choices.append((label, profile_id))
    return choices


def _optimization_diagnostics_markdown(ctx: AppContext) -> str:
    diagnostics = getattr(ctx, "optimization_diagnostics", None)
    if diagnostics is None or not callable(getattr(diagnostics, "status_markdown", None)):
        return "**Optimization diagnostics unavailable.**"
    try:
        return diagnostics.status_markdown()
    except Exception as exc:
        return f"**Optimization diagnostics failed:** `{exc}`"


def _worker_tenants_markdown() -> str:
    registry = WorkerTenantRegistry()
    lines = ["**Engines**"]
    for status in registry.statuses().values():
        ready = "Ready" if status.ready else "Needs setup"
        state = "Enabled" if status.enabled else "Disabled"
        if status.uses_studio_venv:
            state = f"{state}, uses Studio runtime"
        issues = _engine_issue_summary(status.messages)
        lines.append(f"- **{status.label}:** {ready} ({state}) - {issues}")
        lines.append(f"  - Runtime: `{status.python_exe}`")
        lines.append(f"  - Worker file: `{status.worker_script}`")
        if status.repo_dir is not None:
            lines.append(f"  - Engine files: `{status.repo_dir}`")
    lines.append("")
    lines.append("Disabled or missing engines never block the core UI. Enable an engine in `engines.json`, run its bootstrap script, then refresh this panel.")
    return "\n".join(lines)


def _engine_issue_summary(messages: tuple[str, ...]) -> str:
    if not messages:
        return "ready"
    friendly: list[str] = []
    for message in messages:
        if "enabled=true" in message:
            friendly.append("not enabled")
        elif "runtime missing" in message or "python missing" in message:
            friendly.append("runtime folder missing")
        elif "AIWF worker missing" in message:
            friendly.append("app worker file missing")
        elif "repo folder missing" in message or "entry script missing" in message:
            friendly.append("engine files missing")
        else:
            friendly.append(message)
    return "; ".join(dict.fromkeys(friendly))


def _probe_worker_markdown(engine: str) -> str:
    label = engine
    try:
        label = WorkerTenantRegistry().status(engine).label
        result = WorkerProbeService().probe(engine)
    except Exception as exc:
        return f"**{label} probe failed before launch:** `{exc}`"
    status = "OK" if result.ok else "Failed"
    lines = [f"**{label} probe:** {status} - {result.message}", f"Request: `{result.request_path}`"]
    for event in result.events[-5:]:
        kind = event.get("kind", "event")
        message = event.get("message") or event.get("detail") or ""
        lines.append(f"- `{kind}` {message}")
    return "\n".join(lines)


def _launch_component_values(settings: LaunchSettings) -> list:
    return [
        settings.listen,
        settings.port,
        settings.gradio_auth,
        settings.api_cors_origins,
        settings.api_rate_limit_per_minute,
        settings.block_private_download_urls,
        settings.share,
        settings.autolaunch,
        settings.theme,
        settings.cpu,
        settings.xformers,
        settings.opt_sdp_attention,
        settings.opt_split_attention,
        settings.medvram,
        settings.lowvram,
        settings.no_half,
        settings.fp8,
        settings.async_offload,
        settings.pinned_memory,
        settings.cuda_malloc,
        settings.directml,
        settings.api,
        settings.nowebui,
        settings.models_dir,
        settings.ckpt_dir,
        settings.output_dir,
        settings.extra_model_dirs,
        settings.extra_ckpt_dirs,
    ]


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
                            live_preview_decoder = gr.Radio(
                                label="Preview decoder",
                                choices=[("VAE latent decode", "vae")],
                                value=(
                                    ctx.settings.live_preview_decoder
                                    if ctx.settings.live_preview_decoder == "vae"
                                    else "vae"
                                ),
                                info="TAESD fast preview is not exposed until backend support is available.",
                            )
                            preview_every = gr.Slider(
                                label="Update preview every N steps",
                                minimum=1,
                                maximum=20,
                                step=1,
                                value=ctx.settings.show_progress_every_n_steps,
                                interactive=(
                                    ctx.settings.enable_live_preview
                                    and ctx.settings.live_preview_decoder == "vae"
                                ),
                            )
                            preview_hint = gr.Markdown(
                                ctx.settings.live_preview_summary(),
                                elem_classes=["aiwf-settings-hint"],
                            )
                            live_preview_title_progress = gr.Checkbox(
                                label="Show generation progress in browser title",
                                value=ctx.settings.live_preview_title_progress,
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
                            filename_pattern = gr.Textbox(
                                label="Filename pattern",
                                value=ctx.settings.filename_pattern,
                                info="Tokens: [datetime] [date] [time] [seed] [model_name] [width] [height] [seq]. "
                                "Files are never overwritten - a -N suffix is added on collision.",
                            )
                            save_grid = gr.Checkbox(
                                label="Save a grid for batches",
                                value=ctx.settings.save_grid,
                                info="When a run produces multiple images, also save a combined grid (grid- prefix).",
                            )
                            save_sidecar_txt = gr.Checkbox(
                                label="Save sidecar .txt (parameters next to image)",
                                value=ctx.settings.save_sidecar_txt,
                            )
                            save_before_hires = gr.Checkbox(
                                label="Save image before hires fix (planned)",
                                value=ctx.settings.save_before_hires,
                                info="Saved as a preference now; takes effect once backend pre-hires capture lands.",
                            )
                            save_interrupted = gr.Checkbox(
                                label="Save interrupted generations (planned)",
                                value=ctx.settings.save_interrupted,
                                info="Saved as a preference now; takes effect once partial-image capture lands.",
                            )
                            gr.Markdown("Metadata & PNG Info", elem_classes=["aiwf-section-label"])
                            metadata_include_model_hash = gr.Checkbox(
                                label="Include model hash in saved parameters",
                                value=ctx.settings.metadata_include_model_hash,
                            )
                            metadata_include_vae_hash = gr.Checkbox(
                                label="Include VAE hash in saved parameters",
                                value=ctx.settings.metadata_include_vae_hash,
                            )
                            metadata_include_lora_hashes = gr.Checkbox(
                                label="Include LoRA hashes in saved parameters",
                                value=ctx.settings.metadata_include_lora_hashes,
                            )
                            metadata_include_app_version = gr.Checkbox(
                                label="Include AIWF Studio version in saved parameters",
                                value=ctx.settings.metadata_include_app_version,
                            )
                            metadata_include_optimization_profile = gr.Checkbox(
                                label="Include optimization profile in saved parameters",
                                value=ctx.settings.metadata_include_optimization_profile,
                            )
                            optimization_profile = gr.Radio(
                                label="Generation profile",
                                choices=_optimization_profile_choices(ctx),
                                value=ctx.settings.optimization_profile_id,
                                info="Recorded now; backend-changing optimizations remain gated by launch flags and benchmark receipts.",
                            )
                            optimization_diagnostics = gr.Markdown(
                                _optimization_diagnostics_markdown(ctx),
                                elem_classes=["aiwf-settings-hint"],
                            )
                            pnginfo_send_to_studio = gr.Checkbox(
                                label="Switch to Studio after PNG Info Send",
                                value=ctx.settings.pnginfo_send_to_studio,
                            )
                            pnginfo_clear_after_apply = gr.Checkbox(
                                label="Clear queued PNG Info after applying in Studio",
                                value=ctx.settings.pnginfo_clear_after_apply,
                            )
                            gr.Markdown("Sampler & guidance", elem_classes=["aiwf-section-label"])
                            auto_cfg_for_distilled = gr.Checkbox(
                                label="Auto-fix CFG for distilled models",
                                value=ctx.settings.auto_cfg_for_distilled,
                                info="Clamps CFG on Lightning/Hyper/Turbo/LCM models to prevent overexposure.",
                            )
                            use_default_negative = gr.Checkbox(
                                label="Use a default negative prompt when left blank",
                                value=ctx.settings.use_default_negative,
                            )
                            default_negative_prompt = gr.Textbox(
                                label="Default negative prompt (blank = built-in)",
                                value=ctx.settings.default_negative_prompt,
                                lines=2,
                                info="Generic quality terms only; no style/subject words.",
                            )
                            gr.Markdown("Gallery & viewer", elem_classes=["aiwf-section-label"])
                            gallery_height = gr.Slider(
                                label="Gallery height (px)",
                                minimum=120,
                                maximum=1200,
                                step=40,
                                value=ctx.settings.gallery_height,
                                info="Height of the image gallery panel in the Studio tab.",
                            )
                            gallery_columns = gr.Slider(
                                label="Gallery columns",
                                minimum=1,
                                maximum=8,
                                step=1,
                                value=ctx.settings.gallery_columns,
                                info="Number of columns in the batch results gallery.",
                            )
                            send_seed_on_click = gr.Checkbox(
                                label="Send seed to img2img when clicking a result",
                                value=ctx.settings.send_seed_on_click,
                            )
                            send_size_on_click = gr.Checkbox(
                                label="Send image size to img2img when clicking a result",
                                value=ctx.settings.send_size_on_click,
                            )

                            gr.Markdown("Download safety", elem_classes=["aiwf-section-label"])
                            prefer_safetensors = gr.Checkbox(
                                label="Warn before downloading .ckpt or .pt files",
                                value=ctx.settings.prefer_safetensors,
                                info="Safetensors files are safer — .ckpt/.pt can execute arbitrary code on load.",
                            )
                            write_download_receipts = gr.Checkbox(
                                label="Write download receipts (.json alongside each file)",
                                value=ctx.settings.write_download_receipts,
                                info="Saves URL, hash, and timestamp next to every downloaded model file.",
                            )

                            with gr.Row(elem_classes=["aiwf-settings-actions"]):
                                save_btn = gr.Button("Save workspace settings", variant="primary")
                                refresh_ui_btn = gr.Button(
                                    "Refresh UI",
                                    elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"],
                                )
                            status = gr.Markdown("", elem_classes=["aiwf-status-bar"])

                with gr.Tab("Generation defaults"):
                    with gr.Row(equal_height=False, elem_classes=["aiwf-settings-grid"]):
                        with gr.Column(scale=1, min_width=320, elem_classes=["aiwf-panel"]):
                            gr.Markdown("Studio starting values", elem_classes=["aiwf-section-label"])
                            gr.Markdown(
                                "These are the initial values loaded into Studio. Save, then use Refresh UI or reopen Studio to apply.",
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
                            gr.Markdown(
                                "Resolution, sampler, and clip skip defaults are stored in `config.json`.",
                                elem_classes=["aiwf-settings-hint"],
                            )

                with gr.Tab("Model paths"):
                    with gr.Row(equal_height=False, elem_classes=["aiwf-settings-grid"]):
                        with gr.Column(scale=1, min_width=320, elem_classes=["aiwf-panel"]):
                            gr.Markdown("Current runtime paths", elem_classes=["aiwf-section-label"])
                            gr.Markdown(
                                "These are the folders the current session is using right now.",
                                elem_classes=["aiwf-settings-paths"],
                            )
                            model_paths_runtime = gr.Markdown(
                                _model_paths_markdown(ctx),
                                elem_classes=["aiwf-page-path"],
                            )
                        with gr.Column(scale=1, min_width=320, elem_classes=["aiwf-panel"]):
                            gr.Markdown("Next-start library folders", elem_classes=["aiwf-section-label"])
                            gr.Markdown(
                                "Change the primary folders or add extra scan roots. These apply on the next app start after saving the launch profile.",
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
                            launch_extra_model_dirs = gr.Textbox(
                                label="Extra model library folders",
                                lines=4,
                                value=launch.extra_model_dirs,
                                placeholder="One folder per line",
                                info="Scans each folder for checkpoints, LoRAs, VAEs, embeddings, ControlNet, and enhance models on next start.",
                            )
                            launch_extra_ckpt_dirs = gr.Textbox(
                                label="Extra checkpoint folders",
                                lines=4,
                                value=launch.extra_ckpt_dirs,
                                placeholder="One folder per line",
                                info="Adds dedicated checkpoint-only search roots on next start.",
                            )
                        with gr.Column(scale=1, min_width=320, elem_classes=["aiwf-panel"]):
                            gr.Markdown("Import from another install", elem_classes=["aiwf-section-label"])
                            gr.Markdown(
                                "Point this at another project and AIWF will add the shared model folders for you instead of making you wire each library by hand.",
                                elem_classes=["aiwf-settings-paths"],
                            )
                            import_root = gr.Textbox(
                                label="External project folder",
                                placeholder=r"C:\path\to\stable-diffusion-webui or C:\path\to\ComfyUI",
                            )
                            with gr.Row(elem_classes=["aiwf-settings-actions"]):
                                import_a1111_btn = gr.Button(
                                    "Import AUTOMATIC1111",
                                    elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"],
                                )
                                import_comfy_btn = gr.Button(
                                    "Import ComfyUI",
                                    elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"],
                                )
                            import_status = gr.Markdown("", elem_classes=["aiwf-status-bar"])

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
                            launch_cors = gr.Textbox(
                                label="API CORS origins",
                                value=launch.api_cors_origins,
                                placeholder="https://example.local, http://127.0.0.1:3000",
                                info="Comma-separated browser origins allowed to call the API. Blank disables CORS.",
                            )
                            launch_rate_limit = gr.Number(
                                label="API rate limit / minute",
                                value=launch.api_rate_limit_per_minute,
                                precision=0,
                                info="0 disables the limiter. Applies per client IP to /api and /sdapi routes.",
                            )
                            launch_block_private_urls = gr.Checkbox(
                                label="Block private-network direct download URLs",
                                value=launch.block_private_download_urls,
                                info="Prevents direct custom downloads from loopback, LAN, multicast, and reserved IPs.",
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
                            launch_async_offload = gr.Checkbox(
                                label="Wan async offload (preload low during high denoise)",
                                value=launch.async_offload,
                                info="Hides disk I/O while the high transformer runs. On by default for 16 GB cards.",
                            )
                            launch_pinned_memory = gr.Checkbox(
                                label="Wan pinned CPU cache (fast PCIe swaps)",
                                value=launch.pinned_memory,
                                info="Page-locks Wan high/low weights in RAM for ~1s VRAM swaps.",
                            )
                            launch_cuda_malloc = gr.Checkbox(
                                label="cudaMallocAsync allocator",
                                value=launch.cuda_malloc,
                                info="Comfy --cuda-malloc parity. Reduces VRAM fragmentation on Ada.",
                            )
                            launch_directml = gr.Checkbox(
                                label="DirectML — AMD/Intel GPU on Windows",
                                value=launch.directml,
                                info="Requires `pip install torch-directml`. NVIDIA cards should leave this off.",
                            )

                        with gr.Column(scale=1, min_width=320, elem_classes=["aiwf-panel"]):
                            gr.Markdown("Saved profile snapshot", elem_classes=["aiwf-section-label"])
                            gr.Markdown(
                                "Folder overrides and extra scan roots live in the Model paths tab. This panel is here so launch changes still have a quick status readout.",
                                elem_classes=["aiwf-settings-paths"],
                            )
                            launch_status = gr.Markdown(
                                format_launch_status(LaunchSettings.from_runtime_flags(ctx.flags), saved_launch),
                                elem_classes=["aiwf-settings-hint"],
                            )
                            session_snapshot = gr.Markdown(
                                _session_snapshot_markdown(ctx),
                                elem_classes=["aiwf-settings-paths"],
                            )

                    launch_preview = gr.Markdown(
                        f"**Next start command**  \n`{launch.command_preview()}`",
                        elem_classes=["aiwf-launch-preview"],
                    )
                    with gr.Row(elem_classes=["aiwf-settings-actions"]):
                        refresh_launch_btn = gr.Button(
                            "Refresh launch settings",
                            elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"],
                        )
                        save_launch_btn = gr.Button("Save launch profile", variant="primary")
                    launch_save_status = gr.Markdown("", elem_classes=["aiwf-status-bar"])


                with gr.Tab("Engines & pipelines"):
                    gr.Markdown(
                        "Pipelines are generation methods inside Studio. Engines are optional add-on runners for video, training, and other heavy jobs.",
                        elem_classes=["aiwf-page-intro"],
                    )

                    with gr.Row(equal_height=False, elem_classes=["aiwf-settings-grid"]):
                        with gr.Column(scale=1, min_width=320, elem_classes=["aiwf-panel"]):
                            gr.Markdown("Pipelines", elem_classes=["aiwf-section-label"])
                            engine_backend = gr.Radio(
                                label="Image pipeline",
                                choices=_pipeline_choices(ctx),
                                value=launch.inference_backend,
                                info="Diffusers is the default reference path. ONNX Runtime uses AIWF's own sampler path.",
                            )
                            pipeline_status = gr.Markdown(
                                _pipeline_status_markdown(ctx),
                                elem_classes=["aiwf-settings-paths"],
                            )
                            with gr.Group(visible=(launch.inference_backend == "onnx")) as onnx_group:
                                engine_onnx_dir = gr.Textbox(
                                    label="ONNX models directory",
                                    value=ctx.settings.onnx_model_dir,
                                    placeholder=r"C:\models\onnx  (leave blank → models/onnx inside data_dir)",
                                    info="Directory containing subdirectories with text_encoder/, unet/, vae_decoder/ ONNX files.",
                                )
                                engine_onnx_provider = gr.Radio(
                                    label="ORT execution provider",
                                    choices=["auto", "cuda", "directml", "cpu"],
                                    value=launch.onnx_provider,
                                    info="auto tries CUDA → DirectML → CPU in order.",
                                )

                        with gr.Column(scale=1, min_width=320, elem_classes=["aiwf-panel"]):
                            gr.Markdown("Optimizations", elem_classes=["aiwf-section-label"])
                            gr.Markdown(
                                "These flags are set at startup via environment variables. "
                                "Save then restart to apply.",
                                elem_classes=["aiwf-settings-hint"],
                            )
                            engine_cuda_graphs = gr.Checkbox(
                                label="CUDA Graphs (AIWF_CUDA_GRAPHS)",
                                value=launch.cuda_graphs,
                                info="Capture and replay the UNet forward pass. ~5–15% throughput gain on NVIDIA Ada/Ampere.",
                            )
                            engine_torchao = gr.Checkbox(
                                label="TorchAO int8 weight-only quantization (AIWF_TORCHAO)",
                                value=launch.torchao,
                                info="Halves UNet weight memory. Requires torchao installed.",
                            )
                            engine_fp8 = gr.Checkbox(
                                label="FP8 weight-only quantization (AIWF_FP8)",
                                value=launch.fp8_quant,
                                info="Requires NVIDIA Ada Lovelace (RTX 40xx) or newer and torchao.",
                            )
                            engine_torch_compile = gr.Checkbox(
                                label="torch.compile reduce-overhead (AIWF_TORCH_COMPILE)",
                                value=launch.torch_compile,
                                info="Fuses ops in the UNet. First run is slow; subsequent runs are faster.",
                            )
                            engine_channels_last = gr.Checkbox(
                                label="Channels-last memory layout (AIWF_CHANNELS_LAST)",
                                value=launch.channels_last,
                                info="Best for SD 1.x UNet (Conv2D-heavy). Skip for transformer-based models.",
                            )
                            engine_nvenc = gr.Checkbox(
                                label="NVENC GPU video encoding (AIWF_NVENC)",
                                value=launch.nvenc,
                                info="Uses h264_nvenc instead of libx264 for video export. NVIDIA only.",
                            )
                            engine_hevc = gr.Checkbox(
                                label="Prefer HEVC / H.265 (AIWF_HEVC)",
                                value=launch.hevc,
                                info="Uses hevc_nvenc or libx265. Smaller files, slower encode.",
                            )

                        with gr.Column(scale=1, min_width=320, elem_classes=["aiwf-panel"]):
                            gr.Markdown("Engines", elem_classes=["aiwf-section-label"])
                            worker_tenant_status = gr.Markdown(
                                _worker_tenants_markdown(),
                                elem_classes=["aiwf-settings-paths"],
                            )
                            refresh_engines_btn = gr.Button(
                                "Refresh engines and pipelines",
                                elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"],
                            )
                            probe_wan_btn = gr.Button(
                                "Probe Wan Video Engine",
                                elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"],
                            )
                            worker_probe_status = gr.Markdown("", elem_classes=["aiwf-status-bar"])

                    with gr.Row(elem_classes=["aiwf-settings-actions"]):
                        save_engine_btn = gr.Button("Save engine settings", variant="primary")
                    engine_save_status = gr.Markdown("", elem_classes=["aiwf-status-bar"])

                with gr.Tab("Access & security"):
                    with gr.Row(equal_height=False, elem_classes=["aiwf-settings-grid"]):
                        with gr.Column(scale=1, min_width=320, elem_classes=["aiwf-panel"]):
                            gr.Markdown("Security", elem_classes=["aiwf-section-label"])
                            gr.Markdown(_security_markdown(), elem_classes=["aiwf-settings-paths"])
                            security_warnings = api_security_warnings(
                                listen=launch.listen,
                                gradio_auth=launch.gradio_auth,
                                api=launch.api,
                                nowebui=launch.nowebui,
                            )
                            gr.Markdown(
                                (
                                    "**Current saved-profile warnings**  \n"
                                    + "  \n".join(f"- {warning}" for warning in security_warnings)
                                    if security_warnings
                                    else "**Current saved profile:** no network auth warnings."
                                ),
                                elem_classes=["aiwf-settings-hint"],
                            )
                        with gr.Column(scale=1, min_width=320, elem_classes=["aiwf-panel"]):
                            gr.Markdown("Connection guidance", elem_classes=["aiwf-section-label"])
                            gr.Markdown(
                                "Tailscale support is built in to the Remote access tab. "
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

                with gr.Tab("Remote access"):
                    with gr.Row(equal_height=False, elem_classes=["aiwf-settings-hero"]):
                        with gr.Column(scale=2, min_width=340, elem_classes=["aiwf-panel", "aiwf-remote-panel"]):
                            gr.Markdown("Remote access", elem_classes=["aiwf-section-label"])
                            gr.Markdown(
                                "Use Tailscale or your local network for phone and tablet access. Refresh after changing network state, VPNs, or launch settings.",
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
                            gr.Markdown("This session", elem_classes=["aiwf-section-label"])
                            session_snapshot_remote = gr.Markdown(
                                _session_snapshot_markdown(ctx),
                                elem_classes=["aiwf-settings-paths"],
                            )
                            remote_paths = gr.Markdown(
                                _model_paths_markdown(ctx),
                                elem_classes=["aiwf-page-path"],
                            )

        launch_inputs = [
            launch_listen,
            launch_port,
            launch_auth,
            launch_cors,
            launch_rate_limit,
            launch_block_private_urls,
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
            launch_async_offload,
            launch_pinned_memory,
            launch_cuda_malloc,
            launch_directml,
            launch_api,
            launch_nowebui,
            launch_models_dir,
            launch_ckpt_dir,
            launch_output_dir,
            launch_extra_model_dirs,
            launch_extra_ckpt_dirs,
        ]
        extra_model_dirs_index = len(launch_inputs) - 2
        extra_ckpt_dirs_index = len(launch_inputs) - 1

        def refresh_launch_profile():
            ctx.load_settings()
            settings = _launch_form_values(ctx)
            return (
                *_launch_component_values(settings),
                f"**Next start command**  \n`{settings.command_preview()}`",
                format_launch_status(LaunchSettings.from_runtime_flags(ctx.flags), settings),
                _session_snapshot_markdown(ctx),
                "**Launch settings refreshed from disk.**",
            )

        refresh_launch_btn.click(
            refresh_launch_profile,
            outputs=[*launch_inputs, launch_preview, launch_status, session_snapshot, launch_save_status],
            show_progress=False,
        )

        def refresh_access():
            qr_image, qr_text = _connect_qr(ctx)
            return (
                _remote_access_markdown(ctx),
                _session_snapshot_markdown(ctx),
                _model_paths_markdown(ctx),
                gr.update(value=qr_image, visible=qr_image is not None),
                qr_text,
            )

        refresh_network.click(
            refresh_access,
            outputs=[remote_access, session_snapshot_remote, remote_paths, connect_qr, qr_status],
            show_progress=False,
        )
        refresh_ui_btn.click(
            fn=None,
            js="() => window.location.reload()",
            inputs=None,
            outputs=None,
            show_progress=False,
        )


        # ── Engine tab ──────────────────────────────────────────────────────
        def _toggle_onnx_group(backend_val: str):
            return gr.update(visible=(backend_val == "onnx"))

        engine_backend.change(
            _toggle_onnx_group,
            inputs=[engine_backend],
            outputs=[onnx_group],
            show_progress=False,
        )

        def save_engine_settings(
            backend, onnx_dir, onnx_provider,
            cuda_graphs, torchao, fp8, torch_compile, channels_last, nvenc, hevc,
        ):
            import os as _os
            # Persist onnx_model_dir to UserSettings
            ctx.settings.onnx_model_dir = (onnx_dir or "").strip()
            ctx.save_settings()

            # Persist pipeline and optimization flags to the next-start profile.
            saved = ctx.load_launch_settings() or LaunchSettings.from_runtime_flags(ctx.flags)
            settings = saved.model_copy(
                update={
                    "inference_backend": backend,
                    "onnx_provider": onnx_provider,
                    "cuda_graphs": bool(cuda_graphs),
                    "torchao": bool(torchao),
                    "fp8_quant": bool(fp8),
                    "torch_compile": bool(torch_compile),
                    "channels_last": bool(channels_last),
                    "nvenc": bool(nvenc),
                    "hevc": bool(hevc),
                }
            )
            ctx.save_launch_settings(settings)

            # Apply env vars immediately for the current session
            flag_map = {
                "AIWF_CUDA_GRAPHS":   cuda_graphs,
                "AIWF_TORCHAO":       torchao,
                "AIWF_FP8":           fp8,
                "AIWF_TORCH_COMPILE": torch_compile,
                "AIWF_CHANNELS_LAST": channels_last,
                "AIWF_NVENC":         nvenc,
                "AIWF_HEVC":          hevc,
            }
            for k, v in flag_map.items():
                _os.environ[k] = "1" if v else "0"

            return (
                "**Engine settings saved.** Restart AIWF Studio to activate the selected pipeline and flags."
            )

        save_engine_btn.click(
            save_engine_settings,
            inputs=[
                engine_backend, engine_onnx_dir, engine_onnx_provider,
                engine_cuda_graphs, engine_torchao, engine_fp8,
                engine_torch_compile, engine_channels_last, engine_nvenc, engine_hevc,
            ],
            outputs=[engine_save_status],
            show_progress=False,
        )
        # ── /Engine tab ──────────────────────────────────────────────────────

        def refresh_engine_panel():
            ctx.load_settings()
            settings = _launch_form_values(ctx)
            return (
                _worker_tenants_markdown(),
                gr.update(choices=_pipeline_choices(ctx), value=settings.inference_backend),
                _pipeline_status_markdown(ctx),
                gr.update(visible=(settings.inference_backend == "onnx")),
                ctx.settings.onnx_model_dir,
                settings.onnx_provider,
                settings.cuda_graphs,
                settings.torchao,
                settings.fp8_quant,
                settings.torch_compile,
                settings.channels_last,
                settings.nvenc,
                settings.hevc,
                "**Engines and pipelines refreshed from disk.**",
            )

        refresh_engines_btn.click(
            refresh_engine_panel,
            outputs=[
                worker_tenant_status,
                engine_backend,
                pipeline_status,
                onnx_group,
                engine_onnx_dir,
                engine_onnx_provider,
                engine_cuda_graphs,
                engine_torchao,
                engine_fp8,
                engine_torch_compile,
                engine_channels_last,
                engine_nvenc,
                engine_hevc,
                engine_save_status,
            ],
            show_progress=False,
        )
        probe_wan_btn.click(
            lambda: _probe_worker_markdown("wan"),
            outputs=[worker_probe_status],
            show_progress=True,
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

        def toggle_preview_controls(enabled: bool, decoder: str, steps: float):
            preview_steps = max(1, min(20, int(steps or 1)))
            supported = (decoder or "vae") == "vae"
            if enabled and supported:
                summary = (
                    "Live preview every step (VAE decode)"
                    if preview_steps == 1
                    else f"Live preview every {preview_steps} steps (VAE decode)"
                )
            else:
                summary = "Live preview off"
            return gr.update(interactive=enabled and supported), summary

        enable_live_preview.change(
            toggle_preview_controls,
            inputs=[enable_live_preview, live_preview_decoder, preview_every],
            outputs=[preview_every, preview_hint],
            show_progress=False,
        )

        live_preview_decoder.change(
            toggle_preview_controls,
            inputs=[enable_live_preview, live_preview_decoder, preview_every],
            outputs=[preview_every, preview_hint],
            show_progress=False,
        )

        preview_every.change(
            toggle_preview_controls,
            inputs=[enable_live_preview, live_preview_decoder, preview_every],
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
                cors,
                rate_limit,
                block_private_urls,
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
                async_offload,
                pinned_memory,
                cuda_malloc,
                directml,
                api,
                nowebui,
                models_dir,
                ckpt_dir,
                output_dir,
                extra_model_dirs,
                extra_ckpt_dirs,
            ) = values
            engine_profile = ctx.load_launch_settings() or LaunchSettings.from_runtime_flags(ctx.flags)
            return LaunchSettings(
                listen=bool(listen),
                port=int(port),
                gradio_auth=(auth or "").strip(),
                api_cors_origins=(cors or "").strip(),
                api_rate_limit_per_minute=max(0, int(rate_limit or 0)),
                block_private_download_urls=bool(block_private_urls),
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
                async_offload=bool(async_offload),
                pinned_memory=bool(pinned_memory),
                cuda_malloc=bool(cuda_malloc),
                directml=bool(directml),
                inference_backend=engine_profile.inference_backend,
                onnx_provider=engine_profile.onnx_provider,
                cuda_graphs=engine_profile.cuda_graphs,
                torchao=engine_profile.torchao,
                fp8_quant=engine_profile.fp8_quant,
                torch_compile=engine_profile.torch_compile,
                channels_last=engine_profile.channels_last,
                nvenc=engine_profile.nvenc,
                hevc=engine_profile.hevc,
                api=bool(api),
                nowebui=bool(nowebui),
                models_dir=(models_dir or "").strip(),
                ckpt_dir=(ckpt_dir or "").strip(),
                output_dir=(output_dir or "").strip(),
                extra_model_dirs="\n".join(_paths_text(extra_model_dirs)),
                extra_ckpt_dirs="\n".join(_paths_text(extra_ckpt_dirs)),
            )

        def import_external_project(root_value, importer, *launch_values):
            root_text = (root_value or "").strip()
            if not root_text:
                return (
                    gr.update(),
                    gr.update(),
                    "**Pick the other project's folder first.**",
                    gr.update(),
                )

            values = list(launch_values)
            existing_model_dirs = values[extra_model_dirs_index]
            existing_ckpt_dirs = values[extra_ckpt_dirs_index]
            try:
                imported = importer(root_text)
                merged_model_dirs, merged_ckpt_dirs = merge_imported_path_text(
                    existing_model_dirs,
                    existing_ckpt_dirs,
                    imported,
                )
                values[extra_model_dirs_index] = merged_model_dirs
                values[extra_ckpt_dirs_index] = merged_ckpt_dirs
                settings = _launch_settings_from_inputs(values[: len(launch_inputs)])
            except ValueError as exc:
                return gr.update(), gr.update(), f"**Import failed** - {exc}", gr.update()
            return (
                merged_model_dirs,
                merged_ckpt_dirs,
                f"**{imported.source} imported.** {imported.summary}",
                f"**Next start command**  \n`{settings.command_preview()}`",
            )

        import_a1111_btn.click(
            lambda root_value, *launch_values: import_external_project(
                root_value,
                import_automatic1111_paths,
                *launch_values,
            ),
            inputs=[import_root, *launch_inputs],
            outputs=[launch_extra_model_dirs, launch_extra_ckpt_dirs, import_status, launch_preview],
            show_progress=False,
        )

        import_comfy_btn.click(
            lambda root_value, *launch_values: import_external_project(
                root_value,
                import_comfyui_paths,
                *launch_values,
            ),
            inputs=[import_root, *launch_inputs],
            outputs=[launch_extra_model_dirs, launch_extra_ckpt_dirs, import_status, launch_preview],
            show_progress=False,
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
                return f"**Error** — {exc}", gr.update(), gr.update(), gr.update(), gr.update()

            ctx.save_launch_settings(settings)
            status_text = (
                "**Launch profile saved.** Restart AIWF Studio to apply it. "
                f"Saved to `{ctx.launch_settings_path.name}`."
            )
            return (
                status_text,
                f"**Next start command**  \n`{settings.command_preview()}`",
                format_launch_status(LaunchSettings.from_runtime_flags(ctx.flags), settings),
                _model_paths_markdown(ctx),
                _model_paths_markdown(ctx),
            )

        save_launch_btn.click(
            save_launch_options,
            inputs=launch_inputs,
            outputs=[launch_save_status, launch_preview, launch_status, model_paths_runtime, remote_paths],
            show_progress=False,
        )

        def persist(
            preview_enabled,
            preview_decoder,
            preview_steps,
            preview_title_on,
            save,
            embed,
            t2i,
            i2i,
            inpaint,
            fmt,
            quality,
            fname_pattern,
            grid_on,
            sidecar_on,
            before_hires_on,
            interrupted_on,
            include_model_hash,
            include_vae_hash,
            include_lora_hashes,
            include_app_version,
            include_optimization_profile,
            optimization_profile_id,
            send_pnginfo_to_studio,
            clear_pnginfo_after_apply,
            auto_cfg_distilled,
            use_default_neg,
            default_neg_text,
            gallery_h,
            gallery_cols,
            send_seed,
            send_size,
            prefer_safe,
            write_receipts,
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
            ctx.settings.live_preview_decoder = preview_decoder if preview_decoder == "vae" else "vae"
            ctx.settings.show_progress_every_n_steps = int(preview_steps)
            ctx.settings.live_preview_title_progress = bool(preview_title_on)
            ctx.settings.save_images = save
            ctx.settings.embed_metadata = embed
            ctx.settings.txt2img_output_subdir = t2i
            ctx.settings.img2img_output_subdir = i2i
            ctx.settings.inpaint_output_subdir = inpaint
            ctx.settings.image_format = (fmt or "png").lower()
            ctx.settings.image_quality = int(quality or 95)
            ctx.settings.filename_pattern = (fname_pattern or "[datetime]").strip() or "[datetime]"
            ctx.settings.save_grid = bool(grid_on)
            ctx.settings.save_sidecar_txt = bool(sidecar_on)
            ctx.settings.save_before_hires = bool(before_hires_on)
            ctx.settings.save_interrupted = bool(interrupted_on)
            ctx.settings.metadata_include_model_hash = bool(include_model_hash)
            ctx.settings.metadata_include_vae_hash = bool(include_vae_hash)
            ctx.settings.metadata_include_lora_hashes = bool(include_lora_hashes)
            ctx.settings.metadata_include_app_version = bool(include_app_version)
            ctx.settings.metadata_include_optimization_profile = bool(include_optimization_profile)
            if optimization_profile_id in {value for _label, value in _optimization_profile_choices(ctx)}:
                ctx.settings.optimization_profile_id = optimization_profile_id
            ctx.settings.pnginfo_send_to_studio = bool(send_pnginfo_to_studio)
            ctx.settings.pnginfo_clear_after_apply = bool(clear_pnginfo_after_apply)
            ctx.settings.auto_cfg_for_distilled = bool(auto_cfg_distilled)
            ctx.settings.use_default_negative = bool(use_default_neg)
            ctx.settings.default_negative_prompt = default_neg_text or ""
            ctx.settings.gallery_height = int(gallery_h or 480)
            ctx.settings.gallery_columns = int(gallery_cols or 2)
            ctx.settings.send_seed_on_click = bool(send_seed)
            ctx.settings.send_size_on_click = bool(send_size)
            ctx.settings.prefer_safetensors = bool(prefer_safe)
            ctx.settings.write_download_receipts = bool(write_receipts)
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
                live_preview_decoder,
                preview_every,
                live_preview_title_progress,
                save_images,
                embed_metadata,
                txt2img_dir,
                img2img_dir,
                inpaint_dir,
                image_format,
                image_quality,
                filename_pattern,
                save_grid,
                save_sidecar_txt,
                save_before_hires,
                save_interrupted,
                metadata_include_model_hash,
                metadata_include_vae_hash,
                metadata_include_lora_hashes,
                metadata_include_app_version,
                metadata_include_optimization_profile,
                optimization_profile,
                pnginfo_send_to_studio,
                pnginfo_clear_after_apply,
                auto_cfg_for_distilled,
                use_default_negative,
                default_negative_prompt,
                gallery_height,
                gallery_columns,
                send_seed_on_click,
                send_size_on_click,
                prefer_safetensors,
                write_download_receipts,
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
                outputs=[remote_access, session_snapshot_remote, remote_paths, connect_qr, qr_status],
                show_progress=False,
            )
            tab.select(
                refresh_engine_panel,
                outputs=[
                    worker_tenant_status,
                    engine_backend,
                    pipeline_status,
                    onnx_group,
                    engine_onnx_dir,
                    engine_onnx_provider,
                    engine_cuda_graphs,
                    engine_torchao,
                    engine_fp8,
                    engine_torch_compile,
                    engine_channels_last,
                    engine_nvenc,
                    engine_hevc,
                    engine_save_status,
                ],
                show_progress=False,
            )
            tab.select(
                lambda: _optimization_diagnostics_markdown(ctx),
                outputs=[optimization_diagnostics],
                show_progress=False,
            )
