from __future__ import annotations

import html
import platform
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr

from aiwf.bootstrap import AppContext
from aiwf.core.domain.generation import GenerationMode, GenerationRequest, JobState
from aiwf.core.domain.models import SCHEDULE_TYPES, normalize_schedule_id_for_sampler
from aiwf.web.modern.dataset_reference import (
    dataset_summary_markdown,
    filter_records,
    gallery_items,
    load_manifest,
    resolve_dataset_dir,
    table_rows,
)


_STATIC_DIR = Path(__file__).resolve().parents[3] / "static"
_PAGE_NAMES = ("Create", "Models", "Data", "Services", "Settings")


@lru_cache(maxsize=4)
def _static_text(name: str) -> str:
    path = _STATIC_DIR / name
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _modern_theme(dark: bool = True) -> gr.Theme:
    if not dark:
        return gr.themes.Soft(
            primary_hue=gr.themes.colors.teal,
            secondary_hue=gr.themes.colors.blue,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("DM Sans"),
            font_mono=gr.themes.GoogleFont("IBM Plex Mono"),
        )
    return (
        gr.themes.Base(
            primary_hue=gr.themes.colors.teal,
            secondary_hue=gr.themes.colors.blue,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("DM Sans"),
            font_mono=gr.themes.GoogleFont("IBM Plex Mono"),
        )
        .set(
            body_background_fill="#090b10",
            body_text_color="#f4f7fb",
            block_background_fill="#111722",
            block_border_color="rgba(255,255,255,0.08)",
            block_border_width="1px",
            block_radius="8px",
            input_background_fill="#0b1018",
            input_border_color="rgba(255,255,255,0.11)",
            input_radius="8px",
            button_primary_background_fill="linear-gradient(180deg, #77d7c0 0%, #3e9f8a 100%)",
            button_primary_text_color="#06100d",
            button_secondary_background_fill="#171f2c",
            button_secondary_text_color="#c8d1df",
            button_secondary_background_fill_hover="#1d2736",
            border_color_primary="rgba(255,255,255,0.10)",
            color_accent="#77d7c0",
            color_accent_soft="rgba(119,215,192,0.14)",
            link_text_color="#9be8d4",
            shadow_drop="0 18px 60px rgba(0,0,0,0.45)",
        )
    )


def _safe_call(default: Any, func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception:
        return default


def _checkpoint_choices(ctx: AppContext) -> tuple[list[str], dict[str, str], str | None]:
    checkpoints = _safe_call([], ctx.generation.list_checkpoints)
    choices = [checkpoint.title for checkpoint in checkpoints]
    mapping = {checkpoint.title: checkpoint.id for checkpoint in checkpoints}
    selected = None
    last_id = getattr(ctx.settings, "last_checkpoint_id", None)
    for checkpoint in checkpoints:
        if checkpoint.id == last_id:
            selected = checkpoint.title
            break
    return choices, mapping, selected or (choices[0] if choices else None)


def _sampler_choices(ctx: AppContext) -> tuple[list[str], dict[str, str], str | None]:
    samplers = _safe_call([], ctx.generation.list_samplers)
    choices = [sampler.label for sampler in samplers]
    mapping = {sampler.label: sampler.id for sampler in samplers}
    selected = next(
        (sampler.label for sampler in samplers if sampler.id == getattr(ctx.settings, "default_sampler", "euler_a")),
        choices[0] if choices else None,
    )
    return choices, mapping, selected


def _model_table(ctx: AppContext) -> tuple[list[list[str]], str]:
    checkpoints = _safe_call([], ctx.generation.list_checkpoints)
    loras = _safe_call([], ctx.generation.list_loras)
    vaes = _safe_call([], ctx.generation.list_vaes)
    rows = []
    for item in checkpoints[:80]:
        rows.append(["Checkpoint", item.title, item.architecture, item.filename])
    for item in loras[:80]:
        rows.append(["LoRA", item.title, item.architecture, item.filename])
    for item in vaes[:40]:
        rows.append(["VAE", item.title, "", item.filename])
    status = f"Found {len(checkpoints)} checkpoints, {len(loras)} LoRAs, and {len(vaes)} VAEs."
    return rows, status


def _recent_gallery(ctx: AppContext) -> list[Any]:
    images = []
    for job in _safe_call([], ctx.generation.recent_jobs, 12):
        if job.state != JobState.COMPLETED or job.result is None:
            continue
        images.extend(job.result.images[:2])
    return images[:12]


def _runtime_html(ctx: AppContext) -> str:
    try:
        import torch

        torch_version = torch.__version__.split("+", 1)[0]
    except Exception:
        torch_version = "unavailable"
    device = _safe_call("Unknown", ctx.generation.backend.devices.describe)
    if device.startswith("CUDA ("):
        device = device.removeprefix("CUDA (").rstrip(")").split(",", 1)[0]
    cards = [
        ("Python", platform.python_version(), "mint"),
        ("Torch", torch_version, "blue"),
        ("Device", device, "amber"),
        ("Backend", getattr(ctx.flags, "inference_backend", "diffusers"), "violet"),
    ]
    return "".join(
        (
            f'<div class="modern-stat modern-stat-{tone}">'
            f'<span>{html.escape(label)}</span><strong>{html.escape(str(value))}</strong></div>'
        )
        for label, value, tone in cards
    )


def _onboarding_html(ctx: AppContext) -> str:
    avatar = html.escape(getattr(ctx.settings, "github_avatar_url", "") or "https://github.com/nawnie.png?size=160")
    return f"""
    <div class="modern-onboarding-card">
        <div class="modern-avatar-wrap">
            <img src="{avatar}" alt="nawnie GitHub avatar" class="modern-avatar">
            <div><strong>nawnie</strong><span>Local AIWF workspace</span></div>
        </div>
        <div>
            <h2>Welcome to AIWF Studio</h2>
            <p>This modern shell keeps the first run calm: create images quickly, then open advanced tools only when you need them.</p>
            <p>Models, datasets, private tokens, and generated outputs stay local unless you explicitly enable sharing or remote access.</p>
        </div>
    </div>
    """


def _show_page(page: str):
    return [
        page,
        *[gr.update(visible=name == page) for name in _PAGE_NAMES],
        f"Current workspace: **{page}**",
    ]


def _persist_onboarding(ctx: AppContext, *, page: str = "Create"):
    ctx.settings.modern_onboarding_seen = True
    ctx.save_settings()
    return [gr.update(visible=False), *_show_page(page)]


def _dataset_view(query: str, asset_type: str):
    records = load_manifest()
    root = resolve_dataset_dir()
    filtered = filter_records(records, query=query, asset_type=asset_type)
    return (
        dataset_summary_markdown(records, root) + f"\n\nShowing **{len(filtered)}** matching records.",
        gallery_items(filtered),
        table_rows(filtered),
    )


def _run_txt2img(
    ctx: AppContext,
    checkpoint_map: dict[str, str],
    sampler_map: dict[str, str],
    prompt: str,
    negative: str,
    checkpoint_title: str | None,
    sampler_label: str | None,
    scheduler_label: str | None,
    steps: int,
    cfg_scale: float,
    width: int,
    height: int,
    seed: int,
    batch_size: int,
    batch_count: int,
):
    if not (prompt or "").strip():
        raise gr.Error("Enter a prompt first.")
    if not checkpoint_title or checkpoint_title not in checkpoint_map:
        raise gr.Error("Choose a checkpoint in Models or add one to your model folder.")
    sampler_id = sampler_map.get(sampler_label or "", "euler_a")
    schedule_id = normalize_schedule_id_for_sampler(
        sampler_id,
        next((item.id for item in SCHEDULE_TYPES if item.label == scheduler_label), "automatic"),
    )
    request = GenerationRequest(
        mode=GenerationMode.TXT2IMG,
        prompt=prompt,
        negative_prompt=negative or "",
        checkpoint_id=checkpoint_map[checkpoint_title],
        sampler=sampler_id,
        scheduler=schedule_id,
        steps=int(steps),
        cfg_scale=float(cfg_scale),
        width=int(width),
        height=int(height),
        seed=int(seed),
        batch_size=int(batch_size),
        batch_count=int(batch_count),
        clip_skip=int(getattr(ctx.settings, "default_clip_skip", 1)),
    )
    last_preview = None
    session_images: list = []
    session_seeds: list[int] = []
    gallery_columns = min(max(1, int(getattr(ctx.settings, "gallery_columns", 2) or 2)), 4)

    def _gallery_update(images: list) -> gr.Update:
        return gr.update(
            value=list(images),
            visible=len(images) > 1,
            columns=min(gallery_columns, max(1, len(images))),
        )

    for event in ctx.generation.submit_streaming(request):
        if event[0] == "progress":
            _kind, step, total, message, preview = event
            if preview is not None:
                last_preview = preview
            yield (
                gr.update(value=last_preview, visible=last_preview is not None),
                _gallery_update(session_images) if session_images else gr.update(),
                f"**Running** - step {step}/{total}: {message}",
            )
        elif event[0] == "batch_images":
            _kind, batch_images, batch_seeds = event
            session_images.extend(batch_images)
            session_seeds.extend(batch_seeds)
            primary = session_images[-1] if session_images else None
            seeds = ", ".join(str(seed) for seed in session_seeds[:8])
            yield (
                gr.update(value=primary, visible=primary is not None),
                _gallery_update(session_images),
                f"**Batch complete** - {len(session_images)} image(s), seed {seeds or 'recorded in metadata'}",
            )
        else:
            _kind, job = event
            if job.result is None:
                yield gr.update(), gr.update(), f"**Error** - {job.error or job.state.value}"
                return
            if not session_images and job.result.images:
                session_images.extend(job.result.images)
                session_seeds.extend(job.result.seeds)
            images = session_images or job.result.images
            primary = images[-1] if images else None
            seeds = ", ".join(str(seed) for seed in (session_seeds or job.result.seeds)[:8])
            yield (
                gr.update(value=primary, visible=primary is not None),
                _gallery_update(images),
                f"**Done** - {len(images)} image(s), seed {seeds or 'recorded in metadata'}",
            )


def create_modern_web_ui(ctx: AppContext) -> tuple[gr.Blocks, object, str, str]:
    checkpoint_choices, checkpoint_map, checkpoint_value = _checkpoint_choices(ctx)
    sampler_choices, sampler_map, sampler_value = _sampler_choices(ctx)
    schedule_choices = [item.label for item in SCHEDULE_TYPES]
    dataset_records = load_manifest()
    dataset_root = resolve_dataset_dir()
    model_rows, model_status = _model_table(ctx)

    with gr.Blocks(title="AIWF Studio Modern", elem_classes=["aiwf-modern-app"]) as demo:
        gr.HTML(
            """
            <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
            <meta name="mobile-web-app-capable" content="yes">
            """,
            visible=False,
        )
        active_page = gr.State("Create")
        checkpoint_state = gr.State(checkpoint_map)
        sampler_state = gr.State(sampler_map)

        with gr.Row(elem_classes=["modern-shell"], equal_height=False):
            with gr.Column(scale=1, min_width=210, elem_classes=["modern-sidebar"]):
                gr.HTML(
                    """
                    <div class="modern-brand">
                        <div class="modern-mark">AI</div>
                        <div><strong>AIWF Studio</strong><span>Modern shell</span></div>
                    </div>
                    """
                )
                nav_create = gr.Button("Create", elem_classes=["modern-nav-button"])
                nav_models = gr.Button("Models", elem_classes=["modern-nav-button"])
                nav_data = gr.Button("Data", elem_classes=["modern-nav-button"])
                nav_services = gr.Button("Services", elem_classes=["modern-nav-button"])
                nav_settings = gr.Button("Settings", elem_classes=["modern-nav-button"])
                gr.HTML('<div class="modern-sidebar-note">Simple first. Full control when you open it.</div>')

            with gr.Column(scale=6, elem_classes=["modern-main"]):
                gr.HTML(
                    f"""
                    <header class="modern-topbar">
                        <div><h1>AIWF Studio</h1><p>Local image, video, model, and dataset workspace.</p></div>
                        <div class="modern-runtime-grid">{_runtime_html(ctx)}</div>
                    </header>
                    """
                )
                route_status = gr.Markdown("Current workspace: **Create**", elem_classes=["modern-route-status"])

                with gr.Column(
                    visible=not getattr(ctx.settings, "modern_onboarding_seen", False),
                    elem_classes=["modern-onboarding"],
                ) as onboarding_panel:
                    gr.HTML(_onboarding_html(ctx))
                    with gr.Row(elem_classes=["modern-onboarding-actions"]):
                        onboarding_start = gr.Button("Get started", variant="primary")
                        onboarding_advanced = gr.Button("Show advanced tools")
                        onboarding_data = gr.Button("Open dataset reference")

                with gr.Column(visible=True, elem_classes=["modern-page modern-create-page"]) as page_create:
                    with gr.Row(equal_height=False, elem_classes=["modern-workbench"]):
                        with gr.Column(scale=7, elem_classes=["modern-panel modern-prompt-panel"]):
                            gr.HTML("<div class='modern-section-title'><span>Create</span><strong>Image generation</strong></div>")
                            prompt = gr.Textbox(
                                label="Prompt",
                                lines=5,
                                placeholder="Describe what you want to generate...",
                            )
                            with gr.Accordion("Negative prompt and prompt sources", open=False):
                                negative = gr.Textbox(label="Negative prompt", lines=3)
                                gr.Markdown("Prompt files, styles, and wildcard routing will live here as the shell expands.")
                            with gr.Row():
                                checkpoint = gr.Dropdown(
                                    label="Checkpoint",
                                    choices=checkpoint_choices,
                                    value=checkpoint_value,
                                    interactive=True,
                                )
                                mode = gr.Radio(
                                    label="Mode",
                                    choices=["txt2img", "img2img", "inpaint"],
                                    value="txt2img",
                                )
                            with gr.Row():
                                width = gr.Slider(256, 2048, value=ctx.settings.default_width, step=8, label="Width")
                                height = gr.Slider(256, 2048, value=ctx.settings.default_height, step=8, label="Height")
                            with gr.Row():
                                steps = gr.Slider(1, 80, value=ctx.settings.default_steps, step=1, label="Steps")
                                cfg = gr.Slider(0, 20, value=ctx.settings.default_cfg_scale, step=0.5, label="CFG")
                                seed = gr.Number(value=-1, precision=0, label="Seed")
                            with gr.Accordion("Advanced generation controls", open=False):
                                with gr.Row():
                                    sampler = gr.Dropdown(
                                        label="Sampler",
                                        choices=sampler_choices,
                                        value=sampler_value,
                                    )
                                    scheduler = gr.Dropdown(
                                        label="Scheduler",
                                        choices=schedule_choices,
                                        value="Automatic",
                                    )
                                with gr.Row():
                                    batch_size = gr.Slider(1, 8, value=1, step=1, label="Batch size")
                                    batch_count = gr.Slider(1, 8, value=1, step=1, label="Batch count")
                                gr.Markdown("HR, refiner, ControlNet, LoRA, and metadata controls will be added here without crowding the first screen.")
                            run_btn = gr.Button("Generate", variant="primary", elem_classes=["modern-generate"])

                        with gr.Column(scale=5, elem_classes=["modern-panel modern-output-panel"]):
                            gr.HTML("<div class='modern-section-title'><span>Output</span><strong>Preview and recent work</strong></div>")
                            output = gr.Image(label="Primary output", visible=True, elem_classes=["modern-output-image"])
                            status = gr.Markdown("Ready.", elem_classes=["modern-run-status"])
                            gallery = gr.Gallery(
                                label="Batch outputs",
                                value=[],
                                visible=False,
                                columns=2,
                                object_fit="contain",
                            )
                            recent = gr.Gallery(
                                label="Recent outputs",
                                value=_recent_gallery(ctx),
                                columns=4,
                                object_fit="cover",
                                elem_classes=["modern-recent-gallery"],
                            )

                with gr.Column(visible=False, elem_classes=["modern-page"]) as page_models:
                    gr.HTML("<div class='modern-page-heading'><h2>Models</h2><p>Installed checkpoints, LoRAs, and VAEs stay visible without dominating the creation screen.</p></div>")
                    refresh_models = gr.Button("Refresh model scan")
                    model_status_md = gr.Markdown(model_status)
                    model_table = gr.Dataframe(
                        headers=["Kind", "Title", "Architecture", "File"],
                        value=model_rows,
                        interactive=False,
                        wrap=True,
                        elem_classes=["modern-dataframe"],
                    )

                with gr.Column(visible=False, elem_classes=["modern-page"]) as page_data:
                    gr.HTML("<div class='modern-page-heading'><h2>Data</h2><p>Read-only reference view for the MoK Gradio dataset used to guide this UI overhaul.</p></div>")
                    dataset_status = gr.Markdown(dataset_summary_markdown(dataset_records, dataset_root))
                    with gr.Row():
                        dataset_query = gr.Textbox(label="Reference search", placeholder="Try Blocks, local LLM, layout, event...")
                        dataset_type = gr.Dropdown(
                            label="Asset type",
                            choices=["All", "capture", "diagram", "flowchart", "graph"],
                            value="All",
                        )
                        dataset_search = gr.Button("Search")
                    dataset_gallery = gr.Gallery(
                        label="Reference gallery",
                        value=gallery_items(dataset_records),
                        columns=3,
                        object_fit="contain",
                        elem_classes=["modern-reference-gallery"],
                    )
                    dataset_table = gr.Dataframe(
                        headers=["ID", "Type", "Split", "Chapter", "Caption", "Status"],
                        value=table_rows(dataset_records),
                        interactive=False,
                        wrap=True,
                        elem_classes=["modern-dataframe"],
                    )

                with gr.Column(visible=False, elem_classes=["modern-page"]) as page_services:
                    gr.HTML(
                        """
                        <div class="modern-page-heading"><h2>Services</h2><p>Advanced tools are grouped by workflow so users do not have to scan a long tab strip.</p></div>
                        <div class="modern-service-grid">
                            <div><span>Video</span><strong>Wan, RIFE, audio-video</strong><p>Generate, interpolate, and prepare media.</p></div>
                            <div><span>Image tools</span><strong>Inpaint, enhance, segment, face swap</strong><p>Open only when the task needs them.</p></div>
                            <div><span>Library</span><strong>Models, history, PNG info</strong><p>Inspect sources and recover previous work.</p></div>
                            <div><span>Training</span><strong>Datasets and ED2</strong><p>Keep training setup away from beginner generation.</p></div>
                            <div><span>Chat</span><strong>Local assistant workspace</strong><p>Planning and tool help without leaving AIWF.</p></div>
                            <div><span>Workflows</span><strong>Composable pipelines</strong><p>Opt-in automation for repeatable jobs.</p></div>
                        </div>
                        """
                    )

                with gr.Column(visible=False, elem_classes=["modern-page"]) as page_settings:
                    gr.HTML("<div class='modern-page-heading'><h2>Settings</h2><p>Modern shell preferences. Core runtime settings remain shared with AIWF Studio.</p></div>")
                    avatar_url = gr.Textbox(
                        label="GitHub avatar URL",
                        value=getattr(ctx.settings, "github_avatar_url", "https://github.com/nawnie.png?size=160"),
                    )
                    reset_onboarding = gr.Button("Show onboarding on next modern launch")
                    settings_status = gr.Markdown(f"Settings file: `{ctx.settings_path}`")

        pages = [page_create, page_models, page_data, page_services, page_settings]
        nav_outputs = [active_page, *pages, route_status]
        nav_create.click(lambda: _show_page("Create"), outputs=nav_outputs, show_progress=False)
        nav_models.click(lambda: _show_page("Models"), outputs=nav_outputs, show_progress=False)
        nav_data.click(lambda: _show_page("Data"), outputs=nav_outputs, show_progress=False)
        nav_services.click(lambda: _show_page("Services"), outputs=nav_outputs, show_progress=False)
        nav_settings.click(lambda: _show_page("Settings"), outputs=nav_outputs, show_progress=False)

        onboarding_start.click(
            lambda: _persist_onboarding(ctx, page="Create"),
            outputs=[onboarding_panel, active_page, *pages, route_status],
            show_progress=False,
        )
        onboarding_advanced.click(
            lambda: _persist_onboarding(ctx, page="Services"),
            outputs=[onboarding_panel, active_page, *pages, route_status],
            show_progress=False,
        )
        onboarding_data.click(
            lambda: _persist_onboarding(ctx, page="Data"),
            outputs=[onboarding_panel, active_page, *pages, route_status],
            show_progress=False,
        )
        def _run_txt2img_event(*args):
            yield from _run_txt2img(ctx, *args)

        run_btn.click(
            _run_txt2img_event,
            inputs=[
                checkpoint_state,
                sampler_state,
                prompt,
                negative,
                checkpoint,
                sampler,
                scheduler,
                steps,
                cfg,
                width,
                height,
                seed,
                batch_size,
                batch_count,
            ],
            outputs=[output, gallery, status],
        )
        dataset_search.click(
            _dataset_view,
            inputs=[dataset_query, dataset_type],
            outputs=[dataset_status, dataset_gallery, dataset_table],
            show_progress=False,
        )
        refresh_models.click(lambda: _model_table(ctx), outputs=[model_table, model_status_md], show_progress=False)

        def _save_avatar(url: str):
            ctx.settings.github_avatar_url = (url or "").strip() or "https://github.com/nawnie.png?size=160"
            ctx.save_settings()
            return "**Saved.** Restart or reload the modern shell to refresh the onboarding avatar."

        avatar_url.submit(_save_avatar, inputs=[avatar_url], outputs=[settings_status], show_progress=False)

        def _reset_onboarding():
            ctx.settings.modern_onboarding_seen = False
            ctx.save_settings()
            return "**Saved.** Onboarding will show on the next modern launch."

        reset_onboarding.click(_reset_onboarding, outputs=[settings_status], show_progress=False)

    return (
        demo,
        _modern_theme(dark=ctx.flags.theme == "dark"),
        _static_text("modern.css"),
        "",
    )
