from __future__ import annotations

import os
import queue
import threading
from pathlib import Path

import gradio as gr

from aiwf.bootstrap import AppContext
from aiwf.core.domain.engine import EngineTenant
from aiwf.infrastructure.safetensors_metadata import read_safetensors_metadata
from aiwf.services.model_download import CATEGORY_LABELS, browse_links_html, inspect_custom_input, is_unsafe_download_format
from aiwf.services.model_download_catalog import QUICK_START_BUNDLES
from aiwf.services.model_info_lookup import get_model_info_lookup
from aiwf.services.model_ops import ModelOpsService, PreflightResult, inspect_model_asset
from aiwf.services.process_supervisor import get_process_supervisor
from aiwf.services.civitai_browser import CivitAIBrowser
from aiwf.web.registry import WebRegistry


def _model_op_tenant(preflight: PreflightResult) -> EngineTenant | None:
    """Return the GPU tenant needed by a model operation, if any."""
    if preflight.command is None:
        return None
    if preflight.command.name in {"model-ops-lora-fuse", "model-ops-convert"}:
        return EngineTenant.IMAGE
    return None


def _cn_download_choices(ctx: AppContext) -> list[tuple[str, str]]:
    choices = []
    for item in ctx.controlnet.list_downloadable():
        mark = "  ✓ installed" if ctx.controlnet.is_installed(item) else ""
        choices.append((f"{item.title} · {item.size_mb}MB{mark}", item.key))
    return choices


def _catalog_choices(ctx: AppContext, categories: list[str] | None = None) -> list[tuple[str, str]]:
    allowed = set(categories or CATEGORY_LABELS.keys())
    choices: list[tuple[str, str]] = []
    for item in ctx.model_download.list_catalog():
        if item.category not in allowed:
            continue
        choices.append(
            (item.choice_label(installed=ctx.model_download.is_catalog_installed(item)), item.key)
        )
    return choices


def _catalog_category_label(key: str | None, ctx: AppContext) -> str:
    if not key:
        return "_Pick a catalog model to see its save category._"
    item = ctx.model_download.find_catalog(key)
    if item is None:
        return "_Unknown catalog entry._"
    folder = ctx.model_download.destination_dir(item.category)
    label = CATEGORY_LABELS.get(item.category, item.category)
    notes = f"  \n_{item.notes}_" if item.notes else ""
    return f"**Category:** {label}  \n**Folder:** `{folder}`{notes}"


def _run_download(worker, progress_q: queue.Queue, result: dict) -> None:
    def _worker():
        try:
            worker(on_progress=lambda done, total: progress_q.put((done, total)))
            result["ok"] = True
        except Exception as exc:
            result["error"] = str(exc)
        finally:
            progress_q.put(None)

    threading.Thread(target=_worker, daemon=True).start()


def _cn_installed_md(ctx: AppContext) -> str:
    installed = [item for item in ctx.controlnet.list_downloadable() if ctx.controlnet.is_installed(item)]
    extra = [
        m
        for m in ctx.controlnet.list_models()
        if not any(Path(m.path).name == i.filename for i in installed)
    ]
    if not installed and not extra:
        return "_No ControlNet models yet. Pick one above and download._"
    lines = [f"- **{item.title}** → `{item.filename}`" for item in installed]
    lines += [f"- {m.title} → `{m.path}`" for m in extra]
    return "**Installed ControlNet models**  \n" + "  \n".join(lines)


def register_model_manager(registry: WebRegistry) -> None:
    @registry.tab("Models", order=10)
    def build(ctx: AppContext, tab: gr.Tab | None = None) -> None:
        enable_wip_model_ops = os.environ.get("AIWF_ENABLE_WIP_MODEL_OPS", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        enable_wip_tabs = os.environ.get("AIWF_ENABLE_WIP_TABS", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        catalog = ctx.models

        with gr.Column(elem_classes=["aiwf-models"]):
            with gr.Column(elem_classes=["aiwf-page-header"]):
                gr.Markdown("Models", elem_classes=["aiwf-section-label"])
                gr.Markdown(
                    "Manage downloads, checkpoints, LoRAs, and model utilities.",
                    elem_classes=["aiwf-page-intro"],
                )
                folder_help = gr.Markdown(catalog.models_folder_help(), elem_classes=["aiwf-page-path"])

            with gr.Tabs(elem_classes=["aiwf-model-tabs"]):
                with gr.Tab("Download", elem_classes=["aiwf-model-tab"]):
                    dl = ctx.model_download
                    all_categories = [key for _, key in dl.category_choices()]
                    with gr.Column(elem_classes=["aiwf-panel"]):
                        gr.Markdown("Download models", elem_classes=["aiwf-section-label"])
                        gr.HTML(browse_links_html(), elem_classes=["aiwf-external-links-wrap"])
                        dl_folder_help = gr.Markdown(dl.folder_paths_help(), elem_classes=["aiwf-settings-paths"])

                    with gr.Column(elem_classes=["aiwf-panel"]):
                        gr.Markdown("Quick start", elem_classes=["aiwf-section-label"])
                        gr.Markdown(
                            "Download the smallest usable model set for a feature.",
                            elem_classes=["aiwf-settings-paths"],
                        )
                        with gr.Row():
                            qs_video_btn  = gr.Button("Video (Wan 2.2 I2V)",  variant="secondary")
                            qs_rife_btn   = gr.Button(
                                "Frame interpolation",
                                variant="secondary",
                            )
                            qs_seg_btn    = gr.Button("Segmentation (SAM)",    variant="secondary")
                            qs_fs_btn     = gr.Button(
                                "Face swap (ReActor)",
                                variant="secondary",
                                interactive=enable_wip_tabs,
                            )
                        with gr.Row():
                            qs_sd_btn     = gr.Button("Text-to-image SD1.5",   variant="secondary")
                            qs_sdxl_btn   = gr.Button("Text-to-image SDXL",    variant="secondary")
                        qs_status = gr.Markdown("", elem_classes=["aiwf-status-bar"])

                    with gr.Column(elem_classes=["aiwf-panel"]):
                        gr.Markdown("Curated catalog", elem_classes=["aiwf-section-label"])
                        gr.Markdown(
                            "Starter Hugging Face, CivitAI, and direct-download models.",
                            elem_classes=["aiwf-settings-paths"],
                        )
                        catalog_filter = gr.CheckboxGroup(
                            label="Show categories",
                            choices=dl.category_choices(),
                            value=all_categories,
                        )
                        with gr.Row():
                            catalog_select = gr.Dropdown(
                                label="Catalog model",
                                choices=_catalog_choices(ctx, all_categories),
                                value=None,
                                scale=4,
                            )
                            catalog_dl_btn = gr.Button("Download", variant="primary", scale=1)
                        catalog_category = gr.Markdown(
                            _catalog_category_label(None, ctx),
                            elem_classes=["aiwf-settings-hint"],
                        )
                        catalog_status = gr.Markdown("", elem_classes=["aiwf-status-bar"])

                    with gr.Column(elem_classes=["aiwf-panel"]):
                        gr.Markdown("Custom download", elem_classes=["aiwf-section-label"])
                        gr.Markdown(
                            "Paste a Hugging Face repo, CivitAI URL, or direct file URL.",
                            elem_classes=["aiwf-settings-paths"],
                        )
                        custom_source = gr.Radio(
                            label="Source",
                            choices=[
                                ("Hugging Face", "huggingface"),
                                ("CivitAI", "civitai"),
                                ("Direct URL", "direct"),
                            ],
                            value="huggingface",
                        )
                        custom_url = gr.Textbox(
                            label="Repo or URL",
                            placeholder="runwayml/stable-diffusion-v1-5 or https://civitai.com/models/4384",
                            lines=1,
                        )
                        custom_filename = gr.Textbox(
                            label="Hugging Face file path (optional)",
                            placeholder="v1-5-pruned-emaonly.safetensors",
                            info="Required for HF file downloads. Leave empty only for Wan Diffusers folder repos.",
                        )
                        custom_category = gr.Radio(
                            label="Save as category",
                            choices=dl.category_choices(),
                            value="checkpoint",
                            info="Determines the destination folder under models/.",
                        )
                        with gr.Row():
                            custom_check_btn = gr.Button("Check URL", elem_classes=["aiwf-btn-ghost"])
                            custom_dl_btn = gr.Button("Download custom", variant="primary")
                        custom_status = gr.Markdown("", elem_classes=["aiwf-status-bar"])

                with gr.Tab("Checkpoints", elem_classes=["aiwf-model-tab"]):
                    with gr.Row(elem_classes=["aiwf-panel"]):
                        ckpt_select = gr.Dropdown(
                            label="Checkpoint",
                            choices=catalog.checkpoint_choices(),
                            value=catalog.checkpoint_choices()[0][1] if catalog.checkpoint_choices() else None,
                            allow_custom_value=False,
                            scale=4,
                        )
                        ckpt_refresh = gr.Button("Refresh", elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"], scale=1)

                    ckpt_details = gr.Markdown(
                        catalog.checkpoint_details(catalog.find_checkpoint(
                            catalog.checkpoint_choices()[0][1] if catalog.checkpoint_choices() else None
                        )),
                        elem_classes=["aiwf-model-details"],
                    )

                with gr.Tab("LoRAs", elem_classes=["aiwf-model-tab"]):
                    with gr.Row(elem_classes=["aiwf-panel"]):
                        lora_select = gr.Dropdown(
                            label="LoRA",
                            choices=catalog.lora_choices(),
                            value=catalog.lora_choices()[0][1] if catalog.lora_choices() else None,
                            allow_custom_value=False,
                            scale=4,
                        )
                        lora_refresh = gr.Button("Refresh", elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"], scale=1)

                    lora_details = gr.Markdown(
                        catalog.lora_details(catalog.find_lora(
                            catalog.lora_choices()[0][1] if catalog.lora_choices() else None
                        )),
                        elem_classes=["aiwf-model-details"],
                    )

                    with gr.Column(elem_classes=["aiwf-panel", "aiwf-lora-config"]):
                        gr.Markdown("LoRA shortcut & prompt words", elem_classes=["aiwf-section-label"])
                        gr.Markdown(
                            "Set the alias and trigger words used by `*lora:alias` in Image prompts.",
                            elem_classes=["aiwf-settings-paths"],
                        )
                        with gr.Row():
                            lora_alias = gr.Textbox(
                                label="Shortcut alias",
                                placeholder="clearskin",
                                info="Used as *lora:clearskin — lowercase, no spaces",
                                scale=2,
                            )
                            lora_strength = gr.Slider(
                                0,
                                2,
                                value=1.0,
                                step=0.05,
                                label="Default strength",
                                info="Weight applied when the shortcut expands",
                                scale=2,
                            )
                        lora_keywords = gr.Textbox(
                            label="Trigger words",
                            lines=2,
                            placeholder="clear skin, smooth skin",
                            info="Inserted into the prompt when the shortcut is used. Edit to match your workflow.",
                        )
                        with gr.Row():
                            save_lora = gr.Button("Save LoRA settings", variant="primary")
                            copy_shortcut = gr.Button("Copy shortcut", elem_classes=["aiwf-btn-ghost"])

                    lora_status = gr.Markdown("", elem_classes=["aiwf-settings-hint"])

                with gr.Tab("Model Mixing", elem_classes=["aiwf-model-tab"]):
                    model_ops = ModelOpsService(ctx.flags)
                    mix_command_state = gr.State(None)
                    with gr.Column(elem_classes=["aiwf-panel"]):
                        gr.Markdown("LoRA fuse", elem_classes=["aiwf-section-label"])
                        if not enable_wip_model_ops:
                            gr.Markdown(
                                "Model mixing is disabled on stable `main`. Use the `dev` branch after backing up models.",
                                elem_classes=["aiwf-settings-paths"],
                            )
                        mix_base = gr.Dropdown(
                            label="Base checkpoint",
                            choices=catalog.checkpoint_choices(),
                            value=catalog.checkpoint_choices()[0][1] if catalog.checkpoint_choices() else None,
                            allow_custom_value=False,
                        )
                        mix_lora_slots = []
                        with gr.Accordion("LoRA stack", open=True, elem_classes=["aiwf-prompt-tools"]):
                            for index in range(4):
                                with gr.Row(elem_classes=["aiwf-lora-stack-row"]):
                                    mix_lora_pick = gr.Dropdown(
                                        label=f"LoRA {index + 1}",
                                        choices=catalog.lora_choices(),
                                        value=None,
                                        scale=4,
                                    )
                                    mix_lora_weight = gr.Slider(
                                        0.0,
                                        2.0,
                                        value=1.0,
                                        step=0.05,
                                        label="Weight",
                                        scale=2,
                                    )
                                    mix_lora_slots.extend([mix_lora_pick, mix_lora_weight])
                        mix_lora_output = gr.Textbox(label="Output folder name", value="fused_lora_model")
                        with gr.Row():
                            mix_lora_check = gr.Button(
                                "Check LoRA fuse",
                                elem_classes=["aiwf-btn-ghost"],
                                interactive=enable_wip_model_ops,
                            )
                            mix_lora_run = gr.Button(
                                "Run LoRA fuse",
                                variant="primary",
                                interactive=enable_wip_model_ops,
                            )

                    with gr.Column(elem_classes=["aiwf-panel"]):
                        gr.Markdown("Checkpoint blend", elem_classes=["aiwf-section-label"])
                        with gr.Row():
                            blend_left = gr.Dropdown(
                                label="First checkpoint",
                                choices=catalog.checkpoint_choices(),
                                value=catalog.checkpoint_choices()[0][1] if catalog.checkpoint_choices() else None,
                                allow_custom_value=False,
                            )
                            blend_right = gr.Dropdown(
                                label="Second checkpoint",
                                choices=catalog.checkpoint_choices(),
                                value=catalog.checkpoint_choices()[1][1] if len(catalog.checkpoint_choices()) > 1 else None,
                                allow_custom_value=False,
                            )
                        blend_ratio = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="Second checkpoint weight")
                        blend_output = gr.Textbox(label="Output checkpoint name", value="blended_model")
                        with gr.Row():
                            blend_check = gr.Button(
                                "Check blend",
                                elem_classes=["aiwf-btn-ghost"],
                                interactive=enable_wip_model_ops,
                            )
                            blend_run = gr.Button(
                                "Run blend",
                                variant="primary",
                                interactive=enable_wip_model_ops,
                            )
                    mix_status = gr.Markdown("", elem_classes=["aiwf-status-bar"])

                with gr.Tab("Conversion & Quantization", elem_classes=["aiwf-model-tab"]):
                    conversion_command_state = gr.State(None)
                    with gr.Column(elem_classes=["aiwf-panel"]):
                        gr.Markdown("Model type change", elem_classes=["aiwf-section-label"])
                        if not enable_wip_model_ops:
                            gr.Markdown(
                                "Conversion and quantization jobs are disabled on stable `main`. Use `dev` for experiments.",
                                elem_classes=["aiwf-settings-paths"],
                            )
                        convert_source = gr.Textbox(
                            label="Source path",
                            placeholder=str(ctx.flags.resolved_ckpt_dir()),
                        )
                        convert_arch = gr.Radio(
                            label="Architecture hint",
                            choices=[("Auto/unknown", "unknown"), ("SD 1.x", "sd15"), ("SDXL", "sdxl"), ("Wan/video", "wan"), ("Flux", "flux")],
                            value="unknown",
                        )
                        convert_operation = gr.Radio(
                            label="Operation",
                            choices=[
                                ("Single file -> Diffusers folder", "single-to-diffusers"),
                                ("Diffusers folder -> single file", "diffusers-to-single"),
                                ("Diffusers/single -> ONNX folder", "onnx-export"),
                            ],
                            value="single-to-diffusers",
                        )
                        convert_output = gr.Textbox(label="Output name", value="converted_model")
                        with gr.Row():
                            convert_inspect = gr.Button(
                                "Inspect source",
                                elem_classes=["aiwf-btn-ghost"],
                                interactive=enable_wip_model_ops,
                            )
                            convert_check = gr.Button(
                                "Check conversion",
                                elem_classes=["aiwf-btn-ghost"],
                                interactive=enable_wip_model_ops,
                            )
                            convert_run = gr.Button(
                                "Run conversion",
                                variant="primary",
                                interactive=enable_wip_model_ops,
                            )
                        convert_status = gr.Markdown("", elem_classes=["aiwf-status-bar"])

                    quant_command_state = gr.State(None)
                    with gr.Column(elem_classes=["aiwf-panel"]):
                        gr.Markdown("Quantization preflight", elem_classes=["aiwf-section-label"])
                        quant_source = gr.Textbox(
                            label="Source path",
                            placeholder=str(ctx.flags.resolved_ckpt_dir()),
                        )
                        quant_target = gr.Radio(
                            label="Target component",
                            choices=[("Full image model", "model"), ("VAE (preflight only)", "vae"), ("Text encoder", "text_encoder")],
                            value="model",
                        )
                        quant_arch = gr.Radio(
                            label="Architecture hint",
                            choices=[("Auto/unknown", "unknown"), ("SD 1.x", "sd15"), ("SDXL", "sdxl"), ("Wan/video", "wan"), ("Flux", "flux")],
                            value="unknown",
                        )
                        quant_choice = gr.Radio(
                            label="Quant/storage choice",
                            choices=[
                                ("BF16 compatibility", "bf16"),
                                ("FP16 compatibility", "fp16"),
                                ("FP8 experiment", "fp8"),
                                ("INT8 experiment", "int8"),
                                ("NVFP4 storage/compression only", "nvfp4"),
                                ("GGUF later lane", "gguf"),
                            ],
                            value="bf16",
                        )
                        quant_output = gr.Textbox(label="Output name", value="quantized_model")
                        with gr.Row():
                            quant_check = gr.Button(
                                "Check quantization",
                                elem_classes=["aiwf-btn-ghost"],
                                interactive=enable_wip_model_ops,
                            )
                            quant_run = gr.Button(
                                "Run quantization job",
                                variant="primary",
                                interactive=enable_wip_model_ops,
                            )
                        quant_status = gr.Markdown("", elem_classes=["aiwf-status-bar"])

                with gr.Tab("ControlNet", elem_classes=["aiwf-model-tab"]):
                    with gr.Column(elem_classes=["aiwf-panel"]):
                        gr.Markdown("Download ControlNet models", elem_classes=["aiwf-section-label"])
                        gr.Markdown(
                            "SD1.5 ControlNet-v1.1 Light checkpoints download "
                            f"into `{ctx.controlnet.models_dir()}` and appear in Image Advanced.",
                            elem_classes=["aiwf-settings-paths"],
                        )
                        with gr.Row():
                            cn_dl_select = gr.Dropdown(
                                label="ControlNet to download",
                                choices=_cn_download_choices(ctx),
                                value=None,
                                scale=4,
                            )
                            cn_dl_btn = gr.Button("Download", variant="primary", scale=1)
                            cn_dl_refresh = gr.Button(
                                "Refresh", elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"], scale=1
                            )
                        cn_dl_status = gr.Markdown("", elem_classes=["aiwf-status-bar"])
                        cn_installed = gr.Markdown(_cn_installed_md(ctx), elem_classes=["aiwf-model-details"])

                with gr.Tab("Model Info", elem_classes=["aiwf-model-tab"]):
                    with gr.Column(elem_classes=["aiwf-panel"]):
                        gr.Markdown("Model Info Lookup", elem_classes=["aiwf-section-label"])
                        gr.Markdown(
                            "Fetch metadata from Hugging Face, CivitAI, or Ollama.",
                            elem_classes=["aiwf-settings-paths"],
                        )
                        with gr.Row():
                            info_query = gr.Textbox(
                                label="Model ID / URL / name",
                                placeholder=(
                                    "stabilityai/stable-diffusion-xl-base-1.0  "
                                    "or  https://civitai.com/models/4384  "
                                    "or  llama3:8b"
                                ),
                                scale=8,
                            )
                            info_lookup_btn = gr.Button("Look up", variant="primary", scale=1)
                        info_result = gr.Markdown("", elem_classes=["aiwf-model-details"])

                with gr.Tab("Browse", elem_classes=["aiwf-model-tab"]):
                    _civitai = CivitAIBrowser(
                        api_token=getattr(ctx.settings, "civitai_token", None)
                    )
                    with gr.Tabs():
                        # ── Installed ──────────────────────────────────────
                        with gr.Tab("Installed"):
                            with gr.Column(elem_classes=["aiwf-panel"]):
                                gr.Markdown(
                                    "Model files found on disk",
                                    elem_classes=["aiwf-section-label"],
                                )
                                gr.Markdown(
                                    "Model files grouped by folder.",
                                    elem_classes=["aiwf-settings-paths"],
                                )
                                with gr.Row():
                                    installed_refresh_btn = gr.Button(
                                        "Refresh", elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"]
                                    )
                                installed_result = gr.Markdown(
                                    _civitai.installed_summary(ctx.flags),
                                    elem_classes=["aiwf-model-details"],
                                )

                        # ── HF Hub ─────────────────────────────────────────
                        with gr.Tab("HF Hub"):
                            with gr.Column(elem_classes=["aiwf-panel"]):
                                gr.Markdown(
                                    "Search Hugging Face",
                                    elem_classes=["aiwf-section-label"],
                                )
                                gr.Markdown(
                                    "Enter a repo ID or keyword. Add an HF token in Settings for private repos.",
                                    elem_classes=["aiwf-settings-paths"],
                                )
                                with gr.Row():
                                    hf_query = gr.Textbox(
                                        label="Repo ID or keyword",
                                        placeholder="stabilityai/stable-diffusion-xl-base-1.0",
                                        scale=7,
                                    )
                                    hf_search_btn = gr.Button(
                                        "Look up", variant="primary", scale=1
                                    )
                                hf_result = gr.Markdown("", elem_classes=["aiwf-model-details"])

                        # ── CivitAI ────────────────────────────────────────
                        with gr.Tab("CivitAI"):
                            with gr.Column(elem_classes=["aiwf-panel"]):
                                gr.Markdown(
                                    "Browse CivitAI",
                                    elem_classes=["aiwf-section-label"],
                                )
                                gr.Markdown(
                                    "Search CivitAI. Add a token in Settings for private/authenticated downloads.",
                                    elem_classes=["aiwf-settings-paths"],
                                )
                                with gr.Row():
                                    civitai_query = gr.Textbox(
                                        label="Search",
                                        placeholder="DreamShaper, realistic portrait, anime ...",
                                        scale=5,
                                    )
                                    civitai_type = gr.Dropdown(
                                        label="Type",
                                        choices=[
                                            ("All", ""),
                                            ("Checkpoint", "Checkpoint"),
                                            ("LoRA", "LORA"),
                                            ("Embedding", "TextualInversion"),
                                            ("ControlNet", "Controlnet"),
                                            ("VAE", "VAE"),
                                            ("Upscaler", "Upscaler"),
                                        ],
                                        value="",
                                        scale=2,
                                    )
                                    civitai_search_btn = gr.Button(
                                        "Search", variant="primary", scale=1
                                    )
                                with gr.Row():
                                    civitai_nsfw = gr.Checkbox(
                                        label="Show NSFW models",
                                        value=False,
                                        scale=3,
                                    )
                                    civitai_sort = gr.Dropdown(
                                        label="Sort",
                                        choices=[
                                            ("Most downloaded", "Most Downloaded"),
                                            ("Highest rated", "Highest Rated"),
                                            ("Newest", "Newest"),
                                        ],
                                        value="Most Downloaded",
                                        scale=3,
                                    )
                                gr.Markdown(
                                    "**Quick presets**",
                                    elem_classes=["aiwf-settings-paths"],
                                )
                                with gr.Row():
                                    civitai_preset_ckpt = gr.Button(
                                        "Top Checkpoints",
                                        elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"],
                                        scale=1,
                                    )
                                    civitai_preset_lora = gr.Button(
                                        "Top LoRAs",
                                        elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"],
                                        scale=1,
                                    )
                                    civitai_preset_vae = gr.Button(
                                        "Top VAEs",
                                        elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"],
                                        scale=1,
                                    )
                                    civitai_preset_newest = gr.Button(
                                        "Newest",
                                        elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"],
                                        scale=1,
                                    )
                                # Carousel: thumbnails from search results
                                civitai_gallery = gr.Gallery(
                                    label="Preview images (click to view model details)",
                                    columns=5,
                                    height=280,
                                    object_fit="cover",
                                    show_label=True,
                                    visible=False,
                                    elem_classes=["aiwf-civitai-gallery"],
                                )
                                civitai_result = gr.Markdown(
                                    "", elem_classes=["aiwf-model-details"]
                                )
                                # Pagination state: cursor string, empty = first page
                                civitai_cursor_state = gr.State(value="")
                                civitai_prev_cursor_state = gr.State(value="")
                                # Gallery index map: list[int] mapping gallery idx -> result.models idx
                                civitai_gallery_map = gr.State(value=[])
                                # Last search result cached for gallery click lookup
                                civitai_last_result = gr.State(value=None)
                                with gr.Row():
                                    civitai_prev_btn = gr.Button(
                                        "◀ Previous", visible=False,
                                        elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"],
                                        scale=1,
                                    )
                                    civitai_next_btn = gr.Button(
                                        "Next ▶", visible=False,
                                        elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"],
                                        scale=1,
                                    )


        def _download_bundle(bundle_key: str):
            keys = QUICK_START_BUNDLES.get(bundle_key, [])
            if not keys:
                yield f"Unknown bundle: {bundle_key}"
                return
            lines: list[str] = []
            total = len(keys)
            for i, key in enumerate(keys):
                item = ctx.model_download.find_catalog(key)
                if item is None:
                    lines.append(f"⚠ Unknown catalog key: `{key}`")
                    yield "\n\n".join(lines)
                    continue
                prefix = f"[{i + 1}/{total}] **{item.title}**"
                if ctx.model_download.is_catalog_installed(item):
                    lines.append(f"✓ {prefix} — already installed")
                    yield "\n\n".join(lines)
                    continue
                lines.append(f"⬇ {prefix} — downloading…")
                yield "\n\n".join(lines)
                progress_q: queue.Queue = queue.Queue()
                result: dict = {}

                def _work(on_progress, _key=key):
                    ctx.model_download.download_catalog(_key, on_progress=on_progress)

                _run_download(_work, progress_q, result)
                while True:
                    update = progress_q.get()
                    if update is None:
                        break
                    done, total_bytes = update
                    if total_bytes:
                        pct = int(100 * done / total_bytes)
                        lines[-1] = f"⬇ {prefix} — {pct}% ({done / 1_000_000:.0f} MB)"
                        yield "\n\n".join(lines)

                if result.get("error"):
                    lines[-1] = f"✗ {prefix} — {result['error']}"
                else:
                    folder = ctx.model_download.destination_dir(item.category)
                    lines[-1] = f"✓ {prefix} → `{folder}`"
                yield "\n\n".join(lines)
            yield "\n\n".join(lines) + "\n\n**Done.**"

        for _btn, _key in [
            (qs_video_btn,  "video"),
            (qs_rife_btn,   "rife"),
            (qs_seg_btn,    "seg"),
            (qs_fs_btn,     "faceswap"),
            (qs_sd_btn,     "sd"),
            (qs_sdxl_btn,   "sdxl"),
        ]:
            _btn.click(_download_bundle, inputs=gr.State(_key), outputs=[qs_status], show_progress="minimal")

        def _refresh_catalog_choices(categories: list[str]):
            allowed = categories or all_categories
            return gr.update(choices=_catalog_choices(ctx, allowed))

        catalog_filter.change(
            _refresh_catalog_choices,
            inputs=[catalog_filter],
            outputs=[catalog_select],
            show_progress=False,
        )

        def _show_catalog_category(key: str | None):
            return _catalog_category_label(key, ctx)

        catalog_select.change(
            _show_catalog_category,
            inputs=[catalog_select],
            outputs=[catalog_category],
            show_progress=False,
        )

        def _inspect_url(source: str, url: str, filename: str):
            new_source, new_url, new_filename, status = inspect_custom_input(
                source=source,
                url_or_repo=url,
                filename=filename,
            )
            fname = new_filename or filename or ""
            if fname and is_unsafe_download_format(fname) and getattr(ctx.settings, "prefer_safetensors", True):
                unsafe_note = (
                    "\n\n⚠ **Unsafe format:** `.ckpt` / `.pt` files are Python pickles and can "
                    "run arbitrary code when loaded. Prefer `.safetensors` when available."
                )
                status = (status or "") + unsafe_note
            return (
                gr.update(value=new_source),
                gr.update(value=new_url or url),
                gr.update(value=new_filename or filename),
                status,
            )

        custom_url.change(
            _inspect_url,
            inputs=[custom_source, custom_url, custom_filename],
            outputs=[custom_source, custom_url, custom_filename, custom_status],
            show_progress=False,
        )
        custom_filename.change(
            _inspect_url,
            inputs=[custom_source, custom_url, custom_filename],
            outputs=[custom_source, custom_url, custom_filename, custom_status],
            show_progress=False,
        )
        custom_check_btn.click(
            _inspect_url,
            inputs=[custom_source, custom_url, custom_filename],
            outputs=[custom_source, custom_url, custom_filename, custom_status],
            show_progress=False,
        )

        def _download_catalog(key: str | None, categories: list[str]):
            if not key:
                raise gr.Error("Pick a catalog model to download.")
            item = ctx.model_download.find_catalog(key)
            if item is None:
                raise gr.Error("Unknown catalog model.")
            if ctx.model_download.is_catalog_installed(item):
                yield f"**{item.title}** is already installed.", gr.update()
                return

            progress_q: queue.Queue = queue.Queue()
            result: dict = {}

            def work(on_progress):
                ctx.model_download.download_catalog(key, on_progress=on_progress)

            _run_download(work, progress_q, result)
            yield f"**Downloading {item.title}…**", gr.update()

            while True:
                update = progress_q.get()
                if update is None:
                    break
                done, total = update
                if total:
                    pct = int(100 * done / total)
                    yield (
                        f"**Downloading {item.title}…** {pct}% ({done / 1_000_000:.1f} MB)",
                        gr.update(),
                    )

            if result.get("error"):
                yield f"**Download failed** — {result['error']}", gr.update()
                return
            folder = ctx.model_download.destination_dir(item.category)
            yield (
                f"**{item.title} installed** → `{folder}`",
                gr.update(choices=_catalog_choices(ctx, categories or all_categories)),
            )

        catalog_dl_btn.click(
            _download_catalog,
            inputs=[catalog_select, catalog_filter],
            outputs=[catalog_status, catalog_select],
            show_progress="minimal",
        )

        def _download_custom(source: str, url: str, filename: str, category: str):
            if not (url or "").strip():
                raise gr.Error("Enter a repo or URL.")
            if category not in CATEGORY_LABELS:
                raise gr.Error("Pick a save category.")
            label = CATEGORY_LABELS[category]
            progress_q: queue.Queue = queue.Queue()
            result: dict = {}
            dest_path: dict[str, str] = {}

            def work(on_progress):
                path = ctx.model_download.download_custom(
                    source=source,
                    url_or_repo=url,
                    category=category,
                    filename=filename,
                    on_progress=on_progress,
                )
                dest_path["path"] = str(path)

            _run_download(work, progress_q, result)
            yield f"**Downloading to {label}…**"

            while True:
                update = progress_q.get()
                if update is None:
                    break
                done, total = update
                if total:
                    pct = int(100 * done / total)
                    yield f"**Downloading…** {pct}% ({done / 1_000_000:.1f} MB)"

            if result.get("error"):
                yield f"**Download failed** — {result['error']}"
                return
            yield f"**Saved** → `{dest_path.get('path', '')}`"

        custom_dl_btn.click(
            _download_custom,
            inputs=[custom_source, custom_url, custom_filename, custom_category],
            outputs=[custom_status],
            show_progress="minimal",
        )

        def _cn_refresh_list():
            return gr.update(choices=_cn_download_choices(ctx)), _cn_installed_md(ctx)

        cn_dl_refresh.click(_cn_refresh_list, outputs=[cn_dl_select, cn_installed], show_progress=False)

        def _cn_download(key):
            if not key:
                raise gr.Error("Pick a ControlNet model to download.")
            item = ctx.controlnet.find_downloadable(key)
            if item is None:
                raise gr.Error("Unknown ControlNet model.")
            if ctx.controlnet.is_installed(item):
                yield (
                    f"**{item.title}** is already installed.",
                    gr.update(choices=_cn_download_choices(ctx)),
                    _cn_installed_md(ctx),
                )
                return

            progress_q: queue.Queue = queue.Queue()
            result: dict = {}

            def worker():
                try:
                    ctx.controlnet.download_model(
                        key,
                        on_progress=lambda done, total: progress_q.put((done, total)),
                    )
                    result["ok"] = True
                except Exception as exc:  # surfaced to the UI below
                    result["error"] = str(exc)
                finally:
                    progress_q.put(None)

            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            yield f"**Downloading {item.title}…**", gr.update(), gr.update()

            while True:
                update = progress_q.get()
                if update is None:
                    break
                done, total = update
                if total:
                    pct = int(100 * done / total)
                    yield (
                        f"**Downloading {item.title}…** {pct}% ({done / 1_000_000:.0f} MB)",
                        gr.update(),
                        gr.update(),
                    )

            if result.get("error"):
                yield (
                    f"**Download failed** — {result['error']}",
                    gr.update(choices=_cn_download_choices(ctx)),
                    _cn_installed_md(ctx),
                )
                return
            yield (
                f"**{item.title} installed.** It's ready in Image -> ControlNet.",
                gr.update(choices=_cn_download_choices(ctx)),
                _cn_installed_md(ctx),
            )

        cn_dl_btn.click(
            _cn_download,
            inputs=[cn_dl_select],
            outputs=[cn_dl_status, cn_dl_select, cn_installed],
            show_progress="minimal",
        )

        # ----------------------------------------------------------------
        # Model Info Lookup callbacks
        # ----------------------------------------------------------------

        def _do_info_lookup(query: str) -> str:
            query = (query or "").strip()
            if not query:
                return "_Enter a HuggingFace repo ID, CivitAI URL/number, or Ollama model name._"
            try:
                lookup = get_model_info_lookup()
                settings = getattr(ctx, "settings", None)
                hf_token = (getattr(settings, "huggingface_token", None) or "").strip()
                civitai_token = (getattr(settings, "civitai_token", None) or "").strip()
                result = lookup.lookup_auto(
                    query,
                    hf_token=hf_token,
                    civitai_token=civitai_token,
                )
                if result is None:
                    return (
                        f"_No result found for `{query}`.  \n"
                        "Check the ID/URL is correct, or that Ollama is running for local model names._"
                    )
                return result.summary_markdown()
            except Exception as exc:
                return f"_Lookup error: {exc}_"

        info_lookup_btn.click(
            _do_info_lookup,
            inputs=[info_query],
            outputs=[info_result],
            show_progress="minimal",
        )
        info_query.submit(
            _do_info_lookup,
            inputs=[info_query],
            outputs=[info_result],
            show_progress="minimal",
        )
        # ── Browse tab handlers ────────────────────────────────────────────

        def _refresh_installed():
            return _civitai.installed_summary(ctx.flags)

        installed_refresh_btn.click(
            _refresh_installed,
            outputs=[installed_result],
            show_progress="minimal",
        )

        def _hf_lookup(query: str):
            query = (query or "").strip()
            if not query:
                return "_Enter a Hugging Face repo ID (e.g. `stabilityai/stable-diffusion-xl-base-1.0`)._"
            from aiwf.services.model_info_lookup import get_model_info_lookup
            lookup = get_model_info_lookup(ctx)
            info = lookup.fetch(query)
            if info is None:
                return f"_Nothing found for `{query}`. Check the repo ID or network connection._"
            return info.summary_markdown()

        hf_search_btn.click(
            _hf_lookup,
            inputs=[hf_query],
            outputs=[hf_result],
            show_progress="minimal",
        )
        hf_query.submit(
            _hf_lookup,
            inputs=[hf_query],
            outputs=[hf_result],
            show_progress="minimal",
        )

        def _civitai_render(result, prefer_safe: bool) -> str:
            """Render a CivitAISearchResult to markdown with download links."""
            if not result.ok:
                return f"\u26a0 {result.error}"
            if not result.models:
                return "_No results found. Try a different keyword or type._"
            lines = [f"**{result.total_count} results** (showing {len(result.models)})\n"]
            for m in result.models:
                card = m.summary_markdown()
                ver = m.latest_version
                if ver and ver.download_url:
                    url = ver.download_url
                    ext = url.rsplit(".", 1)[-1].lower() if "." in url else ""
                    safe_ext = ext in {"safetensors", "gguf", "bin"}
                    if prefer_safe and not safe_ext:
                        card += (
                            f"\n\n> \u26a0 Download format `.{ext}` blocked "
                            "(prefer_safetensors is on). Find a `.safetensors` version "
                            "or disable in Settings."
                        )
                    else:
                        card += (
                            f"\n\n\U0001f4e5 **Download:** "
                            f"[{ver.name} ({ver.size_label()})]({url})"
                        )
                lines.append(card)
                lines.append("---")
            return "\n\n".join(lines)

        def _civitai_run(result, prefer_safe):
            """Shared output builder for all search/page functions."""
            next_cursor = result.next_cursor or ""
            md = _civitai_render(result, prefer_safe)
            # Gallery
            gallery_pairs = _civitai.gallery_images(result)
            gallery_map = _civitai.gallery_index_map(result)
            gallery_update = gr.update(
                value=gallery_pairs,
                visible=bool(gallery_pairs),
            )
            return md, next_cursor, gallery_update, gallery_map, result

        def _civitai_search(query: str, type_filter: str, show_nsfw: bool, sort: str):
            query = (query or "").strip()
            types = [type_filter] if type_filter else None
            prefer_safe = getattr(ctx.settings, "prefer_safetensors", True)
            result = _civitai.search(
                query, types=types, nsfw=show_nsfw, sort=sort, limit=10
            )
            md, next_cursor, gallery_update, gallery_map, res = _civitai_run(result, prefer_safe)
            return (
                md,
                next_cursor,
                "",
                gr.update(visible=bool(next_cursor)),
                gr.update(visible=False),
                gallery_update,
                gallery_map,
                res,
            )

        def _civitai_next(query: str, type_filter: str, show_nsfw: bool, sort: str,
                          cursor: str, prev_cursor: str):
            query = (query or "").strip()
            types = [type_filter] if type_filter else None
            prefer_safe = getattr(ctx.settings, "prefer_safetensors", True)
            result = _civitai.search(
                query, types=types, nsfw=show_nsfw, sort=sort, limit=10, cursor=cursor
            )
            md, next_cursor, gallery_update, gallery_map, res = _civitai_run(result, prefer_safe)
            return (
                md,
                next_cursor,
                cursor,
                gr.update(visible=bool(next_cursor)),
                gr.update(visible=True),
                gallery_update,
                gallery_map,
                res,
            )

        def _civitai_prev(query: str, type_filter: str, show_nsfw: bool, sort: str,
                          prev_cursor: str):
            query = (query or "").strip()
            types = [type_filter] if type_filter else None
            prefer_safe = getattr(ctx.settings, "prefer_safetensors", True)
            result = _civitai.search(
                query, types=types, nsfw=show_nsfw, sort=sort, limit=10,
                cursor=prev_cursor if prev_cursor else None,
            )
            md, next_cursor, gallery_update, gallery_map, res = _civitai_run(result, prefer_safe)
            return (
                md,
                next_cursor,
                "",
                gr.update(visible=bool(next_cursor)),
                gr.update(visible=False),
                gallery_update,
                gallery_map,
                res,
            )

        def _civitai_gallery_select(evt: gr.SelectData, gallery_map: list, last_result):
            """When user clicks a thumbnail, show that model's full detail card."""
            if last_result is None or not gallery_map:
                return ""
            gallery_idx = evt.index
            if gallery_idx >= len(gallery_map):
                return ""
            model_idx = gallery_map[gallery_idx]
            if model_idx >= len(last_result.models):
                return ""
            model = last_result.models[model_idx]
            prefer_safe = getattr(ctx.settings, "prefer_safetensors", True)
            card = model.summary_markdown()
            ver = model.latest_version
            if ver and ver.download_url:
                url = ver.download_url
                ext = url.rsplit(".", 1)[-1].lower() if "." in url else ""
                safe_ext = ext in {"safetensors", "gguf", "bin"}
                if prefer_safe and not safe_ext:
                    card += (
                        f"\n\n> \u26a0 Format `.{ext}` blocked (prefer_safetensors). "
                        "Find a `.safetensors` version or disable in Settings."
                    )
                else:
                    card += f"\n\n\U0001f4e5 **Download:** [{ver.name} ({ver.size_label()})]({url})"
            return card

        _civitai_outputs = [
            civitai_result,
            civitai_cursor_state,
            civitai_prev_cursor_state,
            civitai_next_btn,
            civitai_prev_btn,
            civitai_gallery,
            civitai_gallery_map,
            civitai_last_result,
        ]

        civitai_search_btn.click(
            _civitai_search,
            inputs=[civitai_query, civitai_type, civitai_nsfw, civitai_sort],
            outputs=_civitai_outputs,
            show_progress="minimal",
        )
        civitai_query.submit(
            _civitai_search,
            inputs=[civitai_query, civitai_type, civitai_nsfw, civitai_sort],
            outputs=_civitai_outputs,
            show_progress="minimal",
        )
        civitai_next_btn.click(
            _civitai_next,
            inputs=[civitai_query, civitai_type, civitai_nsfw, civitai_sort,
                    civitai_cursor_state, civitai_prev_cursor_state],
            outputs=_civitai_outputs,
            show_progress="minimal",
        )
        civitai_prev_btn.click(
            _civitai_prev,
            inputs=[civitai_query, civitai_type, civitai_nsfw, civitai_sort,
                    civitai_prev_cursor_state],
            outputs=_civitai_outputs,
            show_progress="minimal",
        )


        civitai_gallery.select(
            _civitai_gallery_select,
            inputs=[civitai_gallery_map, civitai_last_result],
            outputs=[civitai_result],
            show_progress=False,
        )

        def _run_model_op(preflight: PreflightResult | None):
            if preflight is None or preflight.command is None:
                yield "_Run a successful preflight first._"
                return
            if not preflight.ok:
                yield preflight.markdown()
                return
            process_supervisor = get_process_supervisor()
            tenant = _model_op_tenant(preflight)
            lines = [preflight.markdown(), "\n**Log**"]
            yield "\n".join(lines)
            try:
                if tenant is None:
                    for line in process_supervisor.start(preflight.command.name, preflight.command, check=True):
                        lines.append(f"`{line}`")
                        yield "\n".join(lines[-60:])
                else:
                    with ctx.supervisor.tenant_session(
                        tenant,
                        reason=f"Model operation: {preflight.command.name}",
                    ):
                        for line in process_supervisor.start(preflight.command.name, preflight.command, check=True):
                            lines.append(f"`{line}`")
                            yield "\n".join(lines[-60:])
                lines.append("\n**Done.** Refresh model lists if you created a new model.")
                yield "\n".join(lines[-60:])
            except Exception as exc:
                lines.append(f"\n**Error:** {exc}")
                yield "\n".join(lines[-60:])

        def _selected_loras_from_slots(*slot_values):
            selected = []
            weights = []
            seen = set()
            for lora_id, weight in zip(slot_values[0::2], slot_values[1::2]):
                if not lora_id or lora_id in seen:
                    continue
                lora = catalog.find_lora(lora_id)
                if lora is None:
                    continue
                seen.add(lora_id)
                selected.append(lora)
                weights.append(str(float(weight)))
            return selected, ",".join(weights)

        def _check_lora_fuse(base_id, output_name, *slot_values):
            loras, weights = _selected_loras_from_slots(*slot_values)
            result = model_ops.preflight_lora_fuse(
                catalog.find_checkpoint(base_id),
                loras,
                weights=weights,
                output_name=output_name,
            )
            return result.markdown(), result

        def _check_blend(left_id, right_id, ratio, output_name):
            result = model_ops.preflight_checkpoint_blend(
                catalog.find_checkpoint(left_id),
                catalog.find_checkpoint(right_id),
                ratio=float(ratio),
                output_name=output_name,
            )
            return result.markdown(), result

        def _inspect_source(path, arch):
            asset = inspect_model_asset(path or "", architecture=arch)
            lines = [
                f"**Path:** `{asset.path}`",
                f"**Exists:** {'yes' if asset.exists else 'no'}",
                f"**Family:** {asset.family}",
                f"**Storage:** {asset.storage}",
                f"**Architecture:** {asset.architecture}",
                f"**Dtype hint:** {asset.dtype_hint}",
                f"**Quant hint:** {asset.quant_hint}",
            ]
            if asset.metadata:
                lines.append(f"**Safetensors metadata keys:** {len(asset.metadata)}")
            return "\n".join(lines)

        def _check_conversion(path, operation, output_name, arch):
            result = model_ops.preflight_conversion(
                source_path=path,
                operation=operation,
                output_name=output_name,
                architecture=arch,
            )
            return result.markdown(), result

        def _check_quant(path, target, quant, output_name, arch):
            result = model_ops.preflight_quantization(
                source_path=path,
                target=target,
                quant=quant,
                output_name=output_name,
                architecture=arch,
            )
            return result.markdown(), result

        mix_lora_check.click(
            _check_lora_fuse,
            inputs=[mix_base, mix_lora_output, *mix_lora_slots],
            outputs=[mix_status, mix_command_state],
            show_progress=False,
        )
        mix_lora_run.click(
            _run_model_op,
            inputs=[mix_command_state],
            outputs=[mix_status],
            show_progress="minimal",
        )
        blend_check.click(
            _check_blend,
            inputs=[blend_left, blend_right, blend_ratio, blend_output],
            outputs=[mix_status, mix_command_state],
            show_progress=False,
        )
        blend_run.click(
            _run_model_op,
            inputs=[mix_command_state],
            outputs=[mix_status],
            show_progress="minimal",
        )
        convert_inspect.click(
            _inspect_source,
            inputs=[convert_source, convert_arch],
            outputs=[convert_status],
            show_progress=False,
        )
        convert_check.click(
            _check_conversion,
            inputs=[convert_source, convert_operation, convert_output, convert_arch],
            outputs=[convert_status, conversion_command_state],
            show_progress=False,
        )
        convert_run.click(
            _run_model_op,
            inputs=[conversion_command_state],
            outputs=[convert_status],
            show_progress="minimal",
        )
        quant_check.click(
            _check_quant,
            inputs=[quant_source, quant_target, quant_choice, quant_output, quant_arch],
            outputs=[quant_status, quant_command_state],
            show_progress=False,
        )
        quant_run.click(
            _run_model_op,
            inputs=[quant_command_state],
            outputs=[quant_status],
            show_progress="minimal",
        )

        # -- Popular preset buttons -------------------------------------------
        _PRESETS = {
            "ckpt":   ("", "Checkpoint", "Most Downloaded"),
            "lora":   ("", "LORA",       "Most Downloaded"),
            "vae":    ("", "VAE",         "Most Downloaded"),
            "newest": ("", "",            "Newest"),
        }

        def _preset_click(preset_key, show_nsfw):
            query, type_filter, sort = _PRESETS[preset_key]
            search_outputs = _civitai_search(query, type_filter, show_nsfw, sort)
            return search_outputs + (
                gr.update(value=query),
                gr.update(value=type_filter),
                gr.update(value=sort),
            )

        _preset_outputs = _civitai_outputs + [civitai_query, civitai_type, civitai_sort]

        civitai_preset_ckpt.click(
            lambda nsfw: _preset_click("ckpt", nsfw),
            inputs=[civitai_nsfw], outputs=_preset_outputs, show_progress="minimal",
        )
        civitai_preset_lora.click(
            lambda nsfw: _preset_click("lora", nsfw),
            inputs=[civitai_nsfw], outputs=_preset_outputs, show_progress="minimal",
        )
        civitai_preset_vae.click(
            lambda nsfw: _preset_click("vae", nsfw),
            inputs=[civitai_nsfw], outputs=_preset_outputs, show_progress="minimal",
        )
        civitai_preset_newest.click(
            lambda nsfw: _preset_click("newest", nsfw),
            inputs=[civitai_nsfw], outputs=_preset_outputs, show_progress="minimal",
        )


        def refresh_checkpoints():
            checkpoints = catalog.refresh_checkpoints()
            choices = catalog.checkpoint_choices()
            default = choices[0][1] if choices else None
            second = choices[1][1] if len(choices) > 1 else None
            details = catalog.checkpoint_details(catalog.find_checkpoint(default))
            return (
                gr.update(choices=choices, value=default),
                details,
                catalog.models_folder_help(),
                gr.update(choices=choices, value=default),
                gr.update(choices=choices, value=default),
                gr.update(choices=choices, value=second),
            )

        ckpt_refresh.click(
            refresh_checkpoints,
            outputs=[ckpt_select, ckpt_details, folder_help, mix_base, blend_left, blend_right],
            show_progress=False,
        )

        def show_checkpoint(checkpoint_id: str | None):
            return catalog.checkpoint_details(catalog.find_checkpoint(checkpoint_id))

        ckpt_select.change(
            show_checkpoint,
            inputs=[ckpt_select],
            outputs=[ckpt_details],
            show_progress=False,
        )

        def refresh_loras():
            catalog.refresh_loras()
            choices = catalog.lora_choices()
            default = choices[0][1] if choices else None
            lora = catalog.find_lora(default)
            return (
                gr.update(choices=choices, value=default),
                catalog.lora_details(lora),
                *_lora_form_values(catalog, lora),
                *(gr.update(choices=choices, value=None) for _ in mix_lora_slots[0::2]),
            )

        def _lora_form_values(catalog_svc, lora):
            if lora is None:
                return gr.update(value=""), gr.update(value=1.0), gr.update(value="")
            alias = catalog_svc.alias_for_lora(lora.id) or ""
            strength = catalog_svc.lora_strength(lora.id)
            keywords = catalog_svc.lora_keywords(lora.id)
            return gr.update(value=alias), gr.update(value=strength), gr.update(value=keywords)

        lora_refresh.click(
            refresh_loras,
            outputs=[lora_select, lora_details, lora_alias, lora_strength, lora_keywords, *mix_lora_slots[0::2]],
            show_progress=False,
        )

        def show_lora(lora_id: str | None):
            lora = catalog.find_lora(lora_id)
            return (
                catalog.lora_details(lora),
                *_lora_form_values(catalog, lora),
            )

        lora_select.change(
            show_lora,
            inputs=[lora_select],
            outputs=[lora_details, lora_alias, lora_strength, lora_keywords],
            show_progress=False,
        )

        def save_lora_settings(lora_id: str | None, alias: str, strength: float, keywords: str):
            if not lora_id:
                return "_Select a LoRA first._"
            lora = catalog.find_lora(lora_id)
            if lora is None:
                return "_LoRA not found._"
            catalog.set_lora_config(
                lora.id,
                alias=alias.strip().lower(),
                strength=float(strength),
                keywords=keywords.strip(),
            )
            ctx.save_settings()
            return f"\u2713 Saved settings for **{lora.title}**."

        save_lora.click(
            save_lora_settings,
            inputs=[lora_select, lora_alias, lora_strength, lora_keywords],
            outputs=[lora_status],
            show_progress=False,
        )

        def copy_lora_shortcut(lora_id: str | None, alias: str):
            if not lora_id or not alias.strip():
                return "_Set an alias first, then copy._"
            shortcut = f"*lora:{alias.strip().lower()}"
            return f"\u2713 Shortcut copied: `{shortcut}`"

        copy_shortcut.click(
            copy_lora_shortcut,
            inputs=[lora_select, lora_alias],
            outputs=[lora_status],
            show_progress=False,
        )
