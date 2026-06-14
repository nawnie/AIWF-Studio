from __future__ import annotations

import queue
import threading
from pathlib import Path

import gradio as gr

from aiwf.bootstrap import AppContext
from aiwf.infrastructure.safetensors_metadata import read_safetensors_metadata
from aiwf.services.model_download import CATEGORY_LABELS, browse_links_html, inspect_custom_input
from aiwf.services.model_download_catalog import QUICK_START_BUNDLES
from aiwf.web.registry import WebRegistry


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
    @registry.tab("Models", order=15)
    def build(ctx: AppContext, tab: gr.Tab | None = None) -> None:
        catalog = ctx.models

        with gr.Column(elem_classes=["aiwf-models"]):
            with gr.Column(elem_classes=["aiwf-page-header"]):
                gr.Markdown("Model Manager", elem_classes=["aiwf-section-label"])
                gr.Markdown(
                    "Browse checkpoints and LoRAs. Safetensors headers are read when available "
                    "so you can see training info and suggested trigger words.",
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
                            "Download the smallest usable model set for each feature. "
                            "Each button downloads all required files in sequence.",
                            elem_classes=["aiwf-settings-paths"],
                        )
                        with gr.Row():
                            qs_video_btn  = gr.Button("Video (Wan 2.2 I2V)",  variant="secondary")
                            qs_rife_btn   = gr.Button("Frame interpolation",   variant="secondary")
                            qs_seg_btn    = gr.Button("Segmentation (SAM)",    variant="secondary")
                            qs_fs_btn     = gr.Button("Face swap (ReActor)",   variant="secondary")
                        with gr.Row():
                            qs_sd_btn     = gr.Button("Text-to-image SD1.5",   variant="secondary")
                            qs_sdxl_btn   = gr.Button("Text-to-image SDXL",    variant="secondary")
                        qs_status = gr.Markdown("", elem_classes=["aiwf-status-bar"])

                    with gr.Column(elem_classes=["aiwf-panel"]):
                        gr.Markdown("Curated catalog", elem_classes=["aiwf-section-label"])
                        gr.Markdown(
                            "Starter list of Hugging Face, CivitAI, and direct-download models. "
                            "Pick one — the category decides which folder receives the file.",
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
                            "Paste a **Hugging Face** repo (`user/model` + filename), "
                            "**CivitAI** model page URL, or any **direct** file URL. "
                            "Choose a category so the app saves to the correct folder. "
                            "For **Wan Diffusers folder**, paste the Hugging Face repo and leave the file path empty.",
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
                            "Set a short alias and trigger words. In Studio, type `*lora:youralias` "
                            "in the prompt — when you hit **Generate**, it expands to the LoRA tag "
                            "and these words automatically.",
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

                with gr.Tab("ControlNet", elem_classes=["aiwf-model-tab"]):
                    with gr.Column(elem_classes=["aiwf-panel"]):
                        gr.Markdown("Download ControlNet models", elem_classes=["aiwf-section-label"])
                        gr.Markdown(
                            "SD1.5 ControlNet-v1.1 **Light** checkpoints (~129 MB each) download "
                            f"into `{ctx.controlnet.models_dir()}`. They use the same v1.1 "
                            "preprocessors as full models but are about 5× smaller. Already-downloaded "
                            "full fp16 checkpoints in that folder still work. Models appear in "
                            "Studio → Advanced → ControlNet automatically.",
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
                f"**{item.title} installed.** It's ready in Studio → ControlNet.",
                gr.update(choices=_cn_download_choices(ctx)),
                _cn_installed_md(ctx),
            )

        cn_dl_btn.click(
            _cn_download,
            inputs=[cn_dl_select],
            outputs=[cn_dl_status, cn_dl_select, cn_installed],
            show_progress="minimal",
        )

        def refresh_checkpoints():
            checkpoints = catalog.refresh_checkpoints()
            choices = catalog.checkpoint_choices()
            default = choices[0][1] if choices else None
            details = catalog.checkpoint_details(catalog.find_checkpoint(default))
            return (
                gr.update(choices=choices, value=default),
                details,
                catalog.models_folder_help(),
            )

        ckpt_refresh.click(
            refresh_checkpoints,
            outputs=[ckpt_select, ckpt_details, folder_help],
            show_progress=False,
        )

        def show_checkpoint(checkpoint_id: str | None):
            return catalog.checkpoint_details(catalog.find_checkpoint(checkpoint_id))

        ckpt_select.change(show_checkpoint, inputs=[ckpt_select], outputs=[ckpt_details], show_progress=False)

        def refresh_loras():
            catalog.refresh_loras()
            choices = catalog.lora_choices()
            default = choices[0][1] if choices else None
            lora = catalog.find_lora(default)
            return (
                gr.update(choices=choices, value=default),
                catalog.lora_details(lora),
                *_lora_form_values(catalog, lora),
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
            outputs=[lora_select, lora_details, lora_alias, lora_strength, lora_keywords],
            show_progress=False,
        )

        def show_lora(lora_id: str | None):
            lora = catalog.find_lora(lora_id)
            return (catalog.lora_details(lora), *_lora_form_values(catalog, lora))

        lora_select.change(
            show_lora,
            inputs=[lora_select],
            outputs=[lora_details, lora_alias, lora_strength, lora_keywords],
            show_progress=False,
        )

        def _save_lora_config(lora_id: str | None, alias: str, strength: float, keywords: str):
            if not lora_id:
                return "_Select a LoRA first._"
            catalog.set_lora_config(lora_id, alias=alias, strength=strength, keywords=keywords)
            try:
                ctx.save_settings()
            except Exception:
                pass
            token = catalog.keyword_token((alias or "").strip() or lora_id)
            return f"Saved. Shortcut: `{token}`"

        save_lora.click(
            _save_lora_config,
            inputs=[lora_select, lora_alias, lora_strength, lora_keywords],
            outputs=[lora_status],
            show_progress=False,
        )

        def _copy_shortcut(lora_id: str | None, alias: str):
            if not lora_id:
                return "_Select a LoRA first._"
            token = catalog.keyword_token((alias or "").strip() or lora_id)
            return f"Copy this into your prompt: `{token}`"

        copy_shortcut.click(
            _copy_shortcut,
            inputs=[lora_select, lora_alias],
            outputs=[lora_status],
            show_progress=False,
        )
