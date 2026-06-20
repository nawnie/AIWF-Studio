from __future__ import annotations

import time

import gradio as gr
from PIL import Image as PILImage

from aiwf.bootstrap import AppContext
from aiwf.dev.diagnostics import (
    trace_exception_safe,
    trace_job_record_state,
    trace_studio_generate,
    trace_studio_request_built,
)
from aiwf.core.domain.enhance import RestoreOptions
from aiwf.core.domain.faceswap import FaceSwapOptions
from aiwf.core.domain.generation import GenerationMode, GenerationRequest, JobState
from aiwf.core.domain.models import SCHEDULE_TYPES
from aiwf.core.domain.segment import SegmentRequest
from aiwf.core.domain.segment_presets import (
    CUSTOM_SEGMENT_PRESET_ID,
    resolve_segment_text_prompt,
    segment_mask_preset_choices,
)
from aiwf.core.infotext import infotext_to_request_updates, parse_infotext
from aiwf.core.tags import format_tags_display, parse_tags
from aiwf.infrastructure.diffusers.mask import (
    editor_from_mask,
    inpaint_session_background,
    prepare_outpaint,
    resolve_inpaint_mask,
)
from aiwf.infrastructure.faceswap import FaceSwapUnavailable
from aiwf.web.components.checkpoints import checkpoint_dropdown, format_model_status, refresh_checkpoints
from aiwf.web.components.results import format_generation_outputs, results_gallery
from aiwf.web.studio.catalogs import StudioCatalogs
from aiwf.web.studio.controlnet_stack import StudioControlNetSlot, build_controlnet_stack
from aiwf.web.studio.constants import EMPTY_CANVAS, MODE_TITLES, MODES, TOOLBAR_HINTS
from aiwf.web.studio.handlers import compare as compare_handlers
from aiwf.web.studio.handlers import inpaint as inpaint_handlers
from aiwf.web.studio.handlers import models as model_handlers
from aiwf.web.studio.handlers import prompts as prompt_handlers
from aiwf.web.studio.handlers import reactor as reactor_handlers
from aiwf.web.tabs.segment import build_segment_panel
from aiwf.web.studio.handlers import styles as style_handlers
from aiwf.services.prompt_tools import PromptToolsService
from aiwf.web.studio.resolution import (
    ASPECT_RATIO_PRESETS,
    BUCKET_CHOICES,
    DEFAULT_UPLOAD_BUCKET,
    GENERATION_SIZE_PRESETS,
    NON_SQUARE_ASPECT_RATIO_PRESETS,
    dimensions_from_generation_preset,
    resize_to_bucket,
)
from aiwf.web.studio.helpers import (
    align_compare_pair,
    format_tag_summary,
    generation_style_fields,
    load_uploaded_image,
    mode_from_label,
    paste_control_values,
    segment_source_image,
)
from aiwf.web.studio.mode_ui import apply_mode_ui, on_mode_change
from aiwf.web.studio.session import StudioSession
from aiwf.web.studio.summaries import (
    model_help_markdown as _model_help_md,
    result_summary_markdown as _result_summary_md,
)

def _legacy_result_summary_md(job, new_seed, job_status):
    """Rich post-generation summary: prompt, steps, CFG, sampler, size, timing,
    speed, and LoRAs — shown in the Studio status area after each render."""
    import re as _re

    req = job.request
    res = job.result
    if new_seed >= 0:
        head = f"**Done** \u2014 seed **{new_seed}**"
    elif job_status.startswith("**"):
        head = job_status
    else:
        head = f"**Done** \u2014 {job_status}"
    lines = [head]

    prompt = (req.prompt or "").strip().replace("\n", " ")
    if prompt:
        lines.append("_" + (prompt if len(prompt) <= 90 else prompt[:87] + "\u2026") + "_")

    bits = [f"{req.steps} steps", f"CFG {req.cfg_scale:g}", str(req.sampler)]
    sched = getattr(req, "scheduler", "automatic")
    if sched and sched != "automatic":
        bits.append(str(sched))
    bits.append(f"{req.width}\u00d7{req.height}")
    if getattr(req, "batch_size", 1) * getattr(req, "batch_count", 1) > 1:
        bits.append(f"{req.batch_size}\u00d7{req.batch_count} batch")
    lines.append(" \u00b7 ".join(bits))

    elapsed = float(getattr(res, "elapsed_seconds", 0.0) or 0.0)
    if elapsed > 0:
        total_steps = max(1, int(req.steps)) * max(1, len(res.images))
        speed = total_steps / elapsed
        unit = f"{speed:.2f} it/s" if speed >= 1 else f"{1.0 / speed:.2f} s/it"
        lines.append(f"\u23f1 {elapsed:.1f}s \u00b7 {unit}")

    loras = list(dict.fromkeys(_re.findall(r"<lora:([^:>]+)", req.prompt or "")))
    if loras:
        lines.append("LoRA: " + ", ".join(loras))

    return "  \n".join(lines)


def _legacy_model_help_md(ckpt_title):
    """One-line guidance for the selected model (distilled vs standard)."""
    from aiwf.core.model_profile import detect_model_profile

    prof = detect_model_profile(ckpt_title)
    if prof.is_distilled:
        return (
            f"**{prof.title}** \u2014 {prof.help_text}  \n"
            f"Suggested: CFG **{prof.recommended_cfg:g}**, **{prof.recommended_steps}** steps, "
            f"sampler **{prof.recommended_sampler}**, schedule **{prof.recommended_scheduler}**."
        )
    return f"**{prof.title}** \u2014 {prof.help_text}"


def build_studio_tab(ctx: AppContext, tab: gr.Tab | None = None) -> None:
    service = ctx.generation
    catalogs = StudioCatalogs.from_context(ctx)
    session = StudioSession()
    samplers = service.list_samplers()
    vaes = service.list_vaes()
    vae_choices = [("Automatic", None)] + [(v.title, v.id) for v in vaes]
    default_sampler_label = catalogs.default_sampler_label
    default_schedule_label = catalogs.default_schedule_label
    sampler_map = catalogs.sampler_map
    sampler_id_to_label = catalogs.sampler_id_to_label
    schedule_map = catalogs.schedule_map
    default_resolution_size = min(
        GENERATION_SIZE_PRESETS,
        key=lambda size: abs(size - max(ctx.settings.default_width, ctx.settings.default_height)),
    )
    default_aspect = ctx.settings.default_width / max(1, ctx.settings.default_height)
    default_resolution_ratio = min(
        ASPECT_RATIO_PRESETS,
        key=lambda item: abs(default_aspect - (int(item[1].split(":", 1)[0]) / int(item[1].split(":", 1)[1]))),
    )[1]

    studio_root = gr.Column(elem_classes=["aiwf-studio", "aiwf-mode-txt2img"])
    with studio_root:
        with gr.Row(elem_classes=["aiwf-studio-header"]):
            mode_title = gr.Markdown(MODE_TITLES["txt2img"], elem_classes=["aiwf-mode-title"])
            mode_toggle = gr.Radio(
                label="Mode",
                choices=[label for _, label in MODES],
                value="Text",
                container=False,
                elem_classes=["aiwf-mode-toggle"],
            )

        with gr.Row(equal_height=False, elem_classes=["aiwf-studio-body"]):
            with gr.Column(scale=4, min_width=340, elem_classes=["aiwf-sidebar"]):
                with gr.Column(elem_classes=["aiwf-panel"]):
                    gr.Markdown("Model", elem_classes=["aiwf-section-label"])
                    checkpoint, checkpoint_map = checkpoint_dropdown(ctx)
                    model_status = gr.Markdown(format_model_status(ctx), elem_classes=["aiwf-model-status"])
                    model_help = gr.Markdown(
                        _model_help_md(checkpoint.value),
                        elem_classes=["aiwf-model-help", "aiwf-settings-paths"],
                    )
                    refresh = gr.Button("Refresh models", elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"])

                with gr.Column(elem_classes=["aiwf-panel", "aiwf-generate-panel"]):
                    with gr.Row(elem_classes=["aiwf-action-bar", "aiwf-mobile-dock"]):
                        generate = gr.Button(
                            "Generate",
                            variant="primary",
                            scale=3,
                            elem_classes=["aiwf-generate-btn"],
                            elem_id="aiwf-generate",
                        )
                        interrupt = gr.Button("Stop", elem_classes=["aiwf-btn-stop"])
                        continuous_toggle = gr.Checkbox(
                            label="Continuous",
                            value=False,
                            scale=1,
                            elem_classes=["aiwf-continuous-toggle"],
                        )
                    gr.Markdown(
                        '<kbd>Shift</kbd> + <kbd>Enter</kbd> in the prompt runs Generate.',
                        elem_classes=["aiwf-hotkey-hint"],
                    )

                with gr.Column(elem_classes=["aiwf-panel", "aiwf-prompt-panel"]):
                    gr.Markdown("Prompt", elem_classes=["aiwf-section-label"])
                    prompt = gr.Textbox(
                        label="Prompt",
                        show_label=False,
                        lines=4,
                        placeholder="{red|blue} coat, __style__, (detail:1.2), *lora:alias",
                        elem_classes=["aiwf-prompt-input"],
                        elem_id="aiwf-prompt",
                    )
                    negative = gr.Textbox(
                        label="Negative prompt",
                        show_label=False,
                        lines=2,
                        placeholder="Optional — elements to exclude",
                        elem_classes=["aiwf-negative-input"],
                    )
                    with gr.Accordion("Style presets", open=False, elem_classes=["aiwf-prompt-tools"]):
                        gr.Markdown(
                            "Presets wrap your prompt with `{prompt}` — some lead in "
                            "(e.g. *a high quality photo of …*), others add detail tags after. "
                            "Select a preset to read or edit its templates, then **Save preset**.",
                            elem_classes=["aiwf-settings-paths"],
                        )
                        with gr.Row():
                            style_select = gr.Dropdown(
                                label="Style preset",
                                choices=ctx.prompts.style_choices(),
                                value=None,
                                allow_custom_value=False,
                                scale=3,
                            )
                            apply_style_btn = gr.Button(
                                "Apply to prompt",
                                elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"],
                                scale=1,
                            )
                        style_name_input = gr.Textbox(
                            label="Preset name",
                            placeholder="e.g. Quality — Standard",
                            info="Used when saving a new preset or renaming on save",
                        )
                        style_template_prompt = gr.Textbox(
                            label="Style template (positive)",
                            lines=3,
                            placeholder="a high quality photo of {prompt}, masterpiece, best quality, highly detailed",
                        )
                        style_template_negative = gr.Textbox(
                            label="Style template (negative)",
                            lines=2,
                            placeholder="{prompt}, worst quality, low quality, blurry",
                        )
                        style_preview = gr.Markdown("", visible=False, elem_classes=["aiwf-settings-paths"])
                        with gr.Row():
                            save_style_btn = gr.Button("Save preset", elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"])
                            reset_style_btn = gr.Button(
                                "Reset to default",
                                elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"],
                                interactive=False,
                            )
                            delete_style_btn = gr.Button("Delete preset", elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"])
                        gr.Markdown("Embeddings", elem_classes=["aiwf-section-label"])
                        with gr.Row():
                            embedding_pick = gr.Dropdown(
                                label="Embedding",
                                choices=[(e.title, e.id) for e in ctx.generation.list_embeddings()],
                                value=None,
                                scale=3,
                            )
                            embedding_refresh = gr.Button(
                                "Refresh",
                                elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"],
                                scale=1,
                            )
                        with gr.Row():
                            add_embedding_btn = gr.Button(
                                "Add to prompt", elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"]
                            )
                            add_embedding_neg_btn = gr.Button(
                                "Add to negative", elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"]
                            )
                    with gr.Accordion("LoRAs", open=False, elem_classes=["aiwf-prompt-tools"]):
                        gr.Markdown(
                            "Build a compact LoRA stack, then apply it to the prompt. Saved trigger "
                            "words and default strengths come from the **Models** tab.",
                            elem_classes=["aiwf-settings-paths"],
                        )
                        lora_choices = ctx.models.lora_choices()
                        with gr.Column(elem_classes=["aiwf-lora-stack"]):
                            lora_stack_components = []
                            for index in range(4):
                                with gr.Row(elem_classes=["aiwf-lora-stack-row"]):
                                    stack_pick = gr.Dropdown(
                                        label=f"LoRA {index + 1}",
                                        choices=lora_choices,
                                        value=None,
                                        scale=4,
                                    )
                                    stack_strength = gr.Slider(
                                        0.0,
                                        2.0,
                                        value=1.0,
                                        step=0.05,
                                        label="Strength",
                                        scale=2,
                                    )
                                    lora_stack_components.extend([stack_pick, stack_strength])
                        lora_stack_keywords = gr.Checkbox(
                            label="Include saved trigger words",
                            value=True,
                            elem_classes=["aiwf-compact-check"],
                        )
                        with gr.Row():
                            lora_pick = gr.Dropdown(
                                label="Quick add one LoRA",
                                choices=lora_choices,
                                value=None,
                                scale=4,
                            )
                            lora_pick_refresh = gr.Button(
                                "Refresh",
                                elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"],
                                scale=1,
                            )
                        lora_pick_strength = gr.Slider(
                            0.0, 2.0, value=0.8, step=0.05, label="Strength"
                        )
                        with gr.Row():
                            apply_lora_stack_btn = gr.Button("Apply stack", variant="primary")
                            add_lora_btn = gr.Button("Add one", elem_classes=["aiwf-btn-ghost"])

                    with gr.Accordion("Prompt file & wildcards", open=False, elem_classes=["aiwf-prompt-tools"]):
                        gr.Markdown(
                            "Load prompts from `prompts/` (random line per run). "
                            "Use `wildcards/name.txt` via `__name__`.",
                            elem_classes=["aiwf-settings-paths"],
                        )
                        gr.Markdown(ctx.prompts.folder_help(), elem_classes=["aiwf-settings-paths"])
                        use_prompt_file = gr.Checkbox(label="Load from prompt file", value=False)
                        with gr.Row():
                            prompt_file = gr.Dropdown(
                                label="Prompt file",
                                choices=ctx.prompts.list_prompt_files(),
                                value=None,
                                allow_custom_value=False,
                                scale=4,
                            )
                            refresh_prompt_files = gr.Button("Refresh", elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"], scale=1)
                        prompt_file_preview = gr.Markdown(
                            "_Select a prompt file to preview._",
                            elem_classes=["aiwf-prompt-preview"],
                        )

                    tags_input = gr.Textbox(
                        label="Tags",
                        show_label=False,
                        lines=1,
                        placeholder="#portrait  #client-work  #wip",
                        info="Embedded in PNG metadata — searchable in Library",
                        elem_classes=["aiwf-tags-input"],
                    )
                    quick_tag = gr.Dropdown(
                        label="Quick add tag",
                        choices=ctx.tags.recent_tag_choices(),
                        value=None,
                        allow_custom_value=False,
                        elem_classes=["aiwf-quick-tag"],
                    )

                with gr.Column(elem_classes=["aiwf-panel"]):
                    gr.Markdown("Parameters", elem_classes=["aiwf-section-label"])
                    with gr.Row():
                        sampler = gr.Dropdown(
                            label="Sampler",
                            choices=[s.label for s in samplers],
                            value=default_sampler_label,
                        )
                        steps = gr.Slider(1, 150, value=ctx.settings.default_steps, step=1, label="Steps")
                    with gr.Row():
                        scheduler = gr.Dropdown(
                            label="Schedule type",
                            choices=[s.label for s in SCHEDULE_TYPES],
                            value=default_schedule_label,
                        )
                        cfg = gr.Slider(1, 30, value=ctx.settings.default_cfg_scale, step=0.5, label="CFG scale")
                        seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                    with gr.Row():
                        reuse_seed = gr.Button(
                            "Reuse last seed",
                            elem_classes=["aiwf-btn-sm", "aiwf-preset-action"],
                        )
                        randomize_seed = gr.Button(
                            "Random seed",
                            elem_classes=["aiwf-btn-sm", "aiwf-preset-action"],
                        )

                    txt2img_panel = gr.Column(elem_classes=["aiwf-mode-panel"])
                    with txt2img_panel:
                        with gr.Column(elem_classes=["aiwf-resolution-presets"]):
                            with gr.Row(elem_classes=["aiwf-resolution-row"]):
                                gr.HTML('<div class="aiwf-resolution-heading">Size</div>')
                                resolution_size = gr.Radio(
                                    show_label=False,
                                    container=False,
                                    choices=[(str(size), size) for size in GENERATION_SIZE_PRESETS],
                                    value=default_resolution_size,
                                    elem_classes=["aiwf-resolution-toggle", "aiwf-resolution-size"],
                                )
                            with gr.Row(elem_classes=["aiwf-resolution-row"]):
                                gr.HTML('<div class="aiwf-resolution-heading">Ratio</div>')
                                with gr.Column(elem_classes=["aiwf-resolution-ratio-stack"]):
                                    resolution_ratio = gr.Radio(
                                        show_label=False,
                                        container=False,
                                        choices=list(NON_SQUARE_ASPECT_RATIO_PRESETS),
                                        value=(default_resolution_ratio if default_resolution_ratio != "1:1" else None),
                                        elem_classes=["aiwf-resolution-toggle", "aiwf-resolution-ratio"],
                                    )
                                    resolution_ratio_square = gr.Radio(
                                        show_label=False,
                                        container=False,
                                        choices=[("1:1", "1:1")],
                                        value=("1:1" if default_resolution_ratio == "1:1" else None),
                                        elem_classes=[
                                            "aiwf-resolution-toggle",
                                            "aiwf-resolution-ratio",
                                            "aiwf-resolution-ratio-square",
                                        ],
                                    )
                        with gr.Row():
                            width = gr.Slider(64, 2048, value=ctx.settings.default_width, step=8, label="Width")
                            height = gr.Slider(64, 2048, value=ctx.settings.default_height, step=8, label="Height")
                        with gr.Row():
                            batch_size = gr.Slider(1, 8, value=1, step=1, label="Batch")
                            batch_count = gr.Slider(1, 8, value=1, step=1, label="Count")

                inpaint_panel = gr.Accordion(
                    "Inpaint & mask",
                    open=True,
                    visible=False,
                    elem_classes=["aiwf-inpaint-panel"],
                )
                with inpaint_panel:
                    gr.Markdown(
                        "Crop-and-stitch (**Only masked**) runs diffusion on a tight crop then pastes back. "
                        "Use **mask blur** + **seam erode** to soften edges.",
                        elem_classes=["aiwf-settings-paths"],
                    )
                    inpaint_denoise = gr.Slider(
                        0,
                        1,
                        value=0.75,
                        step=0.01,
                        label="Denoising strength",
                        info="Strength applied inside the painted mask",
                    )
                    with gr.Row():
                        mask_blur = gr.Slider(
                            0,
                            64,
                            value=8,
                            step=1,
                            label="Mask blur",
                            info="Gaussian blur on the composite mask — higher = softer seams",
                        )
                        seam_erode = gr.Slider(
                            0,
                            16,
                            value=1,
                            step=1,
                            label="Seam erode",
                            info="Shrink paste mask inward to eat diffused edge halos",
                        )
                    inpaint_area = gr.Radio(
                        ["Whole picture", "Only masked"],
                        value="Only masked",
                        label="Inpaint area",
                        info="Only masked = crop-and-stitch (recommended for segments)",
                    )
                    inpaint_padding = gr.Slider(
                        0,
                        128,
                        value=32,
                        step=4,
                        label="Crop padding",
                        info="Context pixels around mask bbox (Only masked)",
                    )
                    masked_content = gr.Dropdown(
                        choices=["fill", "original", "latent noise", "latent nothing"],
                        value="latent noise",
                        label="Masked content",
                        info="How the masked region is initialized before diffusion",
                    )
                    inpaint_area.change(
                        lambda area: gr.update(visible=(area == "Only masked")),
                        inputs=[inpaint_area],
                        outputs=[inpaint_padding],
                        show_progress=False,
                    )
                    inpaint_source = gr.Radio(
                        label="Inpaint source",
                        choices=[
                            ("Original image", "original"),
                            ("Last result", "result"),
                        ],
                        value="original",
                    )
                    with gr.Accordion("Segment (SAM)", open=False, elem_classes=["aiwf-prompt-tools"]):
                        gr.Markdown(ctx.segment.folder_help(), elem_classes=["aiwf-settings-paths"])
                        sam_mask_preset = gr.Dropdown(
                            label="What to mask",
                            choices=segment_mask_preset_choices(),
                            value="person",
                        )
                        sam_custom_prompt = gr.Textbox(
                            label="Custom prompt",
                            placeholder="person.hat",
                            visible=False,
                        )
                        with gr.Row():
                            sam_model = gr.Dropdown(
                                label="SAM model",
                                choices=[(m.title, m.id) for m in ctx.segment.list_models()],
                                value=(ctx.segment.list_models()[0].id if ctx.segment.list_models() else None),
                                scale=4,
                            )
                            sam_refresh = gr.Button(
                                "Refresh SAM",
                                elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"],
                                scale=1,
                            )
                        with gr.Row():
                            sam_threshold = gr.Slider(0.05, 0.95, value=0.3, step=0.05, label="Detection threshold")
                            sam_mask_index = gr.Slider(0, 2, value=0, step=1, label="Mask candidate")
                        with gr.Row():
                            sam_dilation = gr.Slider(0, 64, value=4, step=1, label="Mask expand")
                            sam_mask_blur = gr.Slider(
                                0,
                                64,
                                value=8,
                                step=1,
                                label="Mask blur",
                                info="Softens SAM mask edges before inpaint",
                            )
                        with gr.Row():
                            sam_preview_btn = gr.Button("Preview candidates", elem_classes=["aiwf-btn-ghost"])
                            sam_mask_btn = gr.Button("Generate & send mask", variant="primary", elem_classes=["aiwf-btn-ghost"])
                        sam_candidates = gr.Gallery(
                            label="SAM candidates",
                            columns=3,
                            height=150,
                            object_fit="contain",
                            visible=False,
                        )
                    with gr.Accordion("Outpaint (extend canvas)", open=False, elem_classes=["aiwf-prompt-tools"]):
                        with gr.Row():
                            op_left = gr.Slider(0, 512, value=0, step=8, label="Left px")
                            op_right = gr.Slider(0, 512, value=0, step=8, label="Right px")
                        with gr.Row():
                            op_up = gr.Slider(0, 512, value=0, step=8, label="Up px")
                            op_down = gr.Slider(0, 512, value=0, step=8, label="Down px")
                        op_fill = gr.Radio(["edge", "reflect", "noise"], value="edge", label="Seed new area")
                        op_overlap = gr.Slider(0, 64, value=8, step=1, label="Seam overlap")
                        op_btn = gr.Button("Prepare outpaint canvas", elem_classes=["aiwf-btn-ghost"])

                with gr.Accordion("Prompt Tools", open=False, elem_classes=["aiwf-prompt-tools"]):
                    gr.Markdown(
                        "Inspect local checkpoints / LoRAs, read model metadata, build prompt drafts, "
                        "and get settings recommendations — all without loading weights.",
                        elem_classes=["aiwf-settings-hint"],
                    )
                    with gr.Tabs():
                        with gr.Tab("Inspector"):
                            with gr.Row():
                                pt_inspect_btn = gr.Button("Scan checkpoints & LoRAs", elem_classes=["aiwf-btn-ghost"], scale=1)
                            pt_inspect_out = gr.Markdown("", elem_classes=["aiwf-settings-hint"])

                        with gr.Tab("Metadata"):
                            pt_meta_path = gr.Textbox(
                                label="Safetensors path",
                                placeholder=r"C:\models\my_model.safetensors",
                                info="Reads metadata only; weights are not loaded.",
                            )
                            with gr.Row():
                                pt_meta_btn = gr.Button("Read metadata", elem_classes=["aiwf-btn-ghost"], scale=1)
                            pt_meta_out = gr.Markdown("", elem_classes=["aiwf-settings-hint"])

                        with gr.Tab("Prompt builder"):
                            with gr.Row():
                                pt_subject = gr.Textbox(label="Subject", placeholder="a cat sitting on a windowsill", scale=3)
                                pt_style = gr.Textbox(label="Style name (optional)", placeholder="watercolor", scale=1)
                            pt_loras = gr.Textbox(
                                label="LoRA names (comma-separated)",
                                placeholder="detail_tweaker, film_grain",
                                info="These become <lora:name:1.0> tags in the prompt.",
                            )
                            pt_neg = gr.Textbox(label="Negative additions (optional)", placeholder="blurry, watermark")
                            with gr.Row():
                                pt_build_btn = gr.Button("Build prompt draft", variant="secondary", scale=1)
                                pt_apply_btn = gr.Button("Apply to prompt", elem_classes=["aiwf-btn-ghost"], scale=1)
                            pt_draft_out = gr.Textbox(label="Draft", interactive=False, lines=3)

                        with gr.Tab("Recommend settings"):
                            with gr.Row():
                                pt_arch = gr.Dropdown(
                                    label="Architecture",
                                    choices=["sd15", "sdxl", "wan"],
                                    value="sd15",
                                    scale=1,
                                )
                                pt_goal = gr.Dropdown(
                                    label="Goal",
                                    choices=["speed", "balanced", "quality"],
                                    value="balanced",
                                    scale=1,
                                )
                            with gr.Row():
                                pt_rec_btn = gr.Button("Get recommendations", elem_classes=["aiwf-btn-ghost"], scale=1)
                            pt_rec_out = gr.Markdown("", elem_classes=["aiwf-settings-hint"])

                advanced_accordion = gr.Accordion("Advanced", open=False, elem_classes=["aiwf-advanced"])
                with advanced_accordion:
                    cooldown_seconds = gr.Slider(
                        0,
                        120,
                        value=ctx.settings.generation_cooldown_seconds,
                        step=1,
                        label="Wait between runs (seconds)",
                        info="Pause between continuous generations to reduce heat on mobile GPUs",
                    )
                    clip_skip = gr.Slider(1, 12, value=ctx.settings.default_clip_skip, step=1, label="Clip skip")
                    with gr.Row():
                        vae = gr.Dropdown(label="VAE", choices=vae_choices, value=None, scale=4)
                        vae_refresh = gr.Button("Refresh VAEs", elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"], scale=1)

                    img2img_advanced = gr.Column(visible=False, elem_classes=["aiwf-advanced-mode"])
                    with img2img_advanced:
                        gr.Markdown("Image2Image", elem_classes=["aiwf-advanced-mode-label"])
                        denoise = gr.Slider(
                            0,
                            1,
                            value=0.75,
                            step=0.01,
                            label="Denoising strength",
                            info="How much of the source image to replace — lower keeps more detail",
                        )

                    pnginfo_hint = gr.Markdown("", visible=False, elem_classes=["aiwf-settings-paths"])
                    paste_text = gr.Textbox(label="Paste infotext", lines=3, placeholder="Paste generation parameters…")
                    with gr.Row():
                        paste_btn = gr.Button("Apply infotext", elem_classes=["aiwf-btn-ghost"])
                        paste_bridge_btn = gr.Button("From PNG Info", elem_classes=["aiwf-btn-ghost"])
                        reuse_last_btn = gr.Button("Reuse last generation", elem_classes=["aiwf-btn-ghost"])

            with gr.Column(scale=8, elem_classes=["aiwf-canvas-column"]):
                workspace_shell = gr.Column(elem_classes=["aiwf-workspace", "aiwf-mode-txt2img"])

                with workspace_shell:
                    with gr.Row(elem_classes=["aiwf-workspace-toolbar"]):
                        with gr.Row(elem_classes=["aiwf-tool-group"]):
                            upload_resolution = gr.Dropdown(
                                label="Upload size",
                                choices=BUCKET_CHOICES,
                                value=DEFAULT_UPLOAD_BUCKET,
                                visible=False,
                                elem_classes=["aiwf-upload-resolution"],
                                scale=2,
                                info="Longest edge after upload — 768 is the 8GB VRAM cap",
                            )
                            upload_btn = gr.UploadButton(
                                "Upload",
                                file_types=["image"],
                                file_count="single",
                                visible=False,
                                elem_classes=["aiwf-tool-btn", "aiwf-upload-btn"],
                            )
                            use_result_btn = gr.Button(
                                "Use output",
                                visible=False,
                                elem_classes=["aiwf-tool-btn"],
                            )
                            edit_mask_btn = gr.Button(
                                "Paint mask",
                                visible=False,
                                elem_classes=["aiwf-tool-btn"],
                            )
                            compare_btn = gr.Button(
                                "Compare",
                                visible=False,
                                elem_classes=["aiwf-tool-btn", "aiwf-tool-compare"],
                            )
                        workspace_hint = gr.Markdown(
                            TOOLBAR_HINTS["txt2img"],
                            elem_classes=["aiwf-workspace-hint"],
                        )

                    with gr.Column(elem_classes=["aiwf-canvas-stage"]):
                        empty_canvas = gr.Markdown(
                            EMPTY_CANVAS["txt2img"],
                            elem_classes=["aiwf-empty-canvas"],
                            visible=True,
                        )
                        mask_editor = gr.ImageEditor(
                            type="pil",
                            brush=gr.Brush(colors=["#FFFFFF"], default_size=48),
                            eraser=gr.Eraser(default_size=48),
                            layers=True,
                            height=640,
                            fixed_canvas=False,
                            visible=False,
                            show_label=False,
                            elem_classes=["aiwf-mask-editor"],
                        )

                        workspace_image = gr.Image(
                            type="pil",
                            interactive=False,
                            show_label=False,
                            elem_classes=["aiwf-workspace-image"],
                        )

                        compare_slider = gr.ImageSlider(
                            type="pil",
                            interactive=False,
                            show_label=False,
                            visible=False,
                            height=640,
                            slider_position=50,
                            elem_classes=["aiwf-compare"],
                        )

                gallery = results_gallery(
                    visible=False,
                    columns=getattr(ctx.settings, "gallery_columns", 2),
                    height=getattr(ctx.settings, "gallery_height", None) or None,
                )
                with gr.Accordion("Generation parameters", open=False, elem_classes=["aiwf-meta-accordion"]):
                    tag_summary = gr.Markdown("", elem_classes=["aiwf-tag-summary"], visible=False)
                    info = gr.Textbox(
                        label="Infotext",
                        lines=2,
                        show_label=False,
                        buttons=["copy"],
                        elem_classes=["aiwf-gen-info"],
                    )
                with gr.Accordion("Segment", open=False, elem_classes=["aiwf-prompt-tools", "aiwf-segment-panel"]):
                    build_segment_panel(ctx)
                with gr.Accordion("ControlNet", open=False, elem_classes=["aiwf-prompt-tools", "aiwf-controlnet-panel"]):
                    gr.Markdown(
                        "Guide txt2img / img2img with up to three stacked control images (edges, pose, depth...). "
                        "Download models in **Models -> ControlNet**. Match SD1.5 ControlNets with SD1.5 checkpoints and SDXL ControlNets with SDXL checkpoints.",
                        elem_classes=["aiwf-settings-paths"],
                    )
                    cn_enable = gr.Checkbox(label="Enable ControlNet unit 1", value=False)
                    _cn_models = ctx.controlnet.list_models()
                    cn_model = gr.Dropdown(
                        label="Unit 1 model",
                        choices=[(m.title, m.id) for m in _cn_models],
                        value=(_cn_models[0].id if _cn_models else None),
                    )
                    cn_module = gr.Dropdown(
                        label="Unit 1 preprocessor",
                        choices=ctx.controlnet.list_modules(),
                        value="canny",
                        info="`none` = use your image as-is (precomputed control map)",
                    )
                    cn_image = gr.Image(label="Unit 1 control image", type="pil", sources=["upload", "clipboard"])
                    with gr.Row():
                        cn_weight = gr.Slider(0, 2, value=1.0, step=0.05, label="Weight")
                        cn_guidance_start = gr.Slider(0, 1, value=0.0, step=0.01, label="Guidance start")
                        cn_guidance_end = gr.Slider(0, 1, value=1.0, step=0.01, label="Guidance end")
                    with gr.Row():
                        cn_threshold_a = gr.Slider(1, 255, value=100, step=1, label="Threshold A")
                        cn_threshold_b = gr.Slider(1, 255, value=200, step=1, label="Threshold B")
                    with gr.Row():
                        cn_refresh = gr.Button("Refresh models", elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"])
                        cn_preview_btn = gr.Button("Preview unit 1", elem_classes=["aiwf-btn-ghost"])
                    cn_preview = gr.Image(label="Unit 1 control map preview", type="pil", interactive=False)
                    with gr.Accordion("ControlNet unit 2", open=False, elem_classes=["aiwf-controlnet-unit"]):
                        cn2_enable = gr.Checkbox(label="Enable ControlNet unit 2", value=False)
                        cn2_model = gr.Dropdown(
                            label="Unit 2 model",
                            choices=[(m.title, m.id) for m in _cn_models],
                            value=(_cn_models[0].id if _cn_models else None),
                        )
                        cn2_module = gr.Dropdown(
                            label="Unit 2 preprocessor",
                            choices=ctx.controlnet.list_modules(),
                            value="depth",
                            info="`none` = use your image as-is (precomputed control map)",
                        )
                        cn2_image = gr.Image(label="Unit 2 control image", type="pil", sources=["upload", "clipboard"])
                        with gr.Row():
                            cn2_weight = gr.Slider(0, 2, value=0.8, step=0.05, label="Weight")
                            cn2_guidance_start = gr.Slider(0, 1, value=0.0, step=0.01, label="Guidance start")
                            cn2_guidance_end = gr.Slider(0, 1, value=1.0, step=0.01, label="Guidance end")
                        with gr.Row():
                            cn2_threshold_a = gr.Slider(1, 255, value=100, step=1, label="Threshold A")
                            cn2_threshold_b = gr.Slider(1, 255, value=200, step=1, label="Threshold B")
                        cn2_preview_btn = gr.Button("Preview unit 2", elem_classes=["aiwf-btn-ghost"])
                        cn2_preview = gr.Image(label="Unit 2 control map preview", type="pil", interactive=False)
                    with gr.Accordion("ControlNet unit 3", open=False, elem_classes=["aiwf-controlnet-unit"]):
                        cn3_enable = gr.Checkbox(label="Enable ControlNet unit 3", value=False)
                        cn3_model = gr.Dropdown(
                            label="Unit 3 model",
                            choices=[(m.title, m.id) for m in _cn_models],
                            value=(_cn_models[0].id if _cn_models else None),
                        )
                        cn3_module = gr.Dropdown(
                            label="Unit 3 preprocessor",
                            choices=ctx.controlnet.list_modules(),
                            value="openpose",
                            info="`none` = use your image as-is (precomputed control map)",
                        )
                        cn3_image = gr.Image(label="Unit 3 control image", type="pil", sources=["upload", "clipboard"])
                        with gr.Row():
                            cn3_weight = gr.Slider(0, 2, value=0.6, step=0.05, label="Weight")
                            cn3_guidance_start = gr.Slider(0, 1, value=0.0, step=0.01, label="Guidance start")
                            cn3_guidance_end = gr.Slider(0, 1, value=1.0, step=0.01, label="Guidance end")
                        with gr.Row():
                            cn3_threshold_a = gr.Slider(1, 255, value=100, step=1, label="Threshold A")
                            cn3_threshold_b = gr.Slider(1, 255, value=200, step=1, label="Threshold B")
                        cn3_preview_btn = gr.Button("Preview unit 3", elem_classes=["aiwf-btn-ghost"])
                        cn3_preview = gr.Image(label="Unit 3 control map preview", type="pil", interactive=False)
                txt2img_advanced = gr.Column(elem_classes=["aiwf-advanced-mode", "aiwf-side-advanced-mode"])
                with txt2img_advanced:
                    with gr.Accordion("Hires fix", open=False, elem_classes=["aiwf-prompt-tools", "aiwf-hires-panel"]):
                        enable_hr = gr.Checkbox(label="Enable hires fix")
                        with gr.Row():
                            hr_scale = gr.Slider(1.0, 4.0, value=2.0, step=0.05, label="Upscale")
                            hr_steps = gr.Slider(1, 150, value=20, step=1, label="Hires steps")
                            hr_denoise = gr.Slider(0, 1, value=0.35, step=0.01, label="Hires denoise")
                        hr_upscaler = gr.Dropdown(
                            label="Hires upscaler",
                            choices=[("Lanczos", "lanczos"), ("Bicubic", "bicubic"), ("Nearest", "nearest")],
                            value=ctx.settings.default_hr_upscaler,
                        )
                with gr.Accordion("ReActor", open=False, elem_classes=["aiwf-prompt-tools", "aiwf-reactor-panel"]):
                    gr.Markdown(
                        "Swap a **source face** onto the current canvas result. "
                        "Optional light img2img pass (default **0.13** denoise) blends swap seams — "
                        "same pattern as ComfyUI ReActor → img2img workflows.",
                        elem_classes=["aiwf-settings-paths"],
                    )
                    gr.Markdown(ctx.faceswap.folder_help(), elem_classes=["aiwf-settings-paths"])
                    with gr.Row(equal_height=False):
                        reactor_source = gr.Image(
                            label="Source face",
                            type="pil",
                            sources=["upload", "clipboard"],
                            scale=1,
                        )
                        with gr.Column(scale=1):
                            reactor_source_index = gr.Number(
                                value=0, precision=0, label="Source face #", info="0 = first detected face"
                            )
                            reactor_target_index = gr.Number(
                                value=-1, precision=0, label="Target face #", info="-1 = swap every face"
                            )
                    with gr.Row():
                        reactor_restore = gr.Checkbox(label="Restore face after swap", value=True)
                        _reactor_restorers = [(m.title, m.id) for m in ctx.enhance.list_restorers()]
                        reactor_restorer = gr.Dropdown(
                            label="Restorer",
                            choices=_reactor_restorers,
                            value=(_reactor_restorers[0][1] if _reactor_restorers else None),
                        )
                    with gr.Row():
                        reactor_visibility = gr.Slider(0, 1, value=1.0, step=0.05, label="Restore visibility")
                        reactor_cf_weight = gr.Slider(0, 1, value=0.5, step=0.05, label="CodeFormer weight")
                    _reactor_models = [(m.title, m.id) for m in ctx.faceswap.list_models()]
                    reactor_model = gr.Dropdown(
                        label="Swap model",
                        choices=_reactor_models,
                        value=(_reactor_models[0][1] if _reactor_models else None),
                        info="inswapper .onnx in models/insightface. Download one in the Face Swap tab.",
                    )
                    with gr.Row():
                        reactor_gender_source = gr.Dropdown(
                            label="Source gender",
                            choices=[("Any", 0), ("Female", 1), ("Male", 2)],
                            value=0,
                        )
                        reactor_gender_target = gr.Dropdown(
                            label="Target gender",
                            choices=[("Any", 0), ("Female", 1), ("Male", 2)],
                            value=0,
                        )
                    reactor_mask = gr.Checkbox(
                        label="Face mask correction",
                        value=False,
                        info="Feathers the swapped face to reduce pixelation around the contour.",
                    )
                    reactor_at_gen = gr.Checkbox(
                        label="Auto-swap on every generated image",
                        value=False,
                        info="When on, ReActor runs automatically on each image as it is generated.",
                    )
                    reactor_blend = gr.Checkbox(
                        label="Blend seams (img2img)",
                        value=True,
                        info="Runs a light img2img pass with the current prompt to smooth swap edges",
                    )
                    reactor_blend_denoise = gr.Slider(
                        0,
                        1.0,
                        value=0.13,
                        step=0.01,
                        label="Blend denoise",
                        info="0.13 is a typical seam-blend strength; raise for stronger restyling",
                    )
                    reactor_btn = gr.Button(
                        "Swap face on result",
                        variant="primary",
                        elem_classes=["aiwf-generate-btn", "aiwf-btn-sm"],
                    )
                status = gr.Markdown("**Ready** — configure parameters and generate", elem_classes=["aiwf-status-bar"])
                send_to_wan = gr.Button(
                    "Send result → Video",
                    elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"],
                    elem_id="aiwf-send-to-video",
                )

                def _send_to_wan(image):
                    if image is None:
                        raise gr.Error("Generate an image first, then send it to Video.")
                    ctx.infotext_bridge.set_image(image)
                    return "**Sent to Video** — opening the tab…"

                send_to_wan.click(
                    _send_to_wan,
                    inputs=[workspace_image],
                    outputs=[status],
                    js="""(image) => {
                        const switchVideo = () => {
                            const buttons = [...document.querySelectorAll('.aiwf-nav-tabs [role="tab"], .aiwf-nav-tabs .tab-nav button, [role="tab"]')];
                            const tab = buttons.find((button) => ((button.textContent || '').trim().toLowerCase()).startsWith('video'));
                            if (tab) {
                                tab.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, view: window }));
                            }
                        };
                        setTimeout(switchVideo, 60);
                        setTimeout(switchVideo, 350);
                        setTimeout(switchVideo, 800);
                        return [image];
                    }""",
                )
                gr.HTML(
                    '<div id="aiwf-progress-wrap" class="aiwf-progress-wrap" hidden>'
                    '<div class="aiwf-progress-meta">'
                    '<div class="aiwf-progress-track"><div id="aiwf-progress-fill" class="aiwf-progress-fill"></div></div>'
                    '<span id="aiwf-progress-step" class="aiwf-progress-step"></span>'
                    '<span id="aiwf-progress-elapsed" class="aiwf-progress-elapsed"></span>'
                    "</div></div>",
                    elem_classes=["aiwf-progress-host"],
                )
                gr.HTML(
                    '<div id="aiwf-client-error-tray" class="aiwf-client-error-tray" hidden>'
                    '<strong>Browser error</strong> '
                    '<span class="aiwf-client-error-text"></span> '
                    '<button type="button" class="aiwf-client-error-copy">Copy</button>'
                    "</div>",
                    elem_classes=["aiwf-client-error-wrap"],
                )

    # Gradio state mirrors backend choices that can change while the tab stays
    # mounted, so callbacks do not rely on stale dropdown labels.
    state = gr.State(checkpoint_map)
    last_seed = gr.State(-1)
    gallery_seeds = gr.State([])
    last_result = gr.State(None)
    last_before = gr.State(None)
    show_compare = gr.State(False)
    show_editor = gr.State(False)
    # ImageEditor gather_state breaks when the component is hidden (second Generate).
    # Generation reads this state instead of the live editor widget.
    mask_editor_value = gr.State(None)

    mode_outputs = [
        mode_title,
        txt2img_panel,
        txt2img_advanced,
        img2img_advanced,
        inpaint_panel,
        advanced_accordion,
        upload_btn,
        upload_resolution,
        edit_mask_btn,
        use_result_btn,
        workspace_image,
        mask_editor,
        workspace_hint,
        empty_canvas,
        workspace_shell,
        studio_root,
        checkpoint,
        state,
    ]

    mode_toggle.change(
        lambda *a: on_mode_change(ctx, *a),
        inputs=[mode_toggle, show_editor, checkpoint],
        outputs=[*mode_outputs, show_editor, show_compare, compare_slider, compare_btn],
        show_progress=False,
    )

    mask_editor.change(
        lambda value: value,
        inputs=[mask_editor],
        outputs=[mask_editor_value],
        show_progress=False,
    )

    def do_refresh(mode_label, current_ckpt):
        mode = mode_from_label(mode_label)
        update, new_map = refresh_checkpoints(
            ctx, rescan=True, current_value=current_ckpt
        )
        return update, format_model_status(ctx), new_map

    refresh.click(
        do_refresh,
        inputs=[mode_toggle, checkpoint],
        outputs=[checkpoint, model_status, state],
        show_progress=False,
    )

    def _on_checkpoint_change(ckpt_title, ckpt_map):
        """Remember the selected checkpoint without loading model weights."""
        help_md = _model_help_md(ckpt_title)
        if not ckpt_title or not ckpt_map:
            return gr.update(), help_md
        ckpt_id = ckpt_map.get(ckpt_title)
        if ckpt_id is None:
            return gr.update(value=f"**Error:** unknown checkpoint {ckpt_title}"), help_md
        try:
            ctx.generation.remember_checkpoint_selection(ckpt_id)
            base_status = format_model_status(ctx)
            return gr.update(value=f"**Selected:** {ckpt_title}\n\n{base_status}"), help_md
        except Exception as exc:
            return gr.update(value=f"**Selection failed:** {ckpt_title}: {exc}"), help_md

    checkpoint.change(
        _on_checkpoint_change,
        inputs=[checkpoint, state],
        outputs=[model_status, model_help],
        show_progress=False,
    )

    def stop_generation():
        session.loop_active = False
        service.interrupt()
        return "**Stopping** — interrupt requested", gr.update(value=False)

    interrupt.click(stop_generation, outputs=[status, continuous_toggle])

    def _reuse_last_seed(last_value):
        if int(last_value or -1) < 0:
            raise gr.Error("Generate an image first to reuse its seed.")
        return int(last_value)

    reuse_seed.click(_reuse_last_seed, inputs=[last_seed], outputs=[seed], show_progress=False)
    randomize_seed.click(lambda: -1, outputs=[seed], show_progress=False)

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

    def _refresh_vaes(current=None):
        vaes = ctx.generation.refresh_vae_catalog()
        choices = [("Automatic", None)] + [(item.title, item.id) for item in vaes]
        ids = {item.id for item in vaes}
        value = current if current in ids else None
        return gr.update(choices=choices, value=value)

    vae_refresh.click(_refresh_vaes, inputs=[vae], outputs=[vae], show_progress=False)

    def _refresh_sam_models(current=None):
        models = ctx.segment.refresh_models()
        ids = {model.id for model in models}
        value = current if current in ids else (models[0].id if models else None)
        return gr.update(choices=[(model.title, model.id) for model in models], value=value)

    sam_refresh.click(_refresh_sam_models, inputs=[sam_model], outputs=[sam_model], show_progress=False)

    def append_quick_tag(current: str, selected: str | None):
        if not selected:
            return current or "", gr.update(value=None)
        tags = parse_tags(current or "")
        if selected not in tags:
            tags.append(selected)
        return format_tags_display(tags), gr.update(value=None)

    quick_tag.change(append_quick_tag, inputs=[tags_input, quick_tag], outputs=[tags_input, quick_tag])

    def apply_paste(text, mode_label):
        if not text.strip():
            raise gr.Error("Paste infotext first.")
        mode = mode_from_label(mode_label)
        gen_mode = {
            "txt2img": GenerationMode.TXT2IMG,
            "img2img": GenerationMode.IMG2IMG,
            "inpaint": GenerationMode.INPAINT,
        }[mode]
        updates = infotext_to_request_updates(parse_infotext(text), gen_mode)
        controls = paste_control_values(
            updates,
            sampler_id_to_label=sampler_id_to_label,
            default_sampler_label=default_sampler_label,
        )
        return (
            controls["prompt"],
            controls["negative_prompt"],
            gr.update(value=controls["sampler"]),
            gr.update(value=controls["scheduler"]),
            controls["steps"],
            controls["cfg_scale"],
            controls["width"],
            controls["height"],
            controls["seed"],
            controls["clip_skip"],
            controls["enable_hr"],
            controls["hr_scale"],
            controls["hr_steps"],
            controls["hr_denoising_strength"],
            controls["img2img_denoise"],
            controls["inpaint_denoise"],
            controls["mask_blur"],
            controls["tags"],
        )

    paste_outputs = [
        prompt, negative, sampler, scheduler, steps, cfg,
        width, height, seed, clip_skip,
        enable_hr, hr_scale, hr_steps, hr_denoise,
        denoise, inpaint_denoise, mask_blur, tags_input,
    ]
    paste_btn.click(apply_paste, inputs=[paste_text, mode_toggle], outputs=paste_outputs)

    def _on_lora_pick(lora_id):
        if not lora_id:
            return gr.update()
        return gr.update(value=ctx.models.lora_strength(lora_id))

    lora_pick.change(_on_lora_pick, inputs=[lora_pick], outputs=[lora_pick_strength], show_progress=False)
    for stack_pick, stack_strength in zip(lora_stack_components[0::2], lora_stack_components[1::2]):
        stack_pick.change(_on_lora_pick, inputs=[stack_pick], outputs=[stack_strength], show_progress=False)

    def _refresh_lora_picker(current, *stack_current):
        ctx.models.refresh_loras()
        choices = ctx.models.lora_choices()
        ids = {value for _, value in choices}
        updates = [gr.update(choices=choices, value=current if current in ids else None)]
        for value in stack_current:
            updates.append(gr.update(choices=choices, value=value if value in ids else None))
        return tuple(updates)

    lora_pick_refresh.click(
        _refresh_lora_picker,
        inputs=[lora_pick, *lora_stack_components[0::2]],
        outputs=[lora_pick, *lora_stack_components[0::2]],
        show_progress=False,
    )

    def add_lora_to_prompt(current_prompt, lora_id, strength):
        lora = ctx.models.find_lora(lora_id)
        if lora is None:
            raise gr.Error("Pick a LoRA first (hit Refresh if the list is empty).")
        if f"<lora:{lora.id}:" in (current_prompt or ""):
            raise gr.Error("That LoRA is already in the prompt — edit its strength there.")
        tag = f"<lora:{lora.id}:{float(strength):g}>"
        keywords = (ctx.models.lora_keywords(lora.id) or "").strip()
        addition = f"{tag}, {keywords}" if keywords else tag
        text = (current_prompt or "").rstrip().rstrip(",")
        return f"{text}, {addition}" if text else addition

    add_lora_btn.click(
        add_lora_to_prompt,
        inputs=[prompt, lora_pick, lora_pick_strength],
        outputs=[prompt],
        show_progress=False,
    )
    apply_lora_stack_btn.click(
        lambda current, *values: prompt_handlers.apply_lora_stack_to_prompt(ctx, current, *values),
        inputs=[prompt, *lora_stack_components, lora_stack_keywords],
        outputs=[prompt],
        show_progress=False,
    )

    def _refresh_embedding_picker(current):
        items = ctx.generation.refresh_embedding_catalog()
        choices = [(e.title, e.id) for e in items]
        ids = {value for _, value in choices}
        return gr.update(choices=choices, value=current if current in ids else None)

    embedding_refresh.click(
        _refresh_embedding_picker,
        inputs=[embedding_pick],
        outputs=[embedding_pick],
        show_progress=False,
    )

    def _append_token(text, token):
        base = (text or "").rstrip().rstrip(",")
        return f"{base}, {token}" if base else token

    def add_embedding_to(field_value, embedding_id):
        if not embedding_id:
            raise gr.Error("Pick an embedding first (hit Refresh if the list is empty).")
        if embedding_id in (field_value or ""):
            raise gr.Error("That embedding is already in the prompt.")
        return _append_token(field_value, embedding_id)

    add_embedding_btn.click(
        add_embedding_to,
        inputs=[prompt, embedding_pick],
        outputs=[prompt],
        show_progress=False,
    )
    add_embedding_neg_btn.click(
        add_embedding_to,
        inputs=[negative, embedding_pick],
        outputs=[negative],
        show_progress=False,
    )

    def _pnginfo_pending_hint():
        if ctx.infotext_bridge.pending_text:
            return gr.update(
                value="**PNG Info waiting** — click **From PNG Info** to apply parameters.",
                visible=True,
            )
        return gr.update(value="", visible=False)

    def load_bridge(mode_label):
        if ctx.settings.pnginfo_clear_after_apply:
            text = ctx.infotext_bridge.consume_pending()
        else:
            text = ctx.infotext_bridge.pending_text
        if not text:
            raise gr.Error("No parameters waiting. Use PNG Info → Send first.")
        applied = apply_paste(text, mode_label)
        return (text, *applied, gr.update(value="", visible=False))

    paste_bridge_btn.click(
        load_bridge,
        inputs=[mode_toggle],
        outputs=[paste_text, *paste_outputs, pnginfo_hint],
    )

    def _last_generation_infotext() -> str | None:
        for job in ctx.generation.recent_jobs(10):
            result = getattr(job, "result", None)
            if result is not None and getattr(result, "infotexts", None):
                return result.infotexts[-1]
        # Fall back to the newest saved image's embedded parameters
        # (covers reuse across app restarts).
        try:
            root = ctx.flags.resolved_output_dir()
            candidates = sorted(
                root.rglob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True
            )[:20]
            for path in candidates:
                try:
                    with PILImage.open(path) as img:
                        text = (getattr(img, "text", None) or {}).get("parameters")
                    if text:
                        return text
                except Exception:
                    continue
        except Exception:
            pass
        return None

    def reuse_last_generation(mode_label):
        text = _last_generation_infotext()
        if not text:
            raise gr.Error(
                "No previous generation found — generate once first "
                "(or enable PNG metadata embedding in Settings)."
            )
        applied = apply_paste(text, mode_label)
        return (text, *applied)

    reuse_last_btn.click(
        reuse_last_generation,
        inputs=[mode_toggle],
        outputs=[paste_text, *paste_outputs],
        show_progress=False,
    )

    def handle_upload(file_obj, bucket, mode_label, editing_mask, current_ckpt=None):
        image = load_uploaded_image(file_obj)
        if image is None:
            return (
                gr.update(),
                gr.update(),
                editing_mask,
                gr.update(),
                gr.update(),
                *apply_mode_ui(ctx, mode_label, editing_mask, current_ckpt=current_ckpt),
            )

        image, size_note = resize_to_bucket(image, int(bucket or 0))
        mode = mode_from_label(mode_label)
        if mode == "inpaint":
            editor_val = {"background": image, "layers": [], "composite": None}
            session.sam_mask = None
            session.inpaint.original = image.copy()
            session.inpaint.mask = None
            mode_ui = apply_mode_ui(ctx, mode_label, True, current_ckpt=current_ckpt)
            return (
                gr.update(value=editor_val),
                gr.update(value=image, visible=True),
                True,
                editor_val,
                f"**Loaded** — {size_note}",
                *mode_ui,
            )
        if mode == "img2img":
            mode_ui = apply_mode_ui(ctx, mode_label, False, current_ckpt=current_ckpt)
            return (
                gr.update(),
                gr.update(value=image, visible=True),
                False,
                gr.update(),
                f"**Loaded** — {size_note}",
                *mode_ui,
            )
        return (
            gr.update(),
            gr.update(),
            editing_mask,
            gr.update(),
            gr.update(),
            *apply_mode_ui(ctx, mode_label, editing_mask, current_ckpt=current_ckpt),
        )

    upload_btn.upload(
        handle_upload,
        inputs=[upload_btn, upload_resolution, mode_toggle, show_editor, checkpoint],
        outputs=[mask_editor, workspace_image, show_editor, mask_editor_value, status, *mode_outputs],
        show_progress=False,
    )

    def start_mask_edit(mode_label, workspace_img, source_choice, editor_value, current_ckpt=None):
        background = inpaint_session_background(source_choice, workspace_img, editor_value, session.inpaint_session)
        if background is None:
            raise gr.Error("Upload an image first.")
        editor_val = {"background": background, "layers": [], "composite": None}
        if session.inpaint.mask is not None:
            editor_val = editor_from_mask(background, session.inpaint.mask)
        session.sam_mask = session.inpaint.mask
        mode_ui = apply_mode_ui(ctx, mode_label, True, current_ckpt=current_ckpt)
        return gr.update(value=editor_val), True, editor_val, *mode_ui

    edit_mask_btn.click(
        start_mask_edit,
        inputs=[mode_toggle, workspace_image, inpaint_source, mask_editor_value, checkpoint],
        outputs=[mask_editor, show_editor, mask_editor_value, *mode_outputs],
    )

    def _on_sam_preset_change(preset_id):
        return gr.update(visible=preset_id == CUSTOM_SEGMENT_PRESET_ID)

    sam_mask_preset.change(
        _on_sam_preset_change,
        inputs=[sam_mask_preset],
        outputs=[sam_custom_prompt],
        show_progress=False,
    )

    def _run_sam(
        preset_id,
        custom_prompt,
        model_id,
        threshold,
        mask_index,
        dilation,
        mask_blur_amount,
        source_image,
        editor_value,
        mode_label,
        current_ckpt=None,
    ):
        segment_source = segment_source_image(source_image, editor_value)
        if segment_source is None:
            # Robust fallback for the embedded Segment accordion inside Inpaint mode.
            # In inpaint flows the current working image may live in session.inpaint.original
            # (set by upload, start mask edit, use result, outpaint, etc.) rather than the
            # top-level workspace_image or the current mask_editor dict.
            if session.inpaint.original is not None:
                segment_source = session.inpaint.original
            else:
                raise gr.Error("Upload or generate an image first.")
        prompt = resolve_segment_text_prompt(preset_id, custom_prompt)
        if not prompt:
            raise gr.Error("Choose what to mask, or select Custom and enter a prompt.")
        request = SegmentRequest(
            text_prompt=prompt,
            box_threshold=float(threshold),
            mask_index=int(mask_index),
            dilation=int(dilation or 0),
            mask_blur=int(mask_blur_amount or 0),
        )
        mask, preview, candidates, message = ctx.segment.segment(
            segment_source, request, model_id=model_id or None
        )
        # Keep the SAM mask as authoritative data — the Gradio ImageEditor does
        # not reliably round-trip programmatically-set layers, so generation
        # reads this directly instead of re-parsing painted layers.
        session.sam_mask = mask
        session.inpaint.mask = mask.copy()
        editor_val = editor_from_mask(segment_source, mask)
        gallery = gr.update(value=[preview, *candidates], visible=True)
        mode_ui = apply_mode_ui(ctx, mode_label, True, current_ckpt=current_ckpt)
        # Auto-configure the inpaint controls for good results with auto-generated
        # segment masks (GroundingDINO + SAM). "Only masked" + "latent noise" (or fill)
        # makes the masked region actually get replaced instead of looking like no-op.
        return (
            gr.update(value=editor_val),
            True,
            gallery,
            f"**SAM:** {message}",
            gr.update(value="Only masked"),
            gr.update(value=32),
            gr.update(value="latent noise"),
            gr.update(value=8),
            gr.update(value=1),
            editor_val,
            *mode_ui,
        )

    _sam_inputs = [
        sam_mask_preset,
        sam_custom_prompt,
        sam_model,
        sam_threshold,
        sam_mask_index,
        sam_dilation,
        sam_mask_blur,
        workspace_image,
        mask_editor_value,
        mode_toggle,
        checkpoint,
    ]
    # When sending a segment/grounding mask into inpaint, we also configure the
    # new inpaint options to sensible values so the mask actually produces visible
    # changes (instead of the old "original content + whole picture" behavior that
    # often looked like "nothing happened").
    _sam_outputs = [
        mask_editor,
        show_editor,
        sam_candidates,
        status,
        inpaint_area,
        inpaint_padding,
        masked_content,
        mask_blur,
        seam_erode,
        mask_editor_value,
        *mode_outputs,
    ]
    sam_mask_btn.click(_run_sam, inputs=_sam_inputs, outputs=_sam_outputs, show_progress="minimal")
    sam_preview_btn.click(_run_sam, inputs=_sam_inputs, outputs=_sam_outputs, show_progress="minimal")

    def _cn_models_update(current=None):
        models = ctx.controlnet.list_models()
        ids = {m.id for m in models}
        value = current if current in ids else (models[0].id if models else None)
        return gr.update(choices=[(m.title, m.id) for m in models], value=value)

    def _cn_models_update_all(current1=None, current2=None, current3=None):
        return (
            _cn_models_update(current1),
            _cn_models_update(current2),
            _cn_models_update(current3),
        )

    cn_refresh.click(
        _cn_models_update_all,
        inputs=[cn_model, cn2_model, cn3_model],
        outputs=[cn_model, cn2_model, cn3_model],
        show_progress=False,
    )

    def _cn_preview(image, module, threshold_a, threshold_b):
        if image is None:
            raise gr.Error("Upload a control image first.")
        return ctx.controlnet.preprocess(
            image,
            module or "none",
            processor_res=512,
            threshold_a=float(threshold_a),
            threshold_b=float(threshold_b),
        )

    cn_preview_btn.click(
        _cn_preview,
        inputs=[cn_image, cn_module, cn_threshold_a, cn_threshold_b],
        outputs=[cn_preview],
        show_progress="minimal",
    )
    cn2_preview_btn.click(
        _cn_preview,
        inputs=[cn2_image, cn2_module, cn2_threshold_a, cn2_threshold_b],
        outputs=[cn2_preview],
        show_progress="minimal",
    )
    cn3_preview_btn.click(
        _cn_preview,
        inputs=[cn3_image, cn3_module, cn3_threshold_a, cn3_threshold_b],
        outputs=[cn3_preview],
        show_progress="minimal",
    )

    def _prepare_outpaint(source_image, editor_value, mode_label, left, right, up, down, fill, overlap):
        src = segment_source_image(source_image, editor_value)
        if src is None:
            raise gr.Error("Upload or generate an image first.")
        try:
            padded, mask = prepare_outpaint(
                src,
                left=int(left),
                right=int(right),
                up=int(up),
                down=int(down),
                fill=fill,
                mask_overlap=int(overlap),
            )
        except ValueError as exc:
            raise gr.Error(str(exc))
        # Authoritative mask consumed by Generate (same path as SAM masks).
        session.sam_mask = mask
        session.inpaint.original = padded.copy()
        session.inpaint.mask = mask.copy()
        editor_val = {"background": padded, "layers": [], "composite": None}
        mode_ui = apply_mode_ui(ctx, mode_label, True)
        new_px = padded.size
        return (
            gr.update(value=editor_val),
            True,
            f"**Outpaint ready** — canvas {new_px[0]}×{new_px[1]}. Set a prompt and Generate.",
            editor_val,
            *mode_ui,
        )

    op_btn.click(
        _prepare_outpaint,
        inputs=[workspace_image, mask_editor_value, mode_toggle, op_left, op_right, op_up, op_down, op_fill, op_overlap],
        outputs=[mask_editor, show_editor, status, mask_editor_value, *mode_outputs],
        show_progress="minimal",
    )

    # ── Prompt Tools accordion handlers ─────────────────────────────────────────
    _pt_service = PromptToolsService(
        checkpoint_dir=ctx.flags.resolved_ckpt_dir(),
        lora_dir=ctx.flags.resolved_models_dir() / "Loras",
        output_dir=ctx.flags.resolved_output_dir(),
    )

    def _pt_inspect():
        try:
            ckpts = _pt_service.list_local_checkpoints()
            loras = _pt_service.list_local_loras()
            lines = ["**Checkpoints**"]
            if ckpts:
                for c in ckpts[:20]:
                    size_mb = c.get("size_bytes", 0) / (1024 * 1024)
                    lines.append(f"- `{c['name']}` — {size_mb:.0f} MB")
                if len(ckpts) > 20:
                    lines.append(f"… and {len(ckpts)-20} more")
            else:
                lines.append("_No checkpoints found._")
            lines.append("")
            lines.append("**LoRAs**")
            if loras:
                for l in loras[:20]:
                    size_mb = l.get("size_bytes", 0) / (1024 * 1024)
                    lines.append(f"- `{l['name']}` — {size_mb:.0f} MB")
                if len(loras) > 20:
                    lines.append(f"… and {len(loras)-20} more")
            else:
                lines.append("_No LoRAs found._")
            return "\n".join(lines)
        except Exception as exc:
            return f"**Error:** {exc}"

    pt_inspect_btn.click(
        _pt_inspect,
        inputs=[],
        outputs=[pt_inspect_out],
        show_progress="minimal",
    )

    def _pt_read_meta(path_str):
        if not path_str or not path_str.strip():
            return "_Enter a path above._"
        try:
            meta = _pt_service.read_safetensors_metadata(path_str.strip())
            if not meta:
                return "_No metadata found in header._"
            import json as _json
            lines = []
            for k, v in meta.items():
                if k.startswith("__"):
                    continue  # skip tensor shape entries
                val_str = _json.dumps(v) if not isinstance(v, str) else v
                lines.append(f"**{k}**: {val_str[:200]}")
            return "\n\n".join(lines) or "_Metadata present but empty._"
        except Exception as exc:
            return f"**Error:** {exc}"

    pt_meta_btn.click(
        _pt_read_meta,
        inputs=[pt_meta_path],
        outputs=[pt_meta_out],
        show_progress="minimal",
    )

    def _pt_build_draft(subject, style, loras_str, neg):
        try:
            lora_names = [l.strip() for l in (loras_str or "").split(",") if l.strip()]
            result = _pt_service.build_prompt_draft(
                subject=subject or "",
                style_name=style or "",
                lora_names=lora_names,
                negative=neg or "",
            )
            return result.get("positive", "")
        except Exception as exc:
            return f"Error: {exc}"

    pt_build_btn.click(
        _pt_build_draft,
        inputs=[pt_subject, pt_style, pt_loras, pt_neg],
        outputs=[pt_draft_out],
        show_progress="minimal",
    )

    def _pt_apply_draft(draft_text, current_prompt):
        if not draft_text:
            return gr.update()
        # Append to existing prompt if non-empty, else replace
        if current_prompt and current_prompt.strip():
            return gr.update(value=current_prompt.rstrip(", ") + ", " + draft_text)
        return gr.update(value=draft_text)

    pt_apply_btn.click(
        _pt_apply_draft,
        inputs=[pt_draft_out, prompt],
        outputs=[prompt],
        show_progress=False,
    )

    def _pt_recommend(arch, goal):
        try:
            rec = _pt_service.recommend_settings(architecture=arch, goal=goal)
            if not rec:
                return "_No recommendation for this combination._"
            lines = []
            for k, v in rec.items():
                lines.append(f"**{k}**: {v}")
            return "\n\n".join(lines)
        except Exception as exc:
            return f"**Error:** {exc}"

    pt_rec_btn.click(
        _pt_recommend,
        inputs=[pt_arch, pt_goal],
        outputs=[pt_rec_out],
        show_progress="minimal",
    )
    # ── /Prompt Tools ─────────────────────────────────────────────────────────

    def use_result_as_source(result_image, mode_label):
        if result_image is None:
            raise gr.Error("Generate an image first.")
        mode = mode_from_label(mode_label)
        if mode == "inpaint":
            editor_val = {"background": result_image, "layers": [], "composite": None}
            if session.inpaint.mask is not None:
                editor_val = editor_from_mask(result_image, session.inpaint.mask)
            session.sam_mask = session.inpaint.mask
            mode_ui = apply_mode_ui(ctx, mode_label, True)
            return gr.update(value=editor_val), gr.update(value=None), True, editor_val, *mode_ui
        mode_ui = apply_mode_ui(ctx, mode_label, False)
        return gr.update(), gr.update(value=result_image), False, gr.update(), *mode_ui

    use_result_btn.click(
        use_result_as_source,
        inputs=[workspace_image, mode_toggle],
        outputs=[mask_editor, workspace_image, show_editor, mask_editor_value, *mode_outputs],
    )

    def _refresh_prompt_files():
        files = ctx.prompts.list_prompt_files()
        return gr.update(choices=files, value=files[0][1] if files else None)

    refresh_prompt_files.click(_refresh_prompt_files, outputs=[prompt_file], show_progress=False)

    def _preview_prompt_file(path: str | None):
        return ctx.prompts.preview_prompt_file(path)

    prompt_file.change(_preview_prompt_file, inputs=[prompt_file], outputs=[prompt_file_preview], show_progress=False)

    def _load_style_editor(style_name):
        from aiwf.core.domain.prompt_style import PromptStyle
        from aiwf.core.domain.style_presets import is_builtin_style, style_preview_text

        if not style_name:
            return "", "", "", gr.update(value="", visible=False), gr.update(interactive=False)

        style = ctx.prompts.find_style(style_name)
        if style is None:
            return "", "", "", gr.update(value="", visible=False), gr.update(interactive=False)

        return (
            style.name,
            style.prompt,
            style.negative_prompt,
            gr.update(value=style_preview_text(style), visible=True),
            gr.update(interactive=is_builtin_style(style_name)),
        )

    style_select.change(
        _load_style_editor,
        inputs=[style_select],
        outputs=[
            style_name_input,
            style_template_prompt,
            style_template_negative,
            style_preview,
            reset_style_btn,
        ],
        show_progress=False,
    )

    def _refresh_style_preview(template_prompt, template_negative):
        from aiwf.core.domain.prompt_style import PromptStyle
        from aiwf.core.domain.style_presets import style_preview_text

        if not (template_prompt or "").strip() and not (template_negative or "").strip():
            return gr.update(value="", visible=False)
        preview_style = PromptStyle(name="", prompt=template_prompt or "", negative_prompt=template_negative or "")
        return gr.update(value=style_preview_text(preview_style), visible=True)

    for editor_input in (style_template_prompt, style_template_negative):
        editor_input.change(
            _refresh_style_preview,
            inputs=[style_template_prompt, style_template_negative],
            outputs=[style_preview],
            show_progress=False,
        )

    def _apply_style_to_prompt(template_prompt, template_negative, prompt_text, negative_text):
        from aiwf.core.domain.prompt_style import PromptStyle, apply_prompt_style

        if not (template_prompt or "").strip() and not (template_negative or "").strip():
            raise gr.Error("Select or edit a style preset first.")
        style = PromptStyle(name="", prompt=template_prompt or "", negative_prompt=template_negative or "")
        positive, negative = apply_prompt_style(style, prompt_text, negative_text)
        return positive, negative

    apply_style_btn.click(
        _apply_style_to_prompt,
        inputs=[style_template_prompt, style_template_negative, prompt, negative],
        outputs=[prompt, negative],
        show_progress=False,
    )

    def _save_prompt_style(name, template_prompt, template_negative, selected_name):
        from aiwf.core.domain.prompt_style import PromptStyle

        clean = (name or "").strip()
        if not clean:
            raise gr.Error("Enter a preset name.")
        if not (template_prompt or "").strip() and not (template_negative or "").strip():
            raise gr.Error("Enter at least one style template (positive or negative).")
        if selected_name and selected_name != clean:
            ctx.prompts.delete_style(selected_name, ctx_save=None)
        ctx.prompts.save_style(
            PromptStyle(name=clean, prompt=template_prompt or "", negative_prompt=template_negative or ""),
            ctx_save=ctx.save_settings,
        )
        choices = ctx.prompts.style_choices()
        return gr.update(choices=choices, value=clean), clean

    save_style_btn.click(
        _save_prompt_style,
        inputs=[style_name_input, style_template_prompt, style_template_negative, style_select],
        outputs=[style_select, style_name_input],
        show_progress=False,
    )

    def _reset_prompt_style(style_name):
        from aiwf.core.domain.style_presets import is_builtin_style, style_preview_text

        if not style_name:
            raise gr.Error("Select a built-in preset to reset.")
        if not is_builtin_style(style_name):
            raise gr.Error("Only built-in presets can be reset to default.")
        preset = ctx.prompts.reset_style_to_default(style_name, ctx_save=ctx.save_settings)
        if preset is None:
            raise gr.Error("Preset not found.")
        return preset.prompt, preset.negative_prompt, gr.update(value=style_preview_text(preset), visible=True)

    reset_style_btn.click(
        _reset_prompt_style,
        inputs=[style_select],
        outputs=[style_template_prompt, style_template_negative, style_preview],
        show_progress=False,
    )

    def _delete_prompt_style(name):
        if not name:
            raise gr.Error("Select a preset to delete.")
        ctx.prompts.delete_style(name, ctx_save=ctx.save_settings)
        return (
            gr.update(choices=ctx.prompts.style_choices(), value=None),
            "",
            "",
            "",
            gr.update(value="", visible=False),
            gr.update(interactive=False),
        )

    delete_style_btn.click(
        _delete_prompt_style,
        inputs=[style_select],
        outputs=[
            style_select,
            style_name_input,
            style_template_prompt,
            style_template_negative,
            style_preview,
            reset_style_btn,
        ],
        show_progress=False,
    )

    def _generation_request(
        mode_label,
        editing_mask,
        prompt_text,
        negative_text,
        ckpt_title,
        sampler_label,
        scheduler_label,
        step_count,
        cfg_scale,
        clip_skip_value,
        w,
        h,
        bs,
        bc,
        seed_value,
        vae_id,
        hires_enabled,
        hires_scale,
        hires_steps,
        hires_denoise,
        hires_upscaler,
        img2img_denoise,
        inpaint_denoise_value,
        mask_blur_value,
        seam_erode_value,
        inpaint_area_value,
        inpaint_padding_value,
        masked_content_value,
        source_image,
        editor_value,
        ckpt_map,
        tags_text,
        use_file,
        prompt_file_path,
        dynamic_seed,
        style_name,
        style_template_prompt,
        style_template_negative,
        cn_enable,
        cn_model_id,
        cn_module,
        cn_image,
        cn_weight,
        cn_guidance_start,
        cn_guidance_end,
        cn_threshold_a,
        cn_threshold_b,
        cn2_enable,
        cn2_model_id,
        cn2_module,
        cn2_image,
        cn2_weight,
        cn2_guidance_start,
        cn2_guidance_end,
        cn2_threshold_a,
        cn2_threshold_b,
        cn3_enable,
        cn3_model_id,
        cn3_module,
        cn3_image,
        cn3_weight,
        cn3_guidance_start,
        cn3_guidance_end,
        cn3_threshold_a,
        cn3_threshold_b,
        inpaint_source_choice,
    ):
        if not ckpt_map or not ckpt_title:
            raise gr.Error("No checkpoint available. Refresh models.")

        if use_file and not prompt_file_path and not (prompt_text or "").strip():
            raise gr.Error("Select a prompt file or enter a prompt.")

        mode = mode_from_label(mode_label)
        ckpt_id = ckpt_map.get(ckpt_title)
        tags = parse_tags(tags_text or "")
        style_fields = generation_style_fields(style_name, style_template_prompt, style_template_negative)
        before_image = None
        init_images = None
        mask_images = None

        if mode == "txt2img":
            request = GenerationRequest(
                mode=GenerationMode.TXT2IMG,
                prompt=prompt_text,
                negative_prompt=negative_text,
                prompt_file=prompt_file_path,
                use_prompt_file=bool(use_file),
                prompt_seed=dynamic_seed,
                **style_fields,
                tags=tags,
                steps=int(step_count),
                cfg_scale=float(cfg_scale),
                width=int(w),
                height=int(h),
                seed=int(seed_value),
                sampler=sampler_map.get(sampler_label, "euler_a"),
                scheduler=schedule_map.get(scheduler_label, "automatic"),
                batch_size=int(bs),
                batch_count=int(bc),
                clip_skip=int(clip_skip_value),
                enable_hr=bool(hires_enabled),
                hr_scale=float(hires_scale),
                hr_steps=int(hires_steps),
                hr_denoising_strength=float(hires_denoise),
                hr_upscaler=str(hires_upscaler or ctx.settings.default_hr_upscaler),
                checkpoint_id=ckpt_id,
                vae_id=vae_id,
            )
        elif mode == "img2img":
            if source_image is None:
                raise gr.Error("Upload an image first.")
            before_image = source_image.copy()
            init_images = [source_image]
            request = GenerationRequest(
                mode=GenerationMode.IMG2IMG,
                prompt=prompt_text,
                negative_prompt=negative_text,
                prompt_file=prompt_file_path,
                use_prompt_file=bool(use_file),
                prompt_seed=dynamic_seed,
                **style_fields,
                tags=tags,
                steps=int(step_count),
                cfg_scale=float(cfg_scale),
                seed=int(seed_value),
                sampler=sampler_map.get(sampler_label, "euler_a"),
                scheduler=schedule_map.get(scheduler_label, "automatic"),
                denoising_strength=float(img2img_denoise),
                clip_skip=int(clip_skip_value),
                checkpoint_id=ckpt_id,
            )
        else:
            background = inpaint_session_background(
                inpaint_source_choice,
                source_image,
                editor_value,
                session.inpaint_session,
            )
            if background is None:
                raise gr.Error("Upload an image and paint a mask.")

            mask = resolve_inpaint_mask(
                editor_value,
                session.inpaint_session,
                session.sam_mask,
                background.size,
                editing_mask=bool(editing_mask),
            )
            if mask is None or mask.getbbox() is None:
                raise gr.Error(
                    "No mask found. Paint over the area, use Segment, or click **Paint mask** to restore the last mask."
                )

            session.inpaint.mask = mask.copy()
            if session.inpaint.original is None:
                session.inpaint.original = background.copy()

            before_image = background.copy()
            init_images = [background]
            mask_images = [mask]
            request = GenerationRequest(
                mode=GenerationMode.INPAINT,
                prompt=prompt_text,
                negative_prompt=negative_text,
                prompt_file=prompt_file_path,
                use_prompt_file=bool(use_file),
                prompt_seed=dynamic_seed,
                **style_fields,
                tags=tags,
                steps=int(step_count),
                cfg_scale=float(cfg_scale),
                seed=int(seed_value),
                sampler=sampler_map.get(sampler_label, "euler_a"),
                scheduler=schedule_map.get(scheduler_label, "automatic"),
                denoising_strength=float(inpaint_denoise_value),
                mask_blur=int(mask_blur_value),
                seam_erode=int(seam_erode_value or 0),
                inpaint_only_masked=(inpaint_area_value == "Only masked"),
                inpaint_masked_padding=int(inpaint_padding_value),
                inpaint_mask_content=str(masked_content_value or "original"),
                clip_skip=int(clip_skip_value),
                checkpoint_id=ckpt_id,
            )

        control_images = None
        try:
            checkpoint_architecture = None
            if ckpt_id:
                checkpoint_architecture = ctx.generation.resolve_checkpoint(ckpt_id).architecture
            units, control_images_list = build_controlnet_stack(
                slots=[
                    StudioControlNetSlot(
                        "ControlNet unit 1",
                        bool(cn_enable),
                        cn_model_id,
                        cn_module,
                        cn_image,
                        float(cn_weight),
                        float(cn_guidance_start),
                        float(cn_guidance_end),
                        float(cn_threshold_a),
                        float(cn_threshold_b),
                    ),
                    StudioControlNetSlot(
                        "ControlNet unit 2",
                        bool(cn2_enable),
                        cn2_model_id,
                        cn2_module,
                        cn2_image,
                        float(cn2_weight),
                        float(cn2_guidance_start),
                        float(cn2_guidance_end),
                        float(cn2_threshold_a),
                        float(cn2_threshold_b),
                    ),
                    StudioControlNetSlot(
                        "ControlNet unit 3",
                        bool(cn3_enable),
                        cn3_model_id,
                        cn3_module,
                        cn3_image,
                        float(cn3_weight),
                        float(cn3_guidance_start),
                        float(cn3_guidance_end),
                        float(cn3_threshold_a),
                        float(cn3_threshold_b),
                    ),
                ],
                mode=mode,
                controlnet=ctx.controlnet,
                checkpoint_architecture=checkpoint_architecture,
            )
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        if units:
            request = request.model_copy(update={"controlnet_units": units})
            control_images = control_images_list

        trace_studio_request_built(
            mode=mode,
            width=getattr(request, "width", None),
            height=getattr(request, "height", None),
            init_count=len(init_images or []),
            mask_count=len(mask_images or []),
            control_count=len(control_images or []),
            checkpoint_id=ckpt_id,
        )
        return request, init_images, mask_images, before_image, mode, control_images

    def _progress_outputs(mode_label, message, preview_image=None, hold_image=None):
        mode_ui = apply_mode_ui(ctx, mode_label, False, hide_empty=True)
        if preview_image is not None:
            workspace_update = gr.update(value=preview_image, visible=True)
        elif hold_image is not None:
            workspace_update = gr.update(value=hold_image, visible=True)
        else:
            workspace_update = gr.update()
        status_text = message if message.startswith("**") else f"**{message}**"
        return (
            workspace_update,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(),
            gr.update(visible=False, value=""),
            "",
            status_text,
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            False,
            False,
            *mode_ui,
        )

    def _finished_outputs(mode_label, job, before_image, *, continuous_on: bool):
        mode_ui = apply_mode_ui(ctx, mode_label, False, hide_empty=True)
        can_compare = before_image is not None

        if job.result is None:
            session.loop_active = False
            if job.state == JobState.CANCELLED:
                status_text = "**Stopped** — generation cancelled"
            else:
                err = job.error or job.state.value
                status_text = f"**Error** — {err}"
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False, value=[]),
                gr.update(visible=False, value=""),
                "",
                status_text,
                -1,
                [],
                None,
                before_image,
                gr.update(),
                gr.update(value=False),
                False,
                False,
                *apply_mode_ui(ctx, mode_label, False),
            )

        infotext = job.result.infotexts[0] if job.result.infotexts else ""
        primary, images, infotext, job_status = format_generation_outputs(
            job.result.images,
            infotext,
            job.state.value,
        )
        gallery_update = gr.update(value=images, visible=len(images) > 1, columns=min(2, len(images)))
        new_seed = job.result.seeds[0] if job.result.seeds else -1
        done_status = _result_summary_md(job, new_seed, job_status)
        applied_tags = job.request.tags
        if applied_tags:
            ctx.tags.remember_tags(applied_tags, save=ctx.save_settings)
        tag_line = format_tag_summary(applied_tags)
        quick_tag_update = gr.update(choices=ctx.tags.recent_tag_choices())
        toggle_update = gr.update(value=continuous_on) if continuous_on else gr.update()
        return (
            gr.update(value=primary, visible=True),
            gr.update(visible=False),
            gr.update(visible=can_compare, value="Compare"),
            gallery_update,
            gr.update(value=tag_line, visible=bool(tag_line)),
            infotext,
            done_status,
            new_seed,
            list(job.result.seeds),
            primary,
            before_image,
            quick_tag_update,
            toggle_update,
            False,
            False,
            *mode_ui,
        )

    def _run_once(mode_label, all_inputs, *, keep_continuous_toggle: bool, image_postprocess=None):
        try:
            request, init_images, mask_images, before_image, mode, control_images = _generation_request(
                *all_inputs
            )
        except Exception as exc:
            trace_exception_safe("studio.request_build", exc, mode=mode_label)
            raise
        hold_image = before_image or (init_images[0] if init_images else None)
        yield _progress_outputs(mode_label, "Queued", hold_image=hold_image)
        try:
            for event in service.submit_streaming(
                request,
                init_images=init_images,
                mask_images=mask_images,
                control_images=control_images,
                image_postprocess=image_postprocess,
            ):
                if event[0] == "progress":
                    _, _step, _total, message, preview = event
                    yield _progress_outputs(
                        mode_label,
                        message,
                        preview_image=preview,
                        hold_image=hold_image if preview is None else None,
                    )
                else:
                    _, job = event
                    trace_job_record_state(job.id, job.state, job.error)
                    yield _finished_outputs(
                        mode_label,
                        job,
                        before_image if mode != "txt2img" else None,
                        continuous_on=keep_continuous_toggle,
                    )
        except Exception as exc:
            trace_exception_safe(
                "studio.generate_stream",
                exc,
                mode=mode_label,
                checkpoint_id=request.checkpoint_id,
            )
            raise

    def run(
        mode_label,
        editing_mask,
        prompt_text,
        negative_text,
        ckpt_title,
        sampler_label,
        scheduler_label,
        step_count,
        cfg_scale,
        clip_skip_value,
        w,
        h,
        bs,
        bc,
        seed_value,
        vae_id,
        hires_enabled,
        hires_scale,
        hires_steps,
        hires_denoise,
        hires_upscaler,
        img2img_denoise,
        inpaint_denoise_value,
        mask_blur_value,
        seam_erode_value,
        inpaint_area_value,
        inpaint_padding_value,
        masked_content_value,
        source_image,
        editor_value,
        ckpt_map,
        tags_text,
        use_file,
        prompt_file_path,
        style_name,
        style_template_prompt,
        style_template_negative,
        cn_enable,
        cn_model,
        cn_module,
        cn_image,
        cn_weight,
        cn_guidance_start,
        cn_guidance_end,
        cn_threshold_a,
        cn_threshold_b,
        cn2_enable,
        cn2_model,
        cn2_module,
        cn2_image,
        cn2_weight,
        cn2_guidance_start,
        cn2_guidance_end,
        cn2_threshold_a,
        cn2_threshold_b,
        cn3_enable,
        cn3_model,
        cn3_module,
        cn3_image,
        cn3_weight,
        cn3_guidance_start,
        cn3_guidance_end,
        cn3_threshold_a,
        cn3_threshold_b,
        inpaint_source,
        continuous_enabled,
        cooldown_wait,
        *reactor_args,
    ):
        session.loop_active = True
        ctx.settings.generation_cooldown_seconds = float(cooldown_wait or 0)
        ctx.save_settings()

        post_swap = None
        if reactor_args:
            (rx_on, rx_src, rx_sidx, rx_tidx, rx_restore, rx_restorer,
             rx_vis, rx_cf, rx_model, rx_gs, rx_gt, rx_mask) = reactor_args
            if rx_on and rx_src is not None:
                def post_swap(image):
                    try:
                        opts = FaceSwapOptions(
                            source_face_index=max(0, int(rx_sidx or 0)),
                            target_face_index=int(rx_tidx if rx_tidx is not None else -1),
                            model_id=rx_model or "inswapper_128",
                            gender_source=int(rx_gs or 0),
                            gender_target=int(rx_gt or 0),
                            mask_face=bool(rx_mask),
                            restore_face=bool(rx_restore),
                            restorer_id=rx_restorer,
                            restore_visibility=float(rx_vis),
                            codeformer_weight=float(rx_cf),
                        )
                        rfn = None
                        if rx_restore and rx_restorer:
                            def rfn(im):
                                return ctx.enhance.restore(
                                    im,
                                    RestoreOptions(
                                        model_id=rx_restorer,
                                        visibility=float(rx_vis),
                                        codeformer_weight=float(rx_cf),
                                    ),
                                )
                        return ctx.faceswap.swap(image, rx_src, opts, restore_fn=rfn)
                    except Exception as exc:
                        trace_exception_safe("studio.reactor_at_gen", exc)
                        return image

        try:
            run_number = 0
            while session.loop_active:
                run_number += 1
                trace_studio_generate(
                    run_number=run_number,
                    mode_label=mode_label,
                    continuous=bool(continuous_enabled),
                    editing_mask=bool(editing_mask),
                    has_source=source_image is not None,
                    has_editor_value=editor_value is not None,
                    cn_enabled=bool(cn_enable),
                    input_count=len(generate_inputs),
                )
                if continuous_enabled and run_number > 1:
                    yield _progress_outputs(mode_label, f"Run {run_number}", hold_image=source_image)

                base_seed = int(seed_value)
                if base_seed < 0:
                    dynamic_seed = None
                elif continuous_enabled:
                    dynamic_seed = base_seed + run_number - 1
                else:
                    dynamic_seed = base_seed

                request_inputs = (
                    mode_label,
                    editing_mask,
                    prompt_text,
                    negative_text,
                    ckpt_title,
                    sampler_label,
                    scheduler_label,
                    step_count,
                    cfg_scale,
                    clip_skip_value,
                    w,
                    h,
                    bs,
                    bc,
                    seed_value,
                    vae_id,
                    hires_enabled,
                    hires_scale,
                    hires_steps,
                    hires_denoise,
                    hires_upscaler,
                    img2img_denoise,
                    inpaint_denoise_value,
                    mask_blur_value,
                    seam_erode_value,
                    inpaint_area_value,
                    inpaint_padding_value,
                    masked_content_value,
                    source_image,
                    editor_value,
                    ckpt_map,
                    tags_text,
                    use_file,
                    prompt_file_path,
                    dynamic_seed,
                    style_name,
                    style_template_prompt,
                    style_template_negative,
                    cn_enable,
                    cn_model,
                    cn_module,
                    cn_image,
                    cn_weight,
                    cn_guidance_start,
                    cn_guidance_end,
                    cn_threshold_a,
                    cn_threshold_b,
                    cn2_enable,
                    cn2_model,
                    cn2_module,
                    cn2_image,
                    cn2_weight,
                    cn2_guidance_start,
                    cn2_guidance_end,
                    cn2_threshold_a,
                    cn2_threshold_b,
                    cn3_enable,
                    cn3_model,
                    cn3_module,
                    cn3_image,
                    cn3_weight,
                    cn3_guidance_start,
                    cn3_guidance_end,
                    cn3_threshold_a,
                    cn3_threshold_b,
                    inpaint_source,
                )

                for update in _run_once(
                    mode_label,
                    request_inputs,
                    keep_continuous_toggle=continuous_enabled and session.loop_active,
                    image_postprocess=post_swap,
                ):
                    yield update

                if not session.loop_active:
                    break
                if not continuous_enabled:
                    break

                wait_s = max(0, int(cooldown_wait or 0))
                for remaining in range(wait_s, 0, -1):
                    if not session.loop_active:
                        break
                    yield _progress_outputs(mode_label, f"Cooling — next run in {remaining}s")
                    time.sleep(1)
        finally:
            session.loop_active = False

    def _run_reactor(
        workspace_result,
        stored_result,
        source_image,
        source_idx,
        target_idx,
        do_restore,
        restorer_id,
        visibility,
        cf_weight,
        do_blend,
        blend_denoise,
        prompt_text,
        negative_text,
        ckpt_title,
        sampler_label,
        step_count,
        cfg_scale,
        clip_skip_value,
        seed_value,
        vae_id,
        style_name,
        ckpt_map,
        tags_text,
        use_file,
        prompt_file_path,
        model_id_in,
        gender_src_in,
        gender_tgt_in,
        mask_in,
    ):
        target = stored_result or workspace_result
        if target is None:
            raise gr.Error("Generate an image first, then run ReActor on the result.")
        if source_image is None:
            raise gr.Error("Upload a source face image.")

        options = FaceSwapOptions(
            source_face_index=max(0, int(source_idx or 0)),
            target_face_index=int(target_idx if target_idx is not None else -1),
            model_id=model_id_in or "inswapper_128",
            gender_source=int(gender_src_in or 0),
            gender_target=int(gender_tgt_in or 0),
            mask_face=bool(mask_in),
            restore_face=bool(do_restore),
            restorer_id=restorer_id,
            restore_visibility=float(visibility),
            codeformer_weight=float(cf_weight),
        )

        restore_fn = None
        if do_restore and restorer_id:
            def restore_fn(image):
                return ctx.enhance.restore(
                    image,
                    RestoreOptions(
                        model_id=restorer_id,
                        visibility=float(visibility),
                        codeformer_weight=float(cf_weight),
                    ),
                )

        try:
            swapped = ctx.faceswap.swap(target, source_image, options, restore_fn=restore_fn)
        except FaceSwapUnavailable as exc:
            raise gr.Error(str(exc))

        result_image = swapped
        status_parts = ["**ReActor complete.**"]

        if do_blend:
            if not ckpt_map or not ckpt_title:
                raise gr.Error("No checkpoint available for seam blend.")
            blend_request = GenerationRequest(
                mode=GenerationMode.IMG2IMG,
                prompt=prompt_text,
                negative_prompt=negative_text,
                prompt_file=prompt_file_path,
                use_prompt_file=bool(use_file),
                style_name=style_name or None,
                tags=parse_tags(tags_text or ""),
                steps=int(step_count),
                cfg_scale=float(cfg_scale),
                seed=int(seed_value),
                sampler=sampler_map.get(sampler_label, "euler_a"),
                denoising_strength=float(blend_denoise),
                clip_skip=int(clip_skip_value),
                checkpoint_id=ckpt_map.get(ckpt_title),
                vae_id=vae_id,
            )
            job = service.submit(blend_request, init_images=[swapped])
            if job.result is None or not job.result.images:
                raise gr.Error(job.error or "Seam blend img2img failed.")
            result_image = job.result.images[0]
            status_parts.append(f"Seam blend at **{float(blend_denoise):.2f}** denoise.")

        return (
            gr.update(value=result_image, visible=True),
            result_image,
            " ".join(status_parts),
            gr.update(visible=True, value="Compare"),
            gr.update(visible=False),
        )

    reactor_btn.click(
        _run_reactor,
        inputs=[
            workspace_image,
            last_result,
            reactor_source,
            reactor_source_index,
            reactor_target_index,
            reactor_restore,
            reactor_restorer,
            reactor_visibility,
            reactor_cf_weight,
            reactor_blend,
            reactor_blend_denoise,
            prompt,
            negative,
            checkpoint,
            sampler,
            steps,
            cfg,
            clip_skip,
            seed,
            vae,
            style_select,
            state,
            tags_input,
            use_prompt_file,
            prompt_file,
            reactor_model,
            reactor_gender_source,
            reactor_gender_target,
            reactor_mask,
        ],
        outputs=[workspace_image, last_result, status, compare_btn, empty_canvas],
        show_progress="minimal",
    )

    def toggle_compare(showing, before, after):
        if before is None or after is None:
            raise gr.Error("Generate an image first to compare.")
        aligned_before, aligned_after = align_compare_pair(before, after)
        new_show = not showing
        if new_show:
            return (
                True,
                gr.update(visible=False),
                gr.update(visible=True, value=(aligned_before, aligned_after)),
                gr.update(value="Hide compare"),
            )
        return (
            False,
            gr.update(visible=True, value=after),
            gr.update(visible=False),
            gr.update(value="Compare"),
        )

    compare_btn.click(
        toggle_compare,
        inputs=[show_compare, last_before, last_result],
        outputs=[show_compare, workspace_image, compare_slider, compare_btn],
    )

    def _on_gallery_select(evt: gr.SelectData, seeds: list, img_w: int, img_h: int):
        """Promote selected gallery image to workspace; optionally send seed/size."""
        selected_image = evt.value
        if isinstance(selected_image, dict):
            selected_image = selected_image.get("image") or selected_image.get("value")

        seed_update = gr.update()
        width_update = gr.update()
        height_update = gr.update()

        if getattr(ctx.settings, "send_seed_on_click", True) and seeds:
            idx = evt.index if isinstance(evt.index, int) else (evt.index[0] if evt.index else 0)
            if 0 <= idx < len(seeds):
                seed_update = gr.update(value=seeds[idx])

        if getattr(ctx.settings, "send_size_on_click", True) and selected_image is not None:
            try:
                from PIL import Image as _PILImage
                if isinstance(selected_image, _PILImage.Image):
                    width_update = gr.update(value=selected_image.width)
                    height_update = gr.update(value=selected_image.height)
            except Exception:
                pass

        return selected_image, seed_update, width_update, height_update

    gallery.select(
        _on_gallery_select,
        inputs=[gallery_seeds, width, height],
        outputs=[workspace_image, seed, width, height],
        show_progress=False,
    )

    generate_outputs = [
        workspace_image,
        compare_slider,
        compare_btn,
        gallery,
        tag_summary,
        info,
        status,
        last_seed,
        gallery_seeds,
        last_result,
        last_before,
        quick_tag,
        continuous_toggle,
        show_compare,
        show_editor,
        *mode_outputs,
    ]

    generate_inputs = [
        mode_toggle,
        show_editor,
        prompt,
        negative,
        checkpoint,
        sampler,
        scheduler,
        steps,
        cfg,
        clip_skip,
        width,
        height,
        batch_size,
        batch_count,
        seed,
        vae,
        enable_hr,
        hr_scale,
        hr_steps,
        hr_denoise,
        hr_upscaler,
        denoise,
        inpaint_denoise,
        mask_blur,
        seam_erode,
        inpaint_area,
        inpaint_padding,
        masked_content,
        workspace_image,
        mask_editor_value,
        state,
        tags_input,
        use_prompt_file,
        prompt_file,
        style_select,
        style_template_prompt,
        style_template_negative,
        cn_enable,
        cn_model,
        cn_module,
        cn_image,
        cn_weight,
        cn_guidance_start,
        cn_guidance_end,
        cn_threshold_a,
        cn_threshold_b,
        cn2_enable,
        cn2_model,
        cn2_module,
        cn2_image,
        cn2_weight,
        cn2_guidance_start,
        cn2_guidance_end,
        cn2_threshold_a,
        cn2_threshold_b,
        cn3_enable,
        cn3_model,
        cn3_module,
        cn3_image,
        cn3_weight,
        cn3_guidance_start,
        cn3_guidance_end,
        cn3_threshold_a,
        cn3_threshold_b,
        inpaint_source,
        continuous_toggle,
        cooldown_seconds,
        reactor_at_gen,
        reactor_source,
        reactor_source_index,
        reactor_target_index,
        reactor_restore,
        reactor_restorer,
        reactor_visibility,
        reactor_cf_weight,
        reactor_model,
        reactor_gender_source,
        reactor_gender_target,
        reactor_mask,
    ]

    generate.click(
        run,
        inputs=generate_inputs,
        outputs=generate_outputs,
        show_progress="minimal",
    )
    prompt.submit(
        run,
        inputs=generate_inputs,
        outputs=generate_outputs,
        show_progress="minimal",
    )

    def _on_studio_tab_select(mode_label, cn_current, cn2_current, cn3_current, current_ckpt):
        mode = mode_from_label(mode_label)
        # Tab activation is the cheap sync point for disk-visible model changes;
        # avoid rescanning on every control edit.
        ckpt_update, new_map = refresh_checkpoints(
            ctx, rescan=True, current_value=current_ckpt
        )
        return (
            ckpt_update,
            format_model_status(ctx),
            new_map,
            _cn_models_update(cn_current),
            _cn_models_update(cn2_current),
            _cn_models_update(cn3_current),
            _pnginfo_pending_hint(),
        )

    if tab is not None:
        tab.select(
            _on_studio_tab_select,
            inputs=[mode_toggle, cn_model, cn2_model, cn3_model, checkpoint],
            outputs=[checkpoint, model_status, state, cn_model, cn2_model, cn3_model, pnginfo_hint],
            show_progress=False,
        )
