from __future__ import annotations

import time

import gradio as gr
from PIL import Image as PILImage

from aiwf.bootstrap import AppContext
from aiwf.core.domain.enhance import RestoreOptions
from aiwf.core.domain.faceswap import FaceSwapOptions
from aiwf.core.domain.generation import GenerationMode, GenerationRequest, JobState
from aiwf.core.domain.models import SCHEDULE_TYPES
from aiwf.core.infotext import infotext_to_request_updates, parse_infotext
from aiwf.core.tags import format_tags_display, parse_tags
from aiwf.core.domain.controlnet import ControlNetUnit
from aiwf.core.domain.segment import SegmentRequest
from aiwf.core.domain.segment_presets import (
    CUSTOM_SEGMENT_PRESET_ID,
    resolve_segment_text_prompt,
    segment_mask_preset_choices,
)
from aiwf.infrastructure.faceswap import FaceSwapUnavailable
from aiwf.infrastructure.diffusers.mask import (
    editor_from_mask,
    inpaint_session_background,
    mask_from_editor,
    prepare_outpaint,
    resolve_inpaint_mask,
)
from aiwf.web.components.checkpoints import checkpoint_dropdown, format_model_status, refresh_checkpoints
from aiwf.web.components.results import format_generation_outputs, results_gallery
from aiwf.web.registry import WebRegistry

MODES = [
    ("txt2img", "Text"),
    ("img2img", "Image2Image"),
    ("inpaint", "Inpaint"),
]

MODE_TITLES = {
    "txt2img": '<span class="aiwf-mode-kicker">Mode</span> Text to image',
    "img2img": '<span class="aiwf-mode-kicker">Mode</span> Image to image',
    "inpaint": '<span class="aiwf-mode-kicker">Mode</span> Inpaint & edit',
}

TOOLBAR_HINTS = {
    "txt2img": "Prompt → generate",
    "img2img": "Upload → vary",
    "inpaint_edit": "Paint mask → generate",
    "inpaint_result": "Original or last result — Generate again",
}

EMPTY_CANVAS = {
    "txt2img": (
        '<div class="aiwf-empty-state">'
        '<div class="aiwf-empty-state-icon" aria-hidden="true">◇</div>'
        '<p class="aiwf-empty-state-title">Canvas ready</p>'
        '<p class="aiwf-empty-state-desc">Write a prompt and press Generate. '
        "Live previews appear here while the model runs.</p></div>"
    ),
    "img2img": (
        '<div class="aiwf-empty-state">'
        '<div class="aiwf-empty-state-icon" aria-hidden="true">◇</div>'
        '<p class="aiwf-empty-state-title">Source image required</p>'
        '<p class="aiwf-empty-state-desc">Upload a reference image, tune denoising strength, '
        "then generate your variation.</p></div>"
    ),
    "inpaint_edit": (
        '<div class="aiwf-empty-state">'
        '<div class="aiwf-empty-state-icon" aria-hidden="true">◇</div>'
        '<p class="aiwf-empty-state-title">Paint your mask</p>'
        '<p class="aiwf-empty-state-desc">Brush over the region to replace. White areas are inpainted; '
        "the rest of the image stays intact.</p></div>"
    ),
    "inpaint_result": (
        '<div class="aiwf-empty-state">'
        '<div class="aiwf-empty-state-icon" aria-hidden="true">◇</div>'
        '<p class="aiwf-empty-state-title">Result canvas</p>'
        '<p class="aiwf-empty-state-desc">Your output appears here. Pick <strong>Original image</strong> or '
        "<strong>Last result</strong> under Inpaint source, then Generate again with the same mask.</p></div>"
    ),
}


def _mode_from_label(label: str) -> str:
    for mode_id, mode_label in MODES:
        if mode_label == label or mode_id == label:
            return mode_id
    return "txt2img"


def _paste_control_values(
    updates: dict,
    *,
    sampler_id_to_label: dict[str, str],
    default_sampler_label: str,
) -> dict[str, object]:
    sampler_label = sampler_id_to_label.get(updates.get("sampler", "euler_a"), default_sampler_label)
    schedule_labels = {s.id: s.label for s in SCHEDULE_TYPES}
    denoise_strength = updates.get("denoising_strength", 0.75)
    return {
        "prompt": updates.get("prompt", ""),
        "negative_prompt": updates.get("negative_prompt", ""),
        "sampler": sampler_label,
        "scheduler": schedule_labels.get(updates.get("scheduler", "automatic"), "Automatic"),
        "steps": updates.get("steps", 20),
        "cfg_scale": updates.get("cfg_scale", 7.0),
        "width": updates.get("width", 512),
        "height": updates.get("height", 512),
        "seed": updates.get("seed", -1),
        "clip_skip": updates.get("clip_skip", 1),
        "enable_hr": updates.get("enable_hr", False),
        "hr_scale": updates.get("hr_scale", 2.0),
        "hr_steps": updates.get("hr_steps", 20),
        "hr_denoising_strength": updates.get("hr_denoising_strength", 0.35),
        "img2img_denoise": denoise_strength,
        "inpaint_denoise": denoise_strength,
        "mask_blur": updates.get("mask_blur", 4),
        "tags": format_tags_display(updates.get("tags", [])),
    }


def _align_compare_pair(before: PILImage.Image | None, after: PILImage.Image | None):
    if before is None or after is None:
        return before, after
    if before.size != after.size:
        before = before.resize(after.size, PILImage.Resampling.LANCZOS)
    return before, after


def _format_tag_summary(tags: list[str]) -> str:
    if not tags:
        return ""
    return "**Tags** " + " · ".join(f"`#{tag}`" for tag in tags)


def _inpaint_background(editing_mask: bool, editor_value, source_image):
    if editing_mask and isinstance(editor_value, dict):
        return editor_value.get("background")
    if source_image is not None:
        return source_image
    return None


def _generation_style_fields(
    style_name: str | None,
    template_prompt: str | None,
    template_negative: str | None,
) -> dict[str, str | None]:
    name = (style_name or "").strip() or None
    positive = (template_prompt or "").strip() or None
    negative = (template_negative or "").strip() or None
    if not name and not positive and not negative:
        return {
            "style_name": None,
            "style_prompt_template": None,
            "style_negative_template": None,
        }
    return {
        "style_name": name,
        "style_prompt_template": positive,
        "style_negative_template": negative,
    }


def _segment_source_image(source_image, editor_value):
    if isinstance(editor_value, dict) and editor_value.get("background") is not None:
        return editor_value.get("background")
    return source_image


def register_studio(registry: WebRegistry) -> None:
    @registry.tab("Studio", order=1)
    def build(ctx: AppContext, tab: gr.Tab | None = None) -> None:
        service = ctx.generation
        samplers = service.list_samplers()
        vaes = service.list_vaes()
        vae_choices = [("Automatic", None)] + [(v.title, v.id) for v in vaes]

        sampler_map = {s.label: s.id for s in samplers}
        sampler_id_to_label = {s.id: s.label for s in samplers}
        _fallback_sampler = samplers[1].label if len(samplers) > 1 else (samplers[0].label if samplers else None)
        default_sampler_label = sampler_id_to_label.get(ctx.settings.default_sampler, _fallback_sampler)
        schedule_map = {s.label: s.id for s in SCHEDULE_TYPES}
        schedule_id_to_label = {s.id: s.label for s in SCHEDULE_TYPES}
        default_schedule_label = schedule_id_to_label.get(
            getattr(ctx.settings, "default_scheduler", "automatic"), "Automatic"
        )

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
                                "Pick a LoRA and **Add to prompt** — inserts the `<lora:name:strength>` tag "
                                "plus any saved trigger words. Set aliases, default strength, and trigger "
                                "words in the **Models** tab.",
                                elem_classes=["aiwf-settings-paths"],
                            )
                            with gr.Row():
                                lora_pick = gr.Dropdown(
                                    label="LoRA",
                                    choices=ctx.models.lora_choices(),
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
                            add_lora_btn = gr.Button("Add to prompt", elem_classes=["aiwf-btn-ghost"])

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
                            reuse_seed = gr.Button("Reuse last seed", elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"])
                            randomize_seed = gr.Button("Random seed", elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"])

                        txt2img_panel = gr.Column(elem_classes=["aiwf-mode-panel"])
                        with txt2img_panel:
                            with gr.Row():
                                width = gr.Slider(64, 2048, value=ctx.settings.default_width, step=8, label="Width")
                                height = gr.Slider(64, 2048, value=ctx.settings.default_height, step=8, label="Height")
                            with gr.Row():
                                batch_size = gr.Slider(1, 8, value=1, step=1, label="Batch")
                                batch_count = gr.Slider(1, 8, value=1, step=1, label="Count")

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

                        with gr.Accordion("ControlNet", open=False, elem_classes=["aiwf-prompt-tools"]):
                            gr.Markdown(
                                "Guide txt2img / img2img with a control image (edges, pose, depth…). "
                                "Download models in **Models → ControlNet**. SD1.5 ControlNet pairs with SD1.5 checkpoints only.",
                                elem_classes=["aiwf-settings-paths"],
                            )
                            cn_enable = gr.Checkbox(label="Enable ControlNet", value=False)
                            _cn_models = ctx.controlnet.list_models()
                            cn_model = gr.Dropdown(
                                label="ControlNet model",
                                choices=[(m.title, m.id) for m in _cn_models],
                                value=(_cn_models[0].id if _cn_models else None),
                            )
                            cn_module = gr.Dropdown(
                                label="Preprocessor",
                                choices=ctx.controlnet.list_modules(),
                                value="canny",
                                info="`none` = use your image as-is (precomputed control map)",
                            )
                            cn_image = gr.Image(label="Control image", type="pil", sources=["upload", "clipboard"])
                            with gr.Row():
                                cn_weight = gr.Slider(0, 2, value=1.0, step=0.05, label="Weight")
                                cn_guidance_start = gr.Slider(0, 1, value=0.0, step=0.01, label="Guidance start")
                                cn_guidance_end = gr.Slider(0, 1, value=1.0, step=0.01, label="Guidance end")
                            with gr.Row():
                                cn_threshold_a = gr.Slider(1, 255, value=100, step=1, label="Canny low")
                                cn_threshold_b = gr.Slider(1, 255, value=200, step=1, label="Canny high")
                            with gr.Row():
                                cn_refresh = gr.Button("Refresh models", elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"])
                                cn_preview_btn = gr.Button("Preview control map", elem_classes=["aiwf-btn-ghost"])
                            cn_preview = gr.Image(label="Control map preview", type="pil", interactive=False)

                        txt2img_advanced = gr.Column(elem_classes=["aiwf-advanced-mode"])
                        with txt2img_advanced:
                            with gr.Accordion("Hires fix", open=False):
                                enable_hr = gr.Checkbox(label="Enable hires fix")
                                with gr.Row():
                                    hr_scale = gr.Slider(1.0, 4.0, value=2.0, step=0.05, label="Upscale")
                                    hr_steps = gr.Slider(1, 150, value=20, step=1, label="Hires steps")
                                    hr_denoise = gr.Slider(0, 1, value=0.35, step=0.01, label="Hires denoise")

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

                        inpaint_advanced = gr.Column(visible=False, elem_classes=["aiwf-advanced-mode"])
                        with inpaint_advanced:
                            gr.Markdown("Inpaint & mask", elem_classes=["aiwf-advanced-mode-label"])
                            inpaint_denoise = gr.Slider(
                                0,
                                1,
                                value=0.75,
                                step=0.01,
                                label="Denoising strength",
                                info="Strength applied inside the painted mask",
                            )
                            mask_blur = gr.Slider(
                                0,
                                64,
                                value=4,
                                step=1,
                                label="Mask blur",
                                info="Softens mask edges for smoother blends with the original image",
                            )
                            inpaint_area = gr.Radio(
                                ["Whole picture", "Only masked"],
                                value="Whole picture",
                                label="Inpaint area",
                                info="Only masked crops to the mask (+padding) and runs diffusion at region resolution before pasting back.",
                            )
                            inpaint_padding = gr.Slider(
                                0,
                                128,
                                value=32,
                                step=4,
                                label="Only masked padding",
                                info="Context pixels around the mask bbox (only used for 'Only masked').",
                                visible=False,
                            )
                            masked_content = gr.Dropdown(
                                choices=["fill", "original", "latent noise", "latent nothing"],
                                value="original",
                                label="Masked content",
                                info="Initialization of the area under the mask before generation (like A1111).",
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
                                info="Original re-runs the same mask on your first upload. "
                                "Last result chains another pass on the output.",
                            )
                            with gr.Accordion("Segment settings", open=False, elem_classes=["aiwf-prompt-tools"]):
                                gr.Markdown(
                                    "Auto-mask a region for inpaint using Segment Anything. "
                                    "Pick what to mask from the list, then send the mask to the editor.",
                                    elem_classes=["aiwf-settings-paths"],
                                )
                                gr.Markdown(ctx.segment.folder_help(), elem_classes=["aiwf-settings-paths"])
                                sam_mask_preset = gr.Dropdown(
                                    label="What to mask",
                                    choices=segment_mask_preset_choices(),
                                    value="person",
                                    info="Uses Grounding DINO to find the region, then SAM to build the mask",
                                )
                                sam_custom_prompt = gr.Textbox(
                                    label="Custom prompt",
                                    placeholder="person.hat  (separate categories with .)",
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
                                    sam_threshold = gr.Slider(
                                        0.05, 0.95, value=0.3, step=0.05,
                                        label="Detection threshold",
                                        info="Lower = detect more / looser boxes",
                                    )
                                    sam_mask_index = gr.Slider(
                                        0, 2, value=0, step=1,
                                        label="Mask candidate",
                                        info="SAM returns 3 masks — pick which to use",
                                    )
                                sam_dilation = gr.Slider(0, 64, value=4, step=1, label="Mask expand (dilation)")
                                with gr.Row():
                                    sam_preview_btn = gr.Button("Preview candidates", elem_classes=["aiwf-btn-ghost"])
                                    sam_mask_btn = gr.Button("Generate & send mask", variant="primary", elem_classes=["aiwf-btn-ghost"])
                                sam_candidates = gr.Gallery(
                                    label="SAM candidates (0 · 1 · 2)",
                                    columns=3,
                                    height=150,
                                    object_fit="contain",
                                    visible=False,
                                )

                            with gr.Accordion("Outpaint (extend canvas)", open=False, elem_classes=["aiwf-prompt-tools"]):
                                gr.Markdown(
                                    "Extend the image outward, then **Generate** to fill the new area. "
                                    "Works best with a higher denoising strength.",
                                    elem_classes=["aiwf-settings-paths"],
                                )
                                with gr.Row():
                                    op_left = gr.Slider(0, 512, value=0, step=8, label="Left px")
                                    op_right = gr.Slider(0, 512, value=0, step=8, label="Right px")
                                with gr.Row():
                                    op_up = gr.Slider(0, 512, value=0, step=8, label="Up px")
                                    op_down = gr.Slider(0, 512, value=0, step=8, label="Down px")
                                op_fill = gr.Radio(
                                    ["edge", "reflect", "noise"],
                                    value="edge",
                                    label="Seed the new area with",
                                )
                                op_overlap = gr.Slider(0, 64, value=8, step=1, label="Seam overlap")
                                op_btn = gr.Button("Prepare outpaint canvas", elem_classes=["aiwf-btn-ghost"])

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

                    gallery = results_gallery(visible=False)
                    with gr.Accordion("Generation parameters", open=False, elem_classes=["aiwf-meta-accordion"]):
                        tag_summary = gr.Markdown("", elem_classes=["aiwf-tag-summary"], visible=False)
                        info = gr.Textbox(
                            label="Infotext",
                            lines=2,
                            show_label=False,
                            buttons=["copy"],
                            elem_classes=["aiwf-gen-info"],
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
                        reactor_blend = gr.Checkbox(
                            label="Blend seams (img2img)",
                            value=True,
                            info="Runs a light img2img pass with the current prompt to smooth swap edges",
                        )
                        reactor_blend_denoise = gr.Slider(
                            0,
                            0.5,
                            value=0.13,
                            step=0.01,
                            label="Blend denoise",
                            info="0.13 is a typical seam-blend strength",
                        )
                        reactor_btn = gr.Button(
                            "Swap face on result",
                            variant="primary",
                            elem_classes=["aiwf-generate-btn", "aiwf-btn-sm"],
                        )
                    status = gr.Markdown("**Ready** — configure parameters and generate", elem_classes=["aiwf-status-bar"])

        state = gr.State(checkpoint_map)
        last_seed = gr.State(-1)
        last_result = gr.State(None)
        last_before = gr.State(None)
        show_compare = gr.State(False)
        show_editor = gr.State(False)
        loop_ctrl = {"active": True}
        sam_state = {"mask": None}
        inpaint_session = {"original": None, "mask": None}

        def _load_uploaded_image(file_obj):
            if file_obj is None:
                return None
            from PIL import Image as PILImage

            if isinstance(file_obj, list):
                if not file_obj:
                    return None
                file_obj = file_obj[0]
            path = getattr(file_obj, "name", file_obj)
            if isinstance(path, dict):
                path = path.get("path") or path.get("name")
            return PILImage.open(path).convert("RGB")

        def apply_mode_ui(mode_label, editing_mask, current_ckpt=None, *, hide_empty: bool = False):
            mode = _mode_from_label(mode_label)
            is_txt = mode == "txt2img"
            is_img = mode == "img2img"
            is_inpaint = mode == "inpaint"
            inpaint_editing = is_inpaint and editing_mask

            ckpt_update, new_map = refresh_checkpoints(
                ctx, current_value=current_ckpt
            )

            if is_txt:
                hint = TOOLBAR_HINTS["txt2img"]
                empty = EMPTY_CANVAS["txt2img"]
                show_empty = True
            elif is_img:
                hint = TOOLBAR_HINTS["img2img"]
                empty = EMPTY_CANVAS["img2img"]
                show_empty = True
            elif inpaint_editing:
                hint = TOOLBAR_HINTS["inpaint_edit"]
                empty = EMPTY_CANVAS["inpaint_edit"]
                show_empty = False
            else:
                hint = TOOLBAR_HINTS["inpaint_result"]
                empty = EMPTY_CANVAS["inpaint_result"]
                show_empty = True

            if hide_empty:
                show_empty = False

            return (
                gr.update(value=MODE_TITLES[mode]),
                gr.update(visible=is_txt),
                gr.update(visible=is_txt),
                gr.update(visible=is_img),
                gr.update(visible=is_inpaint),
                gr.update(open=is_img or is_inpaint),
                gr.update(visible=is_img or is_inpaint),
                gr.update(visible=(is_img or is_inpaint) and not inpaint_editing),
                gr.update(visible=inpaint_editing),
                gr.update(visible=not inpaint_editing),
                gr.update(visible=inpaint_editing),
                gr.update(value=hint),
                gr.update(value=empty, visible=show_empty),
                gr.update(elem_classes=["aiwf-workspace", f"aiwf-mode-{mode}"]),
                gr.update(elem_classes=["aiwf-studio", f"aiwf-mode-{mode}"]),
                ckpt_update,
                new_map,
            )

        mode_outputs = [
            mode_title,
            txt2img_panel,
            txt2img_advanced,
            img2img_advanced,
            inpaint_advanced,
            advanced_accordion,
            upload_btn,
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
        def on_mode_change(mode_label, editing_mask, current_ckpt=None):
            mode = _mode_from_label(mode_label)
            if mode == "inpaint":
                editing_mask = True
            else:
                editing_mask = False
            return (
                *apply_mode_ui(mode_label, editing_mask, current_ckpt=current_ckpt),
                editing_mask,
                False,
                gr.update(visible=False),
                gr.update(visible=False, value="Compare"),
            )

        mode_toggle.change(
            on_mode_change,
            inputs=[mode_toggle, show_editor, checkpoint],
            outputs=[*mode_outputs, show_editor, show_compare, compare_slider, compare_btn],
            show_progress=False,
        )

        def do_refresh(mode_label, current_ckpt):
            mode = _mode_from_label(mode_label)
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
            """Load the selected checkpoint immediately when the user clicks a model in the dropdown.

            This gives instant feedback (logs + progress) instead of lazy-loading only on Generate.
            The actual heavy work (from_single_file, attention opts, VAE, embeddings, etc.) happens here.
            """
            if not ckpt_title or not ckpt_map:
                return gr.update()
            ckpt_id = ckpt_map.get(ckpt_title)
            if ckpt_id is None:
                return gr.update(value=f"**Error:** unknown checkpoint {ckpt_title}")
            try:
                ctx.generation.load_checkpoint(ckpt_id)
                base_status = format_model_status(ctx)
                return gr.update(value=f"**Loaded:** {ckpt_title}\n\n{base_status}")
            except Exception as exc:
                # Inner load_checkpoint already logged details; just surface to UI status.
                return gr.update(value=f"**Load failed:** {ckpt_title} — {exc}")

        checkpoint.change(
            _on_checkpoint_change,
            inputs=[checkpoint, state],
            outputs=[model_status],
            show_progress=True,
        )

        def stop_generation():
            loop_ctrl["active"] = False
            service.interrupt()
            return "**Stopping** — interrupt requested", gr.update(value=False)

        interrupt.click(stop_generation, outputs=[status, continuous_toggle])

        def _reuse_last_seed(last_value):
            if int(last_value or -1) < 0:
                raise gr.Error("Generate an image first to reuse its seed.")
            return int(last_value)

        reuse_seed.click(_reuse_last_seed, inputs=[last_seed], outputs=[seed], show_progress=False)
        randomize_seed.click(lambda: -1, outputs=[seed], show_progress=False)

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
            mode = _mode_from_label(mode_label)
            gen_mode = {
                "txt2img": GenerationMode.TXT2IMG,
                "img2img": GenerationMode.IMG2IMG,
                "inpaint": GenerationMode.INPAINT,
            }[mode]
            updates = infotext_to_request_updates(parse_infotext(text), gen_mode)
            controls = _paste_control_values(
                updates,
                sampler_id_to_label=sampler_id_to_label,
                default_sampler_label=samplers[1].label,
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

        def _refresh_lora_picker(current):
            ctx.models.refresh_loras()
            choices = ctx.models.lora_choices()
            ids = {value for _, value in choices}
            return gr.update(choices=choices, value=current if current in ids else None)

        lora_pick_refresh.click(
            _refresh_lora_picker, inputs=[lora_pick], outputs=[lora_pick], show_progress=False
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
            text = ctx.infotext_bridge.consume_pending() or ctx.infotext_bridge.pending_text
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

        def handle_upload(file_obj, mode_label, editing_mask, current_ckpt=None):
            image = _load_uploaded_image(file_obj)
            if image is None:
                return gr.update(), gr.update(), editing_mask, *apply_mode_ui(mode_label, editing_mask, current_ckpt=current_ckpt)

            mode = _mode_from_label(mode_label)
            if mode == "inpaint":
                editor_val = {"background": image, "layers": [], "composite": None}
                sam_state["mask"] = None
                inpaint_session["original"] = image.copy()
                inpaint_session["mask"] = None
                mode_ui = apply_mode_ui(mode_label, True, current_ckpt=current_ckpt)
                return (
                    gr.update(value=editor_val),
                    gr.update(value=None),
                    True,
                    *mode_ui,
                )
            if mode == "img2img":
                mode_ui = apply_mode_ui(mode_label, False, current_ckpt=current_ckpt)
                return gr.update(), gr.update(value=image), False, *mode_ui
            return gr.update(), gr.update(), editing_mask, *apply_mode_ui(mode_label, editing_mask, current_ckpt=current_ckpt)

        upload_btn.upload(
            handle_upload,
            inputs=[upload_btn, mode_toggle, show_editor, checkpoint],
            outputs=[mask_editor, workspace_image, show_editor, *mode_outputs],
            show_progress=False,
        )

        def start_mask_edit(mode_label, workspace_img, source_choice, editor_value, current_ckpt=None):
            background = inpaint_session_background(source_choice, workspace_img, editor_value, inpaint_session)
            if background is None:
                raise gr.Error("Upload an image first.")
            editor_val = {"background": background, "layers": [], "composite": None}
            if inpaint_session.get("mask") is not None:
                editor_val = editor_from_mask(background, inpaint_session["mask"])
            sam_state["mask"] = inpaint_session.get("mask")
            mode_ui = apply_mode_ui(mode_label, True, current_ckpt=current_ckpt)
            return gr.update(value=editor_val), True, *mode_ui

        edit_mask_btn.click(
            start_mask_edit,
            inputs=[mode_toggle, workspace_image, inpaint_source, mask_editor, checkpoint],
            outputs=[mask_editor, show_editor, *mode_outputs],
        )

        def _on_sam_preset_change(preset_id):
            return gr.update(visible=preset_id == CUSTOM_SEGMENT_PRESET_ID)

        sam_mask_preset.change(
            _on_sam_preset_change,
            inputs=[sam_mask_preset],
            outputs=[sam_custom_prompt],
            show_progress=False,
        )

        def _run_sam(preset_id, custom_prompt, model_id, threshold, mask_index, dilation, source_image, editor_value, mode_label, current_ckpt=None):
            segment_source = _segment_source_image(source_image, editor_value)
            if segment_source is None:
                # Robust fallback for the embedded Segment accordion inside Inpaint mode.
                # In inpaint flows the current working image may live in inpaint_session["original"]
                # (set by upload, start mask edit, use result, outpaint, etc.) rather than the
                # top-level workspace_image or the current mask_editor dict.
                if inpaint_session.get("original") is not None:
                    segment_source = inpaint_session["original"]
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
            )
            mask, preview, candidates, message = ctx.segment.segment(
                segment_source, request, model_id=model_id or None
            )
            # Keep the SAM mask as authoritative data — the Gradio ImageEditor does
            # not reliably round-trip programmatically-set layers, so generation
            # reads this directly instead of re-parsing painted layers.
            sam_state["mask"] = mask
            inpaint_session["mask"] = mask.copy()
            editor_val = editor_from_mask(segment_source, mask)
            gallery = gr.update(value=[preview, *candidates], visible=True)
            mode_ui = apply_mode_ui(mode_label, True, current_ckpt=current_ckpt)
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
                *mode_ui,
            )

        _sam_inputs = [
            sam_mask_preset,
            sam_custom_prompt,
            sam_model,
            sam_threshold,
            sam_mask_index,
            sam_dilation,
            workspace_image,
            mask_editor,
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
            *mode_outputs,
        ]
        sam_mask_btn.click(_run_sam, inputs=_sam_inputs, outputs=_sam_outputs, show_progress="minimal")
        sam_preview_btn.click(_run_sam, inputs=_sam_inputs, outputs=_sam_outputs, show_progress="minimal")

        def _cn_models_update(current=None):
            models = ctx.controlnet.list_models()
            ids = {m.id for m in models}
            value = current if current in ids else (models[0].id if models else None)
            return gr.update(choices=[(m.title, m.id) for m in models], value=value)

        cn_refresh.click(_cn_models_update, inputs=[cn_model], outputs=[cn_model], show_progress=False)

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

        def _prepare_outpaint(source_image, editor_value, mode_label, left, right, up, down, fill, overlap):
            src = _segment_source_image(source_image, editor_value)
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
            sam_state["mask"] = mask
            inpaint_session["original"] = padded.copy()
            inpaint_session["mask"] = mask.copy()
            editor_val = {"background": padded, "layers": [], "composite": None}
            mode_ui = apply_mode_ui(mode_label, True)
            new_px = padded.size
            return (
                gr.update(value=editor_val),
                True,
                f"**Outpaint ready** — canvas {new_px[0]}×{new_px[1]}. Set a prompt and Generate.",
                *mode_ui,
            )

        op_btn.click(
            _prepare_outpaint,
            inputs=[workspace_image, mask_editor, mode_toggle, op_left, op_right, op_up, op_down, op_fill, op_overlap],
            outputs=[mask_editor, show_editor, status, *mode_outputs],
            show_progress="minimal",
        )

        def use_result_as_source(result_image, mode_label):
            if result_image is None:
                raise gr.Error("Generate an image first.")
            mode = _mode_from_label(mode_label)
            if mode == "inpaint":
                editor_val = {"background": result_image, "layers": [], "composite": None}
                if inpaint_session.get("mask") is not None:
                    editor_val = editor_from_mask(result_image, inpaint_session["mask"])
                sam_state["mask"] = inpaint_session.get("mask")
                mode_ui = apply_mode_ui(mode_label, True)
                return gr.update(value=editor_val), gr.update(value=None), True, *mode_ui
            mode_ui = apply_mode_ui(mode_label, False)
            return gr.update(), gr.update(value=result_image), False, *mode_ui

        use_result_btn.click(
            use_result_as_source,
            inputs=[workspace_image, mode_toggle],
            outputs=[mask_editor, workspace_image, show_editor, *mode_outputs],
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
            img2img_denoise,
            inpaint_denoise_value,
            mask_blur_value,
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
            inpaint_source_choice,
        ):
            if not ckpt_map or not ckpt_title:
                raise gr.Error("No checkpoint available. Refresh models.")

            if use_file and not prompt_file_path and not (prompt_text or "").strip():
                raise gr.Error("Select a prompt file or enter a prompt.")

            mode = _mode_from_label(mode_label)
            ckpt_id = ckpt_map.get(ckpt_title)
            tags = parse_tags(tags_text or "")
            style_fields = _generation_style_fields(style_name, style_template_prompt, style_template_negative)
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
                    inpaint_session,
                )
                if background is None:
                    raise gr.Error("Upload an image and paint a mask.")

                mask = resolve_inpaint_mask(
                    editor_value,
                    inpaint_session,
                    sam_state.get("mask"),
                    background.size,
                    editing_mask=bool(editing_mask),
                )
                if mask is None or mask.getbbox() is None:
                    raise gr.Error(
                        "No mask found. Paint over the area, use Segment, or click **Paint mask** to restore the last mask."
                    )

                inpaint_session["mask"] = mask.copy()
                if inpaint_session.get("original") is None:
                    inpaint_session["original"] = background.copy()

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
                    inpaint_only_masked=(inpaint_area_value == "Only masked"),
                    inpaint_masked_padding=int(inpaint_padding_value),
                    inpaint_mask_content=str(masked_content_value or "original"),
                    clip_skip=int(clip_skip_value),
                    checkpoint_id=ckpt_id,
                )

            control_images = None
            if (
                cn_enable
                and cn_model_id
                and cn_image is not None
                and mode in ("txt2img", "img2img")
            ):
                unit = ControlNetUnit(
                    enabled=True,
                    model=cn_model_id,
                    module=cn_module or "none",
                    weight=float(cn_weight),
                    guidance_start=float(cn_guidance_start),
                    guidance_end=float(cn_guidance_end),
                    threshold_a=float(cn_threshold_a),
                    threshold_b=float(cn_threshold_b),
                )
                request = request.model_copy(update={"controlnet_units": [unit]})
                control_images = [cn_image]

            return request, init_images, mask_images, before_image, mode, control_images

        def _progress_outputs(mode_label, message, preview_image=None):
            mode_ui = apply_mode_ui(mode_label, False, hide_empty=True)
            if preview_image is not None:
                workspace_update = gr.update(value=preview_image, visible=True)
            else:
                workspace_update = gr.update(visible=True)
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
                False,
                False,
                *mode_ui,
            )

        def _finished_outputs(mode_label, job, before_image, *, continuous_on: bool):
            mode_ui = apply_mode_ui(mode_label, False, hide_empty=True)
            can_compare = before_image is not None

            if job.result is None:
                loop_ctrl["active"] = False
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
                    None,
                    before_image,
                    gr.update(),
                    gr.update(value=False),
                    False,
                    False,
                    *apply_mode_ui(mode_label, False),
                )

            infotext = job.result.infotexts[0] if job.result.infotexts else ""
            primary, images, infotext, job_status = format_generation_outputs(
                job.result.images,
                infotext,
                job.state.value,
            )
            gallery_update = gr.update(value=images, visible=len(images) > 1, columns=min(2, len(images)))
            new_seed = job.result.seeds[0] if job.result.seeds else -1
            if new_seed >= 0:
                done_status = f"**Done** — seed **{new_seed}**"
            elif job_status.startswith("**"):
                done_status = job_status
            else:
                done_status = f"**Done** — {job_status}"
            applied_tags = job.request.tags
            if applied_tags:
                ctx.tags.remember_tags(applied_tags, save=ctx.save_settings)
            tag_line = _format_tag_summary(applied_tags)
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
                primary,
                before_image,
                quick_tag_update,
                toggle_update,
                False,
                False,
                *mode_ui,
            )

        def _run_once(mode_label, all_inputs, *, keep_continuous_toggle: bool):
            request, init_images, mask_images, before_image, mode, control_images = _generation_request(*all_inputs)
            yield _progress_outputs(mode_label, "Queued")
            for event in service.submit_streaming(
                request,
                init_images=init_images,
                mask_images=mask_images,
                control_images=control_images,
            ):
                if event[0] == "progress":
                    _, _step, _total, message, preview = event
                    yield _progress_outputs(mode_label, message, preview)
                else:
                    _, job = event
                    yield _finished_outputs(
                        mode_label,
                        job,
                        before_image if mode != "txt2img" else None,
                        continuous_on=keep_continuous_toggle,
                    )

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
            img2img_denoise,
            inpaint_denoise_value,
            mask_blur_value,
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
            inpaint_source,
            continuous_enabled,
            cooldown_wait,
        ):
            loop_ctrl["active"] = True
            ctx.settings.generation_cooldown_seconds = float(cooldown_wait or 0)
            ctx.save_settings()

            try:
                run_number = 0
                while loop_ctrl["active"]:
                    run_number += 1
                    if continuous_enabled and run_number > 1:
                        yield _progress_outputs(mode_label, f"Run {run_number}")

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
                        img2img_denoise,
                        inpaint_denoise_value,
                        mask_blur_value,
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
                        inpaint_source,
                    )

                    for update in _run_once(
                        mode_label,
                        request_inputs,
                        keep_continuous_toggle=continuous_enabled and loop_ctrl["active"],
                    ):
                        yield update

                    if not loop_ctrl["active"]:
                        break
                    if not continuous_enabled:
                        break

                    wait_s = max(0, int(cooldown_wait or 0))
                    for remaining in range(wait_s, 0, -1):
                        if not loop_ctrl["active"]:
                            break
                        yield _progress_outputs(mode_label, f"Cooling — next run in {remaining}s")
                        time.sleep(1)
            finally:
                loop_ctrl["active"] = False

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
        ):
            target = stored_result or workspace_result
            if target is None:
                raise gr.Error("Generate an image first, then run ReActor on the result.")
            if source_image is None:
                raise gr.Error("Upload a source face image.")

            options = FaceSwapOptions(
                source_face_index=max(0, int(source_idx or 0)),
                target_face_index=int(target_idx if target_idx is not None else -1),
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
            ],
            outputs=[workspace_image, last_result, status, compare_btn, empty_canvas],
            show_progress="minimal",
        )

        def toggle_compare(showing, before, after):
            if before is None or after is None:
                raise gr.Error("Generate an image first to compare.")
            aligned_before, aligned_after = _align_compare_pair(before, after)
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

        generate_outputs = [
            workspace_image,
            compare_slider,
            compare_btn,
            gallery,
            tag_summary,
            info,
            status,
            last_seed,
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
            denoise,
            inpaint_denoise,
            mask_blur,
            inpaint_area,
            inpaint_padding,
            masked_content,
            workspace_image,
            mask_editor,
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
            inpaint_source,
            continuous_toggle,
            cooldown_seconds,
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

        def _on_studio_tab_select(mode_label, cn_current, current_ckpt):
            mode = _mode_from_label(mode_label)
            ckpt_update, new_map = refresh_checkpoints(
                ctx, rescan=True, current_value=current_ckpt
            )
            return (
                ckpt_update,
                format_model_status(ctx),
                new_map,
                _cn_models_update(cn_current),
                _pnginfo_pending_hint(),
            )

        if tab is not None:
            tab.select(
                _on_studio_tab_select,
                inputs=[mode_toggle, cn_model, checkpoint],
                outputs=[checkpoint, model_status, state, cn_model, pnginfo_hint],
                show_progress=False,
            )
