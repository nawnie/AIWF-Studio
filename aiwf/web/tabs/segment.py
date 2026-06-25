from __future__ import annotations

import gradio as gr

from aiwf.bootstrap import AppContext
from aiwf.core.domain.segment import SegmentPoint, SegmentRequest
from aiwf.core.domain.segment_presets import (
    CUSTOM_SEGMENT_PRESET_ID,
    resolve_segment_preset_config,
    resolve_segment_text_prompt,
    segment_mask_preset_choices,
)
from aiwf.infrastructure.diffusers.mask import editor_from_mask
from aiwf.web.registry import WebRegistry


def _model_choices(models) -> list[tuple[str, str]]:
    return [(model.title, model.id) for model in models]


def _preset_note(preset_id: str | None) -> str:
    config = resolve_segment_preset_config(preset_id)
    note = str(config.get("note") or "")
    return (
        f"**Preset edge profile:** threshold `{float(config['box_threshold']):.2f}` · "
        f"candidate `{int(config['mask_index'])}` · dilation `{int(config['dilation'])}` · "
        f"blur `{int(config['mask_blur'])}` · feather `{int(config['feather'])}`  \n{note}"
    )


def build_segment_panel(ctx: AppContext) -> None:
    """Segment Anything panel — embeddable in Studio or as a standalone tab."""
    service = ctx.segment
    models = service.list_models()
    person = resolve_segment_preset_config("person")

    with gr.Column(elem_classes=["aiwf-segment"]):
        with gr.Column(elem_classes=["aiwf-page-header"]):
            gr.Markdown("Segment (SAM)", elem_classes=["aiwf-section-label"])
            gr.Markdown(
                "Generate inpaint masks with presets, click points, or a custom prompt. "
                "Presets include detection and edge-treatment settings—not just a noun in a box.",
                elem_classes=["aiwf-page-intro"],
            )
            gr.Markdown(service.folder_help(), elem_classes=["aiwf-page-path"])

        with gr.Row(equal_height=False, elem_classes=["aiwf-segment-workspace"]):
            with gr.Column(scale=1, min_width=300, elem_classes=["aiwf-panel"]):
                source = gr.Image(label="Source image", type="pil", sources=["upload", "clipboard"])
                sam_model = gr.Dropdown(
                    label="SAM model",
                    choices=_model_choices(models),
                    value=models[0].id if models else None,
                )
                mask_preset = gr.Dropdown(
                    label="What to mask",
                    choices=segment_mask_preset_choices(),
                    value="person",
                    info="Used when no click points are set",
                )
                custom_prompt = gr.Textbox(
                    label="Custom prompt",
                    placeholder="person.hat  (separate categories with .)",
                    visible=False,
                )
                preset_guidance = gr.Markdown(_preset_note("person"), elem_classes=["aiwf-stage-note"])
                box_threshold = gr.Slider(
                    0.05, 0.9, value=person["box_threshold"], step=0.01, label="Detection threshold"
                )
                with gr.Row():
                    point_x = gr.Number(value=0, precision=0, label="Point X")
                    point_y = gr.Number(value=0, precision=0, label="Point Y")
                    point_label = gr.Radio(["Include", "Exclude"], value="Include", label="Point type")
                mask_index = gr.Slider(
                    0, 2, value=person["mask_index"], step=1, label="Mask candidate", info="SAM multimask index"
                )
                with gr.Row():
                    dilation = gr.Slider(0, 64, value=person["dilation"], step=1, label="Dilation")
                    mask_blur = gr.Slider(0, 64, value=person["mask_blur"], step=1, label="Internal blur")
                    feather = gr.Slider(0, 64, value=person["feather"], step=1, label="Edge feather")
                refresh_models = gr.Button("Refresh models", elem_classes=["aiwf-btn-ghost", "aiwf-btn-sm"])
                run_segment = gr.Button("Generate mask", variant="primary", elem_classes=["aiwf-generate-btn"])

            with gr.Column(scale=1, min_width=300, elem_classes=["aiwf-panel"]):
                mask_output = gr.Image(label="Mask", type="pil", interactive=False)
                preview_output = gr.Image(label="Preview", type="pil", interactive=False)
                mask_gallery = gr.Gallery(label="SAM candidates", columns=3, object_fit="contain")
                status = gr.Markdown("**Ready**", elem_classes=["aiwf-status-bar"])
                editor_export = gr.JSON(label="ImageEditor payload (for workflows)", visible=False)

    def _refresh():
        refreshed = service.refresh_models()
        return gr.update(choices=_model_choices(refreshed), value=refreshed[0].id if refreshed else None)

    refresh_models.click(_refresh, outputs=[sam_model], show_progress=False)

    def _on_preset_change(preset_id):
        config = resolve_segment_preset_config(preset_id)
        return (
            gr.update(visible=preset_id == CUSTOM_SEGMENT_PRESET_ID),
            gr.update(value=float(config["box_threshold"])),
            gr.update(value=int(config["mask_index"])),
            gr.update(value=int(config["dilation"])),
            gr.update(value=int(config["mask_blur"])),
            gr.update(value=int(config["feather"])),
            _preset_note(preset_id),
        )

    mask_preset.change(
        _on_preset_change,
        inputs=[mask_preset],
        outputs=[custom_prompt, box_threshold, mask_index, dilation, mask_blur, feather, preset_guidance],
        show_progress=False,
    )

    def _segment(
        image,
        model_id,
        preset_id,
        custom_text,
        threshold,
        px,
        py,
        p_label,
        m_index,
        dilate,
        mask_blur_val,
        feather_val,
    ):
        if image is None:
            raise gr.Error("Upload an image first.")
        if not model_id and not service.list_models():
            raise gr.Error(f"No SAM models in {service.sam_dir()}")

        has_points = int(px or 0) > 0 or int(py or 0) > 0
        text_prompt = resolve_segment_text_prompt(preset_id, custom_text)
        if not has_points and not text_prompt:
            raise gr.Error("Choose what to mask, set click points, or enter a custom prompt.")

        request = SegmentRequest(
            text_prompt=text_prompt,
            box_threshold=float(threshold),
            points=[SegmentPoint(x=int(px), y=int(py), label=0 if p_label == "Exclude" else 1)]
            if has_points
            else [],
            mask_index=int(m_index),
            dilation=int(dilate),
            mask_blur=int(mask_blur_val),
            feather=int(feather_val),
        )

        mask, preview, candidates, message = service.segment(image, request, model_id=model_id)
        gallery = [mask, *candidates]
        editor = editor_from_mask(image, mask)
        return mask, preview, gallery, f"**{message}**", editor

    run_segment.click(
        _segment,
        inputs=[
            source,
            sam_model,
            mask_preset,
            custom_prompt,
            box_threshold,
            point_x,
            point_y,
            point_label,
            mask_index,
            dilation,
            mask_blur,
            feather,
        ],
        outputs=[mask_output, preview_output, mask_gallery, status, editor_export],
        show_progress="minimal",
    )


def register_segment(registry: WebRegistry) -> None:
    @registry.tab("Segment", order=21)
    def build(ctx: AppContext, tab: gr.Tab | None = None) -> None:
        build_segment_panel(ctx)
